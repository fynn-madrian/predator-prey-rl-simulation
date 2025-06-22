import numpy as np
from pettingzoo import ParallelEnv
from gym import spaces
from render import render
import random
import json
import jsonlines
import os
import datetime
from algorithms import get_model
from environment_object import generate_all_objects, Field, place_non_overlapping
from numba import njit
from helpers import collides_with, convert_observation, point_to_segment_distance, first_hit_polyline, first_hit_circles
import heapq

from collections import deque, defaultdict
from algorithms import load_model
import tensorflow as tf

MOVE_BIN_VALUES = [-1.0, -0.25, 0.0, 0.25, 1.0]
ATK_BIN_VALUES = [-0.3, -0.1, 0.0, 0.1, 0.3]
SPIN_BIN_VALUES = [-0.75,  0.0, 0.75]


class CustomEnvironment(ParallelEnv):

    metadata = {"name": "plutonian_insects"}

    def __init__(self, config_path=None, config=None, folder_path=None, **kwargs):

        if "seed" in kwargs:
            np.random.seed(kwargs["seed"])
            random.seed(kwargs["seed"])
            tf.random.set_seed(kwargs["seed"])

        super().__init__()

        if config_path and config:
            raise ValueError(
                "Both config and config_path provided. Please provide only one.")

        if config:
            self.config = config
        elif config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise ValueError(
                "No config provided. Please provide a config or config_path.")

        self.map_size = self.config.get("map_size")
        self.map_config = self.config.get("map_config")

        self.objects = self.generate_map()
        self.max_age = self.config.get("max_age")
        self.max_speed = self.config.get("max_speed")
        self.predator_fov = self.config.get("predator_fov")
        self.prey_fov = self.config.get("prey_fov")
        self.vision_rays = self.config.get("vision_rays")
        self.vision_range = self.config.get("vision_range")
        self.ray_all_type_mapping = {
            0: "none",
            1: "out_of_bounds",
            2: "predator",
            3: "River",
            4: "Rock",
            5: "Field",
        }
        self.ray_all_type_mapping_inv = {v: k for k, v in self.ray_all_type_mapping.items()}
        self.all_type_count = len(self.ray_all_type_mapping)

        self.sequence_length = self.config.get("sequence_length")
        self.buffer_size = self.config.get("buffer_size")
        self.agent_detection_radius = self.config.get(
            "agent_detection_radius")
        self.agent_collision_radius = self.config.get(
            "agent_collision_radius")
        self.group_paths = {0: "predator", 1: "prey"}

        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.step_count = 0
        self.goal_pos = None
        self.any_field_depleted = False
        if folder_path is None:
            self.folder_path = os.path.join(
                "logs", self.timestamp)
        else:
            self.folder_path = folder_path
        self.model_save_path = os.path.join(self.folder_path, "models")
        self.scenario = self.config.get("scenario", "navigate")
        self.agent_data = self.generate_agents(self.scenario)
        if self.scenario == "navigate":

            self.start_pos = np.array(
                self.config.get("start_pos", [0, 0]), dtype=np.float32)
            self.goal_pos = np.array(
                self.config.get("goal_pos", [self.map_size-1, self.map_size-1]), dtype=np.float32)
            # Reward weights
            self.time_penalty = self.config.get("time_penalty", 0.01)
            self.progress_scaling = self.config.get("progress_scaling", 1.0)
            self.collision_penalty = self.config.get("collision_penalty", -0.5)
            self.goal_bonus = self.config.get("goal_bonus", 10.0)

    def _sample_start_goal(self):

        margin = 5
        max_try = 200
        diag_thresh = self.map_size * 0.3
        for _ in range(max_try):
            s = np.random.uniform(
                margin, self.map_size - margin, size=2).astype(np.float32)
            g = np.random.uniform(
                margin, self.map_size - margin, size=2).astype(np.float32)

            # distance check
            if np.linalg.norm(g - s) < diag_thresh:
                continue

            def free(p):
                return all(
                    getattr(o, "is_passable", True) or not collides_with(o, p)
                    for o in self.objects
                )
            if free(s) and free(g):
                return s, g

        # fallback: centre & opposite corner
        return (np.array([margin, margin], dtype=np.float32),
                np.array([self.map_size - margin, self.map_size - margin], dtype=np.float32))

    def reset(self, seed=None, options=None, models=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)

        self.model_save_path = os.path.join(self.folder_path, "models")
        os.makedirs(self.folder_path, exist_ok=True)
        for grp in self.group_paths.values():
            os.makedirs(os.path.join(self.folder_path,
                        "models", grp), exist_ok=True)
            os.makedirs(os.path.join(self.folder_path, grp), exist_ok=True)

        # Clear field‑depleted flag for new episode
        self.any_field_depleted = False
        # Generate a fresh map for this episode
        self.objects = self.generate_map()

        # Reset or initialize agents based on scenario
        for agent in self.agent_data.values():
            # Common resets
            agent.age = 0
            agent.sequence_buffer = []
            agent.initial_sequence_state = None
            agent.hidden_state = None
            agent.stale_count = 0
            agent.reward_penalty = 0.0
            agent.reward_boost = 0.0
            agent.collided = False
            agent.previous_position.clear()

            if getattr(self, "scenario", None) == "navigate":
                # Sample new start and goal positions
                self.start_pos, self.goal_pos = self._sample_start_goal()
                # Place all agents at the common start position
                agent.position = self.start_pos.copy()
                # Facing directly toward goal
                direction = self.goal_pos - self.start_pos
                norm = np.linalg.norm(direction)
                if norm > 1e-8:
                    agent.facing = direction / norm
                else:
                    agent.facing = np.array([1.0, 0.0], dtype=np.float32)
                agent.prev_goal_dist = float(
                    np.linalg.norm(agent.position - self.goal_pos))
            else:
                # For non-navigation scenarios, randomly place each agent
                placed = False
                while not placed:
                    loc = np.array([np.random.uniform(0, self.map_size),
                                    np.random.uniform(0, self.map_size)])
                    # Check collision with impassable objects
                    if not any(
                        (not obj.is_passable and collides_with(obj, loc))
                        for obj in self.objects
                    ):
                        agent.position = loc
                        agent.velocity = np.zeros_like(agent.velocity)
                        # Default facing vector
                        agent.facing = np.array([1.0, 0.0], dtype=np.float32)
                        placed = True

        self.agents = list(self.agent_data.keys())

        # Build initial observations and info dicts
        observations = {
            str(agent_id): self.get_observation(agent)
            for agent_id, agent in self.agent_data.items()
        }
        info = {
            str(agent_id): {
                "value": agent.V,
                "log_prob": agent.log_prob,
                "food_collected": 0.0
            }
            for agent_id, agent in self.agent_data.items()
        }

        self.step_count = 0
        self.conflict_agents = []

        return observations, info

    def step(self, actions):
        x_idxs = {aid: action["x_dir"]
                  for aid, action in actions.items()}
        y_idxs = {aid: action["y_dir"]
                  for aid, action in actions.items()}
        spin_idxs = {aid: action["spin"]
                     for aid, action in actions.items()}

        pre_move_data = {
            aid: {
                "position": agent.position.copy(),
                "velocity": agent.velocity.copy(),
                "facing": agent.facing.copy(),
                "age": agent.age,
                "stale_count": agent.stale_count,
                "group": agent.group,
            }
            for aid, agent in self.agent_data.items()
        }

        # Reset per-step hit flags
        for agent in self.agent_data.values():
            agent.was_hit_this_step = False

        # track per-step metrics
        food_collected = {}           # agent_id -> total food gathered this step
        predator_hit_reward = {}      # agent_id -> reward from hitting prey this step

        attacking_agents = []
        for agent_id, agent in self.agent_data.items():
            if agent.group == 0:
                attacking_agents.append(agent_id)

        attacking_agents = np.random.permutation(attacking_agents)

        for agent_id in attacking_agents:
            agent = self.agent_data[agent_id]
            spin_idx = spin_idxs[agent_id]
            turn_accel = SPIN_BIN_VALUES[spin_idx] * \
                agent.max_turn_acceleration
            agent.turn_velocity += turn_accel

            rel_x = ATK_BIN_VALUES[x_idxs[agent_id]]
            rel_y = ATK_BIN_VALUES[y_idxs[agent_id]]
            vec = np.array([rel_x, rel_y], dtype=np.float32)
            agent.attack(vec)

        # record predator hit rewards generated during resolve_fights
        for aid, agent in self.agent_data.items():
            if agent.reward_boost > 0:
                predator_hit_reward[aid] = agent.reward_boost

        moving_agents = []
        for agent_id, agent in self.agent_data.items():
            if agent.group == 1:
                moving_agents.append(agent_id)

        moving_agents = np.random.permutation(moving_agents)

        for agent_id in moving_agents:
            agent = self.agent_data[agent_id]
            spin_idx = spin_idxs[agent_id]
            turn_accel = SPIN_BIN_VALUES[spin_idx] * \
                agent.max_turn_acceleration
            agent.turn_velocity += turn_accel
            idx_x = x_idxs[agent_id]
            idx_y = y_idxs[agent_id]
            rel_x = MOVE_BIN_VALUES[idx_x]
            rel_y = MOVE_BIN_VALUES[idx_y]
            vec = np.array([rel_x, rel_y], dtype=np.float32)
            agent.move(vec)

        prey_agents = [agent for agent in self.agent_data.values()
                       if agent.group == 1]
        food_sources = [obj for obj in self.objects if obj.is_food]

        agent_gather_map = {}
        for agent in prey_agents:
            for food_obj in food_sources:
                if collides_with(food_obj, agent.position) and agent.ID in moving_agents:
                    if food_obj not in agent_gather_map:
                        agent_gather_map[food_obj] = []
                    agent_gather_map[food_obj].append(agent)
                    break

        for food_obj, gatherers in agent_gather_map.items():
            available = food_obj.food
            gather_weights = []

            for _ in gatherers:
                gather_weights.append(random.uniform(
                    0.8, 1.2))

            weight_sum = sum(gather_weights)
            gathered_total = 0

            for agent, weight in zip(gatherers, gather_weights):
                proportion = weight / weight_sum

                max_gather = max(0.5, food_obj.food * 0.15)
                available_to_agent = available * proportion
                gathered = min(available_to_agent, max_gather)
                agent.gather(gathered)
                food_collected[agent.ID] = food_collected.get(
                    agent.ID, 0.0) + gathered
                food_obj.food -= gathered
                gathered_total += gathered

        for agent_id in self.agent_data.keys():
            agent = self.agent_data[agent_id]
            agent.increase_age()

        rewards = {}
        next_observations = {}

        # Record whether any food field is fully depleted **before** regeneration/respawn
        self.any_field_depleted = any(
            o.is_food and o.food <= 0 for o in self.objects)
        # regenerate partially depleted fields (unless a prey is on them)
        for obj in self.objects:
            if obj.is_food and obj.food > 0:
                if not any(
                    collides_with(obj, agent.position)
                    for agent in self.agent_data.values()
                    if agent.group == 1
                ):
                    multiplier = random.uniform(0.002, 0.005)
                    obj.food = min(obj.max_food, obj.food +
                                   obj.radius * multiplier)

        # remove fully depleted fields and respawn replacements using non-overlapping placement
        depleted = [o for o in self.objects if o.is_food and o.food <= 0]
        for field in depleted:
            self.objects.remove(field)
        for _ in depleted:
            new_fields = place_non_overlapping(
                Field, 1, self.map_config, self.map_size, "Field_base_radius",
                self.objects,
                lambda: random.randint(*self.map_config["Field_food_range"]),
                max_food=self.map_config["Field_max_food"]
            )
            if new_fields:
                self.objects.append(new_fields[0])

        self.agents = list(self.agent_data.keys())

        terminations = {}
        truncations = {}

        for agent_id, agent in self.agent_data.items():
            key = str(agent_id)
            truncations[key] = agent.age >= self.max_age
            terminations[key] = self.check_termination(agent)

        for agent_id, agent in self.agent_data.items():
            observation = self.get_observation(agent)
            next_observations[str(agent_id)] = observation
            rewards[str(agent_id)] = self.calculate_reward(agent, observation)

        self.agents = list(self.agent_data.keys())

        # (Removed the old commented max_cycles snippet here)

        infos = {
            str(agent_id): {
                "value": agent.V,
                "log_prob": agent.log_prob,
                "food_collected": float(food_collected.get(str(agent_id), 0.0))
            }
            for agent_id, agent in self.agent_data.items()
        }

        self.step_count += 1

        """        if self.max_cycles is not None and self.step_count >= self.max_cycles:
            truncations = {agent_id: True for agent_id in self.agents}"""
        hit_flags = {str(aid): getattr(agent, "was_hit_this_step", False)
                     for aid, agent in self.agent_data.items()}

        predator_positions = np.array(
            [a.position for a in self.agent_data.values() if a.group == 0]
        )
        dist_to_predator = {}
        if predator_positions.size:
            for aid, agent in self.agent_data.items():
                if agent.group == 0:
                    dist_to_predator[str(aid)] = 0.0
                else:
                    dist_to_predator[str(aid)] = float(
                        np.min(np.linalg.norm(
                            predator_positions - agent.position, axis=1))
                    )
        else:
            for aid in self.agent_data:
                dist_to_predator[str(aid)] = None
        self.log_step(pre_move_data, rewards, actions,
                      food_collected, predator_hit_reward,
                      hit_flags, dist_to_predator, terminations, truncations)

        # Mark hidden‑state reset for agents that finished; actual sequence flush
        # happens in the main loop *after* the current timestep is appended.
        for agent_id, agent in self.agent_data.items():
            key = str(agent_id)
            if terminations.get(key, False) or truncations.get(key, False):
                agent.hidden_state = None

        return next_observations, rewards, terminations, truncations, infos

    def find_path(self, start, goal, step_size=5, max_steps=50):
        """
        Fast, greedy path approximation.
        Attempts to move from start to goal with obstacle avoidance.
        Returns a list of waypoints or None if blocked.
        """
        direction = goal - start
        distance = np.linalg.norm(direction)
        if distance < step_size:
            return [start, goal]

        direction = direction / distance
        path = [start]
        current = start.copy()
        attempts = 0

        while attempts < max_steps:
            attempts += 1
            next_point = current + direction * step_size

            # Clip to map bounds
            if np.any(next_point < 0) or np.any(next_point >= self.map_size):
                return None

            # Check for collision
            blocked = any(
                not getattr(obj, "is_passable", True) and
                collides_with(obj, next_point)
                for obj in self.objects
            )

            if blocked:
                # Try minor detours: left/right/orthogonal
                perp = np.array([-direction[1], direction[0]]) * step_size * 0.75
                for offset in [perp, -perp]:
                    detour = current + direction * step_size + offset
                    if np.all(0 <= detour) and np.all(detour < self.map_size):
                        if not any(not getattr(obj, "is_passable", True) and
                                collides_with(obj, detour) for obj in self.objects):
                            next_point = detour
                            break
                else:
                    return None  # No viable detour

            path.append(next_point)
            current = next_point

            if np.linalg.norm(goal - current) < step_size:
                path.append(goal)
                return path

        return None

    def generate_agents(self, scenario):
        agents = {}
        location = np.array(
            [np.random.randint(0, self.map_size), np.random.randint(0, self.map_size)])
        while any([not obj.is_passable and collides_with(obj, location) for obj in self.objects]):
            location = np.array(
                [np.random.randint(0, self.map_size), np.random.randint(0, self.map_size)])

        model = get_model(log_dir=self.folder_path)

        new_agent = Agent(1, location, env=self, model=model,
                          max_speed=self.max_speed, max_age=self.max_age)

        agents[str(new_agent.ID)] = new_agent

        if scenario == "full" or scenario == "flee":
            # Add a predator agent
            location = np.array(
                [np.random.randint(0, self.map_size), np.random.randint(0, self.map_size)])
            while any([not obj.is_passable and collides_with(obj, location) for obj in self.objects]):
                location = np.array(
                    [np.random.randint(0, self.map_size), np.random.randint(0, self.map_size)])

            new_predator = Agent(0, location, env=self, max_speed=self.max_speed, max_age=self.max_age)
            agents[str(new_predator.ID)] = new_predator

        return agents

    def generate_map(self):
        map_objects = generate_all_objects(self.map_config, self.map_size)

        return map_objects

    def render(self):
        render(self.objects, self.agent_data, goal=self.goal_pos)

    def observation_space(self, agent_id=None):
        observation_space = spaces.Dict({
            "rays": spaces.Box(
                low=0, high=1.0,
                shape=(self.vision_rays, 10),
                dtype=np.float32
            ),
            "facing": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "velocity": spaces.Box(low=-5.0, high=5.0, shape=(2,), dtype=np.float32),
            "good_vector": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "good_distance": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "good_info": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "bad_vector": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "bad_distance": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })
        return observation_space

    def action_space(self, agent_id=None):
        return spaces.Dict({
            "x":    spaces.Discrete(len(MOVE_BIN_VALUES)),
            "y":    spaces.Discrete(len(MOVE_BIN_VALUES)),
            "spin": spaces.Discrete(len(SPIN_BIN_VALUES))
        })

    def get_observation(self, agent):
        rays = self.cast_rays(agent)
        facing = agent.facing
        good_direction, good_distance, bad_vector, bad_distance = agent.get_help_vector(
            rays)

        observation = {
            "rays": rays,
            "facing": facing,
            "position": agent.position,
            "velocity": agent.velocity,
            "good_vector": good_direction,
            "good_distance": good_distance,
            "bad_vector": bad_vector,
            "bad_distance": bad_distance,
        }
        return observation

    def remove_agent(self, agent_id):
        if len(self.agent_data[agent_id].buffer) > 0:
            print(
                f"Optimizing model for agent {agent_id} with buffer size {len(self.agent_data[agent_id].buffer)}")
            self.agent_data[agent_id].optimize_model()
        else:
            print(
                f"Skipping optimization for agent {agent_id} — buffer is empty.")

        model_path = os.path.join(
            self.model_save_path, self.group_paths[self.agent_data[agent_id].group], f"agent_{agent_id}_model.weights.h5")

        self.agent_data[agent_id].model.save_weights(model_path)

        print(
            f"Agent {agent_id} has been terminated. Model saved to {model_path}.")
        del self.agent_data[agent_id]
        self.agents.remove(agent_id)

    def cast_rays(self, agent):
        N, R, S = self.vision_rays, self.vision_range, 20
        ray_thick = 1.0
        fov = np.deg2rad(self.predator_fov if agent.group == 0 else self.prey_fov)

        base_ang = np.arctan2(-agent.facing[1], agent.facing[0])
        angles   = np.linspace(-fov/2, fov/2, N) + base_ang
        dirs     = np.stack([np.cos(angles), -np.sin(angles)], axis=1)
        d_grid   = np.linspace(0.1, R, S, dtype=np.float32)
        pts      = agent.position[None, None, :] + dirs[:, None, :] * d_grid[None,:,None]

        # ------------------------------------------------------------------ OOB
        dist_all = np.full(N, R, np.float32)
        code_all = np.zeros(N, np.int32)
        fill_pct = np.zeros(N, np.float32)

        mask_oob  = (pts[...,0] < 0) | (pts[...,0] >= self.map_size) | \
                    (pts[...,1] < 0) | (pts[...,1] >= self.map_size)
        first_oob = mask_oob.argmax(1)
        has_oob   = mask_oob.any(1)
        dist_all[has_oob] = d_grid[first_oob[has_oob]]
        code_all[has_oob] = 1      # code 1 = map edge

        # ----------------------------------------------------------- other AGENTS
        others   = [o for o in self.agent_data.values() if o.ID != agent.ID]
        if others:
            op    = np.array([o.position for o in others], np.float32)
            orad  = self.agent_detection_radius * np.ones(len(others), np.float32)
            ocod  = np.array([2 if o.group == 0 else 3 for o in others], np.int32)

            hit_s, hit_o = first_hit_circles(pts, op, orad, ray_thick)
            hit_mask = hit_s != -1
            idxs     = np.where(hit_mask & (code_all == 0))[0]
            dist_all[idxs] = d_grid[hit_s[idxs]]
            code_all[idxs] = ocod[hit_o[idxs]]

        # ----------------------------------------------------- static CIRCLES/RECTS
        circ_pos, circ_rad, circ_code, circ_fill = [], [], [], []
        for obj in self.objects:
            if hasattr(obj, "radius") and not hasattr(obj, "points"):
                circ_pos.append(obj.position)
                circ_rad.append(obj.radius)
                circ_code.append(self.ray_all_type_mapping_inv[type(obj).__name__])
                circ_fill.append(
                    obj.food/obj.max_food if type(obj).__name__ == "Field" else 0.0)

        if circ_pos:
            cp   = np.vstack(circ_pos).astype(np.float32)
            cr   = np.array(circ_rad, np.float32)
            hit_s, hit_o = first_hit_circles(pts, cp, cr, ray_thick)
            hit_mask = (hit_s != -1) & (code_all == 0)
            idxs     = np.where(hit_mask)[0]
            dist_all[idxs] = d_grid[hit_s[idxs]]
            code_all[idxs] = np.array(circ_code, np.int32)[hit_o[idxs]]
            fill_pct[idxs] = np.array(circ_fill, np.float32)[hit_o[idxs]]

        # --------------------------------------------------------- poly-line RIVERS
        for obj in self.objects:
            if not hasattr(obj, "points"):          # not a river
                continue
            for i in range(N):
                if code_all[i] != 0:                # ray already hit something
                    continue
                j = first_hit_polyline(pts[i], obj.np_points,
                                    np.float32(obj.radius), np.float32(ray_thick))
                if j != -1:
                    dist_all[i] = d_grid[j]
                    code_all[i] = self.ray_all_type_mapping_inv["River"]

        # ----------------------------------------------------- goal (navigate only)
        if getattr(self, "scenario", None) == "navigate":
            goal_pos = self.goal_pos.astype(np.float32)
            GOAL_R   = 3.0
            goal_code = self.ray_all_type_mapping_inv["Field"]
            for i in range(N):
                if code_all[i] != 0:
                    continue
                d2 = ((pts[i] - goal_pos)**2).sum(1)
                hit = np.where(d2 < (GOAL_R + ray_thick)**2)[0]
                if hit.size:
                    j = int(hit[0])
                    dist_all[i] = d_grid[j]
                    code_all[i] = goal_code
                    fill_pct[i] = 1.0

        # --------------------------------------------------------------- features
        feats = np.zeros((N, 4 + self.all_type_count), np.float32)
        for i in range(N):
            feats[i, :3] = [dist_all[i] / R,               # normalised distance
                            np.sin(angles[i]), np.cos(angles[i])]
            feats[i, 3 + code_all[i]] = 1.0                # one-hot for hit type
            feats[i, -1] = fill_pct[i]                     # field fill %
        return feats



    def calculate_reward(self, agent, observation):
        return agent.get_reward(observation)

    def check_termination(self, agent):

        if self.scenario == "navigate":
            dist = np.linalg.norm(agent.position - self.goal_pos)
            goal_tol = self.config.get("goal_tolerance", 3.0)
            return dist <= goal_tol

        elif self.scenario == "flee":
            # In flee scenario, we terminate if agent is hit by predator
            return agent.was_hit_this_step

        elif self.scenario == "gather":
            # terminate when any field was depleted this step
            return self.any_field_depleted

        elif self.scenario == "full":
            # In full scenario, terminate if agent is hit OR any field was depleted
            if agent.was_hit_this_step:
                return True
            return self.any_field_depleted

    def get_object_dicts(self):
        data = []
        for obj in self.objects:
            data.append(obj.to_dict())
        return data

    def log_step(self, pre_move_data, rewards, actions,
                 food_collected, predator_hit_reward,
                 hit_flags, dist_to_predator, terminations, truncations):
        env_data = {
            "step": self.step_count,
            "objects": self.get_object_dicts(),
            "goal": self.goal_pos.tolist() if getattr(self, "scenario", None) == "navigate" else None,
        }

        env_json_path = os.path.join(self.folder_path, "environment.jsonl")
        with jsonlines.open(env_json_path, "a") as f:
            f.write(env_data)

        for agent_id in pre_move_data.keys():
            termination_flag = terminations.get(agent_id, False)
            truncation_flag = truncations.get(agent_id, False)
            agent_data = {
                "step": self.step_count,
                "position":  pre_move_data[agent_id]["position"].tolist(),
                "age":  pre_move_data[agent_id]["age"],
                "x_bin": int(actions[agent_id]["x_dir"]),
                "y_bin": int(actions[agent_id]["y_dir"]),
                "spin_bin": int(actions[agent_id]["spin"]),
                "velocity":  pre_move_data[agent_id]["velocity"].tolist(),
                "food_collected": float(food_collected.get(agent_id, 0.0)),
                "predator_hit_reward": float(predator_hit_reward.get(agent_id, 0.0)),
                "reward": float(rewards[agent_id]),
                "facing":  pre_move_data[agent_id]["facing"].tolist(),
                "group":  pre_move_data[agent_id]["group"],
                "termination": bool(termination_flag),
                "truncation": bool(truncation_flag),
                "staleness": pre_move_data[agent_id]["stale_count"],
                "hit_by_predator": bool(hit_flags.get(agent_id, False)),
                "dist_to_predator": dist_to_predator.get(agent_id, None),
            }

            agent_json_path = os.path.join(
                self.folder_path, self.group_paths[pre_move_data[agent_id]["group"]], f"agent_{agent_id}.jsonl")
            with jsonlines.open(agent_json_path, "a") as f:
                f.write(agent_data)

    def path_collides(self, start, end, steps=20, radius=0, ignore_objects=None):

        if ignore_objects is None:
            ignore_objects = []

        path = np.linspace(start, end, steps)

        if radius > 0:
            margin = radius
        else:
            margin = 0
        if np.any(path < margin) or np.any(path >= self.map_size - margin):
            return True

        others = [o for o in self.objects if not o.is_passable and hasattr(o, "shape")
                  and o not in ignore_objects and o.shape != "polyline"]
        rivers = [o for o in self.objects if not o.is_passable and getattr(o, "shape", None) == "polyline"
                  and o not in ignore_objects]

        if others:
            obj_positions = np.array([o.position for o in others])
            obj_radii = np.array([o.radius for o in others])

            if radius == 0:
                for point in path:
                    dists = np.linalg.norm(obj_positions - point, axis=1)
                    if np.any(dists < obj_radii):
                        return True
            else:
                offsets = radius * np.stack([
                    np.cos(np.linspace(0, 2 * np.pi, 8, endpoint=False)),
                    np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False))
                ], axis=1)
                all_points = (path[:, None, :] +
                              offsets[None, :, :]).reshape(-1, 2)
                dists = np.linalg.norm(
                    obj_positions[:, None, :] - all_points[None, :, :], axis=2)
                if np.any(dists < obj_radii[:, None]):
                    return True

        for point in path:
            for river in rivers:
                for i in range(len(river.points) - 1):
                    if point_to_segment_distance(point, river.points[i], river.points[i + 1]) < river.radius:
                        return True
        return False

    def resolve_collision(self, position, target_position, steps=20, radius=0):

        path = np.linspace(position, target_position, steps)

        others = [o for o in self.objects if not o.is_passable and hasattr(
            o, "shape") and o.shape != "polyline"]
        rivers = [o for o in self.objects if not o.is_passable and getattr(
            o, "shape", None) == "polyline"]

        if others:
            obj_positions = np.array([o.position for o in others])
            obj_radii = np.array([o.radius for o in others])

        for i, point in enumerate(path):

            if radius > 0:
                if np.any(point - radius < 0) or np.any(point + radius >= self.map_size):
                    return path[i - 1] if i > 0 else position
            else:
                if np.any(point < 0) or np.any(point >= self.map_size):
                    return path[i - 1] if i > 0 else position

            if others:
                if radius > 0:
                    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
                    offsets = radius * \
                        np.stack([np.cos(angles), np.sin(angles)], axis=1)
                    points = point + offsets
                else:
                    points = np.array([point])

                dists = np.linalg.norm(
                    obj_positions[:, None, :] - points[None, :, :], axis=2)
                if np.any(dists < obj_radii[:, None]):
                    return path[i - 1] if i > 0 else position

            for river in rivers:
                for j in range(len(river.points) - 1):
                    if point_to_segment_distance(point, river.points[j], river.points[j + 1]) < river.radius:
                        return path[i - 1] if i > 0 else position

        return target_position

    def check_agent_collision(self, agent, position, target_position, steps=20):
        path = np.linspace(position, target_position, steps)
        others = [a for a in self.agent_data.values() if a.ID != agent.ID]

        if not others:
            return None, None

        agent_positions = np.array([a.position for a in others])
        agent_ids = [a.ID for a in others]

        dists = np.linalg.norm(
            agent_positions[:, None, :] - path[None, :, :], axis=2)
        collision = np.where(dists < self.agent_collision_radius)

        if collision[0].size > 0:
            agent_idx, step_idx = collision[0][0], collision[1][0]
            return agent_ids[agent_idx], path[step_idx]

        return None, None

    def check_pushable_collision(self, position, target_position, steps=20):
        path = np.linspace(position, target_position, steps)

        objs = [o for o in self.objects if o.is_pushable]
        if not objs:
            return False, None

        obj_positions = np.array([o.position for o in objs])
        dists = np.linalg.norm(
            obj_positions[:, None, :] - path[None, :, :], axis=2)
        idx = np.where(dists < np.array([o.radius for o in objs])[:, None])

        if idx[0].size > 0:
            return True, objs[idx[0][0]]
        return False, None

    def push_object(self, agent, object, direction_vector):

        full_distance = np.linalg.norm(direction_vector)
        if full_distance == 0:
            return agent.position

        to_obj_dist = np.linalg.norm(
            agent.position - object.position) - object.radius
        remaining_distance = max(0, full_distance - to_obj_dist)
        push_distance = remaining_distance / 2

        push_vector = (direction_vector / full_distance) * push_distance

        collision_point = agent.position + \
            (direction_vector / full_distance) * to_obj_dist

        position = collision_point + push_vector
        object_position = object.position + push_vector

        object_collision = self.path_collides(
            object.position, object_position, radius=object.radius * 0.75, ignore_objects=[object]
        )

        agent_collision = self.path_collides(
            agent.position, position, radius=self.agent_collision_radius, ignore_objects=[
                object]
        )

        if object_collision:
            object_position = self.resolve_collision(
                object.position, object_position, radius=object.radius * 0.75)

        if agent_collision:
            position = self.resolve_collision(
                agent.position, position, radius=self.agent_collision_radius)

        self.objects.remove(object)
        object.position = list(object_position)
        self.objects.append(object)

        separation_vector = np.array(position) - np.array(object.position)
        distance = np.linalg.norm(separation_vector)
        min_distance = object.radius + self.agent_collision_radius + 0.5

        if distance < min_distance:
            if distance == 0:
                separation_vector = np.random.rand(2) - 0.5
                distance = np.linalg.norm(separation_vector)
            correction = separation_vector / distance * min_distance
            position = np.array(object.position) + correction

        return position


class Agent:
    W_APPROACH = 1
    _id_counter = 0

    def __init__(self, group, position, env=None, model=None, facing=[1, 1], ID=None, age=0, max_speed=5, max_age=5_000_000):
        self.group = group
        self.position = position
        self.age = age
        self.model = model
        self.max_age = max_age
        self.max_turn_acceleration = np.pi / 24
        self.turn_velocity = 0.0
        self.turn_friction = 0.5
        self.stale_count = 0

        if ID is None:
            self.ID = str(self._get_next_id())
        else:
            self.ID = str(ID)

        self.env = env
        self.position = np.array(position)
        self.facing = np.array(facing)
        self.prev_dist = None
        self.reward_penalty = 0.0
        self.reward_boost = 0.0
        self.velocity = np.array([0.0, 0.0])
        self.max_speed = max_speed
        self.sequence_buffer = []
        self.buffer = []
        self.hidden_state_buffer = []
        self.initial_sequence_state = None
        self.V = 0
        self.log_prob = {}
        self.previous_position = deque(maxlen=5)
        # for collision-avoidance tracking
        self.collided = False
        self.prev_goal_dist = None
        self.hidden_state = None
        self.last_target_food = None
        self.last_target_fill = 0.0
        self.last_food_dist = None
        # No-hit reward tracking for prey
        self.steps_since_hit = 0
        self.no_hit_reward_increment = 0.03
        self.no_hit_reward_cap = 10

        self.last_visible_fill = 0.0      # highest food-fill seen in current frame
        self.prev_facing = np.array(facing)
        self.visit_counts = defaultdict(int)
        self.prev_facing = np.array(facing)

        self.exploration_coef = 0.5
        self.exploration_damping = 0.7

        self.cell_visit_threshold = 25
        self.over_stay_penalty   = 0.2
    @classmethod
    def _get_next_id(cls):
        cls._id_counter += 1
        return cls._id_counter

    def get_action(self, observation):
        if self.group == 0:
            action = {
                "x_dir": 0,
                "y_dir": 0,
                "spin": 0
            }
            rays = observation["rays"]
            target_dir, _, _, _ = self.get_help_vector(rays)

            # current facing from the obs:
            f = observation["facing"]
            # angle predator is looking (radians)
            θ_f = np.arctan2(f[1], f[0])
            # angle to prey
            θ_t = np.arctan2(target_dir[1], target_dir[0])
            # smallest signed difference in [-π, π]
            Δ = (θ_t - θ_f + np.pi) % (2*np.pi) - np.pi

            # choose spin bin: left, straight, right
            if Δ > +0.1:
                # SPIN_BIN_VALUES[2] == +0.75 rad/sec² (turn right)
                spin_idx = 2
            elif Δ < -0.1:
                # SPIN_BIN_VALUES[0] == -0.75 rad/sec² (turn left)
                spin_idx = 0
            else:
                spin_idx = 1   # no turn

            # discretize attack bins exactly as before
            atk_vals = np.array(ATK_BIN_VALUES)
            x_idx = int(np.argmin(np.abs(atk_vals - target_dir[0])))
            y_idx = int(np.argmin(np.abs(atk_vals - target_dir[1])))

            return {"x_dir": x_idx, "y_dir": y_idx, "spin": spin_idx}, self.hidden_state

        flat, rays = convert_observation(observation)
        if self.hidden_state is None:
            self.hidden_state = self.model.get_initial_state()

        if self.initial_sequence_state is None:
            self.initial_sequence_state = self.hidden_state

        action = self.model.select_action(
            [(flat, rays)], hidden_state=self.hidden_state)
        self.V = action["value"]
        self.log_prob = action["logp_total"]
        self.hidden_state = action["lstm_state"]

        # Custom discretization for group 0 (predators)

        return action, self.hidden_state

    def optimize_model(self):
        if self.group == 0:
            # Predator group does not use the buffer
            return
        buffer = list(self.buffer)
        hidden_states = self.hidden_state_buffer

        self.model.optimize_model(buffer=buffer, hidden_states=hidden_states)

    def gather(self, food):
        self.reward_boost += food * 2.5
        self.stale_count = 0

    def increase_age(self):
        self.age += 1

    def get_reward(self, observation):
        if self.group == 0:
            # Predator group does not use the reward system
            return 0.0

        if self.env.scenario == "navigate":
            curr_pos = self.position
            # attempt to get last position; if not available, use current
            if len(self.previous_position) >= 2:
                prev_pos = self.previous_position[-2]
            else:
                prev_pos = curr_pos
            # movement vector and distance
            movement_vec = curr_pos - prev_pos
            # direction toward goal from previous position
            goal_vec = self.env.goal_pos - prev_pos
            goal_dist = float(np.linalg.norm(goal_vec))
            if goal_dist > 1e-8:
                goal_dir = goal_vec / goal_dist
            else:
                goal_dir = np.zeros_like(goal_vec)
            # project movement onto goal direction (only forward component)
            forward_component = float(np.dot(movement_vec, goal_dir))
            # Non-linear proximity scaling: strong boost within last 7.5 units, slight adjustment otherwise
            threshold = 7.5
            max_dist = np.linalg.norm(self.env.goal_pos - prev_pos)
            if goal_dist <= threshold:
                # strong non-linear boost near goal: quadratic ramp
                proximity = (threshold - goal_dist) / threshold
                scale_factor = 1.25 + proximity * proximity
            elif max_dist > threshold:
                # slight linear adjustment when far: up to +10% at threshold boundary
                deep_proximity = (max_dist - goal_dist) / \
                    (max_dist - threshold)
                scale_factor = 1.0 + 0.1 * deep_proximity
            else:
                scale_factor = 1.0
            reward = self.env.progress_scaling * forward_component * \
                scale_factor - self.env.time_penalty
            # obstacle proximity penalty: penalize being too close to walls
            # rays[:,0] is normalized distance to obstacle [0..1]
            ray_dists = observation["rays"][:, 0]
            min_ray = float(np.min(ray_dists))
            # scale penalty: more severe when closer to obstacles
            if min_ray < 0.25:
                reward -= (1.0 - min_ray)

            # increased collision penalty: double the configured penalty
            if self.collided:
                reward += self.env.collision_penalty * 2.0
            # goal bonus for reaching within tolerance
            if goal_dist <= self.env.config.get("goal_tolerance", 3.0):
                reward += self.env.goal_bonus
            # update for next step
            self.prev_goal_dist = goal_dist
            self.collided = False

            return float(reward)

        elif self.env.scenario == "gather":
            penalty_amount = self.reward_penalty
            reward =- penalty_amount
            self.reward_penalty = 0.0

            food_approach_reward = 0.0

            fill_pcts = observation["rays"][:, -1]
            visible = float(np.max(fill_pcts))
            AWARENESS_THRESHOLD = 15.0
            food_objs = [o for o in self.env.objects if o.is_food]
            if food_objs:
                positions = np.vstack([o.position for o in food_objs])
                radii = np.array([o.radius for o in food_objs])
                shapes_rect = np.array(
                    [o.shape == "rectangle" for o in food_objs])
                dists = np.zeros(len(food_objs), dtype=float)
                if np.any(shapes_rect):
                    rect_pos = positions[shapes_rect]
                    rect_half = radii[shapes_rect]
                    diffs = np.abs(rect_pos - self.position) - \
                        rect_half[:, None]
                    diffs = np.maximum(diffs, 0.0)
                    rect_d = np.hypot(diffs[:, 0], diffs[:, 1])
                    dists[shapes_rect] = rect_d
                circ_mask = ~shapes_rect
                if np.any(circ_mask):
                    circ_pos = positions[circ_mask]
                    circ_rad = radii[circ_mask]
                    raw = np.linalg.norm(
                        circ_pos - self.position, axis=1) - circ_rad
                    raw = np.maximum(raw, 0.0)
                    dists[circ_mask] = raw
                idx = np.argmin(dists)
                nearest_food_obj = food_objs[idx]
                nearest_dist = dists[idx]
            else:
                nearest_food_obj = None
                nearest_dist = float("inf")

            if nearest_food_obj is not None and (visible > 0.0 or nearest_dist <= AWARENESS_THRESHOLD):
                fill_now = nearest_food_obj.food / nearest_food_obj.max_food
                if self.last_food_dist is None:
                    self.last_food_dist = nearest_dist

                # how much closer we got this step
                approach_delta = self.last_food_dist - nearest_dist

                # only reward if there’s actually food and real progress
                if fill_now > 0:
                    fill_factor = fill_now          # 0.0 – 1.0

                    # scale reward by how close we already are
                    closeness = 1.0 - min(nearest_dist, AWARENESS_THRESHOLD) / AWARENESS_THRESHOLD
                    # (alternatively, use 1/(nearest_dist+ε) for sharper ramp-up near)

                    food_approach_reward = approach_delta \
                                        * self.W_APPROACH \
                                        * fill_factor \
                                        * closeness

                    reward += food_approach_reward

                # update for next frame
                self.last_food_dist = nearest_dist
                # update last distance when aware
                self.last_food_dist = nearest_dist
                self.last_target_food = nearest_food_obj
                self.last_target_fill = fill_now
            else:
                # Reset last_food_dist when not aware of field
                self.last_food_dist = None

            boost = self.reward_boost
            reward += boost
            self.reward_boost = 0.0
            cell = (int(self.position[0] // 5), int(self.position[1] // 5))
            cnt  = self.visit_counts[cell]
            if cnt < self.cell_visit_threshold:
                # positive bonus for first few visits
                bonus = self.exploration_coef / ((cnt + 1) ** self.exploration_damping)
            else:
                # after threshold, negative penalty grows with each extra visit
                extra = cnt - self.cell_visit_threshold + 1
                bonus = - self.over_stay_penalty * extra
                bonus = max(bonus, -1.5)  # cap the penalty

            self.visit_counts[cell] += 1
            reward += bonus

            if any(fill_pcts > 0.0):
                reward += 0.5

            return reward

        elif self.env.scenario == "flee":
            # thresholded distance-delta reward
            dist_delta = 0.0
            no_hit_reward = 0.0
            reward = - self.reward_penalty
            self.reward_penalty = 0.0
            # Updated no-hit reward logic
            if not self.was_hit_this_step:
                self.steps_since_hit += 1
                if self.steps_since_hit > 25:
                    # Start rewarding after 25 steps, with a steeper increase
                    step_offset = self.steps_since_hit - 25
                    no_hit_reward = self.no_hit_reward_increment * step_offset
                    if no_hit_reward > self.no_hit_reward_cap:
                        no_hit_reward = self.no_hit_reward_cap
                    reward += no_hit_reward
            else:
                self.steps_since_hit = 0
                reward -= 10.0

            # current distance to closest predator
            predators = [a for a in self.env.agent_data.values()
                         if a.group == 0]
            if predators:
                predator_positions = np.array([p.position for p in predators])
                dists = np.linalg.norm(
                    predator_positions - self.position, axis=1)
                closest_predator_dist = np.min(dists)
                self.dist_to_predator = closest_predator_dist
            else:
                self.dist_to_predator = None
            # reward based on delta distance only within escape radius
            if self.prev_dist is not None and self.dist_to_predator is not None:
                dist_delta = self.dist_to_predator - self.prev_dist
                if dist_delta > 0.0:
                    reward += dist_delta * 1.5
                else:
                    reward += dist_delta * 0.5

            self.prev_dist = self.dist_to_predator

            return reward

        elif self.env.scenario == "full":
            # reward for time survived, food gathered, no collisions, approaching VISIBLE food
            reward = -self.reward_penalty
            self.reward_penalty = 0.0
            fine_cell = (int(self.position[0]//15), int(self.position[1]//15))
            count = self.visit_counts[fine_cell]
            exp_reward = self.exploration_coef / \
                ((count + 1) ** self.exploration_damping)
            self.visit_counts[fine_cell] = count + 1
            reward += exp_reward

            no_hit_reward = 0.0
            dist_delta = 0.0
            if not self.was_hit_this_step:
                self.steps_since_hit += 1
                no_hit_reward = self.no_hit_reward_increment * self.steps_since_hit
                if no_hit_reward > self.no_hit_reward_cap:
                    no_hit_reward = self.no_hit_reward_cap
                reward += no_hit_reward
            else:
                self.steps_since_hit = 0
                reward -= 10.0

            # current distance to closest predator
            predators = [a for a in self.env.agent_data.values()
                         if a.group == 0]
            if predators:
                predator_positions = np.array([p.position for p in predators])
                dists = np.linalg.norm(
                    predator_positions - self.position, axis=1)
                closest_predator_dist = np.min(dists)
                self.dist_to_predator = closest_predator_dist
            else:
                self.dist_to_predator = None
            # reward based on delta distance to closest predator
            if self.prev_dist is not None and self.dist_to_predator is not None:
                dist_delta = self.dist_to_predator - self.prev_dist
                reward += dist_delta * 0.25
            self.prev_dist = self.dist_to_predator
            # reward for food gathered
            food_gathered = self.reward_boost
            reward += food_gathered
            self.reward_boost = 0.0

            # reward for approaching visible food
            food_approach_reward = 0.0
            fill_pcts = observation["rays"][:, -1]
            visible = float(np.max(fill_pcts))
            AWARENESS_THRESHOLD = 15.0
            food_objs = [o for o in self.env.objects if o.is_food]
            if food_objs:
                positions = np.vstack([o.position for o in food_objs])
                radii = np.array([o.radius for o in food_objs])
                shapes_rect = np.array(
                    [o.shape == "rectangle" for o in food_objs])
                dists = np.zeros(len(food_objs), dtype=float)
                if np.any(shapes_rect):
                    rect_pos = positions[shapes_rect]
                    rect_half = radii[shapes_rect]
                    diffs = np.abs(rect_pos - self.position) - \
                        rect_half[:, None]
                    diffs = np.maximum(diffs, 0.0)
                    rect_d = np.hypot(diffs[:, 0], diffs[:, 1])
                    dists[shapes_rect] = rect_d
                circ_mask = ~shapes_rect
                if np.any(circ_mask):
                    circ_pos = positions[circ_mask]
                    circ_rad = radii[circ_mask]
                    raw = np.linalg.norm(
                        circ_pos - self.position, axis=1) - circ_rad
                    raw = np.maximum(raw, 0.0)
                    dists[circ_mask] = raw
                idx = np.argmin(dists)
                nearest_food_obj = food_objs[idx]
                nearest_dist = dists[idx]
            else:
                nearest_food_obj = None
                nearest_dist = float("inf")

            if nearest_food_obj is not None and (visible > 0.0 or nearest_dist <= AWARENESS_THRESHOLD):
                fill_now = nearest_food_obj.food / nearest_food_obj.max_food
                # Only reward approach when outside the field
                if nearest_dist > 0.0:
                    if self.last_food_dist is None:
                        self.last_food_dist = nearest_dist
                    approach_delta = self.last_food_dist - nearest_dist
                    if approach_delta > 0.2 and fill_now > 0:
                        fill_factor = fill_now
                        food_approach_reward = approach_delta * self.W_APPROACH * fill_factor
                        reward += food_approach_reward
                    # Update last_food_dist for next step
                    self.last_food_dist = nearest_dist
                else:
                    # Agent is inside field: no approach reward
                    food_approach_reward = 0.0
                    self.last_food_dist = None
            else:
                # Reset last_food_dist when not aware
                self.last_food_dist = None

            # reward only when visibility genuinely improves
            if visible > self.last_visible_fill + 0.05:
                reward += 2.0 * (visible - self.last_visible_fill)
            self.last_visible_fill = visible

            return reward

    def move(self, accel):
        # reset collision flag at start of move
        self.collided = False
        self.velocity = self.velocity + accel

        delta_theta = self.turn_velocity
        if abs(delta_theta) > 1e-8:
            fx, fy = self.facing
            cos_t, sin_t = np.cos(delta_theta), np.sin(delta_theta)
            self.facing = np.array([fx * cos_t - fy * sin_t,
                                    fx * sin_t + fy * cos_t])
            self.facing /= (np.linalg.norm(self.facing) + 1e-8)
        self.turn_velocity *= self.turn_friction

        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed

        target_pos = self.position + self.velocity
        resolved_pos = self.env.resolve_collision(self.position, target_pos)

        if not np.allclose(resolved_pos, target_pos):
            self.collided = True
            self.reward_penalty += 2.5

        if not np.allclose(resolved_pos, target_pos):
            collision_normal = resolved_pos - target_pos
            if np.linalg.norm(collision_normal) > 1e-5:
                collision_normal /= np.linalg.norm(collision_normal)
                slide_dir = self.velocity - collision_normal * \
                    np.dot(self.velocity, collision_normal)
                self.velocity = slide_dir * 0.25

        actual_movement = np.linalg.norm(resolved_pos - self.position)
        # self.reward_penalty -= actual_movement
        self.stale_count = 0 if actual_movement > 0.3 else self.stale_count + 1

        self.position = resolved_pos
        self.velocity *= 0.9

        self.previous_position.append(self.position)

    def attack(self, direction_vector):
        accel = direction_vector
        self.velocity += accel

        delta_theta = self.turn_velocity
        if abs(delta_theta) > 1e-8:
            fx, fy = self.facing
            cos_t, sin_t = np.cos(delta_theta), np.sin(delta_theta)
            self.facing = np.array([fx * cos_t - fy * sin_t,
                                    fx * sin_t + fy * cos_t])
            self.facing /= (np.linalg.norm(self.facing) + 1e-8)

        self.turn_velocity *= self.turn_friction

        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            lerp_factor = (self.max_speed / speed) ** 0.5
            self.velocity = self.velocity * lerp_factor

        target_position = self.position + self.velocity
        resolved_position = self.env.resolve_collision(
            self.position, target_position)

        if not np.allclose(resolved_position, target_position):
            self.reward_penalty += 4

        actual_movement = np.linalg.norm(resolved_position - self.position)

        if not np.allclose(resolved_position, target_position):
            collision_vector = resolved_position - target_position
            if np.linalg.norm(collision_vector) > 1e-5:
                collision_normal = collision_vector / \
                    np.linalg.norm(collision_vector)

                tangent_velocity = self.velocity - \
                    np.dot(self.velocity, collision_normal) * collision_normal
                self.velocity = tangent_velocity * 0.4
                # --- slide‑along‑obstacle tweak to stop predators getting stuck on rocks ---
                if self.group == 0 and np.allclose(resolved_position, self.position):
                    # Try nudging perpendicular to current facing (left, then right) to “slide” off the obstacle
                    perp = np.array(
                        [-self.facing[1], self.facing[0]], dtype=np.float32)
                    for sign in (1.0, -1.0):
                        nudge = perp * 0.5 * sign      # small sideways step
                        slide_target = self.position + nudge
                        slide_resolved = self.env.resolve_collision(
                            self.position, slide_target)
                        if not np.allclose(slide_resolved, self.position):
                            resolved_position = slide_resolved
                            self.velocity = nudge
                            # update movement metric for stale_count logic
                            actual_movement = np.linalg.norm(
                                resolved_position - self.position)
                            break

        target_agent, agent_collision_point = self.env.check_agent_collision(
            self, self.position, resolved_position
        )

        if target_agent is not None:
            self.position = agent_collision_point
            self.velocity = np.array([0.0, 0.0])
            self.env.agent_data[target_agent].was_hit_this_step = True
            self.env.agent_data[target_agent].reward_penalty += 2.5

        self.position = resolved_position
        self.velocity *= 0.75

        # self.reward_penalty += actual_movement * 0.2
        self.stale_count = 0 if actual_movement > 0.3 else self.stale_count + 1

        # --- extra escape logic for predators boxed in by U‑shaped rivers ---
        if self.group == 0 and self.stale_count >= 3:
            # Rotate facing by a random small angle and take a modest forward step
            angle = random.uniform(-np.pi / 2, np.pi / 2)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            fx, fy = self.facing
            self.facing = np.array([fx * cos_a - fy * sin_a,
                                    fx * sin_a + fy * cos_a])
            self.facing /= (np.linalg.norm(self.facing) + 1e-8)
            self.velocity = self.facing * 0.5
            # Reset stale counter so repeated nudges aren’t immediate
            self.stale_count = 0

        self.previous_position.append(self.position)

    def append_to_buffer(self):
        self.buffer.append(list(self.sequence_buffer))
        self.sequence_buffer.clear()
        self.hidden_state_buffer.append(self.initial_sequence_state)
        # print(len(self.hidden_state_buffer))
        self.initial_sequence_state = None


    def get_help_vector(self, rays):

        env, pos = self.env, self.position
        max_distance = env.map_size * np.sqrt(2.0)

        if self.group == 0:
            prey = [a for a in env.agent_data.values() if a.group == 1]
            if not prey:
                result = (np.zeros(2, np.float32), 0.0,
                        np.zeros(2, np.float32), 0.0)
            else:
                positions = np.vstack([a.position for a in prey])
                dists     = np.linalg.norm(positions - pos, axis=1)
                idx       = np.argmin(dists)
                closest, closest_dist = prey[idx], dists[idx]

                lead_t        = env.config.get("predator_intercept_time", 3.0)
                predicted_pos = closest.position + closest.velocity * lead_t
                predicted_pos = np.clip(predicted_pos, 0.0, env.map_size)

                if not env.path_collides(pos, predicted_pos):
                    direction = (predicted_pos - pos) / (closest_dist + 1e-8)
                else:
                    path = env.find_path(pos, predicted_pos)
                    if path and len(path) > 1:
                        vec = path[1] - pos
                        nrm = np.linalg.norm(vec)
                        direction = vec / nrm if nrm > 1e-8 else np.zeros(2, np.float32)
                    else:
                        base = predicted_pos - pos
                        nrm  = np.linalg.norm(base)
                        base_dir = base / nrm if nrm > 1e-8 else np.zeros(2, np.float32)
                        direction = base_dir          # default
                        for ang in (0, 20, -20, 40, -40, 60, -60, 80, -80):
                            c, s = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
                            rot  = np.array([base_dir[0]*c - base_dir[1]*s,
                                            base_dir[0]*s + base_dir[1]*c],
                                            dtype=np.float32)
                            if not env.path_collides(pos, pos + rot * min(8.0, closest_dist)):
                                direction = rot
                                break

                result = (direction, closest_dist / max_distance,
                        np.zeros(2, np.float32), 0.0)


        elif env.scenario == "flee":
            predators = [a for a in env.agent_data.values() if a.group == 0]
            if not predators:
                result = (np.zeros(2, np.float32), 0.0,
                        np.zeros(2, np.float32), 0.0)
            else:
                positions = np.vstack([a.position for a in predators])
                dists     = np.linalg.norm(positions - pos, axis=1)
                idx       = np.argmin(dists)
                diff      = positions[idx] - pos
                dir_away  = diff / (np.linalg.norm(diff) + 1e-8)
                result    = (np.zeros(2, np.float32), 0.0,
                            dir_away,               dists[idx] / max_distance)

        elif env.scenario == "navigate":
            vec  = env.goal_pos - pos
            dist = np.linalg.norm(vec)
            direction = vec / dist if dist > 1e-8 else np.zeros(2, np.float32)
            result = (direction, dist / max_distance,
                    np.zeros(2, np.float32), 0.0)


        elif env.scenario in ("gather", "full"):

            foods = [o for o in env.objects if getattr(o, "is_food", False)]
            if not foods:
                food_vec, food_dist = np.zeros(2, np.float32), 0.0
            else:
                f_pos  = np.vstack([o.position for o in foods])
                f_rad  = np.array([o.radius for o in foods], np.float32)
                f_rect = np.array([o.shape == "rectangle" for o in foods])

                diff  = pos - f_pos
                dists = np.empty(len(foods), np.float32)
                dirs  = np.empty_like(diff)

                # rectangles
                if f_rect.any():
                    half = f_rad[f_rect][:, None]
                    clamped = np.maximum(np.minimum(diff[f_rect], half), -half)
                    edge_vec = clamped + f_pos[f_rect] - pos
                    dists[f_rect] = np.linalg.norm(edge_vec, axis=1)
                    to_ctr = f_pos[f_rect] - pos
                    dirs[f_rect] = (to_ctr.T /
                                    (np.linalg.norm(to_ctr, axis=1) + 1e-8)).T

                # circles
                if (~f_rect).any():
                    diff_c = diff[~f_rect]
                    dist_c = np.linalg.norm(diff_c, axis=1)
                    dists[~f_rect] = np.maximum(dist_c - f_rad[~f_rect], 0.0)
                    dirs[~f_rect]  = (diff_c.T / (dist_c + 1e-8)).T

                idx = np.argmin(dists)
                food_vec   = dirs[idx]
                food_dist  = dists[idx] / max_distance

                if dists[idx] >= 10.5 and np.max(rays[:, -1]) <= 0.0:
                    food_vec, food_dist = np.zeros(2, np.float32), 0.0


            if env.scenario == "gather":
                result = (food_vec, food_dist,
                        np.zeros(2, np.float32), 0.0)

            else:
                prey = [a for a in env.agent_data.values() if a.group == 1]
                if not prey:
                    prey_vec, prey_dist = np.zeros(2, np.float32), 0.0
                else:
                    positions = np.vstack([a.position for a in prey])
                    dists     = np.linalg.norm(positions - pos, axis=1)
                    idx       = np.argmin(dists)
                    prey_vec  = (positions[idx] - pos) / (dists[idx] + 1e-8)
                    prey_dist = dists[idx] / max_distance

                result = (food_vec, food_dist,
                        prey_vec, prey_dist)


        else:
            result = (np.zeros(2, np.float32), 0.0,
                    np.zeros(2, np.float32), 0.0)

        return result
