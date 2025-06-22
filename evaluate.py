import tensorflow as tf
import os
import numpy as np
import random
from custom_environment import CustomEnvironment
from algorithms import load_model, get_model
from helpers import convert_observation, collides_with

evaluation_config = {
    "map_size": 100,
    "base_population_per_group": 1,
    "reproduction_cooldown": 100,
    "max_age": 5_000_000,
    "scenario": "full",
    "map_config": {
        "Rock": 6,
        "River": 1,
        "Field": 1,
        "Forest": 0,
        "Field_food_range": [10, 20],
        "Field_base_radius": 12,
        "Field_max_food": 25,
        "River_base_radius": 5,
        "Rock_base_radius": 2,
    },
    "render_enabled": True,
    "predator_fov": 120,
    "prey_fov": 180,
    "vision_range": 20,
    "vision_rays": 15,
    "agent_detection_radius": 1,
    "agent_collision_radius": 2,
    "buffer_size": 16,
    "mutation_rate": 0.0,
    "sequence_length": 32,
    "max_speed": 7.5,
    "stale_truncation": 100,
    "max_agent_count": 2,
}


# metrics for each scenario
# gather: average time between food pickups, overall food collected
# navigate: average distance to goal, time taken to reach goal
# flee: average distance to predator, time alive
# full: average distance to goal, time taken to reach goal, average distance to predator, time alive


def evaluate(seed, steps=1000, log_dir=None):

    # Disable standard logging; only enable if log_dir is provided
    run_config = evaluation_config.copy()
    run_config["render_enabled"] = bool(log_dir)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    env = CustomEnvironment(config=run_config, seed=seed, folder_path=log_dir)
    observations, infos = env.reset()

    total_reward = 0
    total_steps = 0
    total_distance_to_goal = 0
    total_distance_to_predator = 0
    time_alive = 0
    total_food_collected = 0

    model_path = "good_flee/models/prey/agent_2_model_5000000.weights.h5"
    if env.scenario == "gather" or env.scenario == "navigate":
        predators = []
        for agent in env.agent_data.values():
            if agent.group == 0:
                predators.append(agent)
        for agent in predators:
            env.remove_agent(agent.ID)

    prey_agent = [a for a in env.agent_data.values() if a.group == 1][0]
    prey_agent.model.load_weights(model_path)

    for step in range(steps):
        actions = {}
        obs = observations[prey_agent.ID]
        for agent_id, agent in env.agent_data.items():
            obs = observations[agent_id]
            action, lstm_state = agent.get_action(obs)
            actions[agent_id] = action

        observations, rewards, terminations, truncations, infos = env.step(
            actions)
        total_reward += rewards[prey_agent.ID]
        total_steps += 1

        done = terminations[prey_agent.ID] or truncations[prey_agent.ID]

        if env.scenario == "navigate":
            # Compute distance to goal directly
            prey_pos = env.agent_data[prey_agent.ID].position
            goal_dist = np.linalg.norm(prey_pos - env.goal_pos)
            total_distance_to_goal += goal_dist
            if done:
                break

        elif env.scenario == "flee":
            # Compute distance to closest predator
            prey_pos = env.agent_data[prey_agent.ID].position
            predator_positions = [
                agent.position for agent in env.agent_data.values() if agent.group == 0]
            if predator_positions:
                dists = [np.linalg.norm(pos - prey_pos)
                         for pos in predator_positions]
                total_distance_to_predator += min(dists)
            if not done:
                time_alive += 1
            else:
                break

        elif env.scenario == "gather":
            # Accumulate food collected from infos
            food = infos[prey_agent.ID].get("food_collected", 0)
            total_food_collected += food

        elif env.scenario == "full":
            # Combine navigate and flee metrics
            prey_pos = env.agent_data[prey_agent.ID].position
            goal_dist = 0
            total_distance_to_goal += goal_dist
            predator_positions = [
                agent.position for agent in env.agent_data.values() if agent.group == 0]
            if predator_positions:
                dists = [np.linalg.norm(pos - prey_pos)
                         for pos in predator_positions]
                total_distance_to_predator += min(dists)
            if not done:
                time_alive += 1
            else:
                break

    # Compute average time between pickups for gather scenario
    avg_time_between = total_food_collected and (
        total_steps / total_food_collected) or 0.0

    return {
        "total_reward": total_reward,
        "average_steps": total_steps / steps if steps > 0 else 0.0,
        "average_distance_to_goal": total_distance_to_goal / steps if steps > 0 else 0.0,
        "average_distance_to_predator": total_distance_to_predator / steps if steps > 0 else 0.0,
        "time_alive": time_alive,
        "total_food_collected": total_food_collected,
        "average_time_between_pickups": avg_time_between
    }


def aggregate_and_log(num_runs=100, steps=1000, top_k=5, log_root="evaluation_logs"):
    results = []
    for seed in range(num_runs):
        res = evaluate(seed, steps, log_dir=log_root)
        results.append((seed, res))
    # Compute average metrics
    metrics_sum = {}
    for _, res in results:
        for k, v in res.items():
            metrics_sum[k] = metrics_sum.get(k, 0) + v
    avg_metrics = {k: metrics_sum[k]/num_runs for k in metrics_sum}
    print("Average Metrics over {} runs:".format(num_runs))
    for k, v in avg_metrics.items():
        print(f"{k}: {v}")
    # Log top K episodes
    os.makedirs(log_root, exist_ok=True)
    # Sort by survival time
    top_seeds = sorted(results, key=lambda x: x[1].get(
        "time_alive", 0), reverse=True)[:top_k]
    for seed, _ in top_seeds:
        dirpath = os.path.join(log_root, f"episode_{seed}")
        print(f"Logging episode {seed} to {dirpath}")
        evaluate(seed, steps, log_dir=dirpath)


if __name__ == "__main__":
    aggregate_and_log()
    exit()
    # create animation for top seeps

    import shutil
    import json
    import cv2
    import pandas as pd

    from custom_environment import Agent
    from render import render

    def reconstruct_objects(obj_dicts):
        from environment_object import River, Field, Rock, Forest
        objs = []
        for obj in obj_dicts:
            if obj["type"] == "River":
                instance = River(obj["points"], radius=obj["radius"])
            elif obj["type"] == "Field":
                instance = Field(obj["position"], obj["food"],
                                 obj["max_food"], radius=obj["radius"])
            elif obj["type"] == "Rock":
                instance = Rock(obj["position"], radius=obj["radius"])
            elif obj["type"] == "Forest":
                instance = Forest(obj["position"], radius=obj["radius"])
            objs.append(instance)
        return objs

    def create_video_for_run(log_dir, start_step=0, end_step=1000):
        tmp_dir = "tmp_visualizations"
        os.makedirs(tmp_dir, exist_ok=True)

        env_path = os.path.join(log_dir, "environment.jsonl")

        env_data = []
        with open(env_path, "r") as f:
            for idx, line in enumerate(f):
                entry = json.loads(line)
                entry["step"] = idx
                if start_step <= idx <= end_step:
                    env_data.append(entry)
                if idx == end_step:
                    break

        species_dirs = ["prey", "predator"]
        agent_data_by_id = {}

        for species_dir in species_dirs:
            species = 0 if species_dir == "predator" else 1
            species_path = os.path.join(log_dir, species_dir)
            if not os.path.isdir(species_path):
                continue
            for file in os.listdir(species_path):
                agent_id = file.replace("agent_", "").replace(".jsonl", "")
                file_path = os.path.join(species_path, file)
                agent_records = []
                with open(file_path, "r") as af:
                    for idx, line in enumerate(af):
                        if start_step <= idx <= end_step:
                            record = json.loads(line)
                            record["step"] = idx
                            agent_records.append(record)
                        if idx > end_step:
                            break
                if agent_records:
                    df = pd.DataFrame(agent_records)
                    agent_data_by_id[agent_id] = [df, species]

        for env_entry in env_data:
            step = env_entry["step"]
            objects = reconstruct_objects(env_entry["objects"])
            goal = env_entry.get("goal", None)

            agents = {}
            for agent_id, (df, species) in agent_data_by_id.items():
                row = df[df["step"] == step]
                if row.empty:
                    continue
                row = row.iloc[0]
                agent = Agent(
                    group=species,
                    position=row["position"],
                    age=row["age"],
                    facing=row["facing"],
                    max_speed=5,
                    max_age=10000,
                    ID=int(agent_id)
                )
                agent.velocity = row["velocity"]
                agents[agent_id] = agent

            if agents or objects:
                render(objects, agents,
                       savedir=os.path.join(tmp_dir, f"{step}.png"), goal=goal)

        images = [img for img in os.listdir(tmp_dir) if img.endswith(".png")]
        images = [img for img in images if start_step <=
                  int(img.split(".")[0]) <= end_step]
        images.sort(key=lambda x: int(x.split(".")[0]))

        if not images:
            print(
                f"No images found for run in {log_dir}, skipping video generation.")
            return

        frame = cv2.imread(os.path.join(tmp_dir, images[0]))
        height, width, layers = frame.shape
        video_name = os.path.join(log_dir, f'video_{start_step}.avi')

        video = cv2.VideoWriter(
            video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(tmp_dir, image)))

        video.release()
        cv2.destroyAllWindows()
        shutil.rmtree(tmp_dir)

    for run in os.listdir("evaluation_logs"):
        log_dir = os.path.join("evaluation_logs", run)
        if os.path.isdir(log_dir) and "episode_" in run:
            print(f"Creating video for {log_dir}")
            create_video_for_run(log_dir)