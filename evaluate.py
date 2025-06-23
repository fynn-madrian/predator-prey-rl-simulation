import tensorflow as tf
import os
import numpy as np
import json
import pandas as pd
import cv2
import shutil
from algorithms import AgentNet
from custom_environment import CustomEnvironment, Agent
from algorithms import load_model, get_model
from helpers import convert_observation, collides_with
from render import render


# metrics for each scenario
# gather: average time between food pickups, overall food collected
# navigate: average distance to goal, time taken to reach goal
# flee: average distance to predator, time alive
# full: average distance to goal, time taken to reach goal, average distance to predator, time alive

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


def create_video_for_run(log_dir, start_step=0, end_step=None, seed=None):
    tmp_dir = "tmp_visualizations"
    os.makedirs(tmp_dir, exist_ok=True)

    env_path = os.path.join(log_dir, "environment.jsonl")
    # Read entire environment log once
    with open(env_path, "r") as f:
        all_env_lines = f.readlines()

    last_logged_step = len(all_env_lines) - 1
    # If caller supplied no end_step or an oversized one, clamp it
    if end_step is None or end_step > last_logged_step:
        end_step = last_logged_step

    env_data = []
    for idx in range(start_step, end_step + 1):
        entry = json.loads(all_env_lines[idx])
        entry["step"] = idx
        env_data.append(entry)

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
    video_name = os.path.join(log_dir, f'video_{seed}.avi')

    video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(tmp_dir, image)))

    video.release()
    cv2.destroyAllWindows()
    shutil.rmtree(tmp_dir)


def evaluate(seed, steps=480, log_dir=None, model_path=None):

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

    # For food pickup intervals
    pickup_intervals = []
    last_pickup_step = 0

    if model_path is None:
        raise ValueError("Must provide model_path to evaluate()")
    if env.scenario == "gather" or env.scenario == "navigate":
        predators = []
        for agent in env.agent_data.values():
            if agent.group == 0:
                predators.append(agent)
        for agent in predators:
            env.remove_agent(agent.ID)

    prey_agent = [a for a in env.agent_data.values() if a.group == 1][0]
    prey_agent.model.load_weights(
        model_path)

    for step in range(steps):
        actions = {}
        for agent_id, agent in env.agent_data.items():
            obs = observations[agent_id]
            action, lstm_state = agent.get_action(obs)
            actions[agent_id] = action

        observations, rewards, terminations, truncations, infos = env.step(
            actions)
        total_reward += rewards[prey_agent.ID]
        total_steps += 1

        done = terminations[str(prey_agent.ID)
                            ] or truncations[str(prey_agent.ID)]
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
            if food > 0:
                interval = step - last_pickup_step
                pickup_intervals.append(interval)
                last_pickup_step = step
            total_food_collected += food
            if done:
                break

        elif env.scenario == "full":
            # Combine navigate and flee metrics
            prey_pos = env.agent_data[prey_agent.ID].position
            goal_dist = 0
            total_distance_to_goal += goal_dist
            predator_positions = [
                agent.position for agent in env.agent_data.values() if agent.group == 0]
            food = infos[prey_agent.ID].get("food_collected", 0)
            if food > 0:
                interval = step - last_pickup_step
                pickup_intervals.append(interval)
                last_pickup_step = step
            total_food_collected += food
            if predator_positions:
                dists = [np.linalg.norm(pos - prey_pos)
                         for pos in predator_positions]
                total_distance_to_predator += min(dists)
            if not done:
                time_alive += 1
            else:
                break

    # Compute average time between food pickups
    if pickup_intervals:
        avg_time_between_pickups = sum(
            pickup_intervals) / len(pickup_intervals)
    else:
        avg_time_between_pickups = total_steps
    # For navigate scenario, average time to goal; otherwise zero
    if env.scenario == "navigate":
        avg_time_to_goal = total_steps
    else:
        avg_time_to_goal = 0

    env.flush_logs()
    return {
        "total_reward": total_reward,
        "average_steps": total_steps / steps if steps > 0 else 0.0,
        "average_distance_to_goal": total_distance_to_goal / steps if steps > 0 else 0.0,
        "average_distance_to_predator": total_distance_to_predator / steps if steps > 0 else 0.0,
        "time_alive": time_alive,
        "total_food_collected": total_food_collected,
        "average_time_between_pickups": avg_time_between_pickups,
        "average_time_to_goal": avg_time_to_goal
    }


def aggregate_and_log(model_path, num_runs=100, steps=1000, top_k=5, log_root="evaluation_logs"):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model_dir = os.path.join(log_root, model_name)
    os.makedirs(model_dir, exist_ok=True)
    results = []
    for seed in range(num_runs):
        run_log_dir = os.path.join(model_dir, f"run_{seed}")
        res = evaluate(seed, steps, log_dir=run_log_dir,
                       model_path=model_path)
        results.append((seed, res))
    # Compute average metrics
    metrics_sum = {}
    for _, res in results:
        for k, v in res.items():
            metrics_sum[k] = metrics_sum.get(k, 0) + v
    avg_metrics = {k: metrics_sum[k]/num_runs for k in metrics_sum}
    # Write metrics and config/seeds to text file
    metrics_file = os.path.join(model_dir, "metrics.txt")
    with open(metrics_file, "w") as mf:
        mf.write(f"Model: {model_path}\n")
        mf.write(f"Config: {evaluation_config}\n")
        mf.write(f"Seeds: {list(range(num_runs))}\n")
        mf.write("Average Metrics:\n")
        for k, v in avg_metrics.items():
            mf.write(f"{k}: {v}\n")

    # Select top runs according to scenario-specific criteria
    scenario = evaluation_config["scenario"]
    if scenario == "flee":
        # longest time alive
        results.sort(key=lambda x: x[1]["time_alive"], reverse=True)
    elif scenario in ("gather", "navigate"):
        # shortest episodes (fewest steps)
        results.sort(key=lambda x: x[1]["average_steps"])
    elif scenario == "full":
        # highest reward
        results.sort(key=lambda x: x[1]["total_reward"], reverse=True)
    else:
        # fallback to highest reward
        results.sort(key=lambda x: x[1]["total_reward"], reverse=True)
    top_runs = results[:top_k]

    visualize = True
    if visualize:
        # Generate videos for all runs under the model directory
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        model_dir = os.path.join("evaluation_logs", model_name)
        for seed, _ in top_runs:
            print(f"Creating video for run {seed} in {model_dir}")
            run_dir = os.path.join(model_dir, f"run_{seed}")
            create_video_for_run(run_dir, start_step=0,
                                 end_step=steps, seed=seed)


if __name__ == "__main__":
    evaluation_config = {
        "map_size": 100,
        "base_population_per_group": 1,
        "reproduction_cooldown": 100,
        "max_age": 480,
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
            "Rock_base_radius": 6,
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
    model_path = "/Users/fynnmadrian/Downloads/model/flee_full.weights.h5"
    aggregate_and_log(model_path)
