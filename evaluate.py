import matplotlib.pyplot as plt
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
from collections import defaultdict
import pickle
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

def evaluate(seed, max_steps=480, log_dir=None, model_path=None):
    """
    Run one episode under the FULL scenario.
    Returns per-episode metrics, with None for metrics that didn't occur:
      - time_alive: steps until prey is eaten (None if not eaten)
      - time_to_goal: steps until a field is emptied (None if no field cleared)
      - total_reward: cumulative reward across steps
      - episode_length: number of steps run (termination or max_steps)
      - total_food_collected: total food picked up by prey
      - termination_reason: 'eaten', 'cleared', or 'timeout'
    """
    config = evaluation_config.copy()
    config['render_enabled'] = bool(log_dir)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    env = CustomEnvironment(config=config, seed=seed, folder_path=log_dir)
    observations, infos = env.reset()

    if model_path is None:
        raise ValueError("Must provide model_path to evaluate()")
    prey = next(a for a in env.agent_data.values() if a.group == 1)
    prey.model.load_weights(model_path)

    total_reward = 0.0
    total_food = 0.0
    time_alive = None
    time_to_goal = None
    termination = 'timeout'

    for step in range(max_steps):
        actions = {}
        for agent_id, agent in env.agent_data.items():
            obs = observations[agent_id]
            action, _ = agent.get_action(obs)
            actions[agent_id] = action

        observations, rewards, terminations, truncations, infos = env.step(actions)
        total_reward += rewards.get(str(prey.ID), 0.0)
        total_food += infos.get(str(prey.ID), {}).get('food_collected', 0.0)

        if terminations.get(str(prey.ID), False):
            if getattr(prey, 'was_hit_this_step', False):
                time_alive = step + 1
                termination = 'eaten'
            elif env.any_field_depleted:
                time_to_goal = step + 1
                termination = 'cleared'
            break

    ep_length = step + 1
    env.flush_logs()
    return {
        'time_alive': time_alive,
        'time_to_goal': time_to_goal,
        'total_reward': total_reward,
        'episode_length': ep_length,
        'total_food_collected': total_food,
        'termination_reason': termination
    }

# --------------------------------------------------
# Aggregation and comparative plotting
# --------------------------------------------------
def aggregate_and_plot(models_dict, seeds=range(1000), max_steps=480, log_root='evaluation_logs'):
    os.makedirs(log_root, exist_ok=True)
    summary = {}
    raw_metrics = {}
    # Collect metrics per label
    for label, model_paths in models_dict.items():
        metrics = defaultdict(list)
        term_counts = {'eaten': 0, 'cleared': 0, 'timeout': 0}

        for model_path in model_paths:
            for seed in seeds:
                result = evaluate(
                    seed,
                    max_steps,
                    log_dir=os.path.join(log_root, label, f'run_{seed}'),
                    model_path=model_path
                )
                # time metrics
                if result['time_alive'] is not None:
                    metrics['time_alive'].append(result['time_alive'])
                if result['time_to_goal'] is not None:
                    metrics['time_to_goal'].append(result['time_to_goal'])
                # reward per step
                rpt = result['total_reward'] / result['episode_length']
                metrics['reward_per_step'].append(rpt)
                # other metrics
                metrics['episode_length'].append(result['episode_length'])
                metrics['total_food_collected'].append(result['total_food_collected'])
                term_counts[result['termination_reason']] += 1
        raw_metrics[label] = {'metrics': metrics, 'termination_counts': term_counts}
        summary[label] = {
            'avg_time_alive': np.mean(metrics['time_alive']) if metrics['time_alive'] else None,
            'avg_time_to_goal': np.mean(metrics['time_to_goal']) if metrics['time_to_goal'] else None,
            'avg_reward_per_step': np.mean(metrics['reward_per_step']),
            'avg_episode_length': np.mean(metrics['episode_length']),
            'sum_food_collected': np.sum(metrics['total_food_collected']),
            'termination_counts': term_counts
        }
    save_path = os.path.join(log_root, 'evaluation_data.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({'summary': summary, 'raw_metrics': raw_metrics}, f)
    print(f"Saved aggregated data to {save_path}")
    labels = list(summary.keys())
    # Comparative bar charts for each metric
    comp_metrics = [
        ('avg_time_alive', 'Average Time Alive'),
        ('avg_time_to_goal', 'Average Time to Goal'),
        ('avg_reward_per_step', 'Average Reward per Step'),
        ('avg_episode_length', 'Average Episode Length'),
        ('sum_food_collected', 'Total Food Collected')
    ]

    for key, title in comp_metrics:
        values = [summary[label][key] or 0 for label in labels]
        plt.figure()
        plt.bar(labels, values)
        plt.ylabel(title)
        plt.title(f'{title} by Model Group')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(log_root, f'{key}_comparison.png'))
        plt.close()

    # Comparative pie charts for termination reasons per label
    for label in labels:
        counts = list(summary[label]['termination_counts'].values())
        reasons = list(summary[label]['termination_counts'].keys())
        plt.figure()
        plt.pie(counts, labels=reasons, autopct='%1.1f%%', startangle=90)
        plt.title(f'Termination Breakdown: {label}')
        plt.tight_layout()
        plt.savefig(os.path.join(log_root, f'{label}_termination_pie.png'))
        plt.close()

    return summary



if __name__ == "__main__":
    evaluation_config = {
        "map_size": 100,
        "base_population_per_group": 1,
        "max_age": 480,
        "scenario": "full",
        "map_config": {
            "Rock": 6,
            "River": 1,
            "Field": 1,
            "Forest": 0,
            "Field_food_range": [10, 20],
            "Field_base_radius": 15,
            "Field_max_food": 35,
            "River_base_radius": 5,
            "Rock_base_radius": 5,
        },
        "render_enabled": True,
        "predator_fov": 120,
        "prey_fov": 180,
        "vision_range": 35,
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
    models_dict = {
    "Flee": [
        "/home/fynnm/full_models/179/2025-07-01-18-22-24/models/prey/agent_1_model_5000000.weights.h5",
        "/home/fynnm/full_models/179/2025-07-02-14-41-52/models/prey/agent_1_model_5000000.weights.h5",
    ],
    "Navigate": [
        "/home/fynnm/full_models/215/2025-07-01-18-25-14/models/prey/agent_1_model_5000000.weights.h5",
        "/home/fynnm/full_models/215/2025-07-02-14-46-48/models/prey/agent_1_model_5000000.weights.h5",
    ],
    "Explore": [
        "/home/fynnm/full_models/models_mac/2025-07-02-14-41-30/models/prey/agent_1_model_5000000.weights.h5",
        "/home/fynnm/full_models/models_mac/2025-07-03-22-53-28/models/prey/agent_1_model_5000000.weights.h5",
    ],
    "Scratch": ["/home/fynnm/full_models/2025-06-30-23-48-16/full_from_scratch_new.weights.h5",
    "/home/fynnm/full_models/2025-07-03-23-13-24/models/prey/agent_1_model_5000000.weights.h5"]
    }

    aggregate_and_plot(models_dict)
