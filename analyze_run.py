import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

run = sorted(os.listdir("logs"))

current_run = [r for r in run if os.path.isdir(os.path.join("logs", r))][-1]

run_path = os.path.join("./logs", current_run)

full_rewards = {"prey": []}
full_food_collected = {"prey": []}
full_pred_hit = {"prey": []}


species_paths = {
    species: os.path.join(run_path, species)
    for species in ["prey"]
}


window_size = 25000

# Streaming computation of windowed metrics without loading entire files
# Determine max_step across all species by counting lines
max_step = 0
for species, species_path in species_paths.items():
    for fname in os.listdir(species_path):
        path = os.path.join(species_path, fname)
        try:
            with open(path, 'r') as f:
                count = sum(1 for _ in f)
            max_step = max(max_step, count)
        except Exception:
            continue

print(f"Max step across all agents: {max_step}")

# Compute number of windows and time points
if max_step > 0:
    num_windows = (max_step + window_size - 1) // window_size
    time_points = [min(window_size * i, max_step)
                   for i in range(1, num_windows + 1)]
else:
    time_points = []
    num_windows = 0

# Initialize sums and counts per species per window
reward_sums = {s: [0] * num_windows for s in species_paths}
reward_counts = {s: [0] * num_windows for s in species_paths}
food_sums = {s: [0] * num_windows for s in species_paths}
food_counts = {s: [0] * num_windows for s in species_paths}

# Initialize heatmap bins for positions
heatmap = np.zeros((25, 25), dtype=int)
x_bin_size = 100 / 25
y_bin_size = 100 / 25

# Process each file line by line
for species, species_path in species_paths.items():
    for fname in os.listdir(species_path):
        path = os.path.join(species_path, fname)
        try:
            with open(path, 'r') as f:
                for line_idx, line in enumerate(f):
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    window_idx = line_idx // window_size
                    if window_idx >= num_windows:
                        continue
                    # Update reward sums and counts
                    reward = data.get("reward", 0)
                    reward_sums[species][window_idx] += reward
                    reward_counts[species][window_idx] += 1
                    # Update food sums and counts
                    food = data.get("food_collected", 0)
                    food_sums[species][window_idx] += food
                    food_counts[species][window_idx] += 1
                    # Update heatmap for position
                    pos = data.get("position")
                    if pos and isinstance(pos, list) and len(pos) == 2:
                        x, y = pos
                        x_idx = int(x / x_bin_size)
                        y_idx = int(y / y_bin_size)
                        if 0 <= x_idx < 25 and 0 <= y_idx < 25:
                            heatmap[x_idx, y_idx] += 1
        except Exception:
            continue

# Compute average metrics per window
reward_by_species = {s: [] for s in species_paths}
for species in species_paths:
    for i in range(num_windows):
        if reward_counts[species][i]:
            reward_by_species[species].append(
                reward_sums[species][i] / reward_counts[species][i])
        else:
            reward_by_species[species].append(0)

food_by_species = {s: [] for s in species_paths}
for species in species_paths:
    for i in range(num_windows):
        if food_counts[species][i]:
            food_by_species[species].append(
                food_sums[species][i] / food_counts[species][i])
        else:
            food_by_species[species].append(0)

# Assign to full collections
full_rewards = {s: reward_by_species[s] for s in species_paths}
full_food_collected = {s: food_by_species[s] for s in species_paths}
# hit metric not computed streaming
full_pred_hit = {s: [0] * num_windows for s in species_paths}

print(f"Average reward per {window_size}-step window:")
for species, rewards in reward_by_species.items():
    print(f"{species}: {rewards}")

# Plot average reward per window
time_points = list(range(len(full_rewards["prey"])))
plt.figure(figsize=(10, 5))
plt.plot(time_points, full_rewards["prey"], label="Prey")
os.makedirs("graphs", exist_ok=True)
plt.xlabel("Window")
plt.ylabel("Average Reward")
plt.title(f"Average Reward per {window_size}-step Window by Species")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"graphs/rewards.png")
plt.close()

print("Saved: graphs/rewards.png")

# Plot average food collected per window
time_windows = list(range(len(full_food_collected["prey"])))
plt.figure(figsize=(10, 5))
plt.plot(time_windows,
         full_food_collected["prey"], label="Prey Food Collected")
plt.xlabel("Window")
plt.ylabel("Average Food Collected")
plt.title("Average Food Collected per Window by Species")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/food_collected_by_window.png")
plt.close()

print("Saved: graphs/food_collected_by_window.png")

# Plot position heatmap
if np.any(heatmap):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        heatmap.T, origin='lower',
        extent=[0, 100, 0, 100],
        aspect='auto'
    )
    fig.colorbar(im, ax=ax)
    ax.set_title("Agent Position Heatmap")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    os.makedirs("graphs", exist_ok=True)
    fig.savefig("graphs/position_heatmap.png")
    plt.close(fig)

print("Saved: graphs/position_heatmap.png")


metrics_path = os.path.join(run_path, "train_metrics.jsonl")

# Initialize lists for each metric
steps = []
loss_total = []
loss_policy = []
loss_value = []
entropy_mean = []
entropy_coef = []
grad_norm = []

with open(metrics_path) as f:
    for idx, line in enumerate(f):
        steps.append(idx)
        data = json.loads(line)
        loss_total.append(data["loss_total"])
        loss_policy.append(data["loss_policy"])
        loss_value.append(data["loss_value"])
        entropy_mean.append(data["entropy_mean"])
        entropy_coef.append(data["entropy_coef"])
        grad_norm.append(data["grad_norm"])

# Dictionary of metric names to their corresponding lists
metrics = {
    "loss_total": loss_total,
    "loss_policy": loss_policy,
    "loss_value": loss_value,
    "entropy_mean": entropy_mean,
    "entropy_coef": entropy_coef,
    "grad_norm": grad_norm,
}

# Plot each metric over steps and save to graphs directory
for metric_name, values in metrics.items():
    plt.figure(figsize=(10, 5))
    plt.plot(steps, values, label=metric_name)
    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} over Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"graphs/{metric_name}.png")
    plt.close()

print("Saved training metrics plots")

# === Plotting time to goal per episode (streaming) ===
episode_lengths = []
for species, species_path in species_paths.items():
    for fname in os.listdir(species_path):
        path = os.path.join(species_path, fname)
        try:
            with open(path, 'r') as f:
                prev_step = 0
                for line_idx, line in enumerate(f):
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("termination") == True or data.get("truncation") == True:
                        current_step = line_idx
                        length = current_step - prev_step
                        episode_lengths.append(length)
                        prev_step = current_step
        except Exception:
            continue

print(f"Total episodes: {len(episode_lengths)}")
print(f"Episode lengths: {episode_lengths}")

if episode_lengths:
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(episode_lengths)),
             episode_lengths, marker="o", linestyle="-")
    plt.xlabel("Episode Index")
    plt.ylabel("Episode Length (steps)")
    plt.title("Time to Goal (or Max Length if Not Reached) per Episode")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("graphs", exist_ok=True)
    plt.savefig("graphs/episode_lengths.png")
    plt.close()
    print("Saved: graphs/episode_lengths.png")
