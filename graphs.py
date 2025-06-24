import numpy as np
import matplotlib.pyplot as plt

# Helper and plotting functions
import os
import json


def compute_reward_by_window(run_dir, window_size):
    print(
        f"compute_reward_by_window: run_dir={run_dir}, window_size={window_size}")
    # Scan all .jsonl files in run_dir for reward per line
    max_step = 0
    for fname in os.listdir(run_dir):
        path = os.path.join(run_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'r') as f:
                count = sum(1 for _ in f)
            max_step = max(max_step, count)
        except Exception:
            continue

    print(f"  max_step = {max_step}")

    if max_step <= 0:
        return [], []

    num_windows = (max_step + window_size - 1) // window_size
    time_points = [min(window_size * i, max_step)
                   for i in range(1, num_windows + 1)]

    print(f"  num_windows = {num_windows}, time_points = {time_points}")

    reward_sums = [0] * num_windows
    reward_counts = [0] * num_windows

    for fname in os.listdir(run_dir):
        path = os.path.join(run_dir, fname)
        try:
            with open(path, 'r') as f:
                for idx, line in enumerate(f):
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    window_idx = idx // window_size
                    if window_idx >= num_windows:
                        continue
                    reward_sums[window_idx] += data.get("reward", 0)
                    reward_counts[window_idx] += 1
        except Exception:
            continue

    print(f"  reward_sums = {reward_sums}")
    print(f"  reward_counts = {reward_counts}")

    rewards = []
    for i in range(num_windows):
        if reward_counts[i]:
            rewards.append(reward_sums[i] / reward_counts[i])
        else:
            rewards.append(0)
    print(f"  rewards = {rewards}")
    return rewards, time_points


def plot_normalized_learning_curves(run_dirs_dict, window_size=25000, save_path="graphs/normalized_learning_curves.png"):
    print(
        f"plot_normalized_learning_curves: run_dirs_dict={run_dirs_dict}, window_size={window_size}, save_path={save_path}")
    os.makedirs("graphs", exist_ok=True)
    plt.figure(figsize=(12, 6))  # wider figure for readability
    # window size for moving average smoothing (increased for stronger smoothing)
    smooth_window = 11

    # iterate over categories, each mapping to multiple run paths
    for category, run_paths in run_dirs_dict.items():
        all_smoothed = []
        all_time_points = []
        # process each run in this category
        for run_dir in run_paths:
            data_dir = run_dir
            prey_dir = os.path.join(run_dir, "prey")
            if os.path.isdir(prey_dir):
                data_dir = prey_dir
            print(f"  [{category}] data_dir = {data_dir}")
            rewards, time_points = compute_reward_by_window(
                data_dir, window_size)
            if not rewards:
                print(f"  [{category}] no data in {data_dir}, skipping")
                continue
            # normalize per-run
            min_r, max_r = min(rewards), max(rewards)
            if max_r > min_r:
                norm = [(r - min_r) / (max_r - min_r) for r in rewards]
            else:
                norm = [0] * len(rewards)
            # smooth per-run
            if len(norm) >= smooth_window:
                smooth = np.convolve(norm, np.ones(
                    smooth_window)/smooth_window, mode='same')
            else:
                smooth = norm
            all_smoothed.append(smooth)
            all_time_points.append(time_points)
        # skip if no runs contributed
        if not all_smoothed:
            continue
        # align lengths by trimming to shortest
        min_len = min(len(s) for s in all_smoothed)
        tp = all_time_points[0][:min_len]
        arr = np.array([s[:min_len] for s in all_smoothed])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        # plot mean trend
        line, = plt.plot(tp, mean, label=category, linewidth=2)
        # shaded deviation
        plt.fill_between(tp, mean - std, mean + std,
                         alpha=0.2, color=line.get_color())

    plt.xlabel("Training Step", fontsize=14)
    plt.ylabel("Normalized Reward", fontsize=14)
    plt.title("Normalized Learning Curves Comparison", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# get mutliple runs and plot comparison

if __name__ == "__main__":
    # mapping category names to lists of run directories
    run_dirs_dict = {
        "good_flee": ["/Users/fynnmadrian/plutonian_insects/good_flee"],
        "recent_logs": ["/Users/fynnmadrian/plutonian_insects/logs/2025-06-20-00-07-29", "/Users/fynnmadrian/plutonian_insects/logs/2025-06-19-10-35-46"]
    }
    plot_normalized_learning_curves(run_dirs_dict)
