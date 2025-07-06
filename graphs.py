import numpy as np
import matplotlib.pyplot as plt
import os
import json
from matplotlib.patches import Patch

def compute_reward_by_window(run_dir, window_size):
    """
    Compute average reward in non-overlapping windows of size `window_size`.
    """
    max_step = 0
    for fname in os.listdir(run_dir):
        path = os.path.join(run_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, 'r') as f:
                count = sum(1 for _ in f)
            max_step = max(max_step, count)
            max_step = min(max_step, 5_000_000)
        except Exception:
            continue
    if max_step <= 0:
        return [], []

    num_windows = (max_step + window_size - 1) // window_size
    time_points = [min(window_size * i, max_step) for i in range(1, num_windows + 1)]

    reward_sums = [0.0] * num_windows
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
                    w = idx // window_size
                    if w < num_windows:
                        reward_sums[w] += data.get("reward", 0)
                        reward_counts[w] += 1
        except Exception:
            continue

    rewards = [(reward_sums[i] / reward_counts[i]) if reward_counts[i] else 0.0 for i in range(num_windows)]
    return rewards, time_points


def plot_normalized_learning_curves(run_dirs_dict, window_size=50000, save_path="graphs/normalized_learning_curves.png"):
    """
    Plot normalized learning curves with mean and ±1 standard deviation shading.

    - Y-axis fixed from 0 to 1.
    - X-axis starts at 0.
    - Shaded area indicates ±1 std dev across runs.
    - Uses a consistent color scheme (tab10) for categories.
    - High-DPI output suitable for publication.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 6), dpi=300)

    smooth_window = 11
    all_rewards = []
    for paths in run_dirs_dict.values():
        for d in paths:
            data_dir = os.path.join(d, "prey") if os.path.isdir(os.path.join(d, "prey")) else d
            r, _ = compute_reward_by_window(data_dir, window_size)
            all_rewards.extend(r)
    global_min, global_max = (min(all_rewards), max(all_rewards)) if all_rewards else (0.0, 1.0)

    mixed = any(len(v) > 1 for v in run_dirs_dict.values())
    cmap = plt.get_cmap('tab10')

    for idx, (cat, paths) in enumerate(run_dirs_dict.items()):
        color = cmap(idx)
        if not mixed and len(paths) == 1:
            d = paths[0]
            data_dir = os.path.join(d, "prey") if os.path.isdir(os.path.join(d, "prey")) else d
            rewards, times = compute_reward_by_window(data_dir, window_size)
            if not rewards:
                continue
            norm = [(r - global_min) / (global_max - global_min) if global_max > global_min else 0.0 for r in rewards]
            if len(norm) >= smooth_window:
                pad = smooth_window // 2
                padded = np.pad(norm, (pad, pad), mode='edge')
                smooth = np.convolve(padded, np.ones(smooth_window)/smooth_window, mode='valid')
                times_smooth = times
            else:
                smooth, times_smooth = norm, times
            plt.plot(times, norm, linestyle=':', linewidth=1, alpha=0.8, color=color)
            plt.plot(times_smooth, smooth, label=cat, linewidth=2, color=color)
        else:
            sm_list, tp_list = [], []
            for d in paths:
                data_dir = os.path.join(d, "prey") if os.path.isdir(os.path.join(d, "prey")) else d
                rewards, times = compute_reward_by_window(data_dir, window_size)
                if not rewards:
                    continue
                norm = [(r - global_min) / (global_max - global_min) if global_max > global_min else 0.0 for r in rewards]
                if len(norm) >= smooth_window:
                    pad = smooth_window // 2
                    padded = np.pad(norm, (pad, pad), mode='edge')
                    smooth = np.convolve(padded, np.ones(smooth_window)/smooth_window, mode='valid')
                    times_smooth = times
                else:
                    smooth, times_smooth = norm, times
                sm_list.append(smooth)
                tp_list.append(times_smooth)
            if not sm_list:
                continue
            min_len = min(len(s) for s in sm_list)
            tp = tp_list[0][:min_len]
            runs_array = np.array([s[:min_len] for s in sm_list])
            mean = runs_array.mean(axis=0)
            std = runs_array.std(axis=0, ddof=1)
            plt.plot(tp, mean, label=cat, linewidth=2, color=color)
            plt.fill_between(tp, mean - std, mean + std, alpha=0.2, color=color)

    plt.xlabel("Training Step", fontsize=14)
    plt.ylabel("Normalized Reward", fontsize=14)
    plt.title("Normalized Learning Curves Comparison", fontsize=16)

    plt.xlim(left=0)
    plt.ylim(0, 1)

    handles, labels = plt.gca().get_legend_handles_labels()
    std_patch = Patch(facecolor='grey', alpha=0.2, label='±1 std dev')
    handles.append(std_patch)
    plt.legend(handles=handles, fontsize=12)

    plt.grid(alpha=0.3, linestyle='--')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    run_dirs_dict = {
        "pretrained flee": [
            "/home/fynnm/full_models/179/2025-07-01-18-22-24",
            "/home/fynnm/full_models/179/2025-07-02-14-41-52",
        ],
        "pretrained navigation": [
            "/home/fynnm/full_models/215/2025-07-01-18-25-14",
            "/home/fynnm/full_models/215/2025-07-02-14-46-48",
        ],
        "pretrained exploration": [
            "/home/fynnm/full_models/models_mac/2025-07-02-14-41-30",
            "/home/fynnm/full_models/models_mac/2025-07-03-22-53-28",
        ],
        "from scratch": [
            "/home/fynnm/full_models/2025-06-30-23-48-16",
            "/home/fynnm/full_models/2025-07-03-23-13-24"
        ]
    }
    plot_normalized_learning_curves(run_dirs_dict)
