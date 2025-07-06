import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

# Publication-quality global styling
plt.rcParams.update({
    'figure.figsize': (8, 6),
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'legend.fontsize': 11
})

# Paths and loading
data_root = 'evaluation_logs'
metrics_file = 'evaluation_data.pkl'
os.makedirs(data_root, exist_ok=True)
with open(metrics_file, 'rb') as f:
    data = pickle.load(f)
summary = data['summary']
raw_metrics = data['raw_metrics']

# Label formatting: break long names and customize "Scratch"
def wrap_label(label):
    return label.replace(' ', '\n', 1)
label_map = {
    'Flee': 'Flee\n(Pretrained)',
    'Navigate': 'Navigate\n(Pretrained)',
    'Explore': 'Explore\n(Pretrained)',
    'Scratch': 'From\nScratch'
}
models = list(summary.keys())
nice_labels = [label_map.get(m, wrap_label(m)) for m in models]

# Color setup
cmap = plt.get_cmap('tab10')
bar_colors = [cmap(i) for i in range(len(models))]

# Function to save bar charts with half-thickness black borders
def save_bar_chart(key, ylabel, title):
    values = [summary[m].get(key, 0) for m in models]
    fig, ax = plt.subplots()
    bars = ax.bar(
        nice_labels,
        values,
        color=bar_colors,
        edgecolor='black',
        linewidth=0.5
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    filename = f"{key}_comparison.png"
    fig.savefig(os.path.join(data_root, filename), dpi=300)
    plt.close(fig)
    print(f"Saved: {filename}")

# Generate official-quality comparison bars
stats = [
    ('avg_time_alive', 'Time Alive (steps)', 'Time Alive Across Models'),
    ('avg_time_to_goal', 'Time to Goal (steps)', 'Time to Goal Across Models'),
    ('avg_reward_per_step', 'Reward per Step', 'Average Reward per Step Across Models'),
    ('avg_episode_length', 'Episode Length (steps)', 'Average Episode Length Across Models'),
    ('sum_food_collected', 'Food Collected', 'Total Food Collected Across Models')
]
for key, ylabel, title in stats:
    save_bar_chart(key, ylabel, title)

# Unified term colors for pies
term_reasons = sorted({r for info in raw_metrics.values() for r in info['termination_counts']})
term_colors = {r: cmap(i) for i, r in enumerate(term_reasons)}

# Pie charts with legend percentages and outlines
for model, info in raw_metrics.items():
    counts = info['termination_counts']
    labels = list(counts.keys())
    sizes = np.array([counts[r] for r in labels])
    colors = [term_colors[r] for r in labels]
    total = sizes.sum()
    legend_entries = [f"{r} ({cnt/total*100:.1f}%)" for r, cnt in zip(labels, sizes)]
    fig, ax = plt.subplots()
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops={'edgecolor':'black','linewidth':0.5}
    )
    ax.legend(
        wedges,
        legend_entries,
        title='Termination Reasons',
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
    model_label = label_map.get(model, wrap_label(model))
    ax.set_title(f'Termination Breakdown: {model_label}')
    ax.axis('equal')
    fig.tight_layout()
    filename = f"{model}_termination_pie.png"
    fig.savefig(os.path.join(data_root, filename), dpi=300)
    plt.close(fig)
    print(f"Saved: {filename}")

# Successful termination comparison
reason_key = 'cleared'
fig, ax = plt.subplots()
success_rates = [
    (info['termination_counts'].get(reason_key, 0) / sum(info['termination_counts'].values()) * 100)
    if sum(info['termination_counts'].values()) > 0 else 0
    for info in raw_metrics.values()
]
bars = ax.bar(
    nice_labels,
    success_rates,
    color=bar_colors,
    edgecolor='black',
    linewidth=0.5
)
ax.set_ylabel('Successful Terminations (%)')
ax.set_title('Successful Termination Rates Across Models')
plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
for bar in bars:
    h = bar.get_height()
    ax.annotate(f'{h:.1f}%%',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=10)
fig.tight_layout()
filename = 'successful_comparison.png'
fig.savefig(os.path.join(data_root, filename), dpi=300)
plt.close(fig)
print(f"Saved: {filename}")
