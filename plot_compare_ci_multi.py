import numpy as np
import json
import matplotlib.pyplot as plt
import argparse


def load_data(npy_file, json_file):
    """
    Load data from exported npy + json metadata
    Args:
        npy_file (str): Path to the .npy file containing runs data.
        json_file (str): Path to the .json file containing metadata.
    Returns:
        x (np.ndarray): Episode axis values.
        runs (np.ndarray): Array of shape [num_seeds, num_points] with episodic returns.
    """
    runs = np.load(npy_file)                     # shape: [num_seeds, num_points]
    with open(json_file, "r") as f:
        stats = json.load(f)
    x = np.array(stats["x"], dtype=float)        # episode axis
    return x, runs

def compute_95_ci(runs):
    """
    Compute 95% confidence interval for given runs data.
    Args:
        runs (np.ndarray): Array of shape [num_seeds, num_points] with episodic returns.
    Returns:
        mean (np.ndarray): Mean episodic return across runs.
        low (np.ndarray): Lower bound of 95% confidence interval.
        high (np.ndarray): Upper bound of 95% confidence interval.
    """
    mean = runs.mean(axis=0)
    std = runs.std(axis=0)
    n = runs.shape[0]

    se = std / np.sqrt(n)              # Standard error
    ci95 = 1.96 * se                   # 95% confidence interval

    low = mean - ci95
    high = mean + ci95

    return mean, low, high


# ----------------------------------------------------------
# Plot comparison (publication-ready)
# ----------------------------------------------------------
def plot_ci_compare_multi(datasets, interval, title, output):
    plt.style.use("seaborn-v0_8-white")
    plt.figure(figsize=(12, 5))

    # Four unique colours
    colors = {
        "CartPole-Base": "blue",
        "CartPole-Parseval": "red",
        "Acrobot-Base": "green",
        "Acrobot-Parseval": "orange",
    }

    ymax = -999

    # Plot all curves
    for label, x, mean, low, high in datasets:
        color = colors[label]

        plt.plot(x, mean, color=color, linewidth=2.2, label=label)
        plt.fill_between(x, low, high, color=color, alpha=0.15)

        ymax = max(ymax, high.max())

    # Task-switch markers
    text_y = ymax * 0.80
    plt.text(110, text_y, "T1", fontsize=12, weight="bold")

    for i in range(1, 4):
        ep = i * interval
        plt.axvline(ep, linestyle="--", color="gray", linewidth=1.2, alpha=0.6)
        task_id = (i % 4) + 1
        plt.text(ep + 110, text_y, f"T{task_id}", fontsize=12, weight="bold")

    plt.xlabel("Episode", fontsize=13)
    plt.ylabel("Episodic Return", fontsize=13)
    plt.title(title, fontsize=15, weight="bold")
    plt.legend(fontsize=11, ncol=2)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"[SAVED] {output}")

# ----------------------------------------------------------
# Main CLI entry
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cartpole_base", required=True)
    parser.add_argument("--cartpole_parseval", required=True)
    parser.add_argument("--acrobot_base", required=True)
    parser.add_argument("--acrobot_parseval", required=True)
    parser.add_argument("--tag", default="episodic_return")
    parser.add_argument("--max_x", type=int, default=1200)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--title", default="Base vs Parseval PPO (95% CI)")
    parser.add_argument("--output", default="compare_ci_clean.png")

    args = parser.parse_args()

    # Load all four datasets
    xb_c, rb_c = load_data(
        f"{args.cartpole_base}/avg_{args.tag}_runs.npy",
        f"{args.cartpole_base}/avg_{args.tag}_stats.json"
    )

    xp_c, rp_c = load_data(
        f"{args.cartpole_parseval}/avg_{args.tag}_runs.npy",
        f"{args.cartpole_parseval}/avg_{args.tag}_stats.json"
    )

    xb_a, rb_a = load_data(
        f"{args.acrobot_base}/avg_{args.tag}_runs.npy",
        f"{args.acrobot_base}/avg_{args.tag}_stats.json"
    )

    xp_a, rp_a = load_data(
        f"{args.acrobot_parseval}/avg_{args.tag}_runs.npy",
        f"{args.acrobot_parseval}/avg_{args.tag}_stats.json"
    )

    # Apply truncation
    if args.max_x:
        mask = xb_c <= args.max_x; xb_c = xb_c[mask]; rb_c = rb_c[:, mask]
        mask = xp_c <= args.max_x; xp_c = xp_c[mask]; rp_c = rp_c[:, mask]
        mask = xb_a <= args.max_x; xb_a = xb_a[mask]; rb_a = rb_a[:, mask]
        mask = xp_a <= args.max_x; xp_a = xp_a[mask]; rp_a = rp_a[:, mask]

    # Compute CI
    mean_cb, low_cb, high_cb = compute_95_ci(rb_c)
    mean_cp, low_cp, high_cp = compute_95_ci(rp_c)
    mean_ab, low_ab, high_ab = compute_95_ci(rb_a)
    mean_ap, low_ap, high_ap = compute_95_ci(rp_a)

    # Combine into one structure
    datasets = [
        ("CartPole-Base", xb_c, mean_cb, low_cb, high_cb),
        ("CartPole-Parseval", xp_c, mean_cp, low_cp, high_cp),
        ("Acrobot-Base", xb_a, mean_ab, low_ab, high_ab),
        ("Acrobot-Parseval", xp_a, mean_ap, low_ap, high_ap),
    ]

    # Single 4-curve plot
    plot_ci_compare_multi(
        datasets,
        args.interval,
        args.title,
        args.output,
    )


# example usage:
# python plot_compare_ci_multi.py \
#     --cartpole_base results/CartPole-v1/base \
#     --cartpole_parseval results/CartPole-v1/parseval \
#     --acrobot_base results/Acrobot-v1/base \
#     --acrobot_parseval results/Acrobot-v1/parseval \
#     --tag episodic_return \
#     --interval 300 \
#     --max_x 1200 \
#     --title "CI Comparison Across Both Environments" \
#     --output results/episodic_return.png

# python plot_compare_ci_multi.py \
#     --cartpole_base results/CartPole-v1/base \
#     --cartpole_parseval results/CartPole-v1/parseval \
#     --acrobot_base results/Acrobot-v1/base \
#     --acrobot_parseval results/Acrobot-v1/parseval \
#     --tag actor_cosine_sim_layer_2 \
#     --interval 300 \
#     --max_x 1200 \
#     --title "CI Comparison Across Both Environments" \
#     --output results/actor_cosine_sim_layer_2.png

# python plot_compare_ci_multi.py \
#     --cartpole_base results/CartPole-v1/base \
#     --cartpole_parseval results/CartPole-v1/parseval \
#     --acrobot_base results/Acrobot-v1/base \
#     --acrobot_parseval results/Acrobot-v1/parseval \
#     --tag critic_cosine_sim_layer_2 \
#     --interval 300 \
#     --max_x 1200 \
#     --title "CI Comparison Across Both Environments" \
#     --output results/critic_cosine_sim_layer_2.png