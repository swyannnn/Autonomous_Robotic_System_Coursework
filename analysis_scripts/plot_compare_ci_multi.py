# EXAMPLE USAGE:
# python plot_compare_ci_multi.py \
#     --cartpole_base results/CartPole-v1/base \
#     --cartpole_parseval results/CartPole-v1/parseval \
#     --acrobot_base results/Acrobot-v1/base \
#     --acrobot_parseval results/Acrobot-v1/parseval \
#     --tag episodic_return \
#     --interval 300 \
#     --max_x 1200 \
#     --title "Episodic Return Across Environments: Base vs Parseval PPO (95% CI)" \
#     --output results/episodic_return.png
 
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse


def load_data(npy_file: str, json_file: str) -> tuple:
    """
    Load run data from .npy and .json files.
    Args:
        npy_file (str): Path to the .npy file containing run data.
        json_file (str): Path to the .json file containing stats data.
    Returns:
        x (np.ndarray): X-axis values (e.g., episodes).
        runs (np.ndarray): Run data array of shape [num_seeds, num_points].
    """
    runs = np.load(npy_file)  # shape: [num_seeds, num_points]
    with open(json_file, "r") as f:
        stats = json.load(f)
    x = np.array(stats["x"], dtype=float)
    return x, runs


def compute_95_ci(runs: np.ndarray) -> tuple:
    """
    Compute 95% confidence intervals for the given runs.
    Args:
        runs (np.ndarray): Array of shape [num_seeds, num_points].
    Returns:
        mean (np.ndarray): Mean values across runs.
        low (np.ndarray): Lower bound of 95% CI.
        high (np.ndarray): Upper bound of 95% CI.
    """
    mean = runs.mean(axis=0)
    std = runs.std(axis=0)
    n = runs.shape[0]
    se = std / np.sqrt(n)
    ci95 = 1.96 * se

    return mean, mean - ci95, mean + ci95


def plot_ci_compare_multi(datasets: list, interval: int, title: str, output: str) -> None:
    """
    Plot multiple datasets with 95% confidence intervals.
    Args:
        datasets (list): List of tuples (label, x, mean, low, high).
        interval (int): Interval for task markers.
        title (str): Plot title.
        output (str): Output file path.
    Returns:
        None
    """
    plt.style.use("seaborn-v0_8-white")
    plt.figure(figsize=(12, 5))

    colors = {
        "CartPole-Base": "blue",
        "CartPole-Parseval": "red",
        "Acrobot-Base": "green",
        "Acrobot-Parseval": "orange",
    }

    ymax = -1e9

    for label, x, mean, low, high in datasets:
        color = colors[label]

        plt.plot(x, mean, color=color, linewidth=2.2, label=label)
        plt.fill_between(x, low, high, color=color, alpha=0.15)

        ymax = max(ymax, high.max())

    # Task markers (still optional)
    text_y = ymax * 0.70
    plt.text(x[0] + 120, text_y, "T1", fontsize=12, weight="bold")

    # Only draw lines inside the plotted range
    for i in range(1, 4):
        ep = i * interval
        if x[0] <= ep <= x[-1]:
            plt.axvline(ep, linestyle="--", color="gray", linewidth=1.2, alpha=0.6)
            plt.text(ep + 120, text_y, f"T{i+1}", fontsize=12, weight="bold")

    plt.xlabel("Episode", fontsize=13)
    plt.ylabel("Episodic Return", fontsize=13)
    plt.title(title, fontsize=15, weight="bold")
    plt.legend(fontsize=11, ncol=2)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"[SAVED] {output}")

def main(args):
    """
    Main function to load data, compute CIs, and plot comparisons.
    Args:
        args: Command-line arguments.
    Returns:
        None
    """
    # Load CartPole
    xb_c, rb_c = load_data(f"{args.cartpole_base}/avg_{args.tag}_runs.npy",
                           f"{args.cartpole_base}/avg_{args.tag}_stats.json")
    xp_c, rp_c = load_data(f"{args.cartpole_parseval}/avg_{args.tag}_runs.npy",
                           f"{args.cartpole_parseval}/avg_{args.tag}_stats.json")

    # Load Acrobot
    xb_a, rb_a = load_data(f"{args.acrobot_base}/avg_{args.tag}_runs.npy",
                           f"{args.acrobot_base}/avg_{args.tag}_stats.json")
    xp_a, rp_a = load_data(f"{args.acrobot_parseval}/avg_{args.tag}_runs.npy",
                           f"{args.acrobot_parseval}/avg_{args.tag}_stats.json")

    # ----- APPLY RANGE SLICE -----
    def slice_range(x, runs, min_x, max_x):
        mask = (x >= min_x) & (x <= max_x)
        return x[mask], runs[:, mask]

    xb_c, rb_c = slice_range(xb_c, rb_c, args.min_x, args.max_x)
    xp_c, rp_c = slice_range(xp_c, rp_c, args.min_x, args.max_x)
    xb_a, rb_a = slice_range(xb_a, rb_a, args.min_x, args.max_x)
    xp_a, rp_a = slice_range(xp_a, rp_a, args.min_x, args.max_x)

    # Compute CI
    mean_cb, low_cb, high_cb = compute_95_ci(rb_c)
    mean_cp, low_cp, high_cp = compute_95_ci(rp_c)
    mean_ab, low_ab, high_ab = compute_95_ci(rb_a)
    mean_ap, low_ap, high_ap = compute_95_ci(rp_a)

    # TODO: remove hardcoding of +600 for Acrobot x-axis shift
    datasets = [
        ("CartPole-Base", xb_c, mean_cb, low_cb, high_cb),
        ("CartPole-Parseval", xp_c, mean_cp, low_cp, high_cp),
        ("Acrobot-Base", xb_a, mean_ab, low_ab, high_ab),
        ("Acrobot-Parseval", xp_a, mean_ap, low_ap, high_ap),
    ]

    plot_ci_compare_multi(
        datasets=datasets,
        interval=args.interval,
        title=args.title,
        output=args.output,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cartpole_base", required=True)
    parser.add_argument("--cartpole_parseval", required=True)
    parser.add_argument("--acrobot_base", required=True)
    parser.add_argument("--acrobot_parseval", required=True)
    parser.add_argument("--tag", default="episodic_return")
    parser.add_argument("--min_x", type=int, default=0)
    parser.add_argument("--max_x", type=int, default=1200)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--title", default="Base vs Parseval PPO (95% CI)")
    parser.add_argument("--output", default="compare_ci_clean.png")

    args = parser.parse_args()
    main(args)