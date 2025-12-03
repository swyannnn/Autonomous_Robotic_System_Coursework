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
def plot_ci_compare(
    x_base, mean_base, low_base, high_base,
    x_par, mean_par, low_par, high_par,
    interval, title, output
):
    """
    Plot comparison of two methods with 95% confidence intervals.
    Args:
        x_base (np.ndarray): Episode axis for Base PPO.
        mean_base (np.ndarray): Mean episodic return for Base PPO.
        low_base (np.ndarray): Lower bound of 95% CI for Base PPO.
        high_base (np.ndarray): Upper bound of 95% CI for Base PPO.
        x_par (np.ndarray): Episode axis for Parseval PPO.
        mean_par (np.ndarray): Mean episodic return for Parseval PPO.
        low_par (np.ndarray): Lower bound of 95% CI for Parseval PPO.
        high_par (np.ndarray): Upper bound of 95% CI for Parseval PPO.
        interval (int): Episode interval for task switches.
        title (str): Title of the plot.
        output (str): Path to save the output plot.
    Returns:
        None
    """

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(12, 5))

    # ------------------------------------------------------
    # Plot Base PPO (smooth mean + thick CI band)
    # ------------------------------------------------------
    plt.plot(x_base, mean_base, color="blue", linewidth=2.2, label="Base PPO")
    plt.fill_between(x_base, low_base, high_base, color="blue", alpha=0.18)

    # ------------------------------------------------------
    # Plot Parseval PPO (smooth mean + thick CI band)
    # ------------------------------------------------------
    plt.plot(x_par, mean_par, color="red", linewidth=2.2, label="Parseval PPO")
    plt.fill_between(x_par, low_par, high_par, color="red", alpha=0.18)

    # ------------------------------------------------------
    # Task switch lines + labels
    # ------------------------------------------------------
    num_tasks = 4
    ymax = max(high_base.max(), high_par.max())
    text_y = ymax * 0.80

    # Draw T1 manually at the start
    plt.text(110, text_y, "T1", fontsize=12, weight="bold")

    for i in range(1, 4):  # or however many boundaries you want
        ep = i * interval
        plt.axvline(ep, linestyle="--", color="gray", linewidth=1.2, alpha=0.7)

        # Cyclic task index: 1→T2, 2→T3, 3→T4, 4→T1, 5→T2, ...
        task_id = (i % num_tasks) + 1     # maps 1→2, 2→3, 3→4, 4→1 correctly

        plt.text(ep + 110, text_y, f"T{task_id}", fontsize=12, weight="bold")


    # ------------------------------------------------------
    # Labels & aesthetics
    # ------------------------------------------------------
    plt.xlabel("Episode", fontsize=13)
    plt.ylabel("Episodic Return", fontsize=13)
    plt.title(title, fontsize=15, weight="bold")
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(output, dpi=300)
    print(f"[SAVED] {output}")


# ----------------------------------------------------------
# Main CLI entry
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_path", required=True)
    parser.add_argument("--parseval_path", required=True)
    parser.add_argument("--tag", default="episodic_return")
    parser.add_argument("--max_x", type=int, default=1200)


    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--title", default="Base vs Parseval PPO (95% CI)")
    parser.add_argument("--output", default="compare_ci_clean.png")

    args = parser.parse_args()

    xb, rb = load_data(
        f"{args.base_path}/avg_{args.tag}_runs.npy",
        f"{args.base_path}/avg_{args.tag}_stats.json"
    )

    xp, rp = load_data(
        f"{args.parseval_path}/avg_{args.tag}_runs.npy",
        f"{args.parseval_path}/avg_{args.tag}_stats.json"
    )

    # Apply truncation if max_x is specified
    if args.max_x is not None:
        # Base PPO slice
        mask_b = xb <= args.max_x
        xb = xb[mask_b]
        rb = rb[:, mask_b]

        # Parseval PPO slice
        mask_p = xp <= args.max_x
        xp = xp[mask_p]
        rp = rp[:, mask_p]

    # Compute confidence intervals
    mean_b, low_b, high_b = compute_95_ci(rb)
    mean_p, low_p, high_p = compute_95_ci(rp)

    # Plot comparison
    plot_ci_compare(
        xb, mean_b, low_b, high_b,
        xp, mean_p, low_p, high_p,
        args.interval,
        args.title,
        args.output
    )

# example usage:
# python plot_compare_ci.py \
# --base_path "results/CartPole-v1/base" \
# --parseval_path "results/CartPole-v1/parseval" \
# --interval 300 \
# --title "Base vs Parseval PPO on CartPole-v1 (95% Confidence Interval)" \
# --output "results/CartPole-v1/compare_ci.png"

# python plot_compare_ci.py \
# --base_path "results/Acrobot-v1/base" \
# --parseval_path "results/Acrobot-v1/parseval" \
# --interval 300 \
# --title "Base vs Parseval PPO on Acrobot-v1 (95% Confidence Interval )" \
# --output "results/Acrobot-v1/compare_ci.png"

# python plot_compare_ci.py \
# --base_path "results/Acrobot-v1/base" \
# --parseval_path "results/Acrobot-v1/parseval" \
# --interval 300 \
# --title "Critic Cosine Similarity Layer 2" \
# --output "results/Acrobot-v1/compare_ci_critic_cosine_sim_layer_2.png" \
# --tag "critic_cosine_sim_layer_2"

# python plot_compare_ci.py \
# --base_path "results/Acrobot-v1/base" \
# --parseval_path "results/Acrobot-v1/parseval" \
# --interval 300 \
# --title "Actor Cosine Similarity Layer 2" \
# --output "results/Acrobot-v1/compare_ci_actor_cosine_sim_layer_2.png" \
# --tag "actor_cosine_sim_layer_2"

# python plot_compare_ci.py \
# --base_path "results/CartPole-v1/base" \
# --parseval_path "results/CartPole-v1/parseval" \
# --interval 300 \
# --title "Critic Cosine Similarity Layer 2" \
# --output "results/CartPole-v1/compare_ci_critic_cosine_sim_layer_2.png" \
# --tag "critic_cosine_sim_layer_2"

# python plot_compare_ci.py \
# --base_path "results/CartPole-v1/base" \
# --parseval_path "results/CartPole-v1/parseval" \
# --interval 300 \
# --title "Actor Cosine Similarity Layer 2" \
# --output "results/CartPole-v1/compare_ci_actor_cosine_sim_layer_2.png" \
# --tag "actor_cosine_sim_layer_2"