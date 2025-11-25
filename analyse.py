import os
import numpy as np
from glob import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse


# ------------------------------------------------------
# Load scalars from TensorBoard tfevents (NO TensorFlow)
# ------------------------------------------------------
def load_scalars(folder, tag):
    event_files = glob(os.path.join(folder, "**", "*.tfevents.*"), recursive=True)
    if len(event_files) == 0:
        raise ValueError(f"No event files found in: {folder}")

    event_file = sorted(event_files)[-1]  # pick latest

    ea = EventAccumulator(event_file)
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        raise ValueError(f"Tag '{tag}' not found in logs: {folder}")

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


# ------------------------------------------------------
# Extract episodic returns + task IDs
# ------------------------------------------------------
def extract_task_episode_pairs(folder):
    _, returns = load_scalars(folder, "charts/episodic_return")
    _, task_ids = load_scalars(folder, "charts/task_id_episode")

    # Align lengths
    L = min(len(returns), len(task_ids))
    returns = returns[:L]
    task_ids = task_ids[:L].astype(int)
    episodes = np.arange(L)

    return episodes, task_ids, returns


# ------------------------------------------------------
# Build the Task × Task evaluation matrix
# ------------------------------------------------------
def build_eval_matrix(task_ids, returns, num_tasks):
    # find episode indices when each task finishes
    boundaries = []
    cur = task_ids[0]
    for i in range(1, len(task_ids)):
        if task_ids[i] != cur:
            boundaries.append(i - 1)
            cur = task_ids[i]
    boundaries.append(len(task_ids) - 1)

    R = np.zeros((num_tasks, num_tasks))

    for i in range(num_tasks):
        end_idx = boundaries[i]

        for j in range(num_tasks):
            mask = task_ids[:end_idx + 1] == j
            if np.sum(mask) > 0:
                R[i, j] = np.mean(returns[:end_idx + 1][mask])
            else:
                R[i, j] = np.nan

    return R


# ------------------------------------------------------
# Continual Learning Metrics
# ------------------------------------------------------
def continual_rl_metrics(R):
    num_tasks = R.shape[0]

    # Average performance
    AP = np.mean([R[i, i] for i in range(num_tasks)])

    # Forgetting
    forgetting = np.zeros(num_tasks)
    for t in range(1, num_tasks):
        best_prev = np.nanmax(R[:t, t])
        final = R[-1, t]
        forgetting[t] = best_prev - final

    overall_F = np.mean(forgetting[1:])

    # Forward Transfer
    FT = np.zeros(num_tasks - 1)
    for t in range(num_tasks - 1):
        FT[t] = R[t, t + 1] - R[t, t]

    return {
        "R": R,
        "AP": AP,
        "Forgetting_per_task": forgetting,
        "Overall_forgetting": overall_F,
        "Forward_transfer": FT
    }


# ------------------------------------------------------
# Pretty print results
# ------------------------------------------------------
def print_matrix(R):
    print("\n=== Task x Task Evaluation Matrix (R) ===")
    for row in R:
        print(" ".join(f"{x:7.2f}" if not np.isnan(x) else "   nan " for x in row))


def print_results(res, label):
    print("\n==============================")
    print(" RESULTS FOR:", label)
    print("==============================")
    print_matrix(res["R"])
    print("\nAverage Performance (AP):", round(res["AP"], 3))
    print("Forgetting per task:", res["Forgetting_per_task"])
    print("Overall Forgetting (F):", round(res["Overall_forgetting"], 3))
    print("Forward Transfer (FT):", res["Forward_transfer"], "\n")


# ------------------------------------------------------
# Main
# ------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="folder with Base PPO tfevents")
    parser.add_argument("--parseval", required=True, help="folder with Parseval PPO tfevents")
    parser.add_argument("--tasks_num", type=int, default=4, help="number of tasks")
    args = parser.parse_args()

    print("Loading Base PPO logs...")
    _, task_ids, returns = extract_task_episode_pairs(args.base)
    R_base = build_eval_matrix(task_ids, returns, args.tasks_num)
    base_res = continual_rl_metrics(R_base)

    print("Loading Parseval PPO logs...")
    _, task_ids, returns = extract_task_episode_pairs(args.parseval)
    R_parseval = build_eval_matrix(task_ids, returns, args.tasks_num)
    parseval_res = continual_rl_metrics(R_parseval)

    print_results(base_res, "Base PPO")
    print_results(parseval_res, "Parseval PPO")

# example usage:
# python report.py --base /media/nine/HD_1/HD_2_from_seven/Yann/robotics/COMP4082_ARS/scratch/runs/CartPole-v1__main__1__1764075972 \
# --parseval /media/nine/HD_1/HD_2_from_seven/Yann/robotics/COMP4082_ARS/scratch/runs/CartPole-v1__parseval__1__1764078329 --tasks_num 4