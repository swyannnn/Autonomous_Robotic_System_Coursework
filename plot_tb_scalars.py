import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os
import json
import csv

def load_scalar(event_file, tag):
    """
    Loads a scalar tag from a TensorBoard event file.
    Returns steps and values as numpy arrays.
    Args:
        event_file (str): Path to the TensorBoard event file.
        tag (str): The scalar tag to load.
    Returns:
        steps (np.ndarray): Array of steps.
        values (np.ndarray): Array of scalar values.
    """
    ea = EventAccumulator(event_file)
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        return None, None

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=np.float32)
    values = np.array([e.value for e in events], dtype=np.float32)
    return steps, values


def load_runs(run_dirs, tag):
    """
    Loads the specified scalar tag from multiple TensorBoard event files.
    Args:
        run_dirs (list of str): List of directories containing TensorBoard event files.
        tag (str): The scalar tag to load.
    Returns:
        all_steps (list of np.ndarray): List of steps arrays for each run.
        all_values (list of np.ndarray): List of values arrays for each run.
    """
    all_steps, all_values = [], []

    for run in run_dirs:
        event_files = glob.glob(os.path.join(run, "events.*"))
        if len(event_files) == 0:
            continue

        event_file = sorted(event_files)[-1]
        print(f"[LOAD] {event_file}")
        s, v = load_scalar(event_file, tag)
        if s is None:
            print(f"[WARN] Tag {tag} missing in {event_file}")
            continue

        all_steps.append(s)
        all_values.append(v)

    return all_steps, all_values

def extract_switch_points(steps, interval, cycles):
    """
    Finds the closest steps to the target evaluation points.
    Args:
        steps (np.ndarray): Array of steps.
        interval (int): Interval between tasks.
        cycles (int): Number of tasks.
    Returns:
        switch_points (list of int): List of steps closest to each target evaluation point.
    """
    switch_points = []
    targets = np.arange(interval, interval * (cycles + 1), interval)

    for t in targets:
        idx = (np.abs(steps - t)).argmin()
        switch_points.append(steps[idx])
    return switch_points


def build_performance_matrix(run_dir, num_tasks, interval, cycles):
    """
    Builds a performance matrix from TensorBoard logs for a single run.
    Args:
        run_dir (str): Directory containing TensorBoard event files.
        num_tasks (int): Number of tasks.
        interval (int): Interval between tasks.
        cycles (int): Number of tasks.
    Returns:
        perf_mat (np.ndarray): Performance matrix of shape (num_tasks, cycles)."""
    perf_mat = np.zeros((num_tasks, cycles))

    event_files = glob.glob(os.path.join(run_dir, "events.*"))
    if len(event_files) == 0:
        print(f"[WARN] No event files in {run_dir}")
        return None

    event_file = sorted(event_files)[-1]
    ea = EventAccumulator(event_file)
    ea.Reload()

    for t in range(num_tasks):
        tag = f"eval/task_{t}/mean_return"
        if tag not in ea.Tags().get("scalars", []):
            print(f"[WARN] Missing tag {tag} in {event_file}")
            return None

        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])

        switch_points = extract_switch_points(steps, interval, cycles)

        for i, sp in enumerate(switch_points):
            idx = (np.abs(steps - sp)).argmin()
            perf_mat[t, i] = values[idx]

    return perf_mat

def compute_cl_metrics(perf):
    """
    Computes Continual Learning metrics from the performance matrix.
    Args:
        perf (np.ndarray): Performance matrix of shape (T, C).
    Returns:
        metrics (dict): Dictionary containing CL metrics.
    """
    T, C = perf.shape

    diag = np.array([perf[t, t] for t in range(T)])
    final = perf[:, C - 1]

    bwt_per_task = final[:T - 1] - diag[:T - 1]
    BWT = bwt_per_task.mean()

    CF = -BWT
    CF_tasks = -bwt_per_task

    fwt_values = [perf[j, j - 1] for j in range(1, T)]
    FWT = np.mean(fwt_values)

    SPB = FWT - abs(BWT)

    return {
        "diag": diag.tolist(),
        "final": final.tolist(),
        "FWT": float(FWT),
        "BWT": float(BWT),
        "CF": float(CF),
        "SPB": float(SPB),
        "CF_per_task": CF_tasks.tolist(),
        "BWT_per_task": bwt_per_task.tolist(),
    }

def interpolate(all_steps, all_values, num_points=2000):
    """
    Interpolates multiple runs to a common x-axis.
    Args:
        all_steps (list of np.ndarray): List of steps arrays for each run.
        all_values (list of np.ndarray): List of values arrays for each run.
        num_points (int): Number of points for interpolation.
    Returns:
        common_x (np.ndarray): Common x-axis.
        interpolated (np.ndarray): Interpolated values of shape (num_runs, num_points).
    """
    xmin = max(s[0] for s in all_steps)
    xmax = min(s[-1] for s in all_steps)
    common_x = np.linspace(xmin, xmax, num_points)
    interpolated = [np.interp(common_x, s, v) for s, v in zip(all_steps, all_values)]
    return common_x, np.stack(interpolated)

def plot_task_segments(
    x,
    runs,
    output,
    title,
    ma_window=51,
):
    """
    Plots the mean and standard deviation of runs with task segments.
    Args:
        x (np.ndarray): Common x-axis.
        runs (np.ndarray): Interpolated values of shape (num_runs, num_points).
        interval (int): Interval between tasks.
        output (str): Output file path for the plot.
        title (str): Title of the plot.
        num_tasks (int): Number of tasks.
        cycles (int): Number of tasks.
        ma_window (int): Window size for moving average.
    """
    mean = runs.mean(axis=0)
    std = runs.std(axis=0)

    # ----- SAVE MEAN/STDEV -----
    json.dump({
        "x": x.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }, open(output.replace(".png", "_stats.json"), "w"), indent=4)

    np.save(output.replace(".png", "_runs.npy"), runs)

    print(f"[SAVED] episodic return stats → {output.replace('.png','_stats.json')}")
    print(f"[SAVED] interpolated runs → {output.replace('.png','_runs.npy')}")

    # ----- PLOT -----
    plt.figure(figsize=(10, 6))
    mean_smoothed = mean
    if ma_window < len(mean):
        kernel = np.ones(ma_window) / ma_window
        mean_smoothed = np.convolve(mean, kernel, mode="valid")
        x_ma = x[ma_window - 1:]
        plt.plot(x_ma, mean_smoothed, label="Moving Avg", color="black")

    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.plot(x, mean, label="Mean", color="blue")

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(alpha=0.3)
    plt.savefig(output, dpi=300)
    print(f"[SAVED] plot → {output}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--ma_window", type=int, default=51)
    parser.add_argument("--compute_metrics", action="store_true")
    args = parser.parse_args()

    # ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # expand glob
    run_dirs = []
    for p in args.runs:
        run_dirs.extend(glob.glob(p))
    run_dirs = sorted(run_dirs)

    steps_list, values_list = load_runs(run_dirs, args.tag)
    x, runs_interp = interpolate(steps_list, values_list)

    title = args.title if args.title else args.tag
    plot_task_segments(x, runs_interp, args.output, title, ma_window=args.ma_window)

    if args.compute_metrics:
        matrices = []
        for r in run_dirs:
            mat = build_performance_matrix(r, num_tasks=4, interval=args.interval, cycles=args.cycles)
            if mat is not None:
                matrices.append(mat)

        avg_mat = np.mean(np.stack(matrices), axis=0)
        np.savetxt(f"{os.path.dirname(args.output)}/performance_matrix.csv", avg_mat, delimiter=",")

        metrics = compute_cl_metrics(avg_mat)
        json.dump(metrics, open(f"{os.path.dirname(args.output)}/cl_metrics.json", "w"), indent=4)

        print("[SAVED] performance_matrix.csv, cl_metrics.json")

# example usage:
# python plot_tb_scalars.py --runs "runs/CartPole-v1__base_*" --tag "charts/episodic_return" \
# --output "results_export/CartPole-v1/base/avg_episodic_return.png" --interval 300 --cycles 4 --compute_metrics \
# --title "CartPole-v1 Base Algorithm" 

# python plot_tb_scalars.py --runs "runs/CartPole-v1__parseval_*" --tag "charts/episodic_return" \
# --output "results_export/CartPole-v1/parseval/avg_episodic_return.png" --interval 300 --cycles 4 --compute_metrics \
# --title "CartPole-v1 Parseval Algorithm" 

# python plot_tb_scalars.py --runs "runs/Acrobot-v1__base_*" --tag "charts/episodic_return" \
# --output "results_export/Acrobot-v1/base/avg_episodic_return.png" --interval 300 --cycles 4 --compute_metrics \
# --title "Acrobot-v1 Base Algorithm" 

# python plot_tb_scalars.py --runs "runs/Acrobot-v1__parseval_*" --tag "charts/episodic_return" \
# --output "results_export/Acrobot-v1/parseval/avg_episodic_return.png" --interval 300 --cycles 4 --compute_metrics \
# --title "Acrobot-v1 Parseval Algorithm" 

# FOR LAYER COSINE SIMILARITY:

# python plot_tb_scalars.py --runs "runs/Acrobot-v1__parseval_*" --tag "agent/actor_cosine_sim_layer_2" \
# --output "results_export/Acrobot-v1/parseval/avg_actor_cosine_sim_layer_2.png" --interval 300 --cycles 4 \
# --title "Acrobot-v1 Parseval Algorithm" 

# python plot_tb_scalars.py --runs "runs/Acrobot-v1__base_*" --tag "agent/actor_cosine_sim_layer_2" \
# --output "results_export/Acrobot-v1/base/avg_actor_cosine_sim_layer_2.png" --interval 300 --cycles 4 \
# --title "Acrobot-v1 Base Algorithm" 

# python plot_tb_scalars.py --runs "runs/Acrobot-v1__parseval_*" --tag "agent/actor_cosine_sim_layer_2" \
# --output "results_export/Acrobot-v1/parseval/avg_actor_cosine_sim_layer_2.png" --interval 300 --cycles 4 \
# --title "Acrobot-v1 Parseval Algorithm" 

# python plot_tb_scalars.py --runs "runs/Acrobot-v1__base_*" --tag "agent/actor_cosine_sim_layer_2" \
# --output "results_export/Acrobot-v1/base/avg_actor_cosine_sim_layer_2.png" --interval 300 --cycles 4 \
# --title "Acrobot-v1 Base Algorithm" 