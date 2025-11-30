import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os
import json
import csv


# ============================================================
# LOAD A SCALAR TAG FROM A SINGLE EVENT FILE
# ============================================================
def load_scalar(event_file, tag):
    ea = EventAccumulator(event_file)
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        return None, None

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=np.float32)
    values = np.array([e.value for e in events], dtype=np.float32)
    return steps, values


# ============================================================
# LOAD ALL RUNS AND EXTRACT THE TAGS
# ============================================================
def load_runs(run_dirs, tag):
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


# ============================================================
# FIND EVALUATION POINTS (EPISODE COUNTS WHERE SWITCH OCCURS)
# ============================================================
def extract_switch_points(steps, interval, cycles):
    """Returns the steps closest to switch boundaries: 300, 600, ..."""
    switch_points = []
    targets = np.arange(interval, interval * (cycles + 1), interval)

    for t in targets:
        idx = (np.abs(steps - t)).argmin()
        switch_points.append(steps[idx])
    return switch_points


# ============================================================
# BUILD PERFORMANCE MATRIX FOR ONE RUN
# ============================================================
def build_performance_matrix(run_dir, num_tasks, interval, cycles):
    perf_mat = np.zeros((num_tasks, cycles))  # task × switch_index

    event_files = glob.glob(os.path.join(run_dir, "events.*"))
    if len(event_files) == 0:
        print(f"[WARN] No event files in {run_dir}")
        return None

    event_file = sorted(event_files)[-1]
    ea = EventAccumulator(event_file)
    ea.Reload()

    # For each task t, load scalar
    for t in range(num_tasks):
        tag = f"eval/task_{t}/mean_return"
        if tag not in ea.Tags().get("scalars", []):
            print(f"[WARN] Missing tag {tag} in {event_file}")
            return None

        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])

        # find switch episode points
        switch_points = extract_switch_points(steps, interval, cycles)

        for i, sp in enumerate(switch_points):
            idx = (np.abs(steps - sp)).argmin()
            perf_mat[t, i] = values[idx]

    return perf_mat


# ============================================================
# CALCULATE CL METRICS: FWT, BWT, CF, SPB
# ============================================================
def compute_cl_metrics(perf):
    """
    perf = [num_tasks × cycles]
    perf[t][i] = performance on task t after training task i
    """

    T, C = perf.shape

    # diagonal: after training the task itself
    diag = np.array([perf[t, t] for t in range(T)])

    # final: after last cycle
    final = perf[:, C - 1]

    # --- BWT (Backward Transfer) ---
    bwt_values = final[:T - 1] - diag[:T - 1]
    BWT = bwt_values.mean()

    # --- Catastrophic Forgetting (CF) ---
    CF = -BWT  # definition: forgetting = -BWT (common CL definition)
    CF_tasks = -bwt_values

    # --- FWT (Forward Transfer) ---
    # performance on task j *before* it was trained: perf[j][j-1]
    # baseline = 0 (CartPole random policy)
    fwt_values = []
    for j in range(1, T):
        fwt_values.append(perf[j, j - 1])  # before training j

    FWT = np.mean(fwt_values)

    # --- Stability–Plasticity Balance (SPB) ---
    SPB = FWT - abs(BWT)

    return {
        "FWT": float(FWT),
        "BWT": float(BWT),
        "CF": float(CF),
        "SPB": float(SPB),
        "CF_per_task": CF_tasks.tolist(),
        "BWT_per_task": bwt_values.tolist(),
    }


# ============================================================
# PLOT AVERAGED EPISODIC RETURN (YOUR ORIGINAL PLOT)
# ============================================================
def interpolate(all_steps, all_values, num_points=2000):
    xmin = max(s[0] for s in all_steps)
    xmax = min(s[-1] for s in all_steps)
    common_x = np.linspace(xmin, xmax, num_points)
    interpolated = [np.interp(common_x, s, v) for s, v in zip(all_steps, all_values)]
    return common_x, np.stack(interpolated)


def plot_task_segments(
    x,
    runs,
    interval,
    output,
    title,
    num_tasks=4,
    cycles=None,
    ma_window=51,
):
    mean = runs.mean(axis=0)
    std = runs.std(axis=0)

    max_episode_data = x[-1]
    if cycles is None:
        cycles = int(np.ceil(max_episode_data / interval))

    max_episode_plot = cycles * interval
    mask_all = x <= max_episode_plot
    x_plot = x[mask_all]
    mean_plot = mean[mask_all]
    std_plot = std[mask_all]

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_tasks))
    task_labels_added = set()

    for cycle in range(cycles):
        t = cycle % num_tasks
        start_ep = cycle * interval
        end_ep = (cycle + 1) * interval

        mask = (x_plot >= start_ep) & (x_plot < end_ep)
        if mask.sum() == 0:
            continue

        label = f"Task {t}" if f"Task {t}" not in task_labels_added else None
        task_labels_added.add(f"Task {t}")

        plt.plot(
            x_plot[mask],
            mean_plot[mask],
            color=colors[t],
            linewidth=2,
            label=label,
        )
        plt.fill_between(
            x_plot[mask],
            mean_plot[mask] - std_plot[mask],
            mean_plot[mask] + std_plot[mask],
            color=colors[t],
            alpha=0.15,
        )

    if ma_window > 1 and ma_window < len(mean_plot):
        kernel = np.ones(ma_window) / ma_window
        ma = np.convolve(mean_plot, kernel, mode="valid")
        x_ma = x_plot[ma_window - 1:]
        plt.plot(x_ma, ma, color="black", linewidth=2, label="Moving Avg")

    plt.xticks(np.arange(0, max_episode_plot + interval, interval))
    plt.grid(alpha=0.3)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Episodic Return")
    plt.legend(title="Task Intervals")

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"[SAVED] {output}")


# ============================================================
# MAIN ENTRY
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

    # expand glob
    run_dirs = []
    for p in args.runs:
        run_dirs.extend(glob.glob(p))
    run_dirs = sorted(run_dirs)

    # ===== PLOT =====
    steps_list, values_list = load_runs(run_dirs, args.tag)
    x, runs_interp = interpolate(steps_list, values_list)
    title = args.title if args.title else args.tag
    plot_task_segments(x, runs_interp, args.interval, args.output, title, cycles=args.cycles)

    # ===== COMPUTE METRICS =====
    if args.compute_metrics:
        print("\n[Computing Continual Learning Metrics...]")

        matrices = []
        for r in run_dirs:
            mat = build_performance_matrix(r, num_tasks=4, interval=args.interval, cycles=args.cycles)
            if mat is not None:
                matrices.append(mat)

        if len(matrices) == 0:
            print("[ERROR] No matrices extracted.")
            exit()

        avg_mat = np.mean(np.stack(matrices), axis=0)
        metrics = compute_cl_metrics(avg_mat)

        # Save CSV
        np.savetxt("performance_matrix.csv", avg_mat, delimiter=",")
        with open("metrics_summary.json", "w") as f:
            json.dump(metrics, f, indent=4)

        print("\n=== METRICS SUMMARY ===")
        print(json.dumps(metrics, indent=4))


# Example usage:
# python plot_tb_scalars.py \
#     --runs runs_exp/Pendulum-v1__base__* \
#     --tag charts/episodic_return \
#     --output episodic_return_base_pendulum.png \
#     --interval 300 \
#     --cycles 4 \
#     --ma_window 51 \
#     --title "Pendulum-v1 (PPO Agent with Base Algorithm)" \
#     --compute_metrics 
