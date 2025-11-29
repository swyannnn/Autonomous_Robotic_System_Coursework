import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os


def load_scalar(event_file, tag):
    ea = EventAccumulator(event_file)
    ea.Reload()

    if tag not in ea.Tags()['scalars']:
        print(f"[WARN] Tag {tag} not found in {event_file}")
        return None, None

    events = ea.Scalars(tag)

    steps = np.array([e.step for e in events], dtype=np.float32)
    values = np.array([e.value for e in events], dtype=np.float32)

    return steps, values


def load_runs(run_dirs, tag):
    all_steps = []
    all_values = []

    for run in run_dirs:
        event_files = glob.glob(os.path.join(run, "events.*"))
        if len(event_files) == 0:
            continue

        event_file = sorted(event_files)[-1]
        print(f"[LOAD] {event_file}")

        steps, values = load_scalar(event_file, tag)
        if steps is None:
            continue

        all_steps.append(steps)
        all_values.append(values)

    return all_steps, all_values


def interpolate(all_steps, all_values, num_points=2000):
    xmin = max([s[0] for s in all_steps])
    xmax = min([s[-1] for s in all_steps])

    common_x = np.linspace(xmin, xmax, num_points)
    interpolated = []

    for s, v in zip(all_steps, all_values):
        interp = np.interp(common_x, s, v)
        interpolated.append(interp)

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

    # if cycles not provided, cover all data
    if cycles is None:
        cycles = int(np.ceil(max_episode_data / interval))

    max_episode_plot = cycles * interval

    # restrict to desired episode range (e.g. 0–1200)
    mask_all = x <= max_episode_plot
    x_plot = x[mask_all]
    mean_plot = mean[mask_all]
    std_plot = std[mask_all]

    plt.figure(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, num_tasks))
    task_labels_added = set()

    # draw task-colored segments
    for cycle in range(cycles):
        t = cycle % num_tasks
        start_ep = cycle * interval
        end_ep = (cycle + 1) * interval

        mask = (x_plot >= start_ep) & (x_plot < end_ep)
        if mask.sum() == 0:
            continue

        label = f"Task {t}"
        if label in task_labels_added:
            plot_label = None
        else:
            plot_label = label
            task_labels_added.add(label)

        plt.plot(
            x_plot[mask],
            mean_plot[mask],
            color=colors[t],
            linewidth=2,
            label=plot_label,
        )

        plt.fill_between(
            x_plot[mask],
            mean_plot[mask] - std_plot[mask],
            mean_plot[mask] + std_plot[mask],
            color=colors[t],
            alpha=0.15,
        )

    # moving average (over the averaged curve)
    if ma_window is not None and ma_window > 1:
        if ma_window > len(mean_plot):
            ma_window = len(mean_plot)
        kernel = np.ones(ma_window) / ma_window
        ma = np.convolve(mean_plot, kernel, mode="valid")
        x_ma = x_plot[ma_window - 1 :]
        plt.plot(
            x_ma,
            ma,
            color="black",
            linewidth=2,
            label=f"Moving Avg (w={ma_window})",
        )

    # Formatting
    xticks = np.arange(0, int(max_episode_plot) + interval, interval)
    plt.xticks(xticks)

    plt.grid(alpha=0.3)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Episodic Return")
    plt.legend(title="Task Intervals")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"[SAVED] {output}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument(
        "--cycles",
        type=int,
        default=4,
        help="Number of task cycles (intervals) to plot. 4 with interval=300 -> 1200 episodes.",
    )
    parser.add_argument(
        "--ma_window",
        type=int,
        default=51,
        help="Moving average window (in points along the interpolated curve).",
    )
    args = parser.parse_args()

    # Expand glob patterns
    run_dirs = []
    for p in args.runs:
        run_dirs.extend(glob.glob(p))

    steps_list, values_list = load_runs(run_dirs, args.tag)
    if len(steps_list) == 0:
        print("[ERROR] No valid runs.")
        exit()

    x, runs = interpolate(steps_list, values_list)
    title = args.title if args.title else args.tag

    plot_task_segments(
        x,
        runs,
        args.interval,
        args.output,
        title,
        num_tasks=4,
        cycles=args.cycles,
        ma_window=args.ma_window,
    )

# Example usage:
# python plot_tb_scalars.py \
#     --runs runs/CartPole-v1__parseval__* \
#     --tag charts/episodic_return \
#     --output episodic_return_parseval.png \
#     --interval 300 \
#     --cycles 4 \
#     --ma_window 51 \
#     --title "CartPole-v1 (PPO Agent with Parseval Regularization)"
