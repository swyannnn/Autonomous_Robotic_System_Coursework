
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, os


def load_scalar(event_file, tag):
    ea = EventAccumulator(event_file)
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        return None, None

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def load_runs(pattern, tag):
    run_dirs = sorted(glob.glob(pattern))
    steps_list, values_list = [], []

    for run in run_dirs:
        event_files = glob.glob(os.path.join(run, "events.*"))
        if not event_files:
            continue
        
        event_file = sorted(event_files)[-1]
        s, v = load_scalar(event_file, tag)
        if s is None: 
            continue
        
        steps_list.append(s)
        values_list.append(v)

    return steps_list, values_list


def interpolate(steps_list, values_list, num_points=2000):
    xmin = max(s[0] for s in steps_list)
    xmax = min(s[-1] for s in steps_list)
    x_common = np.linspace(xmin, xmax, num_points)

    vals_interp = [np.interp(x_common, s, v) for s, v in zip(steps_list, values_list)]
    return x_common, np.stack(vals_interp)


def plot_comparison(
        x, 
        base_vals, 
        parseval_vals, 
        interval=300,
        cycles=4,
        output="comparison.png"
):
    base_mean = base_vals.mean(axis=0)
    parseval_mean = parseval_vals.mean(axis=0)

    ma = lambda y, w: np.convolve(y, np.ones(w)/w, mode="valid")

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # Base PPO raw curve
    ax.plot(x, base_mean, color="blue", alpha=0.35, label="Base PPO")
    # Base PPO moving average
    base_ma = ma(base_mean, 51)
    ax.plot(x[50:], base_ma, color="blue", linewidth=2)

    # Parseval PPO raw curve
    ax.plot(x, parseval_mean, color="red", alpha=0.35, label="Parseval PPO")
    # Parseval PPO moving average
    parseval_ma = ma(parseval_mean, 51)
    ax.plot(x[50:], parseval_ma, color="red", linewidth=2)

    # --- Task segment shading (optional, keep if you like the bands) ---
    for i in range(cycles):
        start = i * interval
        end = (i + 1) * interval
        ax.axvspan(start, end, alpha=0.05, color="gray")

    # --- Vertical dotted lines at boundaries: 0, 300, 600, 900, 1200 ---
    for i in range(cycles + 1):
        x_pos = i * interval
        ax.axvline(x_pos, linestyle="--", color="gray", linewidth=1, alpha=0.8)

    # --- Task labels T1, T2, T3, T4 in middle of each segment ---
    ymin, ymax = ax.get_ylim()
    for i in range(cycles):
        x_mid = (i + 0.5) * interval
        ax.text(
            x_mid,
            ymax * 0.95,          # slightly below top
            f"T{i+1}",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Comparison of Base PPO vs Parseval PPO (CartPole-v1)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic Return")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print("[Saved]", output)



if __name__ == "__main__":
    # Example usage section (edit paths)
    base_steps, base_vals = load_runs("runs/CartPole-v1__base__*", "charts/episodic_return")
    p_steps, p_vals = load_runs("runs/CartPole-v1__parseval__*", "charts/episodic_return")

    x, base_interp = interpolate(base_steps, base_vals)
    _, parseval_interp = interpolate(p_steps, p_vals)

    plot_comparison(x, base_interp, parseval_interp)
