import torch
import numpy as np
import json
import glob
import os
from agent import BasePPOAgent, ParsevalPPOAgent
from utils import make_env
from tasks import TaskManager
import gymnasium as gym



# =============================================================
#  Evaluate a single model on all tasks
# =============================================================
def evaluate_model_on_all_tasks(agent, env, task_manager, num_tasks, eval_episodes, device):
    task_returns = {}

    for task_id in range(num_tasks):
        task_manager.set_task(task_id)

        returns = []
        for _ in range(eval_episodes):
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).to(device)

            total_reward = 0
            done = False

            while not done:
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(obs)

                action_env = int(action.item())
                obs, reward, terminated, truncated, _ = env.step(action_env)
                obs = torch.tensor(obs, dtype=torch.float32).to(device)

                total_reward += reward
                done = terminated or truncated

            returns.append(total_reward)

        task_returns[task_id] = float(np.mean(returns))

    return task_returns



# =============================================================
#  Forgetting calculation for a single trial (4×4 matrix)
#  perf: dictionary  perf[model_id][task_id]
# =============================================================
def compute_forgetting_single_trial(perf, num_tasks=4):

    forgetting = {}
    for old_task in range(num_tasks):
        forgetting[old_task] = {}

        base_perf = perf[old_task][old_task]   # performance right after learning that task

        for checkpoint in range(num_tasks):
            if checkpoint <= old_task:
                forgetting[old_task][checkpoint] = None
            else:
                forgetting_value = base_perf - perf[checkpoint][old_task]
                forgetting[old_task][checkpoint] = float(forgetting_value)

    return forgetting



# =============================================================
#  Evaluate all trials across all checkpoints
# =============================================================
def evaluate_multi_trial_forgetting(
    root_pattern,
    env_id="CartPole-v1",
    algorithm="base",
    num_tasks=4,
    eval_episodes=20,
    device="cuda"
):

    # -----------------------------------
    # Detect all trial folders
    # -----------------------------------
    base_paths = sorted(glob.glob(root_pattern))
    print(f"Found {len(base_paths)} trials.")
    assert len(base_paths) > 0, "No trials found — check your glob pattern"

    # -----------------------------------
    # Create environment instance
    # -----------------------------------
    env = make_env(env_id, capture_video=False, run_name="multi_eval")()
    task_manager = TaskManager(env)

    # -----------------------------------
    # Select agent type
    # -----------------------------------
    def create_agent():
        return BasePPOAgent(env).to(device) if algorithm == "base" else ParsevalPPOAgent(env).to(device)

    # Storage
    perf_all_trials = []
    forgetting_all_trials = []

    # ============================================================
    #  Loop: each trial
    # ============================================================
    for trial_id, base_path in enumerate(base_paths):
        print(f"\n===============================")
        print(f"      Trial {trial_id+1}: {base_path}")
        print(f"===============================")

        model_paths = [
            os.path.join(base_path, "episode_300.pth"),
            os.path.join(base_path, "episode_600.pth"),
            os.path.join(base_path, "episode_900.pth"),
            os.path.join(base_path, "episode_1200.pth"),
        ]

        # ------------------------------
        # Check files exist
        # ------------------------------
        for p in model_paths:
            assert os.path.exists(p), f"Missing checkpoint: {p}"

        # ------------------------------
        # Evaluate this trial
        # ------------------------------
        perf_this_trial = {}

        for checkpoint_id, model_path in enumerate(model_paths):
            print(f"\nEvaluating checkpoint {checkpoint_id}: {model_path}")

            agent = create_agent()
            agent.load_state_dict(torch.load(model_path, map_location=device))
            agent.eval()

            perf_this_trial[checkpoint_id] = evaluate_model_on_all_tasks(
                agent, env, task_manager, num_tasks, eval_episodes, device
            )

            for t in range(num_tasks):
                print(f"  Model {checkpoint_id} on Task T{t+1}: {perf_this_trial[checkpoint_id][t]:.2f}")

        perf_all_trials.append(perf_this_trial)

        # ------------------------------
        # Compute forgetting for this trial
        # ------------------------------
        forgetting_matrix = compute_forgetting_single_trial(perf_this_trial, num_tasks)
        forgetting_all_trials.append(forgetting_matrix)

    env.close()


    # ============================================================
    #  COMPUTE MEAN FORGETTING MATRIX OVER TRIALS
    # ============================================================
    forgetting_mean = {}
    forgetting_std = {}

    for old_task in range(num_tasks):
        forgetting_mean[old_task] = {}
        forgetting_std[old_task] = {}

        for checkpoint in range(num_tasks):
            values = []

            for trial in forgetting_all_trials:
                val = trial[old_task][checkpoint]
                if val is not None:
                    values.append(val)

            if len(values) == 0:
                forgetting_mean[old_task][checkpoint] = None
                forgetting_std[old_task][checkpoint] = None
            else:
                forgetting_mean[old_task][checkpoint] = float(np.mean(values))
                forgetting_std[old_task][checkpoint]  = float(np.std(values))


    # ============================================================
    # PRINT MEAN FORGETTING TABLE
    # ============================================================
    print("\n\n=====================================")
    print("        MEAN FORGETTING TABLE")
    print("=====================================")
    header = "Task | " + " | ".join([f"T{t+1}" for t in range(num_tasks)])
    print(header)
    print("-" * len(header))

    for old_task in range(num_tasks):
        row = f"T{old_task+1} | "
        for checkpoint in range(num_tasks):
            if forgetting_mean[old_task][checkpoint] is None:
                row += " n/a |"
            else:
                row += f" {forgetting_mean[old_task][checkpoint]:.3f} |"
        print(row)

    return perf_all_trials, forgetting_mean, forgetting_std



# ============================================================
# RUN EXAMPLE
# ============================================================

root_pattern = "/media/nine/HD_1/HD_2_from_seven/Yann/robotics/COMP4082_ARS/scratch/runs/CartPole-v1__parseval__*"

perf_all_trials, forgetting_mean, forgetting_std = evaluate_multi_trial_forgetting(
    root_pattern=root_pattern,
    env_id="CartPole-v1" if "CartPole" in root_pattern else "Pendulum-v1",
    algorithm="base" if "base" in root_pattern else "parseval",
    num_tasks=4,
    eval_episodes=20,
    device="cuda"
)
