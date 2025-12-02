import torch
import numpy as np
import json
import glob
import os
from agent import PPOAgent, BasePPOAgent
from utils import make_env
from tasks import TaskManager


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

                obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                obs = torch.tensor(obs, dtype=torch.float32).to(device)
                total_reward += reward
                done = terminated or truncated

            returns.append(total_reward)

        task_returns[task_id] = float(np.mean(returns))

    return task_returns


def compute_forgetting_single_trial(perf, num_tasks=4):
    forgetting = {}
    for old_task in range(num_tasks):
        forgetting[old_task] = {}
        base_perf = perf[old_task][old_task]

        for checkpoint in range(num_tasks):
            if checkpoint <= old_task:
                forgetting[old_task][checkpoint] = None
            else:
                forgetting_value = base_perf - perf[checkpoint][old_task]
                forgetting[old_task][checkpoint] = float(forgetting_value)

    return forgetting


def evaluate_multi_trial_forgetting(
    root_pattern,
    env_id="CartPole-v1",
    algorithm="base",
    num_tasks=4,
    eval_episodes=20,
    episode_per_task=300,
    device="cuda",
    output_dir="results_export"
):
    os.makedirs(output_dir, exist_ok=True)
    base_paths = sorted(glob.glob(root_pattern))
    print(f"Found {len(base_paths)} trials.")

    env = make_env(env_id, capture_video=False, run_name="multi_eval")()
    task_manager = TaskManager(env)

    perf_all_trials = []
    forgetting_all_trials = []

    for trial_id, base_path in enumerate(base_paths):
        print(f"\n=== Trial {trial_id+1}: {base_path} ===")

        model_paths = [
            os.path.join(base_path, f"episode_{(t+1)*episode_per_task}.pth") for t in range(num_tasks)
        ]

        perf_this_trial = {}

        for checkpoint_id, model_path in enumerate(model_paths):
            print(f"Evaluating checkpoint {checkpoint_id+1}/{num_tasks}: {model_path}")
            agent = PPOAgent(env, add_diag_layer=False).to(device) if algorithm == "base" else PPOAgent(env).to(device)
            # agent = BasePPOAgent(env).to(device) if algorithm == "base" else PPOAgent(env).to(device)
            agent.load_state_dict(torch.load(model_path, map_location=device))
            agent.eval()

            perf_this_trial[checkpoint_id] = evaluate_model_on_all_tasks(
                agent, env, task_manager, num_tasks, eval_episodes, device
            )

        perf_all_trials.append(perf_this_trial)

        forgetting_matrix = compute_forgetting_single_trial(perf_this_trial, num_tasks)
        forgetting_all_trials.append(forgetting_matrix)

    env.close()

    forgetting_mean = {}
    forgetting_std = {}

    for old_task in range(num_tasks):
        forgetting_mean[old_task] = {}
        forgetting_std[old_task] = {}

        for checkpoint in range(num_tasks):
            values = [
                trial[old_task][checkpoint]
                for trial in forgetting_all_trials
                if trial[old_task][checkpoint] is not None
            ]

            if len(values) == 0:
                forgetting_mean[old_task][checkpoint] = None
                forgetting_std[old_task][checkpoint] = None
            else:
                forgetting_mean[old_task][checkpoint] = float(np.mean(values))
                forgetting_std[old_task][checkpoint]  = float(np.std(values))

    # ======= SAVE DATA =======
    json.dump(perf_all_trials, open(f"{output_dir}/perf_all_trials.json", "w"), indent=4)
    json.dump(forgetting_mean, open(f"{output_dir}/forgetting_mean.json", "w"), indent=4)
    json.dump(forgetting_std, open(f"{output_dir}/forgetting_std.json", "w"), indent=4)
    # table
    with open(f"{output_dir}/forgetting_table.csv", "w") as f:
        for t in range(num_tasks):
            row = [forgetting_mean[t][c] if forgetting_mean[t][c] is not None else "" for c in range(num_tasks)]
            f.write(",".join(map(str, row)) + "\n")

    print("[SAVED] perf_all_trials.json, forgetting_mean.json, forgetting_std.json, forgetting_table.csv")

    return perf_all_trials, forgetting_mean, forgetting_std

if __name__ == "__main__":
    root_pattern = "runs/Acrobot-v1__parseval_*"
    env_id = "Acrobot-v1"
    algorithm = "parseval" if "parseval" in root_pattern else "base"
    evaluate_multi_trial_forgetting(
        root_pattern=root_pattern,
        env_id=env_id,
        algorithm=algorithm,
        num_tasks=4,
        eval_episodes=20,
        episode_per_task=300,
        device="cuda", 
        output_dir=f"results_export/{env_id}/{algorithm}"
    )