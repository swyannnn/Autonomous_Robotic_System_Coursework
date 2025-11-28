import torch
import numpy as np
from agent import BasePPOAgent, ParsevalPPOAgent
from utils import make_env
from tasks import TaskManager


import torch
import numpy as np
import json
from agent import BasePPOAgent, ParsevalPPOAgent
from utils import make_env
from tasks import TaskManager


def evaluate_model_all_tasks(
    model_path,
    best_json_path,
    env_id="CartPole-v1",
    algorithm="base",
    num_tasks=4,
    eval_episodes=10,
    device="cuda",
    success_threshold=500,
):
    """
    Loads a saved PPO model and evaluates it on ALL tasks.
    Computes mean return, success rate, and forgetting.
    """

    # ------------------------------
    # Load best_per_task from training
    # ------------------------------
    with open(best_json_path, "r") as f:
        best_per_task = json.load(f)

    # Convert keys to int for safety
    best_per_task = {int(k): float(v) for k, v in best_per_task.items()}

    # ------------------------------
    # Create environment + agent
    # ------------------------------
    env = make_env(env_id, capture_video=False, run_name="eval")()
    task_manager = TaskManager(env)

    if algorithm == "base":
        agent = BasePPOAgent(env).to(device)
    else:
        agent = ParsevalPPOAgent(env).to(device)

    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    # ==============================
    #   Evaluate all tasks
    # ==============================

    all_task_returns = {}
    all_task_success = {}
    forgetting_scores = {}

    for task_id in range(num_tasks):
        print(f"\n==============================")
        print(f" Evaluating Task {task_id + 1}")
        print(f"==============================")

        # Apply task physics
        task_manager.set_task(task_id)

        returns = []
        successes = []

        # Run several episodes
        for ep in range(eval_episodes):
            obs, _ = env.reset()
            obs = torch.tensor(obs, dtype=torch.float32).to(device)

            total_reward = 0
            done = False

            while not done:
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(obs)

                # convert tensor to int
                action_int = int(action.item())

                obs, reward, terminated, truncated, _ = env.step(action_int)
                obs = torch.tensor(obs, dtype=torch.float32).to(device)

                total_reward += reward
                done = terminated or truncated

            returns.append(total_reward)
            successes.append(1 if total_reward >= success_threshold else 0)

            print(f"Episode {ep+1}: return = {total_reward}")

        mean_return = float(np.mean(returns))
        success_rate = float(np.mean(successes))

        all_task_returns[task_id] = mean_return
        all_task_success[task_id] = success_rate

        # -----------------------------------------
        # Compute forgetting:
        # F_t = best_per_task[t] - current_performance
        # -----------------------------------------
        prev_best = best_per_task.get(task_id, 0)
        forgetting = max(prev_best - mean_return, 0.0)
        forgetting_scores[task_id] = forgetting

        print(f"Mean return:   {mean_return:.2f}")
        print(f"Success rate:  {success_rate:.2f}")
        print(f"Best ever:     {prev_best:.2f}")
        print(f"Forgetting:    {forgetting:.2f}")

    env.close()

    # ==============================
    # Summary
    # ==============================
    print("\n=====================================")
    print(" FINAL EVALUATION SUMMARY")
    print("=====================================")
    print("Task → Mean Return   Success   Forgetting")
    for t in range(num_tasks):
        print(f"T{t+1}:   {all_task_returns[t]:7.2f}   {all_task_success[t]:.2f}      {forgetting_scores[t]:.2f}")

    return all_task_returns, all_task_success, forgetting_scores

model_path = "/media/nine/HD_1/HD_2_from_seven/Yann/robotics/COMP4082_ARS/scratch/runs/CartPole-v1__base__1__1764343795/episode_900.pth"
evaluate_model_all_tasks(
    model_path = model_path,
    best_json_path = "/media/nine/HD_1/HD_2_from_seven/Yann/robotics/COMP4082_ARS/scratch/runs/CartPole-v1__base__1__1764343795/best_per_task.json",
    env_id="CartPole-v1",
    algorithm="base" if "base" in model_path else "parseval",
    num_tasks=4,
    eval_episodes=20,
    device="cuda",
    success_threshold=500,
)
