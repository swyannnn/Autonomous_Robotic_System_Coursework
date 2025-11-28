from typing import Callable
import numpy as np
import gymnasium as gym
import torch


def evaluate_agent(agent, make_env, args, device, success_threshold=500, task_manager=None, task_id=None, eval_episodes=10):
    """
    Evaluate the agent for `eval_episodes` episodes.
    Returns:
        mean_return: float
        success_rate: float (0.0 to 1.0)
    """
    # Create separate eval environment (num_envs = 1)
    eval_env = make_env(args.env_id, False, f"{args.exp_name}_eval")()

    if task_manager is not None and task_id is not None:
        params = task_manager.tasks[task_id]
        task_manager.apply_params(eval_env, params)

    agent.eval()

    obs, _ = eval_env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    episodic_returns = []
    successes = 0  # count number of successful episodes

    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions, _, _, _ = agent.get_action_and_value(obs)

        next_obs, reward, terminated, truncated, info = eval_env.step(actions.cpu().numpy())

        done = terminated or truncated

        if done:
            # Gymnasium puts episode info directly in info dict
            if "episode" in info:
                episodic_returns.append(info["episode"]["r"])
                length = info["episode"]["l"]

                if length >= success_threshold:
                    successes += 1

            # must reset immediately
            next_obs, _ = eval_env.reset()

        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

    eval_env.close()
    agent.train()

    mean_return = float(np.mean(episodic_returns))
    success_rate = successes / eval_episodes

    return mean_return, success_rate
