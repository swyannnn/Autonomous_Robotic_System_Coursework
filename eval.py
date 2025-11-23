from typing import Callable
import numpy as np
import gymnasium as gym
import torch


def evaluate_agent(agent, make_env, args, device, task_manager=None, task_id=None, eval_episodes=10):
    """
    Evaluate the agent for `eval_episodes` episodes.
    Returns:
        mean_return: float
        success_rate: float (0.0 to 1.0)
    """
    # Create separate eval environment (num_envs = 1)
    eval_envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, 0, False, f"{args.exp_name}_eval")]
    )

    # unwrap and apply selected task parameters
    if task_manager is not None and task_id is not None:
        raw_env = task_manager.unwrap_single(eval_envs.envs[0]) 
        params = task_manager.tasks[task_id]
        task_manager.apply_params(raw_env, params)

    agent.eval()

    obs, _ = eval_envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    episodic_returns = []
    successes = 0  # count number of successful episodes

    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions, _, _, _ = agent.get_action_and_value(obs)

        next_obs, _, _, _, infos = eval_envs.step(actions.cpu().numpy())

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is None:
                    continue
                if "episode" not in info:
                    continue

                # record return
                episodic_returns.append(info["episode"]["r"])

                # detect success (CartPole-specific: time limit reached)
                terminated = info.get("terminated", False)
                truncated = info.get("truncated", False)

                if (not terminated) and truncated:
                    successes += 1

        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

    eval_envs.close()
    agent.train()

    mean_return = float(np.mean(episodic_returns))
    success_rate = successes / eval_episodes

    return mean_return, success_rate


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from cleanrl.ppo_continuous_action import Agent, make_env

    model_path = hf_hub_download(
        repo_id="sdpkjc/Hopper-v4-ppo_continuous_action-seed1", filename="ppo_continuous_action.cleanrl_model"
    )
    evaluate(
        model_path,
        make_env,
        "Hopper-v4",
        eval_episodes=10,
        run_name=f"eval",
        Model=Agent,
        device="cpu",
        capture_video=False,
    )
