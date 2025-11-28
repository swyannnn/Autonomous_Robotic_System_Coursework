import torch
import numpy as np
from agent import BasePPOAgent, ParsevalPPOAgent
from utils import make_env
from tasks import TaskManager


def evaluate_saved_model(
    model_path,
    env_id="CartPole-v1",
    algorithm="base",
    num_episodes=10,
    device="cuda",
    task_id=None,
):
    """
    Loads a saved PPO model and evaluates it.

    Args:
        model_path: path to the saved agent_final.pth
        env_id: Gym environment id
        algorithm: "base" or "parseval"
        num_episodes: number of eval episodes
        device: "cuda" or "cpu"
        task_id: if using TaskManager, which task to set before evaluation
    """

    # ------------------------------
    # 1. Create environment
    # ------------------------------
    env = make_env(env_id, capture_video=False, run_name="eval")()
    task_manager = TaskManager(env)

    # Set specific task if requested
    if task_id is not None:
        task_manager.set_task(task_id)

    # ------------------------------
    # 2. Load agent architecture
    # ------------------------------
    if algorithm == "base":
        agent = BasePPOAgent(env).to(device)
    else:
        agent = ParsevalPPOAgent(env).to(device)

    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    # ------------------------------
    # 3. Run evaluation
    # ------------------------------
    returns = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32).to(device)

        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs)

            obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            obs = torch.tensor(obs, dtype=torch.float32).to(device)

            done = terminated or truncated
            total_reward += reward

        returns.append(total_reward)

        print(f"Episode {ep+1}: return = {total_reward}")

    env.close()

    print("\n==========================")
    print(f" Mean return over {num_episodes} episodes: {np.mean(returns):.2f}")
    print("==========================")

    return np.mean(returns)



evaluate_saved_model(
    model_path = "/media/nine/HD_1/HD_2_from_seven/Yann/robotics/COMP4082_ARS/scratch/runs/CartPole-v1__base__1__1764338987/episode_900.pth",
    env_id="CartPole-v1",
    algorithm="base",
    num_episodes=20,
    task_id=3,    # if you want to test task 3 (0-based)
)
