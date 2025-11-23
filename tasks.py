# tasks.py
import gymnasium as gym

def get_cartpole_base_env(env):
    """Unwrap all layers and return the raw CartPoleEnv."""
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e  # must be CartPoleEnv

def set_task_parameters(cart, masspole, force_mag):
    """Apply physics modifications to CartPole."""
    # Set mass of pole
    cart.masspole = masspole
    cart.force_mag = force_mag

    # Need to recompute dependent terms
    cart.total_mass = cart.masspole + cart.masscart
    cart.polemass_length = cart.masspole * cart.length

class TaskManager:
    def __init__(self, envs):
        """
        envs is your SyncVectorEnv
        """
        self.envs = envs

        # List of raw environments
        self.raw_envs = [get_cartpole_base_env(env) for env in envs.envs]

        # Define 4 tasks
        self.tasks = [
            {"masspole": 0.05, "force_mag": 5.0},   # T1
            {"masspole": 0.05, "force_mag": 15.0},  # T2
            {"masspole": 0.20, "force_mag": 5.0},   # T3
            {"masspole": 0.20, "force_mag": 15.0},  # T4
        ]

    def set_task(self, task_index: int):
        """Apply the selected task to all environments."""
        params = self.tasks[task_index]
        for cart in self.raw_envs:
            set_task_parameters(cart, params["masspole"], params["force_mag"])
        print(f"[TaskManager] Switched to Task {task_index+1}: {params}")

    def unwrap_single(self, env):
        """Fully unwrap a single env from wrappers."""
        e = env
        while hasattr(e, "env"):
            e = e.env
        return e

    def apply_params(self, env, params):
        """Apply params to one raw env."""
        env.masspole = params["masspole"]
        env.force_mag = params["force_mag"]
        env.total_mass = env.masspole + env.masscart
        env.polemass_length = env.masspole * env.length
