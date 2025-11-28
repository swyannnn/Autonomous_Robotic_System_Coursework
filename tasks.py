def set_task_parameters(cart, masspole, force_mag):
    """Apply physics modifications to CartPole."""
    # Set mass of pole
    cart.masspole = masspole
    cart.force_mag = force_mag

    # Need to recompute dependent terms
    cart.total_mass = cart.masspole + cart.masscart
    cart.polemass_length = cart.masspole * cart.length


class TaskManager:
    def __init__(self, env_or_envs):
        """
        Accepts either:
        - a single gym.Env
        - or a vectorized SyncVectorEnv (envs.envs)
        """
        self.tasks = [
            {"masspole": 0.05, "force_mag": 5.0},   # T1
            {"masspole": 0.05, "force_mag": 15.0},  # T2
            {"masspole": 0.20, "force_mag": 5.0},   # T3
            {"masspole": 0.20, "force_mag": 15.0},  # T4
        ]

        # Vectorized env mode
        if hasattr(env_or_envs, "envs"):
            self.vector_env = True
            self.envs = env_or_envs
            # Proper unwrapping
            self.raw_envs = [e.unwrapped for e in env_or_envs.envs]

        # Single-env mode
        else:
            self.vector_env = False
            self.env = env_or_envs
            self.raw_env = env_or_envs.unwrapped


    def set_task(self, task_index: int):
        """Apply selected task to all envs or single env."""
        params = self.tasks[task_index]

        set_task_parameters(self.raw_env, params["masspole"], params["force_mag"])
        print(f"[TaskManager] Switched to Task {task_index + 1}: {params}")


    def apply_params(self, raw_env, params):
        """Apply task parameters to a given raw env (used in evaluation)."""
        set_task_parameters(raw_env, params["masspole"], params["force_mag"])
