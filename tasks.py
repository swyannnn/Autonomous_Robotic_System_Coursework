def set_task_parameters(env, masspole, force_mag):
    """Apply physics modifications to CartPole."""
    # Set mass of pole
    env.unwrapped.masspole = masspole
    env.unwrapped.force_mag = force_mag
    # Need to recompute dependent terms
    env.unwrapped.total_mass = env.unwrapped.masspole + env.unwrapped.masscart
    env.unwrapped.polemass_length = env.unwrapped.masspole * env.unwrapped.length


class TaskManager:
    def __init__(self, env):
        """
        Accepts either:
        - a single gym.Env
        - or a vectorized SyncVectorEnv (envs.envs)
        """
        self.tasks = [
            {"masspole": 1.0, "force_mag": 10.0},   # T1
            {"masspole": 1.0, "force_mag": 31.0},  # T2
            {"masspole": 7.0, "force_mag": 10.0},   # T3
            {"masspole": 7.0, "force_mag": 31.0},  # T4
        ]
        self.env = env

    def set_task(self, task_index: int):
        """Apply selected task to all envs or single env."""
        params = self.tasks[task_index]

        set_task_parameters(self.env, params["masspole"], params["force_mag"])
        print(f"[TaskManager] Switched to Task {task_index + 1}: {params}")

    def apply_params(self, raw_env, params):
        """Apply task parameters to a given raw env (used in evaluation)."""
        set_task_parameters(raw_env, params["masspole"], params["force_mag"])
