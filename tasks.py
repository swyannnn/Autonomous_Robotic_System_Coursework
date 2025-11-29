import yaml

def set_task_cartpole_parameters(env, params):
    """Apply physics modifications to CartPole."""
    gravity = params["gravity"]
    masspole = params["masspole"]
    force_mag = params["force_mag"]

    # Set mass of pole
    env.unwrapped.masspole = masspole
    env.unwrapped.force_mag = force_mag
    # Need to recompute dependent terms
    env.unwrapped.total_mass = env.unwrapped.masspole + env.unwrapped.masscart
    env.unwrapped.polemass_length = env.unwrapped.masspole * env.unwrapped.length
    env.unwrapped.gravity = gravity

def set_task_mountaincar_parameters(env, params):
    """Apply physics modifications to MountainCar."""
    gravity = params["gravity"]
    force = params["force"]

    env.unwrapped.gravity = gravity
    env.unwrapped.force = force

class TaskManager:
    def __init__(self, env, task_config_file="task_config.yaml"):
        """
        Accepts either:
        - a single gym.Env
        - or a vectorized SyncVectorEnv (envs.envs)
        """
        with open(task_config_file, "r") as f:
            self.tasks = yaml.safe_load(f)[env.spec.id]
        self.env = env
        self.set_task_parameters = set_task_cartpole_parameters if "CartPole" in env.spec.id \
                        else set_task_mountaincar_parameters

    def set_task(self, task_index: int):
        """Apply selected task to all envs or single env."""
        params = self.tasks[task_index]

        self.set_task_parameters(self.env, params)
        print(f"[TaskManager] Switched to Task {task_index + 1}: {params}")

    def apply_params(self, raw_env, params):
        """Apply task parameters to a given raw env (used in evaluation)."""
        self.set_task_parameters(raw_env, params)
