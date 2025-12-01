import yaml

def set_task_cartpole_parameters(env, params):
    """Apply physics modifications to CartPole."""
    gravity = params["gravity"]
    masspole = params["masspole"]
    force_mag = params["force_mag"]
    # length = params["length"]

    # Set mass of pole
    env.unwrapped.masspole = masspole
    env.unwrapped.force_mag = force_mag
    # env.unwrapped.length = length
    # Need to recompute dependent terms
    env.unwrapped.total_mass = env.unwrapped.masspole + env.unwrapped.masscart
    env.unwrapped.polemass_length = env.unwrapped.masspole * env.unwrapped.length
    env.unwrapped.gravity = gravity

def set_task_mountaincar_parameters(env, params):
    """Apply physics modifications to MountainCar."""
    power = params["power"]
    max_speed = params["max_speed"]
    goal_position = params["goal_position"]
    goal_velocity = params["goal_velocity"]

    env.unwrapped.power = power
    env.unwrapped.max_speed = max_speed
    env.unwrapped.goal_position = goal_position
    env.unwrapped.goal_velocity = goal_velocity

def set_task_pendulum_parameters(env, params):
    """Apply physics modifications to Pendulum."""
    gravity = params["g"]
    max_torque = params["max_torque"]
    mass = params["m"]
    length = params["l"]

    env.unwrapped.max_torque = max_torque
    env.unwrapped.m = mass
    env.unwrapped.l = length
    env.unwrapped.g = gravity

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

        if env.spec.id == "CartPole-v1":
            self.set_task_parameters = set_task_cartpole_parameters
        elif env.spec.id == "MountainCarContinuous-v0":
            self.set_task_parameters = set_task_mountaincar_parameters
        elif env.spec.id == "Pendulum-v1":
            self.set_task_parameters = set_task_pendulum_parameters

    def set_task(self, task_index: int):
        """Apply selected task to all envs or single env."""
        params = self.tasks[task_index]

        self.set_task_parameters(self.env, params)
        print(f"[TaskManager] Switched to Task {task_index + 1}: {params}")

    def apply_params(self, raw_env, params):
        """Apply task parameters to a given raw env (used in evaluation)."""
        self.set_task_parameters(raw_env, params)

    def log_task_params(self, logger, task_index: int, episode_count: int):
        """Log current task parameters."""
        params = self.tasks[task_index]
        for param in params:
            # get it from env.unwrapped to ensure we log the actual applied value
            params[param] = getattr(self.env.unwrapped, param)
        for key, value in params.items():
            logger.add_scalar(f"task/{key}", value, episode_count)
