import yaml

def set_task_cartpole_parameters(env: object, params: dict) -> None:
    """
    Apply physics modifications to CartPole.
    Args:
        env: Gymnasium CartPole environment
        params: Dictionary of parameters to modify
    """

    u = env.unwrapped

    if "masspole" in params:
        u.masspole = params["masspole"]
    if "force_mag" in params:
        u.force_mag = params["force_mag"]
    if "gravity" in params:
        u.gravity = params["gravity"]

    # Need to recompute dependent terms
    u.total_mass = u.masspole + u.masscart
    u.polemass_length = u.masspole * u.length

def set_task_acrobot_parameters(env: object, params: dict) -> None:
    """
    Apply physics modifications to Acrobot (Gymnasium).
    Args:
        env: Gymnasium Acrobot environment
        params: Dictionary of parameters to modify
    """

    u = env.unwrapped

    # Gravity
    if "g" in params:
        u.g = params["g"]

    # Link lengths
    if "LINK_LENGTH_1" in params:
        u.LINK_LENGTH_1 = params["LINK_LENGTH_1"]
    if "LINK_LENGTH_2" in params:
        u.LINK_LENGTH_2 = params["LINK_LENGTH_2"]

    # Link masses
    if "LINK_MASS_1" in params:
        u.LINK_MASS_1 = params["LINK_MASS_1"]
    if "LINK_MASS_2" in params:
        u.LINK_MASS_2 = params["LINK_MASS_2"]

    # Center of mass positions
    if "LINK_COM_POS_1" in params:
        u.LINK_COM_POS_1 = params["LINK_COM_POS_1"]
    if "LINK_COM_POS_2" in params:
        u.LINK_COM_POS_2 = params["LINK_COM_POS_2"]

    # Moment of inertia
    if "LINK_MOI" in params:
        u.LINK_MOI = params["LINK_MOI"]

    # Torque scaling
    if "MAX_TORQUE" in params:
        u.MAX_TORQUE = params["MAX_TORQUE"]

    # Integration timestep
    if "dt" in params:
        u.dt = params["dt"]

    # Angular velocity limits
    if "MAX_VEL_1" in params:
        u.MAX_VEL_1 = params["MAX_VEL_1"]
    if "MAX_VEL_2" in params:
        u.MAX_VEL_2 = params["MAX_VEL_2"]

class TaskManager:
    """
    Manages tasks for a given Gym environment based on a configuration file.
    Args:
        env: Gym environment or vectorized SyncVectorEnv
        task_config_file (str): Path to the YAML file containing task configurations.
    """
    def __init__(self, env, task_config_file="config/task_config.yaml"):
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
        elif env.spec.id == "Acrobot-v1":
            self.set_task_parameters = set_task_acrobot_parameters
        else:
            raise NotImplementedError(f"TaskManager not implemented for env {env.spec.id}")

    def set_task(self, task_index: int) -> None:
        """Apply selected task to all envs or single env."""
        params = self.tasks[task_index]

        self.set_task_parameters(self.env, params)
        print(f"[TaskManager] Switched to Task {task_index + 1}: {params}")

    def apply_params(self, raw_env: object, params: dict) -> None:
        """Apply task parameters to a given raw env (used in evaluation)."""
        self.set_task_parameters(raw_env, params)

    def log_task_params(self, logger: object, task_index: int, episode_count: int) -> None:
        """Log current task parameters."""
        params = self.tasks[task_index]
        for param in params:
            # get it from env.unwrapped to ensure we log the actual applied value
            params[param] = getattr(self.env.unwrapped, param)
        for key, value in params.items():
            logger.add_scalar(f"task/{key}", value, episode_count)
