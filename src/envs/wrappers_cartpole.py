# src/envs/wrappers_cartpole.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces # Import spaces to access observation/action space info

from src.utils.logger_setup import setup_logger

logger = setup_logger(__name__) 

class CartPoleWrapper:

    def __init__(self, env_name: str, render_mode: str | None = None):  # render_mode expects a string or None; default is None.
        logger.info(f"Initializing environment instance: '{env_name}', render_mode: '{render_mode}'...")
        try:
            self.env = gym.make(env_name, render_mode=render_mode) # Create the actual Gym environment instance.
            self.env_name = env_name # Alternatively: self.env_name = self.env.spec.id
            logger.debug(f"Environment instance '{self.env.spec.id}' created successfully!")
            logger.debug(f"Observation space: {self.env.observation_space}")
            logger.debug(f"Action space: {self.env.action_space}")
        except Exception as e: # Exception is the base class for most standard errors
            logger.error(f"Error while creating environment '{env_name}': {e}", exc_info=True)
            raise 

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]: # Example: env.reset(options={"low": -0.05, "high": 0.05}) narrows initial-state ranges
        logger.debug(f"seed={seed}, initial-state options={options}, resetting environment...")
        try:
            initial_state, info = self.env.reset(seed=seed, options=options) # Call underlying env.reset
            logger.debug("Environment reset successful!")
            return initial_state, info
        except Exception as e:
            logger.error(f"Error while resetting environment: {e}", exc_info=True)
            raise

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]: # Avoid overly verbose per-step debug logs
        try: 
            observation, reward, terminated, truncated, info = self.env.step(action) # Call underlying env.step
            return observation, reward, terminated, truncated, info
        except Exception as e:
            logger.error(f"Environment step error after agent action {action}: {e}", exc_info=True)
            raise

    def close(self):
        logger.info("Closing environment...")
        try:
            self.env.close()
            logger.debug("Environment closed successfully!")
        except Exception as e:
            logger.error(f"Error while closing environment: {e}", exc_info=True) # Close failures may not require aborting

    @property
    def observation_space(self) -> spaces.Space:
        return self.env.observation_space # Use .shape (e.g., (4,)) or .shape[0] for the state dimension

    @property  # 我们在这里对两个空间并没有限制返回其维度或大小，而是完整暴露，比如在随机采样中可以直接使用action = self.action_space.sample() ，而不必担心对 self.action_space.n 采样报错
    def action_space(self) -> spaces.Space:
        return self.env.action_space # Use .n for the action count (e.g., 2)

    def render(self): # Render trigger; if not called, render_mode="human" may still not display
        try:
            return self.env.render()
        except Exception as e:
            logger.warning(f"Render error: {e}", exc_info=True) # Render errors typically should not interrupt training