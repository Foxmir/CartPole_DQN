# src/utils/main_common_setup.py
# Shared setup steps used at the start of each main script run: create objects, load config, pass params, build instances
# python -m scripts.step1B_cartpole_presision_analysis

import random
import numpy as np
import tensorflow as tf
import wandb

from src.envs.wrappers_cartpole import CartPoleWrapper
from src.memory.replay_buffer import ReplayBuffer
from src.agents.agent_dqn import DQNAgent

from src.utils.logger_setup import setup_logger
from src.utils.wandb_login import create_wandb_run

logger = setup_logger(__name__)

def create_run_and_get_config(default_config): # Create W&B run and obtain sweep-processed config
    try: # Note: env.reset(seed) and env.action_space.seed(seed) must be set separately after env creation.
        run_name = default_config["agent"]["name"]
        run = create_wandb_run(run_name, config=default_config) # If login fails, run may be None
        config = wandb.config # W&B processes config via sweep and returns it through the global wandb.config
        logger.info(f"Loaded hyperparameter config from W&B Sweeps for this run: {dict(config)}")
    except Exception as e:
        raise
    return run, config


def set_seeds(main_seed): # These RNGs are global singletons; setting them here affects the whole process
    try: # No return needed
        random.seed(main_seed)  # e.g., replay batch = random.sample(...)
        np.random.seed(main_seed) # Affects epsilon-greedy decisions, etc.
        tf.random.set_seed(main_seed) # Controls TF-level randomness (weight init, dropout, tf.random.uniform, etc.)
        logger.info(f"Seed check: random={random.random()}")
        logger.info(f"Seed check: np={np.random.rand()}")
        logger.info(f"Seed check: tf={tf.random.uniform((), dtype=tf.float32).numpy()}")
    except Exception as e:
        raise


def extract_training_params(config):
    logger.info("Reading key dictionaries from sweep config...")
    try:
        num_episodes = config["training"]["num_episodes"]
        initial_collect_size = config["training"]["initial_collect_size"]
        batch_size = config["training"]["batch_size"]
        max_episode_steps = config["environment"]["max_episode_steps"]
        if "evaluator" in config and "n_eval" in config["evaluator"]:
            n_eval = config["evaluator"]["n_eval"]
        else:
            n_eval = 70  # 默认值
            logger.warning(f"⚠️ Warning: missing 'evaluator.n_eval' in config; using default {n_eval}")
        logger.info("Successfully read key dictionaries from sweep config.")
    except Exception as e:
        raise
    return num_episodes, initial_collect_size, batch_size, max_episode_steps, n_eval


def creat_instances(config,create_buffer=True):
    logger.info("Creating instances and passing parameters...")
    try:
        main_seed = config["training"]["main_seed"] # Reset global RNG states; affects downstream randomness implicitly.
        logger.info(f"Using random seed: {main_seed}") # Ensure seeds are set before instance creation and env reset
        set_seeds(main_seed)
        
        env = CartPoleWrapper(env_name=config["environment"]["name"], render_mode=config["environment"]["render_mode"])
        env.action_space.seed(main_seed) # Affects env action_space.sample() used during interaction; must be after env creation
        
        buffer = None

        if create_buffer:
            buffer = ReplayBuffer(capacity=config["memory"]["capacity"], 
                                batch_size=config["training"]["batch_size"])
        agent = DQNAgent(agent_name=config["agent"]["name"], # No need to create MLP separately; DQNAgent creates it internally
                        state_space=env.observation_space, # Pass full Box observation space
                        action_space=env.action_space,  # Pass full Discrete action space
                        learning_rate=config["agent"]["learning_rate"],
                        epsilon=config["agent"]["epsilon_start"],
                        epsilon_decay=config["agent"]["epsilon_decay"],
                        epsilon_min=config["agent"]["epsilon_min"],
                        gamma=config["agent"]["gamma"],
                        tau=config["agent"]["tau"])
        logger.info("env, buffer, agent instances created successfully!")
    except Exception as e:
        raise
    return env, buffer, agent, main_seed