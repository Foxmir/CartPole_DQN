# Path: scripts/step1B_cartpole_presision_analysis.py
# Purpose: Precision/reliability analysis to estimate how many training seeds are needed.
# Run from repo root (CartPole_DQN): python -m scripts.step1B_cartpole_presision_analysis

import sys
import wandb
import numpy as np
import tensorflow as tf
import time, collections
from math import ceil

from src.metrics.average_episode_reward import get_average_episode_reward

from src.utils.evaluator import evaluator
from src.utils.wandb_login import login_wandb
from src.utils.logger_setup import setup_logger
from src.utils.load_yaml_config import load_yaml_config
from src.utils.main_common_setup import create_run_and_get_config, extract_training_params, creat_instances

logger = setup_logger(__name__)

# 1) Login to W&B
if not login_wandb(): 
    logger.critical("!!! [External 1]: W&B login failed, exiting. !!!")
    sys.exit(1)


# 2) Load hyperparameter config (this script uses a fixed config path)
default_config = load_yaml_config("configs/cartpole_dqn_defaults.yaml")  # NOTE: confirm path
if default_config is None:
    logger.critical("!!! [External 2]: Failed to load the hyperparameter config, exiting. !!!") 
    sys.exit(1) 


# 3) Create W&B run + load sweep parameters
logger.info("Preparing to create a W&B run and load sweep parameters...")
try:
    run, config = create_run_and_get_config(default_config) 
except Exception as e:
    logger.error(f"Failed while creating W&B run and/or loading sweep params from config: '{e}'", exc_info=True)
    logger.critical("!!! [External 3]: Failed to create W&B run / load sweep params, exiting. !!!")
    sys.exit(1)


# 4) Read key training params from sweep config
try:
    num_episodes, initial_collect_size, batch_size, max_episode_steps, n_eval = extract_training_params(config)
except Exception as e:
    logger.error(f"Failed to read required fields from sweep config: '{e}'", exc_info=True)
    logger.critical("!!! [External 4]: Failed to read required sweep config fields, exiting. !!!")
    wandb.finish()  # Try to end the run
    sys.exit(1)


# From here on, each main seed goes through the same training+evaluation loop
train_seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]  # 20 different main seeds
train_means = []  # Evaluation means for each training seed
train_sds = []  # Evaluation standard deviations for each training seed

t0 = time.perf_counter()  # ⏱
t_env_step = 0.0  # ⏱
t_learn = 0.0  # ⏱
t_wblog = 0.0  # ⏱
t_eval = 0.0  # ⏱

for seed in train_seeds:
    # 4.1) Override main training seed
    config["training"]["main_seed"] = seed
    logger.info(f"Main training seed set to: {seed}")

    # 4.2) Create module instances
    try:  # Instance constructors already log enough detail and raise on failure
        env, buffer, agent, main_seed = creat_instances(config,create_buffer=True) 
        dummy = tf.zeros((1,) + env.observation_space.shape, dtype=tf.float32)  # Force-build networks to create weights
        _ = agent.online_network(dummy, training=False) 
        _ = agent.target_network(dummy, training=False)
        agent.target_network.set_weights(agent.online_network.get_weights())  # Sync target weights after build
        logger.debug("Network weights initialized via dummy input; target network synced to online network.")
    except Exception as e:
        logger.error(f"Failed while creating instances / wiring config: '{e}'", exc_info=True)
        logger.critical("!!! [Internal 4]: Failed to create instances, exiting. !!!")
        wandb.finish()  # Try to end the run
        sys.exit(1)


    # 5) Training loop (similar logic to the default main)
    logger.info("\nStarting training...")
    total_steps = 0  # Total steps across episodes
    rewards_records = collections.deque(maxlen=100)
    env_reset_seed_base = 10000 * main_seed  # Avoid accidental correlations between env resets and main seed
    wandb.define_metric("episode")  # W&B uses a global step by default; define a custom x-axis
    wandb.define_metric("episode_reward", step_metric="episode")

    loss = float('nan')  # Before initial_collect_size, learn() won't run; keep placeholders for logging
    gradients_norm = float('nan')
    norm_loss = float('nan')
    norm_gradients_norm = float('nan')

    for i in range(num_episodes):
        episode_reward = 0
        try:
            current_state,_ = env.reset(seed= env_reset_seed_base + i)  # '_' is info dict; intentionally unused
        except Exception as e:
            logger.error(f"Failed to reset environment: '{e}'", exc_info=True)
            logger.critical("!!! [Internal 5]: Environment reset failed during training, exiting. !!!")
            wandb.finish()
            sys.exit(1)

        for step_in_episode in range(max_episode_steps):
            total_steps += 1
            try:
                # (1) Interact with the real environment
                action = agent.select_action(current_state)
                start = time.perf_counter()  # ⏱
                next_state, reward, terminated, truncated,_ = env.step(action)
                t_env_step += time.perf_counter() - start  # ⏱
                done = terminated or truncated
                train_done = terminated  # For training targets, only natural termination is treated as terminal

                # (2) Store transition into replay buffer
                buffer.add(current_state, action, reward, next_state, train_done)

                # (3) Learn
                if len(buffer) >= initial_collect_size:  # Ensure enough diverse samples before learning
                    next_states_tensor,states_tensor, actions_tensor, rewards_tensor, dones_tensor = buffer.sample()
                    start = time.perf_counter()  # ⏱
                    loss, gradients_norm, norm_loss, norm_gradients_norm = agent.learn(next_states_tensor,states_tensor, actions_tensor, rewards_tensor, dones_tensor)
                    t_learn += time.perf_counter() - start  # ⏱

                # (4) Episode bookkeeping
                current_state = next_state
                episode_reward += reward
                
                if done:
                    rewards_records.append(episode_reward)
                    average_episode_reward = get_average_episode_reward(rewards_records)

                    if (i + 1) % 50 == 0 or i == 0:
                        logger.info(f"--- Episode {i+1} finished. Average episode reward: {average_episode_reward} ---")
                    
                    break
            except Exception as e:
                logger.error(f"Error during per-step training loop: {e}", exc_info=True)
                logger.critical("!!! [Internal 5]: Training loop failed, exiting. !!!")
                wandb.finish()
                sys.exit(1)


    # 6) After all episodes, evaluate via the evaluator protocol
    try:
        start = time.perf_counter()  # ⏱
        mean, sd, scores_list = evaluator(agent, env, eva_seed_num=n_eval, eva_env_reset_seed_base=20000)
        train_means.append(mean)
        train_sds.append(sd)
        t_eval += time.perf_counter() - start  # ⏱
        print(f"Finished training with seed={seed}. Evaluation over {n_eval} episodes: mean={mean:.2f}, std={sd:.2f}")

        env.close()
        logger.info("Environment closed successfully.")
    except Exception as e:
        logger.critical("!!! [External 6]: Evaluation and/or environment close failed, exiting. !!!")
        wandb.finish()  # Try to end the run
        sys.exit(1)


# 7) Compute seed count estimate from collected data
try:
    target_error = 10  # Acceptable error: +/- 10 points
    train_seed_sd = np.std(np.array(train_means), ddof=1)
    n_train = ceil(1.96 * train_seed_sd / target_error ** 2)  # ceil = round up
    logger.info(
        f"Per-seed evaluation means: {train_means}\n"
        f"Per-seed evaluation stds:   {train_sds}\n"
        f"Std across training seeds:  {train_seed_sd}\n"
        f"Estimated #training seeds:  {n_train}\n"
    )
except Exception as e:
    logger.error(f"Failed while computing required training seed count: '{e}'", exc_info=True)
    logger.critical("!!! [External 7]: Failed to compute required seed count, exiting. !!!")
    wandb.finish()  # Try to end the run
    sys.exit(1)


# 8) Upload summary to W&B and print performance breakdown
try:
    wandb.log({
        "train_means": train_means,
        "train_seed_sd": train_seed_sd,
        "n_train": n_train
        })
    wandb.finish()

    # ⏱⏱⏱
    total_wall = time.perf_counter() - t0  # Wall-clock time
    env_share = 100.0 * t_env_step / total_wall if total_wall > 0 else 0.0
    learn_share = 100.0 * t_learn / total_wall if total_wall > 0 else 0.0
    eval_share = 100.0 * t_eval / total_wall if total_wall > 0 else 0.0
    other_share = 100.0 - (env_share + learn_share + eval_share)
    print(f"""
            [Performance Summary]
            - Total Steps: {total_steps}
            - Wall Time:   {total_wall/60:.2f}min
            - Time Shares:
            - Env Step:  {env_share:.1f}%
            - Learn:     {learn_share:.1f}%
            - Evaluation: {eval_share:.1f}%
            - Other:     {other_share:.1f}%
            """)
except Exception as e:
    logger.error(
        f"Failed during W&B upload/finish and/or timing summary print (non-critical): '{e}'",
        exc_info=True,
    )
    wandb.finish()  # Try to end the run
    sys.exit(1)

logger.info("\n--- Step1B precision analysis: training and evaluation completed successfully. ---\n")