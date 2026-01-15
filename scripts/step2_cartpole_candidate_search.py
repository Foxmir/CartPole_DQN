# Path: scripts/step2_cartpole_candidate_search.py
# Purpose: Step2 candidate search via W&B sweeps (Bayes); train/evaluate multiple seeds and log a single objective metric.

# RL loop: get state -> select action -> step env -> store transition -> learn

import os
import sys
import wandb
import argparse
import numpy as np
import tensorflow as tf
import time

from src.utils.evaluator import evaluator
from src.utils.main_common_setup import create_run_and_get_config, extract_training_params, creat_instances
from src.utils.load_yaml_config import load_yaml_config
from src.utils.wandb_login import login_wandb
from src.utils.logger_setup import setup_logger

logger = setup_logger(__name__)  # Logger for this module

def main():
    # 1) Login to W&B
    if not login_wandb():
        logger.critical("!!! [External 1]: W&B login failed, exiting. !!!")
        sys.exit(1)


    # 2) Parse CLI args + load config (W&B sweeps can override via injected args; keeping CLI support for convenience)
    logger.info("Parsing command-line args and loading config...")
    try:
        parser = argparse.ArgumentParser(description="Select which config YAML to load from configs/")
        parser.add_argument(
            "--config",
            type=str,
            default="cartpole_dqn_good.yaml",
            help="Config file name under configs/ (including .yaml)",
        )
        args, _ = parser.parse_known_args()  # Ignore unknown args to stay compatible with sweep controllers
        config_path_to_load = os.path.join("configs", args.config)
        logger.info(f"Parsed config path: {config_path_to_load}. Loading...")

        default_config = load_yaml_config(config_path=config_path_to_load)
        if default_config is None:
            logger.critical("!!! [External 2]: Failed to load config YAML, exiting. !!!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed while parsing args/loading config: '{e}'", exc_info=True)
        logger.critical("!!! [External 2]: Failed to parse args / load config, exiting. !!!")
        sys.exit(1)


    # 3) Create W&B run and load sweep parameters
    logger.info("Creating W&B run and loading sweep parameters...")
    try:
        run, config = create_run_and_get_config(default_config) 
    except Exception as e:
        logger.error(f"Failed while creating W&B run / loading sweep params: '{e}'", exc_info=True)
        logger.critical("!!! [External 3]: Failed to create W&B run / load sweep params, exiting. !!!")
        wandb.finish()  # Safe even if run creation failed
        sys.exit(1)


    # 4) Read training parameters from sweep config
    try:
        num_episodes, initial_collect_size, batch_size, max_episode_steps, n_eval = extract_training_params(config)
    except Exception as e:
        logger.error(f"Failed to read required fields from sweep config: '{e}'", exc_info=True)
        logger.critical("!!! [External 4]: Failed to read sweep config fields, exiting. !!!")
        wandb.finish()
        sys.exit(1)


    # From here on, each main seed goes through the same train+eval loop
    train_seeds = [42, 43, 44, 45, 46]  # Fixed: 5 seeds for Bayes optimization
    train_means_list = []  # Evaluation means per training seed
    train_sds_list = []  # Evaluation stds per training seed (usually not used for optimization)

    t0 = time.perf_counter()  # ⏱
    t_env_step = 0.0  # ⏱
    t_learn = 0.0  # ⏱
    t_eval = 0.0  # ⏱

        
    for seed in train_seeds:
        # 4.1) Override main training seed
        config["training"]["main_seed"] = seed
        logger.info(f"Main training seed set to: {seed}")

        # 4.2) Create instances
        try:
            env, buffer, agent, main_seed = creat_instances(config,create_buffer=True)
            dummy = tf.zeros((1,) + env.observation_space.shape, dtype=tf.float32)  # Force-build network weights
            _ = agent.online_network(dummy, training=False) 
            _ = agent.target_network(dummy, training=False)
            agent.target_network.set_weights(agent.online_network.get_weights())
            logger.debug("Network weights initialized via dummy input; target synced to online.")
        except Exception as e:
            logger.error(f"Failed while creating instances/building dummy weights: '{e}'", exc_info=True)
            logger.critical("!!! [External 4]: Failed to create instances / build dummy weights, exiting. !!!")
            wandb.finish()
            sys.exit(1)


        # 5) Training loop
        logger.info("\nStarting training...")
        env_reset_seed_base = 10000 * main_seed  # Avoid accidental correlations between env reset seeds and main seed
        
        loss = float('nan')  # learn() won't run before initial_collect_size; keep placeholders
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
                try:
                    env.close()  # Best-effort
                except:
                    pass
                wandb.finish()
                sys.exit(1)

            for step_in_episode in range(max_episode_steps):
                try:
                    # (1) Interact with the environment
                    action = agent.select_action(current_state)
                    start = time.perf_counter()  # ⏱
                    next_state, reward, terminated, truncated,_ = env.step(action)
                    t_env_step += time.perf_counter() - start  # ⏱
                    done = terminated or truncated
                    train_done = terminated  # For training targets, only natural termination is treated as terminal

                    # (2) Store transition
                    buffer.add(current_state, action, reward, next_state, train_done)

                    # (3) Learn
                    if len(buffer) >= initial_collect_size:
                        next_states_tensor,states_tensor, actions_tensor, rewards_tensor, dones_tensor = buffer.sample()
                        start = time.perf_counter()  # ⏱
                        loss, gradients_norm, norm_loss, norm_gradients_norm = agent.learn(next_states_tensor,states_tensor, actions_tensor, rewards_tensor, dones_tensor)
                        t_learn += time.perf_counter() - start  # ⏱

                    # (4) Episode bookkeeping
                    current_state = next_state
                    episode_reward += reward
                    
                    if done:
                        break
                except Exception as e:
                    logger.error(f"Error during training loop: {e}", exc_info=True)
                    logger.warning("Training loop error; exiting.")
                    try:
                        env.close()
                    except:
                        pass
                    wandb.finish()
                    sys.exit(1)


        # (5) After training under this seed, evaluate via the evaluator protocol
        logger.info(f"\n--- Training finished for seed {seed}. Starting evaluation... ---")
        try:
            start = time.perf_counter()  # ⏱
            mean, sd, scores_list = evaluator(agent, env, eva_seed_num=n_eval, eva_env_reset_seed_base=20000)
            train_means_list.append(mean)
            train_sds_list.append(sd)
            t_eval += time.perf_counter() - start  # ⏱
            logger.info(f"--- Evaluation finished for seed {seed}. Mean return: {mean:.2f}, std: {sd:.2f} ---\n")

            env.close()
            logger.info("Environment closed successfully.")
        except Exception as e:
            logger.critical("!!! [Internal 5]: Failed during post-training evaluation for this seed, exiting. !!!")
            try:
                env.close()
            except:
                pass
            wandb.finish()
            sys.exit(1)


    # 6) Aggregate across seeds, log metrics to W&B, and print timing breakdown
    try:
        config_all_train_seed_mean = np.mean(np.array(train_means_list))
        config_all_train_seed_sd = np.std(np.array(train_means_list), ddof=1)
        logger.info(
            f"""\n[Aggregate across training seeds]\n"
            f"- Per-seed mean returns: {train_means_list}\n"
            f"- Per-seed eval stds:    {train_sds_list} (usually not used for selection)\n"
            f"- Objective mean:        {config_all_train_seed_mean:.2f} (Bayes optimization metric)\n"
            f"- Objective std:         {config_all_train_seed_sd:.2f}\n"
            f"""
        )
        wandb.log({
            "train_means_list": train_means_list,
            "train_sds_list": train_sds_list,
            "config_all_train_seed_mean": config_all_train_seed_mean,
            "config_all_train_seed_sd": config_all_train_seed_sd
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
                - Wall Time:   {total_wall/60:.2f}min
                - Time Shares:
                - Env Step:  {env_share:.1f}%
                - Learn:     {learn_share:.1f}%
                - Eval:      {eval_share:.1f}%
                - Other:     {other_share:.1f}%
                """)
    except Exception as e:
        logger.critical(
            f"Failed during aggregate metric computation / W&B upload / timing summary: '{e}'",
            exc_info=True,
        )
        wandb.finish()
        sys.exit(1)
    
    logger.info("\n--- Step2 candidate search (Bayes sweep) completed successfully. ---\n")

# Standard entry point
if __name__ == "__main__":
    main()