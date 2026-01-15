# Path: scripts/main_cartpole_dqn.py
# Purpose: Main training entry-point for CartPole DQN experiments.

# Observe state -> choose action -> get env feedback -> store experience -> learn
# Run from repo root (CartPole_DQN):
# python -m scripts.main_cartpole_dqn


import os
import sys
import wandb
import argparse
import collections 
import tensorflow as tf
import time

from src.metrics.average_episode_reward import get_average_episode_reward

from src.utils.main_common_setup import create_run_and_get_config, extract_training_params, creat_instances
from src.utils.artifact_utils import ArtifactManager
from src.utils.device_setup import get_device  # Select CPU/GPU device
from src.utils.load_yaml_config import load_yaml_config
from src.utils.wandb_login import login_wandb
from src.utils.logger_setup import setup_logger

logger = setup_logger(__name__)  # Logger instance (module name: scripts.main_cartpole_dqn)


def main():
    # 1) Login to W&B
    # The helper returns a boolean (False on failure). If login fails, we must exit.
    if not login_wandb():
        logger.critical("!!! [External-1]: W&B login failed. Exiting. !!!")
        sys.exit(1)  # 1 = error exit (0 = success). Raising is also fine in some designs.


    # 2) Parse CLI args + load the selected YAML config
    # This part differs across entry points, so it is not moved into shared modules.
    logger.info("Parsing CLI args and loading config...")
    try:
        parser = argparse.ArgumentParser(description="Select a YAML config file to load")
        parser.add_argument(
            "--config",
            type=str,
            default="cartpole_dqn_good.yaml",
            help="Config filename under configs/ (including .yaml)",
        )
        # Ignore unknown args for compatibility with W&B Sweeps (it may pass extra params).
        args, _ = parser.parse_known_args()
        config_path_to_load = os.path.join("configs", args.config)
        logger.info(f"Config path resolved: {config_path_to_load}. Loading...")
        
        default_config = load_yaml_config(config_path=config_path_to_load)
        if default_config is None:
            logger.critical("!!! [External-2]: Config load failed. Exiting. !!!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error while parsing CLI args: '{e}'", exc_info=True)
        logger.critical("!!! [External-2]: CLI parse failed. Exiting. !!!")
        sys.exit(1)


    # 3) Create W&B run + load sweep params + create artifact manager
    logger.info("Creating W&B run and loading sweep params...")
    try:
        run, config = create_run_and_get_config(default_config) 
        artifact_manager = ArtifactManager(run)
    except Exception as e:
        logger.error(
            f"Failed to create W&B run / load sweep params / prepare artifact paths: '{e}'",
            exc_info=True,
        )
        logger.critical(
            "!!! [External-3]: Failed to create W&B run / load sweep params / prepare artifact paths. Exiting. !!!"
        )
        wandb.finish()
        sys.exit(1)


    # 4) Read key training params from config
    try:
        device = get_device()
        num_episodes, initial_collect_size, batch_size, max_episode_steps, _ = extract_training_params(config)
    except Exception as e:
        logger.error(f"Failed to read required training params from config: '{e}'", exc_info=True)
        logger.critical("!!! [External-4]: Missing/invalid training params. Exiting. !!!")
        wandb.finish()
        sys.exit(1)


    with tf.device(device):
        # 4b) Create module instances + build model weights via a dummy forward pass
        try:
            env, buffer, agent, main_seed = creat_instances(config,create_buffer=True)
            dummy = tf.zeros((1,) + env.observation_space.shape, dtype=tf.float32)
            _ = agent.online_network(dummy, training=False) 
            _ = agent.target_network(dummy, training=False)
            agent.target_network.set_weights(agent.online_network.get_weights())
            logger.debug("Model weights initialized via dummy input; target network synced to online network.")
        except Exception as e:
            logger.error(f"Failed to create instances / build dummy weights: '{e}'", exc_info=True)
            logger.critical("!!! [External-4]: Failed to create instances / build dummy weights. Exiting. !!!")
            wandb.finish()
            sys.exit(1)


        # 5) Training loop
        logger.info("\nStarting training...")
        total_steps = 0  # total steps across episodes
        rewards_records = collections.deque(maxlen=100)
        # Use a large offset to reduce the chance of accidental correlation with other RNG streams.
        env_reset_seed_base = 10000 * main_seed
        # W&B: define a custom x-axis
        wandb.define_metric("episode")
        wandb.define_metric("episode_reward", step_metric="episode")
        
        # Before the replay buffer reaches the warmup size, learning won't run, so keep NaNs for logging.
        loss = float('nan')
        gradients_norm = float('nan')
        norm_loss = float('nan')
        norm_gradients_norm = float('nan')
        best_average_episode_reward = -float('inf')
        
        t0 = time.perf_counter()
        t_env_step = 0.0
        t_learn = 0.0
        t_wblog = 0.0
        t_artifact = 0.0
        
        for i in range(num_episodes):
            episode_reward = 0
            try:
                current_state,_ = env.reset(seed= env_reset_seed_base + i)
            except Exception as e:
                logger.error(f"Error while resetting env: '{e}'", exc_info=True)
                logger.critical("!!! Training: env reset failed. Exiting. !!!")
                wandb.finish()
                sys.exit(1)

            for step_in_episode in range(max_episode_steps):
                total_steps += 1
                # episode_steps += 1
                try:
                    # (1) Sample an action and step the env
                    action = agent.select_action(current_state)
                    start = time.perf_counter()
                    next_state, reward, terminated, truncated,_ = env.step(action)
                    t_env_step += time.perf_counter() - start
                    done = terminated or truncated
                    # For training targets, treat truncation as "not terminal" (bootstrapping allowed).
                    train_done = terminated

                    # (2) Store transition in replay buffer
                    buffer.add(current_state, action, reward, next_state, train_done)

                    # (3) Learn
                    if len(buffer) >= initial_collect_size:
                        next_states_tensor,states_tensor, actions_tensor, rewards_tensor, dones_tensor = buffer.sample()
                        start = time.perf_counter()
                        loss, gradients_norm, norm_loss, norm_gradients_norm = agent.learn(next_states_tensor,states_tensor, actions_tensor, rewards_tensor, dones_tensor)
                        t_learn += time.perf_counter() - start
                        
                        # To reduce overhead, skip per-step W&B logging.
                        # wandb.log({
                        #     "epsilon": agent.epsilon,
                        #     "loss": loss,
                        #     "gradients_norm": gradients_norm,
                        #     "norm_loss": norm_loss,
                        #     "norm_gradients_norm": norm_gradients_norm
                        #     }, step=total_steps)  # Use total cross-episode steps as the x-axis
                    # (4) Episode tail: update state, accumulate reward, check termination
                    current_state = next_state
                    episode_reward += reward
                    
                    if done:
                        rewards_records.append(episode_reward)
                        average_episode_reward = get_average_episode_reward(rewards_records)

                        if average_episode_reward > best_average_episode_reward:
                            start = time.perf_counter()
                            artifact_manager.save_and_upload_to_artifact(agent)
                            best_average_episode_reward = average_episode_reward
                            t_artifact += time.perf_counter() - start

                        if (i + 1) % 50 == 0 or i == 0:
                            logger.info(f"--- Episode {i+1} finished. Average episode reward: {average_episode_reward} ---")
                        
                        start = time.perf_counter()
                        # These are end-of-episode values (not per-step).
                        wandb.log({
                            "episode_reward": episode_reward,
                            "average_episode_reward": average_episode_reward,
                            "epsilon": agent.epsilon,
                            "loss": loss,
                            "gradients_norm": gradients_norm,
                            "norm_loss": norm_loss,
                            "norm_gradients_norm": norm_gradients_norm,
                            "episode":i+1
                            }) 
                        t_wblog += time.perf_counter() - start
                        break
                except Exception as e:
                    logger.error(f"Error during episode training: {e}", exc_info=True)
                    logger.warning("!!! Training error occurred. Exiting. !!!")
                    wandb.finish()
                    sys.exit(1)


        # 6) Cleanup: close W&B, close env, print timing breakdown, check graph compilation
        try:
            wandb.finish()
            env.close()
            logger.info("Environment closed successfully!")
            
            total_wall = time.perf_counter() - t0
            env_share = 100.0 * t_env_step / total_wall if total_wall > 0 else 0.0
            learn_share = 100.0 * t_learn / total_wall if total_wall > 0 else 0.0
            log_share = 100.0 * t_wblog / total_wall if total_wall > 0 else 0.0
            artifact_share = 100.0 * t_artifact / total_wall if total_wall > 0 else 0.0
            other_share = 100.0 - (env_share + learn_share + log_share + artifact_share)
            # Overall throughput (software+hardware efficiency): steps per second (SPS)
            steps_per_sec = total_steps / total_wall if total_wall > 0 else 0.0
            print(f"""
                  [Performance Summary]
                  - Total Steps: {total_steps}
                  - Wall Time:   {total_wall/60:.2f}min
                  - Throughput:  {steps_per_sec:.1f} steps/sec
                  - Time Shares:
                    - Env Step:  {env_share:.1f}%
                    - Learn:     {learn_share:.1f}%
                    - W&B Log:   {log_share:.1f}%
                    - Artifact:  {artifact_share:.1f}%
                    - Other:     {other_share:.1f}%
                  """)
            
            if agent.learn.pretty_printed_concrete_signatures():
                print("✅ learn graph compilation: success")
            else:
                print("❌ learn graph compilation: failed")
            
        except Exception as e:
            logger.error(
                f"Cleanup failed (wandb/env/timing/graph-check) for env '{env.env_name}': '{e}'",
                exc_info=True,
            )
        
        
        logger.info("\n🎉🎉🎉---Training finished successfully!---🎉🎉🎉\n🎉🎉🎉---Nice work!---🎉🎉🎉")

# Standard entry point
if __name__ == "__main__":
    main()