# scripts/step3Worker_cartpole_model_selection.py
# Get state -> select action -> get env feedback -> store experience -> learn
# Step3 worker: the parallel subprocess that does the work


import sys
import json
import argparse
import wandb
import numpy as np
import tensorflow as tf

from src.utils.evaluator import evaluator
from src.utils.main_common_setup import create_run_and_get_config, extract_training_params, creat_instances
from src.utils.load_yaml_config import load_yaml_config
from src.utils.wandb_login import login_wandb
from src.utils.logger_setup import setup_logger

logger = setup_logger(__name__) # Get logger instance using current module name (__name__)


def main():
    # 1️⃣ Log in to W&B
    if not login_wandb(): # Combine action + check; login_wandb() returns False on failure
        logger.critical("!!! [External-1]: W&B login failed; exiting !!!")
        sys.exit(1) # 1 indicates abnormal exit (0 indicates success)


    # 2️⃣ Parse CLI args (config path, candidate id, random seed, output path)
    logger.info("Preparing to parse command-line arguments and load parameters...")
    try: 
        parser = argparse.ArgumentParser(description="Receiving command-line arguments passed from the controller...")
        parser.add_argument("--top10_candidates_yaml", type=str, required=True) # "configs/top10_candidates.yaml"
        parser.add_argument("--candidate_id", type=str, required=True)
        parser.add_argument("--main_seed", type=int, required=True)
        parser.add_argument("--out", type=str, required=True)

        args, _ = parser.parse_known_args() 
        default_config = load_yaml_config(config_path=args.top10_candidates_yaml) # Pass config path; errors handled inside
        if default_config is None:
            logger.critical("!!! [External-2]: Hyperparameter config load failed; exiting !!!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error while parsing command-line arguments: '{e}'", exc_info=True)
        logger.critical("!!! [External-2]: Failed to parse command-line arguments; exiting !!!")
        sys.exit(1)


    # 3️⃣ Load current candidate config, inject seed (to form a standard config), create W&B run, and load sweep params
    logger.info("Preparing to load candidate config, create W&B run, and load sweep parameters...")
    try:
        candidate_config = default_config[args.candidate_id] # Extract candidate config dict
        candidate_config["training"]["main_seed"] = args.main_seed 
        run, config = create_run_and_get_config(candidate_config) 
        logger.info(f"Loaded hyperparameter config from W&B Sweeps for this run: {dict(config)}")
    except Exception as e:
        logger.error(f"Error while loading candidate config + creating W&B run + loading sweep params: '{e}'", exc_info=True)
        logger.critical("!!! [External-3]: Failed to load candidate config/create W&B run/load sweep params; exiting !!!")
        wandb.finish() # Safe even if run creation failed
        sys.exit(1)


    # 4️⃣-1️⃣ Read config from sweeps
    try:
        num_episodes, initial_collect_size, batch_size, max_episode_steps, n_eval = extract_training_params(config)
    except Exception as e:
        logger.error(f"Error while reading key dictionaries from sweep config: '{e}'", exc_info=True)
        logger.critical("!!! [External-4]: Failed to read key dictionaries from sweep config; exiting !!!")
        wandb.finish() # Attempt to finish run
        sys.exit(1)
        

    # 4️⃣-2️⃣ Create module instances + pass parameters
    try: # No need to duplicate logs; instance constructors are already detailed
        env, buffer, agent, main_seed = creat_instances(config,create_buffer=True)
        dummy = tf.zeros((1,) + env.observation_space.shape, dtype=tf.float32)
        _ = agent.online_network(dummy, training=False) 
        _ = agent.target_network(dummy, training=False)
        agent.target_network.set_weights(agent.online_network.get_weights())
        logger.debug("Network weights initialized via dummy input; target network synced to online network.")
    except Exception as e:
        logger.error(f"Error while creating instances/passing params/building dummy weights: '{e}'", exc_info=True)
        logger.critical("!!! [External-4]: Failed to create instances/passing params/building dummy weights; exiting !!!")
        wandb.finish() # Attempt to finish run
        sys.exit(1)


    # 5️⃣ Training loop
    logger.info("\nStarting model training...")
    env_reset_seed_base = 10000 * main_seed
    if main_seed == 2:
        env_reset_seed_base = 10000**2 * main_seed # Avoid env seed 20000 when main_seed=2, which would overlap with eval protocol
    
    loss = float('nan')
    gradients_norm = float('nan') # nan表示：这里本应有一个数值，但因为某种原因它没有一个有效的数值结果
    norm_loss = float('nan')
    norm_gradients_norm = float('nan')
    
    for i in range(num_episodes): 
        episode_reward = 0 # Reset at episode start; accumulate total reward per episode
        try:
            current_state,_ = env.reset(seed= env_reset_seed_base + i) 
        except Exception as e:
            logger.error(f"Error while resetting environment: '{e}'", exc_info=True)
            logger.critical("!!!  [Internal-5] Training: environment reset failed; exiting !!!")
            try:
                env.close() # Try to close the environment on failure
            except:
                pass
            wandb.finish()
            sys.exit(1)

        for step_in_episode in range(max_episode_steps): 
            try:
                # 【1】Sample an action from the real environment and observe the response
                action = agent.select_action(current_state) # Agent selects action
                next_state, reward, terminated, truncated,_ = env.step(action) # Env response
                done = terminated or truncated 
                train_done = terminated 

                # 【2】Store experience to buffer
                buffer.add(current_state, action, reward, next_state, train_done)

                # 【3】Learn
                if len(buffer) >= initial_collect_size: # Start batch learning after buffer reaches baseline size
                    next_states_tensor,states_tensor, actions_tensor, rewards_tensor, dones_tensor = buffer.sample() # Sample from buffer
                    loss, gradients_norm, norm_loss, norm_gradients_norm = agent.learn(next_states_tensor,states_tensor, actions_tensor, rewards_tensor, dones_tensor) 
                
                # 【4】Episode tail: update state, accumulate reward, check termination
                current_state = next_state
                episode_reward += reward
                
                if done: # Even though we track done and train_done, episode termination uses done
                    break # Not an exception; proceed to next episode
            except Exception as e:
                logger.error(f"Error while training within an episode: {e}", exc_info=True)
                logger.warning("!!! Error occurred during training; exiting !!!")
                try:
                    env.close() # Try to close the environment on failure
                except:
                    pass
                wandb.finish()
                sys.exit(1)


    # 【5】After all training episodes under the current seed, run evaluation and upload
    logger.info(f"\n--- All training episodes under main seed {main_seed} finished; preparing to evaluate... ---")
    try:
        mean, sd, scores_list = evaluator(agent, env, eva_seed_num=n_eval, eva_env_reset_seed_base=20000) # Returns mean, sd, and score list
        logger.info(f"--- Candidate {args.candidate_id}: evaluation under main seed {main_seed} finished. Mean episodic reward: {mean:.2f}, SD: {sd:.2f} ---\n")

        run.summary["c_mean"] = float(mean)
        run.summary["c_sd"] = float(sd)
        wandb.finish()

        env.close()
        logger.info("Environment closed successfully!")
    except Exception as e: # 内部有打印logger
        logger.critical("!!! [Internal-5]: Failed while evaluating and uploading after training; exiting !!!")
        try:
            env.close()
            wandb.finish() # 尝试结束run
        except:
            pass
        sys.exit(1)


    # 6️⃣ Write results to JSON file
    try: # out_path is passed in via args.out
        out_dict = {"c_mean": float(mean), "c_sd": float(sd), "c_scores_list": scores_list}
        with open(args.out, 'w') as f:
            json.dump(out_dict, f, indent=2)
        logger.info(f"Training results successfully written to file: {args.out}")
    except Exception as e: # 内部有打印logger
            logger.critical("!!! [External-6]: Failed to write results to JSON file; exiting !!!")
            sys.exit(1)


    logger.info(f"\n🎉---Step3Worker: seed {main_seed}, candidate {args.candidate_id} finished training + evaluation successfully!---🎉\n")

# Standard entry point
if __name__ == "__main__":
    main()