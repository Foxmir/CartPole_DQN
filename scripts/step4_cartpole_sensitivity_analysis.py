# \scripts\step4_cartpole_sensitivity_analysis.py
# Step 4: explore the landscape around the champion hyperparameters selected in Step 3
# We use grid search: within ±10% of each hyperparameter, uniformly sample 5 points.
# The code largely follows the Step 2 main function.


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

logger = setup_logger(__name__)

def main():
    # 1️⃣ Log in to W&B
    if not login_wandb(): 
        logger.critical("!!! [External-1]: W&B login failed; exiting !!!")
        sys.exit(1)


    # 2️⃣ Load config file (no CLI args needed here)
    logger.info("Preparing to load parameters...")
    try: # Use top1 config as the base; W&B sweeps will override defined fields.
        default_config = load_yaml_config(config_path="configs/cartpole_dqn_top1.yaml")
        if default_config is None:
            logger.critical("!!! [External-2]: Hyperparameter config load failed; exiting !!!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error while loading config: '{e}'", exc_info=True)
        logger.critical("!!! [External-2]: Config load failed; exiting !!!")
        sys.exit(1)


    # 3️⃣ Create W&B run and load sweep parameters
    logger.info("Preparing to create W&B run and load sweep parameters...")
    try: 
        run, config = create_run_and_get_config(default_config)

        # # =========== 🔥【临时补丁】手动指定丢失的那组参数 🔥 ===========
        # # 目的：绕过 Sweep Controller，强制补跑丢失的组合
        # # 注意：跑完这次后，请务必把这段代码删掉或注释掉！
        # logger.warning("!!! 注意：正在使用硬编码参数补跑丢失任务 !!!")
        
        # # 1. 强制覆盖 config 中的超参数
        # config.agent['tau'] = 0.03
        # config.agent['epsilon_decay'] = 0.975
        # config.agent['learning_rate'] = 0.00075
        # config.training['batch_size'] = 256
        # config.training['main_seed'] = 502
        
        # # 2. 必须手动更新 wandb.config，否则云端显示的配置还是旧的/默认的
        # #    (因为 create_run_and_get_config 已经 init 过了，我们需要 update)
        # wandb.config.update({
        #     "agent": config.agent,
        #     "training": config.training
        # }, allow_val_change=True)
        
        # logger.warning(f"已强制设定参数: LR={config.agent['learning_rate']}, BS={config.training['batch_size']}, Seed={config.training['main_seed']}")
        # # ==============================================================

    except Exception as e:
        logger.error(f"Error while creating W&B run and loading sweep params: '{e}'", exc_info=True)
        logger.critical("!!! [External-3]: Failed to create W&B run/load sweep params; exiting !!!")
        wandb.finish() # Safe even if run creation failed
        sys.exit(1)


    # 4️⃣-1️⃣ Read config from sweeps
    try:
        num_episodes, initial_collect_size, batch_size, max_episode_steps, n_eval = extract_training_params(config) # n_eval has a defensive default in the helper
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
    loss = float('nan')
    gradients_norm = float('nan')
    norm_loss = float('nan')
    norm_gradients_norm = float('nan')
    
    for i in range(num_episodes): 
        episode_reward = 0
        try:
            current_state,_ = env.reset(seed= env_reset_seed_base + i) # '_' holds the info dict; conventionally unused
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
                action = agent.select_action(current_state) 
                next_state, reward, terminated, truncated,_ = env.step(action) #
                done = terminated or truncated
                train_done = terminated 

                # 【2】Store experience to buffer
                buffer.add(current_state, action, reward, next_state, train_done)

                # 【3】Learn
                if len(buffer) >= initial_collect_size:
                    next_states_tensor,states_tensor, actions_tensor, rewards_tensor, dones_tensor = buffer.sample() 
                    loss, gradients_norm, norm_loss, norm_gradients_norm = agent.learn(next_states_tensor,states_tensor, actions_tensor, rewards_tensor, dones_tensor) 

                # 【4】Episode tail: update state, accumulate reward, check termination
                current_state = next_state
                episode_reward += reward
                
                if done:
                    break
            except Exception as e:
                logger.error(f"Error while training within an episode: {e}", exc_info=True)
                logger.warning("!!! Error occurred during training; exiting !!!")
                try:
                    env.close() # Try to close the environment on failure
                except:
                    pass
                wandb.finish()
                sys.exit(1)


    # 5️⃣ After all training episodes under the current seed, run evaluation and store results
    logger.info(f"\n--- Main seed {main_seed}: training for this parameter set finished; preparing to evaluate... ---")
    try:
        mean, sd, _ = evaluator(agent, env, eva_seed_num=n_eval, eva_env_reset_seed_base=20000) # Also returns scores_list (unused here)
        logger.info(f"--- Main seed {main_seed}: evaluation finished. Mean episodic reward: {mean:.2f}, SD: {sd:.2f} ---\n")

        wandb.log({"final_mean": float(mean), "final_sd": float(sd)}) # 便于某些图标追踪汇总
        run.summary["final_mean"] = float(mean)
        run.summary["final_sd"] = float(sd)
        
        env.close() # 当全部回合都测试完，关闭环境
        logger.info("Environment closed successfully!")
        wandb.finish()
    except Exception as e: # 内部有打印logger
        logger.critical("!!! [External-5]: Failed while evaluating/storing after training; exiting !!!")
        try:
            env.close()
            wandb.finish()
        except:
            pass
        sys.exit(1)

    
    logger.info("\n🎉🎉🎉---Step4 Sensitivity Analysis: data collection completed successfully!---🎉🎉🎉\n"
                "🎉🎉🎉---Enjoy the upcoming statistical analysis!---🎉🎉🎉\n"
                "🎉🎉🎉---Great work and persistence!---🎉🎉🎉")


# Standard entry point
if __name__ == "__main__":
    main()