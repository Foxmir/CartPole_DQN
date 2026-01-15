# src/utils/evaluator.py
# Generic evaluation utility module
# The logic used to determine the evaluator seed count in Step 1 precision analysis is the same
# as the evaluator used during training; they differ only in evaluator seed count/values and
# number of episodes. Therefore we keep a single reusable evaluator here.
import numpy as np
from src.utils.logger_setup import setup_logger

logger = setup_logger(__name__) 

def evaluator(agent, env, eva_seed_num, eva_env_reset_seed_base=20000): 
    scores_list = []
    logger.info(f"Starting evaluation for {eva_seed_num} episodes...")
    try:
        for i in range(eva_seed_num): # Follow the configured number of evaluation episodes
            # if (i) % 10 == 0 or i == 0: # If you're worried about long waits, you can print progress
            #     logger.info(f"--- Evaluation episode {i} running, please wait... ---")

            episode_reward = 0
            current_state,_ = env.reset(seed= eva_env_reset_seed_base + i) # Same env.reset pattern as in the main script

            done = False
            while not done: # Loop ends naturally when done becomes True
                original_epsilon = agent.epsilon
                agent.epsilon = 0 # No exploration during evaluation
                action = agent.select_action(current_state) # Fully greedy policy
                agent.epsilon = original_epsilon # Restore epsilon to avoid side effects for callers
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                current_state = next_state
                episode_reward += reward
            
            # if i % 10 == 0:
                # print(f"Eval {i}: episode_reward={episode_reward}, action={action}")

            scores_list.append(episode_reward)
        logger.info(f"Evaluation completed: {eva_seed_num} episodes.")
    except Exception as e:                  
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise
        
    return np.mean(scores_list), np.std(scores_list), scores_list