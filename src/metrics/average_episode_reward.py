# src/metrics/average_episode_reward.py

from src.utils.logger_setup import setup_logger

logger = setup_logger(__name__) 

def get_average_episode_reward(rewards_records): # Incremental-style update: average reward over recent 100 episodes (hard-coded in main)
    try:
        average_episode_reward = round(sum(rewards_records)/len(rewards_records))
        return average_episode_reward
    except Exception as e:
        logger.error(f"Error while computing average reward over the latest 100 episodes: {e}", exc_info=True)
        raise

