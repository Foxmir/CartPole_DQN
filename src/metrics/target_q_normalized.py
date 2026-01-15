# CartPole_DQN/src/metrics/target_q_normalized.py

import tensorflow as tf

from src.utils.logger_setup import setup_logger

logger = setup_logger(__name__) 

def target_q_normalized(q_targets, loss, gradients_norm): # Use target-network Q values as the normalization factor
    try:
        mean_abs_q_target = tf.reduce_mean(tf.abs(q_targets)) # Mean absolute target Q
        norm_loss = loss / (tf.square(mean_abs_q_target) + 1e-8) # Add epsilon to avoid division by zero
        norm_gradients_norm = gradients_norm / (mean_abs_q_target + 1e-8)
        return norm_loss, norm_gradients_norm
    except Exception as e:
        logger.error(f"Error while creating metrics (normalized loss and gradient norm): {e}", exc_info=True)
        raise