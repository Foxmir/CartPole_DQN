# src/memory/replay_buffer.py

from collections import deque # deque is a double-ended queue; fast append/pop on both ends and drops oldest when maxlen is reached
import random
import tensorflow as tf

from src.utils.logger_setup import setup_logger

logger = setup_logger(__name__) 

class ReplayBuffer:

    def __init__(self, capacity, batch_size):
        logger.info("Initializing replay buffer instance...")
        try:
            self.buffer = deque(maxlen=capacity) # Starts empty (length 0); does not pre-fill zeros
            self.batch_size = batch_size
            logger.debug("Replay buffer instance created successfully!")
        except Exception as e:
            logger.error(f"Error while creating replay buffer instance: {e}", exc_info=True)
            raise
        
    def __len__(self): # Called by len(instance)
        try: # Called frequently; avoid logging unless necessary
            return len(self.buffer) # Use len(instance), not len(instance.buffer)
        except Exception as e:
            logger.error(f"Error while getting replay buffer length: {e}", exc_info=True)
            raise

    def add(self, state, action, reward, next_state, done):
        try:
            experience = (state, action, reward, next_state, done) # Pack one transition tuple: (state, action, reward, next_state, done)
            self.buffer.append(experience)
        except Exception as e:
            logger.error(f"Error while packing/appending experience to buffer: {e}", exc_info=True)
            raise

    def sample(self): # Similar name to random.sample(); this method takes no args
        try:
            batch = random.sample(self.buffer, self.batch_size) # Sample batch_size experiences
            states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor = (
                tf.convert_to_tensor(data, dtype=tf.float32 if i != 1 else tf.int32) # Use int32 for actions; others are float32
                for i, data in enumerate(zip(*batch))
                )
            return next_states_tensor,states_tensor, actions_tensor, rewards_tensor, dones_tensor # Order matches agent.learn() signature
        except Exception as e:
            logger.error(f"Error while sampling/unpacking/converting batch to tensors: {e}", exc_info=True)
            raise