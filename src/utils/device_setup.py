# src/utils/device_setup.py

import tensorflow as tf

from src.utils.logger_setup import setup_logger 

logger = setup_logger(__name__) 

def get_device(): # Check available hardware and return the best device name
    gpus = tf.config.list_physical_devices('GPU') # List all available GPU devices
    if gpus:
        device_name = '/GPU:0' # If GPU exists, use the first one by default
        tf.config.experimental.set_memory_growth(gpus[0], True) # Prevent TF from pre-allocating all VRAM
        logger.info(f"GPU detected. Using device: {device_name}")
    else:
        device_name = '/CPU:0' # If no GPU, use CPU
        logger.info(f"No GPU detected. Using device: {device_name}")
    return device_name