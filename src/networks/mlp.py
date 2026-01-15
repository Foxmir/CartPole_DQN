# src/networks/mlp.py
import tensorflow as tf

from src.utils.logger_setup import setup_logger

logger = setup_logger(__name__) 

class MLP(tf.keras.Model): # Define the DQN network via Keras model subclassing
   
    def __init__(self, state_space, action_space): # __init__ defines layers only; call() defines the forward connections.
        super().__init__() # Must call parent initializer
        logger.info("Initializing neural network instance...")
        try:
            self.dense1 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform', input_shape=(state_space.shape))  # Explicit input shape can speed up build.
            logger.debug(f"Dense layer created: '{self.dense1}'...")
            self.dense2 = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform') # He init matches ReLU
            logger.debug(f"Dense layer created: '{self.dense2}'...")
            # self.dense3 = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform')
            # logger.debug(f"'{self.dense3}'密集层创建成功...")
            self.out = tf.keras.layers.Dense(action_space.n, activation=None) # For a full summary, the model must be fully built.
            logger.debug(f"Output layer created: '{self.out}'...")
            logger.info("Neural network instance created successfully!")
        except Exception as e:
            logger.error(f"Error while creating neural network instance: {e}", exc_info=True)
            raise

    def call(self, inputs): # Define forward pass here; required for subclassed models.
        try: # Forward pass is called frequently; avoid noisy logs
            x = self.dense1(inputs)
            x = self.dense2(x)
            # x = self.dense3(x)
            return self.out(x) # Keras calls call() via __call__ when you do model(inputs)
        except Exception as e:
            logger.error(f"Error during forward pass: {e}", exc_info=True)
            raise
