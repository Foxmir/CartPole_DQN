# src/agents/agent_dqn.py

import numpy as np
import tensorflow as tf

from src.networks.mlp import MLP
from src.metrics.target_q_normalized import target_q_normalized
from src.utils.logger_setup import setup_logger

logger = setup_logger(__name__) 

class DQNAgent:
    def __init__(self, agent_name, state_space, action_space, learning_rate, epsilon, epsilon_decay, epsilon_min, gamma, tau):
        logger.info("Initializing agent instance...")
        try:
            self.agent_name = agent_name # This name may not be used later, but kept for completeness
            self.state_space = state_space
            self.action_space = action_space
            self.learning_rate = learning_rate
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.epsilon_min = epsilon_min
            self.gamma = gamma
            self.tau = tau

            self.online_network = MLP(self.state_space, self.action_space) # Creating the online network instance does not create weights yet.
                                                                  # Weights are created on first call (build), e.g. q_values = self.online_network(state).
            logger.debug("Online network instance created successfully!")
            self.target_network = MLP(self.state_space, self.action_space) # Create target network instance
            logger.debug("Target network instance created successfully!") # Cannot copy weights yet; weights are created after an actual build (e.g., dummy tensor).

            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # Create Adam optimizer instance; the optimizer manages LR internally.
            logger.debug("Adam optimizer instance created successfully!") # You can log type(self.optimizer).__name__ if needed
            self.loss_fn = tf.keras.losses.MeanSquaredError() # Create loss function instance (MSE). Huber is also a common robust choice.
            logger.debug("MSE loss function instance created successfully!") # You can log type(self.loss_fn).__name__ if needed
            logger.info("Agent instance created successfully!")
        except Exception as e:
            logger.error(f"Error while initializing agent (networks/optimizer/loss): {e}", exc_info=True)
            raise
        
    def select_action(self, state): # Select action using ε-greedy; called frequently so keep logging minimal
        try:
            if np.random.rand() < self.epsilon: # Generates a random float in [0, 1), similar to np.random.uniform(0, 1)
                action = self.action_space.sample() # Random sample from the action space; returns a Python int
            else:
                state = tf.convert_to_tensor(state, dtype=tf.float32) # Convert state to tensor (state_dim,)
                state = tf.expand_dims(state, axis=0) # Add batch dimension: (1, state_dim)
                            
                q_values = self.online_network(state) # instance(arguments) triggers Keras __call__ which invokes our call().
                action = tf.argmax(q_values[0]).numpy() # q_values shape (1, action_dim); take batch[0] then argmax

            if self.epsilon > self.epsilon_min: # Linear decay is also possible; here we use exponential decay
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon) # Do not go below minimum
            return action
        
        except Exception as e:
            logger.error(f"Error while selecting action / updating epsilon: {e}", exc_info=True)
            raise

    # Train model: target network inference -> TD target -> online network training -> update online -> update target -> normalized metrics
    @tf.function
    def learn(self, next_states, states, actions, rewards, dones): # Called frequently; keep logging in the main script

        # 1️⃣ Target network inference
        next_q_values = self.target_network(next_states, training = False) # Shape (batch_size, action_size)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1) # Row-wise max over actions

        # 2️⃣ TD target (Bellman)
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values # TensorFlow will broadcast scalars as needed

        # 3️⃣ Train online network: forward -> gather Q(s,a) -> loss
        with tf.GradientTape() as tape: # Records forward ops for backprop; tape is active only inside the block
            q_values = self.online_network(states, training = False) # Shape (batch_size, action_size)

            batch_size = tf.shape(q_values)[0]
            batch_indices = tf.range(batch_size)
            indices = tf.stack([batch_indices, actions], axis=1)  # 形状 (batch_size, 2)
            q_values = tf.gather_nd(q_values, indices)   # 形状 (batch_size,)

            loss = self.loss_fn(target_q_values, q_values) # Scalar loss for the batch

        # 4️⃣ Update online network
        gradients = tape.gradient(loss, self.online_network.trainable_variables)  # model.trainable_variables returns a list of trainable vars
        self.optimizer.apply_gradients(zip(gradients, self.online_network.trainable_variables))

        # 5️⃣ Target network update: soft update (hard update also possible). tau=1 equals hard copy.
        for target_param, online_param in zip(self.target_network.trainable_variables, self.online_network.trainable_variables):
            target_param.assign((1 - self.tau) * target_param + self.tau * online_param)

        # Normalized metrics using TD target Q values as a scale factor
        gradients_norm = tf.linalg.global_norm(gradients)
        norm_loss, norm_gradients_norm = target_q_normalized(target_q_values, loss, gradients_norm)

        return loss, gradients_norm, norm_loss, norm_gradients_norm
    
    # Local save/load of weights is separated from artifact upload/download (handled elsewhere)
    def save_model(self,weights_file_path): # Save online network weights to local path
        try:
            self.online_network.save_weights(weights_file_path)
            logger.debug(f"Model weights saved to: {weights_file_path}")
        except Exception as e:
            raise

    def load_model(self,weights_file_path): # Load online network weights from local path
        try:
            self.online_network.load_weights(weights_file_path) # Only online network is needed for some evaluation workflows
            q = self.online_network(tf.zeros((1, 4)))
            print("DEBUG: prediction after loading weights:", q.numpy())
            logger.info(f"Model weights loaded successfully from {weights_file_path}")
        except Exception as e:
            raise
