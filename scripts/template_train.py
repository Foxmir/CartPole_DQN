# Path: scripts/template_train.py
# Purpose: Training script template (not executed); may contain placeholders and is for reference only.

# Observe state -> choose action -> get env feedback -> store experience -> learn

import wandb  # Kept for reference: run.log(), etc. are still used here.
import time
import os
import json

# Import modules from the local 'src' package
from src.agents.base_agent import BaseAgent
from src.envs.wrappers import BaseWrapper
from src.memory.replay_buffer import ReplayBuffer
from src.networks.mlp import MLP
from src.utils.load_config import load_config
# NOTE: Example change: import the wrapped init helper and keep the login helper.
from src.utils.wandb_utils import login_wandb, initialize_wandb_run

def main():
    """Entry point for a mock training loop using wrapped W&B helpers."""
    print("--- Starting training mock ---")

        # --- Try logging into W&B ---
    if not login_wandb():
            print("Warning: W&B login failed. Continuing without W&B logging.")
            # return  # Intentionally kept as a comment

    # --- Load config ---
    config = load_config()
    if config is None:
        print("Failed to load config. Exiting.")
        return

    # --- Example change: call the wrapped initialization helper ---
    # Instead of calling wandb.init() directly, call our helper and pass the config dict.
    run = initialize_wandb_run(experiment_config=config)
    # 'run' is now a W&B run object or None

    # --- Initialization ---
    print("--- Initializing components ---")
    # Create component instances from the loaded config (logic unchanged)
    agent = BaseAgent(agent_name=config["agent"]["name"],
                      learning_rate=config["agent"]["learning_rate"])
    print(f"Agent '{config['agent']['name']}' created.")

    env = BaseWrapper(env_name=config["environment"]["name"])
    print(f"Environment wrapper for '{config['environment']['name']}' created.")

    buffer = ReplayBuffer(capacity=config["memory"]["capacity"])
    print(f"Replay buffer (capacity {config['memory']['capacity']}) created.")

    network = MLP(input_dim=4, output_dim=2, hidden_dims=config["network"]["hidden_dims"])
    print(f"Network ({config['network']['type']}) created.")
    print("--- Component initialization complete ---")

    # --- Mock training loop ---
    print("--- Entering mock training loop ---")
    # (wandb.log calls remain the same, but you must check whether run is None)
    num_episodes = config["training"]["num_episodes"]
    batch_size = config["training"]["batch_size"]
    total_steps = 0

    for i in range(num_episodes):
        episode_reward = 0
        episode_steps = 0
        print(f"\n--- Episode {i+1}/{num_episodes} ---")
        current_state = env.reset()

        max_episode_steps = 50
        for step in range(max_episode_steps):
            total_steps += 1
            episode_steps += 1

            action = agent.select_action(current_state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            buffer.add(current_state, action, reward, next_state, done)

            loss = 1.0 / (total_steps + 1)

            # --- W&B logging (per step) ---
            if run:  # Check whether the run was initialized successfully
                log_data = {
                    "step": total_steps,
                    "episode": i + 1,
                    "step_reward": reward,
                    "step_loss": loss
                }
                run.log(log_data, step=total_steps)  # Default behavior commits immediately

            if len(buffer) >= batch_size:
                experiences = buffer.sample(batch_size)
                agent.learn(experiences)

            current_state = next_state
            if done:
                print(f"Episode {i+1} ended after {episode_steps} steps. Total reward: {episode_reward}")
                break

        # --- W&B logging (end of episode) ---
        if run:  # Check whether the run was initialized successfully
            run.log({
                "episode_reward": episode_reward,
                "episode_steps": episode_steps
             }, step=total_steps)

    # --- Finish W&B run ---
    if run:  # Check whether the run was initialized successfully
        run.finish()
        print("--- W&B Run finished ---")

    print("\n--- Training mock finished ---")
    env.close()
    print("--- Environment closed ---")

# Standard entry point
if __name__ == "__main__":
    main()