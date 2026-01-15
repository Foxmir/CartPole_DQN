# Path: tool/before_sweeps.py
# Purpose: Convenience script to ensure W&B login works before launching sweeps.

from src.utils.wandb_login import login_wandb

login_wandb()  # The function prints success/failure internally.

# If running the main code once, run the main entry-point script directly.
# For batch runs, use the two commands below to start W&B Sweeps (Bayes/grid search).

# Edit the sweep config and run the sweep creation command first.
# It will return a sweep_id, which you then plug into the agent command.
# wandb sweep configs/bayes_cartpole_dqn.yaml --project RL_Project_Data --entity foxmir-stanford-university

# Template
# wandb agent --count 50 foxmir-stanford-university/RL_Project_Data/YOUR_NEW_SWEEP_ID

# Example: a specific Bayes sweep run
# wandb agent --count 50 foxmir-stanford-university/RL_Project_Data/7h2rkbyy
