# Path: scripts/step1A_cartpole_precision_analysis.py
# Purpose: Precision/reliability analysis to choose the number of evaluation seeds (n_eval).

# Run from the repo root (CartPole_DQN)
# Example (good config, ~500 episodes):
# python -m scripts.step1A_cartpole_precision_analysis --model_artifact_name foxmir-stanford-university/RL_Project_Data/model-p7ci3nct:v137
# Example (good config, ~200 episodes):
# python -m scripts.step1A_cartpole_precision_analysis --model_artifact_name foxmir-stanford-university/RL_Project_Data/model-ji7ko554:v88

import sys
import wandb
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from pathlib import Path
from matplotlib.ticker import MaxNLocator

from src.utils.evaluator import evaluator
from src.utils.wandb_login import login_wandb
from src.utils.logger_setup import setup_logger
from src.utils.artifact_utils import ArtifactManager
from src.utils.load_yaml_config import load_yaml_config
from src.utils.main_common_setup import create_run_and_get_config, extract_training_params, creat_instances

logger = setup_logger(__name__)

def main():
    # 1️⃣ Login to W&B
    if not login_wandb(): 
        logger.critical("!!! [External-1]: W&B login failed. Exiting. !!!")
        sys.exit(1)


    # 2️⃣-1️⃣ Parse CLI args for the model weights artifact name
    logger.info("Parsing command-line arguments...")  # Model artifact names are dynamic; CLI input is the simplest way to provide it.
    try:
        parser = argparse.ArgumentParser(description="Step 1: Precision Analysis")
        parser.add_argument(
            "--model_artifact_name",
            type=str,
            required=False,
            help="Full W&B artifact path for the model weights",
        )  # Example: python -m scripts.step1A_cartpole_precision_analysis --model_artifact_name org/project/model-xxxx:v123
        args = parser.parse_args()  # Strictly parse CLI
        if not args.model_artifact_name:
            args.model_artifact_name = input("Enter the full W&B weights artifact path to load: ").strip()  # Paste from the W&B UI
        artifact_weights_path_to_load = args.model_artifact_name      
    except Exception as e:
        logger.error(f"Error while parsing CLI args: '{e}'", exc_info=True)
        logger.critical("!!! [External-2]: Failed to parse CLI args. Exiting. !!!")
        sys.exit(1)


    # 2️⃣-2️⃣ Load hyperparameter config (this script reads weights from CLI instead)
    good_config = load_yaml_config("configs/cartpole_dqn_good.yaml")  # NOTE: confirm the path
    if good_config is None:
        logger.critical("!!! [External-2]: Failed to load the hyperparameter config. Exiting. !!!")
        sys.exit(1) 
    

    # 3️⃣ Create W&B run, load sweep params, and create artifact manager
    logger.info("Creating W&B run and loading sweep parameters...")
    try:
        run, config = create_run_and_get_config(good_config) 
        artifact_manager = ArtifactManager(run)
    except Exception as e:
        logger.error(
            f"Error while creating W&B run / loading sweep params: '{e}'",
            exc_info=True,
        )
        logger.critical("!!! [External-3]: Failed to create W&B run / load sweep params. Exiting. !!!")
        sys.exit(1)


    # 4️⃣-1️⃣ Read training parameters from sweep config
    try:
        # We do not need n_eval here (even if a default exists) because Step1A is meant to determine it.
        num_episodes, initial_collect_size, batch_size, max_episode_steps, _ = extract_training_params(config)
    except Exception as e:
        logger.error(f"Error while reading key fields from sweep config: '{e}'", exc_info=True)
        logger.critical("!!! [External-4]: Failed to read sweep config fields. Exiting. !!!")
        wandb.finish()  # Try to end the run
        sys.exit(1)


    # 4️⃣-2️⃣ Create instances + build model weights via a dummy forward pass
    try:  # Instances have their own detailed logs; errors will raise.
        # No replay buffer is needed because we are only evaluating.
        env, buffer, agent, _ = creat_instances(config, create_buffer=False)
        # Build the network weights so that loading weights below can overwrite existing variables.
        dummy = tf.zeros((1,) + env.observation_space.shape, dtype=tf.float32)
        # We only need the online network (weights were uploaded for the online network).
        _ = agent.online_network(dummy, training=False)
    except Exception as e:
        logger.error(f"Error while creating instances / building dummy weights: '{e}'", exc_info=True)
        logger.critical("!!! [External-4]: Failed to create instances / build dummy weights. Exiting. !!!")
        wandb.finish()  # Try to end the run
        sys.exit(1)


    # 5️⃣ Download model weights from W&B (avoid local weights to keep a single source of truth)
    try:
        artifact_manager.download_and_load_from_artifact(agent, artifact_weights_path_to_load)
    except Exception as e:  # Internal logger already prints details
        logger.critical("!!! [External-5]: Failed to download model weights. Exiting. !!!")
        wandb.finish()  # Try to end the run
        sys.exit(1)


    # 6️⃣ Core: evaluate with many seeds once, then compute CI half-width for different N
    seed_num = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    eva_outcome_sds = [] # standard deviation
    eva_outcome_ses = [] # standard error 
    eva_outcome_halfwidths = [] # CI half-width = 1.96 * SE

    max_seeds_num = max(seed_num)  # Run the maximum once to save time
    logger.info(f"Running baseline evaluation for {max_seeds_num} episodes...")
    
    try:
        # returns: mean, std, scores_list
        mean_all, sd_all, scores_list = evaluator(agent, env, eva_seed_num=max_seeds_num, eva_env_reset_seed_base=20000)
        print(f"All {max_seeds_num} evaluation episodes: mean={mean_all:.2f}, std={sd_all:.2f}")
    except Exception as e:  # evaluator logs internally
        logger.error(f"[External-6] Baseline evaluation failed: {e}", exc_info=True)
        logger.critical("!!! [External-6]: Baseline evaluation failed. Exiting. !!!")
        wandb.finish()  # Try to end the run
        sys.exit(1)

    env.close()
    logger.info("Environment closed successfully!")


    # 7️⃣ Compute half-widths and pick the smallest N meeting the target threshold
    try:
        logger.info("Computing half-widths for different evaluation sample sizes...")
        for n in seed_num: 
            subset = np.array(scores_list[:n], dtype=np.float64)
            if n >= 2:
                # Use sample std (ddof=1). When n=1, ddof=1 becomes NaN, so guard above.
                sd = subset.std(ddof=1)
                se = sd / np.sqrt(n)  # Standard error of the mean
                halfwidth = 1.96 * se  # ~95% CI half-width
            else:
                sd, se, halfwidth = 0.0, 0.0, 0.0
            eva_outcome_sds.append(sd)
            eva_outcome_ses.append(se)
            eva_outcome_halfwidths.append(halfwidth)

        target_error = 10  # Acceptable error is +/- 10 points
        chosen_n = None
        # Choose the first N whose CI half-width meets the threshold.
        for n, hw in zip(seed_num, eva_outcome_halfwidths):
            if hw <= target_error:
                chosen_n = n
                break
        logger.info(f"Minimum n_eval meeting target_error={target_error} is {chosen_n}")
    except Exception as e:
        logger.error(f"Error while computing half-widths: '{e}'", exc_info=True)
        logger.critical("!!! [External-7]: Failed to compute half-widths. Exiting. !!!")
        wandb.finish()  # Try to end the run
        sys.exit(1)

    # 8️⃣ Plot + save locally + upload to W&B
    try:
        logger.info("Plotting and uploading chart to W&B...")
        plt.figure(figsize=(10, 6), dpi=120)
        plt.plot(seed_num, eva_outcome_halfwidths, marker='o', linestyle='-')  # CI curve
        plt.title("Precision Analysis: 95% CI Half-Width vs N_eval")
        plt.xlabel("Number of Evaluation Episodes (N_eval)")
        plt.ylabel("95% CI Half-Width of Mean Return (≈ 1.96 × SE)")
        plt.grid(True)

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Integer ticks on x-axis
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))  # y-axis format

        ax.axhline(y=target_error, color='red', linestyle='--', linewidth=1)  # Threshold line
        ax.text(seed_num[0], target_error, f" target={target_error}", color='red', va='bottom')

        if chosen_n is not None:  # Highlight the chosen N
            chosen_hw = eva_outcome_halfwidths[seed_num.index(chosen_n)]
            ax.axvline(x=chosen_n, color='green', linestyle='--', linewidth=1)
            ax.scatter([chosen_n], [chosen_hw], color='green', zorder=5)
            ax.text(chosen_n, chosen_hw, f"  chosen N={chosen_n}", color='green', va='bottom')

        project_root = Path(__file__).resolve().parents[1]  # scripts/ parent is usually repo root
        png_path = project_root / "step1A_precision_halfwidth.png"
        plt.savefig(png_path)  # Save figure

        logger.info(f"[Step1A] Image saved to: {png_path}")
        logger.info(f"[Step1A] Image mtime: {png_path.stat().st_mtime}")

        wandb.log({"Step1A_precision_analysis_halfwidth_chart": wandb.Image(str(png_path))})
        plt.close()
        logger.info("Analysis complete; chart uploaded to W&B")
    except Exception as e:
        logger.error(f"Error while plotting/uploading to W&B: '{e}'", exc_info=True)
        logger.critical("!!! [External-8]: Failed to plot/upload to W&B. Exiting. !!!")
        wandb.finish()  # Try to end the run
        sys.exit(1)
    wandb.finish()

if __name__ == "__main__":
    main()