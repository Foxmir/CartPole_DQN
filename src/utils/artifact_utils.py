# \src\utils\artifact_utils.py
# Separate local weight save/load from artifact upload/download
import os
import wandb
from src.utils.logger_setup import setup_logger

logger = setup_logger(__name__)

class ArtifactManager: # Step3 controller also uploads nested YAML artifacts directly, so it may not use this class
    def __init__(self, run):
        self.run = run

    def save_and_upload_to_artifact(self, agent, filename_suffix="_best.weights.h5"): # W&B expects weights to end with .weights.h5
        try:
            model_filename = f"{self.run.name}{filename_suffix}" # Weight filename
            weights_file_path = os.path.join(self.run.dir, model_filename) # wandb.run.dir is the auto-synced run directory
            logger.debug(f"Weights file to save/upload for this run: {weights_file_path}")

            agent.save_model(weights_file_path=weights_file_path) # Save weights locally
            artifact = wandb.Artifact(name=f"model-{self.run.id}", type="model") # Create artifact; run.id helps uniquely identify the run
            artifact.add_file(weights_file_path) # Add local weights file to artifact
            self.run.log_artifact(artifact) # Upload artifact to cloud
            # wandb.save(weights_file_path, policy="live") # Not needed when using artifacts to manage weights
            logger.debug(f"Model saved locally: {weights_file_path}, and uploaded to W&B.")
        except Exception as e:
            logger.error(f"Failed to save locally and upload model weights to W&B artifact: {e}", exc_info=True)
            raise

    def download_and_load_from_artifact(self, agent, artifact_path): # Used mainly for precision analysis when needing a trained agent
        try:
            logger.info(f"Downloading artifact from W&B: {artifact_path}")
            artifact = self.run.use_artifact(artifact_path, type='model') # Tells W&B which artifact to use; handles download/versioning
            artifact_dir = artifact.download() # Downloads to local cache directory and returns the path
            weights_filename = [f for f in os.listdir(artifact_dir) if f.endswith(".weights.h5")][0] # Expect exactly one weights file per artifact version
            weights_path = os.path.join(artifact_dir, weights_filename)
            agent.load_model(weights_path)  
            logger.info("Model loaded successfully; ready to start precision analysis...")
        except Exception as e:
            logger.error(f"Failed to download model artifact from W&B and load weights locally: {e}", exc_info=True)
            raise