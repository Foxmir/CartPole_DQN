# Path: src/utils/load_yaml_config.py
# Purpose: Load a YAML config file from the repo root and return it as a dict.

import yaml
import os

from src.utils.logger_setup import setup_logger

logger = setup_logger(__name__)

def load_yaml_config(config_path="configs/cartpole_dqn_defaults.yaml"):  # config_path is relative to the repo root
    logger.info("Opening the YAML config file...")
    config = None
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
        full_path = os.path.join(project_root, config_path)  # repo_root + relative path
        logger.info(f"Resolved config path: '{full_path}'")
        with open(full_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded and parsed successfully: '{full_path}'")
    except FileNotFoundError:
        logger.error(f"Config file not found: '{full_path}'")
    except yaml.YAMLError as e:  # YAML parse errors
        logger.error(f"Invalid YAML or parse error in '{full_path}': {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error while loading '{full_path}': {e}", exc_info=True)
    return config