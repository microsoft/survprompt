import os
import json
import datetime
import logging
from dotenv import load_dotenv
from dataclasses import asdict

def _create_logdir(log_name: str) -> str:
    """
    Create a unique log directory for the experiment.
    Args:
        log_name: The name of the experiment directory directly under /logs.
        save_subdir: The name of the experiment subdirectory under the experiment directory.
    Returns:
        log_dir: The unique log directory for the experiment.
    """
    # generate a unique log dir based on time and date
    load_dotenv(override=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
    log_dir = os.path.join(os.getenv("BASE_DIR"), "logs", f"{log_name}", f"{date}")
    os.makedirs(log_dir)
    return log_dir

def _save_config(experiment, log_dir: str, pred_cfg=None, exp_cfg=None) -> None:
    """
    Save the configuration of the experiment to files.
    Args:
        experiment: The experiment object containing the configuration.
        log_dir: The directory where the configuration files will be saved.
        pred_cfg: PredictorConfig object (optional)
        exp_cfg: ExperimentConfig object (optional)
    """
    config_dir = os.path.join(log_dir, "config")
    os.makedirs(config_dir, exist_ok=True)

    # Save main experiment config
    config = vars(experiment)
    with open(os.path.join(config_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)
    
    # Save exp_cfg if provided
    if exp_cfg is not None:
        try:
            # Handle both dataclass and regular class objects
            if hasattr(exp_cfg, '__dataclass_fields__'):
                exp_config_dict = asdict(exp_cfg)
            else:
                exp_config_dict = vars(exp_cfg)
            
            with open(os.path.join(config_dir, "exp_config.json"), "w") as f:
                json.dump(exp_config_dict, f, indent=2, default=str)
            logging.info(f"Saved exp_config to: {os.path.join(config_dir, 'exp_config.json')}")
        except Exception as e:
            logging.warning(f"Failed to save exp_config: {e}")
    
    # Save pred_cfg if provided
    if pred_cfg is not None:
        try:
            # Handle both dataclass and regular class objects
            if hasattr(pred_cfg, '__dataclass_fields__'):
                pred_config_dict = asdict(pred_cfg)
            else:
                pred_config_dict = vars(pred_cfg)
            
            with open(os.path.join(config_dir, "pred_config.json"), "w") as f:
                json.dump(pred_config_dict, f, indent=2, default=str)
            logging.info(f"Saved pred_config to: {os.path.join(config_dir, 'pred_config.json')}")
        except Exception as e:
            logging.warning(f"Failed to save pred_config: {e}")