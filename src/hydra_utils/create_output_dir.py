import json
import os
import subprocess
from datetime import datetime

from loguru import logger


class ExperimentUtils:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_git_commit_hash(self):
        try:
            return (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("ascii")
                .strip()
            )
        except Exception as e:
            logger.error(f"Error obtaining Git commit hash: {e}")
            return "unknown"

    def create_directory(self, path):
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except OSError as e:
            logger.error(f"Error creating directory {path}: {e}")
            raise

    def create_experiment_subdirectories(self, main_dir):
        subdirs = ["models", "outputs"]
        paths = {}
        for subdir in subdirs:
            dir_path = os.path.join(main_dir, subdir)
            paths[subdir] = self.create_directory(dir_path)
        return paths

    def prepare_experiment_config(self):
        cfg = self.cfg
        config = {
            "base_dir": "debug" if cfg.task_name == "debug" else cfg.paths.output_dir,
            "task_model": None,
            "extra": None,
            "training_mode": None,
            "params": None,
        }
        params = [
            f"lr-{cfg.model.optimizer.lr}",
            f"wd-{cfg.model.optimizer.weight_decay}",
            f"ep-{cfg.trainer.max_epochs}",
            f"bs-{cfg.data.train_batch_size}",
            f"act-{cfg.model.net.activation}",
            f"opt-{cfg.model.optimizer._target_.split('.')[-1]}",
            f"lrsched-{cfg.model.scheduler._target_.split('.')[-1]}",
        ]
        config["params"] = "_".join(filter(None, params))
        dir_components = [
            config[key]
            for key in [
                "task_model",
                "params",
            ]
            if config[key]
        ]
        config["dir_name"] = "_".join(dir_components)
        return config

    def save_experiment_metadata(self, output_dir):
        cfg = self.cfg
        # cfg.experiment_start_time = datetime.now().isoformat()
        # Add system and environment information if needed
        metadata_path = os.path.join(output_dir, "metadata.json")

        from omegaconf import OmegaConf

        # dumps to file:
        with open(metadata_path, "w") as f:
            OmegaConf.save(cfg, f)

    def get_output_dir(self, project_name):
        # Prepare the configuration for the experiment
        config = self.prepare_experiment_config()

        # Construct the main directory name for the experiment
        main_dir = os.path.join(config["base_dir"], config["dir_name"], project_name)
        main_dir_path = self.create_directory(main_dir)

        # Create subdirectories for models and outputs
        subdirectories = self.create_experiment_subdirectories(main_dir_path)

        # Save the experiment metadata
        self.save_experiment_metadata(main_dir_path)

        # Return paths to the main, model, and output directories
        return main_dir_path, subdirectories
