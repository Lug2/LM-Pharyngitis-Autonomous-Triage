
import os
import yaml
import logging
import pandas as pd
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BenchmarkTask(ABC):
    def __init__(self, output_dir: str, config_overrides: dict = None):
        self.output_dir = output_dir
        self.config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')
        self.config_overrides = config_overrides or {}
        
        # Ensure task-specific subdir exists
        self.task_dir = os.path.join(self.output_dir, self.name.lower())
        os.makedirs(self.task_dir, exist_ok=True)
        
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def run(self) -> dict:
        """
        Executes the benchmark task.
        Returns a dictionary of summary metrics to be aggregated.
        """
        pass

    def load_config(self, config_name: str) -> dict:
        path = os.path.join(self.config_dir, config_name)
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Apply Overrides
        if self.config_overrides.get('n_samples') is not None:
             if 'simulation' in config:
                 config['simulation']['iso_samples'] = self.config_overrides['n_samples']
                 logger.info(f"Override: Set isolation samples to {self.config_overrides['n_samples']}")
        
        if self.config_overrides.get('noise') is not None:
             if 'simulation' in config:
                 config['simulation']['max_noise'] = self.config_overrides['noise']
                 logger.info(f"Override: Set max noise to {self.config_overrides['noise']}")

        if self.config_overrides.get('steps') is not None:
             if 'simulation' in config:
                 config['simulation']['robustness_steps'] = self.config_overrides['steps']
                 logger.info(f"Override: Set robustness steps (resolution) to {self.config_overrides['steps']}")

        if self.config_overrides.get('rob_samples') is not None:
             if 'simulation' in config:
                 config['simulation']['robustness_samples'] = self.config_overrides['rob_samples']
                 logger.info(f"Override: Set robustness samples per step to {self.config_overrides['rob_samples']}")

        return config

    def save_csv(self, df: pd.DataFrame, filename: str):
        path = os.path.join(self.task_dir, filename)
        df.to_csv(path, index=False)
        logger.info(f"Saved {filename} to {path}")

    def save_plot(self, fig, filename: str):
        path = os.path.join(self.task_dir, filename)
        fig.savefig(path)
        logger.info(f"Saved plot {filename} to {path}")

    def get_model_config_path(self, config: dict) -> str:
        """
        Resolves model config path relative to the config file location.
        Config file is in `Benchmark/configs/`.
        """
        rel_path = config.get('paths', {}).get('model_config')
        if not rel_path:
            # Fallback path if not in config
            return os.path.abspath(os.path.join(self.config_dir, "../../inference_model/model_config.yaml"))
            
        return os.path.abspath(os.path.join(self.config_dir, rel_path))

