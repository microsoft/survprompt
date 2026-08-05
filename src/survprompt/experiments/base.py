import os
from dotenv import load_dotenv
from typing import List
import logging
from survprompt.baselines.run_rsf import run_rsf
from survprompt.baselines.run_cox import run_cox
from survprompt.predict_survival import predict_survival
from survprompt.configs.predictor_config import PredictorConfig
from survprompt.configs.exp_config import ExperimentConfig
from survprompt.experiments.utils import _create_logdir, _save_config
from survprompt.data.utils import features_available, process_features
from survprompt.defaults import (
    DEFAULT_DATA_NAME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PRED_ENGINE,
    DEFAULT_REASONING_EFFORT,
)

load_dotenv(override=True)

class Experiment:
    """
    Base class for all experiments.
    """
    def __init__(self, log_name: str = None, save_logs: bool = True):
        self.log_name = log_name
        self.log_dir = self._initialize_logging() if save_logs else None

    def _initialize_logging(self) -> None:
        if self.log_name is None:
            self.log_name = self.__class__.__name__
        self.log_dir = _create_logdir(self.log_name)
        return self.log_dir

    def save_config(self, experiment, pred_cfg=None, exp_cfg=None):
        _save_config(experiment, self.log_dir, pred_cfg, exp_cfg)
        logging.info(f"Saved config to: {self.log_dir}")

    def run(self):
        raise NotImplementedError


class BaselineExperiment(Experiment):
    """
    Base experiment class for reproducing baselines.
    """

    def __init__(
            self,
            baseline: str = "rsf",
            data_name: str = DEFAULT_DATA_NAME,
            features: str = "all",
            cancer_of_interest: str = "nsclc",
            base_dir: str = os.getenv("BASE_DIR"),
            input_dir: str = os.getenv("INPUT_DIR"),
            num_folds: int = 5,
            save_outputs: bool = False,
            save_model: bool = False,
            output_fname: str = "baseline_results",
            log_name: str = None,
            subgroup_features_to_test: List[str] = ['STAGE_IV_DX', 'STAGE_I-III_NOPROG'],
            save_logs: bool = True
    ):
        super().__init__(log_name=log_name, save_logs=save_logs)
        self.baseline = baseline
        self.data_name = data_name
        self.features = process_features(features)
        self.cancer_of_interest = cancer_of_interest
        self.base_dir = base_dir
        self.input_dir = input_dir
        self.num_folds = num_folds
        self.save_outputs = save_outputs
        self.save_model = save_model
        self.output_fname = output_fname
        self.subgroup_features_to_test = subgroup_features_to_test
        self.save_logs = save_logs

    def run(self):
        # Define configs
        exp_cfg = ExperimentConfig(
                        data_name = self.data_name,
                        cancer_of_interest = self.cancer_of_interest,
                        input_dir = self.input_dir,
                        base_dir = self.base_dir,
                        features = self.features,
                        num_folds = self.num_folds,
                        exp_name = self.output_fname,
                        subgroup_features_to_test = self.subgroup_features_to_test,
                        save_logs = self.save_logs
                    )
        if exp_cfg.save_logs:
            self.save_config(experiment=self, pred_cfg=None, exp_cfg=exp_cfg)
        if self.baseline == "rsf":
            run_rsf(exp_cfg=exp_cfg,
                    save_outputs=self.save_outputs,
                    save_model=self.save_model)
        elif self.baseline == 'cox':
            run_cox(exp_cfg=exp_cfg,
                    save_outputs=self.save_outputs,
                    save_model=self.save_model)


class BaseExperiment(Experiment):
    def __init__(
        self,
        data_name: str = DEFAULT_DATA_NAME,
        cancer_of_interest: List[str] | str = "nsclc",
        features: List[str] | str = "all",
        base_dir: str = os.getenv("BASE_DIR"),
        input_dir: str = os.getenv("INPUT_DIR"),
        num_folds: int = 5,
        fold: int = 0,
        prediction_number: int = 0,
        pred_engine: str = DEFAULT_PRED_ENGINE,
        prompting_task: str = "SURV_PROB",
        requests_per_minute: int = 30,
        temperature: float = 0.4,
        max_tokens: int = 128,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        exp_name: str = "zeroshot",
        experiment_type: str = "arbitrary_fixed",
        sample_size: int = -1,
        complete_outputs: bool = True,
        overwrite_outputs: bool = False,
        subgroup_features_to_test: List[str] = ['STAGE_IV_DX', 'STAGE_I-III_NOPROG'],
        log_name: str = None,
        system_prompt_path: str = 'system.j2',
        save_logs: bool = True,
        save_metrics: List[str] = None,
    ):
        super().__init__(log_name=log_name, save_logs=save_logs)
        self.data_name = data_name
        self.features_available = features_available
        self.features = self._process_features(features)
        self.cancer_of_interest = cancer_of_interest
        self.base_dir = base_dir
        self.input_dir = input_dir
        self.num_folds = num_folds
        self.fold = fold
        self.prediction_number = prediction_number
        self.pred_engine = pred_engine
        self.prompting_task = prompting_task
        self.requests_per_minute = requests_per_minute
        self.temperature = temperature
        if pred_engine == DEFAULT_PRED_ENGINE and max_tokens == 128:
            max_tokens = DEFAULT_MAX_TOKENS
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.exp_name = exp_name
        self.exp_type = experiment_type
        self.sample_size = sample_size
        self.complete_outputs = complete_outputs
        self.overwrite_outputs = overwrite_outputs
        self.subgroup_features_to_test = subgroup_features_to_test
        self.system_prompt_path = system_prompt_path
        self.save_logs = save_logs
        self.save_metrics = save_metrics


    def _process_features(self, features: list[str] | str) -> list[str]:
        """Process the 'features' argument to expand 'all'."""
        if features == "all":
            return self.features_available
        if isinstance(features, list) and "all" in features:
            return self.features_available
        return features
    
    def run(self):
        # Define configs
        exp_cfg = ExperimentConfig(
                            data_name = self.data_name,
                            cancer_of_interest = self.cancer_of_interest,
                            input_dir = self.input_dir,
                            base_dir = self.base_dir,
                            features = self.features,
                            num_folds = self.num_folds,
                            sample_size = self.sample_size,
                            fold = self.fold,
                            prediction_number = self.prediction_number,
                            exp_name = self.exp_name,
                            exp_type = self.exp_type,
                            complete_outputs = self.complete_outputs,
                            overwrite_outputs = self.overwrite_outputs,
                            subgroup_features_to_test = self.subgroup_features_to_test,
                            log_dir = self.log_dir,
                            save_logs = self.save_logs,
                            save_metrics = self.save_metrics,
                        )
        pred_cfg = PredictorConfig(
                            pred_engine = self.pred_engine,
                            prompting_task = self.prompting_task,
                            temperature = self.temperature,
                            max_tokens = self.max_tokens,
                            requests_per_minute = self.requests_per_minute,
                            system_prompt_path = self.system_prompt_path,
                            reasoning_effort = self.reasoning_effort,
                        )

        self.exp_cfg = exp_cfg
        self.pred_cfg = pred_cfg
        if exp_cfg.save_logs:
            self.save_config(experiment=self, pred_cfg=pred_cfg, exp_cfg=exp_cfg)
        logging.info(f"Running on {self.data_name}, {self.exp_name} with {self.features}, {self.cancer_of_interest}")
        predict_survival(exp_cfg = exp_cfg, 
                         pred_cfg = pred_cfg)
