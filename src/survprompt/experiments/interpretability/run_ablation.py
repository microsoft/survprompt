#!/usr/bin/env python3
"""
Unified ablation study script for Survprompt and RSF.

This keeps the original command-line workflow from the private ablation scripts,
but makes the experiment-set runner self-contained instead of importing
``run_experiment_set`` from a separate module.

Examples:
    # Run Survprompt and RSF with all selective feature sets
    python run_ablation.py --data_name mskchord --cancer_of_interest nsclc

    # Run only Survprompt with the package default prediction engine
    python run_ablation.py --methods survprompt --features all_features stage_only

    # Run a registry experiment instead of the default engine
    python run_ablation.py --methods survprompt --experiments zeroshot_gpt4o --prompting_tasks SURV_PROB
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys

from dotenv import load_dotenv
from survprompt.defaults import DEFAULT_PRED_ENGINE, DEFAULT_DATA_NAME
from survprompt.experiments.base import BaselineExperiment
from survprompt.experiments.experiments import EXPERIMENT_TO_NAME
from survprompt.experiments.interpretability.feature_sets import ABLATION_FEATURE_SETS

load_dotenv(override=True)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ablation experiments with configurable feature sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--data_name", "--data-name", dest="data_name", type=str, default=DEFAULT_DATA_NAME, help="Name of the dataset to use")
    parser.add_argument("--cancer_of_interest", "--cancer", dest="cancer_of_interest", type=str, default="nsclc", help="Cancer cohort to run")
    parser.add_argument("--base_dir", "--base-dir", dest="base_dir", type=str, default=os.getenv("BASE_DIR"), help="Repository/output base directory")
    parser.add_argument("--input_dir", "--input-dir", dest="input_dir", type=str, default=os.getenv("INPUT_DIR"), help="Input data directory")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["survprompt", "rsf"],
        default=["survprompt", "rsf"],
        help="Methods to run in the ablation study",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=list(ABLATION_FEATURE_SETS.keys()),
        default=list(ABLATION_FEATURE_SETS.keys()),
        help="Feature sets to use in the ablation study. Can specify multiple.",
    )
    parser.add_argument(
        "--log_dir",
        "--log_name",
        dest="log_name",
        default="ablation_study",
        help="Name for the log directory",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[DEFAULT_PRED_ENGINE],
        help="Survprompt experiments to run",
    )
    parser.add_argument(
        "--prompting_tasks",
        nargs="+",
        default=["SURV_PROB"],
        choices=["SURV_PROB", "TTE_OS"],
        help="Prompting tasks to run",
    )
    parser.add_argument("--num_folds", "--num-folds", dest="num_folds", type=int, default=5, help="Number of data folds")
    parser.add_argument("--fold", type=int, default=0, help="Fold used for Survprompt predictions")
    parser.add_argument("--sample_size", "--sample-size", dest="sample_size", type=int, default=0, help="Sample size for experiments (0 = no sampling)")
    parser.add_argument("--max_tokens", "--max-tokens", dest="max_tokens", type=int, default=None, help="Maximum tokens. Omit to use the model default.")
    parser.add_argument("--list_features", "--list-features", dest="list_features", action="store_true", help="List available feature sets and exit")
    parser.add_argument("--prompt_path", "--system-prompt-path", dest="prompt_path", type=str, default="system.j2", help="Path to system prompt template")
    parser.add_argument("--exclude_race", "--exclude-race", dest="exclude_race", action="store_true", help="Accepted for compatibility; race exclusion is not applied here")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Survprompt outputs")
    return parser.parse_args()


def list_available_features() -> None:
    """Print available feature sets and their contents."""
    print("Available feature sets:")
    print("=" * 50)
    for name, features in ABLATION_FEATURE_SETS.items():
        print(f"{name:20} : {features}")
    print()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if args.base_dir is None:
        print("Error: BASE_DIR must be set or passed with --base_dir")
        sys.exit(1)
    if args.input_dir is None:
        print("Error: INPUT_DIR must be set or passed with --input_dir")
        sys.exit(1)

    invalid_features = [f for f in args.features if f not in ABLATION_FEATURE_SETS]
    if invalid_features:
        print(f"Error: Invalid feature sets: {invalid_features}")
        print(f"Available feature sets: {list(ABLATION_FEATURE_SETS.keys())}")
        sys.exit(1)

    invalid_experiments = [exp for exp in args.experiments if exp not in EXPERIMENT_TO_NAME]
    if invalid_experiments:
        print(f"Error: Invalid experiments: {invalid_experiments}")
        print(f"Available experiments: {list(EXPERIMENT_TO_NAME.keys())}")
        sys.exit(1)

    if args.sample_size < 0:
        print("Error: Sample size must be non-negative")
        sys.exit(1)

    if args.max_tokens is not None and args.max_tokens <= 0:
        print("Error: Max tokens must be positive")
        sys.exit(1)


def _selected_feature_sets(use_features: dict[str, list[str]]) -> dict[str, list[str]]:
    return {k: v for k, v in ABLATION_FEATURE_SETS.items() if k in use_features}


def run_experiment_for_features(
    log_name: str,
    experiment: str,
    prompting_task: str,
    use_features: dict[str, list[str]],
    data_name: str,
    cancer_of_interest: str,
    base_dir: str,
    input_dir: str,
    num_folds: int,
    fold: int,
    system_prompt_path: str = "system.j2",
    sample_size: int = 0,
    max_tokens: int | None = None,
    overwrite_outputs: bool = False,
) -> None:
    experiment_class = EXPERIMENT_TO_NAME[experiment]
    feature_set = _selected_feature_sets(use_features)
    for exp_name, features in feature_set.items():
        logging.info("Running experiment: %s with features: %s and prompting task: %s", exp_name, features, prompting_task)

        extra_kwargs = {}
        if max_tokens is not None:
            extra_kwargs["max_tokens"] = max_tokens

        experiment_default = experiment_class(
            data_name=data_name,
            cancer_of_interest=cancer_of_interest,
            sample_size=sample_size,
            base_dir=base_dir,
            input_dir=input_dir,
            log_name=os.path.join(log_name, data_name, experiment, prompting_task, exp_name),
            features=features,
            subgroup_features_to_test=None,
            prompting_task=prompting_task,
            overwrite_outputs=overwrite_outputs,
            temperature=0.0,
            num_folds=num_folds,
            fold=fold,
            system_prompt_path=system_prompt_path,
            save_metrics=None,
            **extra_kwargs,
        )
        experiment_default.run()

def run_rsf_for_features(
    log_name: str,
    use_features: dict[str, list[str]],
    data_name: str,
    cancer_of_interest: str,
    base_dir: str,
    input_dir: str,
    num_folds: int,
) -> None:
    feature_set = _selected_feature_sets(use_features)
    for exp_name, features in feature_set.items():
        logging.info("Running RSF ablation: %s with features: %s", exp_name, features)
        experiment = BaselineExperiment(
            baseline="rsf",
            data_name=data_name,
            features=features,
            cancer_of_interest=cancer_of_interest,
            base_dir=base_dir,
            input_dir=input_dir,
            num_folds=num_folds,
            save_outputs=True,
            save_model=False,
            output_fname=f"RSF_ablation/{exp_name}/results",
            log_name=os.path.join(log_name, data_name, "rsf", exp_name),
            subgroup_features_to_test=None,
        )
        experiment.run()


def run_experiment_set(
    experiments: list[str],
    prompting_tasks: list[str],
    log_name: str,
    use_features: dict[str, list[str]],
    data_name: str,
    cancer_of_interest: str,
    base_dir: str,
    input_dir: str,
    num_folds: int,
    fold: int,
    system_prompt_path: str = "system.j2",
    sample_size: int = 0,
    max_tokens: int | None = None,
    overwrite_outputs: bool = False,
) -> None:
    for experiment in experiments:
        for prompting_task in prompting_tasks:
            start_time = datetime.datetime.now()
            logging.info("===== Running experiment: %s with prompting task: %s", experiment, prompting_task)
            run_name = f"TEST_{log_name}" if sample_size != 0 else log_name
            run_experiment_for_features(
                data_name=data_name,
                cancer_of_interest=cancer_of_interest,
                base_dir=base_dir,
                input_dir=input_dir,
                num_folds=num_folds,
                fold=fold,
                log_name=run_name,
                experiment=experiment,
                prompting_task=prompting_task,
                use_features=use_features,
                system_prompt_path=system_prompt_path,
                sample_size=sample_size,
                max_tokens=max_tokens,
                overwrite_outputs=overwrite_outputs,
            )
            time_taken = datetime.datetime.now() - start_time
            logging.info("Time taken for experiment %s with prompting task %s: %s", experiment, prompting_task, time_taken)


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.list_features:
        list_available_features()
        return

    validate_args(args)
    # Filter feature sets based on command line arguments
    feature_set = {k: v for k, v in ABLATION_FEATURE_SETS.items() if k in args.features}

     # Log the configuration
    logging.info("Starting ablation experiment with configuration:")
    logging.info("  Data name: %s", args.data_name)
    logging.info("  Cancer cohort: %s", args.cancer_of_interest)
    logging.info("  Log directory: %s", args.log_name)
    logging.info("  Methods: %s", args.methods)
    logging.info("  Feature sets: %s", args.features)
    logging.info("  Experiments: %s", args.experiments)
    logging.info("  Prompting tasks: %s", args.prompting_tasks)
    logging.info("  Max tokens: %s", args.max_tokens)
    logging.info("  Sample size: %s", args.sample_size)
    logging.info("  Selected features: %s", feature_set)
    logging.info("  System prompt path: %s", args.prompt_path)

    if "survprompt" in args.methods:
        run_experiment_set(
            data_name=args.data_name,
            cancer_of_interest=args.cancer_of_interest,
            base_dir=args.base_dir,
            input_dir=args.input_dir,
            num_folds=args.num_folds,
            fold=args.fold,
            log_name=args.log_name,
            experiments=args.experiments,
            prompting_tasks=args.prompting_tasks,
            use_features=feature_set,
            sample_size=args.sample_size,
            max_tokens=args.max_tokens,
            system_prompt_path=args.prompt_path,
            overwrite_outputs=args.overwrite,
        )

    if "rsf" in args.methods:
        run_rsf_for_features(
            data_name=args.data_name,
            cancer_of_interest=args.cancer_of_interest,
            base_dir=args.base_dir,
            input_dir=args.input_dir,
            num_folds=args.num_folds,
            log_name=args.log_name,
            use_features=feature_set,
        )

    logging.info("Ablation experiment completed successfully!")


if __name__ == "__main__":
    main()
