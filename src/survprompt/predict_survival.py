from typing import Any, List, Dict, Tuple
import argparse
import logging

import json
import os
import pandas as pd

from survprompt.data.utils import features_available
from survprompt.predictor import OpenAIOutcomePredictor
from survprompt.configs.predictor_config import PredictorConfig
from survprompt.configs.exp_config import ExperimentConfig
from survprompt.predictor_utils import get_tabular_data
from survprompt.defaults import DEFAULT_DATA_NAME, DEFAULT_MAX_TOKENS, DEFAULT_PRED_ENGINE, DEFAULT_REASONING_EFFORT
from survprompt.evaluation.metrics import calculate_c_index, assess_on_subcohorts


def parse_args():
    parser = argparse.ArgumentParser(description='Train Random Survival Forest model with options.')
    # Data options
    parser.add_argument('--data_name', type=str, default=DEFAULT_DATA_NAME, help='Dataset to use')
    parser.add_argument('--input_dir', type=str, default=os.getenv("INPUT_DIR"), help='Directory containing input data')
    parser.add_argument('--features', nargs='+', choices=features_available+['all'], required=True, help='Feature sets to include')
    parser.add_argument('--cancer_of_interest', nargs='+', required=True, help='Cancer types to include')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--sample_size', type=int, default=-1, help='Number of samples to predict')
    parser.add_argument('--fold', type=int, default=0, help='Fold to use')
    parser.add_argument('--prediction_number', type=int, default=0, help='Prediction number to use (ie 0 for first, 1 for second, etc. Used for best of n, ensemble or consistency)')
    # Output options
    parser.add_argument('--base_dir', type=str, default=os.getenv("BASE_DIR"), help='Base directory')
    parser.add_argument('--exp_name', type=str, default='zeroshot', help='Experiment name')
    parser.add_argument('--exp_type', type=str, default='arbitrary_fixed', help='Experiment type')
    parser.add_argument('--complete_outputs', type=bool, default=True, help='Re-prompt model only for NaN predictions')
    parser.add_argument('--overwrite_outputs', action='store_true', help='Overwrite existing prediction outputs if found')
    parser.add_argument('--save_metrics', nargs='+', choices=['c-index', 'coverage'], default=None, help='Metrics to save to file (default: None)')
    
    # Prediction engine options
    parser.add_argument('--pred_engine', type=str, default=DEFAULT_PRED_ENGINE, help='Prediction engine to use')
    parser.add_argument('--prompting_task', type=str, choices=['TTE_OS', 'SURV_PROB'], default='SURV_PROB', help='Prompting task (eg. predict for median OS, survival)')
    parser.add_argument('--requests_per_minute', type=int, default=30, help='Number of requests per minute for OpenAI API')
    parser.add_argument('--temperature', type=float, default=0.4, help='Temperature to use when prompting.')
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS, help='Max tokens to use when prompting. For o1 models corresponds to completion tokens.')
    parser.add_argument('--reasoning_effort', type=str, default=DEFAULT_REASONING_EFFORT, help='Reasoning effort to use when prompting.')
    parser.add_argument('--max_retries', type=int, default=15, help='Max retries for the predictor.')
    parser.add_argument('--use_async', type=bool, default=True, help='Whether to use AsyncAzureOpenAI or not.')
    
    args = parser.parse_args()
    
    return args

def _get_output_fname(exp_cfg: ExperimentConfig,
                     pred_cfg: PredictorConfig) -> str:
    """Get the output filename for the predictions"""
    filename = _get_fname(exp_cfg, pred_cfg)
    save_dir = _get_predictions_dir(exp_cfg, pred_cfg)
    return f'{save_dir}/{filename}'

def _get_logs_fname(exp_cfg: ExperimentConfig,
                    pred_cfg: PredictorConfig) -> str:
    """Get the output filename for the logs"""
    filename = _get_fname(exp_cfg, pred_cfg)
    save_dir = _get_logs_dir(exp_cfg, pred_cfg)
    return f'{save_dir}/{filename}'

def _omit_temperature_suffix(pred_engine: str) -> bool:
    """Reasoning-style GPT-5 deployments do not expose temperature controls."""
    return pred_engine.startswith('gpt-5')

def _get_fname(exp_cfg: ExperimentConfig,
              pred_cfg: PredictorConfig) -> str:
    model_info = f'{pred_cfg.pred_engine}'
    if 'gpt-5' in model_info:
        model_info += f'-{pred_cfg.reasoning_effort}'
    temp_suffix = '' if _omit_temperature_suffix(pred_cfg.pred_engine) else f'_temp{pred_cfg.temperature}'
    filename = f'{exp_cfg.exp_name}_{"_".join(exp_cfg.cancer_of_interest)}_fold{exp_cfg.fold}_{model_info}{temp_suffix}_pred{exp_cfg.prediction_number}'
    if set(features_available) != set(exp_cfg.features):
        filename += f'_{"-".join(exp_cfg.features)}'
    filename += '.csv'
    return filename

def _get_predictions_dir(exp_cfg: ExperimentConfig,
                         pred_cfg: PredictorConfig) -> str:
    """Get directory for the predictions"""
    system_prompt_name = pred_cfg.system_prompt_path.replace('.j2', '')
    save_dir = f'{exp_cfg.base_dir}/results/predictions/{exp_cfg.data_name}/{system_prompt_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def _get_logs_dir(exp_cfg: ExperimentConfig,
                  pred_cfg: PredictorConfig) -> str:
    system_prompt_name = pred_cfg.system_prompt_path.replace('.j2', '')
    log_save_dir = f'{exp_cfg.log_dir}/predictions/{exp_cfg.data_name}/{system_prompt_name}'
    if not os.path.exists(log_save_dir):
        os.makedirs(log_save_dir)
    return log_save_dir

def _get_metrics_fname(exp_cfg: ExperimentConfig,
                       pred_cfg: PredictorConfig,
                       metric: str = 'cindex') -> str:
    """Get the filename to save metrics to"""
    system_prompt_name = pred_cfg.system_prompt_path.replace('.j2', '')
    save_dir = f'{exp_cfg.base_dir}/results/metrics/{metric}/{exp_cfg.data_name}/{system_prompt_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    temp_suffix = '' if _omit_temperature_suffix(pred_cfg.pred_engine) else f'_temp{pred_cfg.temperature}' 
    filename = f'{exp_cfg.exp_name}_{"_".join(exp_cfg.cancer_of_interest)}_fold{exp_cfg.fold}_{pred_cfg.pred_engine}{temp_suffix}_pred{exp_cfg.prediction_number}.json'
    return f'{save_dir}/{filename}'

def get_invalid_samples(original_samples, original_sample_ids, original_samples_stop, original_samples_dead, saved_df):
    """Filter samples to only predict for NaNs"""

    pred_cols =  [col for col in saved_df.columns if col.startswith('pred_')]

    # Filter out rows where all pred_cols are not NaN
    new_saved_df = saved_df.dropna(subset=pred_cols, how='all')
    # Identify rows where any pred_cols are NaN
    nan_ids = saved_df[saved_df[pred_cols].isna().any(axis=1)].sample_ids.tolist()

    new_samples, new_ids, new_stop, new_dead = [], [], [], []
    for i, sample_id in enumerate(original_sample_ids):
        if sample_id in nan_ids:
            new_samples.append(original_samples[i])
            new_ids.append(sample_id)
            new_dead.append(original_samples_dead[i])
            new_stop.append(original_samples_stop[i])

    return new_samples, new_ids, new_stop, new_dead, new_saved_df

def get_predictions_coverage(save_df: pd.DataFrame) -> float:
    """Calculate the coverage of predictions in the saved DataFrame."""
    if 'pred_num_days' in save_df.columns:
        pred_col = 'pred_num_days'
    elif 'pred_prob' in save_df.columns:
        pred_col = 'pred_prob'
    else:
        raise ValueError("No prediction columns found in the DataFrame.")
    
    coverage_perc = save_df[pred_col].notna().sum() / len(save_df['sample_ids']) * 100
    nan_perc = save_df[pred_col].isna().sum() / len(save_df['sample_ids']) * 100
    print(f"Coverage percentage: {coverage_perc:.2f}%, NaN predictions percentage: {nan_perc:.2f}%")
    return coverage_perc, nan_perc

def predict_survival(
        exp_cfg: ExperimentConfig,
        pred_cfg: PredictorConfig
) -> Tuple[List[Any], List[Any], List[Any], List[Any], Any]:
    
    if exp_cfg.features == ['all']:
        exp_cfg.features = features_available

    train_s, val_samples, s_ids, pred_ids_subgroups, s_stop, s_dead = get_tabular_data(exp_cfg)
    output_fname = _get_output_fname(exp_cfg, pred_cfg)
    print(f"Prediction output file: {output_fname}")

    save_df = None
    if not exp_cfg.overwrite_outputs:
        try:
            save_df = pd.read_csv(output_fname)
            logging.info(f"Output file already exists: {output_fname} and overwrite_outputs=False. Loading existing file.")
            if exp_cfg.complete_outputs:
                val_samples, s_ids, s_stop, s_dead, save_df = get_invalid_samples(val_samples, s_ids, s_stop, s_dead, save_df)
                if len(val_samples) > 0:
                    exp_cfg.overwrite_outputs = True
        except FileNotFoundError:
            print(f"Did not find outputfile, but overwrite_outputs not specified. Setting overwrite_outputs to True")
            exp_cfg.overwrite_outputs = True

    if exp_cfg.overwrite_outputs:
        predictor = OpenAIOutcomePredictor(cfg=pred_cfg)

        examples = None
        outcomes_examples = None

        results, formatted_prompts = predictor.predict_outcome(query_pt_dicts=val_samples,
                                            examples_pt_dicts=examples,
                                            examples_pt_outcomes=outcomes_examples,)
        
        
        if pred_cfg.prompting_task == 'TTE_OS':
            save_dict = {
                'sample_ids': s_ids,
                'pred_num_days': [x['num_days'] if x is not None else None for x in results],
                'response_text': [x['response_text'] if x is not None else None for x in results],
                'stop_nonlt': s_stop,
                'dead_nonlt': s_dead
            }
        else:
            save_dict = {
                'sample_ids': s_ids,
                'pred_time': [x['time'] if x is not None else None for x in results],
                'pred_prob': [x['prob'] if x is not None else None for x in results],
                'response_text': [x['response_text'] if x is not None else None for x in results],
                'stop_nonlt': s_stop,
                'dead_nonlt': s_dead
            }

        if save_df is None:
            save_df = pd.DataFrame(save_dict)
        else:
            new_df = pd.DataFrame(save_dict)
            save_df = pd.concat([save_df, new_df], axis=0)
        save_df.dead_nonlt = save_df.dead_nonlt.astype(bool)

        # Save the predictions
        save_df.to_csv(output_fname, index=False)
        logging.info(f"Saved predictions to: {output_fname}")

    # Save predictions to logs
    if exp_cfg.save_logs:
        logs_fname = _get_logs_fname(exp_cfg, pred_cfg)
        save_df.to_csv(logs_fname, index=False)
        logging.info(f"Saved logs to: {logs_fname}")

        prompts = dict(zip(s_ids, formatted_prompts))
        outputs = dict(zip(s_ids, results))
        
        # Save prompts and outputs as JSON files
        logs_dir = _get_logs_dir(exp_cfg, pred_cfg)
        prompts_file = f"{logs_dir}/prompts.json"
        outputs_file = f"{logs_dir}/outputs.json"
        
        with open(prompts_file, 'w') as f:
            json.dump(prompts, f, indent=2)
        logging.info(f"Saved prompts to: {prompts_file}")
        
        with open(outputs_file, 'w') as f:
            json.dump(outputs, f, indent=2)
        logging.info(f"Saved outputs to: {outputs_file}")

    # Get and save metrics
    if exp_cfg.save_metrics is not None:
        if 'c-index' in exp_cfg.save_metrics:
            cindex = calculate_c_index(save_df)
            # print sub-cohort c-indexes as well
            if len(pred_ids_subgroups) > 0:
                print(f"Predicting for subcohorts: {pred_ids_subgroups.keys()}")
                cindex_subcohorts = assess_on_subcohorts(save_df, pred_ids_subgroups)
            else:
                cindex_subcohorts = {}
            cindex_fp = _get_metrics_fname(exp_cfg, pred_cfg, metric='cindex')
            cindex_dict = {'all': cindex}
            cindex_dict.update(cindex_subcohorts)
            print(f"C-index: {cindex_dict}")
            with open(cindex_fp, 'w') as f:
                json.dump(cindex_dict, f)
            print(f"Saved c-index to: {cindex_fp}")
        if 'coverage' in exp_cfg.save_metrics:
            coverage_perc, nan_predictions_perc = get_predictions_coverage(save_df)
            coverage_fp = _get_metrics_fname(exp_cfg, pred_cfg, metric='coverage')
            coverage_dict = {'coverage': coverage_perc, 'nan_predictions': nan_predictions_perc}
            print(f"Coverage: {coverage_dict}")
            with open(coverage_fp, 'w') as f:
                json.dump(coverage_dict, f)
            print(f"Saved coverage to: {coverage_fp}")

if __name__=='__main__':
    args = parse_args()

    # Define configs
    exp_cfg = ExperimentConfig(
                        data_name = args.data_name,
                        cancer_of_interest = args.cancer_of_interest,
                        input_dir = args.input_dir,
                        base_dir=args.base_dir,
                        features = args.features,
                        num_folds = args.num_folds,
                        sample_size = args.sample_size,
                        fold = args.fold,
                        prediction_number = args.prediction_number,
                        exp_name = args.exp_name,
                        exp_type = args.exp_type,
                        complete_outputs = args.complete_outputs,
                        overwrite_outputs = args.overwrite_outputs,
                        save_metrics = args.save_metrics,
                    )
    pred_cfg = PredictorConfig(
                        pred_engine = args.pred_engine,
                        prompting_task = args.prompting_task,
                        temperature = args.temperature,
                        max_tokens = args.max_tokens,
                        reasoning_effort = args.reasoning_effort,
                        requests_per_minute = args.requests_per_minute,
                        max_retries = args.max_retries,
                        use_async = args.use_async,
                    )

    predict_survival(exp_cfg = exp_cfg,
                     pred_cfg = pred_cfg)
