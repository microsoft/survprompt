from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from survprompt.evaluation.metrics import calculate_optimism
from survprompt.configs.exp_config import ExperimentConfig
from survprompt.defaults import SURVPROMPT_HEADLINE_LABEL_CANDIDATES
from survprompt.plots.color_utils import parse_label, get_model_color, get_line_style, TITLE_FONT_SIZE, AXIS_LABEL_FONT_SIZE, TICK_LABEL_FONT_SIZE, LEGEND_FONT_SIZE, GLOBAL_FONT_SIZE
import matplotlib as mpl
import ast
import os, json, glob, re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Update global font properties using constant.
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams["font.size"] = GLOBAL_FONT_SIZE
mpl.rcParams['figure.dpi'] = 300


def save_source_data(pdf_path: str, df: pd.DataFrame) -> str:
    """Write a figure's underlying numbers as a tidy CSV.

    Saved next to the PDF under ``<pdf_dir>/source_data/<pdf_stem>.csv`` so each
    figure has a companion table with every plotted value (and its 95% CI).
    """
    pdf_dir = os.path.dirname(pdf_path)
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = os.path.join(pdf_dir, "source_data")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Source data: {csv_path}")
    return csv_path


def metric_stats_to_long(
    metric_to_stats: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    cancers: List[str],
    models: List[str],
    extra_cols: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Flatten ``{metric: {cancer: {model: {value, ci_lower, ci_upper}}}}`` into a
    tidy long-form DataFrame (one row per metric x cancer x model)."""
    rows = []
    for metric, stats in metric_to_stats.items():
        for cancer in cancers:
            for model in models:
                s = stats.get(cancer, {}).get(model, {})
                row = {
                    "cancer": cancer,
                    "model": model,
                    "metric": metric,
                    "value": s.get("value", np.nan),
                    "ci_lower": s.get("ci_lower", np.nan),
                    "ci_upper": s.get("ci_upper", np.nan),
                }
                if extra_cols:
                    row.update(extra_cols)
                rows.append(row)
    return pd.DataFrame(rows)

# Define model deployment and prompting types
model_deployment_dict = {
    'GPT 4o': 'gpt-4o_2024-08-06',
    'GPT 4o mini': 'gpt-4o-mini_2024-07-18',
    'GPT 4.1': 'gpt-4.1_2025-04-14',
    'GPT 4.1 mini': 'gpt-4.1-mini_2025-04-14',
    'GPT 5.5': 'gpt-5.5_2026-04-24',
    'GPT 5': 'gpt-5',
    'GPT 5.4': 'gpt-5.4-none',
    'GPT 5.4 medium': 'gpt-5.4-medium',
    'GPT 5.5 medium': 'gpt-5.5-medium',
    'GPT 5.6 Sol medium': 'gpt-5.6-sol_2026-07-09-medium',  # headline Survprompt default
    'GPT 5.6 Sol none': 'gpt-5.6-sol_2026-07-09-none',
    'o1': 'o1_2024-12-17',
    }
prompting_mode_dict = {
    '0/None': 'zeroshot',
}


def is_all_models_zero_shot_label(label: str, prompting_task: Optional[str] = None) -> bool:
    """Return True when a label belongs in the supplementary all-models figures."""
    if label in {"Ground Truth", "Training Times/Events"}:
        return False
    if label == "Baseline: RSF":
        return True
    if prompting_task and prompting_task not in label:
        return False
    if "0/None" not in label:
        return False
    return (
        "Temp = 0" in label
        or "o1" in label
        or "GPT 5" in label
    )


def is_survprompt_headline_label(label: str) -> bool:
    return label in SURVPROMPT_HEADLINE_LABEL_CANDIDATES


def resolve_survprompt_headline_label(labels) -> str:
    available = set(labels)
    for candidate in SURVPROMPT_HEADLINE_LABEL_CANDIDATES:
        if candidate in available:
            return candidate
    raise KeyError(
        "Could not find a headline Survprompt label. "
        f"Looked for {SURVPROMPT_HEADLINE_LABEL_CANDIDATES}, "
        f"available labels include: {sorted(available)}"
    )


def format_model_display_label(
    model: str,
    size: str,
    include_reasoning_effort: bool = False,
) -> str:
    """Format a model label for legends while optionally hiding reasoning effort."""
    suffix = ""
    if size == "mini":
        suffix = " mini"
    elif size == "none":
        suffix = " none"
    elif size == "medium" and include_reasoning_effort:
        suffix = " medium"
    return f"{model}{suffix}".strip()

def plot_coverage(
        coverage_df: pd.DataFrame, 
        cancer_of_interest: str = 'nsclc',
        prompting_mode: str = '0/None'
        ) -> None:
    '''
    Plots the coverage of predictions across different models, system prompts, and tasks.
    Args:
        coverage_df (pd.DataFrame): DataFrame containing columns 'system_prompt', 'task', 'model', 'coverage'.
    '''
    # Filter by prompting mode
    coverage_df = coverage_df[coverage_df['prompting_mode'] == prompting_mode]

    # Sort coverage_df by model name
    coverage_df = coverage_df.sort_values(by='model')

    # Plot coverage
    nrows=len(coverage_df['task'].unique())
    ncols=len(coverage_df['system_prompt'].unique())
    fig, axes = plt.subplots(figsize=(12, 8), sharey=True, sharex=True,
                             nrows=nrows, 
                             ncols=ncols)
    for row, (task, task_group) in enumerate(coverage_df.groupby('task')):
        if ncols == 1: 
            axes[row].set_ylabel(task, fontsize=12)
        else:
            axes[row, 0].set_ylabel(task, fontsize=12)
        for col, (system_prompt, system_group) in enumerate(task_group.groupby('system_prompt')):
            curr_plot = axes[row] if ncols == 1 else axes[row, col]
            # add a label for each column
            if row == 0: curr_plot.set_title(system_prompt, fontsize=12)
            # Unique colors for each model
            unique_models = system_group['model'].unique()
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_models)))
            for color, (model, model_group) in zip(colors, system_group.groupby('model')):
                curr_plot.bar(model, model_group['coverage'], color=color)
                curr_plot.tick_params(axis='x', rotation=90)

    fig.text(-0.02, 0.55, 'Coverage (%)', fontsize=20, va='center', rotation='vertical')
    plt.suptitle(f'{prompting_mode.capitalize()} Prediction Coverage by System Prompt, Task and Model ({cancer_of_interest})', fontsize=16)
    plt.tight_layout()

def plot_km(
    ax,
    datasets: List[Tuple[pd.Series, pd.Series, str]],  # list of tuples (T, E, label)
    ci_show: bool = True,
    ci_alpha: float = 0.2,
    show_censors: bool = False,
    xaxis_label: str = None,
    yaxis_label: str = None,
    title: str = None,
    max_xval: Optional[float] = None,
    legend_pos: str = 'upper right',
    include_metrics_in_legend: bool = True,
    xtick_spacing: float = 0.5,
    simplify_legend_labels=None,
    plot_max_obs_line=False,
    model_as_legend_label=False
    ) -> None:
    # Use first dataset as baseline.
    T_base, E_base, base_label = datasets[0]
    T_base = T_base.astype(int) / 365  # convert days to years
    E_base = E_base.astype(int)
    model, method, size, task = parse_label(base_label)
    base_color = get_model_color(model, method, task, simplify=simplify_legend_labels)
    base_ls = get_line_style(size)

    kmf = KaplanMeierFitter()
    kmf.fit(T_base, E_base, label=base_label)
    ax = kmf.plot(ax=ax, ci_show=ci_show, ci_alpha=ci_alpha,
                  show_censors=show_censors, color=base_color, linestyle=base_ls)

    # Plot remaining datasets.
    for (T_pred, E_pred, label) in datasets[1:]:
        model, method, size, task = parse_label(label)

        sample_prediction = None
        for value in T_pred.values:
            if isinstance(value, list):
                sample_prediction = value
                break
            if pd.notna(value):
                sample_prediction = value
                break

        if sample_prediction is None:
            print(f"Warning: No valid predictions found for {label}")
            continue

        if isinstance(sample_prediction, list): # Plot survival using predicted probabilities directly
            all_time_points = []
            all_survival_probs = []
            valid_pred_series = T_pred[T_pred.apply(lambda x: isinstance(x, list))]
            for pred_data in valid_pred_series.values:
                # Handle list format [time_points, probs]
                if isinstance(pred_data, list) and len(pred_data) == 2:
                    time_points, probs = pred_data[0], pred_data[1]
                    
                    # Ensure both time_points and probs are lists and have the same length
                    if isinstance(time_points, list) and isinstance(probs, list) and len(time_points) > 0 and len(time_points) == len(probs):
                        all_time_points.append(time_points)
                        all_survival_probs.append(probs)

            try:
                if len(all_time_points) == 0:
                    print(f"Warning: No valid time/probability pairs found for {label}")
                    continue
                    
                time_grid = np.linspace(0, max([max(times) for times in all_time_points]), len(T_base))
                interpolated_survivals = []
                for times, probs in zip(all_time_points, all_survival_probs):
                    interpolated_survivals.append(np.interp(time_grid, times, probs))

                mean_predicted_survival = np.mean(interpolated_survivals, axis=0)
                std_predicted_survival = np.std(interpolated_survivals, axis=0, ddof=1)
                n_patients = len(interpolated_survivals)
                se_predicted_survival = std_predicted_survival / np.sqrt(n_patients)
                ci_upper = mean_predicted_survival + 1.96 * se_predicted_survival
                ci_lower = mean_predicted_survival - 1.96 * se_predicted_survival

                if include_metrics_in_legend:
                    # Add optimism to label
                    opt_dict = calculate_optimism(df=pd.DataFrame({'pred_prob': T_pred,
                                                                   'pred_event': E_pred,
                                                                   'stop_nonlt': T_base * 365,
                                                                   'dead_nonlt': E_base}))
                    updated_label = f"{label} (opt: {opt_dict['auc_opt']:.3f}, abs_opt: {opt_dict['auc_abs_opt']:.3f}, sight: {opt_dict['sightedness']:.3f})"
                else:
                    updated_label = label
                    
                # Handle RSF special cases for survival curves
                if 'Baseline: RSF (Avg)' in updated_label or 'Baseline: RSF (Survival Curve)' in updated_label:
                    updated_label = "RSF (Avg)"
                    color = 'red'
                elif simplify_legend_labels == 'Survprompt_final':
                    updated_label = "Survprompt"
                    color = get_model_color(model, method, task, simplify=simplify_legend_labels)
                elif model_as_legend_label:
                    updated_label = format_model_display_label(model, size)
                    color = get_model_color(model, method, task, simplify=simplify_legend_labels)
                else:
                    color = get_model_color(model, method, task, simplify=simplify_legend_labels)
                    
                ls = get_line_style(size)
                
                # plot
                ax.plot(time_grid, mean_predicted_survival, label=updated_label, linestyle=ls, color=color)
                ax.fill_between(time_grid, ci_lower, ci_upper, color=color, alpha=0.2)
            except Exception as e:
                print(f"Error in plotting probabilities for {label}: {e}")
                
        elif isinstance(sample_prediction, (int, float, np.number)): # Plot survival using KM on predicted TTE
            if include_metrics_in_legend:
                # Add optimism to label
                opt_dict = calculate_optimism(df=pd.DataFrame({'pred_num_days': T_pred,
                                                               'pred_event': E_pred,
                                                               'stop_nonlt': T_base * 365,
                                                               'dead_nonlt': E_base}))
                updated_label = f"{label} (opt: {opt_dict['auc_opt']:.3f}, abs_opt: {opt_dict['auc_abs_opt']:.3f}, sight: {opt_dict['sightedness']:.3f})"
            else:
                updated_label = label
            
            if simplify_legend_labels and 'RSF' in updated_label:
                updated_label = "Random Survival Forest"
                color = get_model_color(model, method, task, simplify=simplify_legend_labels)
            elif 'Baseline: RSF' in updated_label and 'Survival Curve' not in updated_label and 'Avg' not in updated_label:
                updated_label = "RSF (TTE)"
                color = 'blue'
            elif 'Baseline: RSF (Avg)' in updated_label or 'Baseline: RSF (Survival Curve)' in updated_label:
                updated_label = "RSF (Avg)"
                color = 'red'
            elif model_as_legend_label and 'Random Survival Forest' != updated_label and 'RSF' not in updated_label:
                updated_label = format_model_display_label(model, size)
                color = get_model_color(model, method, task, simplify=simplify_legend_labels)
            else:
                color = get_model_color(model, method, task, simplify=simplify_legend_labels)
                
            ls = get_line_style(size)
            
            # plot
            kmf = KaplanMeierFitter()
            # Drop patients a model didn't cover (NaN predicted time) before the
            # int cast, which cannot represent NaN. Some aligned series can also
            # contain a leading NaN ahead of list-valued predictions.
            valid = T_pred.apply(lambda x: isinstance(x, (int, float, np.number)) and pd.notna(x))
            T_pred_year = T_pred[valid].astype(int) / 365
            E_pred = E_pred[valid].astype(int)
            kmf.fit(T_pred_year, E_pred, label=updated_label)
            ax = kmf.plot(ax=ax, ci_show=ci_show, ci_alpha=ci_alpha,
                  show_censors=show_censors, color=color, linestyle=ls)

    # Plot vertical line for max observed time in baseline
    if plot_max_obs_line:
        ax.axvline(T_base[E_base == 1].max(), color="r", linestyle="--")

    # After plotting
    right_lim = max_xval if max_xval is not None else T_base.max()
    ax.set_xlim(left=0, right=right_lim)
    xticks = np.arange(0, right_lim+xtick_spacing, xtick_spacing)
    ax.set_xticks(xticks)
    
    # Set y-axis ticks at every 0.25, but only label 0, 0.5, and 1
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.25, 0.25))
    ax.set_yticklabels(['0', '', '0.5', '', '1'])
    
    if title:
        ax.set_title(title, fontweight='bold', fontsize=TITLE_FONT_SIZE)
    if xaxis_label:
        ax.set_xlabel(xaxis_label, fontweight='normal', fontsize=AXIS_LABEL_FONT_SIZE, horizontalalignment="center")
    else:
        ax.get_xaxis().set_visible(False)
    if yaxis_label:
        ax.set_ylabel(yaxis_label, fontweight='normal', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis='x', labelrotation=0, labelsize=TICK_LABEL_FONT_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend_pos is not None:
        ax.legend(loc=legend_pos, fontsize=LEGEND_FONT_SIZE, ncol=1 if simplify_legend_labels == 'Survprompt_final' else 3)
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

def generate_score(metric: np.ndarray) -> tuple:
    percentile_val = 1.96
    score = (np.mean(metric), percentile_val * np.std(metric) / np.sqrt(len(metric)))

    return round(score[0], 4), round(score[1], 4)


def generate_score_str(metric: np.ndarray) -> str:
    mean, std = generate_score(metric)
    return str(mean) + " +/- " + str(std)

def process_prediction_file(file_path: str) -> Tuple[Dict[str, Tuple[int, bool]], Dict[str, int]]:
    """
    Read prediction CSV file and output two dictionaries:
      1. Mapping sample_id to a tuple (stop_nonlt, dead_nonlt)
      2. Mapping sample_id to predicted columns

    If stop_nonlt and dead_nonlt are not present, will output None for first tuple element.
    Args:
        file_path (str): Path to the prediction CSV file.

    Returns:
        Tuple[Dict[str, Tuple[int, bool]], Dict[str, int]]:
            - dict1: {sample_id: (stop_nonlt, dead_nonlt)}
            - dict2: {sample_id: [predicted columns]]}
    """
    df = pd.read_csv(
        file_path,
        converters={
            "stop_nonlt": int,
            "dead_nonlt": lambda x: x.strip().lower() == "true"
        }
    )
    if "pred_num_days" in df.columns:
        df["pred_num_days"] = df["pred_num_days"].fillna(-1).astype(float).astype(int)
    if "pred_time" in df.columns:
        df["pred_time"] = df["pred_time"].fillna("[]").apply(ast.literal_eval)
    if "pred_prob" in df.columns:
        df["pred_prob"] = df["pred_prob"].fillna("[]").apply(ast.literal_eval)

    if "dead_nonlt" not in df.columns:
        ground_truth = None
    else:
        ground_truth = {
            row["sample_ids"]: (row["stop_nonlt"], row["dead_nonlt"])
            for _, row in df.iterrows()
        }
    
    prediction = {
        row["sample_ids"]: row["pred_num_days"] if "pred_num_days" in df.columns else [row["pred_time"], row["pred_prob"]]
        for _, row in df.iterrows()
    }
    return ground_truth, prediction

def process_data_for_km(
        cfg: ExperimentConfig,
        prompting_tasks: List[str],
        race_inclusion_path: str = 'incl_race',
        system_prompt_path: str = 'system',
        get_train_times: bool = False,
        load_multiple_runs: bool = False) -> List[Tuple[pd.Series, pd.Series, str]]:
    """
    Processes data for Kaplan-Meier surves by aligning predictions from various models with ground truth.
    Args:
        cfg (ExperimentConfig): Configuration object containing experiment settings.
        prompting_tasks (List[str]): List of prompting tasks being performed.
        race_inclusion_path (str): The path indicating whether race was included or not.
        system_prompt_path (str): The path to the system prompt used.
        get_train_times (bool): If True, returns training observed times and censoring events (for cMAE).
        load_multiple_runs (bool): If True, loads all prediction files (pred0, pred1, pred2, ...) for self-consistency.
    Returns:
        list: A list of tuples, each containing:
            - pd.Series: Time to event or prediction values.
            - pd.Series: Event occurrence (1 for event, 0 for no event).
            - str: Label indicating the source of the data (e.g., model name or "Ground Truth").
    """
    cancer_of_interest = cfg.cancer_of_interest[0] if isinstance(cfg.cancer_of_interest, list) else cfg.cancer_of_interest
    
    # Build baseline model paths (always single run with fold0)
    baseline_paths = {
        'Baseline: RSF': f'rsf_baseline_{cancer_of_interest}_fullfts_pred_fold0',
        'Baseline: Cox': f'cox_baseline_{cancer_of_interest}_fullfts_pred_fold0',
    }
    
    # Build LLM model paths (can have multiple runs)
    llm_model_paths = {}
    for task in prompting_tasks:
        for model_name, deployment in model_deployment_dict.items():
            for prompting_mode_name, prompting_mode in prompting_mode_dict.items():
                if prompting_mode_name == '0/None' and ('GPT 4o' in model_name or 'GPT 4.1' in model_name):
                    # Add temp 0 case (base path without pred suffix)
                    label = f"{task}: {model_name} {prompting_mode_name} Temp = 0"
                    llm_model_paths[label] = f'{prompting_mode}_{task}_{cancer_of_interest}_fold0_{deployment}_temp0.0_pred'
                label = f"{task}: {model_name} {prompting_mode_name}"
                # GPT-5-family result files omit the temperature suffix.
                temp_suffix = '' if deployment.startswith('gpt-5') else '_temp0.4'
                llm_model_paths[label] = f'{prompting_mode}_{task}_{cancer_of_interest}_fold0_{deployment}{temp_suffix}_pred'
    
    ground_truth_df = None
    predictions = {}
    predicted_ids = set()

    def _merge_ground_truth(gt: dict | None) -> None:
        nonlocal ground_truth_df
        if gt is None:
            return
        gt_df = pd.DataFrame(gt).T
        gt_df.columns = ['time', 'event']
        gt_df.index.name = "sample_id"
        ground_truth_df = gt_df if ground_truth_df is None else ground_truth_df.combine_first(gt_df)
    
    # Process baseline models (single run only)
    for label, file_name in baseline_paths.items():
        try:
            file_path = f'{cfg.base_dir}/results/predictions/{cfg.data_name}/{race_inclusion_path}/{file_name}.csv'
            gt, p = process_prediction_file(file_path)

            _merge_ground_truth(gt)
            p_series = pd.Series(p, name=label)
            p_series.index.name = "sample_id"
            predictions[label] = p_series
            predicted_ids.update(p_series.index)
            print(f"File found for model: {label}")
        except FileNotFoundError:
            continue
    
    # Process LLM models (single or multiple runs)
    for label, file_name_base in llm_model_paths.items():
        if load_multiple_runs:
            # Discover every run file (pred0, pred1, ...). Globbing rather than
            # counting up from 0 handles non-contiguous run indices (e.g. a model
            # with pred0/pred2/pred3 but no pred1).
            pred_dir = f'{cfg.base_dir}/results/predictions/{cfg.data_name}/{race_inclusion_path}/{system_prompt_path}'
            run_files = glob.glob(f'{pred_dir}/{file_name_base}*.csv')
            pred_nums = []
            for path in run_files:
                match = re.search(rf'{re.escape(file_name_base)}(\d+)\.csv$', os.path.basename(path))
                if match:
                    pred_nums.append(int(match.group(1)))
            for pred_num in sorted(pred_nums):
                file_name = f"{file_name_base}{pred_num}"
                file_path = f'{pred_dir}/{file_name}.csv'

                try:
                    gt, p = process_prediction_file(file_path)

                    _merge_ground_truth(gt)
                    label_with_run = f"{label} pred{pred_num}"
                    p_series = pd.Series(p, name=label_with_run)
                    p_series.index.name = "sample_id"
                    predictions[label_with_run] = p_series
                    predicted_ids.update(p_series.index)
                    print(f"File found for model: {label_with_run}")
                except FileNotFoundError:
                    continue
        else:
            # Load only pred0
            file_name = f"{file_name_base}0"
            file_path = f'{cfg.base_dir}/results/predictions/{cfg.data_name}/{race_inclusion_path}/{system_prompt_path}/{file_name}.csv'
            
            try:
                gt, p = process_prediction_file(file_path)

                _merge_ground_truth(gt)
                p_series = pd.Series(p, name=label)
                p_series.index.name = "sample_id"
                predictions[label] = p_series
                predicted_ids.update(p_series.index)
                print(f"File found for model: {label}")
            except FileNotFoundError:
                continue

    if not predictions:
        raise FileNotFoundError(
            f"No prediction files found for {cancer_of_interest} "
            f"({cfg.data_name}, tasks={prompting_tasks})."
        )
    
    if ground_truth_df is None: # ground truth was not found in previous files
        file_path = cfg.input_data_paths[cancer_of_interest]
        if cfg.data_name == 'mskchord':
            times_df = pd.read_csv(file_path, index_col='PATIENT_ID')
            times_df['stop_nonlt'] = times_df['stop'] - times_df['entry']
        times_df['dead_nonlt'] = times_df['dead']
        times_df = times_df[times_df['stop_nonlt'] >= 0].reset_index()
        ground_truth_df = times_df.loc[times_df['PATIENT_ID'].isin(predicted_ids), ['PATIENT_ID','stop_nonlt', 'dead_nonlt']]
        ground_truth_df = ground_truth_df.set_index('PATIENT_ID')
        ground_truth_df.columns = ['time', 'event']
        ground_truth_df.index.name = "sample_id"
    
    if get_train_times:
        file_path = cfg.input_data_paths[cancer_of_interest]

        try:
            if cfg.data_name == 'mskchord':
                train_times_df = pd.read_csv(file_path, index_col='PATIENT_ID')
                train_times_df['stop_nonlt'] = train_times_df['stop'] - train_times_df['entry']
            train_times_df['dead_nonlt'] = train_times_df['dead']
            train_times_df = train_times_df[train_times_df['stop_nonlt'] >= 0].reset_index()
            if ground_truth_df is not None:
                train_times_df = train_times_df[~train_times_df['PATIENT_ID'].isin(predicted_ids)]
                train_times_df = train_times_df[['PATIENT_ID', 'stop_nonlt', 'dead_nonlt']]    
            
        except FileNotFoundError:
            print(f"Training times file not found for {cancer_of_interest}.")
            train_times, train_events = None, None
    
    # Build datasets list with aligned predictions and ground truth.
    datasets = [(ground_truth_df['time'], ground_truth_df['event'], "Ground Truth")]
    if get_train_times:
        datasets += [
            (train_times_df['stop_nonlt'], train_times_df['dead_nonlt'], "Training Times/Events")
        ]

    for label, p_series in predictions.items():
        # Align predictions with ground truth sample_ids.
        p_series_aligned = p_series.reindex(ground_truth_df.index)
        # Create an event series of ones (as in your example) or modify if actual events are available.
        event_series = pd.Series([1] * len(p_series_aligned), index=p_series_aligned.index)
        datasets.append((p_series_aligned, event_series, label))
    return datasets

def get_all_predictions_coverage(
        base_dir,
        dataset,
        cancer_of_interest,
        prompting_tasks, 
        system_prompt_paths
) -> Dict[str, Tuple[float, float]]:
    """
    Extract the coverage of predictions across given cancers, prompting tasks and system prompts.
    """
    coverage_list = []
    for system_prompt in system_prompt_paths:
        for task in prompting_tasks:
            coverage_dir = f"{base_dir}/results/metrics/coverage/{dataset}/{system_prompt}/"
            coverage_files = [f for f in os.listdir(coverage_dir) if task in f and cancer_of_interest in f]
            for coverage_file_name in coverage_files:
                coverage_by_model_dict = {'system_prompt': system_prompt, 'task': task, 'cancer_of_interest': cancer_of_interest}

                # Extract prompting mode and model name from the coverage file name
                prompting_mode = next((mode for name, mode in prompting_mode_dict.items() if mode in coverage_file_name), None)
                coverage_by_model_dict['prompting_mode'] = prompting_mode if prompting_mode else 'Unknown Prompting Mode'
                model_name = next((name for name, deployment in model_deployment_dict.items() if deployment in coverage_file_name), None)
                coverage_by_model_dict['model'] = model_name if model_name else 'Unknown Model'
                
                # Extract coverage percentage from the file
                with open(os.path.join(coverage_dir, coverage_file_name), 'r') as f:
                    coverage_by_model_dict.update(json.load(f))

                coverage_list.append(coverage_by_model_dict)
    return pd.DataFrame(coverage_list)
