"""
Generate 4-subplot figure with bar charts and bootstrap confidence intervals for
C-Index, integrated Brier score, mean absolute error (MAE), and censored MAE across
all cancers of interest for baseline models (Cox and Random Survival Forest).
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict

from survprompt.configs.exp_config import ExperimentConfig
from survprompt.plots.color_utils import GLOBAL_FONT_SIZE, get_model_color, parse_label
from survprompt.plots.plot_utils import process_data_for_km
from survprompt.plots.plot_all_metrics import compute_all_metrics_with_bootstrap
from survprompt.defaults import DEFAULT_DATA_NAME

# Set global font properties
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams["font.size"] = GLOBAL_FONT_SIZE

# Load environment and set BASE_DIR
BASE_DIR = os.getenv("BASE_DIR")
if BASE_DIR is None:
    load_dotenv()
    BASE_DIR = os.getenv("BASE_DIR")
if BASE_DIR is None:
    raise ValueError("BASE_DIR environment variable not set")

PLOT_DIR = os.path.join(BASE_DIR, "plots", "all_metrics_baselines")
os.makedirs(PLOT_DIR, exist_ok=True)

BOOTSTRAP_SAMPLES = 1000
CONFIDENCE_ALPHA = 0.05


def plot_metric_barplot(
    metric_stats: Dict[str, Dict[str, Dict[str, float]]],
    cancer_of_interest: List[str],
    model_names: List[str],
    metric_name: str,
    ylabel: str,
    ax,
    show_legend: bool = False
):
    """
    Bar chart with bootstrap confidence intervals for baseline models.
    Uses get_model_color from color_utils for consistent coloring.
    """
    cancer_labels = [ca.upper() for ca in cancer_of_interest]
    
    # Map display names to data labels and get colors
    color_map: Dict[str, str] = {}
    for model in model_names:
        if model == 'Cox Proportional Hazards':
            model_label = 'Baseline: Cox'
        elif model == 'Random Survival Forest':
            model_label = 'Baseline: RSF'
        else:
            model_label = model
        try:
            m, method, size, task = parse_label(model_label)
            color_map[model] = get_model_color(m, method, task)
        except Exception:
            color_map[model] = 'gray'
    
    x = np.arange(len(cancer_labels))
    width = 0.35

    for i, model in enumerate(model_names):
        values = []
        err_low = []
        err_high = []
        for ca in cancer_of_interest:
            stats = metric_stats.get(ca, {}).get(model, {})
            value = stats.get("value", np.nan)
            lower = stats.get("ci_lower", np.nan)
            upper = stats.get("ci_upper", np.nan)
            values.append(value)
            if np.isnan(value) or np.isnan(lower) or np.isnan(upper):
                err_low.append(0.0)
                err_high.append(0.0)
            else:
                err_low.append(max(0.0, value - lower))
                err_high.append(max(0.0, upper - value))

        offset = (i - len(model_names) / 2 + 0.5) * width
        bar = ax.bar(
            x + offset,
            values,
            width,
            label=model,
            color=color_map.get(model, 'gray'),
            alpha=0.85
        )
        yerr = np.array([err_low, err_high])
        ax.errorbar(
            x + offset,
            values,
            yerr=yerr,
            fmt='none',
            ecolor='black',
            elinewidth=1.2,
            capsize=4,
            capthick=1
        )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Cancer")
    ax.set_xticks(x)
    ax.set_xticklabels(cancer_labels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if metric_name == "C-Index":
        ax.set_ylim(0.5, 1.0)

    if show_legend:
        ax.legend(fontsize=8, loc='upper right')
    elif ax.get_legend():
        ax.get_legend().remove()


def main():
    """
    Main function to generate the 4-subplot figure with all metrics for baseline models.
    """
    # Data selection
    dataset = DEFAULT_DATA_NAME
    cancer_of_interest = ['nsclc', 'brca', 'crc', 'panc', 'prostate']
    prompting_tasks = []
    
    # Model names for plotting (baselines only)
    model_names = [
        'Cox Proportional Hazards',
        'Random Survival Forest'
    ]
    
    # Initialize metric dictionaries
    c_index_stats = {}
    brier_score_stats = {}
    mae_stats = {}
    cmae_stats = {}
    
    # Process data for each cancer type
    for ca in cancer_of_interest:
        print(f"Processing {ca}...")
        
        # Get datasets for this cancer type
        cfg = ExperimentConfig(
            base_dir=BASE_DIR,
            data_name=dataset,
            cancer_of_interest=ca,
        )
        ca_datasets = process_data_for_km(
            cfg, prompting_tasks,
            get_train_times=True, system_prompt_path='system'
        )
        # Filter for baseline models of interest
        selected_datasets = [
            x for x in ca_datasets if x[2] in [
                'Ground Truth',
                'Baseline: Cox',
                'Baseline: RSF',
                "Training Times/Events"
            ]
        ]
        
        # Calculate all metrics with unified bootstrap (much faster!)
        time_points = np.arange(0.5, 10.1, 0.5)
        
        c_index_mapped = {}
        brier_mapped = {}
        mae_summary = {}
        cmae_summary = {}
        
        # Process Cox model
        cox_metrics = compute_all_metrics_with_bootstrap(
            selected_datasets,
            'Baseline: Cox',
            time_points,
            n_boot=BOOTSTRAP_SAMPLES,
            alpha=CONFIDENCE_ALPHA
        )
        c_index_mapped['Cox Proportional Hazards'] = cox_metrics['c_index']
        brier_mapped['Cox Proportional Hazards'] = cox_metrics['brier_score']
        mae_summary['Cox Proportional Hazards'] = cox_metrics['mae']
        cmae_summary['Cox Proportional Hazards'] = cox_metrics['cmae']
        
        # Process RSF model
        rsf_metrics = compute_all_metrics_with_bootstrap(
            selected_datasets,
            'Baseline: RSF',
            time_points,
            n_boot=BOOTSTRAP_SAMPLES,
            alpha=CONFIDENCE_ALPHA
        )
        c_index_mapped['Random Survival Forest'] = rsf_metrics['c_index']
        brier_mapped['Random Survival Forest'] = rsf_metrics['brier_score']
        mae_summary['Random Survival Forest'] = rsf_metrics['mae']
        cmae_summary['Random Survival Forest'] = rsf_metrics['cmae']
        
        c_index_stats[ca] = c_index_mapped
        brier_score_stats[ca] = brier_mapped
        mae_stats[ca] = mae_summary
        cmae_stats[ca] = cmae_summary
        
        print(f"  C-Index: {c_index_mapped}")
        print(f"  Brier Score: {brier_mapped}")
        print(
            f"  MAE: "
            f"{[k + ': ' + str(mae_summary[k]['value']) for k in mae_summary]}"
        )
        print(
            f"  cMAE: "
            f"{[k + ': ' + str(cmae_summary[k]['value']) for k in cmae_summary]}"
        )
    
    # Create the 4-subplot figure
    # Subplot 1: C-Index bar plot (higher is better, measures discrimination ability)
    # Subplot 2: Integrated Brier Score bar plot (lower is better, measures calibration)
    # Subplot 3: Mean Absolute Error distribution (lower is better, measures prediction accuracy)
    # Subplot 4: Censored Mean Absolute Error distribution (lower is better, accounts for censoring)
    fig, axes = plt.subplots(4, 1, figsize=(5, 20))
    
    # Plot each metric
    plot_metric_barplot(
        c_index_stats, cancer_of_interest, model_names,
        "C-Index", "C-Index", axes[0], show_legend=True
    )

    plot_metric_barplot(
        brier_score_stats, cancer_of_interest, model_names,
        "Brier Score", "Integrated Brier Score", axes[1]
    )

    plot_metric_barplot(
        mae_stats, cancer_of_interest, model_names,
        "MAE", "Mean Absolute Error (years)", axes[2]
    )

    plot_metric_barplot(
        cmae_stats, cancer_of_interest, model_names,
        "cMAE", "Censored Mean Absolute Error (years)", axes[3]
    )
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "all_metrics_baselines_comparison.pdf"), 
                bbox_inches='tight', format="pdf")
    plt.show()
    print(f"Plot saved to {os.path.join(PLOT_DIR, 'all_metrics_baselines_comparison.pdf')}")


if __name__ == "__main__":
    main()
