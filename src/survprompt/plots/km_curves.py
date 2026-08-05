import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from dotenv import load_dotenv

from survprompt.configs.exp_config import ExperimentConfig
from survprompt.plots.color_utils import GLOBAL_FONT_SIZE
from survprompt.defaults import DEFAULT_DATA_NAME
from survprompt.plots.plot_utils import (
    is_all_models_zero_shot_label,
    plot_km, process_data_for_km, resolve_survprompt_headline_label,
)

# Update global font properties using constant.
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["font.size"] = GLOBAL_FONT_SIZE

# Load environment and set BASE_DIR
load_dotenv()
BASE_DIR = os.getenv("BASE_DIR")
if BASE_DIR is None:
    raise ValueError("BASE_DIR environment variable not set")

PLOT_DIR = os.path.join(BASE_DIR, "plots", "survival_curves")
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_km_curves(
        dataset: str,
        cancers_of_interest: list,
        prompting_tasks: list,
        race_inclusion_path: str,
        system_prompt_path: str,
        filter_by_model: str = None
):
    # Create output directory for this specific configuration
    output_dir = os.path.join(PLOT_DIR, dataset, race_inclusion_path,system_prompt_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1 - Main figure with 0-shot results
    nrows = len(cancers_of_interest)
    ncols = 1
    fig, ax = plt.subplots(nrows, ncols, figsize=(8, 12), constrained_layout=True) 
    
    # Ensure ax is always an array
    if nrows == 1:
        ax = [ax]
    
    for i, cancer_of_interest in enumerate(cancers_of_interest):
        cfg = ExperimentConfig(
            base_dir=BASE_DIR,
            data_name=dataset,
            cancer_of_interest=cancer_of_interest,
        )
        datasets = process_data_for_km(cfg, prompting_tasks, race_inclusion_path, system_prompt_path)
        if filter_by_model:
            resolved_filter = filter_by_model
            if filter_by_model == "Survprompt":
                resolved_filter = resolve_survprompt_headline_label(x[2] for x in datasets)
            datasets_to_plot = [
                x for x in datasets
                if ('Ground Truth' in x[2] or 'Baseline: RSF' in x[2] or resolved_filter in x[2])
            ]
        else:
            datasets_to_plot = [
                x for x in datasets
                if x[2] == "Ground Truth" or x[2] == "Baseline: RSF" or is_all_models_zero_shot_label(x[2])
            ]

        if ncols > 1:
            ax_idx = (i // 2, i % 2)
            xaxis_label = "Years" if ax_idx[0] == 2 else ""
        else:
            ax_idx = (i, )
            xaxis_label = "Years" if ax_idx[0] == nrows - 1 else ""
        legend_pos = 'upper right' if i == 0 else None
        
        plot_km(ax=ax[ax_idx], datasets=datasets_to_plot, show_censors=False, include_metrics_in_legend=False,
                xaxis_label=xaxis_label, xtick_spacing=1.0,
                title=None, legend_pos=legend_pos,
                max_xval=None, 
                simplify_legend_labels='Survprompt_final' if filter_by_model else 'Survprompt_appendix',
                model_as_legend_label=True)
        
        # Position label to the left of the y-axis
        ax[ax_idx].text(-0.25, 0.5, f"{cancer_of_interest.upper()}", 
                       transform=ax[ax_idx].transAxes, 
                       rotation=0, verticalalignment='center', 
                       horizontalalignment='center', fontsize=18, fontweight='bold')
        
    for i in range(len(ax.flat)):
        if i >= len(cancers_of_interest):
            ax.flat[i].axis('off')

    fig.supylabel('Survival Probability', fontsize=GLOBAL_FONT_SIZE + 2)

    file_name = 'all_cancers_0_shot.pdf' if filter_by_model else f'all_cancers_0_shot_all_models_{".".join(prompting_tasks)}.pdf'
    output_path = os.path.join(output_dir, file_name)
    plt.savefig(output_path, bbox_inches='tight', format="pdf")
    print(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    dataset = DEFAULT_DATA_NAME
    cancers_of_interest = ['nsclc', 'brca', 'crc', 'panc', 'prostate']
    race_inclusion_path = 'incl_race'
    system_prompt_path = 'system'
    best_model = 'Survprompt'

    # Best model only
    plot_km_curves(dataset=dataset,
                   cancers_of_interest=cancers_of_interest,
                   prompting_tasks=['SURV_PROB'],
                   race_inclusion_path=race_inclusion_path,
                   system_prompt_path=system_prompt_path,
                   filter_by_model=best_model)

    # All zero-shot models
    plot_km_curves(dataset=dataset,
                   cancers_of_interest=cancers_of_interest,
                   prompting_tasks=['SURV_PROB'],
                   race_inclusion_path=race_inclusion_path,
                   system_prompt_path=system_prompt_path,
                   filter_by_model=None)
    
    plot_km_curves(dataset=dataset,
                   cancers_of_interest=cancers_of_interest,
                   prompting_tasks=['TTE_OS'],
                   race_inclusion_path=race_inclusion_path,
                   system_prompt_path=system_prompt_path,
                   filter_by_model=None)