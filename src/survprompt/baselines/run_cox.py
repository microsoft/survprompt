import os
import argparse

from lifelines import CoxPHFitter

from survprompt.configs.exp_config import ExperimentConfig
from survprompt.baselines.utils import (
    run_cross_validation,
    save_model_outputs,
    save_model_scores,
    save_model_object
)
from survprompt.data.utils import (
    features_available,
    process_features,
    get_sel_fts_labels,
    load_data
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Cox PH model with options.')
    parser.add_argument('--features', nargs='+', choices=features_available+['all'], required=True, help='Feature sets to include')
    parser.add_argument('--exclude_features', nargs='+', default=[], help='Names of individual features to exclude')
    parser.add_argument('--cancer_of_interest', nargs='+', required=True, help='Cancer types to include')
    parser.add_argument('--base_dir', type=str, default=os.getenv("BASE_DIR"), help='Base directory')
    parser.add_argument('--input_dir', type=str, default=os.getenv("INPUT_DIR"), help='Directory containing input data')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--save_outputs', action='store_true', help='Save prediction outputs')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model')
    parser.add_argument('--output_fname', type=str, default='cox', help='Output filename for prediction results')

    args = parser.parse_args()
    args.features = process_features(args.features)
    return args

def run_cox(
        exp_cfg: ExperimentConfig,
        save_outputs: bool,
        save_model: bool
    ):
    cancer2df_master_current_tx = load_data(exp_cfg)

    selected_features, labels = get_sel_fts_labels(exp_cfg.features)
    cox = CoxPHFitter(penalizer=0.1)
    models, scores, estimated_surv_times = run_cross_validation(cancer2df_master_current_tx,
                                                                exp_cfg.cancer_of_interest,
                                                                selected_features,
                                                                labels,
                                                                exp_cfg.num_folds,
                                                                cox,
                                                                subgroup_features_to_test=exp_cfg.subgroup_features_to_test,)    
    
    if save_outputs:
        save_model_outputs(cfg=exp_cfg, scores=estimated_surv_times)
        save_model_scores(cfg=exp_cfg, scores=scores, metric_name='cindex')
    
    if save_model:
        save_model_object(cfg=exp_cfg, models=models)

if __name__ == '__main__':
    args = parse_args()
    exp_cfg = ExperimentConfig(
                    cancer_of_interest = args.cancer_of_interest,
                    input_dir = args.input_dir,
                    base_dir = args.base_dir,
                    features = args.features,
                    num_folds = args.num_folds,
                    exp_name = args.output_fname
                )
    run_cox(exp_cfg=exp_cfg,
            save_outputs=args.save_outputs,
            save_model=args.save_model)
