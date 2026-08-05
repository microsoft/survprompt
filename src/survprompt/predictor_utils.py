"""Utilities for loading and formatting tabular prediction data."""
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from survprompt.configs.exp_config import ExperimentConfig


CATEGORICAL_FEATURES = [
                        ['WHITE','ASIAN','BLACK'], ['MALE'], ['SMOKER'], ['SEQUENCING_YEAR'], ['ANY_PRIOR_TX'],
                        ['STAGE 1','STAGE 2','STAGE 3','STAGE 4'], ['STAGE_IV_DX','STAGE_I-III_NOPROG','STAGE_I-III_PROG'], ['progressed'],
                        ['DMETS_DX_ADRENAL'], ['DMETS_DX_BONE'],['DMETS_DX_BRAIN'],['DMETS_DX_LIVER'],['DMETS_DX_LUNG'],['DMETS_DX_LYMPH'],['DMETS_DX_PLEURA'],['DMETS_DX_OTHER'],
                        ['HAS_Gleason'],['ADENOCARCINOMA'],['SQUAMOUS'],['PDL1'],['HAS_PDL1'],['HR'],['HER2'], ['RECTAL','ASCENDING','CECUM'],['NONADENOCARCINOMA'],['MUCINOUS'],['MSI_OR_dMMR'],['HAS_MSI_OR_dMMR'],
                        ['HAS_CA15-3'],['HAS_CEA'],['HAS_PSA'], ['HAS_CA19-9'],
                        ['KRAS'], ['HRAS'], ['RET'], ['MET'], ['GNAQ'], ['PTEN'], ['KIT'], ['EGFR'], ['FGFR1'], ['FGFR2'], ['FGFR3'], ['PDGFRA'], ['ERBB2'], ['TP53'], ['NRAS'], ['NOTCH1'], ['GNA11'], ['CTNNB1'], ['PIK3CA'], ['IDH1'], ['BRAF'], ['ALK'], ['AKT1'],
                        ['has_brca'], ['has_crc'], ['has_nsclc'], ['has_panc'], ['has_prostate'],
                        ]

CONTINUOUS_FEATURES = ['AGE', 'MAX_CA15-3', 'CA15-3', 'MAX_CEA', 'CEA', 'MAX_PSA', 'PSA', 'MAX_CA19-9', 'CA19-9', 'Gleason']

def df_to_list_dicts(df: pd.DataFrame) -> List[Dict[str, str]]:
    return df.to_dict(orient='records')

def get_tabular_data(exp_cfg: ExperimentConfig) -> Tuple[List[Dict[str, str]], List[str], List[str], List[str]]:
    """
    Get tabular data for the experiment. This function loads the data from the specified input directory, prepares it, and returns the training data,
    and sampled features, IDs, and labels. It also combines all cancer_of_interest into one DataFrame.
    Args
    ----
    exp_cfg: ExperimentConfig
        Experiment configuration object.
    
    Returns
    -------
    combined_train_df: pd.DataFrame
        Training data.
    sample_features: list
        Sampled features.
    sample_ids: list
        Sampled IDs.
    combined_prediction_ids_subgroups: Dict[str, List[str]]
        Prediction IDs for subgroups
    sample_labels_stop: list
        Sampled labels for stop.
    sample_labels_dead: list
        Sampled labels for dead.
    """
    from survprompt.data.utils import load_data, prepare_data, get_sel_fts_labels

    cancer2df_master_current_tx = load_data(exp_cfg)
    selected_features, _ = get_sel_fts_labels(exp_cfg.features, exclude_race=False)
    tabular_data = prepare_data(cancer2df_master_current_tx, exp_cfg.cancer_of_interest, selected_features, exp_cfg.num_folds, exp_cfg.subgroup_features_to_test)

    # Combine all cancer_of_interest into one DataFrame
    combined_train_df = None
    combined_prediction_df = None
    combined_prediction_ids_val = []
    combined_prediction_ids_subgroups = None
    for ca in exp_cfg.cancer_of_interest:
        try:
            data_to_use = tabular_data[ca][exp_cfg.fold]
            train_df = data_to_use['train']
            train_ids = data_to_use['train_ids']
            train_df['patient_id'] = train_ids
            train_df[f'has_{ca}'] = 1

            prediction_df = data_to_use['val_dict']['val']
            prediction_ids_dict = data_to_use['val_ids_dict']
            prediction_ids_val = prediction_ids_dict['val'].values
            prediction_df['patient_id'] = prediction_ids_val
            prediction_df[f'has_{ca}'] = 1

            prediction_ids_subgroups = {k: v for k, v in prediction_ids_dict.items() if k != 'val'}

            combined_prediction_ids_val += list(prediction_ids_val)
            if combined_prediction_ids_subgroups is None:
                combined_prediction_ids_subgroups = prediction_ids_subgroups
            else:
                for k, v in prediction_ids_subgroups.items():
                    if k in combined_prediction_ids_subgroups:
                        combined_prediction_ids_subgroups[k] = pd.concat([combined_prediction_ids_subgroups[k], v], join='outer')
                    else:
                        combined_prediction_ids_subgroups[k] = v

            if combined_train_df is None:
                combined_train_df = train_df
            else:
                combined_train_df = pd.concat([combined_train_df, train_df], join='outer')
            if combined_prediction_df is None:
                combined_prediction_df = prediction_df
            else:
                combined_prediction_df = pd.concat([combined_prediction_df, prediction_df], join='outer')
        except KeyError:
            raise ValueError(f"Selected cancer {ca} not found in data.")

    # Drop patient ids
    combined_train_df = combined_train_df.drop(columns=['patient_id'])
    combined_prediction_df = combined_prediction_df.drop(columns=['patient_id'])

    # Sample val data if sample size specified in config
    if exp_cfg.sample_size > 0:
        sample_indices = list(np.random.choice(range(len(combined_prediction_df)), exp_cfg.sample_size))
    else:
        sample_indices = list(range(len(combined_prediction_df)))

    sample_features = df_to_list_dicts(combined_prediction_df.iloc[sample_indices])
    sample_ids = [combined_prediction_ids_val[i] for i in sample_indices]
    sample_labels_stop = combined_prediction_df['stop_nonlt'].iloc[sample_indices].tolist()
    sample_labels_dead = combined_prediction_df['dead_nonlt'].iloc[sample_indices].tolist()
 
    return combined_train_df, sample_features, sample_ids, combined_prediction_ids_subgroups, sample_labels_stop, sample_labels_dead
