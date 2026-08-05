import pandas as pd
from sklearn.model_selection import KFold

from survprompt.configs.exp_config import ExperimentConfig

FEATURE_TYPE_TO_COLS = {
    'demographics': ['AGE','MALE','WHITE','ASIAN','BLACK','SMOKER'],
    'entry': ['entry'], # Note: we exclude "entry" from the list of training features
    'sequencing_year': ['SEQUENCING_YEAR'], # Note: we exclude "SEQUENCING_YEAR" from the list of training features observing minimal impact: c-idx .757688 with, vs .757746 without
    'treatment': set(['ANY_PRIOR_TX']),
    'stage': ['STAGE 1','STAGE 2','STAGE 3','STAGE 4','STAGE_IV_DX','STAGE_I-III_NOPROG','STAGE_I-III_PROG','progressed'], 
    'met': ['DMETS_DX_ADRENAL','DMETS_DX_BONE','DMETS_DX_BRAIN','DMETS_DX_LIVER','DMETS_DX_LUNG','DMETS_DX_LYMPH','DMETS_DX_PLEURA','DMETS_DX_OTHER'],
    'path': ['Gleason','HAS_Gleason','ADENOCARCINOMA','SQUAMOUS','PDL1','HAS_PDL1','HR','HER2', 'RECTAL','ASCENDING','CECUM','NONADENOCARCINOMA','MUCINOUS','MSI_OR_dMMR','HAS_MSI_OR_dMMR'],
    'lab': ['MAX_CA15-3','CA15-3','HAS_CA15-3', 'MAX_CEA','CEA','HAS_CEA', 'MAX_PSA','PSA','HAS_PSA', 'MAX_CA19-9','CA19-9','HAS_CA19-9'],          
    'genomics': ['KRAS', 'HRAS', 'RET', 'MET', 'GNAQ', 'PTEN', 'KIT', 'EGFR', 'FGFR1', 'FGFR2', 'FGFR3', 'PDGFRA', 'ERBB2', 'TP53', 'NRAS', 'NOTCH1', 'GNA11', 'CTNNB1', 'PIK3CA', 'IDH1', 'BRAF', 'ALK', 'AKT1']
}

COL_LABELS = {
    'demographics': 'Demographics',
    'treatment': 'Treatment',
    'stage': 'Stage/Progression',
    'met': 'Other Met Sites',
    'path': 'Pathology',
    'lab': 'Tumor Markers',
    'genomics': 'Genomics'
}

features_available = ['demographics', 'treatment', 'stage', 'met', 'path', 'lab', 'genomics']

def process_features(features: list[str] | str) -> list[str]:
    """Process the 'features' argument to expand 'all'."""
    if features == "all":
        return features_available
    if isinstance(features, list) and "all" in features:
        return features_available
    return features

def get_sel_fts_labels(features, exclude_race=False):
    """
    Get selected features and labels for the given feature types.
    
    Args:
        features: List of feature type names
        exclude_race: If True, exclude race columns (WHITE, ASIAN, BLACK) from demographics
    
    Returns:
        Tuple of (selected_features, labels)
    """
    selected_features = []
    labels = []
    race_columns = ['WHITE', 'ASIAN', 'BLACK'] if exclude_race else []
    
    for feature_type in features:
        for feature in FEATURE_TYPE_TO_COLS[feature_type]:
            # Skip race columns if exclude_race is True
            if feature not in race_columns:
                selected_features.append(feature)
        labels.append(COL_LABELS[feature_type])
    
    if exclude_race and any(col in race_columns for feature_type in features for col in FEATURE_TYPE_TO_COLS[feature_type]):
        excluded_cols = [col for feature_type in features for col in FEATURE_TYPE_TO_COLS[feature_type] if col in race_columns]
        print(f"Excluding race columns from features: {excluded_cols}")
    
    return selected_features, labels

def load_data(cfg: ExperimentConfig):
    cancer2df_master_current_tx = {}
    for c in cfg.input_data_paths.keys():
        print(f"Loading data for {c} from {cfg.input_data_paths[c]}")

        if cfg.data_name == 'mskchord':
            cancer2df_master_current_tx[c] = pd.read_csv(cfg.input_data_paths[c])

        txhxcols_c = cancer2df_master_current_tx[c].columns[
            cancer2df_master_current_tx[c].columns.str.contains('ANY_CURRENT_|ANY_PREV_', regex=True)
        ]
        txhxcols = FEATURE_TYPE_TO_COLS['treatment'].union(set(txhxcols_c))
    return cancer2df_master_current_tx

def prepare_data(cancer2df_master_current_tx, ca_of_interest, selected_features, num_folds, subgroup_features_to_test=None):
    prepared_data = {}
    for ca in ca_of_interest:
        prepared_data[ca] = {}
        dftemp = cancer2df_master_current_tx[ca]
        dftemp['stop_nonlt'] = dftemp['stop'] - dftemp['entry']
        dftemp['dead_nonlt'] = dftemp['dead']
        dftemp = dftemp[dftemp['stop_nonlt'] >= 0].reset_index()

        kf = KFold(n_splits=num_folds,
                   shuffle=True,
                   random_state=20) #using same random state as original code (20), though og code doesn't shuffle/use random state here
        kf.get_n_splits(dftemp)
        for fold, (train_index, test_index) in enumerate(kf.split(dftemp)):
            features_of_interest = list(set(dftemp.columns).intersection(set(selected_features)))
            features_of_interest += ['stop_nonlt', 'dead_nonlt'] # label columns
            train = dftemp.loc[train_index, features_of_interest]
            val = dftemp.loc[test_index, features_of_interest]

            val_dict = {'val': val}
            val_ids_dict = {'val': dftemp.loc[test_index, 'PATIENT_ID']}

            # Add the subgroup features to the validation set
            if subgroup_features_to_test is not None:
                subgroup_features = [subgroup for subgroup in subgroup_features_to_test if subgroup in val.columns]
                for subgroup in subgroup_features:
                    val_dict[subgroup] = val[val[subgroup].astype(bool)]
                    val_ids_dict[subgroup] = dftemp.loc[test_index, 'PATIENT_ID'][val[subgroup].astype(bool)]

            prepared_data[ca][fold] = {
                'train': train,
                'train_ids': dftemp.loc[train_index, 'PATIENT_ID'],
                'val_dict': val_dict,
                'val_ids_dict': val_ids_dict
            }
    return prepared_data
