import pandas as pd
import numpy as np
import os
import joblib
from copy import deepcopy

from sksurv.ensemble import RandomSurvivalForest
from lifelines import CoxPHFitter

from survprompt.data.utils import prepare_data

def get_estimated_survival_times(model, X_test, threshold=0.5):
    surv_fns = model.predict_survival_function(X_test)
    est_surv_times = []
    for sf in surv_fns:
        if isinstance(model, CoxPHFitter):
            times = surv_fns.index.values
            surv_probs = surv_fns[sf].values
        else:
            times = sf.x
            surv_probs = sf.y
        if surv_probs[-1] > threshold:
            est_surv_times.append(times[-1])
        else:
            est_surv_times.append(times[surv_probs <= threshold][0])
    return est_surv_times

def model_train_generic(model, df_train, df_tests, cois, endsuffix='', target_col=None):
    # Prepare feature subset
    cois = list(set(df_train.columns).intersection(set(cois)))
    if cois:
        print("Using features:", cois)
        X_train = df_train[cois].fillna(0).astype(int)
        if target_col is None:
            # Assume RSF: target is a tuple (dead, stop)
            y_train = df_train[['dead'+endsuffix, 'stop'+endsuffix]].apply(tuple, axis=1).tolist()
            y_train = np.array(y_train, dtype=[('Status','?'), ('Survival_in_days','<f8')])
        else:
            y_train = df_train[target_col]
        
        # Special handling for Cox model
        if isinstance(model, CoxPHFitter):
            X_train_cox = X_train.copy()
                
            # Remove low variance features
            low_variance_cols = X_train_cox.var()[X_train_cox.var() < 0.01].index.tolist()
            if low_variance_cols:
                X_train_cox.drop(columns=low_variance_cols, inplace=True)
                cois = list(set(cois) - set(low_variance_cols))
                print(f"Low variance features removed: {low_variance_cols}")
            
            if not cois:
                print("No features left after removing low variance features.")
                return None, [np.nan]*len(df_tests), [np.nan]*len(df_tests[0]), [np.nan]*len(df_tests[0])
            
            # Add event and duration
            X_train_cox = X_train_cox[cois]
            X_train_cox['duration'] = df_train['stop'+endsuffix]
            X_train_cox['event'] = df_train['dead'+endsuffix]

            # Remove invalid duration/event
            mask = (X_train_cox['duration'] > 0) & (~X_train_cox['duration'].isna()) & (~X_train_cox['event'].isna())
            if len(X_train_cox) - len(X_train_cox[mask]) > 0:
                print(f"Samples with invalid duration/event removed: {len(X_train_cox) - len(X_train_cox[mask])}")
            X_train_cox = X_train_cox[mask]

            model.fit(X_train_cox, duration_col='duration', event_col='event', robust=True)
        else:
            model.fit(X_train, y_train)
        
        scores = []
        estimated_surv = None
        estimated_risk = None
        for i, df_test in enumerate(df_tests):
            X_test = df_test[cois].fillna(0).astype(int)
            y_test_censored = df_test[['dead'+endsuffix, 'stop'+endsuffix]].apply(tuple, axis=1).tolist()
            y_test_censored = np.array(y_test_censored, dtype=[('Status','?'), ('Survival_in_days','<f8')])
            if target_col is None:
                y_test = y_test_censored
            else:
                y_test = df_test[target_col]

            if X_test.empty:
                # If empty test set, then skip
                estimated_risk = [np.nan]*len(y_test)
                score = np.nan
            else:
                try:
                    # RSF/Cox
                    estimated_surv = get_estimated_survival_times(model, X_test)
                except Exception:
                    estimated_surv = model.predict(X_test).tolist()
                if i == 0: # estimate survival time for first eval fold, which contains all patients
                    estimated_surv_tosave = estimated_surv

                try:
                    if isinstance(model, RandomSurvivalForest):
                        score = model.score(X_test, y_test)  # will get c-index by default
                        estimated_risk = model.predict(X_test).tolist()
                    elif isinstance(model, CoxPHFitter):
                        X_test_cox = X_test.copy()
                        X_test_cox['duration'] = df_test['stop'+endsuffix]
                        X_test_cox['event'] = df_test['dead'+endsuffix]
                        score = model.score(X_test_cox, scoring_method='concordance_index')
                        estimated_risk = model.predict_partial_hazard(X_test_cox).values.tolist()
                    else: # use default scoring function for model
                        print(f"Using default scoring function for {model}")
                        score = model.score(X_test, y_test)
                        estimated_risk = [np.nan]*len(y_test)
                except Exception:
                    score = np.nan
            scores.append(score)
        return model, scores, estimated_surv_tosave, estimated_risk
    return None, [np.nan]*len(df_tests), [np.nan]*len(df_tests), [np.nan]*len(df_tests)

def run_cross_validation(cancer2df_master_current_tx, ca_of_interest, variable_list, labels, num_folds, model_to_train, target_col=None,
                         deceased_only=False, subgroup_features_to_test=['STAGE_IV_DX', 'STAGE_I-III_NOPROG']):

    # Set subgroup_features_to_test = None if subgroup_features_to_test=['STAGE_IV_DX', 'STAGE_I-III_NOPROG'] not in variable_list
    if subgroup_features_to_test == ['STAGE_IV_DX', 'STAGE_I-III_NOPROG'] and not any(subgroup in variable_list for subgroup in subgroup_features_to_test):
        subgroup_features_to_test = None

    labels_col_name = "_+_".join(labels)
    models = {}
    estimated_surv_times = {}
    scores = {}

    formatted_data = prepare_data(cancer2df_master_current_tx, ca_of_interest, variable_list, num_folds, subgroup_features_to_test)

    for c in ca_of_interest:
        # dftemp = cancer2df_master_current_tx[c]
        # dftemp['stop_nonlt'] = dftemp['stop'] - dftemp['entry']
        # dftemp['dead_nonlt'] = dftemp['dead']
        # dftemp = dftemp[dftemp['stop_nonlt'] >= 0].sample(frac=1).reset_index()
        # kf = KFold(n_splits=num_folds)  # Define the split - into n_splits folds
        # kf.get_n_splits(dftemp)  # returns the number of splitting iterations in the cross-validator 
        models[c] = {}
        estimated_surv_times[c] = []
        scores[c] = {}
        for f in range(num_folds):
            print(f'{c} fold: {f}')
            train = formatted_data[c][f]['train']
            if deceased_only:
                train = train[train['dead_nonlt'] == 1]

            val_list = list(formatted_data[c][f]['val_dict'].values())

            print(f'Number of samples in train: {len(train)}')
            print(f'Number of samples in eval_sets: {[len(a) for a in val_list]}')
                
            
            fresh_train_model = deepcopy(model_to_train)
            trained_model, scorelist, estimated_survs, estimated_risk = model_train_generic(fresh_train_model, train, val_list, variable_list, '_nonlt', target_col=target_col)

            models[c][f] = trained_model
            survs = pd.DataFrame([formatted_data[c][f]['val_ids_dict']['val'].values, estimated_survs, estimated_risk]).T
            survs.columns = ['sample_ids', 'pred_num_days', 'risk_pred']
            estimated_surv_times[c].append(survs)

            # save scores in format: {fold 1: {labels_col_name: {"pred_num_days": scorelist[1]}, subgroup_features_to_test[0]: {"pred_num_days": scorelist[2]}, subgroup_features_to_test[1]: {"pred_num_days": scorelist[3]}}}
            scores[c][f] = {}
            scores[c][f][labels_col_name] = {'pred_num_days': scorelist[0]}
            if subgroup_features_to_test:
                scores[c][f][f'{subgroup_features_to_test[0]}_{labels_col_name}'] = {'pred_num_days': scorelist[1]}
                if len(subgroup_features_to_test) > 2:
                    scores[c][f][f'{subgroup_features_to_test[1]}_{labels_col_name}'] = {'pred_num_days': scorelist[2]}          

    return models, scores, estimated_surv_times

def save_model_outputs(cfg, scores):
    model_outputs_dir = f"{cfg.base_dir}/results/predictions/{cfg.data_name}"
    for c, model_outputs_by_cancer in scores.items():
        for f, model_outputs in enumerate(model_outputs_by_cancer):
            model_outputs_fp = f"{model_outputs_dir}/{cfg.exp_name}_{c}_fullfts_pred_fold{f}.csv"
            os.makedirs(os.path.dirname(model_outputs_fp), exist_ok=True)
            model_outputs.to_csv(model_outputs_fp, index=False)
    print(f"Saved model predictions to: {model_outputs_dir}")

def save_model_scores(cfg, scores, metric_name='cindex'):
    import json
    scores_dir = f"{cfg.base_dir}/results/metrics/{metric_name}/{cfg.data_name}"
    for c, scores_by_cancer in scores.items():
        for f, scores in scores_by_cancer.items():
            scores_fp = f'{scores_dir}/{cfg.exp_name}_{c}_fullfts_pred_fold{f}.json'
            os.makedirs(os.path.dirname(scores_fp), exist_ok=True)
            with open(scores_fp, 'w') as f:
                json.dump(scores, f)
    print(f"Saved c-index to: {scores_dir}")

def save_model_object(cfg, models):
    models_dir = f"{cfg.base_dir}/results/models/{cfg.data_name}"
    for c, models_by_cancer in models.items():
        for f, model in models_by_cancer.items():
            model_fp = f"{models_dir}/{cfg.exp_name}_{c}_fullfts_model_fold{f}.joblib"
            os.makedirs(os.path.dirname(model_fp), exist_ok=True)
            joblib.dump(model, model_fp)
    print(f"Saved models to: {models_dir}")
