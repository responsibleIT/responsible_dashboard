import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer 
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from scipy.stats import chi2_contingency, f_oneway
from werkzeug.utils import secure_filename
import os
import logging
import uuid
import time
import gc
import shutil
import json
import math
import pickle
import sklearn
import psutil

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key-onboarding-v2-' + str(uuid.uuid4()))
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Constants ---
# For budgetting "parameters" change the N_Permutation_repeats, jobs, min samples or importance size below.
DEFAULT_AGE_BINS = [0, 24, 34, 44, 54, 64, np.inf]
DEFAULT_AGE_LABELS = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']
TEST_SIZE = 0.20
PERMUTATION_IMPORTANCE_SIZE = 0.25
RANDOM_STATE = 42
N_PERMUTATION_REPEATS = 5
PI_N_JOBS = 1
MIN_SAMPLES_FOR_PI = 15
METRICS_FOR_OUTLIERS = ['Demographic Parity', 'Equalized Odds', 'Predictive Parity']
ALL_METRIC_NAMES = ['Demographic Parity', 'Equalized Odds', 'FPR', 'Predictive Parity', 'Group Size']
ALLOWED_DATASET_EXTENSIONS = {'csv'}
ALLOWED_MODEL_EXTENSIONS = {'pkl', 'joblib'} 
ALLOWED_METADATA_EXTENSIONS = {'json'} 


# --- Helper Functions ---
def allowed_file(filename, allowed_extensions=ALLOWED_DATASET_EXTENSIONS):
    """Checks if a filename has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def calculate_fairness_metrics(df_group):
    """Calculates fairness metrics for a given subgroup DataFrame."""
    if df_group.empty:
        return {'Demographic Parity': np.nan, 'Equalized Odds': np.nan, 'FPR': np.nan, 'Predictive Parity': np.nan, 'Group Size': 0}
    TP = df_group[(df_group['y_true'] == 1) & (df_group['y_pred'] == 1)].shape[0]
    FP = df_group[(df_group['y_true'] == 0) & (df_group['y_pred'] == 1)].shape[0]
    TN = df_group[(df_group['y_true'] == 0) & (df_group['y_pred'] == 0)].shape[0]
    FN = df_group[(df_group['y_true'] == 1) & (df_group['y_pred'] == 0)].shape[0]
    predicted_positive = TP + FP; actual_positive = TP + FN; actual_negative = TN + FP; total_group = TP + FP + TN + FN
    dp_rate = predicted_positive / total_group if total_group > 0 else np.nan
    tpr_rate = TP / actual_positive if actual_positive > 0 else np.nan
    fpr_rate = FP / actual_negative if actual_negative > 0 else np.nan
    pp_rate = TP / predicted_positive if predicted_positive > 0 else np.nan
    return {'Demographic Parity': dp_rate, 'Equalized Odds': tpr_rate, 'FPR': fpr_rate, 'Predictive Parity': pp_rate, 'Group Size': total_group}

def parse_budgets(form_data, protected_cols):
    """Parses budget settings from the form data."""
    budgets = {}; enabled_attributes = set(); budget_overall = False
    for attribute in protected_cols:
        if form_data.get(f"budget_enable_{attribute}") == 'yes':
            enabled_attributes.add(attribute); budget_overall = True; attr_budgets = {}
            prefix = f"budget__{attribute}__";
            for key, value in form_data.items():
                if key.startswith(prefix):
                    group_name = key[len(prefix):]
                    if value and value.strip() and value.strip().lower() != 'none':
                        try: limit = int(value); attr_budgets[group_name] = limit if limit >= 0 else None
                        except (ValueError, TypeError): attr_budgets[group_name] = None
                    else: attr_budgets[group_name] = None
            if attr_budgets: budgets[attribute] = attr_budgets
    return budgets, enabled_attributes, budget_overall

def apply_age_binning(df, p_cols):
    """Applies binning to an 'age' column if present and numeric."""
    df_binned = df.copy(); analysis_cols = list(p_cols); binned_col_name = None
    age_col='age'
    analysis_map = {p: p for p in p_cols}

    if age_col in p_cols and age_col in df_binned.columns and pd.api.types.is_numeric_dtype(df_binned[age_col]):
        binned_col_name=f'{age_col}_bin'
        try:
            min_age, max_age = df_binned[age_col].min(), df_binned[age_col].max()
            actual_bins = sorted(list(set([b for b in DEFAULT_AGE_BINS if b >= min_age and b <= max_age])))
            if not actual_bins or min_age < actual_bins[0]: actual_bins.insert(0, -np.inf if min_age < DEFAULT_AGE_BINS[0] else min_age)
            if not actual_bins or max_age >= actual_bins[-1]: actual_bins.append(np.inf if max_age >= DEFAULT_AGE_BINS[-1] else max_age + 1e-9)
            actual_bins = sorted(list(set(actual_bins)))
            actual_bins = [b for i, b in enumerate(actual_bins) if i == 0 or b > actual_bins[i-1] + 1e-9]
            if len(actual_bins) < 2: binned_col_name = None
            else:
                num_bins = len(actual_bins) - 1
                labels = DEFAULT_AGE_LABELS[:num_bins] if len(DEFAULT_AGE_LABELS) >= num_bins else [f'AgeBin_{i+1}' for i in range(num_bins)]
                df_binned[binned_col_name] = pd.cut(df_binned[age_col], bins=actual_bins, labels=labels, right=False, include_lowest=True)
                df_binned[binned_col_name] = df_binned[binned_col_name].astype(str).fillna('NaN_Age_Bin')
                try:
                    analysis_cols[analysis_cols.index(age_col)] = binned_col_name
                    analysis_map[age_col] = binned_col_name
                except ValueError:
                    if binned_col_name not in analysis_cols: analysis_cols.append(binned_col_name)
                    analysis_map[age_col] = binned_col_name
        except Exception as bin_err: logging.error(f"Error binning age: {bin_err}", exc_info=True); binned_col_name = None
    elif age_col in p_cols:
         if age_col not in df_binned.columns: logging.warning(f"Protected attribute '{age_col}' not found.")
         else: logging.warning(f"Protected attribute '{age_col}' is not numeric, cannot bin.")

    return df_binned, list(set(analysis_cols)), analysis_map

def cramers_v(x, y):
    """Calculates Cramer's V statistic for categorical association."""
    contingency_table = pd.crosstab(x, y)
    if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2: return 0.0
    try:
        chi2 = chi2_contingency(contingency_table, correction=False)[0]
        n = contingency_table.sum().sum();
        if n == 0: return 0.0
        phi2 = chi2 / n; r, k = contingency_table.shape
        denominator = min(r - 1, k - 1);
        if denominator == 0: return 0.0
        v = np.sqrt(phi2 / denominator); return np.clip(v, 0.0, 1.0)
    except (ValueError, ZeroDivisionError) as e: return 0.0
    except Exception as e: logging.error(f"Cramer's V Error: {e}", exc_info=True); return 0.0

def create_model(model_config):
    """Creates a scikit-learn model instance based on the configuration."""
    model_type = model_config.get('type', 'logistic_regression')
    params = model_config.get('params', {})
    model = None; logging.info(f"Attempting to create model: Type='{model_type}', Params={params}")
    try:
        if model_type == 'logistic_regression':
            lr_params = { 'C': float(params.get('C', 1.0)), 'solver': str(params.get('solver', 'lbfgs')), 'max_iter': 1000, 'random_state': RANDOM_STATE, 'class_weight': 'balanced' }
            if lr_params['C'] <= 0: lr_params['C'] = 1.0
            valid_solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'];
            if lr_params['solver'] not in valid_solvers: lr_params['solver'] = 'lbfgs'
            model = LogisticRegression(**lr_params);
        elif model_type == 'svm':
            svm_params = { 'C': float(params.get('C', 1.0)), 'kernel': str(params.get('kernel', 'rbf')), 'random_state': RANDOM_STATE, 'class_weight': 'balanced', 'probability': False }
            if svm_params['C'] <= 0: svm_params['C'] = 1.0
            valid_kernels = ['linear', 'poly', 'rbf', 'sigmoid'];
            if svm_params['kernel'] not in valid_kernels: svm_params['kernel'] = 'rbf'
            model = SVC(**svm_params);
        else: model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
    except ValueError as ve: logging.error(f"Value error creating model {model_type}: {ve}. Defaulting."); model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
    except Exception as e: logging.error(f"Unexpected error creating model {model_type}: {e}. Defaulting.", exc_info=True); model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
    if model is None: model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
    return model

# --- Routes ---

@app.route('/')
def landing():
    session.clear(); session.modified = True; clear_uploads_folder();
    session.pop('loaded_model_type_in_pipeline', None)
    session.pop('uploaded_model_file_path', None)
    session.pop('uploaded_metadata_file_path', None)
    session.pop('run_history', None); session.modified = True;
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    session.pop('uploaded_file_path', None); session.pop('columns', None)
    session.pop('target', None); session.pop('protected', None); session.pop('features', None)
    session.pop('model_config', None); session.pop('prev_accuracy', None); session.pop('prev_fairness', None)
    session.pop('prev_features', None); session.pop('prev_importances', None)
    session.pop('budget_info', None); session.pop('proxy_results', None); session.pop('outlier_individuals', None)
    session.pop('outcome_polarity', None); session.pop('run_history', None)
    session.pop('uploaded_model_file_path', None); session.pop('uploaded_metadata_file_path', None)
    session.pop('loaded_model_type_in_pipeline', None)
    session.modified = True
    return render_template('index.html')

@app.route('/onboarding/<int:step>')
def onboarding_step(step):
    """Displays a specific step of the onboarding process using static example data."""
    logging.info(f"Accessed onboarding step {step}")
    total_steps = 11 # MODIFIED: Increased from 8 to 11
    example_model_config = {'type': 'logistic_regression', 'params': {'C': 1.0, 'solver': 'lbfgs'}}
    # ... (rest of your example_data remains the same) ...
    example_data = {
        "all_columns": ['income_k', 'credit_score', 'years_at_job', 'has_guarantor', 'neighbourhood', 'applicant_type', 'approved'],
        "target_col": "approved",
        "protected_cols": ["neighbourhood", "applicant_type"],
        "current_feature_cols": ["income_k", "credit_score", "years_at_job", "has_guarantor"],
        "analysis_cols_map": {"neighbourhood": "neighbourhood", "applicant_type": "applicant_type", "age":"age_bin"}, # Added for completeness
        "current_accuracy": 0.85, "test_set_size": 500, "model_config": example_model_config,
        "original_uploaded_model_type": None, # Will be set if an upload was simulated
        "outcome_polarity": "positive",
        "budget_applied_info": {"active": True, "settings": {"neighbourhood": {"Area A": 3, "Area B": None}},
                               "group_importances": {"neighbourhood": {"Area A": [('credit_score', 0.4), ('income_k', 0.2)], "Area B": [('credit_score', 0.35), ('income_k', 0.25)]},
                                                     "applicant_type": {"Family": [('credit_score', 0.38), ('income_k', 0.22)], "Single": [('credit_score', 0.36), ('income_k', 0.24)]}}},
        "group_importances_display": {"neighbourhood": {"Area A": [('credit_score', 0.4), ('income_k', 0.2)], "Area B": [('credit_score', 0.35), ('income_k', 0.25)]},
                                      "applicant_type": {"Family": [('credit_score', 0.38), ('income_k', 0.22)], "Single": [('credit_score', 0.36), ('income_k', 0.24)]}},
        "group_features_used": {"neighbourhood": {"Area A": ['credit_score_processed', 'income_k_processed'], "Area B": ['credit_score_processed', 'income_k_processed', 'has_guarantor_Yes', 'years_at_job_processed']}},
        "current_fairness_results": {
            "neighbourhood": {
                "groups": {
                    "Area A": {'Demographic Parity': 0.70, 'Equalized Odds': 0.80, 'FPR': 0.15, 'Predictive Parity': 0.88, 'Group Size': 250},
                    "Area B": {'Demographic Parity': 0.40, 'Equalized Odds': 0.65, 'FPR': 0.25, 'Predictive Parity': 0.75, 'Group Size': 250}
                },
                "overall_scores": { 'dp_max_diff': 0.30, 'tpr_max_diff': 0.15, 'fpr_max_diff': 0.10, 'eo_overall_score': 0.15, 'pp_max_diff': 0.13 }
            },
            "applicant_type": {
                "groups": {
                    "Family": {'Demographic Parity': 0.58, 'Equalized Odds': 0.75, 'FPR': 0.18, 'Predictive Parity': 0.82, 'Group Size': 300},
                    "Single": {'Demographic Parity': 0.50, 'Equalized Odds': 0.70, 'FPR': 0.22, 'Predictive Parity': 0.80, 'Group Size': 200}
                },
                 "overall_scores": { 'dp_max_diff': 0.08, 'tpr_max_diff': 0.05, 'fpr_max_diff': 0.04, 'eo_overall_score': 0.05, 'pp_max_diff': 0.02 }
            }
        },
        "metric_names": ALL_METRIC_NAMES, # Make sure ALL_METRIC_NAMES is defined if used
        "proxy_analysis_results": {"neighbourhood": {"income_k": {'test_type': 'ANOVA', 'value': 95.2, 'p_value': 0.000, 'significant': True, 'subgroup_details': {'Area A': 85.5, 'Area B': 55.1}},
                                                  "has_guarantor": {'test_type': "Cramer's V", 'value': 0.05, 'p_value': None, 'significant': False, 'subgroup_details': {'Area A': 'No', 'Area B': 'No'}}},
                                  "applicant_type": {"has_guarantor": {'test_type': "Cramer's V", 'value': 0.18, 'p_value': None, 'significant': True, 'subgroup_details': {'Family': 'No', 'Single': 'Yes'}}}},
        "prev_fairness_results": { # For comparison display in results.html, can be simpler for onboarding
             "neighbourhood": { "groups": {"Area A": {'Demographic Parity': 0.72, 'Equalized Odds': 0.78, 'FPR': 0.16, 'Predictive Parity': 0.89, 'Group Size': 248}, "Area B": {'Demographic Parity': 0.38, 'Equalized Odds': 0.66, 'FPR': 0.24, 'Predictive Parity': 0.73, 'Group Size': 252} } },
             "applicant_type": { "groups": {"Family": {'Demographic Parity': 0.55, 'Equalized Odds': 0.73, 'FPR': 0.19, 'Predictive Parity': 0.80, 'Group Size': 305}, "Single": {'Demographic Parity': 0.52, 'Equalized Odds': 0.71, 'FPR': 0.21, 'Predictive Parity': 0.81, 'Group Size': 195} } }
        },
        "prev_accuracy": 0.84,
        "prev_feature_cols": ["income_k", "credit_score", "years_at_job"],
        "pie_chart_data": { # Simplified example pie data for onboarding visuals
             "neighbourhood": {
                 "Area A": {
                     "Demographic Parity": {
                         "outliers":{"applicant_type":{"labels":["Family","Single"],"values":[20,30], "segment_row_numbers": {"Family": [1,2], "Single": [3,4,5]}}},
                         "non_outliers":{"applicant_type":{"labels":["Family","Single"],"values":[100,100], "segment_row_numbers": {"Family": [6,7], "Single": [8,9,10]}}}
                     },
                     "Equalized Odds": { # True=1, Pred=0 (outlier) vs True=1, Pred=1 (non-outlier)
                         "outliers":{"applicant_type":{"labels":["Family","Single"],"values":[5,15], "segment_row_numbers": {"Family": [11], "Single": [12,13]}}},
                         "non_outliers":{"applicant_type":{"labels":["Family","Single"],"values":[90,80], "segment_row_numbers": {"Family": [14], "Single": [15,16]}}}
                     },
                     "Predictive Parity": { # True=0, Pred=1 (outlier) vs True=1, Pred=1 (non-outlier)
                        "outliers":{"applicant_type":{"labels":["Family","Single"],"values":[10,5], "segment_row_numbers": {"Family": [17], "Single": [18,19]}}},
                        "non_outliers":{"applicant_type":{"labels":["Family","Single"],"values":[115,125], "segment_row_numbers": {"Family": [20], "Single": [21,22]}}}
                     }
                 }
                 # Could add Area B if needed, but one group example is often enough for pie onboarding
             }
        },
        "run_history": [
            {"run_id": 1, "timestamp": "2023-10-26 10:00:00", "accuracy": 0.84, "overall_scores": { "neighbourhood": {"dp_max_diff": 0.34, "eo_overall_score": 0.12, "pp_max_diff": 0.16}, "applicant_type": {"dp_max_diff": 0.03, "eo_overall_score": 0.04, "pp_max_diff": 0.01} }, "config": {"model_type": "LR", "num_selected_features": 3, "outcome_polarity": "positive", "budget_active": False, "proxy_enabled": False} },
            {"run_id": 2, "timestamp": "2023-10-26 10:15:00", "accuracy": 0.85, "overall_scores": { "neighbourhood": {"dp_max_diff": 0.30, "eo_overall_score": 0.15, "pp_max_diff": 0.13}, "applicant_type": {"dp_max_diff": 0.08, "eo_overall_score": 0.05, "pp_max_diff": 0.02} }, "config": {"model_type": "LR (C=0.5)", "num_selected_features": 4, "outcome_polarity": "positive", "budget_active": True, "budget_settings_summary":{"neighbourhood":{"Area A":3}}, "proxy_enabled": True}}
        ],
        "individual_details_map": { # Sample for pie chart clicks
            "1": {"row_number": 1, "y_true": "0", "y_pred": "0", "neighbourhood": "Area A", "applicant_type": "Family"},
            "2": {"row_number": 2, "y_true": "0", "y_pred": "0", "neighbourhood": "Area A", "applicant_type": "Family"},
            "6": {"row_number": 6, "y_true": "1", "y_pred": "1", "neighbourhood": "Area A", "applicant_type": "Family"}
        }
    }
    if step < 1: return redirect(url_for('onboarding_step', step=1))
    if step > total_steps: return redirect(url_for('dashboard'))
    return render_template('onboarding.html', step=step, total_steps=total_steps, **example_data)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles dataset file upload."""
    logging.info("Accessed '/upload' (POST) for dataset.");
    if 'file' not in request.files: flash('No file part selected.', 'error'); return redirect(url_for('dashboard'))
    file = request.files['file'];
    if file.filename == '': flash('No file selected.', 'error'); return redirect(url_for('dashboard'))
    save_path=None
    if file and allowed_file(file.filename, ALLOWED_DATASET_EXTENSIONS):
        filename = secure_filename(file.filename); unique_id = str(uuid.uuid4())
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_dataset_{filename}")
        try:
            clear_uploads_folder()
            file.save(save_path);
            try: columns = pd.read_csv(save_path, nrows=0).columns.tolist();
            except pd.errors.EmptyDataError: flash('CSV empty.', 'error'); os.remove(save_path); return redirect(url_for('dashboard'))
            except Exception as pd_err: flash(f'Error reading headers: {pd_err}.', 'error'); os.remove(save_path); return redirect(url_for('dashboard'))

            session.clear()
            session['uploaded_file_path'] = save_path
            session['columns'] = columns
            session.modified = True;
            logging.info(f"Dataset '{filename}' uploaded successfully. Path: {save_path}. Columns: {columns}")
            return redirect(url_for('configure'))
        except Exception as e:
            logging.error(f"Error processing dataset '{filename}': {e}", exc_info=True);
            flash(f'Error processing dataset: {e}.', 'error')
            if save_path and os.path.exists(save_path): os.remove(save_path)
            session.clear(); session.modified = True; return redirect(url_for('dashboard'))
    else: flash('Invalid file type for dataset (.csv only).', 'error'); return redirect(url_for('dashboard'))


@app.route('/configure')
def configure():
    """Displays the configuration page."""
    if 'uploaded_file_path' not in session or 'columns' not in session:
        flash('Please upload a dataset first to begin configuration.', 'warning')
        return redirect(url_for('dashboard'))

    columns = session.get('columns', [])
    uploaded_dataset_path = session.get('uploaded_file_path')

    if not uploaded_dataset_path or not os.path.exists(uploaded_dataset_path):
        flash('Uploaded dataset file is missing from the server. Please re-upload.', 'error')
        session.pop('uploaded_file_path', None)
        session.pop('columns', None)
        session.modified=True
        return redirect(url_for('dashboard'))
    if not columns:
        flash('No columns were found in the uploaded dataset. Please check your file.', 'error')
        session.pop('target', None); session.pop('protected', None); session.pop('features', None)
        session.pop('model_config', None); session.pop('loaded_model_type_in_pipeline', None)
        session.modified=True
        return redirect(url_for('dashboard'))

    detected_loaded_model_type_str = session.get('loaded_model_type_in_pipeline')

    prev_config = {
        'target': session.get('target'),
        'protected': session.get('protected', []),
        'features': session.get('features', []),
        'model_config': session.get('model_config', None),
        'outcome_polarity': session.get('outcome_polarity', 'positive'),
        'uploaded_model_filename': os.path.basename(session.get('uploaded_model_file_path', '')) if session.get('uploaded_model_file_path') else None,
        'uploaded_metadata_filename': os.path.basename(session.get('uploaded_metadata_file_path', '')) if session.get('uploaded_metadata_file_path') else None,
        'detected_loaded_model_type_str': detected_loaded_model_type_str if detected_loaded_model_type_str and detected_loaded_model_type_str != 'Unknown' else None
    }
    logging.info(f"Configure - Initial prev_config from session: {prev_config}")
    logging.info(f"Configure - detected_loaded_model_type_str from session: {detected_loaded_model_type_str}")

    if prev_config['detected_loaded_model_type_str']:
        normalized_detected_type = prev_config['detected_loaded_model_type_str'].lower()
        current_model_type_in_config = prev_config['model_config'].get('type') if prev_config['model_config'] else None

        if 'logisticregression' in normalized_detected_type or 'logistic_regression' in normalized_detected_type:
            if current_model_type_in_config == 'logistic_regression' and prev_config['model_config'].get('params'):
                pass
            else:
                prev_config['model_config'] = {'type': 'logistic_regression', 'params': {'C': 1.0, 'solver': 'lbfgs'}}
            logging.info(f"Configure - Pre-filling with Logistic Regression due to detected type: {prev_config['detected_loaded_model_type_str']}")
        elif 'svm' in normalized_detected_type or 'svc' in normalized_detected_type:
            if current_model_type_in_config == 'svm' and prev_config['model_config'].get('params'):
                pass
            else:
                prev_config['model_config'] = {'type': 'svm', 'params': {'C': 1.0, 'kernel': 'rbf'}}
            logging.info(f"Configure - Pre-filling with SVM due to detected type: {prev_config['detected_loaded_model_type_str']}")
        else:
            logging.info(f"Configure - Detected type '{prev_config['detected_loaded_model_type_str']}' not directly mappable to LR/SVM pre-fill. Model selection will fall back to defaults if not already set.")
            pass

    if not prev_config['model_config'] or \
       not isinstance(prev_config['model_config'], dict) or \
       prev_config['model_config'].get('type') not in ['logistic_regression', 'svm', 'upload_model']:
        prev_config['model_config'] = {'type': 'logistic_regression', 'params': {'C': 1.0, 'solver': 'lbfgs'}}
        logging.info("Configure - model_config was null or invalid type, defaulted to Logistic Regression.")

    model_cfg_type = prev_config['model_config'].get('type')
    if model_cfg_type == 'logistic_regression' and 'params' not in prev_config['model_config']:
        prev_config['model_config']['params'] = {'C': 1.0, 'solver': 'lbfgs'}
        logging.info("Configure - Added default params for LR as they were missing.")
    elif model_cfg_type == 'svm' and 'params' not in prev_config['model_config']:
        prev_config['model_config']['params'] = {'C': 1.0, 'kernel': 'rbf'}
        logging.info("Configure - Added default params for SVM as they were missing.")
    elif model_cfg_type == 'upload_model' and 'params' not in prev_config['model_config'] :
        prev_config['model_config']['params'] = {}


    logging.info(f"Configure - Final prev_config for template: {prev_config}")
    return render_template('configure.html', columns=columns, prev_config=prev_config)


@app.route('/train', methods=['POST'])
def train_and_evaluate():
    """Handles model training (new or loaded pipeline), fairness evaluation, etc."""
    # --- Performance Tracking Initialization ---
    overall_start_time_perf = time.time()
    process_perf = psutil.Process(os.getpid())
    mem_start_rss_perf = process_perf.memory_info().rss / (1024 * 1024) # in MB
    logging.info(f"PERF_METRIC: TrainEndpointStart, MemoryRSS={mem_start_rss_perf:.2f}MB")
    mem_prev_stage_end_perf = mem_start_rss_perf # For calculating stage-wise memory change

    # --- Initialization ---
    global_preprocessor = None
    global_model = None
    original_feature_to_processed_names_map = {}
    processed_name_to_original_feature_map = {}
    X_train_fit_processed = None
    y_train_fit = None
    P_train_fit_df = None
    feature_names_processed = []
    P_test_orig_df = None
    start_time = time.time()
    loaded_model_details_for_run = None
    original_model_type_from_upload = None

    session.pop('loaded_model_type_in_pipeline', None)
    session.modified = True

    initial_gc_count = gc.collect()
    logging.info(f"--- Entering /train endpoint (Initial GC: {initial_gc_count}) ---")

    uploaded_file_path = session.get('uploaded_file_path')
    original_columns = session.get('columns')
    if not uploaded_file_path or not os.path.exists(uploaded_file_path) or not original_columns:
        flash("Session expired or dataset missing. Please upload again.", "error");
        session.clear(); session.modified = True; return redirect(url_for('dashboard'))

    actual_prev_fairness_data = session.get('prev_fairness', {})
    actual_prev_accuracy = session.get('prev_accuracy')
    logging.info(f"Retrieved from session - actual_prev_accuracy: {actual_prev_accuracy}")
    logging.info(f"Retrieved from session - actual_prev_fairness_data keys: {list(actual_prev_fairness_data.keys()) if actual_prev_fairness_data else 'None'}")

    run_history = session.get('run_history', [])

    current_importances = {}; group_models = {}; group_features_used = {}; proxy_analysis_results = {}
    current_model_config = {}; budget_info = {'active': False, 'settings': {}, 'group_importances': {}}
    pie_chart_data = {}; individual_details_map = {}; current_accuracy = np.nan; current_fairness = {}
    results_group_features_used = {}
    calculated_pi_this_run = False

    try:
        # --- STAGE 0: Form Parsing & Configuration Setup ---
        stage0_start_time_perf = time.time()
        logging.info("Stage 0: Parsing form data and setting up configuration...")
        form_data = request.form
        target_col = form_data.get('target_variable')
        protected_cols = request.form.getlist('protected_attributes')
        features = form_data.getlist('model_features')
        model_type_from_form = form_data.get('model_type', 'logistic_regression')
        outcome_polarity = form_data.get('outcome_polarity', 'positive')

        current_model_config['type'] = model_type_from_form
        current_model_config['params'] = {}

        if model_type_from_form == 'logistic_regression':
            try: current_model_config['params']['C'] = float(form_data.get('lr_c', 1.0))
            except (ValueError, TypeError): current_model_config['params']['C'] = 1.0
            current_model_config['params']['solver'] = form_data.get('lr_solver', 'lbfgs')
        elif model_type_from_form == 'svm':
            try: current_model_config['params']['C'] = float(form_data.get('svm_c', 1.0))
            except (ValueError, TypeError): current_model_config['params']['C'] = 1.0
            current_model_config['params']['kernel'] = form_data.get('svm_kernel', 'rbf')
        elif model_type_from_form == 'upload_model':
            current_model_config['params'] = {}

        if model_type_from_form in ['logistic_regression', 'svm']:
            if current_model_config['params'].get('C', 1.0) <= 0:
                 current_model_config['params']['C'] = 1.0
            if model_type_from_form == 'logistic_regression':
                 valid_solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
                 if current_model_config['params'].get('solver') not in valid_solvers: current_model_config['params']['solver'] = 'lbfgs'
            elif model_type_from_form == 'svm':
                 valid_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
                 if current_model_config['params'].get('kernel') not in valid_kernels: current_model_config['params']['kernel'] = 'rbf'

        budgets, enabled_budget_attrs, budget_overall = parse_budgets(form_data, protected_cols)
        run_proxy = form_data.get('enable_proxy_check') == 'yes'
        budget_info['settings'] = budgets

        if not target_col: flash("Target variable missing.", "error"); return redirect(url_for('configure'))
        if not protected_cols: flash("Protected attributes missing.", "error"); return redirect(url_for('configure'))
        if not features and model_type_from_form != 'upload_model':
             flash("Model features missing. Please select features to use for training.", "error"); return redirect(url_for('configure'))
        if model_type_from_form == 'upload_model' and not features:
            logging.info("Running in 'upload_model' mode. Features will be derived from metadata if not selected by user.")
        if target_col in protected_cols: flash("Target cannot be protected.", "error"); return redirect(url_for('configure'))
        if target_col in features: flash("Target cannot be a model feature.", "error"); return redirect(url_for('configure'))
        overlap_protected_features = set(protected_cols) & set(features)
        if overlap_protected_features: flash(f"Attributes '{', '.join(overlap_protected_features)}' cannot be both Protected and a Model Feature.", "error"); return redirect(url_for('configure'))

        session['target'] = target_col; session['protected'] = protected_cols; session['features'] = features;
        session['model_config'] = current_model_config;
        session['outcome_polarity'] = outcome_polarity
        session.modified = True;
        logging.info(f"Run Config (from form): Target='{target_col}', Protected={protected_cols}, Features (Original)={len(features)}:{features}, Model Config={current_model_config}, Budgeting={budget_overall}, Proxy={run_proxy}")

        stage0_duration_perf = time.time() - stage0_start_time_perf
        mem_after_stage0_perf = process_perf.memory_info().rss / (1024 * 1024)
        logging.info(f"PERF_METRIC: Stage_0_FormParseConfig, Duration={stage0_duration_perf:.4f}s, MemoryRSS={mem_after_stage0_perf:.2f}MB, MemChange={(mem_after_stage0_perf - mem_prev_stage_end_perf):.2f}MB")
        mem_prev_stage_end_perf = mem_after_stage0_perf


        # === STAGE 1: Load User's Dataset, Prep, Split ===
        stage1_start_time_perf = time.time()
        logging.info("Stage 1: Loading User's Dataset, Prepping, Splitting Data...")
        all_cols_to_read = list(set([target_col] + protected_cols + (features if features else [])))
        try:
            read_csv_start_time_perf = time.time()
            df_full = pd.read_csv(uploaded_file_path, usecols=lambda c: c in all_cols_to_read if all_cols_to_read else True)
            read_csv_duration_perf = time.time() - read_csv_start_time_perf
            df_full_shape_perf = df_full.shape if 'df_full' in locals() and df_full is not None else 'N/A'
            logging.info(f"PERF_METRIC: Stage_1_pd_read_csv, Duration={read_csv_duration_perf:.4f}s, DataFrameShape={df_full_shape_perf}")
            df_full = df_full.reset_index().rename(columns={'index': 'row_number'})
        except Exception as e: flash(f"Error reading user dataset (cols: {all_cols_to_read}): {e}.", "error"); return redirect(url_for('configure'))

        essential_cols = list(set([target_col] + protected_cols)); rows_before = len(df_full);
        df_filtered = df_full.dropna(subset=essential_cols).copy(); rows_after = len(df_filtered)
        if rows_after < rows_before: logging.warning(f"Dropped {rows_before - rows_after} rows due to NaNs in essential (target/protected) columns.");
        if len(df_filtered) < 10: flash("Insufficient data after handling NaNs in essential columns (less than 10 rows).", 'error'); return redirect(url_for('configure'))
        if features:
            missing_model_features = [f for f in features if f not in df_filtered.columns]
            if missing_model_features:
                flash(f"Selected model features not found in dataset after essential cleaning: {', '.join(missing_model_features)}. Please check selection or data.", "error")
                return redirect(url_for('configure'))

        P_orig_df_unsplit = df_filtered[protected_cols + ['row_number']].copy()
        del df_full; gc.collect()
        df_binned, analysis_cols_list, analysis_map = apply_age_binning(df_filtered, protected_cols); del df_filtered; gc.collect()

        unique_target = df_binned[target_col].unique();
        if len(unique_target) > 2: flash("Target variable is not binary (has more than 2 unique values).", 'error'); return redirect(url_for('configure'))
        if len(unique_target) <= 1: flash("Target variable has only one unique value. Cannot train model.", 'error'); return redirect(url_for('configure'))
        t_map = {v: i for i, v in enumerate(sorted(unique_target))}; df_binned[target_col] = df_binned[target_col].map(t_map).astype(int);

        # === Proxy Analysis ===
        proxy_analysis_duration_perf = 0 # Initialize
        if run_proxy and model_type_from_form == 'upload_model' and not features:
             logging.info("Proxy analysis skipped for uploaded model as features were not pre-selected by user.")
             run_proxy = False
             proxy_analysis_results = {}
        elif run_proxy and features:
             proxy_analysis_start_time_perf = time.time()
             logging.info("--- Running Proxy Feature Analysis ---");
             LOW_CARDINALITY_THRESHOLD = 10
             for p_orig, p_analysis in analysis_map.items():
                 if p_analysis not in df_binned.columns: continue
                 proxy_results_for_attr={}; protected_series=df_binned[p_analysis].copy()
                 current_model_features_in_df = [f for f in features if f in df_binned.columns]
                 for feature_from_user_selection in current_model_features_in_df:
                     if feature_from_user_selection == p_analysis or feature_from_user_selection == p_orig : continue
                     feature_series=df_binned[feature_from_user_selection].copy()
                     temp_df=pd.DataFrame({'feature': feature_series, 'protected': protected_series}).dropna().copy()
                     if temp_df.empty or temp_df['protected'].nunique()<2 or temp_df['feature'].nunique()<1: continue
                     feature_data=temp_df['feature']; protected_data=temp_df['protected']
                     result={'test_type': None, 'value': np.nan, 'p_value': np.nan, 'significant': False, 'subgroup_details': None}; feature_was_numeric = False
                     try:
                         is_num=pd.api.types.is_numeric_dtype(feature_data); n_unique=feature_data.nunique()
                         is_categorical_like = not is_num or (is_num and n_unique <= LOW_CARDINALITY_THRESHOLD)
                         use_anova = is_num and n_unique > LOW_CARDINALITY_THRESHOLD
                         if is_categorical_like and n_unique > 1:
                             feature_was_numeric=False; v_val=cramers_v(feature_data, protected_data); result.update({'test_type': "Cramer's V", 'value': v_val, 'significant': v_val > 0.15})
                         elif use_anova:
                             feature_was_numeric=True;
                             groups=[feature_data[protected_data==g].values for g in protected_data.unique() if len(feature_data[protected_data==g]) > 1]
                             if len(groups) >= 2:
                                 f_val, p_val = f_oneway(*groups); result.update({'test_type': 'ANOVA', 'value': f_val, 'p_value': p_val, 'significant': p_val < 0.05 if pd.notna(p_val) else False})
                             else: result['test_type']='ANOVA (Skipped - <2 groups after NaN drop)';
                         else: result['test_type']=f'Skipped (Type/Unique: is_num={is_num}, n_unique={n_unique})';
                         if result['significant']:
                             try:
                                 details={}; grouped_data = temp_df.groupby('protected')['feature']
                                 details = grouped_data.mean().round(3).to_dict() if feature_was_numeric else grouped_data.agg(lambda x: x.mode()[0] if not x.mode().empty else 'N/A').to_dict()
                                 result['subgroup_details'] = details
                             except Exception as detail_err: logging.warning(f"Proxy subgroup detail error for {feature_from_user_selection} vs {p_analysis}: {detail_err}")
                     except Exception as stat_err: logging.error(f"Stat test error for {feature_from_user_selection} vs {p_analysis}: {stat_err}"); result['test_type'] = 'Error'
                     if result.get('test_type') and not result['test_type'].startswith('Skipped') and result['test_type'] != 'Error':
                         proxy_results_for_attr[feature_from_user_selection]=result
                 if proxy_results_for_attr:
                     sorted_results=sorted(proxy_results_for_attr.items(), key=lambda item: (not item[1]['significant'], -item[1]['value'] if item[1]['test_type'] == "Cramer's V" else item[1].get('p_value', 1.0)));
                     proxy_analysis_results[p_orig]=dict(sorted_results)
             logging.info(f"--- Proxy Analysis finished: {time.time() - proxy_analysis_start_time_perf:.2f} sec ---")
             proxy_analysis_duration_perf = time.time() - proxy_analysis_start_time_perf
             mem_after_proxy_perf = process_perf.memory_info().rss / (1024 * 1024)
             logging.info(f"PERF_METRIC: Stage_1_ProxyAnalysis, Duration={proxy_analysis_duration_perf:.4f}s, MemoryRSS={mem_after_proxy_perf:.2f}MB")
             if 'temp_df' in locals(): del temp_df; gc.collect()
        elif run_proxy and not features:
             logging.info("Proxy analysis skipped as no features were selected by the user.")
             proxy_analysis_results = {}
        # === End Proxy Analysis ===

        X_orig = df_binned[features] if features else pd.DataFrame(index=df_binned.index)
        y = df_binned[target_col]; P_analysis_df = df_binned[analysis_cols_list]; RN = df_binned['row_number']
        if not X_orig.empty and not (X_orig.index.equals(y.index) and y.index.equals(P_analysis_df.index) and P_analysis_df.index.equals(RN.index) and RN.index.equals(P_orig_df_unsplit.index)):
            common_index = df_binned.index
            X_orig = X_orig.loc[common_index]; y = y.loc[common_index]; P_analysis_df = P_analysis_df.loc[common_index]; RN = RN.loc[common_index]; P_orig_df_unsplit = P_orig_df_unsplit.loc[common_index];
        elif X_orig.empty:
             if not (y.index.equals(P_analysis_df.index) and P_analysis_df.index.equals(RN.index) and RN.index.equals(P_orig_df_unsplit.index)):
                common_index = df_binned.index
                y = y.loc[common_index]; P_analysis_df = P_analysis_df.loc[common_index]; RN = RN.loc[common_index]; P_orig_df_unsplit = P_orig_df_unsplit.loc[common_index];
        del df_binned; gc.collect()

        stratify_option = y if y.value_counts().min() >= 2 else None;
        try:
            X_train_perm_orig, X_test_orig, y_train_perm, y_test, P_train_perm_df, P_test_df, RN_train_perm, RN_test, P_train_orig_df, P_test_orig_df = train_test_split(
                X_orig, y, P_analysis_df, RN, P_orig_df_unsplit, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_option
            )
        except ValueError as e:
            flash(f"Error splitting data: {e}.", "error")
            return redirect(url_for('configure'))
        del X_orig, y, P_analysis_df, RN, P_orig_df_unsplit

        X_train_fit_orig = X_train_perm_orig.copy(); y_train_fit = y_train_perm.copy(); P_train_fit_df = P_train_perm_df.copy()
        X_perm_orig, y_perm, P_perm_df = pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame(); pi_data_available = False
        if budget_overall and model_type_from_form != 'upload_model':
            if X_train_perm_orig.empty:
                logging.warning("X_train_perm_orig is empty before PI split. Budgeting disabled.")
                budget_overall = False; enabled_budget_attrs = set(); pi_data_available = False
            else:
                n_train_p = len(y_train_perm); perm_size_float = n_train_p * PERMUTATION_IMPORTANCE_SIZE; fit_size_float = n_train_p - perm_size_float
                if perm_size_float >= MIN_SAMPLES_FOR_PI and fit_size_float >= MIN_SAMPLES_FOR_PI:
                    stratify_perm = y_train_perm if y_train_perm.value_counts().min() >= 2 else None;
                    try:
                        X_train_fit_orig, X_perm_orig, y_train_fit, y_perm, P_train_fit_df, P_perm_df = train_test_split(
                            X_train_perm_orig, y_train_perm, P_train_perm_df,
                            test_size=PERMUTATION_IMPORTANCE_SIZE, random_state=RANDOM_STATE+1, stratify=stratify_perm )
                        pi_data_available=True;
                    except ValueError:
                        logging.warning("Stratified split for PI failed, using full training set for PI.")
                        X_perm_orig, y_perm, P_perm_df = X_train_perm_orig.copy(), y_train_perm.copy(), P_train_perm_df.copy()
                        pi_data_available = n_train_p >= MIN_SAMPLES_FOR_PI
                elif n_train_p >= MIN_SAMPLES_FOR_PI:
                    logging.info("Not enough samples to split for PI, using full training set for PI.")
                    X_perm_orig, y_perm, P_perm_df = X_train_perm_orig.copy(), y_train_perm.copy(), P_train_perm_df.copy()
                    pi_data_available = True;
                else:
                    logging.warning(f"Not enough samples in training data ({n_train_p}) for Permutation Importance. Budgeting disabled.")
                    pi_data_available = False
                if not pi_data_available: budget_overall=False; enabled_budget_attrs=set()
        elif model_type_from_form == 'upload_model':
             budget_overall = False; enabled_budget_attrs = set(); pi_data_available = False

        del X_train_perm_orig, y_train_perm, P_train_perm_df, RN_train_perm, P_train_orig_df; gc.collect();
        logging.info(f"Stage 1 (Data Prep & Split) Done. PI Data Available: {pi_data_available}, Budget Overall: {budget_overall}")

        stage1_duration_perf = time.time() - stage1_start_time_perf # Includes data loading, proxy, and split
        mem_after_stage1_perf = process_perf.memory_info().rss / (1024 * 1024)
        logging.info(f"PERF_METRIC: Stage_1_LoadPrepSplitProxy, Duration={stage1_duration_perf:.4f}s, MemoryRSS={mem_after_stage1_perf:.2f}MB, MemChange={(mem_after_stage1_perf - mem_prev_stage_end_perf):.2f}MB")
        mem_prev_stage_end_perf = mem_after_stage1_perf


        # === STAGE 1.5: Load Pre-trained PIPELINE and Metadata (if selected) ===
        if model_type_from_form == 'upload_model':
            stage1_5_start_time_perf = time.time()
            logging.info("Stage 1.5: Loading pre-trained pipeline and metadata...")
            model_file_path_from_session = session.get('uploaded_model_file_path')
            metadata_file_path_from_session = session.get('uploaded_metadata_file_path')
            model_file_obj = request.files.get('uploaded_model_file')
            metadata_file_obj = request.files.get('uploaded_metadata_file')
            temp_model_save_path = None; temp_metadata_save_path = None; final_model_path = None; final_metadata_path = None
            try:
                if model_file_obj and model_file_obj.filename != '' and allowed_file(model_file_obj.filename, ALLOWED_MODEL_EXTENSIONS):
                    model_filename = secure_filename(model_file_obj.filename); unique_id=str(uuid.uuid4())
                    temp_model_save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_temp_pipeline_{model_filename}")
                    model_file_obj.save(temp_model_save_path); final_model_path = temp_model_save_path
                    session['uploaded_model_file_path'] = final_model_path
                    logging.info(f"New model file uploaded: {final_model_path}")
                elif model_file_path_from_session and os.path.exists(model_file_path_from_session):
                    final_model_path = model_file_path_from_session
                    logging.info(f"Using previously uploaded model file: {final_model_path}")
                else:
                    flash("Pre-trained pipeline file missing.", "error"); return redirect(url_for('configure'))

                if metadata_file_obj and metadata_file_obj.filename != '' and allowed_file(metadata_file_obj.filename, ALLOWED_METADATA_EXTENSIONS):
                    metadata_filename = secure_filename(metadata_file_obj.filename); unique_id=str(uuid.uuid4())
                    temp_metadata_save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_temp_meta_{metadata_filename}")
                    metadata_file_obj.save(temp_metadata_save_path); final_metadata_path = temp_metadata_save_path
                    session['uploaded_metadata_file_path'] = final_metadata_path
                    logging.info(f"New metadata file uploaded: {final_metadata_path}")
                elif metadata_file_path_from_session and os.path.exists(metadata_file_path_from_session):
                    final_metadata_path = metadata_file_path_from_session
                    logging.info(f"Using previously uploaded metadata file: {final_metadata_path}")
                else:
                    flash("Model metadata file missing.", "error"); return redirect(url_for('configure'))

                try: logging.info(f"Loading pickle from '{final_model_path}' in environment with scikit-learn version: {sklearn.__version__}")
                except Exception: pass
                with open(final_model_path, 'rb') as f_model: loaded_pipeline_object = pickle.load(f_model)
                with open(final_metadata_path, 'r') as f_meta: loaded_model_json_meta = json.load(f_meta)

                if not isinstance(loaded_pipeline_object, Pipeline): raise ValueError(f"Loaded object is type '{type(loaded_pipeline_object).__name__}', expected 'Pipeline'.")
                if 'preprocessor' not in loaded_pipeline_object.named_steps or 'classifier' not in loaded_pipeline_object.named_steps: raise ValueError("Pipeline missing 'preprocessor' or 'classifier' step.")

                original_model_type_from_upload = loaded_model_json_meta.get('model_info',{}).get('model_type_in_pipeline', 'Unknown')
                expected_input_features_from_meta = loaded_model_json_meta.get('feature_info', {}).get('input_features_ordered')
                if not isinstance(expected_input_features_from_meta, list) or not expected_input_features_from_meta:
                    raise ValueError("Metadata missing 'input_features_ordered' or it's empty.")

                if original_model_type_from_upload != 'Unknown':
                    session['loaded_model_type_in_pipeline'] = original_model_type_from_upload
                else:
                    session.pop('loaded_model_type_in_pipeline', None)
                session.modified = True

                if not features:
                    features = expected_input_features_from_meta
                    session['features'] = features
                    logging.info(f"Features for uploaded model taken from metadata: {features}")
                    if not features:
                        flash("Internal Error: Features list is empty after processing metadata for uploaded model.", "error")
                        return redirect(url_for('configure'))
                elif set(features) != set(expected_input_features_from_meta):
                    mismatch_msg = f"Mismatch: Your selected features ({len(features)}) do not exactly match the features the uploaded model expects ({len(expected_input_features_from_meta)} from metadata: {expected_input_features_from_meta}). Please select features matching the metadata."
                    logging.error(mismatch_msg)
                    flash(mismatch_msg, "error")
                    return redirect(url_for('configure'))

                if not X_train_fit_orig.empty and not list(X_train_fit_orig.columns) == expected_input_features_from_meta:
                     logging.warning("Re-slicing X_train_fit_orig columns to match metadata (this indicates potential earlier logic issue).")
                     try: X_train_fit_orig = X_train_fit_orig[expected_input_features_from_meta]
                     except KeyError as ke: flash(f"Error aligning training data features with metadata: Missing columns {ke}", "error"); return redirect(url_for('configure'))
                if not X_test_orig.empty and not list(X_test_orig.columns) == expected_input_features_from_meta:
                     logging.warning("Re-slicing X_test_orig columns to match metadata.")
                     try: X_test_orig = X_test_orig[expected_input_features_from_meta]
                     except KeyError as ke: flash(f"Error aligning test data features with metadata: Missing columns {ke}", "error"); return redirect(url_for('configure'))

                loaded_model_details_for_run = {
                    'pipeline_object': loaded_pipeline_object,
                    'expected_input_features_ordered': expected_input_features_from_meta,
                }
                global_model = loaded_pipeline_object
                logging.info(f"Successfully loaded pre-trained pipeline ({original_model_type_from_upload}). Expected features: {expected_input_features_from_meta}")

            except (FileNotFoundError, pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError, KeyError, ValueError, TypeError, json.JSONDecodeError) as load_err:
                 logging.error(f"Error loading/parsing pre-trained pipeline or metadata: {load_err}", exc_info=True)
                 flash(f"Error loading pre-trained model/metadata: {load_err}. Please check files and try again.", "error")
                 session.pop('uploaded_model_file_path', None); session.pop('uploaded_metadata_file_path', None)
                 session.pop('loaded_model_type_in_pipeline', None); session.modified = True
                 if temp_model_save_path and os.path.exists(temp_model_save_path): os.remove(temp_model_save_path)
                 if temp_metadata_save_path and os.path.exists(temp_metadata_save_path): os.remove(temp_metadata_save_path)
                 return redirect(url_for('configure'))
            except Exception as e:
                logging.error(f"Unexpected error during pipeline/metadata loading: {e}", exc_info=True)
                flash(f"Unexpected error loading pipeline/metadata: {e}.", "error")
                session.pop('uploaded_model_file_path', None); session.pop('uploaded_metadata_file_path', None)
                session.pop('loaded_model_type_in_pipeline', None); session.modified = True
                if temp_model_save_path and os.path.exists(temp_model_save_path): os.remove(temp_model_save_path)
                if temp_metadata_save_path and os.path.exists(temp_metadata_save_path): os.remove(temp_metadata_save_path)
                return redirect(url_for('configure'))

            stage1_5_duration_perf = time.time() - stage1_5_start_time_perf
            mem_after_stage1_5_perf = process_perf.memory_info().rss / (1024 * 1024)
            logging.info(f"PERF_METRIC: Stage_1.5_LoadPipeline, Duration={stage1_5_duration_perf:.4f}s, MemoryRSS={mem_after_stage1_5_perf:.2f}MB, MemChange={(mem_after_stage1_5_perf - mem_prev_stage_end_perf):.2f}MB")
            mem_prev_stage_end_perf = mem_after_stage1_5_perf


        # === STAGE 2: Preprocessing & Global Model Training (if NOT loading model) ===
        elif model_type_from_form != 'upload_model':
            stage2_start_time_perf = time.time()
            logging.info("Stage 2: Preprocessing Fit Data & Training Global Model...")
            if X_train_fit_orig.empty:
                flash("Cannot train model: No feature data available in the training split.", "error")
                return redirect(url_for('configure'))

            num_feats_all = X_train_fit_orig.select_dtypes(include=np.number).columns.tolist()
            cat_feats_all = X_train_fit_orig.select_dtypes(exclude=np.number).columns.tolist()
            num_feats = [f for f in num_feats_all if f in features]
            cat_feats = [f for f in cat_feats_all if f in features]

            logging.info(f"Original num_feats for preprocessor: {num_feats}")
            logging.info(f"Original cat_feats for preprocessor: {cat_feats}")
            num_pipe=Pipeline([('imp', SimpleImputer(strategy='mean')), ('scale', StandardScaler())])
            cat_pipe=Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            transformers=[];
            if num_feats: transformers.append(('num', num_pipe, num_feats))
            if cat_feats: transformers.append(('cat', cat_pipe, cat_feats))
            if not transformers: flash("No model features to process after categorization.", "error"); return redirect(url_for('configure'))
            global_preprocessor=ColumnTransformer(transformers=transformers, remainder='drop', verbose_feature_names_out=False)

            preprocessor_fit_transform_start_time_perf = time.time()
            mem_before_proc_fit_perf = process_perf.memory_info().rss / (1024 * 1024)
            try:
                X_train_fit_processed_np = global_preprocessor.fit_transform(X_train_fit_orig)
                feature_names_processed = list(global_preprocessor.get_feature_names_out())
                logging.info(f"Preprocessing fit complete. Processed shape: {X_train_fit_processed_np.shape}.")
            except Exception as e: flash(f"Error preprocessing data for new model: {e}", "error"); logging.error(f"Preprocessing error: {e}", exc_info=True); return redirect(url_for('configure'))
            preprocessor_fit_transform_duration_perf = time.time() - preprocessor_fit_transform_start_time_perf
            mem_after_proc_fit_perf = process_perf.memory_info().rss / (1024 * 1024)
            logging.info(f"PERF_METRIC: Stage_2_PreprocessorFitTransform, Duration={preprocessor_fit_transform_duration_perf:.4f}s, MemoryRSS={mem_after_proc_fit_perf:.2f}MB, MemChange={(mem_after_proc_fit_perf - mem_before_proc_fit_perf):.2f}MB")

            X_train_fit_processed = X_train_fit_processed_np
            original_feature_to_processed_names_map = {orig_f: [] for orig_f in (num_feats + cat_feats)}
            processed_name_to_original_feature_map = {}
            current_proc_idx = 0
            for name_tx, trans_obj, R_cols_tx in global_preprocessor.transformers_:
                if trans_obj == 'drop': continue
                if isinstance(trans_obj, Pipeline):
                    final_estimator_in_pipe = trans_obj.steps[-1][1]
                    for orig_col_name_tx in R_cols_tx:
                        if isinstance(final_estimator_in_pipe, StandardScaler):
                            processed_name_tx = feature_names_processed[current_proc_idx]
                            if processed_name_tx == orig_col_name_tx:
                                original_feature_to_processed_names_map.setdefault(orig_col_name_tx, []).append(processed_name_tx)
                                processed_name_to_original_feature_map[processed_name_tx] = orig_col_name_tx
                            else: logging.warning(f"Name mismatch for scaled num feat: orig='{orig_col_name_tx}', proc='{processed_name_tx}'")
                            current_proc_idx += 1
                        elif isinstance(final_estimator_in_pipe, OneHotEncoder):
                            try:
                                ohe_output_names_for_this_orig_col = final_estimator_in_pipe.get_feature_names_out([orig_col_name_tx])
                                for _ in ohe_output_names_for_this_orig_col:
                                    processed_name_tx = feature_names_processed[current_proc_idx]
                                    original_feature_to_processed_names_map.setdefault(orig_col_name_tx, []).append(processed_name_tx)
                                    processed_name_to_original_feature_map[processed_name_tx] = orig_col_name_tx
                                    current_proc_idx += 1
                            except Exception as ohe_map_err:
                                logging.error(f"Error mapping OHE names for '{orig_col_name_tx}': {ohe_map_err}.")
                                while current_proc_idx < len(feature_names_processed) and \
                                      (feature_names_processed[current_proc_idx].startswith(orig_col_name_tx + "_") or \
                                       feature_names_processed[current_proc_idx] == orig_col_name_tx) :
                                    processed_name_tx = feature_names_processed[current_proc_idx]
                                    original_feature_to_processed_names_map.setdefault(orig_col_name_tx, []).append(processed_name_tx)
                                    processed_name_to_original_feature_map[processed_name_tx] = orig_col_name_tx
                                    current_proc_idx += 1
            if current_proc_idx != len(feature_names_processed): logging.warning(f"Feature name mapping incomplete: Index {current_proc_idx} vs Total processed {len(feature_names_processed)}.")
            logging.info(f"Original to Processed Names Map created.")
            del X_train_fit_orig

            global_model = create_model(current_model_config); # The classifier
            if X_train_fit_processed.shape[0] != len(y_train_fit): raise ValueError("Shape mismatch before model fit.")

            model_fit_start_time_perf = time.time()
            mem_before_model_fit_perf = process_perf.memory_info().rss / (1024 * 1024)
            try:
                global_model.fit(X_train_fit_processed, y_train_fit);
                logging.info(f"New global model fitting complete.")
            except Exception as e: flash(f"Error training model: {e}", "error"); return redirect(url_for('configure'))
            model_fit_duration_perf = time.time() - model_fit_start_time_perf
            mem_after_model_fit_perf = process_perf.memory_info().rss / (1024 * 1024)
            logging.info(f"PERF_METRIC: Stage_2_ModelFit, Duration={model_fit_duration_perf:.4f}s, MemoryRSS={mem_after_model_fit_perf:.2f}MB, MemChange={(mem_after_model_fit_perf - mem_before_model_fit_perf):.2f}MB")

            gc.collect(); logging.info(f"Stage 2 Done. Global model (classifier) object type: {type(global_model).__name__}")

            stage2_duration_perf = time.time() - stage2_start_time_perf
            mem_after_stage2_perf = process_perf.memory_info().rss / (1024 * 1024)
            logging.info(f"PERF_METRIC: Stage_2_PreprocTrain, Duration={stage2_duration_perf:.4f}s, MemoryRSS={mem_after_stage2_perf:.2f}MB, MemChange={(mem_after_stage2_perf - mem_prev_stage_end_perf):.2f}MB")
            mem_prev_stage_end_perf = mem_after_stage2_perf
        else:
             gc.collect(); logging.info(f"Stage 2 Activities Skipped (Used loaded pipeline).")


        # === STAGE 3: Permutation Importance Calculation ===
        calculated_pi_this_run = False
        current_importances = {}
        if budget_overall and pi_data_available and model_type_from_form != 'upload_model':
            stage3_start_time_perf = time.time()
            if X_perm_orig.empty or not feature_names_processed:
                 logging.warning("Permutation Importance skipped: No data or processed feature names for PI.")
            else:
                logging.info(f"Stage 3: Calculating Group Permutation Importance using {X_perm_orig.shape[0]} samples...")
                pi_start = time.time(); X_perm_processed_np = None; y_perm_values = None
                try:
                    X_perm_processed_np = global_preprocessor.transform(X_perm_orig)
                    y_perm_values = y_perm.values; del X_perm_orig
                    if len(P_perm_df) != len(y_perm_values): raise ValueError("PI data P_perm_df and y_perm length mismatch.")
                    pi_scorer = 'accuracy'
                    for attribute in enabled_budget_attrs:
                        analysis_attr = analysis_map.get(attribute, attribute); importances_for_attr = {}
                        if analysis_attr not in P_perm_df.columns: continue
                        p_perm_df_indexed = P_perm_df.reset_index(drop=True)
                        y_perm_values_indexed = pd.Series(y_perm_values).reset_index(drop=True)
                        x_perm_processed_df_indexed = pd.DataFrame(X_perm_processed_np).reset_index(drop=True)
                        unique_groups_in_perm = p_perm_df_indexed[analysis_attr].dropna().unique()
                        for group_name_pi in unique_groups_in_perm:
                            group_str=str(group_name_pi); pi_res_obj=None
                            try:
                                mask = (p_perm_df_indexed[analysis_attr] == group_name_pi).values
                                n_samples_pi = mask.sum()
                                if n_samples_pi < MIN_SAMPLES_FOR_PI: continue
                                X_pi_np = x_perm_processed_df_indexed[mask].values; y_pi_np = y_perm_values_indexed[mask].values
                                if len(np.unique(y_pi_np)) < 2 : continue
                                if np.isnan(X_pi_np).any() or np.isnan(y_pi_np).any(): continue
                                if len(feature_names_processed) != X_pi_np.shape[1]: continue
                                pi_res_obj = permutation_importance(global_model, X_pi_np, y_pi_np, n_repeats=N_PERMUTATION_REPEATS, random_state=RANDOM_STATE, scoring=pi_scorer, n_jobs=PI_N_JOBS)
                                aggregated_importances_for_group = {}
                                for i, processed_feat_name in enumerate(feature_names_processed):
                                    original_feat_name = processed_name_to_original_feature_map.get(processed_feat_name)
                                    if original_feat_name: aggregated_importances_for_group[original_feat_name] = aggregated_importances_for_group.get(original_feat_name, 0.0) + pi_res_obj.importances_mean[i]
                                imp_list_aggregated = sorted(aggregated_importances_for_group.items(), key=lambda item: item[1], reverse=True)
                                imp_list_final = [(orig_name, agg_score) for orig_name, agg_score in imp_list_aggregated if agg_score > 1e-6]
                                if imp_list_final: importances_for_attr[group_str]=imp_list_final; calculated_pi_this_run=True;
                            except Exception as e: logging.error(f"PI Error for {attribute}/{group_str}: {e}", exc_info=True)
                            finally: del pi_res_obj; gc.collect()
                        if importances_for_attr: current_importances[attribute] = importances_for_attr
                    logging.info(f"Stage 3 (PI) Done. Time: {time.time() - pi_start:.2f} sec. Calculated new PI: {calculated_pi_this_run}")
                except Exception as e: logging.error(f"Error during PI Stage 3 setup/loop: {e}", exc_info=True)
                finally:
                    if 'X_perm_processed_np' in locals() and X_perm_processed_np is not None: del X_perm_processed_np
                    if 'y_perm_values' in locals() and y_perm_values is not None: del y_perm_values
                    if 'P_perm_df' in locals() and P_perm_df is not None: del P_perm_df
                    if 'x_perm_processed_df_indexed' in locals(): del x_perm_processed_df_indexed
                    if 'y_perm_values_indexed' in locals(): del y_perm_values_indexed
                    if 'p_perm_df_indexed' in locals(): del p_perm_df_indexed
                    gc.collect()

            stage3_duration_perf = time.time() - stage3_start_time_perf
            mem_after_stage3_perf = process_perf.memory_info().rss / (1024 * 1024)
            logging.info(f"PERF_METRIC: Stage_3_PermutationImportance, Duration={stage3_duration_perf:.4f}s, MemoryRSS={mem_after_stage3_perf:.2f}MB, MemChange={(mem_after_stage3_perf - mem_prev_stage_end_perf):.2f}MB, CalculatedPI={calculated_pi_this_run}")
            mem_prev_stage_end_perf = mem_after_stage3_perf

        if calculated_pi_this_run:
            budget_info['group_importances'] = current_importances
            session['prev_importances'] = current_importances
        else:
            prev_importances_from_session = session.get('prev_importances', {})
            budget_info['group_importances'] = prev_importances_from_session
            if budget_overall and prev_importances_from_session:
                logging.warning("Using previously stored importances for budgeting.")
                flash("Warning: Using importance scores from a previous run for budgeting.", "warning")
            elif budget_overall and not prev_importances_from_session:
                logging.warning("Budgeting enabled but no importance scores available. Budgeting skipped.");
                flash("Budgeting skipped: Importance scores are not available.", "warning")
                budget_overall=False; enabled_budget_attrs=set()
        session.modified = True; budget_info['active'] = False


        # === STAGE 4: Group-Specific Feature Setup ===
        active_importances_for_budget = budget_info['group_importances']
        results_group_features_used = {}
        if budget_overall and active_importances_for_budget and model_type_from_form != 'upload_model':
            stage4_start_time_perf = time.time()
            logging.info("Stage 4: Setting up Group-Specific Features based on Budgets...")
            stage4_start = time.time()
            for attribute_pa in enabled_budget_attrs:
                imp_for_attr_pa = active_importances_for_budget.get(attribute_pa)
                if not imp_for_attr_pa: continue
                limits_pa = budget_info['settings'].get(attribute_pa, {})
                results_group_features_used[attribute_pa]={}
                for group_str_pa, imp_list_original_features_pa in imp_for_attr_pa.items():
                    if not imp_list_original_features_pa: continue
                    N_budget_pa = limits_pa.get(group_str_pa)
                    top_n_original_features_selected_pa = []
                    if N_budget_pa is None: top_n_original_features_selected_pa = [orig_fname for orig_fname, score in imp_list_original_features_pa]
                    elif isinstance(N_budget_pa, int) and N_budget_pa >= 0: top_n_original_features_selected_pa = [orig_fname for orig_fname, score in imp_list_original_features_pa[:min(N_budget_pa, len(imp_list_original_features_pa))]]
                    else: continue
                    current_group_processed_features_pa = []
                    for original_feat_to_include_pa in top_n_original_features_selected_pa:
                        processed_names_for_original_pa = original_feature_to_processed_names_map.get(original_feat_to_include_pa)
                        if processed_names_for_original_pa: current_group_processed_features_pa.extend(processed_names_for_original_pa)
                    final_processed_features_for_group_pa = list(set(f_proc for f_proc in current_group_processed_features_pa if f_proc in feature_names_processed))
                    if not final_processed_features_for_group_pa and N_budget_pa is not None and N_budget_pa > 0 and top_n_original_features_selected_pa: logging.warning(f"No valid *processed* features for {attribute_pa}/{group_str_pa} after mapping.")
                    results_group_features_used[attribute_pa][group_str_pa] = final_processed_features_for_group_pa
                    budget_info['active'] = True
            if 'X_train_fit_processed' in locals() and X_train_fit_processed is not None: del X_train_fit_processed
            if 'y_train_fit' in locals() and y_train_fit is not None: del y_train_fit
            if 'P_train_fit_df' in locals() and P_train_fit_df is not None: del P_train_fit_df;
            gc.collect();
            logging.info(f"Stage 4 (Group Feature Setup) Done. Budgeting active: {budget_info['active']}. Time: {time.time() - stage4_start:.2f} sec.")

            stage4_duration_perf = time.time() - stage4_start_time_perf
            mem_after_stage4_perf = process_perf.memory_info().rss / (1024 * 1024)
            logging.info(f"PERF_METRIC: Stage_4_GroupFeatureSetup, Duration={stage4_duration_perf:.4f}s, MemoryRSS={mem_after_stage4_perf:.2f}MB, MemChange={(mem_after_stage4_perf - mem_prev_stage_end_perf):.2f}MB, BudgetActive={budget_info['active']}")
            mem_prev_stage_end_perf = mem_after_stage4_perf
        else:
            logging.info("Stage 4: Skipping Group-Specific Feature Setup.")
        results_budget_applied_info = budget_info


        # === STAGE 5: Prediction on Test Set & Evaluation ===
        stage5_start_time_perf = time.time()
        logging.info("Stage 5: Predicting on Test Set and Evaluating Fairness...")
        stage5_start = time.time();
        X_test_processed_df = None
        if model_type_from_form != 'upload_model':
            if X_test_orig.empty: flash("Test data (X_test_orig) is empty. Cannot evaluate new model.", "error"); return redirect(url_for('configure'))
            logging.info(f"Preprocessing entire X_test_orig (shape {X_test_orig.shape}) for new model evaluation...")

            preprocessor_transform_test_start_time_perf = time.time()
            mem_before_proc_test_perf = process_perf.memory_info().rss / (1024 * 1024)
            try:
                X_test_processed_np = global_preprocessor.transform(X_test_orig)
                X_test_processed_df = pd.DataFrame(X_test_processed_np, index=X_test_orig.index, columns=feature_names_processed)
                logging.info(f"Bulk preprocessing of test set complete. Shape: {X_test_processed_df.shape}")
                del X_test_processed_np
            except Exception as e:
                flash(f"Error preprocessing test set for new model: {e}", "error"); logging.error(f"Test set preprocessing failed: {e}", exc_info=True);
                if 'X_test_orig' in locals(): del X_test_orig; gc.collect(); return redirect(url_for('configure'))
            preprocessor_transform_test_duration_perf = time.time() - preprocessor_transform_test_start_time_perf
            mem_after_proc_test_perf = process_perf.memory_info().rss / (1024 * 1024)
            logging.info(f"PERF_METRIC: Stage_5_PreprocessorTransformTest, Duration={preprocessor_transform_test_duration_perf:.4f}s, MemoryRSS={mem_after_proc_test_perf:.2f}MB, MemChange={(mem_after_proc_test_perf - mem_before_proc_test_perf):.2f}MB")


        common_test_index = y_test.index
        if not P_test_df.index.equals(common_test_index): P_test_df = P_test_df.reindex(common_test_index)
        if not RN_test.index.equals(common_test_index): RN_test = RN_test.reindex(common_test_index)
        if P_test_orig_df is not None and not P_test_orig_df.index.equals(common_test_index): P_test_orig_df = P_test_orig_df.reindex(common_test_index)
        elif P_test_orig_df is None: P_test_orig_df = pd.DataFrame(index=common_test_index)

        if model_type_from_form == 'upload_model':
            if not X_test_orig.index.equals(common_test_index): X_test_orig = X_test_orig.reindex(common_test_index)
            if X_test_orig.empty and features: flash("Data for test set features is missing for the uploaded model evaluation.", "error"); return redirect(url_for('configure'))
        elif X_test_processed_df is not None:
             if not X_test_processed_df.index.equals(common_test_index): X_test_processed_df = X_test_processed_df.reindex(common_test_index)

        prediction_loop_start_time_perf = time.time()
        mem_before_pred_loop_perf = process_perf.memory_info().rss / (1024 * 1024)
        y_preds_list=[]; model_usage_log = {"global_pred": 0}
        processed_feature_indices = {name: i for i, name in enumerate(feature_names_processed)} if feature_names_processed else {}
        for idx in common_test_index:
            model_type_used_for_row = "global_pred"
            budget_features_for_this_row_processed = None
            try:
                if idx not in P_test_df.index: y_preds_list.append(0); continue
                p_info_series_row = P_test_df.loc[idx]
                if results_budget_applied_info.get('active', False) and model_type_from_form != 'upload_model':
                     for attr_pa_pred, group_feature_map_pred in results_group_features_used.items():
                         analysis_attr_pa_pred = analysis_map.get(attr_pa_pred, attr_pa_pred)
                         if analysis_attr_pa_pred not in p_info_series_row.index: continue
                         group_val_pred = p_info_series_row.get(analysis_attr_pa_pred); group_test_str_pred = str(group_val_pred) if pd.notna(group_val_pred) else "NaN_Group"
                         if group_test_str_pred in group_feature_map_pred:
                             budget_features_for_this_row_processed = group_feature_map_pred[group_test_str_pred]
                             model_type_used_for_row = f"group_budgeted ({attr_pa_pred}/{group_test_str_pred})"; break
                pred_val = None
                if model_type_from_form == 'upload_model':
                    if idx not in X_test_orig.index: pred_val=0
                    else:
                        row_data_orig_df_pred = X_test_orig.loc[[idx]]
                        try:
                            row_data_orig_reordered_pred = row_data_orig_df_pred[loaded_model_details_for_run['expected_input_features_ordered']]
                            pred_val = global_model.predict(row_data_orig_reordered_pred)[0]
                        except Exception as e_upl_pred: logging.error(f"Error predict upload idx {idx}: {e_upl_pred}"); pred_val = 0
                else:
                    if idx not in X_test_processed_df.index: pred_val=0
                    else:
                        processed_row_np_pred = X_test_processed_df.loc[[idx]].values; classifier_to_use = global_model
                        if budget_features_for_this_row_processed is not None:
                            if not budget_features_for_this_row_processed: pred_val = 0
                            else:
                                try:
                                    processed_row_budgeted_np_pred = processed_row_np_pred.copy()
                                    all_indices_set_pred = set(range(len(feature_names_processed)))
                                    budgeted_indices_set_pred = set(processed_feature_indices[f_name] for f_name in budget_features_for_this_row_processed if f_name in processed_feature_indices)
                                    indices_to_zero_out_pred = list(all_indices_set_pred - budgeted_indices_set_pred)
                                    if indices_to_zero_out_pred: processed_row_budgeted_np_pred[:, indices_to_zero_out_pred] = 0
                                    pred_val = classifier_to_use.predict(processed_row_budgeted_np_pred)[0]
                                except KeyError as ke_pred: pred_val = classifier_to_use.predict(processed_row_np_pred)[0]; model_type_used_for_row += " (Fallback)"
                                except Exception as budget_pred_err_pred: logging.error(f"Error budget predict idx {idx}: {budget_pred_err_pred}"); pred_val = 0; model_type_used_for_row += " (Fallback)"
                        else: pred_val = classifier_to_use.predict(processed_row_np_pred)[0]
                if pred_val is None: pred_val = 0
                y_preds_list.append(pred_val); model_usage_log[model_type_used_for_row] = model_usage_log.get(model_type_used_for_row, 0) + 1
            except Exception as e_pred_loop: logging.error(f"General prediction loop error idx {idx}: {e_pred_loop}"); y_preds_list.append(0); model_usage_log["error_prediction_loop"] = model_usage_log.get("error_prediction_loop", 0) + 1
        if len(y_preds_list) != len(y_test): flash("Internal error: Prediction count mismatch.", "error"); return redirect(url_for('configure'))
        y_pred_series=pd.Series(y_preds_list, index=common_test_index);
        logging.info(f"Prediction Model Usage Counts: {model_usage_log}");

        prediction_loop_duration_perf = time.time() - prediction_loop_start_time_perf
        mem_after_pred_loop_perf = process_perf.memory_info().rss / (1024 * 1024)
        logging.info(f"PERF_METRIC: Stage_5_PredictionLoop, Duration={prediction_loop_duration_perf:.4f}s, MemoryRSS={mem_after_pred_loop_perf:.2f}MB, MemChange={(mem_after_pred_loop_perf - mem_before_pred_loop_perf):.2f}MB, NumPredictions={len(y_preds_list)}")

        if X_test_processed_df is not None: del X_test_processed_df; gc.collect()

        current_accuracy = accuracy_score(y_test, y_pred_series) if not y_test.empty else np.nan
        test_set_size = len(y_pred_series)
        logging.info(f"Overall Accuracy (Current Run): {current_accuracy:.4f}")

        results_df=P_test_df.copy(); results_df['y_true']=y_test.loc[results_df.index]; results_df['y_pred']=y_pred_series.loc[results_df.index]; results_df['row_number'] = RN_test.loc[results_df.index]
        results_full_df = results_df.copy()
        if P_test_orig_df is not None and not P_test_orig_df.empty and 'row_number' in P_test_orig_df.columns:
             orig_protected_cols_for_merge = [col for col in protected_cols if col in P_test_orig_df.columns]
             if orig_protected_cols_for_merge:
                 cols_to_select_from_P_orig = ['row_number'] + orig_protected_cols_for_merge
                 try: p_test_orig_df_unique_rows = P_test_orig_df[cols_to_select_from_P_orig].drop_duplicates(subset=['row_number']); results_full_df = pd.merge(results_full_df, p_test_orig_df_unique_rows, on='row_number', how='left', suffixes=('', '_orig'))
                 except Exception as merge_err: logging.error(f"Error merging P_test_orig_df: {merge_err}")
        if 'row_number' not in results_full_df.columns and RN_test is not None: results_full_df['row_number'] = RN_test.loc[results_full_df.index]

        fairness_calc_start_time_perf = time.time()
        mem_before_fairness_perf = process_perf.memory_info().rss / (1024 * 1024)
        current_fairness = {}
        logging.info("Calculating fairness metrics...")
        for attr_orig_fairness in protected_cols:
            analysis_attr_for_fairness_calc = analysis_map.get(attr_orig_fairness, attr_orig_fairness)
            attr_results_fairness = {'groups': {}, 'overall_scores': {}}; all_group_metrics_for_overall_f = []
            if analysis_attr_for_fairness_calc not in results_df.columns: continue
            try:
                for group_name_f, group_df_f in results_df.groupby(analysis_attr_for_fairness_calc, observed=False, dropna=False):
                    group_str_f = str(group_name_f) if pd.notna(group_name_f) else "NaN_Group";
                    metrics_f = calculate_fairness_metrics(group_df_f)
                    attr_results_fairness['groups'][group_str_f] = metrics_f
                    if metrics_f['Group Size'] > 0: all_group_metrics_for_overall_f.append(metrics_f)
                if len(all_group_metrics_for_overall_f) > 1:
                    def get_valid_rates_f(ml, k): return [m[k] for m in ml if k in m and pd.notna(m[k])]
                    dp_r_f=get_valid_rates_f(all_group_metrics_for_overall_f,'Demographic Parity'); tpr_r_f=get_valid_rates_f(all_group_metrics_for_overall_f,'Equalized Odds'); fpr_r_f=get_valid_rates_f(all_group_metrics_for_overall_f,'FPR'); pp_r_f=get_valid_rates_f(all_group_metrics_for_overall_f,'Predictive Parity')
                    dp_d_f=max(dp_r_f)-min(dp_r_f) if dp_r_f else np.nan; tpr_d_f=max(tpr_r_f)-min(tpr_r_f) if tpr_r_f else np.nan; fpr_d_f=max(fpr_r_f)-min(fpr_r_f) if fpr_r_f else np.nan; pp_d_f=max(pp_r_f)-min(pp_r_f) if pp_r_f else np.nan
                    eo_d_f=[d for d in [tpr_d_f,fpr_d_f] if pd.notna(d)]; eo_s_f=max(eo_d_f) if eo_d_f else np.nan
                    attr_results_fairness['overall_scores']={'dp_max_diff':dp_d_f,'tpr_max_diff':tpr_d_f,'fpr_max_diff':fpr_d_f,'eo_overall_score':eo_s_f,'pp_max_diff':pp_d_f}
                else: attr_results_fairness['overall_scores']={'dp_max_diff':np.nan,'tpr_max_diff':np.nan,'fpr_max_diff':np.nan,'eo_overall_score':np.nan,'pp_max_diff':np.nan}
                current_fairness[attr_orig_fairness] = attr_results_fairness
            except Exception as e_f_calc: logging.error(f"Error calculating fairness for '{attr_orig_fairness}': {e_f_calc}", exc_info=True)

        fairness_calc_duration_perf = time.time() - fairness_calc_start_time_perf
        mem_after_fairness_perf = process_perf.memory_info().rss / (1024 * 1024)
        logging.info(f"PERF_METRIC: Stage_5_FairnessCalc, Duration={fairness_calc_duration_perf:.4f}s, MemoryRSS={mem_after_fairness_perf:.2f}MB, MemChange={(mem_after_fairness_perf - mem_before_fairness_perf):.2f}MB")

        pie_chart_calc_start_time_perf = time.time()
        mem_before_pie_perf = process_perf.memory_info().rss / (1024 * 1024)
        pie_chart_data = {}
        logging.info("Calculating pie chart data...")
        if 'y_pred' in results_full_df.columns and 'y_true' in results_full_df.columns and 'row_number' in results_full_df.columns and len(protected_cols) >= 2:
             for primary_attr_pie_orig in protected_cols:
                analysis_attr_primary_pie_grouping = analysis_map.get(primary_attr_pie_orig, primary_attr_pie_orig)
                if analysis_attr_primary_pie_grouping not in results_full_df.columns: continue
                pie_chart_data[primary_attr_pie_orig] = {}
                other_prot_cols_pie_orig = [p_orig for p_orig in protected_cols if p_orig != primary_attr_pie_orig]
                for group_name_pie, group_df_pie in results_full_df.groupby(analysis_attr_primary_pie_grouping, observed=False, dropna=False):
                    if group_df_pie.empty: continue
                    group_str_pie = str(group_name_pie) if pd.notna(group_name_pie) else "NaN_Group";
                    pie_chart_data[primary_attr_pie_orig][group_str_pie] = {}
                    for metric_name_pie in METRICS_FOR_OUTLIERS:
                        pie_chart_data[primary_attr_pie_orig][group_str_pie][metric_name_pie] = {'outliers':{},'non_outliers':{}}
                        outlier_df_pie, non_outlier_df_pie = pd.DataFrame(), pd.DataFrame()
                        try:
                            if metric_name_pie == 'Demographic Parity': outlier_df_pie,non_outlier_df_pie = group_df_pie[group_df_pie['y_pred']==0].copy(), group_df_pie[group_df_pie['y_pred']==1].copy()
                            elif metric_name_pie == 'Equalized Odds': outlier_df_pie,non_outlier_df_pie = group_df_pie[(group_df_pie['y_true']==1)&(group_df_pie['y_pred']==0)].copy(), group_df_pie[(group_df_pie['y_true']==1)&(group_df_pie['y_pred']==1)].copy()
                            elif metric_name_pie == 'Predictive Parity': outlier_df_pie,non_outlier_df_pie = group_df_pie[(group_df_pie['y_true']==0)&(group_df_pie['y_pred']==1)].copy(), group_df_pie[(group_df_pie['y_true']==1)&(group_df_pie['y_pred']==1)].copy()
                        except KeyError: continue
                        for data_type_pie, df_to_analyze_pie in [('outliers', outlier_df_pie), ('non_outliers', non_outlier_df_pie)]:
                            if df_to_analyze_pie.empty: continue
                            pie_chart_data[primary_attr_pie_orig][group_str_pie][metric_name_pie][data_type_pie] = {}
                            for other_attr_pie_orig_dist in other_prot_cols_pie_orig:
                                col_for_pie_dist = analysis_map.get(other_attr_pie_orig_dist, other_attr_pie_orig_dist)
                                if col_for_pie_dist not in df_to_analyze_pie.columns: pie_chart_data[primary_attr_pie_orig][group_str_pie][metric_name_pie][data_type_pie][other_attr_pie_orig_dist] = {'labels':[],'values':[],'segment_row_numbers':{}}; continue
                                try:
                                    dist_pie = df_to_analyze_pie[col_for_pie_dist].astype(str).value_counts(dropna=False).sort_index()
                                    if not dist_pie.empty:
                                        labels_pie = [str(l) for l in dist_pie.index.tolist()]; values_pie = dist_pie.values.tolist()
                                        seg_rns_pie = {}
                                        for lbl_idx_pie, current_label_val_str_pie in enumerate(labels_pie):
                                            if current_label_val_str_pie.lower() == 'nan' or current_label_val_str_pie == "NaN_Actual_Value" or current_label_val_str_pie == "NaN_Age_Bin": segment_rows_pie = df_to_analyze_pie[df_to_analyze_pie[col_for_pie_dist].isna()]['row_number'].tolist()
                                            else: segment_rows_pie = df_to_analyze_pie[df_to_analyze_pie[col_for_pie_dist].astype(str) == current_label_val_str_pie]['row_number'].tolist()
                                            seg_rns_pie[current_label_val_str_pie] = segment_rows_pie
                                        pie_chart_data[primary_attr_pie_orig][group_str_pie][metric_name_pie][data_type_pie][other_attr_pie_orig_dist] = {'labels':labels_pie, 'values':values_pie, 'segment_row_numbers':seg_rns_pie}
                                    else: pie_chart_data[primary_attr_pie_orig][group_str_pie][metric_name_pie][data_type_pie][other_attr_pie_orig_dist] = {'labels':[],'values':[],'segment_row_numbers':{}}
                                except Exception as e_pie_data: logging.error(f"Pie data error {primary_attr_pie_orig}/{group_str_pie}/{metric_name_pie}/{other_attr_pie_orig_dist} (col '{col_for_pie_dist}'): {e_pie_data}"); pie_chart_data[primary_attr_pie_orig][group_str_pie][metric_name_pie][data_type_pie][other_attr_pie_orig_dist] = {'labels':[],'values':[],'segment_row_numbers':{}}

        pie_chart_calc_duration_perf = time.time() - pie_chart_calc_start_time_perf
        mem_after_pie_perf = process_perf.memory_info().rss / (1024 * 1024)
        logging.info(f"PERF_METRIC: Stage_5_PieChartCalc, Duration={pie_chart_calc_duration_perf:.4f}s, MemoryRSS={mem_after_pie_perf:.2f}MB, MemChange={(mem_after_pie_perf - mem_before_pie_perf):.2f}MB")

        individual_details_map = {}
        logging.info("Preparing individual details map...")
        if 'results_full_df' in locals() and results_full_df is not None and not results_full_df.empty and 'row_number' in results_full_df.columns:
            display_cols_ind = ['row_number','y_true','y_pred'] + [col for col in protected_cols if col in results_full_df.columns]
            for p_orig_ind, p_binned_ind in analysis_map.items():
                if p_binned_ind != p_orig_ind and p_binned_ind in results_full_df.columns and p_binned_ind not in display_cols_ind: display_cols_ind.append(p_binned_ind)
            actual_display_cols_ind = list(set(col for col in display_cols_ind if col in results_full_df.columns))
            if 'row_number' in actual_display_cols_ind:
                try:
                    temp_df_ind = results_full_df[actual_display_cols_ind].copy().drop_duplicates(subset=['row_number'])
                    for col_ind in temp_df_ind.columns:
                        if col_ind == 'row_number': temp_df_ind[col_ind] = temp_df_ind[col_ind].astype(int)
                        else: temp_df_ind[col_ind] = temp_df_ind[col_ind].astype(str)
                    individual_details_map = {str(k):v for k,v in temp_df_ind.set_index('row_number').to_dict(orient='index').items()}
                    del temp_df_ind
                except Exception as e_ind_map: logging.error(f"Error creating individual_details_map: {e_ind_map}")


        run_model_type_for_history = current_model_config.get('type', 'N/A')
        if model_type_from_form == 'upload_model' and original_model_type_from_upload and original_model_type_from_upload != 'Unknown': run_model_type_for_history = f"uploaded ({original_model_type_from_upload})"
        elif model_type_from_form == 'upload_model': run_model_type_for_history = "uploaded (Type Unknown from metadata)"
        current_run_data = { 'run_id': len(run_history) + 1, 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"), 'accuracy': current_accuracy if pd.notna(current_accuracy) else None, 'overall_scores': {}, 'config': { 'model_type': run_model_type_for_history, 'model_params': current_model_config.get('params', {}) if model_type_from_form != 'upload_model' else "N/A (Pre-trained)", 'num_selected_features': len(features), 'outcome_polarity': outcome_polarity, 'budget_active': results_budget_applied_info.get('active', False), 'budget_settings_summary': { attr_hist: { grp_hist: limit_hist for grp_hist, limit_hist in grp_limits_hist.items() if limit_hist is not None } for attr_hist, grp_limits_hist in results_budget_applied_info.get('settings', {}).items() if any(limit_hist is not None for limit_hist in grp_limits_hist.values()) }, 'proxy_enabled': run_proxy } }
        if isinstance(current_fairness, dict):
             for attr_f_hist, attr_data_f_hist in current_fairness.items():
                 if isinstance(attr_data_f_hist, dict) and 'overall_scores' in attr_data_f_hist:
                     scores_f_hist = attr_data_f_hist['overall_scores']; current_run_data['overall_scores'][attr_f_hist] = { 'dp_max_diff': scores_f_hist.get('dp_max_diff') if pd.notna(scores_f_hist.get('dp_max_diff')) else None, 'eo_overall_score': scores_f_hist.get('eo_overall_score') if pd.notna(scores_f_hist.get('eo_overall_score')) else None, 'pp_max_diff': scores_f_hist.get('pp_max_diff') if pd.notna(scores_f_hist.get('pp_max_diff')) else None,}
                 else: current_run_data['overall_scores'][attr_f_hist] = {'dp_max_diff': None, 'eo_overall_score': None, 'pp_max_diff': None}
        run_history.append(current_run_data); session['run_history'] = run_history
        logging.info(f"Added run {current_run_data['run_id']} to history.")

        if 'P_test_orig_df' in locals(): del P_test_orig_df
        if 'results_df' in locals(): del results_df
        if 'P_test_df' in locals(): del P_test_df
        if 'y_test' in locals(): del y_test
        if 'RN_test' in locals(): del RN_test
        if 'results_full_df' in locals(): del results_full_df
        if 'y_pred_series' in locals(): del y_pred_series
        gc.collect()
        logging.info(f"Stage 5 Done. Time: {time.time() - stage5_start:.2f} sec.")

        stage5_duration_perf = time.time() - stage5_start_time_perf # Includes test processing, prediction, fairness, pie
        mem_after_stage5_perf = process_perf.memory_info().rss / (1024 * 1024)
        logging.info(f"PERF_METRIC: Stage_5_PredictEvaluate, Duration={stage5_duration_perf:.4f}s, MemoryRSS={mem_after_stage5_perf:.2f}MB, MemChange={(mem_after_stage5_perf - mem_prev_stage_end_perf):.2f}MB")
        mem_prev_stage_end_perf = mem_after_stage5_perf


        session['prev_fairness'] = current_fairness
        session['prev_accuracy'] = current_accuracy
        session['prev_features'] = features
        session.modified = True
        logging.info("Stored current run results into session for next run's comparison.")

        # === STAGE 6: Prepare Template Context ===
        template_context = {
            'target_col': target_col,
            'protected_cols': protected_cols,
            'current_feature_cols': features,
            'num_selected_features': len(features),
            'analysis_cols_map': analysis_map,
            'model_config': current_model_config,
            'original_uploaded_model_type': original_model_type_from_upload,
            'all_columns': original_columns,
            'current_accuracy': current_accuracy,
            'test_set_size': test_set_size,
            'current_fairness_results': current_fairness,
            'metric_names': ALL_METRIC_NAMES,
            'proxy_analysis_results': proxy_analysis_results,
            'metrics_for_outliers': METRICS_FOR_OUTLIERS,
            'budget_applied_info': results_budget_applied_info,
            'group_importances_display': budget_info['group_importances'],
            'group_features_used': results_group_features_used,
            'pie_chart_data': pie_chart_data,
            'individual_details_map': individual_details_map,
            'outcome_polarity': outcome_polarity,
            'run_history': run_history,
            'actual_prev_fairness_results': actual_prev_fairness_data,
            'actual_prev_accuracy': actual_prev_accuracy
        }
        logging.info("Rendering results.html.")
        return render_template('results.html', **template_context)

    except (FileNotFoundError) as fnf_err:
        logging.error(f"File Not Found Error in /train: {fnf_err}", exc_info=True);
        flash(f"Error: A required data file could not be found: {fnf_err}", 'error'); return redirect(url_for('configure'))
    except (KeyError, ValueError, TypeError) as config_err:
        logging.error(f"Configuration or Data Processing Error in /train: {config_err}", exc_info=True);
        flash(f"Error during model training/evaluation setup or processing: {config_err}.", 'error'); return redirect(url_for('configure'))
    except MemoryError as mem_err:
        logging.error(f"Memory Error in /train: {mem_err}", exc_info=True);
        flash("Server ran out of memory during processing. Please try with a smaller dataset or simpler configuration.", "error");
        session.clear(); session.modified=True; return redirect(url_for('dashboard'))
    except Exception as e:
        logging.exception("!!! Unexpected critical error in /train !!!");
        flash(f"A critical server error occurred during training/evaluation: {e}. Your session has been cleared.", 'error');
        session.clear(); session.modified = True; return redirect(url_for('dashboard'))
    finally:
        # --- Final Cleanup ---
        if 'global_preprocessor' in locals() and global_preprocessor is not None: del global_preprocessor
        if 'global_model' in locals()and global_model is not None: del global_model
        vars_to_del_final = ['X_train_fit_processed', 'y_train_fit', 'P_train_fit_df',
                             'X_train_fit_orig', 'X_perm_orig', 'y_perm', 'P_perm_df',
                             'P_test_orig_df', 'results_full_df', 'results_df',
                             'y_test', 'P_test_df', 'RN_test', 'X_test_orig', 'X_test_processed_df',
                             'x_perm_processed_df_indexed', 'y_perm_values_indexed', 'p_perm_df_indexed']
        for var_name_final in vars_to_del_final:
             if var_name_final in locals() and locals()[var_name_final] is not None:
                 try: del locals()[var_name_final]
                 except NameError: pass
        if 'loaded_model_details_for_run' in locals() and loaded_model_details_for_run is not None:
            if 'pipeline_object' in loaded_model_details_for_run: del loaded_model_details_for_run['pipeline_object']
            del loaded_model_details_for_run
        final_gc_count = gc.collect()

        # --- Performance Tracking Finalization ---
        overall_duration_perf = time.time() - overall_start_time_perf
        mem_end_rss_perf = process_perf.memory_info().rss / (1024 * 1024) # in MB
        logging.info(f"PERF_METRIC: TrainEndpointEnd, Duration={overall_duration_perf:.4f}s, MemoryRSS={mem_end_rss_perf:.2f}MB, MemChangeTotal={(mem_end_rss_perf - mem_start_rss_perf):.2f}MB")
        logging.info(f"--- Exiting /train endpoint --- Final GC collected {final_gc_count} objects. Total Time: {time.time() - start_time:.2f} sec.")

# --- Function to Clear Uploads Folder ---
def clear_uploads_folder():
    """Removes files from the upload folder."""
    folder = app.config['UPLOAD_FOLDER'];
    if not os.path.isdir(folder): return
    items_deleted, items_failed = 0, 0
    try:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path); items_deleted += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path); items_deleted += 1
            except Exception as e:
                logging.error(f'Failed to delete {file_path}. Reason: {e}'); items_failed += 1
        if items_deleted > 0 or items_failed > 0: logging.info(f"Uploads folder cleanup: Deleted {items_deleted}, Failed {items_failed}.")
    except Exception as e:
        logging.error(f"Error listing files in uploads folder '{folder}': {e}")

# --- Main execution ---
if __name__ == '__main__':
    logging.info("Starting Flask application server...")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)