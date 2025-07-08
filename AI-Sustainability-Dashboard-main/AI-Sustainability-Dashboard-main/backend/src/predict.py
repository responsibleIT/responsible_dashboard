import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

MODEL_FILE_PATH = Path("model/lstm_model_4.keras")
X_SCALER_PATH = Path("model/x_dynamic_scaler.pkl")
Y_SCALER_PATH = Path("model/y_acc_scaler.pkl")

INTERMEDIATE_THRESHOLDS = np.round(np.arange(0.1, 10.0, 0.1), 1)
SEQUENCE_LENGTH = len(INTERMEDIATE_THRESHOLDS)

dynamic_feature_cols = ['flops', 'non_zero_params', 'params_reduction_pct', 'flops_reduction_pct']


# --- Helper Function to Prepare Inputs for Prediction from Dictionary ---
def prepare_prediction_inputs_from_dict(
    pruned_data_dict,
    dynamic_cols_list,
    intermediate_thresholds_arr,
    perf_metric_key="performance_metric"
):
    """
    Prepares unscaled dynamic features, acc_0, and acc_10 from the
    pruned_data_dict for a single model instance.
    """
    # Expecting integer keys 0 and 10 for acc_0 and acc_10 as per user's example
    if 0 not in pruned_data_dict: # Check for int key 0
        raise KeyError(f"Key 0 (for {perf_metric_key} at threshold 0.0) not found in pruned_data_dict.")
    if 10 not in pruned_data_dict: # Check for int key 10
        raise KeyError(f"Key 10 (for {perf_metric_key} at threshold 10.0) not found in pruned_data_dict.")

    acc_0_val = pruned_data_dict[0][perf_metric_key]
    acc_10_val = pruned_data_dict[10][perf_metric_key]

    dynamic_features_list = []
    for t_val in intermediate_thresholds_arr: # t_val will be 0.1, 0.2, ... (floats)
        # The intermediate_thresholds_arr contains rounded floats (e.g., 0.1, 0.2)
        # These should directly match the keys used when populating pruned_data_dict for these thresholds.
        if t_val not in pruned_data_dict:
            raise ValueError(f"Threshold {t_val:.1f} not found in pruned_data_dict for intermediate points.")

        current_features = []
        for feat_col in dynamic_cols_list:
            if feat_col not in pruned_data_dict[t_val]:
                raise KeyError(f"Dynamic feature '{feat_col}' not found for threshold {t_val:.1f} in pruned_data_dict.")
            current_features.append(pruned_data_dict[t_val][feat_col])
        dynamic_features_list.append(current_features)

    unscaled_dynamic_features_sample = np.array(dynamic_features_list)

    if unscaled_dynamic_features_sample.shape[0] != len(intermediate_thresholds_arr):
        raise ValueError(f"Mismatch in sequence length for dynamic features. Expected {len(intermediate_thresholds_arr)}, got {unscaled_dynamic_features_sample.shape[0]}.")
    if unscaled_dynamic_features_sample.shape[1] != len(dynamic_cols_list):
        raise ValueError(f"Mismatch in number of dynamic features. Expected {len(dynamic_cols_list)}, got {unscaled_dynamic_features_sample.shape[1]}.")

    return unscaled_dynamic_features_sample, acc_0_val, acc_10_val


# --- Auto-Regressive Prediction Function (WITH acc_10 input) ---
# This function remains unchanged from your original script
def predict_auto_regressively_with_acc10(model_to_use,
                                         unscaled_dynamic_features_sample, # Shape: (seq_len, num_dyn_feat)
                                         unscaled_acc_0,
                                         unscaled_acc_10,
                                         x_dynamic_scaler_model, y_acc_scaler_model,
                                         seq_len, num_dyn_feat):
    
    try:

        # Scale dynamic features
        scaled_dynamic_features = x_dynamic_scaler_model.transform(unscaled_dynamic_features_sample)

        # Scale acc_0 and acc_10 (which are scalar inputs to the sequence)
        scaled_acc_0_val = y_acc_scaler_model.transform(np.array([[unscaled_acc_0]]))[0,0]
        scaled_acc_10_val = y_acc_scaler_model.transform(np.array([[unscaled_acc_10]]))[0,0]

        # Prepare repeated features for acc_0 and acc_10
        scaled_acc_0_feat_repeated = np.tile(scaled_acc_0_val, (seq_len, 1))
        scaled_acc_10_feat_repeated = np.tile(scaled_acc_10_val, (seq_len, 1))

        # Initialize input tensor for the model
        # Features: dynamic_features, acc_0_repeated, acc_10_repeated, lagged_predicted_accuracy
        current_input_X = np.zeros((1, seq_len, num_dyn_feat + 3)) # +3 for acc0, acc10, lagged_acc

        # Populate known parts of the input tensor
        current_input_X[0, :, :num_dyn_feat] = scaled_dynamic_features
        current_input_X[0, :, num_dyn_feat] = scaled_acc_0_feat_repeated[:,0]
        current_input_X[0, :, num_dyn_feat + 1] = scaled_acc_10_feat_repeated[:,0]

        # Auto-regressive prediction loop
        last_predicted_acc_scaled = scaled_acc_0_val # Start with scaled acc_0 as the first "lagged" input
        generated_sequence_scaled = np.zeros(seq_len)

        for t in range(seq_len):
            # Set the lagged accuracy feature for the current step
            current_input_X[0, t, num_dyn_feat + 2] = last_predicted_acc_scaled
            
            # Predict the full sequence, but we only need the prediction at step 't'
            # Model output shape: (batch_size, seq_len, num_output_features_per_step)
            # Assuming model outputs 1 feature (accuracy) per step
            prediction_all_steps_scaled = model_to_use.predict(current_input_X, verbose=0)
            current_step_pred_scaled = prediction_all_steps_scaled[0, t, 0]
            
            generated_sequence_scaled[t] = current_step_pred_scaled
            last_predicted_acc_scaled = current_step_pred_scaled # Update for next iteration
            
        # Inverse transform the generated sequence to get unscaled accuracy values
        return y_acc_scaler_model.inverse_transform(generated_sequence_scaled.reshape(-1,1)).flatten()
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise
    



def predict_with_auto_regressive_model(input_pruned_data, performance_metric_key="performance_metric"):
    try:
        model = load_model(MODEL_FILE_PATH)

        with open(X_SCALER_PATH, 'rb') as f:
            x_dynamic_scaler = pickle.load(f)

        with open(Y_SCALER_PATH, 'rb') as f:
            y_acc_scaler = pickle.load(f)

    except Exception as e:
        print(f"Error: {e}")
        exit()


    try:
        unscaled_dyn_features_sample, unscaled_acc0, unscaled_acc10 = \
            prepare_prediction_inputs_from_dict(
                input_pruned_data,
                dynamic_feature_cols,
                INTERMEDIATE_THRESHOLDS,
                performance_metric_key
            )
        num_dynamic_features_actual = unscaled_dyn_features_sample.shape[1]
        if num_dynamic_features_actual != len(dynamic_feature_cols):
            print(f"Warning: Number of dynamic features from data ({num_dynamic_features_actual}) "
                    f"does not match config ({len(dynamic_feature_cols)}). Using data-derived count.")
        print(f"Successfully prepared inputs: acc_0={unscaled_acc0:.4f}, acc_10={unscaled_acc10:.4f}, "
                f"dynamic features shape={unscaled_dyn_features_sample.shape}")

    except Exception as e:
        print(f"Error preparing inputs: {e}")
        exit()

    predicted_intermediate_accuracies_unscaled = predict_auto_regressively_with_acc10(
        model,
        unscaled_dyn_features_sample,
        unscaled_acc0,
        unscaled_acc10,
        x_dynamic_scaler, y_acc_scaler,
        SEQUENCE_LENGTH, num_dynamic_features_actual
    )

    for i, threshold_val in enumerate(INTERMEDIATE_THRESHOLDS):
        input_pruned_data[threshold_val][performance_metric_key] = predicted_intermediate_accuracies_unscaled[i]


    # Return a dictionary with the updated accuracies including the 0.0 and 10.0 thresholds
    input_pruned_data[0][performance_metric_key] = unscaled_acc0
    input_pruned_data[10][performance_metric_key] = unscaled_acc10
    
    return input_pruned_data