# apps/sustainability_dashboard/backend/src/websocket.py
import os
import json
import asyncio
import copy
import pandas as pd
from flask_socketio import join_room, leave_room  # <- no plain emit import
from dotenv import load_dotenv

from loading import load_huggingface_model, load_local_model
from preprocess import disable_low_weight_neurons
from benchmark import evaluate_model
from predict import predict_with_auto_regressive_model

import glob

load_dotenv()

DEMO_MODE = "false"
UPLOAD_DIR = "uploads"
THRESHOLDS = [i * 0.1 for i in range(1, 100)]

# --- Helper Functions ---
def find_local_model(upload_id: str) -> str | None:
    base = get_upload_path(upload_id)
    for ext in (".h5", ".keras"):
        matches = glob.glob(os.path.join(base, f"*{ext}"))
        if matches:
            return matches[0]
    return None

def get_upload_path(upload_id):
    return os.path.join(UPLOAD_DIR, upload_id)

def get_dataset_path(upload_id):
    return os.path.join(get_upload_path(upload_id), "dataset.csv")

def load_dataset(upload_id):
    return pd.read_csv(get_dataset_path(upload_id))

def save_json_file(upload_id, filename, data):
    path = os.path.join(get_upload_path(upload_id), filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json_file(upload_id, filename):
    path = os.path.join(get_upload_path(upload_id), filename)
    with open(path, 'r') as f:
        return json.load(f)

def has_huggingface_url(upload_id):
    path = os.path.join(get_upload_path(upload_id), "huggingface_url.txt")
    return os.path.exists(path) and open(path).read().strip().lower() != 'none'

def get_huggingface_url(upload_id):
    path = os.path.join(get_upload_path(upload_id), "huggingface_url.txt")
    if not os.path.exists(path):
        return None
    return open(path).read().strip()

def load_selected_columns(upload_id) -> tuple[str, str | None]:
    meta_path = os.path.join(get_upload_path(upload_id), "selected_columns.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            data = json.load(f)
        # only target is required now
        return data.get("text_column"), data.get("target_column")
    # fallback for older flows
    flag_path = os.path.join(get_upload_path(upload_id), "flag.json")
    if os.path.exists(flag_path):
        with open(flag_path, "r") as f:
            data = json.load(f)
        return data.get("text_column"), data.get("target_column")
    return None, None

def infer_label_mapping(series: pd.Series) -> dict[int, str]:
    uniques = list(pd.unique(series))
    try:
        uniques_sorted = sorted(uniques)
    except Exception:
        uniques_sorted = uniques
    return {i: str(uniques_sorted[i]) for i in range(len(uniques_sorted))}

def create_baseline_metrics(metrics, model_info, label_mapping: dict[int, str], threshold=0):
    return {
        "accuracy": metrics['overall']['accuracy'],
        "precision": metrics['overall']['precision'],
        "recall": metrics['overall']['recall'],
        "f1_score": metrics['overall']['f1_score'],
        "per_class": {
            label_mapping[label]: {
                "accuracy": metrics[label]['accuracy'],
                "precision": metrics[label]['precision'],
                "recall": metrics[label]['recall'],
                "f1_score": metrics[label]['f1_score'],
            } for label in metrics if label != 'overall'
        },
        "flops": model_info['original']['flops_estimate'] if threshold == 0 else model_info['after_pruning']['flops_estimate'],
        "non_zero_params": model_info['original']['non_zero_params'] if threshold == 0 else model_info['after_pruning']['non_zero_params'],
        "params_reduction_pct": model_info['after_pruning']['params_reduction_pct'],
        "flops_reduction_pct": model_info['after_pruning']['flops_reduction_pct'],
    }

def create_threshold_data_entry(metrics, threshold):
    return {
        "accuracy": 0,
        "flops": metrics['after_pruning']['flops_estimate'],
        "non_zero_params": metrics['after_pruning']['non_zero_params'],
        "params_reduction_pct": metrics['after_pruning']['params_reduction_pct'],
        "flops_reduction_pct": metrics['after_pruning']['flops_reduction_pct'],
    }

def create_benchmark_data(hf_url, threshold, benchmark, model_info, pruned_data, label_mapping: dict[int, str]):
    data = {
        "model": hf_url,
        "threshold": threshold,
        "overall": {},
        "perClass": {},
        "originalFlops": model_info['original']['flops_estimate'],
        "prunedFlops": model_info['after_pruning']['flops_estimate'],
        "reductionPercentage": model_info['after_pruning']['params_reduction_pct'],
        "originalParameters": model_info['original']['non_zero_params'],
        "prunedParameters": model_info['after_pruning']['non_zero_params'],
    }

    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        data['overall'][metric] = {
            "original": pruned_data['0'][metric],
            "pruned": benchmark['overall'][metric],
        }

    for label, m in benchmark.items():
        if label == 'overall':
            continue
        name = label_mapping.get(label, str(label))
        data['perClass'][name] = {
            k: {
                "original": pruned_data['0']['per_class'][name][k],
                "pruned": m[k],
            } for k in ['accuracy', 'precision', 'recall']
        }
        data['perClass'][name]['f1Score'] = {
            "original": pruned_data['0']['per_class'][name]['f1_score'],
            "pruned": m['f1_score'],
        }
    return data


# --- Main WebSocket Handler ---
def websocket_handlers(socketio):

    @socketio.on('connect')
    def handle_connect():
        print("Client connected")

    @socketio.on('disconnect')
    def handle_disconnect():
        print("Client disconnected")

    @socketio.on('join')
    def on_join(data):
        upload_id = data.get("upload_id")
        if upload_id:
            join_room(upload_id)
        # acknowledge to this client
        socketio.emit("status", {"type": "connection", "status": "connected", "upload_id": upload_id}, to=upload_id)

    @socketio.on('start')
    def handle_start(data):
        upload_id = data.get("upload_id")

        def process():
            try:
                socketio.emit("status", {"message": "Model is being loaded..."}, to=upload_id)

                # Load dataset and selected columns (target only required)
                df = load_dataset(upload_id)
                _, target_col = load_selected_columns(upload_id)
                if target_col and target_col in df.columns:
                    df = df.rename(columns={target_col: 'label'})
                # Build label mapping
                label_mapping = infer_label_mapping(df['label'])

                # Model source
                hf_url = get_huggingface_url(upload_id)
                local_model_path = find_local_model(upload_id)

                if hf_url:
                    model, tokenizer = load_huggingface_model(hf_url)
                    model_name_for_logs = hf_url
                elif local_model_path:
                    model, tokenizer = load_local_model(local_model_path)
                    model_name_for_logs = os.path.basename(local_model_path)
                else:
                    socketio.emit("status", {
                        "type": "error",
                        "message": "No model provided. Enter a Hugging Face repo or upload a .h5/.keras file."
                    }, to=upload_id)
                    return

                # Initial prune at 10%
                model_copy = copy.deepcopy(model)
                pruned_model, model_info = disable_low_weight_neurons(model_copy, 10)

                socketio.emit("status", {"message": "Benchmarking model..."}, to=upload_id)
                baseline = evaluate_model(model, tokenizer, df)
                pruned = evaluate_model(pruned_model, tokenizer, df)

                socketio.emit("status", {"message": "Collecting pruning data..."}, to=upload_id)
                pruned_data = {
                    0: create_baseline_metrics(baseline, model_info, label_mapping, 0),
                    10: create_baseline_metrics(pruned, model_info, label_mapping, 10),
                }

                for t in THRESHOLDS:
                    t = round(t, 1)
                    m_copy = copy.deepcopy(model)
                    p_model, metrics = disable_low_weight_neurons(m_copy, t)
                    pruned_data[t] = create_threshold_data_entry(metrics, t)

                socketio.emit("status", {"message": "Predicting performance..."}, to=upload_id)
                pruned_data = predict_with_auto_regressive_model(pruned_data, "accuracy")

                save_json_file(upload_id, "pruned_threshold_data.json", pruned_data)
                save_json_file(upload_id, "label_mapping.json", label_mapping)

                socketio.emit("status", {
                    "message": f"Benchmark completed successfully for {model_name_for_logs}",
                    "type": "complete"
                }, to=upload_id)

            except Exception as e:
                socketio.emit("status", {"type": "error", "message": f"Start failed: {e}"}, to=upload_id)

        socketio.start_background_task(process)

    @socketio.on('validate')
    def handle_validate(data):
        upload_id = data.get("upload_id")
        threshold = data.get("threshold")

        def process():
            try:
                socketio.emit("status", {"message": "Model is being loaded..."}, to=upload_id)

                df = load_dataset(upload_id)
                _, target_col = load_selected_columns(upload_id)
                if target_col and target_col in df.columns:
                    df = df.rename(columns={target_col: 'label'})

                try:
                    label_mapping = load_json_file(upload_id, "label_mapping.json")
                except Exception:
                    label_mapping = infer_label_mapping(df['label'])

                hf_url = get_huggingface_url(upload_id)
                local_model_path = find_local_model(upload_id)

                if hf_url:
                    model, tokenizer = load_huggingface_model(hf_url)
                    model_id_for_benchmark = hf_url
                elif local_model_path:
                    model, tokenizer = load_local_model(local_model_path)
                    model_id_for_benchmark = os.path.basename(local_model_path)
                else:
                    socketio.emit("status", {
                        "type": "error",
                        "message": "No model available to validate. Run the start step first."
                    }, to=upload_id)
                    return

                pruned_model, model_info = disable_low_weight_neurons(model, threshold)

                socketio.emit("status", {"message": "Benchmarking model..."}, to=upload_id)
                benchmark = evaluate_model(pruned_model, tokenizer, df)

                pruned_data = load_json_file(upload_id, "pruned_threshold_data.json")

                benchmark_data = create_benchmark_data(
                    model_id_for_benchmark, threshold, benchmark, model_info, pruned_data, label_mapping
                )
                save_json_file(upload_id, "benchmark_data.json", benchmark_data)

                socketio.emit("status", {"message": "Validation completed successfully", "type": "complete"}, to=upload_id)

            except FileNotFoundError as e:
                socketio.emit("status", {
                    "type": "error",
                    "message": f"Missing file: {e}. Make sure dataset.csv exists and that you ran the start step."
                }, to=upload_id)
            except Exception as e:
                socketio.emit("status", {"type": "error", "message": f"Validate failed: {e}"}, to=upload_id)

        socketio.start_background_task(process)
