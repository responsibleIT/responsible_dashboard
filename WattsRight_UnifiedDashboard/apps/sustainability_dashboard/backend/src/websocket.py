import os
import json
import asyncio
import copy
import pandas as pd
from flask_socketio import emit, join_room, leave_room
from dotenv import load_dotenv

from loading import load_huggingface_model, load_local_model
from preprocess import disable_low_weight_neurons
from benchmark import evaluate_model
from predict import predict_with_auto_regressive_model

import glob

load_dotenv()

DEMO_MODE = "false" #os.getenv("DEMO", "false").lower() == "true"
UPLOAD_DIR = "uploads"
THRESHOLDS = [i * 0.1 for i in range(1, 100)]
LABEL_MAPPING = {0: 'Ham', 1: 'Spam'}


# --- Helper Functions ---
def find_local_model(upload_id: str) -> str | None:
    """Return the first local model file path (.h5 or .keras) in this upload, if any."""
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

def create_baseline_metrics(metrics, model_info, threshold=0):
    return {
        "accuracy": metrics['overall']['accuracy'],
        "precision": metrics['overall']['precision'],
        "recall": metrics['overall']['recall'],
        "f1_score": metrics['overall']['f1_score'],
        "per_class": {
            LABEL_MAPPING[label]: {
                "accuracy": metrics[label]['accuracy'],
                "precision": metrics[label]['precision'],
                "recall": metrics[label]['recall'],
                "f1_score": metrics[label]['f1_score']
            } for label in metrics if label != 'overall'
        },
        "flops": model_info['original']['flops_estimate'] if threshold == 0 else model_info['after_pruning']['flops_estimate'],
        "non_zero_params": model_info['original']['non_zero_params'] if threshold == 0 else model_info['after_pruning']['non_zero_params'],
        "params_reduction_pct": model_info['after_pruning']['params_reduction_pct'],
        "flops_reduction_pct": model_info['after_pruning']['flops_reduction_pct']
    }

def create_threshold_data_entry(metrics, threshold):
    return {
        "accuracy": 0,
        "flops": metrics['after_pruning']['flops_estimate'],
        "non_zero_params": metrics['after_pruning']['non_zero_params'],
        "params_reduction_pct": metrics['after_pruning']['params_reduction_pct'],
        "flops_reduction_pct": metrics['after_pruning']['flops_reduction_pct']
    }

def create_benchmark_data(hf_url, threshold, gpu, location, benchmark, model_info, pruned_data):
    data = {
        "model": hf_url,
        "threshold": threshold,
        "gpu": gpu,
        "location": location,
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

    for label, metrics in benchmark.items():
        if label == 'overall': continue
        label_name = LABEL_MAPPING.get(label, label)
        data['perClass'][label_name] = {
            metric: {
                "original": pruned_data['0']['per_class'][label_name][metric],
                "pruned": metrics[metric]
            } for metric in ['accuracy', 'precision', 'recall']
        }
        data['perClass'][label_name]['f1Score'] = {
            "original": pruned_data['0']['per_class'][label_name]['f1_score'],
            "pruned": metrics['f1_score']
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
        join_room(upload_id)
        emit("status", {"type": "connection", "status": "connected", "upload_id": upload_id})

    @socketio.on('start')
    def handle_start(data):
        upload_id = data.get("upload_id")

        async def process():
            try:
                emit("status", {"message": "Model is being loaded..."}, to=upload_id)

                # Load dataset (expects uploads/<id>/dataset.csv)
                df = load_dataset(upload_id)

                # Decide model source: Hugging Face URL first, else local model file
                huggingface_url = get_huggingface_url(upload_id)
                local_model_path = find_local_model(upload_id)

                if huggingface_url:
                    model, tokenizer = load_huggingface_model(huggingface_url)
                    model_name_for_logs = huggingface_url
                elif local_model_path:
                    model, tokenizer = load_local_model(local_model_path)
                    model_name_for_logs = os.path.basename(local_model_path)
                else:
                    emit("status", {
                        "type": "error",
                        "message": "No model provided. Enter a Hugging Face repo or upload a .h5/.keras file."
                    }, to=upload_id)
                    return

                # Prune a copy at 10% for the initial pair of baselines
                model_copy = copy.deepcopy(model)
                pruned_model, model_info = disable_low_weight_neurons(model_copy, 10)

                emit("status", {"message": "Benchmarking model..."}, to=upload_id)
                baseline = evaluate_model(model, tokenizer, df)
                pruned = evaluate_model(pruned_model, tokenizer, df)

                emit("status", {"message": "Collecting pruning data..."}, to=upload_id)
                pruned_data = {
                    0: create_baseline_metrics(baseline, model_info, 0),
                    10: create_baseline_metrics(pruned, model_info, 10)
                }

                # Sweep thresholds (0.1 .. 9.9 as defined in THRESHOLDS)
                for t in THRESHOLDS:
                    t = round(t, 1)
                    m_copy = copy.deepcopy(model)
                    p_model, metrics = disable_low_weight_neurons(m_copy, t)
                    pruned_data[t] = create_threshold_data_entry(metrics, t)

                emit("status", {"message": "Predicting performance..."}, to=upload_id)
                pruned_data = predict_with_auto_regressive_model(pruned_data, "accuracy")

                # Persist for the validate step
                save_json_file(upload_id, "pruned_threshold_data.json", pruned_data)

                emit("status", {
                    "message": f"Benchmark completed successfully for {model_name_for_logs}",
                    "type": "complete"
                }, to=upload_id)

            except Exception as e:
                emit("status", {"type": "error", "message": f"Start failed: {e}"}, to=upload_id)

        socketio.start_background_task(process)

    @socketio.on('validate')
    def handle_validate(data):
        upload_id = data.get("upload_id")
        threshold = data.get("threshold")
        gpu = data.get("gpu")
        location = data.get("location")

        async def process():
            try:
                emit("status", {"message": "Model is being loaded..."}, to=upload_id)

                # Load dataset
                df = load_dataset(upload_id)

                # Decide model source again (same rule as in start)
                huggingface_url = get_huggingface_url(upload_id)
                local_model_path = find_local_model(upload_id)

                if huggingface_url:
                    model, tokenizer = load_huggingface_model(huggingface_url)
                    model_id_for_benchmark = huggingface_url
                elif local_model_path:
                    model, tokenizer = load_local_model(local_model_path)
                    model_id_for_benchmark = os.path.basename(local_model_path)
                else:
                    emit("status", {
                        "type": "error",
                        "message": "No model available to validate. Run the start step with a HF URL or upload a model."
                    }, to=upload_id)
                    return

                # Apply chosen threshold and evaluate
                pruned_model, model_info = disable_low_weight_neurons(model, threshold)

                emit("status", {"message": "Benchmarking model..."}, to=upload_id)
                benchmark = evaluate_model(pruned_model, tokenizer, df)

                # Load predictions from start step
                pruned_data = load_json_file(upload_id, "pruned_threshold_data.json")

                # Build and save full benchmark payload
                benchmark_data = create_benchmark_data(
                    model_id_for_benchmark, threshold, gpu, location,
                    benchmark, model_info, pruned_data
                )
                save_json_file(upload_id, "benchmark_data.json", benchmark_data)

                emit("status", {"message": "Validation completed successfully", "type": "complete"}, to=upload_id)

            except FileNotFoundError as e:
                emit("status", {
                    "type": "error",
                    "message": f"Missing file: {e}. Make sure dataset.csv exists and that you ran the start step."
                }, to=upload_id)
            except Exception as e:
                emit("status", {"type": "error", "message": f"Validate failed: {e}"}, to=upload_id)

        socketio.start_background_task(process)
