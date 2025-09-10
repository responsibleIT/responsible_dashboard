# apps/sustainability_dashboard/backend/src/websocket.py
import os
import json
import asyncio
import copy
from flask import request
import pandas as pd
import time
import threading
from contextlib import suppress
from flask_socketio import join_room  # <- no plain emit import
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

def detect_local_device() -> str:
    """
    Try to detect the local accelerator. Returns a human-friendly device name,
    or 'CPU' as a fallback. This function is backend-agnostic and safe.
    """
    # 1) Try PyTorch
    with suppress(Exception):
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            return torch.cuda.get_device_name(idx)

    # 2) Try TensorFlow
    with suppress(Exception):
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # name field can be verbose; return something readable
            return getattr(gpus[0], 'name', 'TensorFlow GPU')

    # 3) Try nvidia-smi directly
    with suppress(Exception):
        import subprocess, shlex
        out = subprocess.check_output(
            shlex.split("nvidia-smi --query-gpu=name --format=csv,noheader"),
            stderr=subprocess.DEVNULL, timeout=1.5
        )
        name = out.decode("utf-8", errors="ignore").strip().splitlines()[0]
        if name:
            return name

    return "CPU"

def sample_gpu_power_background(stop_evt: threading.Event, interval: float = 0.2):
    """
    If NVML is present, sample total board power (Watts) in the background.
    Returns a list that will be populated with float readings.
    If NVML is not available, returns an empty list and does nothing.
    """
    readings: list[float] = []

    def _runner():
        with suppress(Exception):
            import pynvml  # pip install nvidia-ml-py3
            pynvml.nvmlInit()
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                while not stop_evt.is_set():
                    mw = pynvml.nvmlDeviceGetPowerUsage(h)  # milliwatts
                    readings.append(mw / 1000.0)  # -> Watts
                    stop_evt.wait(interval)
            finally:
                with suppress(Exception):
                    pynvml.nvmlShutdown()

    # try to start; if import fails, thread will never add data
    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return readings, t

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

        # respond to just this client if upload_id missing
        if not upload_id:
            socketio.emit(
                "status",
                {"type": "error", "message": "No upload_id provided. Please retry."},
                to=request.sid
            )
            return

        upload_path = get_upload_path(upload_id)
        if not os.path.exists(upload_path):
            socketio.emit(
                "status",
                {"type": "error", "message": f"Upload {upload_id} not found on server."},
                to=request.sid
            )
            return

        def process():
            try:
                socketio.emit("status", {"message": "Model is being loaded..."}, to=upload_id)

                # Load dataset and read chosen target column
                df = load_dataset(upload_id)
                # You persist selected columns as selected_columns.json in /upload
                with suppress(Exception):
                    cols = load_json_file(upload_id, "selected_columns.json")
                    target_col = cols.get("target_column")
                if not target_col or target_col not in df.columns:
                    socketio.emit(
                        "status",
                        {"type": "error", "message": "Target column not found in dataset."},
                        to=upload_id
                    )
                    return  # <-- never use exit()

                # Build label mapping from actual labels
                label_mapping = infer_label_mapping(df[target_col])

                # Model source (HF URL preferred, else local)
                hf_url = get_huggingface_url(upload_id)
                local_model_path = find_local_model(upload_id)
                if hf_url:
                    model, tokenizer = load_huggingface_model(hf_url)
                    model_name_for_logs = hf_url
                elif local_model_path:
                    model, tokenizer = load_local_model(local_model_path)
                    model_name_for_logs = os.path.basename(local_model_path)
                else:
                    socketio.emit(
                        "status",
                        {"type": "error",
                        "message": "No model provided. Enter a Hugging Face repo or upload a .h5/.keras file."},
                        to=upload_id
                    )
                    return

                # Initial prune at 10% for two baselines
                model_copy = copy.deepcopy(model)
                pruned_model, model_info = disable_low_weight_neurons(model_copy, 10)

                socketio.emit("status", {"message": "Benchmarking model..."}, to=upload_id)
                baseline = evaluate_model(model, tokenizer, df, target_col=target_col)
                pruned = evaluate_model(pruned_model, tokenizer, df, target_col=target_col)

                socketio.emit("status", {"message": "Collecting pruning data..."}, to=upload_id)
                pruned_data = {
                    0:  create_baseline_metrics(baseline, model_info, label_mapping, 0),
                    10: create_baseline_metrics(pruned,   model_info, label_mapping, 10),
                }

                # Sweep thresholds
                for t in THRESHOLDS:
                    t = round(t, 1)
                    m_copy = copy.deepcopy(model)
                    _p_model, metrics = disable_low_weight_neurons(m_copy, t)
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
    
    @socketio.on('benchmark_real')
    def handle_benchmark_real(data):
        """
        Run a *real* benchmark on this machine:
        - load model (HF or local)
        - prune at requested threshold
        - time a real evaluation on the uploaded dataset
        - optionally sample real GPU power with NVML
        - persist results under 'realBenchmark' in benchmark_data.json
        """
        upload_id = data.get("upload_id")
        threshold = data.get("threshold")

        async def process():
            try:
                socketio.emit("status", {"message": "Preparing real benchmark..."}, to=upload_id)

                # 1) Load dataset & columns
                df = load_dataset(upload_id)
                # load selected columns if you are using the new target-only flow
                with suppress(Exception):
                    cols = load_json_file(upload_id, "selected_columns.json")
                    target_col = cols.get("target_column")
                    if target_col and target_col in df.columns:
                        # Evaluate path already expects text/label names. If your evaluator
                        # relies on specific names, rename here; otherwise you can skip this.
                        pass

                # 2) Load model (HF first, else local)
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
                        "message": "No model available. Provide a HF repo or upload a model."
                    }, to=upload_id)
                    return

                # 3) Prune to requested threshold
                pruned_model, model_info = disable_low_weight_neurons(model, threshold)

                # 4) Time a *real* evaluation + optional GPU power sampling
                socketio.emit("status", {"message": "Running real benchmark..."}, to=upload_id)
                stop_evt = threading.Event()
                power_readings, power_thread = sample_gpu_power_background(stop_evt, interval=0.25)

                t0 = time.perf_counter()
                real_metrics = evaluate_model(pruned_model, tokenizer, df, target_col=target_col)
                elapsed = time.perf_counter() - t0
                stop_evt.set()
                power_thread.join(timeout=0.5)

                # Aggregate power (if any)
                avg_watts = sum(power_readings) / len(power_readings) if power_readings else None
                energy_j = (avg_watts * elapsed) if avg_watts is not None else None

                # 5) Merge into benchmark_data.json (append 'realBenchmark')
                bench_path = os.path.join(get_upload_path(upload_id), "benchmark_data.json")
                base_payload = {}
                if os.path.exists(bench_path):
                    with open(bench_path, "r", encoding="utf-8") as f:
                        base_payload = json.load(f)

                base_payload.setdefault("model", model_id_for_benchmark)
                base_payload.setdefault("threshold", threshold)

                base_payload["realBenchmark"] = {
                    "device": detect_local_device(),
                    "elapsedSec": elapsed,
                    "samples": int(len(df)),
                    "throughputSamplesPerSec": (len(df) / elapsed) if elapsed > 0 else None,
                    "avgGpuPowerW": avg_watts,       # may be None if NVML unavailable
                    "energyJoules": energy_j,        # may be None
                    "metrics": real_metrics.get("overall", {})  # accuracy/precision/recall/f1 from real run
                }

                with open(bench_path, "w", encoding="utf-8") as f:
                    json.dump(base_payload, f, indent=2)

                socketio.emit("status", {
                    "message": "Real benchmark completed",
                    "type": "complete"
                }, to=upload_id)

            except Exception as e:
                socketio.emit("status", {
                    "type": "error",
                    "message": f"Real benchmark failed: {e}"
                }, to=upload_id)

        socketio.start_background_task(process)

