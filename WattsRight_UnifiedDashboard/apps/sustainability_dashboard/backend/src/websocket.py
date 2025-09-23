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
from pruning import estimate_flops
from benchmark import evaluate_model
from predict import predict_with_auto_regressive_model
from utils.gpu_power import sample_gpu_power_background

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

# --- Main WebSocket Handler ---
def websocket_handlers(socketio):
    def _emit_progress(upload_id: str, done: int, total: int, started_at: float):
        elapsed = time.perf_counter() - started_at
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else None
        socketio.emit("status", {
            "message": f"Evaluating {done}/{total} samples...",
            "eta": eta
        }, to=upload_id)

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

        socketio.emit("status", {"message": f"[START DEBUG] Reading from: {upload_path}"}, to=upload_id)
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

                started_at = time.perf_counter()
                def progress_cb(done, total):
                    _emit_progress(upload_id, done, total, started_at)

                baseline = evaluate_model(model, tokenizer, df, target_col=target_col,
                          progress_cb=progress_cb)
                pruned = evaluate_model(pruned_model, tokenizer, df, target_col=target_col,
                                        progress_cb=progress_cb)

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
                try:
                    pred = predict_with_auto_regressive_model(pruned_data, "accuracy")
                    if isinstance(pred, dict) and pred:
                        pruned_data = pred
                    # else keep the baseline-filled pruned_data we already built
                except Exception:
                    # swallow prediction errors and keep baseline-only curves
                    pass

                save_json_file(upload_id, "pruned_threshold_data.json", pruned_data)
                socketio.emit("status", {"message": f"[START DEBUG] Saved pruned data to {os.path.join(get_upload_path(upload_id), 'pruned_threshold_data.json')}"}, to=upload_id)
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
                started_at = time.perf_counter()

                def progress_cb(done, total):
                    _emit_progress(upload_id, done, total, started_at)

                benchmark = evaluate_model(pruned_model, tokenizer, df, progress_cb=progress_cb)

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
    
    @socketio.on("benchmark_real")
    def handle_benchmark_real(data):
        """
        ORIGINAL -> PRUNE -> PRUNED
        Builds metricCards (kWh & gCO2 per 1000 calls, accuracy %, TFLOPs per call),
        plus rich per-class and realBenchmark blocks. Extremely chatty debug emits.
        """
        import os, json, time, threading, traceback
        from flask import request

        # local import so this stays drop-in
        try:
            from pruning import disable_low_weight_neurons, estimate_flops
        except Exception:
            # disable_low_weight_neurons may already be imported at module level
            disable_low_weight_neurons  # type: ignore

            def estimate_flops(_model, seq_length: int = 128):
                # minimal safe fallback: no FLOPs estimate available
                raise RuntimeError("estimate_flops not available")

        upload_id = (data or {}).get("upload_id")
        try:
            threshold = float((data or {}).get("threshold", 0) or 0.0)
        except Exception:
            threshold = 0.0

        def _emit(msg, typ="debug", extra=None, to_room=None):
            payload = {"type": typ, "message": msg}
            if extra is not None:
                payload["extra"] = extra
            room = to_room or upload_id or request.sid
            try:
                socketio.emit("status", payload, to=room)
            except Exception:
                pass
            print(msg)

        def _pair(a, b):
            # always numbers (never None) so Angular can render safely
            def _num(x):
                try:
                    if x is None: return 0.0
                    # cast numpy types too
                    return float(x)
                except Exception:
                    return 0.0
            return {"original": _num(a), "pruned": _num(b)}

        def _percent(x):
            # 0..1 -> 0..100 ; keep numbers
            try:
                return float(x) * 100.0
            except Exception:
                return 0.0

        def _grid_factor_g_per_kwh(loc: str) -> float:
            # quick-and-safe factors; adjust as you like
            # sources vary—these are ballpark so the card has data
            table = {
                "france": 50.0,
                "netherlands": 475.0,
                "germany": 400.0,
                "uk": 212.0,
                "usa": 386.0,
                "australia": 700.0,
            }
            return table.get((loc or "").lower(), 400.0)

        def process():
            try:
                if not upload_id:
                    _emit("[DEBUG] Missing upload_id for real benchmark.", "error", to_room=request.sid)
                    return

                _emit(f"[DEBUG] Starting benchmark_real | upload_id={upload_id} | threshold={threshold}")

                # 1) DATASET ------------------------------------------------------
                try:
                    df = load_dataset(upload_id)
                    _emit(f"[DEBUG] Dataset loaded: shape={getattr(df, 'shape', None)}; "
                        f"columns={list(getattr(df, 'columns', []))}")
                except Exception as e:
                    _emit(f"[DEBUG] Failed to load dataset: {e}", "error")
                    _emit(traceback.format_exc(), "error")
                    return

                try:
                    sel = load_json_file(upload_id, "selected_columns.json") or {}
                except Exception:
                    sel = {}
                target_col = sel.get("target_column")
                if not target_col or target_col not in getattr(df, "columns", []):
                    target_col = list(getattr(df, "columns", []))[-1] if getattr(df, "columns", []) else None
                if not target_col:
                    _emit("[DEBUG] No target column found.", "error")
                    return
                _emit(f"[DEBUG] Target column resolved: {target_col}")

                samples = int(len(df))

                # 2) MODEL --------------------------------------------------------
                hf_url = None
                local_model_path = None
                try:
                    hf_url = get_huggingface_url(upload_id)
                except Exception:
                    pass
                try:
                    local_model_path = find_local_model(upload_id)
                except Exception:
                    pass

                try:
                    if hf_url:
                        model, tokenizer = load_huggingface_model(hf_url)
                        model_id_for_benchmark = hf_url
                        _emit(f"[DEBUG] Loaded HuggingFace model: {hf_url}")
                    elif local_model_path:
                        model, tokenizer = load_local_model(local_model_path)
                        model_id_for_benchmark = os.path.basename(local_model_path)
                        _emit(f"[DEBUG] Loaded local model: {local_model_path}")
                    else:
                        _emit("[DEBUG] No model found.", "error")
                        return
                except Exception as e:
                    _emit(f"[DEBUG] Model load failed: {e}", "error")
                    _emit(traceback.format_exc(), "error")
                    return

                # 3) ORIGINAL EVAL + POWER ---------------------------------------
                stop_evt = threading.Event()
                power_readings, power_thread = sample_gpu_power_background(stop_evt, interval=0.25)

                def _progress(done, total):
                    try:
                        socketio.emit("status",
                                    {"message": f"Evaluating {done}/{total} samples...", "type": "loading"},
                                    to=upload_id)
                    except Exception:
                        pass

                _emit("[DEBUG] Evaluating ORIGINAL model...")
                t0 = time.perf_counter()
                orig_metrics = evaluate_model(model, tokenizer, df, target_col=target_col, progress_cb=_progress)
                elapsed_orig = time.perf_counter() - t0
                stop_evt.set()
                if power_thread:
                    power_thread.join(timeout=0.5)
                avg_watts_orig = (sum(power_readings) / len(power_readings)) if power_readings else 0.0
                energy_j_orig = avg_watts_orig * elapsed_orig  # Joules (W·s)
                thrpt_orig = (samples / elapsed_orig) if elapsed_orig > 0 else 0.0

                _emit(f"[DEBUG] ORIGINAL: elapsed={elapsed_orig:.3f}s | avg_watts={avg_watts_orig:.3f} "
                    f"| joules={energy_j_orig:.3f} | thrpt={thrpt_orig:.3f}/s")

                # 4) PRUNE + PRUNED EVAL -----------------------------------------
                pruned_model, model_info = disable_low_weight_neurons(model, threshold)
                _emit(f"[DEBUG] Pruned model at threshold={threshold}")

                orig_params = ((model_info or {}).get("original", {}) or {}).get("non_zero_params", 0)
                prun_params = ((model_info or {}).get("after_pruning", {}) or {}).get("non_zero_params", 0)
                _emit(f"[DEBUG] Params non-zero: original={orig_params} | pruned={prun_params}")

                stop_evt2 = threading.Event()
                power_readings2, power_thread2 = sample_gpu_power_background(stop_evt2, interval=0.25)
                _emit("[DEBUG] Evaluating PRUNED model...")
                t1 = time.perf_counter()
                pruned_metrics = evaluate_model(pruned_model, tokenizer, df, target_col=target_col, progress_cb=_progress)
                elapsed_pruned = time.perf_counter() - t1
                stop_evt2.set()
                if power_thread2:
                    power_thread2.join(timeout=0.5)
                avg_watts_pruned = (sum(power_readings2) / len(power_readings2)) if power_readings2 else 0.0
                energy_j_pruned = avg_watts_pruned * elapsed_pruned
                thrpt_pruned = (samples / elapsed_pruned) if elapsed_pruned > 0 else 0.0

                _emit(f"[DEBUG] PRUNED:   elapsed={elapsed_pruned:.3f}s | avg_watts={avg_watts_pruned:.3f} "
                    f"| joules={energy_j_pruned:.3f} | thrpt={thrpt_pruned:.3f}/s")

                # 5) FLOPS (safe) -------------------------------------------------
                orig_flops = pruned_flops = None
                try:
                    orig_flops, _ = estimate_flops(model)          # operations per call
                    pruned_flops, _ = estimate_flops(pruned_model)
                    _emit(f"[DEBUG] FLOPs (per call): original={orig_flops} | pruned={pruned_flops}")
                except Exception as e:
                    _emit(f"[DEBUG] FLOPs estimation failed: {e}")

                # TFLOPs-per-call number for the card (fallback to param counts)
                tflops_orig = (orig_flops / 1e12) if isinstance(orig_flops, (int, float)) else float(orig_params or 0)
                tflops_prun = (pruned_flops / 1e12) if isinstance(pruned_flops, (int, float)) else float(prun_params or 0)

                # 6) Cards: kWh & gCO2 per 1000 calls, accuracy %, TFLOPs --------
                loc = (data or {}).get("location") or "france"
                g_per_kwh = _grid_factor_g_per_kwh(loc)

                # kWh per 1000 calls
                kwh_orig_per_1k = ((energy_j_orig / 3_600_000.0) / max(samples, 1)) * 1000.0
                kwh_prun_per_1k = ((energy_j_pruned / 3_600_000.0) / max(samples, 1)) * 1000.0
                gco2_orig_per_1k = kwh_orig_per_1k * g_per_kwh
                gco2_prun_per_1k = kwh_prun_per_1k * g_per_kwh

                metric_cards = {
                    "power": {        # actually "energy" but UI labels it kWh
                        "original": float(kwh_orig_per_1k),
                        "pruned":   float(kwh_prun_per_1k),
                    },
                    "performance": {
                        "original": float(_percent(orig_metrics.get("overall", {}).get("accuracy", 0.0))),
                        "pruned":   float(_percent(pruned_metrics.get("overall", {}).get("accuracy", 0.0))),
                    },
                    "emissions": {
                        "original": float(gco2_orig_per_1k),
                        "pruned":   float(gco2_prun_per_1k),
                    },
                    "compute": {
                        "original": float(tflops_orig),
                        "pruned":   float(tflops_prun),
                    }
                }
                _emit(f"[DEBUG] metricCards: {metric_cards}")

                # 7) Overall + per-class blocks ----------------------------------
                overall_block = {
                    "accuracy":  _pair(orig_metrics["overall"]["accuracy"],  pruned_metrics["overall"]["accuracy"]),
                    "precision": _pair(orig_metrics["overall"]["precision"], pruned_metrics["overall"]["precision"]),
                    "recall":    _pair(orig_metrics["overall"]["recall"],    pruned_metrics["overall"]["recall"]),
                    "f1Score":   _pair(orig_metrics["overall"]["f1_score"],  pruned_metrics["overall"]["f1_score"]),
                }

                per_class_block = {}
                for lab in set(orig_metrics.keys()) | set(pruned_metrics.keys()):
                    if lab == "overall":
                        continue
                    o = orig_metrics.get(lab, {})
                    p = pruned_metrics.get(lab, {})
                    per_class_block[str(lab)] = {
                        "accuracy":  _pair(o.get("accuracy"),  p.get("accuracy")),
                        "precision": _pair(o.get("precision"), p.get("precision")),
                        "recall":    _pair(o.get("recall"),    p.get("recall")),
                        "f1Score":   _pair(o.get("f1_score"),  p.get("f1_score")),
                    }

                # 8) Compose JSON -------------------------------------------------
                device_label = detect_local_device()
                benchmark_data = {
                    "model": model_id_for_benchmark,
                    "threshold": float(threshold),
                    "gpu": device_label,
                    "location": loc,
                    "originalParameters": int(orig_params or 0),
                    "prunedParameters": int(prun_params or 0),

                    # NEW: metric cards the Angular header uses
                    "metricCards": metric_cards,

                    "overall": overall_block,
                    "perClass": per_class_block,

                    "realBenchmark": {
                        "device": device_label,
                        "elapsedSecOriginal": float(elapsed_orig),
                        "elapsedSecPruned":   float(elapsed_pruned),
                        "samples": samples,
                        "throughputOriginal": float(thrpt_orig),
                        "throughputPruned":   float(thrpt_pruned),
                        "avgGpuPowerWOriginal": float(avg_watts_orig),
                        "avgGpuPowerWPruned":   float(avg_watts_pruned),
                        "energyJoulesOriginal": float(energy_j_orig),
                        "energyJoulesPruned":   float(energy_j_pruned),
                        "tflopsPerCallOriginal": float(tflops_orig),
                        "tflopsPerCallPruned":   float(tflops_prun),
                        "metricsOriginal": orig_metrics.get("overall", {}),
                        "metricsPruned":   pruned_metrics.get("overall", {}),
                    }
                }

                # 9) Save ---------------------------------------------------------
                bench_dir = get_upload_path(upload_id)
                os.makedirs(bench_dir, exist_ok=True)
                bench_path = os.path.join(bench_dir, "benchmark_data.json")
                with open(bench_path, "w", encoding="utf-8") as f:
                    json.dump(benchmark_data, f, indent=2)
                _emit(f"[DEBUG] Wrote benchmark_data.json at {bench_path}")

                _emit("Real benchmark completed", "complete")

            except Exception as e:
                _emit(f"[DEBUG] Real benchmark failed: {e}", "error")
                _emit(traceback.format_exc(), "error")

        socketio.start_background_task(process)





