import os
import signal
import sys
import time
import random
import json

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename

# ---- local imports (backend) ----
from websocket import websocket_handlers
from utils.metrics import calculate_power_consumption, calculate_emissions
from demo import BenchmarkDataError, get_benchmark_from_csv
from model.generative_model.predict_generative import get_dropdown_models
from model.generative_model.benchmark_generative import generate_completion
from loading import load_huggingface_generative_model
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---- constants / paths ----
DEMO_MODE = "false"  # os.getenv("DEMO", "false").lower() == "true"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

GRAPHICSCARD_MAPPING = {
    "NVIDIA A100": {"power": 400, "compute": 78.00},
    "NVIDIA Tesla V100": {"power": 300, "compute": 15.70},
    "NVIDIA T4": {"power": 70, "compute": 65.00},
}
API_PREFIXES = (
    '/upload', '/save_columns', '/settings', '/chart-data',
    '/benchmark', '/socket.io'
)
LOCATION_CARBON_MAPPING = {"france": 50, "netherlands": 263, "germany": 329}
PERFORMANCE_METRICS = ["accuracy"]


def _bundle_path(rel: str) -> str:
    """Resolve path in dev AND when frozen with PyInstaller."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.abspath(os.path.join(base, rel))


# Absolute path to the Angular build directory (dev)
STATIC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "frontend_v2", "dist", "browser")
)

# Debug: print paths being checked
print(f"[Sustainability] Checking STATIC_DIR: {STATIC_DIR}", flush=True)
print(f"[Sustainability] index.html exists: {os.path.exists(os.path.join(STATIC_DIR, 'index.html'))}", flush=True)

# If packaged by PyInstaller, also try resolving via _MEIPASS
if not os.path.exists(os.path.join(STATIC_DIR, "index.html")):
    alt = _bundle_path(os.path.join("apps", "sustainability_dashboard", "frontend_v2", "dist", "browser"))
    print(f"[Sustainability] Trying alternate path: {alt}", flush=True)
    print(f"[Sustainability] alt index.html exists: {os.path.exists(os.path.join(alt, 'index.html'))}", flush=True)
    if os.path.exists(os.path.join(alt, "index.html")):
        STATIC_DIR = alt

print(f"[Sustainability] Final STATIC_DIR: {STATIC_DIR}", flush=True)

# ---- Flask / Socket.IO setup ----
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    allow_upgrades=False,
    logger=True,
    engineio_logger=True,
)

# Register websocket handlers
websocket_handlers(socketio)

# ---- frontend ----
@app.route("/")
def serve_frontend():
    idx = os.path.join(app.static_folder, "index.html")
    if os.path.exists(idx):
        return send_from_directory(app.static_folder, "index.html")
    return (
        "Frontend not bundled (index.html missing). Check Angular build output.",
        500,
    )

# Explicit SPA routes (avoid relying solely on wildcard for common client paths)
@app.route('/pruning-adjustments')
@app.route('/loading-upload')
@app.route('/loading-benchmark')
@app.route('/benchmark-results')
@app.route('/generative-results')
def spa_direct_named():
    idx = os.path.join(app.static_folder or '', 'index.html')
    if app.static_folder and os.path.exists(idx):
        return send_from_directory(app.static_folder, 'index.html')
    return jsonify({"error": "Frontend not bundled"}), 500


# ---- SPA fallback (serve Angular index for client-side routes) ----
@app.route('/<path:subpath>')
def spa_fallback(subpath: str):
    """Return index.html for unknown frontend routes so Angular router can handle them.

    Avoid intercepting known API namespaces and websocket paths.
    """
    # Known backend prefixes; let them 404 naturally or be handled by their own routes
    api_prefixes = (
        'upload', 'benchmark', 'chart-data', 'settings', 'socket.io', 'save_columns',
        'generative', 'api/'
    )
    if subpath.startswith(api_prefixes):
        # Let API 404 be explicit
        return jsonify({"error": f"Not found: {subpath}"}), 404

    idx = os.path.join(app.static_folder or '', 'index.html')
    if app.static_folder and os.path.exists(idx):
        # Debug trace: useful while diagnosing 404s
        # print(f"[SPA Fallback] Serving index.html for path: {subpath}")
        return send_from_directory(app.static_folder, 'index.html')
    return jsonify({"error": "Frontend not bundled"}), 500

# ---- optional: save target column after upload (JSON body) ----
@app.post("/save_columns")
def save_columns():
    data = request.get_json(force=True, silent=True) or {}
    upload_id = data.get("upload_id")
    target_column = data.get("target_column")

    if not upload_id or not target_column:
        return jsonify({"error": "upload_id and target_column are required"}), 400

    upload_path = os.path.join(UPLOAD_DIR, upload_id)
    os.makedirs(upload_path, exist_ok=True)

    payload = {"text_column": None, "target_column": target_column}
    with open(os.path.join(upload_path, "selected_columns.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return jsonify({"ok": True}), 200


# ---- upload endpoint ----
@app.post("/upload")
def upload_data():
    if DEMO_MODE == "true":
        return jsonify({"upload_id": "demo"}), 200

    huggingface_url = request.form.get("huggingface_url", "").strip()
    model_type = request.form.get("model_type", "classification").strip().lower()
    preset_model = request.form.get("preset_model", "").strip()
    model = request.files.get("model")
    dataset = request.files.get("dataset")

    if model_type not in ("classification", "generative"):
        model_type = "classification"

    # basic validation
    if not huggingface_url and not model and not preset_model:
        return jsonify({"error": "Either a HuggingFace URL, a model file, or a preset model must be provided"}), 400
    if not dataset and model_type != "generative":
        return jsonify({"error": "Dataset file is required"}), 400
    if model and not model.filename.lower().endswith((".h5", ".keras")):
        return jsonify({"error": "Model file must be a .h5 or .keras file"}), 400
    if dataset and not dataset.filename.lower().endswith(".csv"):
        return jsonify({"error": "Dataset file must be a .csv file"}), 400

    # create unique upload folder
    subdirectory = f"{int(time.time())}_{random.randint(1000, 9999)}"
    upload_path = os.path.join(UPLOAD_DIR, subdirectory)
    socketio.emit("status", {"message": f"[UPLOAD DEBUG] Saving upload to: {upload_path}"})
    os.makedirs(upload_path, exist_ok=True)

    with open(os.path.join(upload_path, "model_type.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": model_type}, f, ensure_ascii=False, indent=2)

    # persist HF URL
    if huggingface_url:
        with open(os.path.join(upload_path, "huggingface_url.txt"), "w", encoding="utf-8") as f:
            f.write(huggingface_url)

    # persist preset model name
    if preset_model:
        with open(os.path.join(upload_path, "preset_model.txt"), "w", encoding="utf-8") as f:
            f.write(preset_model)
        # Also save as HF URL for model loading (preset names map to HF repos)
        if not huggingface_url:
            with open(os.path.join(upload_path, "huggingface_url.txt"), "w", encoding="utf-8") as f:
                f.write(preset_model)

    # save model if provided
    if model:
        model_path = os.path.join(upload_path, secure_filename(model.filename))
        model.save(model_path)

    # save dataset under a canonical name used by websocket pipeline
    if dataset:
        dataset_path = os.path.join(upload_path, "dataset.csv")
        dataset.save(dataset_path)

    # persist selected columns if the form sent them (target only)
    # expects a JSON string like: {"target_column": "label"}
    selected = request.form.get("selected_columns")
    if selected:
        try:
            sel_obj = json.loads(selected)
            with open(os.path.join(upload_path, "selected_columns.json"), "w", encoding="utf-8") as f:
                json.dump(sel_obj, f, ensure_ascii=False, indent=2)
        except Exception:
            # non-fatal; proceed even if malformed
            pass

    # IMPORTANT: return the id the frontend needs
    return jsonify({"upload_id": subdirectory}), 200

# ---- settings ----
@app.get("/settings")
def get_settings():
    return jsonify(
        {
            "gpus": list(GRAPHICSCARD_MAPPING.keys()),
            "locations": list(LOCATION_CARBON_MAPPING.keys()),
            "metrics": PERFORMANCE_METRICS,
        }
    )

# ---- preset generative models ----
@app.get("/api/preset-models")
def get_preset_model_list():
    """Return the list of preset generative models (val + test)."""
    return jsonify({"models": get_dropdown_models()})

# ---- chart data ----
@app.get("/chart-data/<upload_id>/<gpu>/<location>")
def get_chart_data(upload_id, gpu, location):
    upload_path = os.path.join(UPLOAD_DIR, upload_id)
    pruned_data_path = os.path.join(upload_path, "pruned_threshold_data.json")

    # If the file isn't there or is malformed, return empty series safely
    if not os.path.exists(pruned_data_path):
        return jsonify({
            "tflops": {},
            "power": {},
            "emissions": {},
            "performance": {}
        })

    try:
        with open(pruned_data_path, "r", encoding="utf-8") as f:
            pruned_data = json.load(f) or {}
    except Exception:
        return jsonify({
            "tflops": {},
            "power": {},
            "emissions": {},
            "performance": {}
        })

    # Guard against None
    if pruned_data is None:
        return jsonify({
            "tflops": {},
            "power": {},
            "emissions": {},
            "performance": {}
        })

    # Build the four series. Keys must be strings; values must be numbers.
    tflops, power, emissions, perf = {}, {}, {}, {}

    # GPU/location calculators
    gpu_data = GRAPHICSCARD_MAPPING.get(gpu) or {"power": 0, "compute": 0}
    carbon_intensity = LOCATION_CARBON_MAPPING.get(location, 0)

    perplexity = {}
    for k, v in pruned_data.items():
        # Normalize key: JavaScript uses "0" not "0.0" for whole numbers
        num = float(k)
        key = str(int(num)) if num == int(num) else str(num)
        flops = v.get("flops", 0.0)
        performance = v.get("accuracy", 0.0)

        tflops[key] = flops / 1e12
        power[key] = calculate_power_consumption(gpu_data, flops)
        emissions[key] = calculate_emissions(gpu_data, flops, carbon_intensity)
        perf[key] = performance * 100.0

        if "perplexity" in v:
            perplexity[key] = v["perplexity"]

    result = {
        "tflops": tflops,
        "power": power,
        "emissions": emissions,
        "performance": perf,
    }
    if perplexity:
        result["perplexity"] = perplexity

    return jsonify(result)

# ---- benchmark data ----
@app.get("/benchmark/<upload_id>")
def get_benchmark_data(upload_id):
    upload_path = os.path.join(UPLOAD_DIR, upload_id)
    bench_path = os.path.join(upload_path, "benchmark_data.json")
    if not os.path.exists(bench_path):
        return jsonify({"error": "Benchmark data not found"}), 404

    with open(bench_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model     = data.get("model", "model")
    threshold = data.get("threshold", 0)
    gpu_label = data.get("gpu") or (data.get("realBenchmark") or {}).get("device") or "Detected GPU"
    location  = data.get("location") or "france"
    ci        = LOCATION_CARBON_MAPPING.get(location, 300)  # gCO2/kWh

    # Prefer what benchmark_real already saved
    metric_cards = data.get("metricCards")

    # If not present, compute from the new realBenchmark fields
    if not metric_cards:
        rb = data.get("realBenchmark") or {}
        samples = rb.get("samples") or 0

        # Energy → kWh per 1000 calls (prefer direct energy totals)
        kwh_orig = kwh_prun = None
        if samples and rb.get("energyJoulesOriginal") is not None and rb.get("energyJoulesPruned") is not None:
            kwh_orig = (rb["energyJoulesOriginal"] / samples) / 3_600_000.0 * 1000.0
            kwh_prun = (rb["energyJoulesPruned"]   / samples) / 3_600_000.0 * 1000.0
        else:
            # Fallback: avg power + throughput
            avgW_o = rb.get("avgGpuPowerWOriginal")
            avgW_p = rb.get("avgGpuPowerWPruned")
            tput_o = rb.get("throughputOriginal")
            tput_p = rb.get("throughputPruned")
            if avgW_o and tput_o:
                kwh_orig = ((avgW_o / tput_o) / 3_600_000.0) * 1000.0
            if avgW_p and tput_p:
                kwh_prun = ((avgW_p / tput_p) / 3_600_000.0) * 1000.0

        # Emissions per 1000 from kWh
        gco2_orig = (kwh_orig * ci) if kwh_orig is not None else None
        gco2_prun = (kwh_prun * ci) if kwh_prun is not None else None

        # Performance % from data["overall"]
        ov = data.get("overall") or {}
        acc_o = (ov.get("accuracy") or {}).get("original")
        acc_p = (ov.get("accuracy") or {}).get("pruned")
        perf_o = (acc_o * 100.0) if isinstance(acc_o, (int, float)) else None
        perf_p = (acc_p * 100.0) if isinstance(acc_p, (int, float)) else None

        # TFLOPs per call from new fields
        tflops_o = rb.get("tflopsPerCallOriginal")
        tflops_p = rb.get("tflopsPerCallPruned")

        metric_cards = {
            "power":       {"original": kwh_orig,  "pruned": kwh_prun},
            "emissions":   {"original": gco2_orig, "pruned": gco2_prun},
            "performance": {"original": perf_o,    "pruned": perf_p},
            "compute":     {"original": tflops_o,  "pruned": tflops_p},
        }

    resp = {
        "model": model,
        "gpu": gpu_label,
        "location": location,
        "threshold": threshold,
        "overall": data.get("overall", {}),
        "perClass": data.get("perClass", {}),
        "originalParameters": data.get("originalParameters"),
        "prunedParameters": data.get("prunedParameters"),
        "metricCards": metric_cards,
        "realBenchmark": data.get("realBenchmark", {}),
        "textExamples": data.get("textExamples", []),
        "generationExample": data.get("generationExample"),
    }
    return jsonify(resp)

@app.route("/api/export/<upload_id>", methods=["GET"])
def export_model(upload_id):
    try:
        upload_path = os.path.join(UPLOAD_DIR, upload_id)
        zip_path = os.path.join(upload_path, "pruned_model_hf.zip")
        if not os.path.exists(zip_path):
            return {"error": "Pruned model not found"}, 404

        return send_file(zip_path, as_attachment=True, download_name="pruned_model.zip")
    except Exception as e:
        return {"error": str(e)}, 500


# ---- generative dashboard data ----
@app.get("/generative/<upload_id>")
def get_generative_data(upload_id):
    """Return the structured GenerativeDashboardData for a generative upload."""
    upload_path = os.path.join(UPLOAD_DIR, upload_id)
    data_path = os.path.join(upload_path, "generative_dashboard_data.json")
    if not os.path.exists(data_path):
        return jsonify({"error": "Generative dashboard data not found"}), 404

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return jsonify(data)

# ---- in-memory model cache for generation ----
_generation_cache = {}  # keyed by upload_id → { "original": model, "pruned": model, "tokenizer": tok, "device": dev }


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Generate text from both original and pruned models for side-by-side comparison."""
    body = request.get_json(silent=True) or {}
    upload_id = body.get("upload_id")
    prompt = body.get("prompt", "").strip()

    if not upload_id or not prompt:
        return jsonify({"error": "upload_id and prompt are required"}), 400

    upload_path = os.path.join(UPLOAD_DIR, upload_id)
    pruned_dir = os.path.join(upload_path, "pruned_model_hf")
    hf_url_file = os.path.join(upload_path, "huggingface_url.txt")

    if not os.path.isdir(pruned_dir):
        return jsonify({"error": "No pruned model found. Run a benchmark first."}), 404

    try:
        # Load models (cached per upload_id)
        if upload_id not in _generation_cache:
            # Read original model name
            if not os.path.exists(hf_url_file):
                return jsonify({"error": "Original model reference not found."}), 404

            with open(hf_url_file, "r") as f:
                hf_repo = f.read().strip()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            original_model = AutoModelForCausalLM.from_pretrained(
                hf_repo, torch_dtype=dtype, trust_remote_code=True,
            )
            original_model.to(device).eval()
            original_model.config.pad_token_id = tokenizer.pad_token_id

            pruned_model = AutoModelForCausalLM.from_pretrained(
                pruned_dir, torch_dtype=dtype, trust_remote_code=True,
            )
            pruned_model.to(device).eval()
            pruned_model.config.pad_token_id = tokenizer.pad_token_id

            _generation_cache[upload_id] = {
                "original": original_model,
                "pruned": pruned_model,
                "tokenizer": tokenizer,
                "device": device,
            }

        cached = _generation_cache[upload_id]
        original_text = generate_completion(
            cached["original"], cached["tokenizer"], prompt,
            device=cached["device"], max_new_tokens=80,
        )
        pruned_text = generate_completion(
            cached["pruned"], cached["tokenizer"], prompt,
            device=cached["device"], max_new_tokens=80,
        )

        return jsonify({
            "original": original_text,
            "pruned": pruned_text,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/shutdown', methods=['POST'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    os._exit(0)   # <-- forcefully end Python process
    return 'Server shutting down...'

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def spa_index(path: str):
    # If the request looks like an API/socket/asset route, let Flask handle it
    for p in API_PREFIXES:
        if path.startswith(p.lstrip('/')):
            return ("Not Found", 404)

    # Otherwise return the Angular app shell
    idx = os.path.join(app.static_folder, "index.html")
    if os.path.exists(idx):
        return send_from_directory(app.static_folder, "index.html")
    return ("Frontend not bundled (index.html missing).", 500)

# ---- entrypoint ----
if __name__ == "__main__":
    print("Sustainability dashboard is now running on port 8000")
    # IMPORTANT: run Socket.IO server (not app.run), no reloader
    socketio.run(
        app,
        host="0.0.0.0",
        port=8000,
        debug=True,
        use_reloader=False,
        allow_unsafe_werkzeug=True,  # dev convenience on Windows
    )
