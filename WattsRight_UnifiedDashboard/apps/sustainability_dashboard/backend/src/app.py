import os
import sys
import time
import random
import json

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename

# ---- local imports (backend) ----
from websocket import websocket_handlers
from utils.metrics import calculate_power_consumption, calculate_emissions
from demo import BenchmarkDataError, get_benchmark_from_csv

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
# If packaged by PyInstaller, also try resolving via _MEIPASS
if not os.path.exists(os.path.join(STATIC_DIR, "index.html")):
    alt = _bundle_path(os.path.join("apps", "sustainability_dashboard", "frontend_v2", "dist", "browser"))
    if os.path.exists(os.path.join(alt, "index.html")):
        STATIC_DIR = alt

# ---- Flask / Socket.IO setup ----
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    logger=True, engineio_logger=True)

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
        'upload', 'benchmark', 'chart-data', 'settings', 'socket.io', 'save_columns'
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
    model = request.files.get("model")
    dataset = request.files.get("dataset")

    # basic validation
    if not huggingface_url and not model:
        return jsonify({"error": "Either a HuggingFace URL or a model file must be provided"}), 400
    if not dataset:
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

    # persist HF URL
    if huggingface_url:
        with open(os.path.join(upload_path, "huggingface_url.txt"), "w", encoding="utf-8") as f:
            f.write(huggingface_url)

    # save model if provided
    if model:
        model_path = os.path.join(upload_path, secure_filename(model.filename))
        model.save(model_path)

    # save dataset
    if dataset:
        dataset_path = os.path.join(upload_path, secure_filename(dataset.filename))
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

    # Your GPU/location calculators
    gpu_data = GRAPHICSCARD_MAPPING.get(gpu) or {"power": 0, "compute": 0}
    carbon_intensity = LOCATION_CARBON_MAPPING.get(location, 0)

    for k, v in pruned_data.items():
        key = str(k)  # JSON keys are strings in the frontend anyway
        flops = v.get("flops", 0.0)
        performance = v.get("accuracy", 0.0)

        tflops[key] = flops / 1e12
        power[key] = calculate_power_consumption(gpu_data, flops)
        emissions[key] = calculate_emissions(gpu_data, flops, carbon_intensity)
        perf[key] = performance * 100.0

    return jsonify({
        "tflops": tflops,
        "power": power,
        "emissions": emissions,
        "performance": perf,
    })

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
    gpu_label = data.get("gpu") or "Detected GPU"
    location  = data.get("location") or "france"
    ci = LOCATION_CARBON_MAPPING.get(location, 300)  # gCO2/kWh

    rb = data.get("realBenchmark") or {}
    samples   = rb.get("samples")
    energy_j  = rb.get("energyJoules")
    elapsed   = rb.get("elapsedSec")
    avg_watts = rb.get("avgGpuPowerW")
    tput      = rb.get("throughputSamplesPerSec")
    overall   = rb.get("metrics") or {}

    # Power per 1000 calls (prefer total energy)
    kwh_per_1000 = None
    if energy_j and samples and samples > 0:
        kwh_per_1000 = (energy_j / samples) / 3_600_000.0 * 1000.0
    elif avg_watts and tput and tput > 0:
        kwh_per_1000 = ((avg_watts / tput) / 3_600_000.0) * 1000.0

    emissions_per_1000 = (kwh_per_1000 * ci) if kwh_per_1000 is not None else None

    resp = {
        "model": model,
        "gpu": gpu_label,          # <- shows “NVIDIA GeForce RTX 3060” in UI
        "location": location,
        "threshold": threshold,
        "overall": {
            "accuracy": {
                "original": overall.get("accuracy") or 0.0,
                "pruned":   overall.get("accuracy") or 0.0,
            }
        },
        "metricCards": {
            "power":      {"original": kwh_per_1000,       "pruned": kwh_per_1000},
            "emissions":  {"original": emissions_per_1000, "pruned": emissions_per_1000},
            "performance":{"original": (overall.get("accuracy") or 0.0) * 100.0,
                           "pruned":   (overall.get("accuracy") or 0.0) * 100.0},
            "compute":    {"original": None,               "pruned": None},  # unknown
        },
        "realBenchmark": rb
    }
    return jsonify(resp)


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
