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
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

GRAPHICSCARD_MAPPING = {
    "NVIDIA A100": {"power": 400, "compute": 78.00},
    "NVIDIA Tesla V100": {"power": 300, "compute": 15.70},
    "NVIDIA T4": {"power": 70, "compute": 65.00},
}
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

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",   # simple and works on Windows
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

    if not os.path.exists(pruned_data_path):
        return jsonify({"error": "Pruned threshold data not found"}), 404

    with open(pruned_data_path, "r", encoding="utf-8") as f:
        pruned_data = json.load(f)

    gpu_data = GRAPHICSCARD_MAPPING.get(gpu)
    carbon_intensity = LOCATION_CARBON_MAPPING.get(location)

    tflops_per_threshold = {}
    power_per_threshold = {}
    emissions_per_threshold = {}
    performance_per_threshold = {}

    for key, data in pruned_data.items():
        flops = data.get("flops", 0)
        performance_per_threshold[key] = data.get("accuracy", 0) * 100
        tflops_per_threshold[key] = flops / 1e12
        power_per_threshold[key] = calculate_power_consumption(gpu_data, flops)
        emissions_per_threshold[key] = calculate_emissions(gpu_data, flops, carbon_intensity)

    return jsonify(
        {
            "tflops": tflops_per_threshold,
            "power": power_per_threshold,
            "emissions": emissions_per_threshold,
            "performance": performance_per_threshold,
        }
    )


# ---- benchmark data ----
@app.get("/benchmark/<upload_id>")
def get_benchmark_data(upload_id):
    upload_path = os.path.join(UPLOAD_DIR, upload_id)

    if DEMO_MODE == "true":
        flag_path = os.path.join(upload_path, "flag.json")
        if not os.path.exists(flag_path):
            return jsonify({"error": "Data not found"}), 404

        with open(flag_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model = data.get("model", "my-model")
        threshold = data.get("threshold", 0)
        gpu = data.get("gpu", "NVIDIA A100")
        location = data.get("location", "france")

        try:
            benchmark_data = get_benchmark_from_csv(model, upload_id, threshold, gpu, location)
            return jsonify(benchmark_data)
        except BenchmarkDataError as e:
            return jsonify({"error": str(e)}), e.status_code

    benchmark_data_path = os.path.join(upload_path, "benchmark_data.json")
    if not os.path.exists(benchmark_data_path):
        return jsonify({"error": "Benchmark data not found"}), 404

    with open(benchmark_data_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    gpu_data = GRAPHICSCARD_MAPPING.get(benchmark_data.get("gpu"))
    carbon_intensity = LOCATION_CARBON_MAPPING.get(benchmark_data.get("location"))

    original_flops = benchmark_data.get("originalFlops", 0)
    pruned_flops = benchmark_data.get("prunedFlops", 0)

    benchmark_data["metricCards"] = {
        "power": {
            "original": calculate_power_consumption(gpu_data, original_flops),
            "pruned": calculate_power_consumption(gpu_data, pruned_flops),
        },
        "performance": {
            "original": benchmark_data["overall"]["accuracy"]["original"] * 100,
            "pruned": benchmark_data["overall"]["accuracy"]["pruned"] * 100,
        },
        "emissions": {
            "original": calculate_emissions(gpu_data, original_flops, carbon_intensity),
            "pruned": calculate_emissions(gpu_data, pruned_flops, carbon_intensity),
        },
        "compute": {
            "original": original_flops / 1e12,
            "pruned": pruned_flops / 1e12,
        },
    }

    return jsonify(benchmark_data), 200

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
