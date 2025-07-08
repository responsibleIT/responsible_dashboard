import os
import json
import asyncio
import copy
import pandas as pd
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv

load_dotenv()

from loading import load_huggingface_model, load_local_model
from preprocess import disable_low_weight_neurons
from benchmark import evaluate_model
from predict import predict_with_auto_regressive_model

DEMO_MODE = os.getenv("DEMO", "false").lower() == "true"
UPLOAD_DIR = "uploads"
THRESHOLDS = [i * 0.1 for i in range(1, 100)]
LABEL_MAPPING = {0: 'Ham', 1: 'Spam'}


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, upload_id: str):
        await websocket.accept()
        if upload_id not in self.active_connections:
            self.active_connections[upload_id] = []
        self.active_connections[upload_id].append(websocket)

    def disconnect(self, websocket: WebSocket, upload_id: str):
        if upload_id in self.active_connections:
            self.active_connections[upload_id].remove(websocket)
            if not self.active_connections[upload_id]:
                del self.active_connections[upload_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def send_message_to_upload(self, message: str, upload_id: str):
        if upload_id in self.active_connections:
            for connection in self.active_connections[upload_id]:
                try:
                    await connection.send_text(message)
                except:
                    self.active_connections[upload_id].remove(connection)

    async def broadcast_to_upload(self, data: dict, upload_id: str):
        message = json.dumps(data)
        await self.send_message_to_upload(message, upload_id)

manager = ConnectionManager()


def validate_upload_id(upload_id: str) -> bool:
    """Validate upload ID format and directory existence."""
    if not upload_id:
        return False
    
    upload_path = os.path.join(UPLOAD_DIR, upload_id)
    if not os.path.exists(upload_path):
        return False
    
    if DEMO_MODE:
        # In demo mode, we allow any upload_id format
        return True
    
    try:
        parts = upload_id.split('_')
        if len(parts) != 2:
            return False
        
        random_id = int(parts[1])
        return 1000 <= random_id <= 9999
    except (ValueError, IndexError):
        return False

def get_upload_path(upload_id: str) -> str:
    """Get the upload directory path for a given upload ID."""
    return os.path.join(UPLOAD_DIR, upload_id)

def get_dataset_path(upload_id: str) -> str:
    """Get the dataset CSV path for a given upload ID."""
    return os.path.join(get_upload_path(upload_id), "dataset.csv")

def has_huggingface_url(upload_id: str) -> bool:
    """Check if upload has a valid Hugging Face URL."""
    huggingface_path = os.path.join(get_upload_path(upload_id), "huggingface_url.txt")
    
    if not os.path.exists(huggingface_path):
        return False
    
    with open(huggingface_path, "r") as f:
        url = f.read().strip()
    
    return bool(url) and url.lower() != 'none'

def get_huggingface_url(upload_id: str) -> Optional[str]:
    """Get the Hugging Face URL for a given upload ID."""
    huggingface_path = os.path.join(get_upload_path(upload_id), "huggingface_url.txt")
    
    if not os.path.exists(huggingface_path):
        return None
    
    with open(huggingface_path, "r") as f:
        url = f.read().strip()
    
    return url if url else None

def get_model_path(upload_id: str) -> Optional[str]:
    """Get the local model file path for a given upload ID."""
    upload_path = get_upload_path(upload_id)
    model_files = [f for f in os.listdir(upload_path) if f.endswith(('.h5', '.keras'))]
    
    if not model_files:
        return None
    
    return os.path.join(upload_path, model_files[0])

async def send_message(upload_id: str, message: str, msg_type: str = "loading"):
    """Send a message to all connections for a given upload ID."""
    await manager.broadcast_to_upload({
        "type": msg_type,
        "message": message,
    }, upload_id)

def load_dataset(upload_id: str) -> pd.DataFrame:
    """Load the dataset CSV for a given upload ID."""
    return pd.read_csv(get_dataset_path(upload_id))

def save_json_file(upload_id: str, filename: str, data: dict):
    """Save data as JSON file in the upload directory."""
    file_path = os.path.join(get_upload_path(upload_id), filename)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def load_json_file(upload_id: str, filename: str) -> dict:
    """Load JSON data from file in the upload directory."""
    file_path = os.path.join(get_upload_path(upload_id), filename)
    with open(file_path, "r") as f:
        return json.load(f)

def create_baseline_metrics(metrics: dict, model_info: dict, threshold: float = 0) -> dict:
    """Create standardized metrics dictionary."""
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

def create_threshold_data_entry(metrics: dict, threshold: float) -> dict:
    """Create a threshold data entry with basic metrics."""
    return {
        "accuracy": 0,
        "flops": metrics['after_pruning']['flops_estimate'],
        "non_zero_params": metrics['after_pruning']['non_zero_params'],
        "params_reduction_pct": metrics['after_pruning']['params_reduction_pct'],
        "flops_reduction_pct": metrics['after_pruning']['flops_reduction_pct']
    }

def create_benchmark_data(huggingface_url: str, threshold: float, gpu: str, location: str, 
                         benchmark_data: dict, model_info: dict, pruned_threshold_data: dict) -> dict:
    """Create benchmark data structure for validation results."""
    storable_data = {
        "model": huggingface_url,
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

    # Add overall metrics
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        storable_data['overall'][metric] = {
            "original": pruned_threshold_data['0'][metric],
            "pruned": benchmark_data['overall'][metric],
        }

    # Add per-class metrics
    for label, metrics in benchmark_data.items():
        if label == 'overall':
            continue

        label_name = LABEL_MAPPING.get(label, label)
        storable_data['perClass'][label_name] = {}
        
        for metric in ['accuracy', 'precision', 'recall']:
            storable_data['perClass'][label_name][metric] = {
                "original": pruned_threshold_data['0']['per_class'][label_name][metric],
                "pruned": metrics[metric],
            }
        
        storable_data['perClass'][label_name]['f1Score'] = {
            "original": pruned_threshold_data['0']['per_class'][label_name]['f1_score'],
            "pruned": metrics['f1_score'],
        }

    return storable_data

async def process_start_message(upload_id: str):
    """Process the 'start' message type."""
    await send_message(upload_id, "Model is being loaded...")
    
    df = load_dataset(upload_id)
    
    if not has_huggingface_url(upload_id):
        # TODO: Handle local model benchmark
        return
    
    huggingface_url = get_huggingface_url(upload_id)
    if not huggingface_url:
        return
    
    # Load model and create baseline
    model, tokenizer = load_huggingface_model(huggingface_url)
    model_copy = copy.deepcopy(model)
    pruned_model, model_info = disable_low_weight_neurons(model_copy, 10)

    await send_message(upload_id, "Benchmarking model...")

    # Evaluate baseline and pruned models
    metrics_baseline = evaluate_model(model, tokenizer, df)
    metrics_pruned = evaluate_model(pruned_model, tokenizer, df)

    await send_message(upload_id, "Collecting pruning data...")

    # Create pruned threshold data structure
    pruned_threshold_data = {
        0: create_baseline_metrics(metrics_baseline, model_info, 0),
        10: create_baseline_metrics(metrics_pruned, model_info, 10)
    }

    # Process all thresholds
    for threshold in THRESHOLDS:
        threshold = round(threshold, 1)
        
        model_copy = copy.deepcopy(model)
        pruned_model, metrics = disable_low_weight_neurons(model_copy, threshold)
        
        pruned_threshold_data[threshold] = create_threshold_data_entry(metrics, threshold)
        del model_copy

    await send_message(upload_id, "Predicting performance...")

    # Predict and save results
    pruned_threshold_data = predict_with_auto_regressive_model(pruned_threshold_data, "accuracy")
    save_json_file(upload_id, "pruned_threshold_data.json", pruned_threshold_data)

    # wait 1 second to ensure all files are saved
    await asyncio.sleep(1)
    
    await send_message(upload_id, "Benchmark completed successfully", "complete")

async def process_validate_message(upload_id: str, threshold: float, gpu: str, location: str):
    """Process the 'validate' message type."""
    if not threshold or not isinstance(threshold, (int, float)):
        await send_message(upload_id, "Invalid threshold value", "error")
        return

    # Clean up existing benchmark data
    benchmark_data_path = os.path.join(get_upload_path(upload_id), "benchmark_data.json")
    if os.path.exists(benchmark_data_path):
        os.remove(benchmark_data_path)

    await send_message(upload_id, "Model is being loaded...")

    df = load_dataset(upload_id)

    if not has_huggingface_url(upload_id):
        # TODO: Handle local model validation
        return

    huggingface_url = get_huggingface_url(upload_id)
    if not huggingface_url:
        return

    # Load and prune model
    model, tokenizer = load_huggingface_model(huggingface_url)
    pruned_model, model_info = disable_low_weight_neurons(model, threshold)

    await send_message(upload_id, "Benchmarking model...")

    # Evaluate pruned model
    benchmark_data = evaluate_model(pruned_model, tokenizer, df)
    pruned_threshold_data = load_json_file(upload_id, "pruned_threshold_data.json")

    # Create and save benchmark results
    storable_data = create_benchmark_data(
        huggingface_url, threshold, gpu, location, 
        benchmark_data, model_info, pruned_threshold_data
    )
    
    save_json_file(upload_id, "benchmark_data.json", storable_data)
    await send_message(upload_id, "Validation completed successfully", "complete")

async def demo_start_process(upload_id: str):
    """Demo simulation of the start process with realistic timing."""
    await send_message(upload_id, "Model is being loaded...")
    await asyncio.sleep(2)
    
    await send_message(upload_id, "Benchmarking model...")
    await asyncio.sleep(10)
    
    await send_message(upload_id, "Collecting pruning data...")
    await asyncio.sleep(3)
    
    await send_message(upload_id, "Predicting performance...")
    await asyncio.sleep(2)
    
    await send_message(upload_id, "Analyzation completed successfully", "complete")

async def demo_validate_process(upload_id: str, threshold: float, gpu: str, location: str): 
    if threshold is not None and gpu is not None and location is not None:
        flag_path = os.path.join(get_upload_path(upload_id), "flag.json")
        if os.path.exists(flag_path):
            os.remove(flag_path)

        flag_data = {
            "model": get_huggingface_url(upload_id),
            "threshold": threshold,
            "gpu": gpu,
            "location": location
        }
        
        with open(flag_path, "w") as f:
            json.dump(flag_data, f)
    

    await send_message(upload_id, "Model is being loaded...")
    await asyncio.sleep(2)

    await send_message(upload_id, "Benchmarking model...")
    await asyncio.sleep(5)

    await send_message(upload_id, "Benchmark completed successfully", "complete")

async def handle_websocket_message(websocket: WebSocket, upload_id: str, message: dict):
    """Handle incoming WebSocket messages."""
    msg_type = message.get("type")
    
    if msg_type == "ping":
        await manager.send_personal_message(
            json.dumps({"type": "pong", "timestamp": message.get("timestamp")}),
            websocket
        )
    elif msg_type == "start":
        if DEMO_MODE:
            await demo_start_process(upload_id)
        else:
            await process_start_message(upload_id)
    elif msg_type == "validate":
        if DEMO_MODE:
            await demo_validate_process(
                upload_id, 
                message.get("threshold"), 
                message.get("gpu"), 
                message.get("location")
            )
        else:
            await process_validate_message(
                upload_id, 
                message.get("threshold"), 
                message.get("gpu"), 
                message.get("location")
            )
    else:
        await manager.send_personal_message(
            json.dumps({"type": "error", "message": "Unknown message type"}),
            websocket
        )

async def websocket_endpoint(websocket: WebSocket, upload_id: str):
    """Main WebSocket endpoint handler."""
    if not validate_upload_id(upload_id):
        await websocket.close(code=4001, reason="Invalid upload_id")
        return
    
    await manager.connect(websocket, upload_id)
    
    # Send initial connection confirmation
    await manager.send_personal_message(
        json.dumps({
            "type": "connection",
            "status": "connected",
            "upload_id": upload_id,
            "message": "WebSocket connection established"
        }),
        websocket
    )
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_websocket_message(websocket, upload_id, message)
                    
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({"type": "error", "message": "Invalid JSON format"}),
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, upload_id)
        print(f"Client disconnected from upload_id: {upload_id}")
    except Exception as e:
        print(f"WebSocket error for upload_id {upload_id}: {str(e)}")
        manager.disconnect(websocket, upload_id)