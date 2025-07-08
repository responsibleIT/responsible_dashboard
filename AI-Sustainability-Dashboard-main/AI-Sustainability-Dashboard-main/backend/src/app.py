import os
import uvicorn
import time
import random
import json

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from websocket import websocket_endpoint
from utils.metrics import calculate_power_consumption, calculate_emissions
from demo import get_benchmark_from_csv

DEMO_MODE = os.getenv("DEMO", "false").lower() == "true"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# power: Consumption at peak performance in watts
# compute: Max TFLOPS (floating point operations per second)
GRAPHICSCARD_MAPPING = {
    "NVIDIA A100": {
        "power": 400,
        "compute": 78.00,
    },
    "NVIDIA Tesla V100": {
        "power": 300,
        "compute": 15.70,
    },
    "NVIDIA T4": {
        "power": 70,
        "compute": 65.00,
    },
}

# gCO2 per kWh
LOCATION_CARBON_MAPPING = {
    "france": 50,
    "netherlands": 263,
    "germany": 329,
}

PERFORMANCE_METRICS = [
    "accuracy",
]


app = FastAPI(title="HuggingFace API", description="API for HuggingFace URL and file uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "https://dashboard2.alexandervreeswijk.com"],  # Angular dev server default port
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_data_test(
    huggingface_url: str = Form(default=""),
    model: Optional[UploadFile] = File(None),
    dataset: Optional[UploadFile] = File(None)
):
    if DEMO_MODE:
        return {"upload_id": 'demo'}

    try:    
        has_huggingface_url = (
            huggingface_url and 
            huggingface_url.strip() and 
            huggingface_url.lower() != 'none'
        )
        has_model = model is not None
        has_dataset = dataset is not None
        
        if not has_huggingface_url and not has_model:
            raise HTTPException(status_code=400, detail="Either a HuggingFace URL or a model file must be provided")
        
        if not has_dataset:
            raise HTTPException(status_code=400, detail="Dataset file is required")
        
        if model and not model.filename.endswith(('.h5', '.keras')):
            raise HTTPException(status_code=400, detail="Model file must be a .h5 or .keras file")
        
        if dataset and not dataset.filename.endswith(('.csv')):
            raise HTTPException(status_code=400, detail="Dataset file must be a .csv file")

        # Create unique subdirectory
        timestamp = int(time.time())
        random_id = random.randint(1000, 9999)
        subdirectory = f"{timestamp}_{random_id}"
        upload_path = os.path.join(UPLOAD_DIR, subdirectory)
        os.makedirs(upload_path, exist_ok=True)

        # Save HuggingFace URL
        if has_huggingface_url:
            huggingface_path = os.path.join(upload_path, "huggingface_url.txt")
            with open(huggingface_path, "w") as f:
                f.write(huggingface_url.strip())
        
        # Save model file
        if model:
            extension = model.filename.split('.')[-1]
            model_path = os.path.join(upload_path, f"model.{extension}")
            with open(model_path, "wb") as buffer:
                content = await model.read()
                buffer.write(content)

        # Save dataset file
        if dataset:
            extension = dataset.filename.split('.')[-1]
            dataset_path = os.path.join(upload_path, f"dataset.{extension}")
            with open(dataset_path, "wb") as buffer:
                content = await dataset.read()
                buffer.write(content)
            
        return {"upload_id": subdirectory}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Test error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.websocket("/ws/{upload_id}")
async def websocket_route(websocket: WebSocket, upload_id: str):
    await websocket_endpoint(websocket, upload_id)

@app.get("/settings")
def get_settings():
    return {
        "gpus": list(GRAPHICSCARD_MAPPING.keys()),
        "locations": list(LOCATION_CARBON_MAPPING.keys()),
        "metrics": PERFORMANCE_METRICS,
    }

@app.get("/chart-data/{upload_id}/{gpu}/{location}")
def get_chart_data(upload_id: str, gpu: str, location: str):
    upload_path = os.path.join(UPLOAD_DIR, upload_id)
    gpu = GRAPHICSCARD_MAPPING.get(gpu)
    carbon_intensity = LOCATION_CARBON_MAPPING.get(location)
    
    tflops_per_threshold = {}
    power_per_threshold = {}
    emissions_per_threshold = {}
    performance_per_threshold = {}

    # Load the pruned_threshold_data.json file
    pruned_data_path = os.path.join(upload_path, "pruned_threshold_data.json")
    if not os.path.exists(pruned_data_path):
        raise HTTPException(status_code=404, detail="Pruned threshold data not found")
    
    with open(pruned_data_path, "r") as f:
        pruned_data = json.load(f)

    # For each key in the pruned data
    for key, data in pruned_data.items():
        flops = data.get("flops", 0)
        performance_per_threshold[key] = data.get("accuracy", 0) * 100
        tflops_per_threshold[key] = flops / 1e12  # Convert to TFLOPS
        power_per_threshold[key] = calculate_power_consumption(gpu, flops)
        emissions_per_threshold[key] = calculate_emissions(gpu, flops, carbon_intensity)

    # Prepare the response data
    response_data = {
        "tflops": tflops_per_threshold,
        "power": power_per_threshold,
        "emissions": emissions_per_threshold,
        "performance": performance_per_threshold,
    }

    return JSONResponse(content=response_data)

@app.get("/benchmark/{upload_id}")
def get_benchmark_data(upload_id: str):
    upload_path = os.path.join(UPLOAD_DIR, upload_id)

    if DEMO_MODE:
        # Get the flag.txt file and read the json data
        flag_path = os.path.join(upload_path, "flag.json")
        if not os.path.exists(flag_path):
            raise HTTPException(status_code=404, detail="Data not found")
        with open(flag_path, "r") as f:
            data = json.load(f)
        
        model = data.get("model", "my-model")
        threshold = data.get("threshold", 0)
        gpu = data.get("gpu", "NVIDIA A100")
        location = data.get("location", "france")

        benchmark_data = get_benchmark_from_csv(model, upload_id, threshold, gpu, location)
        return JSONResponse(content=benchmark_data)
    
    # Load the benchmark_data.json file
    benchmark_data_path = os.path.join(upload_path, f"benchmark_data.json")
    if not os.path.exists(benchmark_data_path):
        raise HTTPException(status_code=404, detail="Benchmark data not found")
    
    with open(benchmark_data_path, "r") as f:
        benchmark_data = json.load(f)

    gpu = benchmark_data.get("gpu", None)
    location = benchmark_data.get("location", None)

    if not gpu or not location:
        raise HTTPException(status_code=400, detail="GPU and location must be specified in the benchmark data")
    
    gpu = GRAPHICSCARD_MAPPING.get(gpu)
    carbon_intensity = LOCATION_CARBON_MAPPING.get(location)

    original_flops = benchmark_data.get("originalFlops", 0)
    pruned_flops = benchmark_data.get("prunedFlops", 0)
    
    benchmark_data["metricCards"] = {
        "power": {
            "original": calculate_power_consumption(gpu, original_flops),
            "pruned": calculate_power_consumption(gpu, pruned_flops),
        },
        "performance": {
            "original": benchmark_data["overall"]["accuracy"]["original"] * 100,
            "pruned": benchmark_data["overall"]["accuracy"]["pruned"] * 100,
        },
        "emissions": {
            "original": calculate_emissions(gpu, original_flops, carbon_intensity),
            "pruned": calculate_emissions(gpu, pruned_flops, carbon_intensity),
        },
        "compute": {
            "original": original_flops / 1e12,
            "pruned": pruned_flops / 1e12,
        },
    }

    return JSONResponse(content=benchmark_data)



if __name__ == "__main__":

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)