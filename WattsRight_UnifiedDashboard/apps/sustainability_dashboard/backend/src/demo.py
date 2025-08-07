import pandas as pd
import json
import os

from utils.metrics import calculate_power_consumption, calculate_emissions

UPLOAD_DIR = "uploads"
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

class BenchmarkDataError(Exception):
    """Custom exception for benchmark data errors."""
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code

def get_benchmark_from_csv(model: str, upload_id: str, target_threshold: float, gpu: str, location: str):
    upload_path = os.path.join(UPLOAD_DIR, upload_id)
    csv_path = os.path.join(upload_path, "threshold_data.csv")

    if not os.path.exists(csv_path):
        raise BenchmarkDataError("Threshold data CSV not found", 404)

    df = pd.read_csv(csv_path)

    baseline_row = df[df['threshold'] == 0]
    if baseline_row.empty:
        raise BenchmarkDataError("Baseline data (threshold 0) not found in CSV", 404)
    baseline = baseline_row.iloc[0]

    target_row = df[df['threshold'] == target_threshold]
    if target_row.empty:
        raise BenchmarkDataError(f"Data for threshold {target_threshold} not found in CSV", 404)
    target = target_row.iloc[0]

    gpu_mapped = GRAPHICSCARD_MAPPING.get(gpu)
    carbon_intensity = LOCATION_CARBON_MAPPING.get(location)

    if not gpu_mapped:
        raise BenchmarkDataError(f"Unknown GPU: {gpu}", 400)
    if carbon_intensity is None:
        raise BenchmarkDataError(f"Unknown location: {location}", 400)

    return {
        "model": model,
        "gpu": gpu,
        "location": location,
        "threshold": target_threshold,
        "overall": {
            "accuracy": {
                "original": float(baseline['overall_accuracy']),
                "pruned": float(target['overall_accuracy'])
            },
            "precision": {
                "original": float(baseline['overall_precision']),
                "pruned": float(target['overall_precision'])
            },
            "recall": {
                "original": float(baseline['overall_recall']),
                "pruned": float(target['overall_recall'])
            },
            "f1Score": {
                "original": float(baseline['overall_f1']),
                "pruned": float(target['overall_f1'])
            }
        },
        "perClass": {
            "Ham": {
                "accuracy": {
                    "original": float(baseline['ham_accuracy']),
                    "pruned": float(target['ham_accuracy'])
                },
                "precision": {
                    "original": float(baseline['ham_precision']),
                    "pruned": float(target['ham_precision'])
                },
                "recall": {
                    "original": float(baseline['ham_recall']),
                    "pruned": float(target['ham_recall'])
                },
                "f1Score": {
                    "original": float(baseline['ham_f1']),
                    "pruned": float(target['ham_f1'])
                }
            },
            "Spam": {
                "accuracy": {
                    "original": float(baseline['spam_accuracy']),
                    "pruned": float(target['spam_accuracy'])
                },
                "precision": {
                    "original": float(baseline['spam_precision']),
                    "pruned": float(target['spam_precision'])
                },
                "recall": {
                    "original": float(baseline['spam_recall']),
                    "pruned": float(target['spam_recall'])
                },
                "f1Score": {
                    "original": float(baseline['spam_f1']),
                    "pruned": float(target['spam_f1'])
                }
            }
        },
        "originalFlops": float(baseline['flops']),
        "prunedFlops": float(target['flops']),
        "reductionPercentage": float(target['params_reduction_pct']),
        "originalParameters": int(baseline['non_zero_params']),
        "prunedParameters": int(target['non_zero_params']),
        "metricCards": {
            "power": {
                "original": calculate_power_consumption(gpu_mapped, float(baseline['flops'])),
                "pruned": calculate_power_consumption(gpu_mapped, float(target['flops']))
            },
            "performance": {
                "original": float(baseline['overall_accuracy']) * 100,
                "pruned": float(target['overall_accuracy']) * 100
            },
            "emissions": {
                "original": calculate_emissions(gpu_mapped, float(baseline['flops']), carbon_intensity),
                "pruned": calculate_emissions(gpu_mapped, float(target['flops']), carbon_intensity)
            },
            "compute": {
                "original": float(baseline['flops']) / 1e12,
                "pruned": float(target['flops']) / 1e12
            }
        }
    }
