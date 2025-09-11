# utils/gpu_power.py
import subprocess, threading, time
from typing import List, Tuple, Callable, Optional

def _nvidia_smi_power_sum() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=1.5
        )
        vals = [v.strip() for v in out.splitlines() if v.strip()]
        watts = [float(v) for v in vals if v not in ("N/A", "")]
        return sum(watts) if watts else None
    except Exception:
        return None

def _nvml_power_sum() -> Tuple[Optional[Callable], Optional[Callable]]:
    try:
        import pynvml as nvml
        nvml.nvmlInit()
        def _sum():
            total = 0.0
            count = nvml.nvmlDeviceGetCount()
            for i in range(count):
                h = nvml.nvmlDeviceGetHandleByIndex(i)
                # milliwatts -> watts
                mw = nvml.nvmlDeviceGetPowerUsage(h)  # may raise
                total += (mw or 0) / 1000.0
            return total
        def _shutdown():
            try: nvml.nvmlShutdown()
            except Exception: pass
        return _sum, _shutdown
    except Exception:
        return None, None

def make_power_reader():
    """Return (read_watts: ()->float|None, cleanup: ()->None). Tries NVML, else nvidia-smi."""
    nvml_sum, nvml_shutdown = _nvml_power_sum()
    if nvml_sum:
        return nvml_sum, nvml_shutdown or (lambda: None)
    # fallback
    return _nvidia_smi_power_sum, (lambda: None)

def sample_gpu_power_background(stop_evt: threading.Event, interval: float = 0.25):
    """
    Returns (readings_list, thread). Readings is appended with watts at ~interval.
    """
    readings: List[float] = []
    read_watts, cleanup = make_power_reader()

    def _run():
        try:
            while not stop_evt.is_set():
                w = read_watts()
                if w is not None and w > 0:
                    readings.append(w)
                time.sleep(interval)
        finally:
            cleanup()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return readings, t