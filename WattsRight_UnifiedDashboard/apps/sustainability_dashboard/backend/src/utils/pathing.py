# apps/sustainability_dashboard/backend/src/utils/pathing.py
import os, sys

def backend_root() -> str:
    """
    Returns: .../apps/sustainability_dashboard/backend
    Works both in dev and within a PyInstaller bundle (_MEIPASS).
    """
    here = os.path.dirname(os.path.abspath(__file__))           # .../backend/src/utils
    dev_root = os.path.abspath(os.path.join(here, "..", ".."))  # .../backend
    if hasattr(sys, "_MEIPASS"):
        # Inside the exe; the bundle contains apps/sustainability_dashboard/backend
        bundle_root = os.path.join(
            sys._MEIPASS, "apps", "sustainability_dashboard", "backend"
        )
        if os.path.isdir(bundle_root):
            return bundle_root
    return dev_root

def asset_path(*parts: str) -> str:
    return os.path.join(backend_root(), *parts)

def model_path(name: str = "lstm_model_4.keras") -> str:
    return asset_path("model", name)
