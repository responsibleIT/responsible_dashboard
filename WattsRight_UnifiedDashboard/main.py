import subprocess
import time
import os
import sys
import webbrowser

def resource_path(relative_path):
    """Get path to resource, for PyInstaller or dev"""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Start both dashboard backends
fairness_path = resource_path("apps/fairness_dashboard/flask_ml/app.py")
sustainability_path = resource_path("apps/sustainability_dashboard/backend/src/app.py")

fairness_proc = subprocess.Popen(["python", fairness_path])
sustainability_proc = subprocess.Popen(["python", sustainability_path])

# Wait for servers to start up
time.sleep(5)

# Open frontpage (not served over HTTP, just a file)
frontpage_path = resource_path("frontpage/index.html")
webbrowser.open(f"file://{frontpage_path}")

# Keep app running
try:
    fairness_proc.wait()
    sustainability_proc.wait()
except KeyboardInterrupt:
    fairness_proc.terminate()
    sustainability_proc.terminate()
