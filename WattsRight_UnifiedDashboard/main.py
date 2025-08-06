import subprocess
import time
import os
import sys

def resource_path(relative_path):
    """ Get path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Resolve paths
FAIRNESS_PATH = resource_path("apps/fairness_dashboard/flask_ml/app.py")
SUSTAINABILITY_PATH = resource_path("apps/sustainability_dashboard/backend/src/app.py")
FRONTPAGE_PATH = resource_path("frontpage/index.html")

# Start Flask and FastAPI
fairness_proc = subprocess.Popen(["python", FAIRNESS_PATH])
sustainability_proc = subprocess.Popen(["python", SUSTAINABILITY_PATH])

# Wait for servers to start
time.sleep(5)

# Open frontpage in default browser (reliable in .exe)
try:
    os.startfile(FRONTPAGE_PATH)
except Exception as e:
    print("Failed to open frontpage:", e)

# Wait for shutdown
try:
    fairness_proc.wait()
    sustainability_proc.wait()
except KeyboardInterrupt:
    fairness_proc.terminate()
    sustainability_proc.terminate()
