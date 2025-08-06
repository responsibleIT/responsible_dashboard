import subprocess
import webbrowser
import time
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, 'frozen', False) else __file__))

FAIRNESS_PATH = os.path.join(BASE_DIR, "apps", "fairness_dashboard", "flask_ml", "app.py")
SUSTAINABILITY_PATH = os.path.join(BASE_DIR, "apps", "sustainability_dashboard", "backend", "src", "app.py")
FRONTPAGE_PATH = os.path.join(BASE_DIR, "frontpage", "index.html")

fairness_proc = subprocess.Popen(["python", FAIRNESS_PATH])
sustainability_proc = subprocess.Popen(["python", SUSTAINABILITY_PATH])

time.sleep(3)
webbrowser.open(f"file://{FRONTPAGE_PATH}")

try:
    fairness_proc.wait()
    sustainability_proc.wait()
except KeyboardInterrupt:
    fairness_proc.terminate()
    sustainability_proc.terminate()
