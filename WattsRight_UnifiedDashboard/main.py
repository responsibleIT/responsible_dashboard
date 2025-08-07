import subprocess
import time
import os
import sys
import webbrowser
import socket

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def wait_for_server(port, timeout=15):
    for _ in range(timeout * 2):
        if is_port_open(port):
            return True
        time.sleep(0.5)
    return False

def run_server(script_path, cwd=None):
    return subprocess.Popen([sys.executable, script_path], cwd=cwd)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed: correct path for fairness dashboard
fairness_path = os.path.join(BASE_DIR, "apps", "fairness-dashboard", "flask_ml", "app.py")

# Fixed: correct cwd for sustainability dashboard to resolve imports
sustainability_src = os.path.join(BASE_DIR, "apps", "sustainability_dashboard", "backend", "src")
print("Launching sustainability from:", sustainability_src)
sustainability_path = os.path.join(sustainability_src, "app.py")

# Start servers
fairness_proc = run_server(fairness_path)
sustainability_proc = run_server(sustainability_path, cwd=sustainability_src)

# Open frontpage when both ports are live
if wait_for_server(5000) and wait_for_server(8000):
    frontpage = os.path.join(BASE_DIR, "frontpage", "index.html")
    webbrowser.open(f"file://{frontpage}")
else:
    print("One or both servers failed to start.")

try:
    fairness_proc.wait()
    sustainability_proc.wait()
except KeyboardInterrupt:
    fairness_proc.terminate()
    sustainability_proc.terminate()
