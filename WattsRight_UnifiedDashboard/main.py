import subprocess
import webbrowser
import time
import os

# Set paths
FAIRNESS_PATH = "apps\\fairness_dashboard\\flask_ml\\app.py"
SUSTAINABILITY_PATH = "apps\\sustainability_dashboard\\backend\\src\\app.py"
FRONTPAGE_PATH = os.path.abspath("frontpage/index.html")

# Start Flask (Fairness) on port 5000
fairness_proc = subprocess.Popen(["python", FAIRNESS_PATH])

# Start FastAPI (Sustainability) on port 8000
sustainability_proc = subprocess.Popen([
    "python", "-m", "uvicorn",
    "apps.sustainability_dashboard.backend.src.app:app", 
    "--host", "0.0.0.0", 
    "--port", "8000", 
    "--reload"
])

# Wait a bit for servers to start
time.sleep(3)

# Open the frontpage in the default browser
webbrowser.open(f"file://{FRONTPAGE_PATH}")

# Wait for both processes to complete (until user closes)
try:
    fairness_proc.wait()
    sustainability_proc.wait()
except KeyboardInterrupt:
    print("Shutting down...")
    fairness_proc.terminate()
    sustainability_proc.terminate()
