import os
import sys
import time
import webbrowser
import socket
import subprocess
import tempfile
from pathlib import Path

# --- make console tolerant to unicode when running as .exe ---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --- Optional PyInstaller splash: present only inside the bundled .exe ---
# --- Optional PyInstaller splash (safe import) ---
def _get_pyi_splash():
    try:
        import os
        # Only import if bootloader actually created the splash
        if os.environ.get("_PYI_SPLASH") or os.environ.get("_PYI_SPLASH_IPC"):
            import pyi_splash  # provided by PyInstaller at runtime
            return pyi_splash
    except Exception:
        pass
    return None

pyi_splash = _get_pyi_splash()

def splash_update(msg: str):
    if pyi_splash:
        try:
            pyi_splash.update_text(msg)
        except Exception:
            pass

try:
    import tkinter  # forces PyInstaller to collect Tk/Tcl for the splash
except Exception:
    pass

# -------- Utilities --------
def resource_path(rel_path: str) -> str:
    """Absolute path to a resource whether running from source or PyInstaller one-file."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return str((base / rel_path).resolve())


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for_server(port: int, name: str, timeout: int = 30) -> bool:
    print(f"Waiting for {name} on port {port} (timeout {timeout}s)...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_port_open(port):
            print(f"{name} is up on port {port}.", flush=True)
            return True
        time.sleep(0.5)
        print(".", end="", flush=True)
    print("")  # newline
    print(f"Timed out waiting for {name} on port {port}.", flush=True)
    return False


def run_server(script_path: str, cwd: str | None, log_name: str) -> subprocess.Popen:
    """Start a Python script as a subprocess, logging stdout/stderr to a temp file."""
    log_dir = Path(tempfile.gettempdir()) / "wattsright_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{log_name}.log"
    print(f"Logging {log_name} to: {log_file}", flush=True)
    f = open(log_file, "w", buffering=1, encoding="utf-8", errors="replace")
    return subprocess.Popen(
        [sys.executable, script_path],
        cwd=cwd,
        stdout=f,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        creationflags=0,  # keep visible console for debugging
    )


# -------- Resolve paths inside/beside the bundle --------
FAIRNESS_SCRIPT = resource_path("apps/fairness_dashboard/flask_ml/app.py")

SUSTAINABILITY_DIR = resource_path("apps/sustainability_dashboard/backend/src")
SUSTAINABILITY_SCRIPT = str(Path(SUSTAINABILITY_DIR) / "app.py")

FRONTPAGE_HTML = resource_path("frontpage/index.html")

print("Resolved paths:", flush=True)
print(f"  Fairness script:        {FAIRNESS_SCRIPT}", flush=True)
print(f"  Sustainability script:  {SUSTAINABILITY_SCRIPT}", flush=True)
print(f"  Sustainability cwd:     {SUSTAINABILITY_DIR}", flush=True)
print(f"  Frontpage:              {FRONTPAGE_HTML}", flush=True)

# Sanity checks
missing = []
if not Path(FAIRNESS_SCRIPT).is_file():
    missing.append(FAIRNESS_SCRIPT)
if not Path(SUSTAINABILITY_SCRIPT).is_file():
    missing.append(SUSTAINABILITY_SCRIPT)
if not Path(FRONTPAGE_HTML).is_file():
    missing.append(FRONTPAGE_HTML)

if missing:
    print("Missing required files in the bundle:")
    for m in missing:
        print("  -", m)
    print("Check your .spec datas section.")
    sys.exit(1)

# -------- Start servers --------
print("Launching servers...", flush=True)
fairness_proc = run_server(
    FAIRNESS_SCRIPT,
    cwd=str(Path(FAIRNESS_SCRIPT).parent),
    log_name="fairness",
)
sustainability_proc = run_server(
    SUSTAINABILITY_SCRIPT,
    cwd=SUSTAINABILITY_DIR,
    log_name="sustainability",
)

# -------- Wait for ports and open frontpage --------
# show some progress on the splash while we wait
def splash_update(msg: str):
    if pyi_splash:
        try:
            pyi_splash.update_text(msg)
        except Exception:
            pass

splash_update("Starting services...")
ok1 = wait_for_server(5000, "Fairness dashboard", timeout=40)
splash_update("Fairness ready. Starting sustainability...")
ok2 = wait_for_server(8000, "Sustainability dashboard", timeout=40)

if ok1 and ok2:
    splash_update("Opening frontpage...")
    print("Opening frontpage...", flush=True)
    webbrowser.open(f"file:///{FRONTPAGE_HTML}")
    # hide the splash as soon as we’ve kicked the browser
    if pyi_splash:
        try:
            pyi_splash.close()
        except Exception:
            pass
else:
    print("One or both servers failed to start. See logs in %TEMP%/wattsright_logs.", flush=True)
    # make sure the splash doesn’t hang around forever on error
    if pyi_splash:
        try:
            pyi_splash.close()
        except Exception:
            pass

# -------- Keep parent alive; clean exit on Ctrl+C --------
try:
    fairness_proc.wait()
    sustainability_proc.wait()
except KeyboardInterrupt:
    print("\nKeyboardInterrupt: terminating children...", flush=True)
    fairness_proc.terminate()
    sustainability_proc.terminate()
