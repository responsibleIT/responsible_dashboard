import os
import sys
import time
import runpy
import socket
import webbrowser
import subprocess
import tempfile
from pathlib import Path

# ---------------- Utilities ----------------

def safe_print(*args, **kwargs):
    """Print using a safe encoding (no emoji)."""
    text = " ".join(str(a) for a in args)
    try:
        # write directly to stdout with replacement; works in cp1252 consoles
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
    except Exception:
        # fall back to ascii-only
        sys.stdout.write(text.encode("ascii", "ignore").decode("ascii") + "\n")
        sys.stdout.flush()

def resource_path(rel_path: str) -> str:
    """Return absolute path to a resource whether running from source or PyInstaller one-file."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return str((base / rel_path).resolve())

def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0

def wait_for_server(port: int, name: str, timeout: int = 40) -> bool:
    safe_print(f"Waiting for {name} on port {port} (timeout {timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_port_open(port):
            safe_print(f"{name} is up on port {port}.")
            return True
        time.sleep(0.5)
        sys.stdout.write(".")
        sys.stdout.flush()
    sys.stdout.write("\n")
    safe_print(f"Timed out waiting for {name} on port {port}.")
    return False

def log_file(name: str) -> Path:
    d = Path(tempfile.gettempdir()) / "wattsright_logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.log"

# ------------- Paths inside/beside bundle -------------

FAIRNESS_SCRIPT = resource_path("apps/fairness_dashboard/flask_ml/app.py")

SUSTAINABILITY_DIR = resource_path("apps/sustainability_dashboard/backend/src")
SUSTAINABILITY_SCRIPT = str(Path(SUSTAINABILITY_DIR) / "app.py")

FRONTPAGE_HTML = resource_path("frontpage/index.html")

# ---------------- Child runners ----------------

def run_child(script_path: str, cwd: str) -> None:
    """
    Run a target script in this frozen interpreter (child mode).
    This avoids needing an external python.exe.
    """
    os.chdir(cwd)
    # Ensure the target directory is importable
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    # Make printing safer in child too
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # Hand control to the target app.py as if it were run directly.
    # This will block until that script exits.
    runpy.run_path(script_path, run_name="__main__")

# ---------------- Entry point ----------------

def main():
    # Child mode?
    if any(a.startswith("--child=") for a in sys.argv[1:]):
        # --child=fairness or --child=sustainability
        child = next(a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--child="))
        if child == "fairness":
            run_child(FAIRNESS_SCRIPT, Path(FAIRNESS_SCRIPT).parent.as_posix())
        elif child == "sustainability":
            run_child(SUSTAINABILITY_SCRIPT, SUSTAINABILITY_DIR)
        else:
            safe_print(f"Unknown child target: {child}")
            sys.exit(2)
        return

    # Parent launcher
    safe_print('Resolved paths:')
    safe_print('  Fairness script:       ', FAIRNESS_SCRIPT)
    safe_print('  Sustainability script: ', SUSTAINABILITY_SCRIPT)
    safe_print('  Sustainability cwd:    ', SUSTAINABILITY_DIR)
    safe_print('  Frontpage:             ', FRONTPAGE_HTML)

    missing = []
    if not Path(FAIRNESS_SCRIPT).is_file():        missing.append(FAIRNESS_SCRIPT)
    if not Path(SUSTAINABILITY_SCRIPT).is_file():  missing.append(SUSTAINABILITY_SCRIPT)
    if not Path(FRONTPAGE_HTML).is_file():         missing.append(FRONTPAGE_HTML)

    if missing:
        safe_print("Missing required files in the bundle:")
        for m in missing:
            safe_print("  -", m)
        safe_print("Check your .spec datas section.")
        sys.exit(1)

    safe_print("Launching servers...")

    # Spawn two new instances of THIS exe, each in child mode.
    this_exe = sys.argv[0]  # path to the frozen exe (or to main.py in dev)

    fair_log = log_file("fairness")
    sus_log  = log_file("sustainability")

    fair_out = open(fair_log, "w", buffering=1, encoding="utf-8", errors="replace")
    sus_out  = open(sus_log,  "w", buffering=1, encoding="utf-8", errors="replace")

    safe_print("Logging fairness to:     ", fair_log)
    safe_print("Logging sustainability to:", sus_log)

    fairness_proc = subprocess.Popen(
        [this_exe, "--child=fairness"],
        stdout=fair_out,
        stderr=subprocess.STDOUT,
        env=os.environ.copy()
    )
    sustainability_proc = subprocess.Popen(
        [this_exe, "--child=sustainability"],
        stdout=sus_out,
        stderr=subprocess.STDOUT,
        env=os.environ.copy()
    )

    ok1 = wait_for_server(5000, "Fairness dashboard", timeout=40)
    ok2 = wait_for_server(8000, "Sustainability dashboard", timeout=40)

    if ok1 and ok2:
        safe_print("Opening frontpage...")
        webbrowser.open(f"file:///{FRONTPAGE_HTML}")
    else:
        safe_print("One or both servers failed to start. See logs in %TEMP%/wattsright_logs.")

    try:
        fairness_proc.wait()
        sustainability_proc.wait()
    except KeyboardInterrupt:
        safe_print("KeyboardInterrupt: terminating children...")
        fairness_proc.terminate()
        sustainability_proc.terminate()

if __name__ == "__main__":
    main()
