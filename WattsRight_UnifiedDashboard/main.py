import os
import sys
import time
import webbrowser
import socket
import subprocess
import tempfile
import select          # ensure PyInstaller bundles the C-extension
import multiprocessing
from pathlib import Path

# --- make console tolerant to unicode when running as .exe ---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# -------- Optional PyInstaller splash (safe import) --------
def _get_pyi_splash():
    try:
        # Only import if bootloader actually created the splash
        if os.environ.get("_PYI_SPLASH") or os.environ.get("_PYI_SPLASH_IPC"):
            import pyi_splash  # provided by PyInstaller at runtime
            return pyi_splash
    except Exception:
        pass
    return None

pyi_splash = _get_pyi_splash()

def splash_update(msg: str):
    """Update splash text safely (plain text only)."""
    if pyi_splash:
        try:
            pyi_splash.update_text(msg)
        except Exception:
            pass

# Tk/tcl helps some Windows setups render the splash
try:
    import tkinter  # noqa: F401
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
    # print(f"Waiting for {name} on port {port} (timeout {timeout}s)...", flush=True)
    deadline = time.time() + timeout
    dots = 0
    while time.time() < deadline:
        if is_port_open(port):
            print(f"{name} is up on port {port}.", flush=True)
            return True
        time.sleep(0.5)
        dots += 1
    #     if dots % 4 == 0:
    #         print(".", end="", flush=True)
    # print("")  # newline
    # print(f"Timed out waiting for {name} on port {port}.", flush=True)
    return False

def run_child_script_here(script_path: str, cwd: str | None = None) -> None:
    """Run a python script path in *this* interpreter (used by child role)."""
    if cwd:
        os.chdir(cwd)
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
    import runpy
    runpy.run_path(script_path, run_name="__main__")

def run_server(role: str, script_path: str, cwd: str | None, log_name: str) -> subprocess.Popen:
    """
    Start a child process for given role.
    - When frozen: re-run this same EXE with WR_ROLE set; child will import/run script in-process.
    - From source: spawn the current python with the script path.
    """
    log_dir = Path(tempfile.gettempdir()) / "wattsright_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{log_name}.log"
    # print(f"Logging {log_name} to: {log_file}", flush=True)
    f = open(log_file, "w", buffering=1, encoding="utf-8", errors="replace")

    env = os.environ.copy()
    if getattr(sys, "frozen", False):
        # Relaunch the same exe; child branch will pick WR_ROLE and run script.
        env["WR_ROLE"] = role
        cmd = [sys.executable]
        return subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=0,
        )
    else:
        # Running from source -> just spawn python script normally
        cmd = [sys.executable, script_path]
        return subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=0,
        )

# -------- Resolve paths inside/beside the bundle --------
FAIRNESS_SCRIPT = resource_path("apps/fairness_dashboard/flask_ml/app.py")
SUSTAINABILITY_DIR = resource_path("apps/sustainability_dashboard/backend/src")
SUSTAINABILITY_SCRIPT = str(Path(SUSTAINABILITY_DIR) / "app.py")
FRONTPAGE_HTML = resource_path("frontpage/index.html")

# -------- Frontend Build Helpers --------
SUSTAINABILITY_FRONTEND_DIR = resource_path("apps/sustainability_dashboard/frontend_v2")
SUSTAINABILITY_DIST_DIR = str(Path(SUSTAINABILITY_FRONTEND_DIR) / "dist" / "browser")

def check_npm_available() -> bool:
    """Check if npm is available in the system PATH."""
    try:
        result = subprocess.run(
            ["npm", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            shell=True  # Required for Windows to find npm
        )
        return result.returncode == 0
    except Exception:
        return False

def build_sustainability_frontend() -> bool:
    """
    Build the Angular frontend for the sustainability dashboard.
    Returns True if build succeeds or already exists, False otherwise.
    """
    frontend_dir = Path(SUSTAINABILITY_FRONTEND_DIR)
    dist_index = Path(SUSTAINABILITY_DIST_DIR) / "index.html"
    node_modules = frontend_dir / "node_modules"
    
    # Check if build already exists
    if dist_index.is_file():
        print("Sustainability frontend already built.", flush=True)
        return True
    
    print("Sustainability frontend not built. Starting build process...", flush=True)
    
    # Check npm availability
    if not check_npm_available():
        print("ERROR: npm is not installed or not in PATH.", flush=True)
        print("Please install Node.js from https://nodejs.org/", flush=True)
        return False
    
    # Install dependencies if node_modules doesn't exist
    if not node_modules.is_dir():
        print("Installing npm dependencies (this may take a few minutes)...", flush=True)
        splash_update("Installing npm dependencies...")
        try:
            result = subprocess.run(
                ["npm", "install"],
                cwd=str(frontend_dir),
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for npm install
                shell=True
            )
            if result.returncode != 0:
                print(f"npm install failed:\n{result.stderr}", flush=True)
                return False
            print("npm dependencies installed.", flush=True)
        except subprocess.TimeoutExpired:
            print("npm install timed out.", flush=True)
            return False
        except Exception as e:
            print(f"npm install error: {e}", flush=True)
            return False
    
    # Build the Angular app
    print("Building Angular frontend (this may take a minute)...", flush=True)
    splash_update("Building sustainability frontend...")
    try:
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=str(frontend_dir),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for build
            shell=True
        )
        if result.returncode != 0:
            print(f"npm build failed:\n{result.stderr}", flush=True)
            return False
        print("Angular frontend built successfully.", flush=True)
        return True
    except subprocess.TimeoutExpired:
        print("npm build timed out.", flush=True)
        return False
    except Exception as e:
        print(f"npm build error: {e}", flush=True)
        return False

def ensure_frontends_built() -> bool:
    """
    Ensure all frontend builds are complete before starting servers.
    Only runs when executing from source (not frozen exe).
    Returns True if all builds succeed, False otherwise.
    """
    # Skip build step when running as frozen exe - builds should be pre-bundled
    if getattr(sys, "frozen", False):
        return True
    
    print("=" * 50, flush=True)
    print("Checking frontend builds...", flush=True)
    print("=" * 50, flush=True)
    
    # Sustainability dashboard requires Angular build
    if not build_sustainability_frontend():
        print("Failed to build sustainability frontend.", flush=True)
        return False
    
    # Fairness dashboard uses static files (no build needed)
    print("Fairness dashboard uses static files (no build needed).", flush=True)
    
    print("=" * 50, flush=True)
    print("All frontend builds complete.", flush=True)
    print("=" * 50, flush=True)
    return True

def parent_main(skip_build: bool = False) -> int:
    # ensure uploads dir exists next to exe (or current working dir)
    UPLOADS_DIR = Path("uploads")
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    splash_update("Preparing uploads directory...")
    # print(f"Uploads directory ready at: {UPLOADS_DIR.resolve()}", flush=True)

    # Build frontends if needed (only when running from source)
    if not skip_build:
        splash_update("Checking frontend builds...")
        if not ensure_frontends_built():
            splash_update("Frontend build failed!")
            print("ERROR: Frontend build failed. Cannot start application.", flush=True)
            print("Make sure Node.js/npm is installed and try again.", flush=True)
            return 1

    # print("Resolved paths:", flush=True)
    # print(f"  Fairness script:        {FAIRNESS_SCRIPT}", flush=True)
    # print(f"  Sustainability script:  {SUSTAINABILITY_SCRIPT}", flush=True)
    # print(f"  Sustainability cwd:     {SUSTAINABILITY_DIR}", flush=True)
    # print(f"  Frontpage:              {FRONTPAGE_HTML}", flush=True)

    # Sanity checks
    missing = []
    if not Path(FAIRNESS_SCRIPT).is_file():        missing.append(FAIRNESS_SCRIPT)
    if not Path(SUSTAINABILITY_SCRIPT).is_file():  missing.append(SUSTAINABILITY_SCRIPT)
    if not Path(FRONTPAGE_HTML).is_file():         missing.append(FRONTPAGE_HTML)
    if missing:
        splash_update("Missing files!")
        # print("Missing required files in the bundle:", flush=True)
        for m in missing:
            # print("  -", m)
            pass
        return 1

    # Start servers
    splash_update("Launching servers...")

    fairness_proc = run_server(
        "FAIRNESS",
        FAIRNESS_SCRIPT,
        cwd=str(Path(FAIRNESS_SCRIPT).parent),
        log_name="fairness",
    )
    sustainability_proc = run_server(
        "SUSTAINABILITY",
        SUSTAINABILITY_SCRIPT,
        cwd=SUSTAINABILITY_DIR,
        log_name="sustainability",
    )

    # Splash / progress
    splash_update("Waiting for Fairness dashboard...")
    ok1 = wait_for_server(5000, "Fairness dashboard", timeout=40)
    if ok1:
        splash_update("Fairness ready")
    else:
        splash_update("Fairness failed")

    splash_update("Waiting for Sustainability dashboard...")
    ok2 = wait_for_server(8000, "Sustainability dashboard", timeout=40)
    if ok2:
        splash_update("Sustainability ready")
    else:
        splash_update("Sustainability failed")

    if ok1 and ok2:
        splash_update("Opening frontpage...")
        # print("Opening frontpage...", flush=True)
        webbrowser.open(f"file:///{FRONTPAGE_HTML}")
        if pyi_splash:
            try:
                pyi_splash.close()
            except Exception:
                pass
    else:
        splash_update("Startup failed. Check logs.")
        # print("One or both servers failed to start. See logs in %TEMP%/wattsright_logs.", flush=True)
        if pyi_splash:
            try:
                pyi_splash.close()
            except Exception:
                pass

    # Keep parent alive; clean exit on Ctrl+C
    try:
        fairness_proc.wait()
        sustainability_proc.wait()
    except KeyboardInterrupt:
        # print("\nKeyboardInterrupt: terminating children...", flush=True)
        try: fairness_proc.terminate()
        except Exception: pass
        try: sustainability_proc.terminate()
        except Exception: pass
    return 0

def child_main(role: str) -> int:
    """
    Child branch: run target script *in this process* so frozen exe can act like python.
    """
    try:
        if role == "FAIRNESS":
            run_child_script_here(FAIRNESS_SCRIPT, cwd=str(Path(FAIRNESS_SCRIPT).parent))
        elif role == "SUSTAINABILITY":
            run_child_script_here(SUSTAINABILITY_SCRIPT, cwd=SUSTAINABILITY_DIR)
        else:
            # print(f"Unknown WR_ROLE '{role}'", flush=True)
            return 2
        return 0
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 0
    except Exception as e:
        # Child exceptions go to the role log file via stdout/err redirection
        # print(f"[Child:{role}] Unhandled error: {e}", flush=True)
        import traceback; traceback.print_exc()
        return 3

def print_usage():
    """Print usage information."""
    print("""
WattsRight Unified Dashboard

Usage: python main.py [options]

Options:
  --build-only     Build frontend assets only, don't start servers
  --rebuild        Force rebuild of frontend assets even if they exist
  --skip-build     Skip frontend build check (use existing builds)
  --help, -h       Show this help message

When running without options, the script will:
  1. Check if frontend builds exist (and build if needed)
  2. Start the Fairness and Sustainability dashboard servers
  3. Open the frontpage in your default browser
""", flush=True)

if __name__ == "__main__":
    multiprocessing.freeze_support()  # important for PyInstaller on Windows

    role = os.environ.get("WR_ROLE")
    if role:
        # Child branch: run the dashboard script and exit.
        sys.exit(child_main(role))
    else:
        # Parse command line arguments
        args = sys.argv[1:]
        
        if "--help" in args or "-h" in args:
            print_usage()
            sys.exit(0)
        
        if "--build-only" in args:
            # Just build frontends and exit
            print("Build-only mode: building frontends...", flush=True)
            if ensure_frontends_built():
                print("Build complete!", flush=True)
                sys.exit(0)
            else:
                print("Build failed!", flush=True)
                sys.exit(1)
        
        skip_build = False
        
        if "--rebuild" in args:
            # Force rebuild by removing dist directory
            dist_dir = Path(SUSTAINABILITY_DIST_DIR)
            if dist_dir.exists():
                import shutil
                print(f"Removing existing build: {dist_dir}", flush=True)
                shutil.rmtree(dist_dir, ignore_errors=True)
        
        if "--skip-build" in args:
            # Skip build check - user takes responsibility
            skip_build = True
            print("Skipping frontend build check.", flush=True)
            # Check if dist exists anyway and warn
            if not Path(SUSTAINABILITY_DIST_DIR).exists():
                print("WARNING: Sustainability frontend not built. Dashboard may not work correctly.", flush=True)
        
        # Parent branch: spawn children, wait on ports, open frontpage.
        sys.exit(parent_main(skip_build=skip_build))
