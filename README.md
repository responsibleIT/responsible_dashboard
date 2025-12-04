# Watt's Right Unified Dashboard

The **Watt's Right Unified Dashboard** is a standalone desktop application that brings together two complementary dashboards — **Fairness** and **Sustainability** — into one cohesive interface.  
It allows educators, students, researchers, and practitioners to **analyze, visualize, and reflect on AI systems** from ethical, fairness, and sustainability perspectives.

This unified dashboard is designed to **run anywhere** without installation requirements — users simply double-click the generated `.exe` file to launch the app, which automatically spins up local Flask servers and opens the dashboard in the browser.

Currently the dashboard only functions on **Windows** systems.

---

## 🌍 Overview

Watt's Right aims to help AI practitioners explore responsible AI concepts through interactive analytics and visualizations.

The unified dashboard consists of:

- **Fairness Dashboard** — A Flask-based web app for exploring fairness metrics, user distributions, and comparative visualizations. It allows a developer to upload their model and data in order to visualize the impact of three main fairness metrics on the model's performance ability on groups in the data. In this process, it bridges the gap between the concepts of group and individual fairness, enabling the developer to see direct impact of their decisions on individuals in the data.
  
- **Sustainability Dashboard** — A Flask + Angular dashboard for measuring and visualizing the relationship between model performance and energy efficiency. It allows the practitioner to controllably prune a Large Language Model (LLM) meant for text-classification, facilitating the development of greener LLM solutions. In the final steps, the dashboard utilizes the system's own hardware (GPU if available, otherwise CPU) to show the effect of pruning on the model's performance on classes in the uploaded data. Finally, the pruned model can be exported as a HuggingFace model, allowing the practitioner to directly use their newly pruned model in other solutions.

The dashboards run locally and communicate through Python servers that handle user data, model metrics, and visualization endpoints.

---

## 🧱 Installation (Development Mode)

If you're running this from source (instead of the `.exe`):

### Prerequisites
- **Python 3.10+** with pip
- **Node.js 18+** with npm (required for building the Sustainability dashboard frontend)

### Setup

```bash
git clone https://github.com/responsibleIT/responsible_dashboard.git
cd responsible_dashboard/WattsRight_UnifiedDashboard
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
python main.py
```

When you run `main.py` for the first time, it will automatically:
1. Check if npm is available
2. Install Angular dependencies (`npm install`) if needed
3. Build the Sustainability frontend (`npm run build`) if needed
4. Start both dashboard servers
5. Open the frontpage in your default browser

### Command Line Options

```bash
python main.py [options]

Options:
  --build-only     Build frontend assets only, don't start servers
  --rebuild        Force rebuild of frontend assets even if they exist
  --skip-build     Skip frontend build check (use existing builds)
  --help, -h       Show help message
```

**Note:** The Fairness dashboard uses static files and requires no build step.

---

## 🧰 Building the Standalone Executable

To create a single-file `.exe` so anyone can run the dashboard without installing Python or Node.js:

### Prerequisites for Building
1. Complete the development setup above (including running `main.py` at least once to build the Angular frontend)
2. Ensure PyInstaller is installed:
   ```bash
   pip install pyinstaller
   ```

### Build Steps

```bash
cd WattsRight_UnifiedDashboard
pyinstaller main.spec
```

This generates a single executable at `dist/WattsRightDashboard.exe`.

You can distribute this file directly to others — they don't need Python, Node.js, or any dependencies installed. When launched, it will extract and run everything in a temporary directory, then clean up after closing.

---

## 🚀 Running the Dashboard (End Users)

1. Double-click `WattsRightDashboard.exe`
2. Wait for the splash screen (Watt's Right logo) to disappear
3. Your browser will open the Front Page
4. Choose either:
    - **Fairness Dashboard** (localhost:5000)
    - **Sustainability Dashboard** (localhost:8000)
5. Use the "Shut Down Application" button on the front page to safely stop background servers

---

## 📁 Project Structure

```
responsible_dashboard/
├── README.md
├── LICENSE.md
├── requirements.txt
└── WattsRight_UnifiedDashboard/
    ├── main.py                 # Main entry point
    ├── main.spec               # PyInstaller build configuration
    ├── requirements.txt        # Python dependencies
    ├── frontpage/              # Landing page HTML/CSS/JS
    └── apps/
        ├── fairness_dashboard/     # Flask app (static frontend)
        └── sustainability_dashboard/
            ├── backend/            # Flask + SocketIO backend
            └── frontend_v2/        # Angular frontend (requires build)
```

---

## 👩‍💻 Contributors

Developed in collaboration between University of Applied Sciences Amsterdam, KPN, Blue Field Agency, and BrainCreators as an exploratory KIEM-research project.

---

## 🛠️ Troubleshooting

### Executable Issues
- **Exe doesn't start:** Check for antivirus blocking or permission issues. Try right-clicking and "Run as administrator".
- **Splash screen hangs:** The app may be loading large dependencies. Wait up to 60 seconds on first launch.

### Port Issues
- **Ports 5000 or 8000 in use:** Kill those processes first, or check for other Flask/Node apps running.
- **Windows Firewall prompt:** Allow access when prompted to enable local server communication.

### Development Mode Issues
- **npm not found:** Install Node.js from https://nodejs.org/ and restart your terminal.
- **Angular build fails:** Try running `npm install` manually in `apps/sustainability_dashboard/frontend_v2/`.
- **Python dependencies missing:** Ensure you've activated the virtual environment and run `pip install -r requirements.txt`.

### Logs
Server logs are written to `%TEMP%/wattsright_logs/` (Windows). Check `fairness.log` and `sustainability.log` for detailed error messages.






