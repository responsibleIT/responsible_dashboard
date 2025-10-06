# Watt’s Right Unified Dashboard

The **Watt’s Right Unified Dashboard** is a standalone desktop application that brings together two complementary dashboards — **Fairness** and **Sustainability** — into one cohesive interface.  
It allows educators, students, researchers and practicioners to **analyze, visualize, and reflect on AI systems** from ethical, fairness, and sustainability perspectives.

This unified dashboard is designed to **run anywhere** without installation requirements — users simply double-click the generated `.exe` file to launch the app, which automatically spins up local Flask servers and opens the dashboard in the browser.

Currently the dashboard only functions on **Windows** systems

---

## 🌍 Overview

Watt’s Right aims to help AI practicioners explore responsible AI concepts through interactive analytics and visualizations.

The unified dashboard consists of:

- **Fairness Dashboard** — a Flask-based web app for exploring fairness metrics, user distributions, and comparative visualizations. It allows a developer to upload their model and data in order to visualize the impact of three main fairness metrics
on the model's performance ability on groups in the data. In this process, it bridges the gap between the concepts group and individual fairness, enabling the developer to see direct impact of their decisions on individuals in the data.
  
- **Sustainability Dashboard** — a Flask-based + Angular dashboard for measuring and visualizing the relationship between model performance and energy-efficiëncy. It allows the practicioner to controllably prune a Large Language Model (LLM) meant
for text-classification, facilitating develop greener LLM solutions. In the final steps, the dashboard utilises the systems' own hardware (GPU if able, otherwise CPU) to show the effect of pruning on the model's performance on
classes in the uploaded data. Finally, the pruned model can be exported as a HuggingFace model, allowing the practicioner to directly use their newly pruned model in other solutions.

The dashboards run locally and communicate through Python servers that handle user data, model metrics, and visualization endpoints.

--

## 🧱 Installation (Development Mode)

If you’re running this from source (instead of the `.exe`):

```bash
git clone https://github.com/<your-username>/wattsright-unified-dashboard.git
cd wattsright-unified-dashboard
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
python main.py
```

--

## 🧰 Building the Standalone Executable

To create a single-file .exe so anyone can run the dashboard without installing Python:

1. Make sure PyInstaller is installed
```bash
pip install pyinstaller
```
2. Build the .exe using the provided .spec file
```bash
pyinstaller main.spec
```
3. This generates a single executable under **dist/WattsRightDashboard.exe**

You can send this file directly to others — they don’t need Python or dependencies installed.
When launched, it will extract and run everything in a temporary directory, then clean up after closing.

--

## 🚀 Running the Dashboard (End Users)

1. Double-click WattsRightDashboard.exe
2. Wait for the splash screen (Watt’s Right logo) to disappear
3. Your browser will open the Front Page
4. Choose either:
    - Fairness Dashboard (localhost:5000)
    - Sustainability Dashboard (localhost:8000)
5. Use the “Shut Down Application” button on the front page to safely stop background servers

--

## 👩‍💻 Contributors
Developed in collaboration between University of Applied Sciences Amsterdam, KPN, Blue Field Agency and BrainCreators as an exploratory KIEM-research project.

--

## 🛠️ Troubleshooting

- If the .exe doesn’t start: check for antivirus blocking or permission issues.
- If ports 5000 or 8000 are in use, kill those processes first.
- The splash screen may display dependency loading messages — this is expected.






