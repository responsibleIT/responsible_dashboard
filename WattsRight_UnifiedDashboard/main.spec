# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.building.api import Splash
from PyInstaller.utils.hooks import collect_all, copy_metadata
import os

block_cipher = None

# 1) Top-level Python packages we want fully collected
MODULE_PKGS = [
    "pandas", "numpy", "scipy", "sklearn", "joblib",
    "flask", "flask_cors", "flask_socketio", "werkzeug",
    "jinja2", "itsdangerous", "click", "markupsafe",
    "psutil", "engineio", "socketio", "nvidia-ml-py3",
    "setuptools",  # <-- important to satisfy pyi_rth_setuptools
]

# 2) Distributions to copy metadata for (pip names)
META_PKGS = [
    "pandas", "numpy", "scipy", "scikit-learn", "joblib",
    "Flask", "Flask-Cors", "Flask-SocketIO", "Werkzeug",
    "Jinja2", "itsdangerous", "click", "MarkupSafe",
    "psutil", "python-engineio", "python-socketio",
    "setuptools", "nvidia-ml-py3",
]

datas = [
    # Frontpage
    ('frontpage/index.html', 'frontpage'),
    ('frontpage/public', 'frontpage/public'),

    # ===== Fairness dashboard =====
    ('apps/fairness_dashboard/flask_ml', 'apps/fairness_dashboard/flask_ml'),
    ('apps/fairness_dashboard/flask_ml/app.py', 'apps/fairness_dashboard/flask_ml'),

    # ===== Sustainability backend =====
    ('apps/sustainability_dashboard/backend/src', 'apps/sustainability_dashboard/backend/src'),

    # ===== Sustainability frontend (Angular build) =====
    ('apps/sustainability_dashboard/frontend_v2/dist/browser',
     'apps/sustainability_dashboard/frontend_v2/dist/browser'),

    # Splash image
    ('splashscreen.png', '.'),
]

binaries = []
hiddenimports = []

# Pull full package contents
for pkg in MODULE_PKGS:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += ["select"]

# Metadata (best effort)
for distname in META_PKGS:
    try:
        datas += copy_metadata(distname)
    except Exception:
        pass

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest', 'distutils.tests', 'email.tests'],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ---- Splash object ----
splash = Splash("splashscreen.png",
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(20, 300),
    text_size=20,
    text_color='white'
)

exe = EXE(
    pyz,
    a.scripts, a.binaries, a.zipfiles, a.datas,
    splash,
    splash.binaries,
    name='WattsRightDashboard',
    debug=False,
    strip=False,
    upx=False,
    console=False,                 # keep True while testing
    onefile=True                   # <-- this forces single-file build
)
