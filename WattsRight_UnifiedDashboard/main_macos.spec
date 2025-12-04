# -*- mode: python ; coding: utf-8 -*-
# macOS spec file for WattsRight Dashboard
# Build with: pyinstaller main_macos.spec

from PyInstaller.utils.hooks import collect_all, copy_metadata
import os

block_cipher = None

# 1) Top-level Python packages we want fully collected
MODULE_PKGS = [
    "pandas", "numpy", "scipy", "sklearn", "joblib",
    "flask", "flask_cors", "flask_socketio", "werkzeug",
    "jinja2", "itsdangerous", "click", "markupsafe",
    "psutil", "engineio", "socketio",
    "setuptools",
]

# 2) Distributions to copy metadata for (pip names)
META_PKGS = [
    "pandas", "numpy", "scipy", "scikit-learn", "joblib",
    "Flask", "Flask-Cors", "Flask-SocketIO", "Werkzeug",
    "Jinja2", "itsdangerous", "click", "MarkupSafe",
    "psutil", "python-engineio", "python-socketio",
    "setuptools",
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
]

binaries = []
hiddenimports = []

# Pull full package contents
for pkg in MODULE_PKGS:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

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

# macOS: Create a .app bundle
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WattsRightDashboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # No terminal window
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='WattsRightDashboard',
)

# Create macOS .app bundle
app = BUNDLE(
    coll,
    name='WattsRightDashboard.app',
    icon=None,  # Add your .icns file path here if you have one
    bundle_identifier='com.wattsright.dashboard',
    info_plist={
        'CFBundleName': 'WattsRight Dashboard',
        'CFBundleDisplayName': 'WattsRight Dashboard',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'LSBackgroundOnly': False,
    },
)
