# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, copy_metadata

block_cipher = None

# 1) Modules to collect (import names)
MODULE_PKGS = [
    "pandas", "numpy", "scipy", "sklearn", "joblib",
    "flask", "flask_cors", "flask_socketio", "werkzeug",
    "jinja2", "itsdangerous", "click", "markupsafe",
    "psutil", "engineio", "socketio",
]

# 2) Distributions to copy metadata for (pip project names)
META_PKGS = [
    "pandas", "numpy", "scipy", "scikit-learn", "joblib",
    "Flask", "Flask-Cors", "Flask-SocketIO", "Werkzeug",
    "Jinja2", "itsdangerous", "click", "MarkupSafe",
    "psutil", "python-engineio", "python-socketio",
]

datas = [
    ('frontpage/index.html', 'frontpage'),
    ('frontpage/public', 'frontpage/public'),
    ('uploads', 'uploads'),
    # Keep these as files for Flask
    ('apps/fairness_dashboard/flask_ml', 'apps/fairness_dashboard/flask_ml'),
    ('apps/sustainability_dashboard/backend/src', 'apps/sustainability_dashboard/backend/src'),
]
binaries = []
hiddenimports = []

# Pull package contents
for pkg in MODULE_PKGS:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Copy metadata (don’t fail if something lacks metadata)
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
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='main',
    debug=False,
    strip=False,
    upx=True,
    console=True,   # keep True during testing
)
