# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Micro Post application.

This configuration creates a single executable with no console window,
including all necessary resources and data files.
"""

import os
from pathlib import Path

block_cipher = None

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(SPEC))

# Define data files to include
datas = [
    # UI resources (icons, arrows)
    (os.path.join(project_root, 'src', 'ui', 'resources'), 
     os.path.join('src', 'ui', 'resources')),
    # Documentation
    (os.path.join(project_root, 'docs'), 'docs'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'pandas',
    'numpy',
    'openpyxl',
    'scipy',
    'sklearn',
    'sklearn.decomposition',
    'sklearn.decomposition._pca',
    'matplotlib',
    'matplotlib.backends.backend_qtagg',
    'cv2',
]

a = Analysis(
    [os.path.join(project_root, 'main.py')],
    pathex=[project_root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'test',
        'tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MicroPost',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(project_root, 'src', 'ui', 'resources', 'icon.ico'),
)
