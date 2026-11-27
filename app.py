# Entry point for PhishGuard
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# If executed via a global Python or global streamlit CLI without activating the project venv,
# automatically include the local venv site-packages so all dependencies are found.
_venv_site_packages = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if _venv_site_packages.is_dir() and str(_venv_site_packages) not in sys.path:
    sys.path.insert(1, str(_venv_site_packages))

import streamlit as st
from ui.home import show_home

show_home()