"""
Project configuration — path constants for data and model directories.
"""
from pathlib import Path

# Project root (two levels up from this file: src/core/config.py → root)
BASE_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PDFS_DIR = DATA_DIR / "pdfs"

# Model directory
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist on import
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PDFS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
