"""
WISDM51 ML Pipeline - Modular data processing for sensor data.

Modules:
    - config: Central configuration
    - utils: Data loading and utilities
    - cleaning: Data quality handling
    - windowing: Time-series segmentation
    - features: Feature extraction
    - main: Pipeline orchestration

Usage:
    from pipeline.cleaning import clean_data
    from pipeline.windowing import create_windows
    from pipeline.features import extract_all_features
"""

__version__ = "1.0.0"
__author__ = "WISDM51 ML Pipeline"

from .config import *
from .utils import setup_logging, load_all_raw_data
from .cleaning import clean_data
from .windowing import create_windows
from .features import extract_all_features

__all__ = [
    'clean_data',
    'create_windows',
    'extract_all_features',
    'setup_logging',
    'load_all_raw_data',
]
