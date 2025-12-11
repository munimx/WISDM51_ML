"""
WISDM51 ML Pipeline - Modular data processing for sensor data.

Modules:
    - config: Central configuration
    - utils: Data loading and utilities
    - cleaning: Data quality handling
    - windowing: Time-series segmentation
    - scaling: Data normalization (MinMax, Standard, Robust)
    - features: Feature extraction
    - main: Pipeline orchestration

Usage:
    from pipeline.cleaning import clean_data
    from pipeline.windowing import create_windows
    from pipeline.scaling import apply_scaling, scale_windowed_data_pipeline
    from pipeline.features import extract_all_features
"""

__version__ = "2.0.0"
__author__ = "WISDM51 ML Pipeline"

from .config import *
from .utils import setup_logging, load_all_raw_data
from .cleaning import clean_data
from .windowing import create_windows
from .scaling import apply_scaling, scale_windowed_data_pipeline
from .features import extract_all_features

__all__ = [
    'clean_data',
    'create_windows',
    'apply_scaling',
    'scale_windowed_data_pipeline',
    'extract_all_features',
    'setup_logging',
    'load_all_raw_data',
]
