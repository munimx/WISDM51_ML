"""
Configuration file for WISDM51 ML Pipeline.

This module centralizes all configurable parameters for the data processing pipeline.
No hardcoded paths or constants should exist in other modules.
"""

from pathlib import Path

# ======================== PATHS ========================
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw"
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
DATA_DIR = PIPELINE_DIR / "data"
OUTPUT_DIR = PIPELINE_DIR / "output"

# Activity key mapping
ACTIVITY_KEY = {
    'A': 'walking',
    'B': 'jogging',
    'C': 'stairs',
    'D': 'sitting',
    'E': 'standing',
    'F': 'typing',
    'G': 'teeth_brushing',
    'H': 'eating_soup',
    'I': 'eating_chips',
    'J': 'eating_pasta',
    'K': 'drinking',
    'L': 'eating_sandwich',
    'M': 'kicking',
    'O': 'catch',
    'P': 'dribbling',
    'Q': 'writing',
    'R': 'clapping',
    'S': 'folding'
}

# ======================== SENSOR DATA ========================
SAMPLING_RATE = 20  # Hz (samples per second)
SENSOR_TYPES = ['accel', 'gyro']
DEVICES = ['phone', 'watch']
CHANNELS = ['x', 'y', 'z']

# ======================== CLEANING ========================
# Strategy for handling missing/stuck data
# Options: 'forward_fill', 'interpolate', 'drop'
MISSING_VALUE_STRATEGY = 'interpolate'

# Constant value threshold (for stuck sensor detection)
# If a value repeats for STUCK_SENSOR_THRESHOLD consecutive samples, mark as stuck
STUCK_SENSOR_THRESHOLD = 5  # samples

# ======================== WINDOWING ========================
# Window duration in seconds
WINDOW_LENGTH_SECONDS = 3  # 2s, 3s, or 5s are common choices

# Convert to samples: window_samples = window_seconds * sampling_rate
WINDOW_SAMPLES = int(WINDOW_LENGTH_SECONDS * SAMPLING_RATE)  # 60 samples for 3s at 20Hz

# Overlap ratio (0.0 to 1.0)
WINDOW_OVERLAP = 0.5  # 50% overlap

# Class consistency rule: min percentage of samples from same class
CLASS_CONSISTENCY_THRESHOLD = 0.80  # 80% threshold

# Padding strategy for short segments (< window size)
# Options: 'mean', 'mirror', 'linear_interpolation'
PADDING_STRATEGY = 'mean'

# ======================== FEATURES ========================
# Time-domain features to compute for each window and channel
# These will be computed automatically by the features module
TIME_DOMAIN_FEATURES = [
    'mean',
    'median',
    'std',
    'var',
    'min',
    'max',
    'range',
    'skewness',
    'kurtosis',
    'iqr',
    'mad',
    'rms',
    'zcr',
    'autocorr_lag1',
    'sma',
    'energy',
    'hjorth_activity',
    'hjorth_mobility',
    'hjorth_complexity',
    'peak_count'
]

# Peak detection prominence (for peak counting)
PEAK_PROMINENCE = 0.5

# ======================== OUTPUT ========================
# Final CSV file name
FINAL_CSV = OUTPUT_DIR / 'full_features.csv'
CLEANED_CSV = DATA_DIR / 'cleaned.csv'
WINDOWED_CSV = DATA_DIR / 'windowed.csv'

# ======================== LOGGING ========================
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# ======================== DATA LOADING ========================
# Subjects to process (if None, process all)
SUBJECTS_TO_PROCESS = None  # e.g., [1600, 1601] to limit processing
DEVICES_TO_PROCESS = ['phone']  # 'phone', 'watch', or both
SENSORS_TO_PROCESS = ['accel', 'gyro']  # 'accel', 'gyro', or both
