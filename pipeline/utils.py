"""
Utility functions for data loading, validation, and logging.

This module provides helper functions used throughout the pipeline.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO
from typing import List, Tuple, Optional, Dict
from config import (
    RAW_DATA_DIR, ACTIVITY_KEY, SAMPLING_RATE, 
    DEVICES_TO_PROCESS, SENSORS_TO_PROCESS, SUBJECTS_TO_PROCESS, LOG_LEVEL
)

# ======================== LOGGING ========================
def setup_logging(level: str = LOG_LEVEL) -> logging.Logger:
    """
    Configure and return a logger for the pipeline.
    
    Args:
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, level))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


# ======================== DATA LOADING ========================
def load_raw_sensor_file(filepath: Path) -> pd.DataFrame:
    """
    Load a single raw sensor data file.
    
    Expected format:
    subject_id,activity_code,timestamp,x,y,z;
    
    Note: Raw files may have trailing semicolon that needs to be stripped.
    
    Args:
        filepath (Path): Path to the raw sensor CSV file
        
    Returns:
        pd.DataFrame: Loaded sensor data with standardized columns
    """
    # Read file and strip trailing semicolons
    with open(filepath, 'r') as f:
        lines = [line.rstrip(';\n') for line in f]
    
    data_str = '\n'.join(lines)
    df = pd.read_csv(StringIO(data_str), sep=',', header=None)
    df.columns = ['subject_id', 'activity_code', 'timestamp', 'x', 'y', 'z']
    
    # Convert data types
    df['subject_id'] = df['subject_id'].astype(int)
    df['timestamp'] = df['timestamp'].astype(np.int64)
    df['x'] = df['x'].astype(np.float32)
    df['y'] = df['y'].astype(np.float32)
    df['z'] = df['z'].astype(np.float32)
    
    return df


def discover_raw_files(
    devices: Optional[List[str]] = None,
    sensors: Optional[List[str]] = None,
    subjects: Optional[List[int]] = None
) -> Dict[str, List[Path]]:
    """
    Discover all raw sensor files in the raw data directory.
    
    Args:
        devices (Optional[List[str]]): Devices to include ('phone', 'watch')
        sensors (Optional[List[str]]): Sensors to include ('accel', 'gyro')
        subjects (Optional[List[int]]): Subject IDs to include (e.g., [1600, 1601])
        
    Returns:
        Dict[str, List[Path]]: Dictionary mapping (device, sensor) tuples to file lists
    """
    devices = devices or DEVICES_TO_PROCESS
    sensors = sensors or SENSORS_TO_PROCESS
    subjects = subjects or SUBJECTS_TO_PROCESS
    
    files = {}
    
    for device in devices:
        for sensor in sensors:
            sensor_dir = RAW_DATA_DIR / device / sensor
            
            if not sensor_dir.exists():
                logger = setup_logging()
                logger.warning(f"Directory not found: {sensor_dir}")
                continue
            
            pattern = f"data_*_{sensor}_{device}.txt"
            file_list = sorted(sensor_dir.glob(pattern))
            
            # Filter by subject if specified
            if subjects:
                file_list = [
                    f for f in file_list 
                    if int(f.stem.split('_')[1]) in subjects
                ]
            
            key = f"{device}_{sensor}"
            files[key] = file_list
    
    return files


def load_all_raw_data(
    devices: Optional[List[str]] = None,
    sensors: Optional[List[str]] = None,
    subjects: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load and concatenate all raw sensor data from specified devices/sensors/subjects.
    
    Args:
        devices (Optional[List[str]]): Devices to include
        sensors (Optional[List[str]]): Sensors to include
        subjects (Optional[List[int]]): Subject IDs to include
        
    Returns:
        pd.DataFrame: Combined raw data with 'device' and 'sensor' columns added
    """
    logger = setup_logging()
    files = discover_raw_files(devices, sensors, subjects)
    
    all_data = []
    
    for key, file_list in files.items():
        device, sensor = key.split('_')
        logger.info(f"Loading {len(file_list)} files for {device} {sensor}...")
        
        for filepath in file_list:
            try:
                df = load_raw_sensor_file(filepath)
                df['device'] = device
                df['sensor'] = sensor
                all_data.append(df)
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
    
    if not all_data:
        raise ValueError("No data files found. Check paths and filter parameters.")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded total {len(combined_df)} samples from {len(all_data)} files")
    
    return combined_df


def validate_dataframe(df: pd.DataFrame, stage: str) -> None:
    """
    Validate DataFrame structure and content.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        stage (str): Pipeline stage name (for logging)
        
    Raises:
        ValueError: If validation fails
    """
    logger = setup_logging()
    
    if df.empty:
        raise ValueError(f"[{stage}] Empty DataFrame")
    
    # Different validation schemas for different stages
    if stage == "FEATURES_INPUT":
        # Windowed/flattened data: should have x_00, x_01, ..., x_59, etc.
        required_base = {'subject_id', 'activity_code'}
        has_flattened = any(col.startswith('x_') for col in df.columns) and \
                       any(col.startswith('y_') for col in df.columns) and \
                       any(col.startswith('z_') for col in df.columns)
        
        missing_base = required_base - set(df.columns)
        if missing_base or not has_flattened:
            raise ValueError(f"[{stage}] Missing required flattened time-series columns (x_00...z_59)")
    else:
        # Raw data: should have x, y, z, timestamp
        required_cols = {'subject_id', 'activity_code', 'timestamp', 'x', 'y', 'z'}
        missing_cols = required_cols - set(df.columns)
        
        if missing_cols:
            raise ValueError(f"[{stage}] Missing columns: {missing_cols}")
    
    logger.info(f"[{stage}] DataFrame validation passed: {len(df)} rows, {len(df.columns)} cols")


def map_activity_code_to_label(activity_code: str) -> str:
    """
    Map activity code (A-S) to human-readable label.
    
    Args:
        activity_code (str): Single character activity code
        
    Returns:
        str: Human-readable activity name
        
    Raises:
        KeyError: If code not found in activity key
    """
    if activity_code not in ACTIVITY_KEY:
        raise KeyError(f"Unknown activity code: {activity_code}")
    return ACTIVITY_KEY[activity_code]


# ======================== STATISTICS ========================
def print_data_summary(df: pd.DataFrame) -> None:
    """
    Print summary statistics of the DataFrame.
    
    Args:
        df (pd.DataFrame): Data to summarize
    """
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("DATA SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Subjects: {df['subject_id'].nunique()}")
    logger.info(f"Activities: {df['activity_code'].nunique()}")
    logger.info(f"Devices: {df.get('device', 'N/A').unique()}")
    logger.info(f"Sensors: {df.get('sensor', 'N/A').unique()}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    logger.info("=" * 60)
