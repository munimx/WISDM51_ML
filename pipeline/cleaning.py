"""
Data cleaning module for WISDM51 sensor data.

Handles:
- NaN and inf value detection and handling
- Stuck sensor detection (constant values)
- Outlier handling
- Data validation
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, List
from config import MISSING_VALUE_STRATEGY, STUCK_SENSOR_THRESHOLD
from utils import setup_logging, validate_dataframe

logger = setup_logging()


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect rows with NaN or inf values.
    
    Args:
        df (pd.DataFrame): Input sensor data
        
    Returns:
        pd.DataFrame: Boolean mask of problematic rows
    """
    sensor_cols = ['x', 'y', 'z']
    missing_mask = df[sensor_cols].isna().any(axis=1)
    inf_mask = df[sensor_cols].isin([np.inf, -np.inf]).any(axis=1)
    
    problem_rows = missing_mask | inf_mask
    
    logger.info(f"Detected {problem_rows.sum()} rows with NaN/inf values")
    
    return problem_rows


def detect_stuck_sensors(
    df: pd.DataFrame,
    threshold: int = STUCK_SENSOR_THRESHOLD,
    tolerance: float = 1e-6
) -> pd.DataFrame:
    """
    Detect stuck sensors (constant values for consecutive samples).
    
    A sensor is considered "stuck" if the same value repeats for >= threshold samples.
    
    Args:
        df (pd.DataFrame): Input sensor data
        threshold (int): Number of consecutive identical samples to trigger detection
        tolerance (float): Floating point comparison tolerance
        
    Returns:
        pd.DataFrame: Boolean mask of stuck sensor rows
    """
    sensor_cols = ['x', 'y', 'z']
    stuck_mask = pd.Series(False, index=df.index)
    
    for col in sensor_cols:
        # Compute differences between consecutive values
        diff = df[col].diff().abs()
        
        # Identify constant segments (difference < tolerance)
        constant_mask = diff < tolerance
        
        # Use rolling window to find segments of length >= threshold
        stuck_segments = constant_mask.rolling(window=threshold, min_periods=threshold).sum()
        stuck_mask |= stuck_segments >= threshold
    
    logger.info(f"Detected {stuck_mask.sum()} rows with stuck sensors")
    
    return stuck_mask


def handle_missing_values(
    df: pd.DataFrame,
    problem_mask: pd.Series,
    strategy: str = MISSING_VALUE_STRATEGY
) -> pd.DataFrame:
    """
    Handle missing/invalid values using specified strategy.
    
    Args:
        df (pd.DataFrame): Input sensor data
        problem_mask (pd.Series): Boolean mask of problematic rows
        strategy (str): Strategy to apply ('forward_fill', 'interpolate', 'drop')
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    if strategy == 'drop':
        df_clean = df_clean[~problem_mask].reset_index(drop=True)
        logger.info(f"Dropped {problem_mask.sum()} rows with missing values")
    
    elif strategy == 'forward_fill':
        sensor_cols = ['x', 'y', 'z']
        df_clean[sensor_cols] = df_clean[sensor_cols].fillna(method='ffill')
        df_clean[sensor_cols] = df_clean[sensor_cols].fillna(method='bfill')
        logger.info(f"Applied forward fill to handle missing values")
    
    elif strategy == 'interpolate':
        sensor_cols = ['x', 'y', 'z']
        df_clean[sensor_cols] = df_clean[sensor_cols].interpolate(method='linear')
        # Fill remaining NaN (at boundaries) with forward/backward fill
        df_clean[sensor_cols] = df_clean[sensor_cols].fillna(method='ffill').fillna(method='bfill')
        logger.info(f"Applied linear interpolation to handle missing values")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_clean


def detect_outliers_iqr(
    df: pd.DataFrame,
    iqr_multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Values beyond Q1 - 1.5*IQR and Q3 + 1.5*IQR are flagged.
    
    Args:
        df (pd.DataFrame): Input sensor data
        iqr_multiplier (float): Multiplier for IQR threshold
        
    Returns:
        pd.DataFrame: Boolean mask of outlier rows
    """
    sensor_cols = ['x', 'y', 'z']
    outlier_mask = pd.Series(False, index=df.index)
    
    for col in sensor_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_mask |= col_outliers
    
    logger.info(f"Detected {outlier_mask.sum()} outlier rows (IQR method)")
    
    return outlier_mask


def clean_data(
    df: pd.DataFrame,
    handle_missing: bool = True,
    handle_stuck: bool = True,
    handle_outliers: bool = False
) -> pd.DataFrame:
    """
    Execute complete cleaning pipeline on sensor data.
    
    Steps:
    1. Validate input
    2. Detect and handle missing/inf values
    3. Detect and handle stuck sensors
    4. (Optional) Detect and flag outliers
    
    Args:
        df (pd.DataFrame): Raw sensor data
        handle_missing (bool): Whether to handle missing values
        handle_stuck (bool): Whether to detect/handle stuck sensors
        handle_outliers (bool): Whether to detect outliers
        
    Returns:
        pd.DataFrame: Cleaned sensor data
    """
    logger.info("Starting data cleaning pipeline...")
    
    # Validate input
    validate_dataframe(df, "CLEANING_INPUT")
    
    df_clean = df.copy()
    
    # Handle missing values
    if handle_missing:
        problem_mask = detect_missing_values(df_clean)
        df_clean = handle_missing_values(df_clean, problem_mask, MISSING_VALUE_STRATEGY)
    
    # Handle stuck sensors
    if handle_stuck:
        stuck_mask = detect_stuck_sensors(df_clean)
        if stuck_mask.sum() > 0:
            logger.warning(f"Found stuck sensors. Applying interpolation fix...")
            sensor_cols = ['x', 'y', 'z']
            df_clean.loc[stuck_mask, sensor_cols] = df_clean.loc[stuck_mask, sensor_cols].interpolate(method='linear')
    
    # Detect outliers (logging only, not removing)
    if handle_outliers:
        outlier_mask = detect_outliers_iqr(df_clean)
        logger.info(f"Outliers detected (not removed): {outlier_mask.sum()} rows")
    
    # Final validation
    validate_dataframe(df_clean, "CLEANING_OUTPUT")
    
    logger.info("Data cleaning completed successfully")
    
    return df_clean


def get_cleaning_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
    """
    Generate a report comparing raw vs cleaned data.
    
    Args:
        df_raw (pd.DataFrame): Original raw data
        df_clean (pd.DataFrame): Cleaned data
        
    Returns:
        dict: Report with statistics
    """
    report = {
        'original_rows': len(df_raw),
        'cleaned_rows': len(df_clean),
        'rows_removed': len(df_raw) - len(df_clean),
        'pct_retained': 100 * len(df_clean) / len(df_raw)
    }
    
    logger.info(f"Cleaning Report:")
    logger.info(f"  Original rows: {report['original_rows']}")
    logger.info(f"  Cleaned rows: {report['cleaned_rows']}")
    logger.info(f"  Rows removed: {report['rows_removed']}")
    logger.info(f"  Retention: {report['pct_retained']:.2f}%")
    
    return report
