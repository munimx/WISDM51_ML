"""
Windowing module for time-series segmentation.

Implements sliding-window segmentation with:
- Configurable window length and overlap
- Class consistency validation
- Padding strategies for short segments
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional
from config import (
    SAMPLING_RATE, WINDOW_SAMPLES, WINDOW_OVERLAP,
    CLASS_CONSISTENCY_THRESHOLD, PADDING_STRATEGY
)
from utils import setup_logging, validate_dataframe

logger = setup_logging()


def compute_window_stride(window_samples: int, overlap_ratio: float) -> int:
    """
    Compute the stride (step size) between windows given overlap ratio.
    
    Args:
        window_samples (int): Number of samples per window
        overlap_ratio (float): Overlap as fraction (0.0 to 1.0)
        
    Returns:
        int: Stride in samples
    """
    stride = int(window_samples * (1 - overlap_ratio))
    return stride


def extract_windows_with_stride(
    data: np.ndarray,
    window_samples: int,
    stride: int,
    padding_strategy: str = PADDING_STRATEGY,
    is_activity: bool = False
) -> List[np.ndarray]:
    """
    Extract sliding windows from 1D data array.
    
    Args:
        data (np.ndarray): 1D sensor data array or activity codes array
        window_samples (int): Window length in samples
        stride (int): Step size between windows
        padding_strategy (str): Strategy for padding short last window
        is_activity (bool): If True, don't apply padding (just slice for activities)
        
    Returns:
        List[np.ndarray]: List of windows, each of shape (window_samples,)
    """
    windows = []
    n_samples = len(data)
    
    for start_idx in range(0, n_samples - window_samples + 1, stride):
        end_idx = start_idx + window_samples
        window = data[start_idx:end_idx]
        windows.append(window)
    
    # Handle last incomplete window (only for numeric data, not for activities)
    if not is_activity:
        remaining = n_samples % stride
        if remaining > 0 and remaining < window_samples:
            last_segment = data[-remaining:]
            padded_window = pad_window(last_segment, window_samples, padding_strategy)
            windows.append(padded_window)
    
    return windows


def pad_window(
    segment: np.ndarray,
    target_length: int,
    strategy: str = PADDING_STRATEGY
) -> np.ndarray:
    """
    Pad a short segment to target length.
    
    Args:
        segment (np.ndarray): Short data segment
        target_length (int): Desired output length
        strategy (str): Padding strategy ('mean', 'mirror', 'linear_interpolation')
        
    Returns:
        np.ndarray: Padded segment of length target_length
    """
    if len(segment) == target_length:
        return segment
    
    pad_length = target_length - len(segment)
    
    if strategy == 'mean':
        pad_value = np.mean(segment)
        padding = np.full(pad_length, pad_value, dtype=segment.dtype)
        return np.concatenate([segment, padding])
    
    elif strategy == 'mirror':
        # Mirror the available data
        if len(segment) > 0:
            mirrored = segment[::-1]
            n_repeats = (pad_length // len(segment)) + 1
            padding = np.tile(mirrored, n_repeats)[:pad_length]
        else:
            padding = np.zeros(pad_length, dtype=segment.dtype)
        return np.concatenate([segment, padding])
    
    elif strategy == 'linear_interpolation':
        # Interpolate linearly from last value
        x_old = np.arange(len(segment))
        x_new = np.linspace(0, len(segment) - 1, target_length)
        interpolated = np.interp(x_new, x_old, segment)
        return interpolated
    
    else:
        raise ValueError(f"Unknown padding strategy: {strategy}")


def validate_window_class_consistency(
    activities: np.ndarray,
    threshold: float = CLASS_CONSISTENCY_THRESHOLD
) -> bool:
    """
    Check if a window satisfies class consistency (≥80% from same class).
    
    Args:
        activities (np.ndarray): Array of activity codes for the window
        threshold (float): Minimum fraction from dominant class
        
    Returns:
        bool: True if consistent, False otherwise
    """
    unique, counts = np.unique(activities, return_counts=True)
    max_count = np.max(counts)
    consistency = max_count / len(activities)
    
    return consistency >= threshold


def get_window_label(
    activities: np.ndarray,
    threshold: float = CLASS_CONSISTENCY_THRESHOLD
) -> Optional[str]:
    """
    Get the class label for a window.
    
    Returns the dominant class if consistency threshold is met, else None.
    
    Args:
        activities (np.ndarray): Array of activity codes for the window
        threshold (float): Minimum fraction from dominant class
        
    Returns:
        Optional[str]: Activity code of dominant class, or None if inconsistent
    """
    if not validate_window_class_consistency(activities, threshold):
        return None
    
    unique, counts = np.unique(activities, return_counts=True)
    dominant_class = unique[np.argmax(counts)]
    
    return dominant_class


def create_windows(
    df: pd.DataFrame,
    window_samples: int = WINDOW_SAMPLES,
    overlap: float = WINDOW_OVERLAP,
    class_consistency_threshold: float = CLASS_CONSISTENCY_THRESHOLD,
    padding_strategy: str = PADDING_STRATEGY
) -> pd.DataFrame:
    """
    Create sliding windows from cleaned sensor data.
    
    Process:
    1. Group by (subject, device, sensor)
    2. For each group, extract sliding windows with overlap
    3. Validate class consistency (≥80% samples from same class)
    4. Discard windows that don't meet consistency threshold
    5. Append class label as last column
    
    Output format: Each row is one window with metadata + flattened time-series
    
    Args:
        df (pd.DataFrame): Cleaned sensor data
        window_samples (int): Window length in samples
        overlap (float): Overlap ratio (0.0-1.0)
        class_consistency_threshold (float): Min fraction for class consistency
        padding_strategy (str): Strategy for padding short windows
        
    Returns:
        pd.DataFrame: Windowed data with windows as rows
                     Columns: subject_id, device, sensor, activity_code, 
                             x_0, x_1, ..., x_59, y_0, y_1, ..., y_59, z_0, z_1, ..., z_59
    """
    logger.info(f"Creating windows: {window_samples} samples, {overlap*100:.1f}% overlap")
    
    validate_dataframe(df, "WINDOWING_INPUT")
    
    stride = compute_window_stride(window_samples, overlap)
    logger.info(f"Window stride: {stride} samples")
    
    # Group by subject, device, sensor to maintain data integrity
    groupby_cols = ['subject_id', 'device', 'sensor']
    all_windows = []
    discarded_windows = 0
    
    for group_key, group_df in df.groupby(groupby_cols):
        subject_id, device, sensor = group_key
        
        # Extract raw sensor values and activities
        group_df = group_df.reset_index(drop=True)
        x_data = group_df['x'].values
        y_data = group_df['y'].values
        z_data = group_df['z'].values
        activity_data = group_df['activity_code'].values
        
        # Create windows for each channel
        x_windows = extract_windows_with_stride(x_data, window_samples, stride, padding_strategy)
        y_windows = extract_windows_with_stride(y_data, window_samples, stride, padding_strategy)
        z_windows = extract_windows_with_stride(z_data, window_samples, stride, padding_strategy)
        
        # For activities, convert to numeric indices to allow padding, then convert back
        # This ensures activity_windows has same length as other channels
        activity_indices = np.arange(len(activity_data))
        activity_index_windows = extract_windows_with_stride(activity_indices, window_samples, stride, padding_strategy)
        activity_windows = [[activity_data[int(idx)] for idx in indices] for indices in activity_index_windows]
        
        # Combine windows and validate class consistency
        n_windows = len(x_windows)
        
        for i in range(n_windows):
            # Get class label
            window_activities = activity_windows[i]
            window_label = get_window_label(window_activities, class_consistency_threshold)
            
            # Discard if class inconsistent
            if window_label is None:
                discarded_windows += 1
                continue
            
            # Create window row: metadata + flattened time-series + label
            window_row = {
                'subject_id': subject_id,
                'device': device,
                'sensor': sensor,
            }
            
            # Add time-series values as separate columns (flattened window)
            for j in range(len(x_windows[i])):
                window_row[f'x_{j:02d}'] = x_windows[i][j]
                window_row[f'y_{j:02d}'] = y_windows[i][j]
                window_row[f'z_{j:02d}'] = z_windows[i][j]
            
            # Add label last
            window_row['activity_code'] = window_label
            
            all_windows.append(window_row)
    
    # Create windowed DataFrame
    windowed_df = pd.DataFrame(all_windows)
    
    logger.info(f"Created {len(windowed_df)} windows (discarded {discarded_windows} inconsistent windows)")
    logger.info(f"Windowed data shape: {windowed_df.shape}")
    
    return windowed_df


def get_windowing_report(df_input: pd.DataFrame, df_windowed: pd.DataFrame) -> dict:
    """
    Generate a report on windowing results.
    
    Args:
        df_input (pd.DataFrame): Input cleaned data
        df_windowed (pd.DataFrame): Windowed output
        
    Returns:
        dict: Report statistics
    """
    report = {
        'input_samples': len(df_input),
        'output_windows': len(df_windowed),
        'compression_ratio': len(df_input) / max(len(df_windowed), 1)
    }
    
    logger.info(f"Windowing Report:")
    logger.info(f"  Input samples: {report['input_samples']}")
    logger.info(f"  Output windows: {report['output_windows']}")
    logger.info(f"  Compression ratio: {report['compression_ratio']:.2f}x")
    
    return report
