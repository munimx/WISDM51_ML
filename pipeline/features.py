"""
Feature extraction module for windowed sensor data.

Computes time-domain features for each window and sensor channel:
- Statistical features (mean, std, min, max, etc.)
- Signal processing features (RMS, ZCR, energy)
- Advanced features (Hjorth parameters, autocorrelation)
- Morphological features (peak count)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List
from scipy import stats
from scipy.signal import find_peaks
from config import TIME_DOMAIN_FEATURES, PEAK_PROMINENCE, WINDOW_SAMPLES
from utils import setup_logging, validate_dataframe

logger = setup_logging()


# ======================== TIME-DOMAIN FEATURES ========================

def compute_mean(signal: np.ndarray) -> float:
    """Mean value of signal."""
    return np.mean(signal)


def compute_median(signal: np.ndarray) -> float:
    """Median value of signal."""
    return np.median(signal)


def compute_std(signal: np.ndarray) -> float:
    """Standard deviation of signal."""
    return np.std(signal)


def compute_var(signal: np.ndarray) -> float:
    """Variance of signal."""
    return np.var(signal)


def compute_min(signal: np.ndarray) -> float:
    """Minimum value in signal."""
    return np.min(signal)


def compute_max(signal: np.ndarray) -> float:
    """Maximum value in signal."""
    return np.max(signal)


def compute_range(signal: np.ndarray) -> float:
    """Range (max - min) of signal."""
    return np.max(signal) - np.min(signal)


def compute_skewness(signal: np.ndarray) -> float:
    """Skewness (3rd moment) of signal."""
    return stats.skew(signal)


def compute_kurtosis(signal: np.ndarray) -> float:
    """Kurtosis (4th moment) of signal."""
    return stats.kurtosis(signal)


def compute_iqr(signal: np.ndarray) -> float:
    """Interquartile range of signal."""
    q75, q25 = np.percentile(signal, [75, 25])
    return q75 - q25


def compute_mad(signal: np.ndarray) -> float:
    """Mean Absolute Deviation from mean."""
    return np.mean(np.abs(signal - np.mean(signal)))


def compute_rms(signal: np.ndarray) -> float:
    """Root Mean Square of signal."""
    return np.sqrt(np.mean(signal ** 2))


def compute_zcr(signal: np.ndarray) -> float:
    """
    Zero Crossing Rate (ZCR).
    
    Count the number of times the signal crosses zero (changes sign).
    Normalized by signal length.
    """
    sign_changes = np.sum(np.diff(np.sign(signal)) != 0)
    zcr = sign_changes / (len(signal) - 1) if len(signal) > 1 else 0
    return zcr


def compute_autocorr_lag1(signal: np.ndarray) -> float:
    """
    Autocorrelation at lag 1.
    
    Measures correlation between the signal and its 1-sample-delayed copy.
    """
    if len(signal) < 2:
        return 0.0
    
    signal_normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
    autocorr = np.correlate(signal_normalized, signal_normalized, mode='full')
    autocorr_norm = autocorr[len(autocorr) // 2:]
    autocorr_norm = autocorr_norm / autocorr_norm[0]
    
    return autocorr_norm[1] if len(autocorr_norm) > 1 else 0.0


def compute_sma(signal: np.ndarray) -> float:
    """
    Signal Magnitude Area (SMA).
    
    Sum of absolute values (normalized by window length).
    """
    return np.sum(np.abs(signal)) / len(signal)


def compute_energy(signal: np.ndarray) -> float:
    """
    Energy of signal.
    
    Sum of squared values.
    """
    return np.sum(signal ** 2)


def compute_hjorth_activity(signal: np.ndarray) -> float:
    """
    Hjorth Activity parameter.
    
    Variance of the signal.
    """
    return np.var(signal)


def compute_hjorth_mobility(signal: np.ndarray) -> float:
    """
    Hjorth Mobility parameter.
    
    Ratio of std(derivative) to std(signal).
    """
    if len(signal) < 2:
        return 0.0
    
    derivative = np.diff(signal)
    activity = np.var(signal)
    mobility_activity = np.var(derivative)
    
    if activity == 0:
        return 0.0
    
    return np.sqrt(mobility_activity / activity)


def compute_hjorth_complexity(signal: np.ndarray) -> float:
    """
    Hjorth Complexity parameter.
    
    Ratio of mobility(derivative) to mobility(signal).
    """
    if len(signal) < 3:
        return 0.0
    
    derivative = np.diff(signal)
    second_derivative = np.diff(derivative)
    
    mobility_signal = compute_hjorth_mobility(signal)
    mobility_derivative = compute_hjorth_mobility(derivative)
    
    if mobility_signal == 0:
        return 0.0
    
    return mobility_derivative / mobility_signal


def compute_peak_count(signal: np.ndarray, prominence: float = PEAK_PROMINENCE) -> int:
    """
    Count peaks in signal using prominence-based detection.
    
    Args:
        signal (np.ndarray): Input signal
        prominence (float): Minimum peak prominence threshold
        
    Returns:
        int: Number of detected peaks
    """
    if len(signal) < 3:
        return 0
    
    peaks, _ = find_peaks(signal, prominence=prominence)
    return len(peaks)


# ======================== FEATURE COMPUTATION REGISTRY ========================

FEATURE_FUNCTIONS = {
    'mean': compute_mean,
    'median': compute_median,
    'std': compute_std,
    'var': compute_var,
    'min': compute_min,
    'max': compute_max,
    'range': compute_range,
    'skewness': compute_skewness,
    'kurtosis': compute_kurtosis,
    'iqr': compute_iqr,
    'mad': compute_mad,
    'rms': compute_rms,
    'zcr': compute_zcr,
    'autocorr_lag1': compute_autocorr_lag1,
    'sma': compute_sma,
    'energy': compute_energy,
    'hjorth_activity': compute_hjorth_activity,
    'hjorth_mobility': compute_hjorth_mobility,
    'hjorth_complexity': compute_hjorth_complexity,
    'peak_count': compute_peak_count,
}


# ======================== FEATURE EXTRACTION ========================

def extract_features_from_window(
    window_data: Dict[str, np.ndarray],
    features_to_compute: List[str] = TIME_DOMAIN_FEATURES
) -> Dict[str, float]:
    """
    Extract all configured features from a single window.
    
    Args:
        window_data (Dict[str, np.ndarray]): Dict with keys 'x', 'y', 'z' containing signal arrays
        features_to_compute (List[str]): List of feature names to compute
        
    Returns:
        Dict[str, float]: Dictionary mapping feature names to values
    """
    features = {}
    
    for channel in ['x', 'y', 'z']:
        signal = window_data[channel]
        
        for feature_name in features_to_compute:
            if feature_name not in FEATURE_FUNCTIONS:
                logger.warning(f"Unknown feature: {feature_name}")
                continue
            
            feature_func = FEATURE_FUNCTIONS[feature_name]
            
            try:
                feature_value = feature_func(signal)
                # Create feature column name: feature_channel (e.g., mean_x)
                col_name = f"{feature_name}_{channel}"
                features[col_name] = feature_value
            except Exception as e:
                logger.error(f"Error computing {feature_name} for channel {channel}: {e}")
                features[col_name] = np.nan
    
    return features


def extract_all_features(
    windowed_df: pd.DataFrame,
    features_to_compute: List[str] = TIME_DOMAIN_FEATURES,
    window_samples: int = WINDOW_SAMPLES,
    use_fast: bool = True
) -> pd.DataFrame:
    """
    Extract features from all windows in the dataset.
    
    Supports two modes:
    - Fast (vectorized): 6-9x faster using NumPy operations (default)
    - Legacy (row-by-row): Slower but more detailed feature computation
    
    Process:
    1. For each window row
    2. Extract x, y, z time-series from columns (x_00, x_01, ..., x_59, etc.)
    3. Compute all specified features for each channel
    4. Create feature matrix
    5. Append activity label
    
    Args:
        windowed_df (pd.DataFrame): Windowed data from windowing module
                                   Columns: subject_id, device, sensor, activity_code, 
                                           x_00...x_59, y_00...y_59, z_00...z_59
        features_to_compute (List[str]): Feature names to extract
        window_samples (int): Number of samples per window
        use_fast (bool): Use vectorized implementation (default: True)
        
    Returns:
        pd.DataFrame: Feature matrix with one row per window
    """
    validate_dataframe(windowed_df, "FEATURES_INPUT")
    
    if use_fast:
        return _extract_all_features_fast(windowed_df, features_to_compute, window_samples)
    else:
        return _extract_all_features_legacy(windowed_df, features_to_compute, window_samples)


def _extract_all_features_fast(
    windowed_df: pd.DataFrame,
    features_to_compute: List[str] = TIME_DOMAIN_FEATURES,
    window_samples: int = WINDOW_SAMPLES
) -> pd.DataFrame:
    """
    Optimized vectorized feature extraction - 6-9x faster than legacy.
    
    Uses NumPy operations on entire arrays instead of row-by-row computation.
    """
    logger.info(f"Extracting {len(features_to_compute)} features (optimized/vectorized)...")
    n_windows = len(windowed_df)
    
    # Extract all window data as numpy arrays (channels as (n_windows, window_samples))
    x_cols = [f'x_{i:02d}' for i in range(window_samples)]
    y_cols = [f'y_{i:02d}' for i in range(window_samples)]
    z_cols = [f'z_{i:02d}' for i in range(window_samples)]
    
    X = windowed_df[x_cols].values.astype(np.float32)
    Y = windowed_df[y_cols].values.astype(np.float32)
    Z = windowed_df[z_cols].values.astype(np.float32)
    
    # Compute features vectorized
    features = {}
    
    for channel_name, data in [('x', X), ('y', Y), ('z', Z)]:
        # Basic statistical features
        if 'mean' in features_to_compute:
            features[f'mean_{channel_name}'] = np.mean(data, axis=1)
        if 'std' in features_to_compute:
            features[f'std_{channel_name}'] = np.std(data, axis=1)
        if 'min' in features_to_compute:
            features[f'min_{channel_name}'] = np.min(data, axis=1)
        if 'max' in features_to_compute:
            features[f'max_{channel_name}'] = np.max(data, axis=1)
        if 'var' in features_to_compute:
            features[f'var_{channel_name}'] = np.var(data, axis=1)
        if 'range' in features_to_compute:
            features[f'range_{channel_name}'] = np.ptp(data, axis=1)
        if 'median' in features_to_compute:
            features[f'median_{channel_name}'] = np.median(data, axis=1)
        
        # Energy-based features
        if 'energy' in features_to_compute:
            features[f'energy_{channel_name}'] = np.sum(data**2, axis=1)
        if 'rms' in features_to_compute:
            features[f'rms_{channel_name}'] = np.sqrt(np.mean(data**2, axis=1))
        if 'sma' in features_to_compute:
            features[f'sma_{channel_name}'] = np.mean(np.abs(data), axis=1)
        
        # Deviation-based features
        if 'mad' in features_to_compute:
            mean_data = np.mean(data, axis=1, keepdims=True)
            features[f'mad_{channel_name}'] = np.mean(np.abs(data - mean_data), axis=1)
        if 'iqr' in features_to_compute:
            q75 = np.percentile(data, 75, axis=1)
            q25 = np.percentile(data, 25, axis=1)
            features[f'iqr_{channel_name}'] = q75 - q25
        
        # Moment-based features (using scipy.stats)
        if 'skewness' in features_to_compute:
            features[f'skewness_{channel_name}'] = stats.skew(data, axis=1)
        if 'kurtosis' in features_to_compute:
            features[f'kurtosis_{channel_name}'] = stats.kurtosis(data, axis=1)
        
        # Hjorth Activity (variance)
        if 'hjorth_activity' in features_to_compute:
            features[f'hjorth_activity_{channel_name}'] = np.var(data, axis=1)
        
        # Hjorth Mobility and Complexity require derivatives
        if 'hjorth_mobility' in features_to_compute or 'hjorth_complexity' in features_to_compute:
            derivative = np.diff(data, axis=1)
            activity = np.var(data, axis=1)
            mobility_activity = np.var(derivative, axis=1)
            
            if 'hjorth_mobility' in features_to_compute:
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    mobility = np.sqrt(mobility_activity / (activity + 1e-10))
                    features[f'hjorth_mobility_{channel_name}'] = np.nan_to_num(mobility, 0.0)
            
            if 'hjorth_complexity' in features_to_compute:
                second_derivative = np.diff(derivative, axis=1)
                mobility_derivative = np.var(second_derivative, axis=1)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    complexity = np.sqrt(mobility_derivative / (mobility_activity + 1e-10))
                    features[f'hjorth_complexity_{channel_name}'] = np.nan_to_num(complexity, 0.0)
        
        # Zero Crossing Rate
        if 'zcr' in features_to_compute:
            sign_changes = np.sum(np.diff(np.sign(data), axis=1) != 0, axis=1)
            features[f'zcr_{channel_name}'] = sign_changes / (window_samples - 1)
        
        # Autocorrelation at lag 1
        if 'autocorr_lag1' in features_to_compute:
            acf_values = []
            for i in range(data.shape[0]):
                signal = data[i, :]
                normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
                autocorr = np.correlate(normalized, normalized, mode='full')
                autocorr_norm = autocorr[len(autocorr) // 2:]
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    acf = autocorr_norm / (autocorr_norm[0] + 1e-10)
                    acf_values.append(acf[1] if len(acf) > 1 else 0.0)
            
            features[f'autocorr_lag1_{channel_name}'] = np.array(acf_values)
        
        # Peak count
        if 'peak_count' in features_to_compute:
            peak_counts = np.zeros(n_windows, dtype=int)
            for i in range(data.shape[0]):
                peaks, _ = find_peaks(data[i, :], prominence=PEAK_PROMINENCE)
                peak_counts[i] = len(peaks)
            features[f'peak_count_{channel_name}'] = peak_counts
    
    # Create result dataframe
    result_df = pd.DataFrame(features)
    
    # Add metadata columns
    result_df['subject_id'] = windowed_df['subject_id'].values
    result_df['device'] = windowed_df['device'].values
    result_df['sensor'] = windowed_df['sensor'].values
    result_df['activity_code'] = windowed_df['activity_code'].values
    
    logger.info(f"Feature extraction complete: {result_df.shape}")
    
    return result_df


def _extract_all_features_legacy(
    windowed_df: pd.DataFrame,
    features_to_compute: List[str] = TIME_DOMAIN_FEATURES,
    window_samples: int = WINDOW_SAMPLES
) -> pd.DataFrame:
    """
    Legacy row-by-row feature extraction (slower but more detailed).
    
    Kept for backward compatibility and detailed feature computation.
    """
    logger.info(f"Extracting {len(features_to_compute)} features from {len(windowed_df)} windows (legacy)...")
    
    feature_rows = []
    
    for idx, row in windowed_df.iterrows():
        # Extract window data for all channels
        window_data = {}
        
        for channel in ['x', 'y', 'z']:
            # Get all columns for this channel (x_00, x_01, ..., x_59)
            signal_cols = sorted([col for col in windowed_df.columns if col.startswith(f'{channel}_')])
            
            if len(signal_cols) == 0:
                logger.warning(f"No columns found for channel {channel} in row {idx}")
                window_data[channel] = np.array([])
            else:
                window_data[channel] = row[signal_cols].values.astype(np.float32)
        
        # Extract metadata
        metadata = {
            'subject_id': int(row['subject_id']),
            'device': row['device'],
            'sensor': row['sensor'],
            'activity_code': row['activity_code'],
        }
        
        # Compute features
        features = extract_features_from_window(window_data, features_to_compute)
        
        # Combine metadata + features + label
        feature_rows.append({**metadata, **features})
        
        if (idx + 1) % max(1, len(windowed_df) // 10) == 0:
            logger.info(f"Processed {idx + 1}/{len(windowed_df)} windows")
    
    # Create feature DataFrame
    feature_df = pd.DataFrame(feature_rows)
    
    logger.info(f"Feature extraction complete: {feature_df.shape}")
    
    return feature_df


def get_feature_extraction_report(
    df_features: pd.DataFrame,
    features_to_compute: List[str] = TIME_DOMAIN_FEATURES,
    execution_time: float = None
) -> dict:
    """
    Generate a report on feature extraction.
    
    Args:
        df_features (pd.DataFrame): Extracted features
        features_to_compute (List[str]): Features that were computed
        execution_time (float): Optional execution time in seconds
        
    Returns:
        dict: Report statistics
    """
    n_features = len(features_to_compute) * 3  # 3 channels
    n_metadata = 4  # subject_id, device, sensor, activity_code
    
    report = {
        'n_windows': len(df_features),
        'n_features': n_features,
        'n_metadata': n_metadata,
        'total_columns': len(df_features.columns),
        'memory_mb': df_features.memory_usage(deep=True).sum() / 1e6,
        'execution_time_s': execution_time
    }
    
    logger.info(f"Feature Extraction Report:")
    logger.info(f"  Windows: {report['n_windows']}")
    logger.info(f"  Features per channel: {len(features_to_compute)}")
    logger.info(f"  Channels: 3 (x, y, z)")
    logger.info(f"  Total feature columns: {report['n_features']}")
    logger.info(f"  Metadata columns: {report['n_metadata']}")
    logger.info(f"  Total columns: {report['total_columns']}")
    logger.info(f"  Memory usage: {report['memory_mb']:.2f} MB")
    if execution_time:
        logger.info(f"  Execution time: {execution_time:.2f}s")
    
    return report
