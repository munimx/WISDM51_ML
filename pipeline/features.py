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
    window_samples: int = WINDOW_SAMPLES
) -> pd.DataFrame:
    """
    Extract features from all windows in the dataset.
    
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
        
    Returns:
        pd.DataFrame: Feature matrix with one row per window
    """
    logger.info(f"Extracting {len(features_to_compute)} features from {len(windowed_df)} windows...")
    
    validate_dataframe(windowed_df, "FEATURES_INPUT")
    
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
    features_to_compute: List[str] = TIME_DOMAIN_FEATURES
) -> dict:
    """
    Generate a report on feature extraction.
    
    Args:
        df_features (pd.DataFrame): Extracted features
        features_to_compute (List[str]): Features that were computed
        
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
        'memory_mb': df_features.memory_usage(deep=True).sum() / 1e6
    }
    
    logger.info(f"Feature Extraction Report:")
    logger.info(f"  Windows: {report['n_windows']}")
    logger.info(f"  Features per channel: {len(features_to_compute)}")
    logger.info(f"  Channels: 3 (x, y, z)")
    logger.info(f"  Total feature columns: {report['n_features']}")
    logger.info(f"  Metadata columns: {report['n_metadata']}")
    logger.info(f"  Total columns: {report['total_columns']}")
    logger.info(f"  Memory usage: {report['memory_mb']:.2f} MB")
    
    return report
