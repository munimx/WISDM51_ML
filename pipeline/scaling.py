"""
Data scaling module for windowed sensor data.

Applies different scaling techniques to windowed time-series data:
- Min-Max Scaling (0-1 range)
- Standard Scaling (Z-score normalization)
- Robust Scaling (median and IQR-based)
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from config import WINDOW_SAMPLES, DATA_DIR
from utils import setup_logging

logger = setup_logging()


class MinMaxScaler:
    """Min-Max Scaler: scales data to [0, 1] range."""
    
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """
        Compute min and max for scaling.
        
        Args:
            X (np.ndarray): Data to fit (n_samples, n_features)
        """
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale data to [0, 1] range.
        
        Args:
            X (np.ndarray): Data to transform
            
        Returns:
            np.ndarray: Scaled data
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        # Avoid division by zero
        range_ = self.max_ - self.min_
        range_[range_ == 0] = 1.0
        
        X_scaled = (X - self.min_) / range_
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class StandardScaler:
    """Standard Scaler: Z-score normalization (mean=0, std=1)."""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Compute mean and std for scaling.
        
        Args:
            X (np.ndarray): Data to fit
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize data (Z-score).
        
        Args:
            X (np.ndarray): Data to transform
            
        Returns:
            np.ndarray: Standardized data
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        # Avoid division by zero
        std = self.std_.copy()
        std[std == 0] = 1.0
        
        X_scaled = (X - self.mean_) / std
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class RobustScaler:
    """Robust Scaler: scales using median and IQR (resistant to outliers)."""
    
    def __init__(self):
        self.median_ = None
        self.iqr_ = None
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> 'RobustScaler':
        """
        Compute median and IQR for scaling.
        
        Args:
            X (np.ndarray): Data to fit
        """
        self.median_ = np.median(X, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        self.iqr_ = q75 - q25
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale data using median and IQR.
        
        Args:
            X (np.ndarray): Data to transform
            
        Returns:
            np.ndarray: Scaled data
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        # Avoid division by zero
        iqr = self.iqr_.copy()
        iqr[iqr == 0] = 1.0
        
        X_scaled = (X - self.median_) / iqr
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


def get_time_series_columns(df: pd.DataFrame, window_samples: int = WINDOW_SAMPLES) -> List[str]:
    """
    Get the flattened time-series column names from windowed data.
    
    Args:
        df (pd.DataFrame): Windowed data
        window_samples (int): Number of samples per window
        
    Returns:
        List[str]: List of time-series column names
    """
    time_series_cols = []
    for channel in ['x', 'y', 'z']:
        for i in range(window_samples):
            col_name = f'{channel}_{i:02d}'
            if col_name in df.columns:
                time_series_cols.append(col_name)
    
    return time_series_cols


def apply_scaling(
    df: pd.DataFrame,
    scaler_type: str = 'minmax',
    window_samples: int = WINDOW_SAMPLES
) -> Tuple[pd.DataFrame, object]:
    """
    Apply scaling to windowed time-series data.
    
    Only scales the time-series columns (x_00...z_59), not metadata.
    
    Args:
        df (pd.DataFrame): Windowed data
        scaler_type (str): Type of scaler ('minmax', 'standard', 'robust')
        window_samples (int): Number of samples per window
        
    Returns:
        Tuple[pd.DataFrame, object]: Scaled dataframe and fitted scaler
    """
    logger.info(f"Applying {scaler_type} scaling to windowed data...")
    
    # Get time-series columns
    time_series_cols = get_time_series_columns(df, window_samples)
    
    if len(time_series_cols) == 0:
        raise ValueError("No time-series columns found in dataframe")
    
    logger.info(f"Found {len(time_series_cols)} time-series columns to scale")
    
    # Extract time-series data
    X = df[time_series_cols].values
    
    # Select scaler
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Fit and transform
    X_scaled = scaler.fit_transform(X)
    
    # Create scaled dataframe
    df_scaled = df.copy()
    df_scaled[time_series_cols] = X_scaled
    
    logger.info(f"{scaler_type.capitalize()} scaling complete")
    
    return df_scaled, scaler


def compare_scaling_statistics(
    df_original: pd.DataFrame,
    df_scaled: pd.DataFrame,
    scaler_type: str,
    window_samples: int = WINDOW_SAMPLES
) -> pd.DataFrame:
    """
    Compare statistics before and after scaling.
    
    Args:
        df_original (pd.DataFrame): Original windowed data
        df_scaled (pd.DataFrame): Scaled windowed data
        scaler_type (str): Type of scaling applied
        window_samples (int): Number of samples per window
        
    Returns:
        pd.DataFrame: Comparison statistics
    """
    time_series_cols = get_time_series_columns(df_original, window_samples)
    
    # Compute statistics for original data
    X_orig = df_original[time_series_cols].values
    orig_stats = {
        'mean': np.mean(X_orig),
        'std': np.std(X_orig),
        'min': np.min(X_orig),
        'max': np.max(X_orig),
        'median': np.median(X_orig),
        'q25': np.percentile(X_orig, 25),
        'q75': np.percentile(X_orig, 75)
    }
    
    # Compute statistics for scaled data
    X_scaled = df_scaled[time_series_cols].values
    scaled_stats = {
        'mean': np.mean(X_scaled),
        'std': np.std(X_scaled),
        'min': np.min(X_scaled),
        'max': np.max(X_scaled),
        'median': np.median(X_scaled),
        'q25': np.percentile(X_scaled, 25),
        'q75': np.percentile(X_scaled, 75)
    }
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Statistic': list(orig_stats.keys()),
        'Original': list(orig_stats.values()),
        f'Scaled ({scaler_type})': list(scaled_stats.values())
    })
    
    logger.info(f"\nScaling Statistics Comparison ({scaler_type}):")
    logger.info(f"\n{comparison.to_string(index=False)}")
    
    return comparison


def visualize_scaling_comparison(
    df_original: pd.DataFrame,
    df_minmax: pd.DataFrame,
    df_standard: pd.DataFrame,
    df_robust: pd.DataFrame,
    output_dir: Path,
    window_samples: int = WINDOW_SAMPLES,
    sample_size: int = 1000
):
    """
    Create visualizations comparing original and scaled data distributions.
    
    Args:
        df_original (pd.DataFrame): Original windowed data
        df_minmax (pd.DataFrame): MinMax scaled data
        df_standard (pd.DataFrame): Standard scaled data
        df_robust (pd.DataFrame): Robust scaled data
        output_dir (Path): Directory to save plots
        window_samples (int): Number of samples per window
        sample_size (int): Number of random samples to plot (for efficiency)
    """
    logger.info("Creating scaling comparison visualizations...")
    
    time_series_cols = get_time_series_columns(df_original, window_samples)
    
    # Sample data for visualization (to avoid memory issues)
    if len(df_original) > sample_size:
        sample_idx = np.random.choice(len(df_original), sample_size, replace=False)
    else:
        sample_idx = np.arange(len(df_original))
    
    X_orig = df_original.iloc[sample_idx][time_series_cols].values.flatten()
    X_minmax = df_minmax.iloc[sample_idx][time_series_cols].values.flatten()
    X_standard = df_standard.iloc[sample_idx][time_series_cols].values.flatten()
    X_robust = df_robust.iloc[sample_idx][time_series_cols].values.flatten()
    
    # Create figure with histograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribution Comparison: Before and After Scaling', fontsize=16, fontweight='bold')
    
    # Original data
    axes[0, 0].hist(X_orig, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Original Data', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # MinMax scaled
    axes[0, 1].hist(X_minmax, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Min-Max Scaled [0, 1]', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Standard scaled
    axes[1, 0].hist(X_standard, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Standard Scaled (Z-score)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Robust scaled
    axes[1, 1].hist(X_robust, bins=50, color='plum', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Robust Scaled (Median & IQR)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'scaling_comparison_histograms.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved histogram comparison to: {plot_path}")
    plt.close()
    
    # Create boxplot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_to_plot = [X_orig, X_minmax, X_standard, X_robust]
    labels = ['Original', 'Min-Max', 'Standard (Z-score)', 'Robust']
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, showfliers=False)
    
    # Color each box
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'plum']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Boxplot Comparison: Before and After Scaling', fontsize=14, fontweight='bold')
    ax.set_ylabel('Value Range')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Save boxplot
    plot_path = output_dir / 'scaling_comparison_boxplots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved boxplot comparison to: {plot_path}")
    plt.close()


def scale_windowed_data_pipeline(
    windowed_csv: Path,
    output_dir: Path,
    window_samples: int = WINDOW_SAMPLES,
    create_visualizations: bool = True
) -> Dict[str, Path]:
    """
    Complete scaling pipeline: load windowed data, apply 3 scaling methods, save results.
    
    Args:
        windowed_csv (Path): Path to windowed.csv
        output_dir (Path): Directory to save scaled files
        window_samples (int): Number of samples per window
        create_visualizations (bool): Whether to create comparison plots
        
    Returns:
        Dict[str, Path]: Dictionary mapping scaler types to output file paths
    """
    logger.info("=" * 80)
    logger.info("DATA SCALING PIPELINE")
    logger.info("=" * 80)
    
    # Load windowed data
    logger.info(f"Loading windowed data from: {windowed_csv}")
    df_windowed = pd.read_csv(windowed_csv)
    logger.info(f"Loaded {len(df_windowed)} windows with {len(df_windowed.columns)} columns")
    
    # Apply three scaling methods
    scaling_methods = ['minmax', 'standard', 'robust']
    scaled_dfs = {}
    scaled_paths = {}
    
    for scaler_type in scaling_methods:
        # Apply scaling
        df_scaled, scaler = apply_scaling(df_windowed, scaler_type, window_samples)
        scaled_dfs[scaler_type] = df_scaled
        
        # Save scaled data
        output_path = output_dir / f'windowed_{scaler_type}.csv'
        df_scaled.to_csv(output_path, index=False)
        scaled_paths[scaler_type] = output_path
        logger.info(f"Saved {scaler_type} scaled data to: {output_path}")
        
        # Compare statistics
        compare_scaling_statistics(df_windowed, df_scaled, scaler_type, window_samples)
    
    # Create visualizations
    if create_visualizations:
        visualize_scaling_comparison(
            df_windowed,
            scaled_dfs['minmax'],
            scaled_dfs['standard'],
            scaled_dfs['robust'],
            output_dir,
            window_samples
        )
    
    logger.info("=" * 80)
    logger.info("SCALING COMPLETE")
    logger.info("=" * 80)
    
    return scaled_paths


def get_scaling_report(scaled_paths: Dict[str, Path]) -> dict:
    """
    Generate a report on scaling results.
    
    Args:
        scaled_paths (Dict[str, Path]): Dictionary of scaler types to file paths
        
    Returns:
        dict: Report statistics
    """
    report = {
        'num_scaling_methods': len(scaled_paths),
        'scaling_methods': list(scaled_paths.keys()),
        'output_files': {k: str(v) for k, v in scaled_paths.items()}
    }
    
    logger.info(f"Scaling Report:")
    logger.info(f"  Number of scaling methods: {report['num_scaling_methods']}")
    logger.info(f"  Methods: {', '.join(report['scaling_methods'])}")
    logger.info(f"  Output files generated: {len(report['output_files'])}")
    
    return report
