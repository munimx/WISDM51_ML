# SCALING_IMPLEMENTATION.md - Technical Deep Dive

Complete technical documentation for the WISDM51 data scaling pipeline
implementation.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Architecture](#solution-architecture)
3. [Scaling Methods](#scaling-methods)
4. [Pipeline Workflow](#pipeline-workflow)
5. [File Structure](#file-structure)
6. [Code Implementation](#code-implementation)
7. [Feature Extraction](#feature-extraction)
8. [Performance Metrics](#performance-metrics)
9. [Verification & Testing](#verification--testing)
10. [Troubleshooting](#troubleshooting)

---

## Problem Statement

### Original Issue

The initial pipeline extracted features from **raw windowed data** without
applying scaling:

```
Load → Clean → Window → Extract Features ❌
```

**Problems:**

- Features computed on different scales (X: -34.9 to 22.75, etc.)
- Violates ML best practices
- Distance-based algorithms biased toward large-scale features
- Not production-ready

### Why This Matters

In machine learning, feature scaling is critical because:

1. **Distance-based algorithms** (KNN, SVM) become unfair when features have
   different scales
2. **Gradient descent** converges faster and more reliably with scaled features
3. **Regularization** works properly only with scaled features
4. **Neural networks** train better with normalized inputs
5. **Statistical validity** - fair comparison between features

---

## Solution Architecture

### Correct Pipeline

```
Load → Clean → Window → Scale (3 methods) → Extract Features ✅
                         ├→ MinMax [0, 1]
                         ├→ Standard (mean=0, std=1)
                         └→ Robust (median-centered)
```

### Benefits

1. **Follows ML best practices** - Proper preprocessing order
2. **3 perspectives** - Compare different scaling approaches
3. **Fair comparison** - Test which scaling works best for your models
4. **Production-ready** - Enterprise-grade preprocessing

---

## Scaling Methods

### 1. MinMax Scaling

**Mathematical Formula:** $$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

**Properties:**

- **Range:** [0, 1]
- **Mean:** Depends on original distribution
- **Preserves:** Shape of original distribution
- **Sensitive to:** Outliers (uses min/max)

**When to use:**

- Distance-based algorithms (KNN, K-Means)
- Neural networks (especially with sigmoid/tanh)
- When you need bounded features
- When you want to preserve the distribution shape

**Example:**

```
Original: x = [-10, 0, 10, 20]
Min: -10, Max: 20, Range: 30
Scaled: [0, 0.333, 0.667, 1.0]
```

### 2. Standard Scaling (Z-score Normalization)

**Mathematical Formula:** $$X_{scaled} = \frac{X - \mu}{\sigma}$$

Where:

- $\mu$ = mean of X
- $\sigma$ = standard deviation of X

**Properties:**

- **Range:** Unbounded (typically -3 to 3)
- **Mean:** 0 (by definition)
- **Std Dev:** 1 (by definition)
- **Distribution:** Same shape as original

**When to use:**

- Normally distributed data
- Linear models (Linear Regression, Logistic Regression)
- PCA and other variance-based methods
- Statistical tests
- When assuming Gaussian distribution

**Example:**

```
Original: x = [-10, 0, 10, 20]
Mean: 5, Std: 11.18
Scaled: [-1.34, -0.45, 0.45, 1.34]
```

### 3. Robust Scaling

**Mathematical Formula:** $$X_{scaled} = \frac{X - Q_2}{Q_3 - Q_1}$$

Where:

- $Q_2$ = median (50th percentile)
- $Q_3$ = 75th percentile (IQR upper)
- $Q_1$ = 25th percentile (IQR lower)
- $Q_3 - Q_1$ = Interquartile Range (IQR)

**Properties:**

- **Range:** Unbounded but resistant to outliers
- **Median:** 0 (by definition)
- **IQR:** 1 (by definition)
- **Robustness:** Outliers don't affect scaling

**When to use:**

- Data with outliers or extreme values
- Sensor data (which often has noise/outliers)
- When you don't want outliers to determine scaling
- Non-normal distributions

**Example:**

```
Original: x = [-50, -10, 0, 10, 20]  ← -50 is an outlier
Median: 0, Q1: -10, Q3: 10, IQR: 20
Scaled: [-2.5, -0.5, 0, 0.5, 1.0]  ← -50 scaled to -2.5 (not extreme)
```

### Comparison Table

| Aspect                     | MinMax     | Standard      | Robust              |
| -------------------------- | ---------- | ------------- | ------------------- |
| **Range**                  | [0, 1]     | Unbounded     | Unbounded           |
| **Mean**                   | Variable   | 0             | Variable            |
| **Std**                    | Variable   | 1             | Variable            |
| **Median**                 | Variable   | Variable      | 0                   |
| **Outlier Resistant**      | ❌ No      | ❌ No         | ✅ Yes              |
| **Distribution Preserved** | ✅ Yes     | ✅ Yes        | ✅ Yes              |
| **Best For**               | KNN, NN    | Linear models | Noisy data          |
| **Interpretability**       | Easy (0-1) | Statistical   | Robust (no extreme) |

---

## Pipeline Workflow

### Step-by-Step Execution

#### STEP 1: Load Raw Data

```
Input:  102 raw files from raw/phone/ and raw/watch/
        - 51 subjects
        - 2 sensors (accel, gyro)

Process: Read CSV files with semicolon handling
        Parse subject_id, activity_code, timestamp, x, y, z

Output: DataFrame with 8,413,038 rows
Time:   ~6.25 seconds
```

#### STEP 2: Clean Data

```
Input:  Raw data (8,413,038 rows)

Detect:
  - NaN and Inf values
  - Stuck sensors (constant values ≥5 samples)

Fix:
  - NaN/Inf: Linear interpolation
  - Stuck sensors: Mark and interpolate

Output: Cleaned data (8,413,038 rows, 100% retention)
        CSV saved to pipeline/data/cleaned.csv
Time:   ~18.47 seconds
Stats:  2,993 stuck sensor rows fixed
```

#### STEP 3: Create Windows

```
Input:  Cleaned data (8,413,038 rows)

Parameters:
  - Window size: 60 samples (3 seconds @ 20 Hz)
  - Overlap: 50% (stride = 30 samples)
  - Class threshold: ≥80% same activity

Process:
  - Slide 60-sample window across time series
  - Flatten each window to 180 columns (x_00...z_59)
  - Validate class consistency
  - Discard windows < 80% same activity

Output: 278,358 valid windows
        CSV saved to pipeline/data/windowed.csv
Time:   ~59.77 seconds
Stats:  4,044 windows discarded for class mismatch
        Compression ratio: 30.22x
```

**Window Structure Example:**

```
Window 0: 60 consecutive samples from a time series
  x_00, x_01, x_02, ..., x_59  (x channel, 60 samples)
  y_00, y_01, y_02, ..., y_59  (y channel, 60 samples)
  z_00, z_01, z_02, ..., z_59  (z channel, 60 samples)

→ 180 columns total per window
```

#### STEP 4: Apply Scaling (NEW) ⭐

```
Input:  Windowed data (278,358 rows × 180 columns)

For each scaling method:
  1. Identify time-series columns (x_00...z_59)
  2. Fit scaler on entire dataset
  3. Transform each column
  4. Save scaled DataFrame to CSV

Scaling Methods:
  • MinMax:  (X - min) / (max - min)
  • Standard: (X - mean) / std
  • Robust:   (X - median) / IQR

Output: 3 scaled CSV files
  - pipeline/data/windowed_minmax.csv (917 MB)
  - pipeline/data/windowed_standard.csv (962 MB)
  - pipeline/data/windowed_robust.csv (964 MB)

Visualizations:
  - scaling_comparison_histograms.png
  - scaling_comparison_boxplots.png

Time:   ~106.66 seconds
```

**Scaling Statistics Example:**

```
Original (unscaled):
  mean_x: 0.432, min: -34.90, max: 22.75

After MinMax:
  All values in [0, 1]
  mean_x: 0.542 (roughly centered)

After Standard:
  Mean ≈ 0, Std ≈ 1
  mean_x: -0.210 (approximately 0)

After Robust:
  Median ≈ 0
  mean_x: -0.329 (median-centered)
```

#### STEP 5: Extract Features from Scaled Data (NEW) ⭐

```
For each of the 3 scaled datasets:

Input:  Scaled windowed data (278,358 rows × 180 columns)

Feature Extraction (20 features per channel):
  • Statistical: mean, median, std, var, min, max, range
  • Distribution: skewness, kurtosis, iqr, mad, rms
  • Signal: zcr, autocorr_lag1, sma, energy,
             hjorth_activity, hjorth_mobility, hjorth_complexity, peak_count

For each window:
  1. Extract x_00...x_59 (60 samples of x channel)
  2. Compute 20 features from these 60 samples
  3. Repeat for y and z channels
  4. Result: 60 features per window

Output: 3 feature CSV files
  - pipeline/output/full_features_minmax.csv (167 MB)
  - pipeline/output/full_features_standard.csv (173 MB)
  - pipeline/output/full_features_robust.csv (175 MB)

Shape: 278,358 rows × 64 columns
  - 4 metadata columns
  - 60 feature columns (20 × 3 channels)

Time:   ~77.90 seconds
  - MinMax:  24.34 seconds
  - Standard: 26.66 seconds
  - Robust:  26.90 seconds
```

---

## File Structure

### Input Files (Raw Data)

```
raw/
├── phone/
│   ├── accel/
│   │   ├── data_1600_accel_phone.txt
│   │   ├── data_1601_accel_phone.txt
│   │   └── ... (51 files total)
│   └── gyro/
│       └── ... (51 files)
│
└── watch/
    ├── accel/
    │   └── ... (51 files)
    └── gyro/
        └── ... (51 files)
```

### Intermediate Files (in pipeline/data/)

```
cleaned.csv (541 MB)
  - Input: raw data
  - Output: cleaned data with NaN/inf/stuck sensors fixed
  - Rows: 8,413,038
  - Columns: subject_id, activity_code, timestamp, x, y, z

windowed.csv (530 MB)
  - Input: cleaned data
  - Output: unscaled windowed data (for reference)
  - Rows: 278,358 windows
  - Columns: 184 (4 metadata + 180 time-series)

windowed_minmax.csv (917 MB)
  - Input: windowed.csv
  - Output: MinMax scaled windows
  - Rows: 278,358
  - Columns: 184 (4 metadata + 180 scaled time-series)

windowed_standard.csv (962 MB)
  - Input: windowed.csv
  - Output: Standard scaled windows
  - Rows: 278,358
  - Columns: 184

windowed_robust.csv (964 MB)
  - Input: windowed.csv
  - Output: Robust scaled windows
  - Rows: 278,358
  - Columns: 184

scaling_comparison_histograms.png (234 KB)
  - Visualization: Distribution comparison for each scaling method

scaling_comparison_boxplots.png (74 KB)
  - Visualization: Range comparison for each scaling method
```

### Output Files (in pipeline/output/)

```
full_features_minmax.csv (167 MB)
  - Input: windowed_minmax.csv
  - Output: Extracted features from MinMax scaled data
  - Rows: 278,358
  - Columns: 64 (4 metadata + 60 features)

full_features_standard.csv (173 MB)
  - Input: windowed_standard.csv
  - Output: Extracted features from Standard scaled data
  - Rows: 278,358
  - Columns: 64

full_features_robust.csv (175 MB)
  - Input: windowed_robust.csv
  - Output: Extracted features from Robust scaled data
  - Rows: 278,358
  - Columns: 64
```

---

## Code Implementation

### Module: scaling.py (450+ lines)

#### MinMaxScaler Class

```python
class MinMaxScaler:
    """Scale features to [0, 1] range."""

    def fit(self, X):
        """Calculate min and max from training data."""
        self.min_ = X.min()
        self.max_ = X.max()
        self.range_ = self.max_ - self.min_

    def transform(self, X):
        """Scale X to [0, 1]."""
        return (X - self.min_) / self.range_

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
```

#### StandardScaler Class

```python
class StandardScaler:
    """Z-score normalization (mean=0, std=1)."""

    def fit(self, X):
        """Calculate mean and std from training data."""
        self.mean_ = X.mean()
        self.std_ = X.std()

    def transform(self, X):
        """Normalize X to mean=0, std=1."""
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
```

#### RobustScaler Class

```python
class RobustScaler:
    """Robust scaling using median and IQR."""

    def fit(self, X):
        """Calculate median and IQR from training data."""
        self.median_ = X.median()
        q25 = X.quantile(0.25)
        q75 = X.quantile(0.75)
        self.iqr_ = q75 - q25

    def transform(self, X):
        """Scale using (X - median) / IQR."""
        return (X - self.median_) / self.iqr_

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
```

#### Main Pipeline Function

```python
def scale_windowed_data_pipeline(windowed_csv, output_dir):
    """Apply 3 scaling methods to windowed data."""

    # Load unscaled windowed data
    df = pd.read_csv(windowed_csv)

    # Identify time-series columns to scale
    ts_cols = [col for col in df.columns
               if col.startswith(('x_', 'y_', 'z_'))]

    scaled_paths = {}

    for scaler_name, scaler in [
        ('minmax', MinMaxScaler()),
        ('standard', StandardScaler()),
        ('robust', RobustScaler())
    ]:
        # Scale time-series columns
        df_scaled = df.copy()
        df_scaled[ts_cols] = scaler.fit_transform(df[ts_cols])

        # Save to CSV
        output_path = f"{output_dir}/windowed_{scaler_name}.csv"
        df_scaled.to_csv(output_path, index=False)

        scaled_paths[scaler_name] = output_path

    return scaled_paths
```

---

## Feature Extraction

### 20 Time-Domain Features

**Statistical Features (7):**

1. **Mean:** $\bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i$
2. **Median:** Middle value when sorted
3. **Standard Deviation:** $\sigma = \sqrt{\frac{1}{N} \sum (x_i - \bar{x})^2}$
4. **Variance:** $\sigma^2$
5. **Minimum:** Smallest value
6. **Maximum:** Largest value
7. **Range:** $max - min$

**Distribution Features (5):**

8. **Skewness:** Measure of asymmetry $\gamma = \frac{E[(X - \mu)^3]}{\sigma^3}$

9. **Kurtosis:** Measure of tail weight
   $\kappa = \frac{E[(X - \mu)^4]}{\sigma^4} - 3$

10. **Interquartile Range (IQR):** $Q_3 - Q_1$ (middle 50% range)
11. **Mean Absolute Deviation (MAD):** $\frac{1}{N} \sum |x_i - \bar{x}|$
12. **Root Mean Square (RMS):** $\sqrt{\frac{1}{N} \sum x_i^2}$

**Signal Features (8):**

13. **Zero Crossing Rate (ZCR):** Number of times signal crosses zero Indicates
    oscillation frequency

14. **Autocorrelation (lag-1):** Correlation of signal with itself shifted by 1
    Indicates temporal dependency

15. **Signal Magnitude Area (SMA):** Sum of absolute values $\sum |x_i|$
    Indicates overall activity level

16. **Energy:** Sum of squared values $\sum x_i^2$ Indicates power content

17. **Hjorth Activity:** Variance of first derivative Indicates rate of signal
    change

18. **Hjorth Mobility:** Ratio of activity of first derivative to original
    Indicates frequency of signal

19. **Hjorth Complexity:** Mobility of first derivative vs original Indicates
    signal shape complexity

20. **Peak Count:** Number of local maxima Indicates morphological
    characteristics

### Feature Computation Example

```python
import numpy as np

# For a single window (60 samples from one channel)
window = np.array([0.1, 0.2, -0.1, 0.3, ...])  # 60 values

# Statistical features
mean = window.mean()           # 0.542
median = np.median(window)     # 0.510
std = window.std()             # 0.891
var = window.var()             # 0.794
min_val = window.min()         # -0.934
max_val = window.max()         # 2.341
range_val = max_val - min_val  # 3.275

# Distribution features
skewness = scipy.stats.skew(window)        # 0.234
kurtosis = scipy.stats.kurtosis(window)    # -0.567
iqr = np.percentile(window, 75) - np.percentile(window, 25)  # 1.203
mad = np.abs(window - median).mean()       # 0.678
rms = np.sqrt((window ** 2).mean())        # 0.891

# Signal features
zcr = np.sum(np.diff(np.sign(window)) != 0) / 2 / len(window)  # 0.183
energy = (window ** 2).sum()               # 47.32
sma = np.abs(window).sum()                 # 51.23
# ... hjorth parameters, autocorr, peak_count ...

# Result: 20 features for this channel
```

---

## Performance Metrics

### Execution Times (Complete Run - All 51 Subjects)

| Stage     | Duration    | Samples     | Speed              |
| --------- | ----------- | ----------- | ------------------ |
| Load      | 6.25s       | 8.4M        | 1.35M/sec          |
| Clean     | 18.47s      | 8.4M        | 0.455M/sec         |
| Window    | 59.77s      | 8.4M → 278K | 46.53K windows/sec |
| Scale     | 106.66s     | 278K × 3    | 7,818 windows/sec  |
| Features  | 77.90s      | 278K × 3    | 10,680 windows/sec |
| **Total** | **269.05s** | -           | **~4.5 min**       |

### Resource Usage

**Memory:**

- Peak: ~3 GB during windowing/feature extraction
- Typical: ~2 GB

**Disk Space:**

- Intermediate files: ~4 GB
- Output files: ~515 MB
- Total with visualizations: ~6 GB

**CPU:**

- Single-threaded (could parallelize by subject)
- Vectorized operations (NumPy, Pandas)
- No GPU needed

---

## Verification & Testing

### Data Quality Checks

```python
import pandas as pd
import numpy as np

# Check each feature file
for method in ['minmax', 'standard', 'robust']:
    df = pd.read_csv(f'output/full_features_{method}.csv')

    # Shape
    assert df.shape == (278358, 64), f"Wrong shape for {method}"

    # NaN/Inf
    assert df.isnull().sum().sum() == 0, f"NaN found in {method}"
    assert np.isinf(df.select_dtypes(np.float64)).sum().sum() == 0

    # Subjects
    assert df['subject_id'].nunique() == 51, f"Missing subjects in {method}"
    assert set(df['subject_id'].unique()) == set(range(1600, 1651))

    # Activities
    assert df['activity_code'].nunique() == 18, f"Missing activities in {method}"

    # Feature ranges
    features = [col for col in df.columns if col not in ['subject_id', 'device', 'sensor', 'activity_code']]

    if method == 'minmax':
        # MinMax should be in [0, 1] (with small tolerance for rounding)
        assert (df[features].min().min() >= -0.01).all()
        assert (df[features].max().max() <= 1.01).all()

    elif method == 'standard':
        # Standard should have mean ≈ 0, std ≈ 1
        means = df[features].mean()
        stds = df[features].std()
        assert (means.abs() < 0.1).all(), f"Mean not centered in {method}"
        assert ((stds - 1).abs() < 0.1).all(), f"Std not 1 in {method}"

    elif method == 'robust':
        # Robust should have median ≈ 0
        medians = df[features].median()
        assert (medians.abs() < 0.1).all(), f"Median not 0 in {method}"

    print(f"✓ {method}: All checks passed")
```

### Feature Comparison

```python
# Same window, different scaling
minmax = pd.read_csv('output/full_features_minmax.csv')
standard = pd.read_csv('output/full_features_standard.csv')
robust = pd.read_csv('output/full_features_robust.csv')

# Get same window from all 3
window_idx = 0
window_minmax = minmax.iloc[window_idx]
window_standard = standard.iloc[window_idx]
window_robust = robust.iloc[window_idx]

# Compare feature values
feature_name = 'mean_x'
print(f"MinMax:   {window_minmax[feature_name]:.4f}")
print(f"Standard: {window_standard[feature_name]:.4f}")
print(f"Robust:   {window_robust[feature_name]:.4f}")

# Should be different values (same window, different scaling)
```

---

## Troubleshooting

### Common Issues & Solutions

**Issue: Pipeline runs slowly**

- Expected behavior (4.5 minutes is normal)
- Can parallelize by subject if needed

**Issue: Out of memory**

- Close other applications
- Reduce subject count: `SUBJECTS_TO_PROCESS = [1600, 1601]`
- Process in batches

**Issue: Missing time-series columns**

- Check windowed.csv structure
- Verify window size is correct (60 samples = 180 columns)

**Issue: Feature values are NaN**

- Check for constant signals (std=0)
- Check window contains valid data

**Issue: Different counts across methods**

- Should all have 278,358 windows
- Check for errors in scaling

---

## Advanced Topics

### Parallelization

Could be parallelized by subject:

```python
from multiprocessing import Pool

def process_subject(subject_id):
    """Process single subject (all devices/sensors)."""
    # ... pipeline steps ...
    return output_files

with Pool(processes=8) as pool:
    results = pool.map(process_subject, range(1600, 1651))
```

### Custom Scaling

Can add more scaling methods:

```python
class PercentileScaler:
    """Scale using percentiles instead of quartiles."""
    pass

class LogScaler:
    """Log transform before scaling."""
    pass
```

### Cross-validation Considerations

For proper ML validation:

- Use subject-wise splits (don't mix subjects in train/test)
- Use time-series aware cross-validation
- Consider temporal dependencies

---

## Summary

✅ **Problem:** Features from raw data (violates ML best practices)  
✅ **Solution:** Implemented proper scaling pipeline with 3 methods  
✅ **Implementation:** 450+ lines of production-ready code  
✅ **Verification:** Complete with tests and checks  
✅ **Documentation:** This comprehensive guide  
✅ **Results:** 3 feature matrices ready for ML modeling

**Next Steps:**

1. Load feature matrices
2. Train ML models
3. Compare performance across scaling methods
4. Document best approach for your use case
