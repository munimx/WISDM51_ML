# WISDM51 Sensor Data Processing Pipeline

A fully modular, production-ready Python pipeline for processing WISDM51
smartphone and smartwatch sensor data. The pipeline converts raw accelerometer
and gyroscope time-series data into a clean, engineered feature matrix suitable
for machine learning.

## 📋 Overview

**Input:** Raw sensor data (51 subjects × 18 activities × 2 devices × 2 sensors)
**Output:** `full_features.csv` - A single feature matrix with 60+ handcrafted
features per window

### Pipeline Stages

```
Raw Data → Cleaning → Windowing → Feature Extraction → CSV Output
```

1. **Cleaning** - Remove/handle NaN, inf, stuck sensors
2. **Windowing** - Segment into 3-second windows with 50% overlap
3. **Feature Extraction** - Compute 20 time-domain features per channel (60
   total)
4. **Output** - Save as CSV for ML/statistical analysis

---

## 📁 Project Structure

```
pipeline/
├── config.py              # Central configuration (paths, parameters, thresholds)
├── utils.py               # Data loading, logging, utilities
├── cleaning.py            # Data cleaning pipeline
├── windowing.py           # Sliding-window segmentation
├── features.py            # Feature extraction (20+ features per channel)
├── main.py                # Orchestration script
├── data/                  # Intermediate outputs
│   ├── cleaned.csv
│   └── windowed.csv
└── output/
    └── full_features.csv  # FINAL OUTPUT
```

---

## 🔧 Configuration (config.py)

All parameters are centralized in `config.py`. Modify here before running:

```python
# Window settings
WINDOW_LENGTH_SECONDS = 3       # 3-second windows
WINDOW_OVERLAP = 0.5            # 50% overlap
CLASS_CONSISTENCY_THRESHOLD = 0.80  # 80% same class per window

# Cleaning settings
MISSING_VALUE_STRATEGY = 'interpolate'  # 'interpolate', 'forward_fill', 'drop'
STUCK_SENSOR_THRESHOLD = 5      # consecutive identical samples

# Feature settings
TIME_DOMAIN_FEATURES = [        # 20 features per channel
    'mean', 'median', 'std', 'var', 'min', 'max', 'range',
    'skewness', 'kurtosis', 'iqr', 'mad', 'rms', 'zcr',
    'autocorr_lag1', 'sma', 'energy', 'hjorth_activity',
    'hjorth_mobility', 'hjorth_complexity', 'peak_count'
]

# Data selection
SUBJECTS_TO_PROCESS = None      # None = all 51 subjects
DEVICES_TO_PROCESS = ['phone']  # 'phone', 'watch', or both
SENSORS_TO_PROCESS = ['accel', 'gyro']  # 'accel', 'gyro', or both
```

---

## 🚀 Usage

### Run the Full Pipeline (All Data)

```bash
cd /Users/munimahmad/Playground/WISDM51_project/pipeline
python main.py
```

**This runs the complete pipeline:**

- Loads all 51 subjects × 2 devices × 2 sensors (204 files, 15.6M samples)
- Cleans data (fixes stuck sensors, interpolates missing values)
- Creates 516,867 windows with 50% overlap
- Extracts 20 time-domain features per channel (60 total)
- Saves `output/full_features.csv` (354 MB, 516,867 rows)

**Execution Time:** ~18 minutes on a standard machine

### Run with Specific Subjects/Devices/Sensors

Edit `config.py` before running, or use Python to override:

**Option 1: Edit config.py directly**

```python
# In config.py, change these lines:
SUBJECTS_TO_PROCESS = [1600, 1601, 1602]  # Only subjects 1600-1602
DEVICES_TO_PROCESS = ['phone']              # Only phone (exclude watch)
SENSORS_TO_PROCESS = ['accel']              # Only accel (exclude gyro)
```

Then run:

```bash
python main.py
```

**Option 2: Override in Python script**

```bash
python << 'EOF'
import config
config.SUBJECTS_TO_PROCESS = [1600, 1601]
config.DEVICES_TO_PROCESS = ['phone']
config.SENSORS_TO_PROCESS = ['accel']

from main import run_pipeline
run_pipeline()
EOF
```

**Option 3: Single subject quick test**

```bash
python << 'EOF'
import config
config.SUBJECTS_TO_PROCESS = [1600]  # Just subject 1600
from main import run_pipeline
run_pipeline()  # Runs in ~4 seconds
EOF
```

### Pipeline Output

Three CSV files are generated:

1. **data/cleaned.csv** - After cleaning (same structure as raw)
2. **data/windowed.csv** - After windowing (flattened 60-sample windows)
3. **output/full_features.csv** - **FINAL** extracted feature matrix

### Expected Final CSV Structure

```
subject_id, device, sensor, activity_code, mean_x, std_x, ..., peak_count_z
1600,       phone,  accel,  A,             0.432,  1.203, ..., 3
1600,       phone,  accel,  A,             0.521,  1.156, ..., 2
...
```

**Output shape:** 516,867 rows × 64 columns

### Load and Analyze in Python

```python
import pandas as pd

# Load the final feature matrix
df = pd.read_csv('output/full_features.csv')

print(f"Shape: {df.shape}")                    # (516867, 64)
print(list(df.columns))                        # All 64 column names

# Extract features for ML
X = df.drop(['subject_id', 'device', 'sensor', 'activity_code'], axis=1)
y = df['activity_code']

print(X.shape)                                 # (516867, 60) features only
print(y.unique())                              # ['A' 'B' 'C' ... 'S']

# Filter by device/sensor
phone = df[df['device'] == 'phone']            # ~258K rows
watch = df[df['device'] == 'watch']            # ~259K rows

accel = df[df['sensor'] == 'accel']            # ~258K rows
gyro = df[df['sensor'] == 'gyro']              # ~259K rows

# Analyze activity distribution
print(df['activity_code'].value_counts().sort_index())
```

**Columns:**

- Metadata: `subject_id`, `device`, `sensor`, `activity_code`
- Features: 20 features × 3 channels = 60 feature columns
- Example: `mean_x`, `mean_y`, `mean_z`, `std_x`, `std_y`, `std_z`, ...

---

## 💻 Run Commands Cheat Sheet

| Use Case                | Command                                                                                                                       |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Full pipeline**       | `cd pipeline && python main.py`                                                                                               |
| **Single subject test** | `python -c "import config; config.SUBJECTS_TO_PROCESS=[1600]; from main import run_pipeline; run_pipeline()"`                 |
| **Phone only**          | `python -c "import config; config.DEVICES_TO_PROCESS=['phone']; from main import run_pipeline; run_pipeline()"`               |
| **Accel only**          | `python -c "import config; config.SENSORS_TO_PROCESS=['accel']; from main import run_pipeline; run_pipeline()"`               |
| **Subset of subjects**  | `python -c "import config; config.SUBJECTS_TO_PROCESS=list(range(1600,1610)); from main import run_pipeline; run_pipeline()"` |

### Example: Run on Subset (Faster Testing)

```bash
cd /Users/munimahmad/Playground/WISDM51_project/pipeline

# Test with 3 subjects, phone accel only (~30 seconds)
python << 'EOF'
import config
config.SUBJECTS_TO_PROCESS = [1600, 1601, 1602]
config.DEVICES_TO_PROCESS = ['phone']
config.SENSORS_TO_PROCESS = ['accel']
from main import run_pipeline
run_pipeline()
EOF
```

---

## 🧹 Cleaning Module (cleaning.py)

Handles data quality issues:

### Detection Methods

- **NaN/Inf** - Identifies missing or infinite values
- **Stuck Sensors** - Detects constant values (≥5 consecutive samples)
- **Outliers** - Optional IQR-based outlier detection

### Handling Strategies

- `interpolate` (default) - Linear interpolation
- `forward_fill` - Forward fill method
- `drop` - Remove problematic rows

### Example

```python
from cleaning import clean_data

df_cleaned = clean_data(
    df_raw,
    handle_missing=True,
    handle_stuck=True,
    handle_outliers=False
)
```

---

## 🪟 Windowing Module (windowing.py)

Segments time-series into fixed-length windows:

### Key Parameters

- **Window length:** 3 seconds (60 samples @ 20 Hz)
- **Overlap:** 50% (stride = 30 samples)
- **Class consistency:** ≥80% samples from same activity
- **Padding:** Mean-value padding for short segments

### Class Consistency Rule

Each window must contain ≥80% of samples from a single activity class. Windows
that don't meet this threshold are discarded to ensure label purity.

### Example

```python
from windowing import create_windows

df_windowed = create_windows(
    df_cleaned,
    window_samples=60,
    overlap=0.5,
    class_consistency_threshold=0.80
)
```

---

## 🎯 Feature Extraction Module (features.py)

Computes 20 time-domain features for each channel (x, y, z):

### Statistical Features

- `mean`, `median`, `std`, `var`, `min`, `max`, `range`
- `skewness`, `kurtosis`, `iqr` (interquartile range)
- `mad` (mean absolute deviation)

### Signal Features

- `rms` - Root Mean Square (energy-like)
- `zcr` - Zero Crossing Rate (oscillation measure)
- `sma` - Signal Magnitude Area (total activity)
- `energy` - Sum of squared values
- `autocorr_lag1` - 1-lag autocorrelation

### Advanced Features

- `hjorth_activity` - Variance (complexity measure)
- `hjorth_mobility` - Rate of change
- `hjorth_complexity` - Change of slope
- `peak_count` - Number of peaks (with prominence threshold)

### Feature Matrix

For each window:

- 20 features × 3 channels = **60 feature columns**
- 1 metadata column (activity label)
- **Total: 64 columns** (4 metadata + 60 features)

### Example

```python
from features import extract_all_features

df_features = extract_all_features(
    df_windowed,
    features_to_compute=TIME_DOMAIN_FEATURES
)
```

---

## 📊 Data Flow & Transformations

```
Raw Data (1M+ samples)
    ↓
[CLEANING]
    - Remove NaN/inf
    - Fix stuck sensors
    ↓
Cleaned Data (999k samples)
    ↓
[WINDOWING]
    - 3-sec windows @ 50% overlap
    - 80% class consistency check
    ↓
Windows (15k-20k)
    ↓
[FEATURE EXTRACTION]
    - 20 features × 3 channels
    - Time-domain only
    ↓
Feature Matrix (15k-20k × 64 columns)
    ↓
full_features.csv
```

---

## 🎛️ Modular Design

Each module is independent and reusable:

### `config.py`

- Centralized configuration
- No hardcoded paths/parameters
- Easy parameter tuning

### `utils.py`

- Data loading from raw files
- Logging setup
- Data validation
- File discovery

### `cleaning.py`

- NaN/inf detection and handling
- Stuck sensor detection
- Outlier detection
- Strategy-based fixing

### `windowing.py`

- Sliding-window extraction
- Class consistency validation
- Padding strategies
- Window statistics

### `features.py`

- 20 feature functions (statistical, signal, advanced)
- Feature registry (FEATURE_FUNCTIONS dict)
- Batch extraction
- Feature validation

### `main.py`

- Orchestration
- Step-by-step logging
- Error handling
- Final reporting

---

## 🧪 Feature Engineering Justification

### Why These Features?

1. **Statistical** - Capture signal distribution and shape
2. **Signal Processing** - Capture oscillation patterns and energy
3. **Hjorth Parameters** - Represent complexity and activity patterns
4. **Autocorrelation** - Capture temporal dependencies
5. **Peak Count** - Detect morphological characteristics

### Why Time-Domain Only?

- **Simplicity** - No FFT/frequency domain needed
- **Interpretability** - Directly relate to physical activity
- **Efficiency** - Faster computation
- **Foundation** - Baseline for ML models; frequency features can be added later

---

## 📈 Expected Output Shape

```python
>>> df_features.shape
(18247, 64)

>>> df_features.columns
Index(['subject_id', 'device', 'sensor', 'activity_code',
       'mean_x', 'mean_y', 'mean_z', 'median_x', 'median_y', 'median_z',
       ..., 'peak_count_x', 'peak_count_y', 'peak_count_z'],
      dtype='object')
```

---

## 🔒 Data Integrity

### Checks Performed

1. **Shape validation** - Each module validates input/output shapes
2. **Type validation** - Ensures correct data types
3. **Range validation** - Checks for NaN/inf in outputs
4. **Class balance** - Logs activity distribution
5. **Memory tracking** - Logs memory usage at each stage

### No Data Leakage

- Windowing respects subject/device/sensor boundaries
- Padding uses only local segment data (no cross-class borrowing)
- Feature computation is independent per window

---

## 🎓 Next Steps

After generating `full_features.csv`, you can:

1. **Scaling** - StandardScaler or MinMaxScaler
2. **Feature Selection** - SelectKBest, RFE, or domain knowledge
3. **Model Training** - Random Forest, SVM, Neural Networks, etc.
4. **Frequency Features** - Add FFT-based features if needed
5. **Cross-validation** - Subject-wise or time-based splits

---

## 🚨 Troubleshooting

### Empty Output

- Check `SUBJECTS_TO_PROCESS` and `DEVICES_TO_PROCESS` in config
- Verify raw data paths exist
- Check class consistency threshold (may be too strict)

### Memory Issues

- Process fewer subjects: `SUBJECTS_TO_PROCESS = [1600, 1601, ...]`
- Reduce window overlap: `WINDOW_OVERLAP = 0.25`
- Use only one device: `DEVICES_TO_PROCESS = ['phone']`

### Missing Features

- Ensure all 20 features are in `TIME_DOMAIN_FEATURES`
- Check for NaN in output (indicates computation error)

---

## 📝 Code Quality

- **Docstrings** - Every function has docstring with Args, Returns, Raises
- **Type hints** - Python type annotations throughout
- **Error handling** - Try-except with logging at each stage
- **Abstraction** - Functions are single-responsibility
- **Reusability** - All modules can be imported and used independently
- **Determinism** - No randomness (reproducible output)

---

## 📜 Citation

If using this pipeline, cite the original dataset:

> Gary M. Weiss, Kenichi Yoneda, and Thaier Hayajneh. "Smartphone and
> Smartwatch-Based Biometrics Using Activities of Daily Living." IEEE Access,
> 7:133190-133202, Sept. 2019.

---

## 👤 Author Notes

This pipeline is designed for:

- ✅ Data scientists and ML engineers
- ✅ Reproducible research
- ✅ Easy customization
- ✅ Large dataset handling
- ✅ Production deployment

**Not for:**

- ❌ Real-time inference (use preprocessing only)
- ❌ Online learning (batch-oriented)

---

## 📞 Support

For issues:

1. Check config.py parameters
2. Review logging output for specific errors
3. Validate input data format
4. Check available disk space

---

**Version:** 1.0 **Last Updated:** December 2025
