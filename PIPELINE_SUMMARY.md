# WISDM51 Data Processing Pipeline - Completion Summary

## ✅ Project Status: COMPLETE

The fully modular Python pipeline has been successfully built and tested with
the complete WISDM51 dataset.

---

## Pipeline Overview

The pipeline processes raw sensor data through 4 stages to produce a cleaned,
windowed, and feature-enriched dataset:

### Stage 1: Load Raw Data

- Loads all raw sensor files from the data directory
- Parses CSV format with semicolon handling
- Input: 204 raw sensor files (51 subjects × 2 devices × 2 sensors)
- Output: 15.6M+ samples

### Stage 2: Data Cleaning

- Detects and handles missing values (NaN/inf)
- Identifies stuck sensors (constant values for ≥5 samples)
- Applies linear interpolation for problematic data
- Output: 100% data retention (4,651 stuck sensor rows fixed)

### Stage 3: Windowing

- Creates 3-second sliding windows (60 samples @ 20 Hz)
- 50% overlap between consecutive windows (stride=30)
- Validates class consistency: discards windows with <80% same activity
- Output: 516,867 windows (30.24x compression ratio)

### Stage 4: Feature Extraction

- Extracts 20 time-domain features per channel
- 3 channels (accelerometer x, y, z) = 60 features total
- Features include: mean, median, std, variance, min, max, range, skewness,
  kurtosis, IQR, MAD, RMS, ZCR, autocorrelation, SMA, energy, Hjorth parameters,
  peak count

---

## Final Deliverable: `full_features.csv`

**Location:**
`/Users/munimahmad/Playground/WISDM51_project/pipeline/output/full_features.csv`

**Specifications:**

- **Size:** 354 MB
- **Rows:** 516,867 windows
- **Columns:** 64
  - 4 metadata columns: `subject_id`, `device`, `sensor`, `activity_code`
  - 60 feature columns: 20 features × 3 channels (x, y, z)

**Column Details:**

```
Metadata:
- subject_id (int): 1600-1650 (51 subjects)
- device (str): 'phone' or 'watch'
- sensor (str): 'accel' or 'gyro'
- activity_code (str): A-S (18 activities, excludes N)

Features per channel (20 × 3 channels = 60 total):
- mean_{channel}
- median_{channel}
- std_{channel}
- var_{channel}
- min_{channel}
- max_{channel}
- range_{channel}
- skewness_{channel}
- kurtosis_{channel}
- iqr_{channel} (Interquartile Range)
- mad_{channel} (Mean Absolute Deviation)
- rms_{channel} (Root Mean Square)
- zcr_{channel} (Zero Crossing Rate)
- autocorr_lag1_{channel}
- sma_{channel} (Signal Magnitude Area)
- energy_{channel}
- hjorth_activity_{channel}
- hjorth_mobility_{channel}
- hjorth_complexity_{channel}
- peak_count_{channel}
```

**Data Coverage:**

- ✅ All 51 subjects (1600-1650)
- ✅ Both devices (phone, watch)
- ✅ Both sensors (accel, gyro)
- ✅ 18 activities (A-S, no N)

---

## Project Structure

```
pipeline/
├── config.py              # Configuration hub (paths, parameters, activity mapping)
├── utils.py              # Utilities (logging, file I/O, validation)
├── cleaning.py           # Data cleaning module
├── windowing.py          # Sliding window extraction
├── features.py           # Feature engineering (20 features)
├── main.py              # Pipeline orchestration
├── __init__.py          # Package initialization
├── README.md            # Comprehensive documentation
├── data/                # Intermediate outputs
│   ├── cleaned.csv      # Stage 2 output
│   └── windowed.csv     # Stage 3 output
└── output/
    └── full_features.csv # Final deliverable ✅
```

---

## Execution Performance

### Full Dataset Run (All 51 Subjects, 2 Devices, 2 Sensors)

| Stage     | Duration        | Details                                  |
| --------- | --------------- | ---------------------------------------- |
| Load      | 11.77s          | 204 files, 15.6M samples                 |
| Clean     | 33.94s          | 4,651 stuck sensors fixed                |
| Window    | 115.68s         | 516,867 windows created, 4,044 discarded |
| Features  | 913.23s         | 20 features × 516,867 windows            |
| **Total** | **~18 minutes** | All stages combined                      |

### Single Subject Test (Subject 1600, Phone Accel)

| Stage     | Duration       |
| --------- | -------------- |
| Load      | 0.06s          |
| Clean     | 0.16s          |
| Window    | 0.33s          |
| Features  | 3.66s          |
| **Total** | **~4 seconds** |

---

## Key Implementation Details

### Windowing Algorithm

- **Strategy:** Sliding window with fixed stride
- **Window size:** 60 samples (3 seconds @ 20 Hz)
- **Stride:** 30 samples (50% overlap)
- **Padding:** Mean-value padding for short final segment
- **Class validation:** Minimum 80% samples from same activity (discards
  inconsistent windows)

### Feature Engineering Approach

- **Time-domain analysis:** No frequency-domain processing in this pipeline
- **Per-window features:** Each window generates 20 features per channel
- **Flattened structure:** Window time-series expanded to 60 individual columns
  (x_00...z_59)
- **Statistical robustness:** Includes robust statistics (median, IQR, MAD) to
  handle outliers

### Data Quality

- **Missing data:** Zero NaN/inf rows in cleaned output
- **Stuck sensors:** 4,651 rows fixed via interpolation
- **Class consistency:** 4,044 windows (0.78%) discarded for mixed activity
  windows
- **Data retention:** 100% of raw samples preserved (cleaning preserves all
  rows)

---

## Configuration Parameters

All configurable via `config.py`:

```python
# Window parameters
WINDOW_LENGTH_SECONDS = 3
WINDOW_OVERLAP = 0.5  # 50%
CLASS_CONSISTENCY_THRESHOLD = 0.80  # Minimum % of same activity

# Data path
PROJECT_ROOT = "/Users/munimahmad/Playground/WISDM51_project"
RAW_DATA_DIR = f"{PROJECT_ROOT}/raw"
OUTPUT_DIR = f"{PROJECT_ROOT}/pipeline/output"

# Subject/device/sensor filtering
SUBJECTS_TO_PROCESS = list(range(1600, 1651))  # All 51 subjects
DEVICES_TO_PROCESS = ['phone', 'watch']
SENSORS_TO_PROCESS = ['accel', 'gyro']

# Features (20 time-domain features per channel)
TIME_DOMAIN_FEATURES = [
    'mean', 'median', 'std', 'var', 'min', 'max', 'range',
    'skewness', 'kurtosis', 'iqr', 'mad', 'rms', 'zcr',
    'autocorr_lag1', 'sma', 'energy', 'hjorth_activity',
    'hjorth_mobility', 'hjorth_complexity', 'peak_count'
]
```

---

## Usage Instructions

### Run Full Pipeline (All Data)

```bash
cd pipeline
python3 main.py
```

### Run with Custom Subject/Device/Sensor

```bash
cd pipeline
python3 << 'EOF'
import config
config.SUBJECTS_TO_PROCESS = [1600, 1601]  # Subset
config.DEVICES_TO_PROCESS = ['phone']
config.SENSORS_TO_PROCESS = ['accel']

from main import run_pipeline
run_pipeline()
EOF
```

### Load and Analyze Results

```python
import pandas as pd
df = pd.read_csv('output/full_features.csv')

# 516,867 rows × 64 columns
print(df.shape)

# Columns: subject_id, device, sensor, activity_code, + 60 feature columns
print(list(df.columns))

# Access features for specific channel
x_features = [col for col in df.columns if col.endswith('_x')]
y_features = [col for col in df.columns if col.endswith('_y')]
z_features = [col for col in df.columns if col.endswith('_z')]
```

---

## What's NOT Included (As Per Requirements)

This pipeline focuses solely on data preparation. The following are NOT
included:

- ❌ Feature scaling (StandardScaler, MinMaxScaler)
- ❌ Feature selection/dimensionality reduction
- ❌ Machine learning models
- ❌ Train/test splits
- ❌ Cross-validation

These steps are designed to be performed separately using the output CSV as
input.

---

## Debugging Notes

### Fixed Issues During Development

1. **Raw data format (semicolon):** Added `rstrip(';\n')` in data loading
2. **Activity window mismatch:** Changed activity extraction to use numeric
   indices with padding
3. **Validation schema mismatch:** Updated to support both raw and windowed data
   formats
4. **Feature extraction column detection:** Updated to match zero-padded column
   naming (x_00, x_01, etc.)

### Known Warnings (Non-Critical)

Runtime warnings appear for:

- **Precision loss in skewness/kurtosis:** Occurs when data is nearly constant
  (expected with stuck sensors)
- **Invalid autocorrelation division:** Occurs when autocorrelation is zero
  (expected with constant signals)

These are normal for real sensor data and don't affect pipeline execution.

---

## Next Steps for ML Pipeline

To use this output for machine learning:

1. **Load the CSV:**

   ```python
   import pandas as pd
   df = pd.read_csv('output/full_features.csv')
   ```

2. **Extract features and labels:**

   ```python
   X = df.drop(['subject_id', 'device', 'sensor', 'activity_code'], axis=1)
   y = df['activity_code']
   ```

3. **Apply preprocessing (scaling):**

   ```python
   from sklearn.preprocessing import StandardScaler
   X_scaled = StandardScaler().fit_transform(X)
   ```

4. **Train/test split:**

   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
   ```

5. **Train classifier:**
   ```python
   from sklearn.ensemble import RandomForestClassifier
   clf = RandomForestClassifier()
   clf.fit(X_train, y_train)
   ```

---

## Documentation

Complete technical documentation available in:

- `pipeline/README.md` - Comprehensive guide with architecture, features, and
  troubleshooting

---

## Summary

✅ **Pipeline Status:** Production Ready  
✅ **Output File:** 354 MB CSV with 516,867 windows × 64 columns  
✅ **Data Coverage:** Complete (51 subjects, 2 devices, 2 sensors, 18
activities)  
✅ **Execution Time:** ~18 minutes for full dataset  
✅ **Code Quality:** Fully modular, type-hinted, extensively logged

**Ready for downstream ML pipeline development.**
