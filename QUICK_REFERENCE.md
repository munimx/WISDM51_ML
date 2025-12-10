# WISDM51 Pipeline - Quick Reference

## 📊 Final Output

**File:** `pipeline/output/full_features.csv`  
**Size:** 354 MB | **Rows:** 516,867 | **Columns:** 64

## 🚀 Quick Start

```bash
cd /Users/munimahmad/Playground/WISDM51_project/pipeline
python3 main.py
```

## 📁 Output Structure

```
Metadata (4 columns):
├── subject_id      → 1600-1650 (51 subjects)
├── device          → 'phone' or 'watch'
├── sensor          → 'accel' or 'gyro'
└── activity_code   → 'A'-'S' (18 activities)

Features (60 columns = 20 features × 3 channels):
├── Channel X: mean_x, median_x, std_x, var_x, min_x, max_x, ...
├── Channel Y: mean_y, median_y, std_y, var_y, min_y, max_y, ...
└── Channel Z: mean_z, median_z, std_z, var_z, min_z, max_z, ...

Each channel has 20 features:
mean, median, std, var, min, max, range, skewness, kurtosis, iqr,
mad, rms, zcr, autocorr_lag1, sma, energy, hjorth_activity,
hjorth_mobility, hjorth_complexity, peak_count
```

## 🔧 Configuration

Edit `pipeline/config.py` to customize:

```python
SUBJECTS_TO_PROCESS = [1600, 1601]  # Subset of subjects
DEVICES_TO_PROCESS = ['phone']       # ['phone', 'watch']
SENSORS_TO_PROCESS = ['accel']       # ['accel', 'gyro']
```

## 📈 Pipeline Stages

| #   | Stage    | Input          | Output                | Time |
| --- | -------- | -------------- | --------------------- | ---- |
| 1   | Load     | 204 raw files  | 15.6M samples         | 12s  |
| 2   | Clean    | 15.6M samples  | 15.6M samples (fixed) | 34s  |
| 3   | Window   | 15.6M samples  | 516.9K windows        | 116s |
| 4   | Features | 516.9K windows | 516.9K × 60 features  | 913s |

## 💻 Load & Use in Python

```python
import pandas as pd
import numpy as np

# Load the feature matrix
df = pd.read_csv('pipeline/output/full_features.csv')

# Extract features and labels
X = df.drop(['subject_id', 'device', 'sensor', 'activity_code'], axis=1)
y = df['activity_code']

# Features only for one channel (e.g., accelerometer X)
X_accel_x = df[[col for col in df.columns if col.endswith('_x')]]

# Filter by subject/device/sensor
phone_data = df[df['device'] == 'phone']
subject_1600 = df[df['subject_id'] == 1600]
accel_data = df[df['sensor'] == 'accel']
```

## 📊 Data Statistics

- **Total windows:** 516,867
- **Windows discarded (class inconsistency):** 4,044
- **Raw samples processed:** 15,630,426
- **Compression ratio:** 30.24x
- **Data retention:** 100% (all rows preserved through cleaning)

## 🎯 20 Time-Domain Features (Per Channel)

**Statistical (8):**

- mean, median, std, var, min, max, range, iqr

**Signal Processing (7):**

- rms, zcr, sma, energy, autocorr_lag1, peak_count, mad

**Advanced (5):**

- skewness, kurtosis, hjorth_activity, hjorth_mobility, hjorth_complexity

## ⚙️ Technical Details

**Windowing:**

- Window size: 60 samples (3 seconds @ 20 Hz)
- Stride: 30 samples (50% overlap)
- Class consistency threshold: 80%

**Raw Data Format:**

```
subject_id,activity_code,timestamp,x,y,z;
```

(Note: Trailing semicolon automatically handled)

**Sampling Rate:** 20 Hz (all sensors)

## 🔍 Activity Codes (18 Total)

```
A=Walking     B=Jogging     C=Stairs (up)    D=Stairs (down)
E=Sitting     F=Standing    G=Typing         H=Writing
I=Scrolling   J=Eating      K=Drinking       L=Brushing teeth
M=Using phone O=Running     P=Exercising     Q=Yoga
R=Studying    S=Walking irregular
```

## 📝 Files Generated

```
pipeline/
├── output/
│   └── full_features.csv        ← USE THIS FOR ML
├── data/
│   ├── cleaned.csv              (intermediate)
│   └── windowed.csv             (intermediate)
└── *.py modules
```

## 🛠️ Common Tasks

**Run on subset of data:**

```python
import config
config.SUBJECTS_TO_PROCESS = [1600]
config.DEVICES_TO_PROCESS = ['phone']
config.SENSORS_TO_PROCESS = ['accel']

from main import run_pipeline
run_pipeline()
```

**Check feature statistics:**

```python
df = pd.read_csv('pipeline/output/full_features.csv')
print(df.describe())  # Statistics for all numeric columns
```

**Activity distribution:**

```python
print(df['activity_code'].value_counts().sort_index())
```

**Features per device:**

```python
phone = df[df['device'] == 'phone']
watch = df[df['device'] == 'watch']
print(f"Phone windows: {len(phone)}, Watch windows: {len(watch)}")
```

## ⚠️ Notes

- No feature scaling (do this before ML)
- No feature selection performed
- No train/test split in output
- All 18 activities included (activity N not found in dataset)
- Warnings about precision loss in skewness/kurtosis are normal for constant
  signals

## 📚 More Info

See `PIPELINE_SUMMARY.md` for detailed documentation.
