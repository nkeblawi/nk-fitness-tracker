# Fitness Activity Classifier — Wearable Sensor Data

Supervised ML pipeline that classifies **strength training exercises** and **counts repetitions** from raw accelerometer and gyroscope data recorded by a MetaMotion wrist sensor. Six algorithms are benchmarked across four progressively richer feature sets.

Based on: Hoogendoorn & Funk, *[Machine Learning for the Quantified Self](https://link.springer.com/book/10.1007/978-3-319-66308-1)* (Springer, 2017). Original source: [mhoogen/ML4QS](https://github.com/mhoogen/ML4QS/).

---

## What It Does

### Exercise Classification

Given a window of sensor readings, the model predicts which of five exercises is being performed:

| Label | Exercise |
|---|---|
| `bench` | Barbell bench press |
| `squat` | Barbell back squat |
| `row` | Barbell row |
| `ohp` | Overhead press |
| `dead` | Deadlift |

A `rest` label is also present in the raw data and is excluded from classification.

### Repetition Counting

A separate module (`count_repetitions.py`) applies a **Butterworth low-pass filter** to the scalar magnitude signal and detects local maxima as rep peaks. Cutoff frequencies are tuned per exercise:

| Exercise | Signal | Cutoff (Hz) |
|---|---|---|
| Bench press | `acc_r` | 0.40 |
| Squat | `acc_r` | 0.40 |
| Row | `gyr_x` | 0.65 |
| Overhead press | `acc_r` | 0.50 |
| Deadlift | `acc_r` | 0.375 |

Performance is evaluated with **mean absolute error** between predicted and actual rep counts.

---

## Data

**Source:** MetaMotion wrist sensor, recorded by 5 participants performing heavy (5 reps) and medium (10 reps) sets of each exercise. Download the dataset and unzip into `data/raw/`:

- [MetaMotion.zip](https://secure-res.craft.do/v2/VDcx9pyWxusPMveFX3m6KG6HXbjF2gSLkdV3zTrPX8WWrkjoh6aJinsjsSg9tEdgeMZcjDWdtZd28EhN2o2xY1Ui9TfDF5BLtGfUvYhVMqbVgdBdG7UWggpP3rR3DnS5CP9iupmM9rQQPpc9EREkeFXTSsmWXLbb98D3kdakxcembuRAC65ewTeSez8H1yd1GqFYoL76ZhHHGYrL1a4QgNa3G1pHhMLMViLV1PjeuDVxboZBTgp4S8SUsyZZDTixk5jNFwM8BZxff3Mwd8JtxQYkKkGsj8mVm75oGZaFbSGXAkLTsP/MetaMotion.zip)

**Raw signals:** `acc_x/y/z` (12.5 Hz), `gyr_x/y/z` (25 Hz) — merged and resampled to a **200 ms** epoch using the mean.

**Filename encoding:** each CSV filename encodes `participant-label-category`, which `make_dataset.py` parses to create metadata columns automatically.

Reference report: [Mini Master Project — Context Aware Applications for Strength Training](https://secure-res.craft.do/v2/DkCrM8qa8MpqYUv1hZTca1NmEQN8BUD3jgq4E4hUHHYsSECHyPEAMTuaPRwgmvY9KMGbjTiSXxGeD7e4SJpRu6vjQCpDRVbKBT3ywX4ZgDEdyoWBQqxvdJYxVxyQcMqvptguFPNpAqP4UWV7Ub9hpX9iyYUdXqXhQy4foenh4nasYefmgkpSP3MFzrPaz2Ma6jwhTCgzJSMEvfdNeAywK2Mz1JNqaAk8jUwyVp8zpBNxcQzDiwmvvnWdapkVkZmwRTkNbF3iKM5qbMWgnpQa2fhcEzXebG7qq3tC6etT9mErJRZSBrhEXkvDCRhLnsMD9vPzLALSyuBuX9DR6vfKUUs7qEPXArtHkU52wtg2oWfJShZeHcigvgQhbfgXY1o8QAV8W35YeqQYeVZ8SHwZt9TsfkhUEHReUVBYH7hKKdYEjtsJnkkCZ4ncoC9PSdQsSr8BTb9MbvyZTQfEgvBP2HqmtcM45ZLkj/Mini%20Master%20Project%20-%20Exploring%20the%20Possibilities%20of%20Context%20Aware%20Applications%20for%20Strength%20Training.pdf)

---

## Pipeline — Step by Step

The pipeline is developed across four notebooks and a set of `src/` scripts:

### 1. Data ingestion (`src/data/make_dataset.py`)
- Reads all CSVs from `data/raw/`, splits into accelerometer and gyroscope DataFrames
- Parses participant, exercise label, and set category from filenames
- Sets a millisecond-epoch datetime index; resamples to 200 ms
- Exports: `data/interim/01_data_processed.pkl`

### 2. Outlier removal (`src/features/remove_outliers.py`)
Three methods implemented; **Chauvenet's criterion** was selected for the main pipeline:

| Method | Approach |
|---|---|
| IQR | Flags values beyond Q1 − 1.5×IQR or Q3 + 1.5×IQR |
| Chauvenet's criterion | Flags points with normal-distribution probability < 1/(C×N) |
| Local Outlier Factor (LOF) | Density-based; flags points with anomalously low local density (20 neighbors) |

Detected outliers are replaced with NaN and interpolated downstream.
Exports: `data/interim/02_data_outliers_removed_chauvenets.pkl`

### 3. Feature engineering (`src/features/build_features.py`)

| Feature group | Description |
|---|---|
| Scalar magnitude | `acc_r = √(acc_x² + acc_y² + acc_z²)`, same for gyroscope |
| PCA components | 3 principal components from the 6 raw sensor axes |
| Low-pass filtered | Butterworth filter on `acc_y` (cutoff 1.3 Hz, order 5) |
| Temporal abstractions | Rolling **mean** and **std** per set, window = 5 samples (1 s) |
| Frequency abstractions | Fourier transform per set, window = 14 samples (2.8 s) — produces amplitude and power spectral entropy features per axis |
| Set duration | Seconds from first to last reading in a set |
| K-Means cluster label | 5 clusters fit on `acc_x/y/z` — used as a categorical feature |

Exports: `data/interim/03_data_preprocessed.pkl`

### 4. Model training (`src/models/train_model.py`)

**Feature sets benchmarked:**

| Set | Contents |
|---|---|
| Feature Set 1 | Raw 6-axis sensor readings |
| Feature Set 2 | Set 1 + scalar magnitudes + PCA components |
| Feature Set 3 | Set 2 + temporal abstractions |
| Feature Set 4 | Set 3 + frequency features + cluster label |
| Selected features | 10 features chosen by forward selection (decision tree) |

**Forward selection result** (top 10 features):
`acc_y_freq_0.0_Hz_ws_14`, `gyr_r_freq_0.0_Hz_ws_14`, `set_duration`, `acc_z_freq_0.0_Hz_ws_14`, `cluster`, `gyr_z_freq_2.5_Hz_ws_14`, `gyr_x_freq_1.429_Hz_ws_14`, `gyr_z`, `gyr_z_freq_1.786_Hz_ws_14`, `acc_z_freq_0.357_Hz_ws_14`

**Algorithms evaluated** (`src/models/LearningAlgorithms.py`):

| Algorithm | Notes |
|---|---|
| Decision Tree | Grid search; deterministic |
| Random Forest | Grid search; averaged over N iterations |
| K-Nearest Neighbor | Grid search; deterministic |
| Naive Bayes | No grid search |
| Feedforward Neural Network | Grid search; averaged over N iterations |
| XGBoost | Label-encoded target; no grid search yet |

**Evaluation:** 75/25 train/test split (stratified by label) and a separate participant-held-out split (train on participants B–E, test on participant A).

---

## Project Structure

```
nk-fitness-tracker/
├── environment.yml                         # Conda environment (Python 3.8)
├── notebooks/
│   ├── nk-fitness-tracker-p1-exploratory.ipynb       # EDA and signal visualization
│   ├── nk-fitness-tracker-p2-outlier-removal.ipynb   # Outlier method comparison
│   ├── nk-fitness-tracker-p3-feature-engineering.ipynb
│   └── nk-fitness-tracker-p4-predictive-modeling.ipynb
├── src/
│   ├── data/
│   │   └── make_dataset.py                # Ingest raw CSVs → 01_data_processed.pkl
│   ├── features/
│   │   ├── remove_outliers.py             # IQR, Chauvenet, LOF
│   │   ├── build_features.py              # Full feature engineering pipeline
│   │   ├── count_repetitions.py           # Low-pass peak detection for rep counting
│   │   ├── DataTransformation.py          # LowPassFilter, PCA wrappers
│   │   ├── TemporalAbstraction.py         # Rolling mean/std abstraction
│   │   └── FrequencyAbstraction.py        # Fourier transformation abstraction
│   ├── models/
│   │   ├── LearningAlgorithms.py          # Wrappers for all 6 classifiers
│   │   └── train_model.py                 # Feature set benchmark + final model eval
│   └── visualization/
│       └── visualize.py                   # Plot helpers (signals, peaks, confusion matrix)
├── data/
│   ├── raw/                               # MetaMotion CSVs (not tracked in git)
│   ├── interim/                           # Pickled intermediate DataFrames
│   └── processed/                         # Final model-ready dataset
├── models/                                # Serialized trained models (if exported)
└── references/                            # Data dictionaries and reference materials
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate nk-fitness-tracker
```

**Key dependencies:** Python 3.8, scikit-learn 1.3.2, XGBoost 2.1.0, pandas 2.0.3, numpy 1.24.4, scipy 1.10.1, matplotlib 3.6.2, seaborn 0.13.2.

---

## Running the Pipeline

Run scripts in order, or work through the four notebooks interactively:

```bash
# 1. Ingest raw data
python src/data/make_dataset.py

# 2. Remove outliers (produces 02_data_outliers_removed_chauvenets.pkl)
python src/features/remove_outliers.py

# 3. Build features (produces 03_data_preprocessed.pkl)
python src/features/build_features.py

# 4. Train and evaluate models
python src/models/train_model.py

# Optional: count repetitions (standalone, uses 01_data_processed.pkl)
python src/features/count_repetitions.py
```
