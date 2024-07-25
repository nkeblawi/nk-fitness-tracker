import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

# Load the data with outliers removed
df = pd.read_pickle("../../data/interim/02_data_outliers_removed_chauvenets.pkl")

# Assign the predictor columns (sensor data)
predictor_columns = df.columns[:6]


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# Using the interpolate() method
# --------------------------------------------------------------

for col in predictor_columns:
    df[col] = df[col].interpolate()


# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

for s in df["set"].unique():
    set_start = df[df["set"] == s].index[0]
    set_stop = df[df["set"] == s].index[-1]

    set_duration = set_stop - set_start
    df.loc[(df["set"] == s), "set_duration"] = set_duration.seconds

duration_df = df.groupby(["category"])["set_duration"].mean()
duration_df.iloc[0] / 5
duration_df.iloc[1] / 10


# --------------------------------------------------------------
# Butterworth lowpass filter
# Used to filter out high-frequency noise in each set so that we
# can train the model only on the big movements
# --------------------------------------------------------------

# First make a copy of the dataframe, so we don't break the original
df_lowpass = df.copy()

# Create an instance of the LowPassFilter class
LowPass = LowPassFilter()

fs = 1000 / 200  # sampling frequency (200ms)
cutoff = 1.3  # test this value and adjust as needed

# Apply the low pass filter to the acc_y column
df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)


# --------------------------------------------------------------
# Principal component analysis PCA to reduce dimensionality
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

# Add only the first 3 principal components to the dataframe
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)


# --------------------------------------------------------------
# Temporal abstraction using window_size
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

pred_col_list = list(predictor_columns)
pred_col_list.append("acc_r")  # run this only once
pred_col_list.append("gyr_r")  # run this only once

window_size = int(1000 / 200)  # One second

# Apply the temporal abstraction set by set
df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in pred_col_list:
        subset = NumAbs.abstract_numerical(subset, [col], window_size, "mean")
        subset = NumAbs.abstract_numerical(subset, [col], window_size, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)
df_temporal.info()  # shows rolling averages have only 8,645 non-null values


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

sampling_rate = int(1000 / 200)  # 1 second
window_size = int(2800 / 200)  # 2.8 seconds

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], window_size, sampling_rate)
df_freq.columns  # Check to see if the right number of columns have been added

# Visualize the results
subset = df_freq[df_freq["set"] == 15]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot()

df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier transofrmations to set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(
        subset, pred_col_list, window_size, sampling_rate
    )
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)
df_freq.info()


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()  # drop missing values
df_freq.iloc[::2]  # reduce size of dataset by half to prevent overfitting


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("K")
plt.ylabel("Sum of squared distances")
plt.show()  # plot shows elbow at cluster 5

# Set cluster at 5 and add new column
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Visualize the clusters in 3D
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=f"Cluster {c}")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# Redo above visualization except cluster by labels
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_preprocessed.pkl")
