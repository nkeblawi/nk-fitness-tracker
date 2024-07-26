import numpy as np
import pandas as pd
from src.features.DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None

# Plot settings
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("data/interim/01_data_processed.pkl")
df = df[df["label"] != "rest"]

acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
gyr_r = df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2
df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)


# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------

bench_df = df[df["label"] == "bench"]
squat_df = df[df["label"] == "squat"]
row_df = df[df["label"] == "row"]
ohp_df = df[df["label"] == "ohp"]
dead_df = df[df["label"] == "dead"]


# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

fs = 1000 / 200  # 5 per second
LowPass = LowPassFilter()


# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

# Load the data for one set of each type of exercise
bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]]
row_set = row_df[row_df["set"] == row_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]]
dead_set = dead_df[dead_df["set"] == dead_df["set"].unique()[0]]

# *** Change these variables for exploration ***
column = "acc_r"
set_type = ohp_set
cutoff = 0.5

# Test visualization to see how many reps we should expect to see
set_type[column].plot()

# Now let's see if the LowPass can identify those reps
LowPass.low_pass_filter(
    set_type, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=5
)[column + "_lowpass"].plot()


# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------


def count_reps(dataset, cutoff=0.4, order=5, col="acc_r"):
    data = LowPass.low_pass_filter(
        dataset,
        col=col,
        sampling_frequency=fs,
        cutoff_frequency=cutoff,
        order=order,
    )
    indices = argrelextrema(data[column + "_lowpass"].values, np.greater)
    peaks = data.iloc[indices]

    # fig, ax = plt.subplots()
    # plt.plot(dataset[f"{column}_lowpass"])
    # plt.plot(peaks[f"{column}_lowpass"], "o", color="red")
    # ax.set_xlabel("Time")
    # ax.set_ylabel(f"{column}_lowpass")
    # exercise = dataset["label"].iloc[0].title()
    # category = dataset["category"].iloc[0].title()
    # plt.title(f"{exercise} - {category}: {len(peaks)} Reps")
    # plt.show()

    return len(peaks)


# Use this to test and optimize parameters for each type of exercise
count_reps(set_type, cutoff, col=column)

# Optimized cutoff parameters for each type of exercise and acc_r
cutoffs = {
    "bench": 0.4,
    "squat": 0.4,
    "row": 0.65,
    "ohp": 0.5,  # this works better than 0.35, also kept order at 5
    "dead": 0.375,  # this works better than 0.4 or at 0.35
}

# For rows, use gyr_x instead of acc_r
columns = {
    "bench": "acc_r",
    "squat": "acc_r",
    "row": "gyr_x",
    "ohp": "acc_r",
    "dead": "acc_r",
}

# Use this function to count reps for all exercises based on the optimized parameters
count_reps(bench_set, cutoffs["bench"], col=columns["bench"])


# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

df["reps"] = df["category"].apply(lambda x: 5 if x == "heavy" else 10)
rep_df = df.groupby(["label", "category", "set"])["reps"].max().reset_index()

rep_df["reps_pred"] = 0
for s in df["set"].unique():
    subset = df[df["set"] == s]

    label = subset["label"].iloc[0]
    cutoff = cutoffs[label]
    column = columns[label]

    reps = count_reps(subset, cutoff, col=column)
    rep_df.loc[rep_df["set"] == s, "reps_pred"] = reps


# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

error = mean_absolute_error(rep_df["reps"], rep_df["reps_pred"]).round(2)

rep_df.groupby(["label", "category"])[["reps", "reps_pred"]].mean().plot.bar()

# Error down to 0.6 from 0.68 by optimizing cutoffs for Overhead Presses and Deadlifts
