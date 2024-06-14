import pandas as pd
from glob import glob


# --------------------------------------------------------------
# Set the path to the data files
# --------------------------------------------------------------

data_path = "../../data/raw/*.csv"
files = glob(data_path)

# --------------------------------------------------------------
# Read all files using a function and combine them into 2 sets
# --------------------------------------------------------------


def read_data_from_files(files):

    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    # Loop through all the files in the directory
    for f in files:

        # Create new features from filename - participant, label, category
        participant = f.split("-")[0].replace(data_path.rstrip("*.csv"), "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)

        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        # Organize data into two datasets - Accelerometer and Gyroscope
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])

        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    # Append each file to each corresponding dataset
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    # Remove extra time-based columns (no longer needed)
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]

    # Return the combined datasets
    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)


# --------------------------------------------------------------
# Merge the two datasets into a single DataFrame
# --------------------------------------------------------------

data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

# Include categorical features when resampling to 200ms while
# taking the mean of all numerical values except 'set'
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

data_merged[:1000].resample(rule="200ms").apply(sampling)

# Split by day
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)
data_resampled["set"] = data_resampled["set"].astype("int")

# Check before exporting
# data_resampled.info()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
