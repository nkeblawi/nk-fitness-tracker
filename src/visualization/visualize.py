import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

"""
    EXPLORATORY DATA ANALYSIS & PLOTTING
    
    Includes a variety of visualization techniques to explore the data.
    The data is loaded from the processed data file and visualized in different ways.
    The plots are saved in the ../../reports/figures/ folder.
    The file naming convention is: '{label}-{participant}.png'

"""

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")


# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

# Select a single subset
set_df = df[df["set"] == 1]

# Plot y-values over the duration of this set
plt.plot(set_df["acc_y"])

# Plot y-values over the number of samples in this set
plt.plot(set_df["acc_y"].reset_index(drop=True))


# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

# Plot all data for each exercise
for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

# Plot only the first 100 samples for each exercise
for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Adjust plot style and settings
# --------------------------------------------------------------

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100


# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

# Stacked queries to pull squat data for participant A
# Add reset_index to make sure the index is continuous (No. of samples)
category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()

# Group by category (heavy vs. medium) and plot the y-values
fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel = "acc_y"
ax.set_xlabel = "samples"
plt.legend()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

# Pull bench data for all participants (all categories)
participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()

# Group by participant and plot the y-values for benching
fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel = "acc_y"
ax.set_xlabel = "samples"
plt.legend()


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = "squat"
participant = "A"
all_axis_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)

fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel = "acc_y"
ax.set_xlabel = "samples"
plt.legend()


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

# Loop through all combinations of labels and participants
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        if len(all_axis_df) > 0:
            # Plot if there is any Accelerometer data, otherwise skip to next iteration
            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel = "acc_y"
            ax.set_xlabel = "samples"
            plt.title(f"{label} - {participant}".title())
            plt.legend()

# Loop through all combinations of labels and participants (can be refactored)
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        if len(all_axis_df) > 0:
            # Plot if there is any Gyrometer data, otherwise skip to next iteration
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_ylabel = "gyr_y"
            ax.set_xlabel = "samples"
            plt.title(f"{label} - {participant}".title())
            plt.legend()


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = "row"
participant = "A"
combined_plot_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index(drop=True)
)

# ax[0] for Accelerometer data, ax[1] for Gyrometer data with nrows = 2
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

ax[0].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True
)
ax[1].set_xlabel("samples")


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        combined_plot_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )

        if len(combined_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

            ax[0].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].set_xlabel("samples")
            plt.savefig(f"../../reports/figures/{label.title()}-({participant}).png")
            plt.show()
