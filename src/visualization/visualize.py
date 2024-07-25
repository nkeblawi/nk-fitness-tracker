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
# Plot all exercises
# --------------------------------------------------------------


# Plot all data for each exercise
def plot_exercise_data(df, col):
    for label in df["label"].unique():
        subset = df[df["label"] == label]
        fig, ax = plt.subplots()
        plt.plot(subset[col].reset_index(drop=True), label=label)
        plt.legend()
        plt.show()


# Plot only the first 100 samples for each exercise
def plot_exercise_sample(df, col):
    for label in df["label"].unique():
        subset = df[df["label"] == label]
        fig, ax = plt.subplots()
        plt.plot(subset[:100][col].reset_index(drop=True), label=label)
        plt.legend()
        plt.show()


# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------


# Group by category (heavy vs. medium) and plot the y-values
def plot_category_data(df, col):
    fig, ax = plt.subplots()
    df.groupby(["category"])["acc_y"].plot()
    ax.set_ylabel = "acc_y"
    ax.set_xlabel = "samples"
    plt.legend()


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------


# Group by participant and plot the y-values for benching
def plot_participant_data(df, col):
    fig, ax = plt.subplots()
    df.groupby(["participant"])[col].plot()
    ax.set_ylabel = "acc_y"
    ax.set_xlabel = "samples"
    plt.legend()


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------


def plot_all_axis(df, col_x, col_y, col_z, label, participant):
    fig, ax = plt.subplots()
    df[[col_x, col_y, col_z]].plot(ax=ax)
    ax.set_ylabel = col_y
    ax.set_xlabel = "samples"
    plt.title(f"{label} - {participant}".title())
    plt.legend()


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------


# Loop through all combinations of labels and participants
def plot_all_combinations(df, labels, participants, col_x, col_y, col_z):
    for label in labels:
        for participant in participants:
            all_axis_df = (
                df.query(f"label == '{label}'")
                .query(f"participant == '{participant}'")
                .reset_index()
            )

            if len(all_axis_df) > 0:
                # Plot if there is any Accelerometer data, otherwise skip to next iteration
                plot_all_axis(all_axis_df, col_x, col_y, col_z, label, participant)


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


def combine_plot(df, label, participant):
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


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------


def create_combined_plots(df, labels, participants, filepath):
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
                plt.savefig(f"{filepath}/{label.title()}-({participant}).png")
                plt.close()


# --------------------------------------------------------------
# Create an elbow plot of principal components with explained variances
# --------------------------------------------------------------


def plot_pc_explained_variance(pc_values, predictor_columns):
    plt.figure(figsize=(10, 10))
    plt.bar(range(1, len(predictor_columns) + 1), pc_values)
    plt.xlabel("Principal Component Number")
    plt.ylabel("Explained Variance")
    plt.show()
