import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display
import seaborn as sns
import itertools

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
    plt.figure(figsize=(8, 8))
    plt.plot(range(1, len(predictor_columns) + 1), pc_values)
    plt.xlabel("Principal Component Number")
    plt.ylabel("Explained Variance")
    plt.show()


# --------------------------------------------------------------
# Create an elbow plot of k values with inertias
# --------------------------------------------------------------


def elbow_plot(k_values, components):
    plt.figure(figsize=(7, 7))
    plt.plot(k_values, components)
    plt.xlabel("K")
    plt.ylabel("Sum of squared distances")
    plt.show()


# --------------------------------------------------------------
# Plot outliers in case of a binary outlier score
# --------------------------------------------------------------


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


# --------------------------------------------------------------
# Plot lowpass filter after testing it
# --------------------------------------------------------------


def test_lowpass_filter_plot(dataset, col_name):
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
    ax[0].plot(dataset[col_name].reset_index(drop=True), label="raw data")
    ax[1].plot(
        dataset[col_name + "_lowpass"].reset_index(drop=True),
        label="butterworth filter",
    )
    ax[0].legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True
    )
    ax[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True
    )
    plt.show()


# --------------------------------------------------------------
# Create a 3D cluster plot
# --------------------------------------------------------------


def cluster_plot_3d(df_cluster, cluster_col):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    for c in df_cluster[cluster_col].unique():
        subset = df_cluster[df_cluster[cluster_col] == c]
        ax.scatter(
            subset["acc_x"],
            subset["acc_y"],
            subset["acc_z"],
            label=f"{c}",
            alpha=0.5,
        )
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Create a stacked bar plot to check stratification between training and test sets
# --------------------------------------------------------------


def plot_stratification_on_labels(df_train, y_train, y_test):
    fig, ax = plt.subplots(figsize=(10, 5))
    df_train["label"].value_counts().plot(
        kind="bar", ax=ax, color="lightblue", label="Total"
    )
    y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
    y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Plot accuracy curve on number of selected features
# --------------------------------------------------------------


def plot_accuracy_curve(ordered_scores, max_features):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, max_features + 1, 1), ordered_scores)
    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(1, max_features + 1, 1))
    plt.show()


# --------------------------------------------------------------
# Create a grouped bar plot comparing model results on accuracy
# --------------------------------------------------------------


def plot_model_accuracy_comparison(score_df):
    plt.figure(figsize=(10, 10))
    sns.barplot(x="model", y="accuracy", hue="feature_set", data=score_df)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0.7, 1)
    plt.legend(loc="lower right")
    plt.show()


# --------------------------------------------------------------
# Create a confusion matrixk for classification model results
# --------------------------------------------------------------


def plot_confusion_matrix(
    cm,
    classes,
):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.grid(False)
    plt.show()


# --------------------------------------------------------------
# Create a plot that shows the peaks in a time series
# --------------------------------------------------------------


def plot_peaks(dataset, column, peaks):
    fig, ax = plt.subplots()
    plt.plot(dataset[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"], "o", color="red")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{column}_lowpass")
    exercise = dataset["label"].iloc[0].title()
    category = dataset["category"].iloc[0].title()
    plt.title(f"{exercise} - {category}: {len(peaks)} Reps")
    plt.show()
