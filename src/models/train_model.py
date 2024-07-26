import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from src.models.LearningAlgorithms import ClassificationAlgorithms


# Read in the processed data file
df = pd.read_pickle("data/interim/03_data_preprocessed.pkl")


# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------

df_train = df.drop(["participant", "category", "set"], axis=1)

# "Label" is the target variable
X = df_train.drop("label", axis=1)
y = df_train["label"]

# Split the data into training and test sets using a 75/25 ratio.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

# Group the feature columns
orig_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
scalar_features = ["acc_r", "gyr_r"]
pca_features = ["pca_1", "pca_2", "pca_3"]
time_features = [f for f in df_train.columns if "_temp_" in f]
frequency_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]
cluster_features = ["cluster"]

# Group features into sets
feature_set_1 = list(set(orig_features))
feature_set_2 = list(set(feature_set_1 + scalar_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + frequency_features + cluster_features))


# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------

learner = ClassificationAlgorithms()

max_features = 10

# This takes a long time to run
# selected_features, ordered_features, ordered_scores = learner.forward_selection(
#     max_features, X_train, y_train
# )

# Features selected from last run
selected_features = [
    "acc_y_freq_0.0_Hz_ws_14",
    "gyr_r_freq_0.0_Hz_ws_14",
    "set_duration",
    "acc_z_freq_0.0_Hz_ws_14",
    "cluster",
    "gyr_z_freq_2.5_Hz_ws_14",
    "gyr_x_freq_1.429_Hz_ws_14",
    "gyr_z",
    "gyr_z_freq_1.786_Hz_ws_14",
    "acc_z_freq_0.357_Hz_ws_14",
]

# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------

possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features,
]

feature_names = [
    "Feature Set 1",
    "Feature Set 2",
    "Feature Set 3",
    "Feature Set 4",
    "Selected Features",
]


# Function that runs the grid search
def run_grid_search(X_train, X_test, possible_feature_sets, feature_names, iterations):
    score_df = pd.DataFrame()
    for i, f in zip(range(len(possible_feature_sets)), feature_names):

        print("Feature set:", str(i + 1))
        selected_train_X = X_train[possible_feature_sets[i]]
        selected_test_X = X_test[possible_feature_sets[i]]

        # First run non deterministic classifiers to average their score.
        performance_test_nn = 0
        performance_test_rf = 0

        for it in range(0, iterations):
            print("\tTraining neural network,", it)
            (
                class_train_y,
                class_test_y,
                class_train_prob_y,
                class_test_prob_y,
            ) = learner.feedforward_neural_network(
                selected_train_X,
                y_train,
                selected_test_X,
                gridsearch=False,
            )
            performance_test_nn += accuracy_score(y_test, class_test_y)

            print("\tTraining random forest,", it)
            (
                class_train_y,
                class_test_y,
                class_train_prob_y,
                class_test_prob_y,
            ) = learner.random_forest(
                selected_train_X, y_train, selected_test_X, gridsearch=True
            )
            performance_test_rf += accuracy_score(y_test, class_test_y)

        performance_test_nn = performance_test_nn / iterations
        performance_test_rf = performance_test_rf / iterations

        # And we run our deterministic classifiers:
        print("\tTraining KNN")
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.k_nearest_neighbor(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_knn = accuracy_score(y_test, class_test_y)

        print("\tTraining decision tree")
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.decision_tree(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_dt = accuracy_score(y_test, class_test_y)

        print("\tTraining naive bayes")
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

        performance_test_nb = accuracy_score(y_test, class_test_y)

        print("\tTraining XGBoost")
        # First encode the labels in the train_y table
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.xgboost_classifier(
            selected_train_X, y_encoded, selected_test_X
        )  # add gridsearch later

        class_test_y = le.inverse_transform(class_test_y)
        performance_test_xgb = accuracy_score(y_test, class_test_y)

        # Save results to dataframe
        print("\tSaving results to dataframe")
        models = ["NN", "RF", "KNN", "DT", "NB", "XGB"]
        new_scores = pd.DataFrame(
            {
                "model": models,
                "feature_set": f,
                "accuracy": [
                    performance_test_nn,
                    performance_test_rf,
                    performance_test_knn,
                    performance_test_dt,
                    performance_test_nb,
                    performance_test_xgb,
                ],
            }
        )
        score_df = pd.concat([score_df, new_scores])

    return score_df


# Calls the grid search function - comment out if using notebook
iterations = 1
score_df = run_grid_search(
    X_train, X_test, possible_feature_sets, feature_names, iterations
)


# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.decision_tree(
    X_train[selected_features], y_train, X_test[selected_features], gridsearch=True
)

# Get accuracy score
accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)


# CM plot shows that model is nearly perfect except it misclassified
# an overhead press as a bench press in one instance

# Accuracy score is too good - possible target leakage???

# --------------------------------------------------------------
# Select train and test data based on participant
# --------------------------------------------------------------

participant_df = df.drop(["set", "category"], axis=1)

X_train = participant_df[participant_df["participant"] != "A"].drop("label", axis=1)
y_train = participant_df[participant_df["participant"] != "A"]["label"]

X_test = participant_df[participant_df["participant"] == "A"].drop("label", axis=1)
y_test = participant_df[participant_df["participant"] == "A"]["label"]

X_train = X_train.drop(["participant"], axis=1)
X_test = X_test.drop(["participant"], axis=1)

# # Call the plot_stratification_on_labels() function
# fig, ax = plt.subplots(figsize=(10, 5))
# df_train["label"].value_counts().plot(
#     kind="bar", ax=ax, color="lightblue", label="Total"
# )
# y_train.value_counts().plot(kind="bar", ax=ax, color="dodgerblue", label="Train")
# y_test.value_counts().plot(kind="bar", ax=ax, color="royalblue", label="Test")
# plt.legend()
# plt.show()


# --------------------------------------------------------------
# Use best model again and evaluate results
# --------------------------------------------------------------

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.decision_tree(
    X_train[selected_features], y_train, X_test[selected_features], gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)
# call the plot_confusion_matrix() function


# Get accuracy score (this time it is lower at 96.6%,
# so it performs worse particularly misclassifying deadlifts as rows

### ------------------------------------------------------------------- ###
### CONCLUSION: Decision tree model performs poorly with rows/deadlifts ###
### NEXT ACTION: Try random forest or neural network model and compare  ###
### ------------------------------------------------------------------- ###

# --------------------------------------------------------------
# Try another model with the selected features
# --------------------------------------------------------------

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)
# call the plot_confusion_matrix() function


### ------------------------------------------------------------------- ###
### CONCLUSION: Random forest model with feature set 4 performs better
### but still some error with predicting overhead press vs. bench press
### labels. Next step is to try a neural network model and compare results
### ------------------------------------------------------------------- ###

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.feedforward_neural_network(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)
# call the plot_confusion_matrix() function


### ------------------------------------------------------------------- ###
### CONCLUSION: NN  model with feature set 4 performs slightly better than
### random forest with less error with predicting OHP vs. bench
### Next, try NN with selected features to see if it performs even better.
### ------------------------------------------------------------------- ###

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.feedforward_neural_network(
    X_train[selected_features], y_train, X_test[selected_features], gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)
# call the plot_confusion_matrix() function


# ConvergenceWarning: Stochastic Optimizer: Maximum iterations (1000) reached and the optimization hasn't converged yet.
### ------------------------------------------------------------------- ###
### CONCLUSION: NN  model with selected features performed about the same as
### with the feature set 4. It still has some error with OHP vs. bench pred
### Next, try XGBoost with selected features to see if it performs even better.
### ------------------------------------------------------------------- ###

le = LabelEncoder()
y_encoded = le.fit_transform(y_train)

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.xgboost_classifier(
    X_train[selected_features], y_encoded, X_test[selected_features]
)

class_test_y = le.inverse_transform(class_test_y)
accuracy = accuracy_score(y_test, class_test_y)

classes = le.inverse_transform(class_test_prob_y.columns)
cm = confusion_matrix(y_test, class_test_y, labels=classes)
# call the plot_confusion_matrix() function
