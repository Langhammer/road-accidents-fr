# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: roafr_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# <img src="../images/headers/nb_3.svg"  width="1080" height="220">

# %% [markdown]
# # Import Section

# %%
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.random import sample_without_replacement
from skopt import BayesSearchCV
from xgboost import XGBClassifier, plot_importance

from roaf import parameterization, visualization

# %%
# %matplotlib inline
plt.style.use("dark_background")
plt.set_cmap("Dark2")
sns.set_palette("Dark2")

# %% [markdown]
# # Setup of notebook parameters
# These parameters can be overwritten with papermill parameterization, namely for testing purposes.
#
# <table>
#   <tr>
#     <th> Parameter </td>
#     <th> Description </td>
#   </tr>
#   <tr>
#     <td> fast_execution </td>
#     <td> Should be set to <i>True</i> if the focus is on testing and not on prediction
#       quality </td>
#   </tr>
#   <tr>
#     <td> plot_dir </td>
#     <td> The output directory for the plots </td>
#   </tr>
#   <tr>
#     <td> max_sample_size </td>
#     <td> The number of data points used for machine learning and validation. <br>
#           If set to <i>None</i>, all the data (after undersampling) will be used</td>
#   </tr>
#   <tr>
#     <td> n_plot </td>
#     <td> The number of data points to plot in certain figures </td>
#   </tr>
#   <tr>
#     <td> n_cv </td>
#     <td> Parameter for k-fold cross-validation used in parameter optimization</td>
#   </tr>
#   <tr>
#     <td> n_permutation_repetitions </td>
#     <td> The number of permutations to be performed to find the importance of
#     features in trained models</td>
#   </tr>
#   <tr>
#     <td> n_random_forest_estimators </td>
#     <td> The number of estimators in the random forest model</td>
#   </tr>
#
# </table>

# %% tags=["parameters"]
FAST_EXECUTION = False
PLOT_DIR = "../images/"
MAX_SAMPLE_SIZE = None
N_PLOT = 15
N_CV = 5
N_PERMUTATION_REPETITIONS = 10
N_RANDOM_FOREST_ESTIMATORS = 100
VERBOSE = 0

# %%
df = pd.read_parquet("../data/processed/df_by_user.parquet")

# %% [markdown]
# # Data Preprocessing for Machine Learning
# The data was already cleaned in notebook 1, but this cleaning was meant for general purpose,
# so that more data was retained for visualizations in notebook 2.
#
# In this notebook, the data will be prepared for the application of machine learning models.
# This preparation will be done in these steps:
# 1. Dropping all remaining columns that can not be used for machine learning
# 2. One-hot encoding of categorical variables
# 3. Undersampling to get a balanced subset of data
# 4. Scaling
# 5. Splitting the data into training and testing data
# 6. Saving the preprocessed datasets to make them available for the neural networks in notebook 4.

# %% [markdown]
# ## Dropping all remaining columns that can not be used for machine learning

# %%
df_ml = (
    df.select_dtypes(include=np.number)
    .drop(columns=["accident_id", "accident_id_y"])
    .dropna(axis=1, how="any")
)

# %% [markdown]
# ## One-hot encoding of categorical variables

# %%
df_ml = pd.get_dummies(
    data=df_ml,
    columns=[
        "daylight",
        "built_up_area",
        "intersection_category",
        "weather",
        "collision_category",
        "road_admin_category",
        "traffic_regime",
        "reserved_lane",
        "plane_layout",
        "surface_condition",
        "infrastructure",
        "location",
        "is_weekend",
        "role",
    ],
)

# %% [markdown]
# ## Undersampling to get a balanced subset of data or to decrease computation time

# %%
features = df_ml.drop(columns="severity")
feature_columns = features.columns
target = df_ml["severity"]
random_under_sampler = RandomUnderSampler()
features, target = random_under_sampler.fit_resample(X=features, y=target)

# %%
sample_size = len(target)
if MAX_SAMPLE_SIZE is not None:
    if sample_size > MAX_SAMPLE_SIZE:
        sample_idx = sample_without_replacement(
            n_population=sample_size,
            n_samples=MAX_SAMPLE_SIZE,
            random_state=0,
        )
        features = features.iloc[sample_idx]
        target = target.iloc[sample_idx]

print(sample_size)

# %% [markdown]
# ## Scaling

# %%
scaler = StandardScaler()
features = scaler.fit_transform(features)

# %% [markdown]
# ## Splitting the data into training and testing data

# %%
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=0,
)

# %%
# Reconvert the features to DataFrames in order to keep the feature names
X_train = pd.DataFrame(data=X_train)
X_test = pd.DataFrame(data=X_test)
X_train.columns = feature_columns
X_test.columns = feature_columns

# %% [markdown]
# ## Saving the preprocessed datasets
# The preprocessed and splitted dataset will be exported to parquet so that it can be used in
# notebook 4 (artificial neural networks). Parquet is used again as the file format for its
# low requirements regarding disk space.

# %%
TRAIN_FILENAME = "Xy_train"
TEST_FILENAME = "Xy_test"

if FAST_EXECUTION:
    TRAIN_FILENAME = "TESTING_" + TRAIN_FILENAME
    TEST_FILENAME = "TESTING_" + TEST_FILENAME

Xy_train = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
Xy_test = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

Xy_train.to_parquet("../data/processed/" + TRAIN_FILENAME + ".parquet")
Xy_test.to_parquet("../data/processed/" + TEST_FILENAME + ".parquet")


# %% [markdown]
# # XGBoost
# ## Setup and training

# %%
xgb_clf = XGBClassifier(n_jobs=multiprocessing.cpu_count() // 2)

param_spaces = {
    "max_depth": [2, 4, 6],
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
}

bayes_search = BayesSearchCV(
    xgb_clf, param_spaces, cv=N_CV, n_jobs=2, n_iter=4, verbose=0
)
bayes_search.fit(X_train, y_train)

# %% [markdown]
# ## Evaluation and Interpretation

# %%
best_xgb = bayes_search.best_estimator_
y_pred = best_xgb.predict(X_test)

FILENAME = "rxgb_confusion_matrix"
visualization.savefig(basename=FILENAME, filepath=PLOT_DIR)
visualization.plot_confusion_matrix(y_test, y_pred, "XGBoost", figsize=(4, 4))

print(classification_report(y_true=y_test, y_pred=y_pred))

# %% [markdown]
# When interpreting precision and recall for machine learning models, it is important to keep in
# mind, that the test data is balanced while the real-world data is not. That means, that we would
# expect an accuracy of 0.33 for a random guess.
#
# The model does a good job predicting deadly injuries. Unharmed persons are detected with a high
# recall but with lower precision. It shows a lower performance for injured persons. Here,
# especially the recall is very low.

# %%
p = plot_importance(best_xgb, max_num_features=N_PLOT, height=0.8, grid="off")
p.grid(False)
FILENAME = "rxgb_feat_importance"
visualization.savefig(basename=FILENAME, filepath=PLOT_DIR)

# %% [markdown]
# The feature importance plot enables us to identify the most important features used by XGBoost
# for the classification problem.
# It seems like the location (represented by longitude and latitude) has the highest importance
# in this case.

# %% [markdown]
# # Random Forest
# ## Setup and Training

# %%
random_forest_clf = RandomForestClassifier(n_estimators=N_RANDOM_FOREST_ESTIMATORS)
random_forest_clf.fit(X_train, y_train)

# %% [markdown]
# ## Evaluation and Interpretation

# %%
y_pred_rf = random_forest_clf.predict(X_test)

FILENAME = "rf_confusion_matrix"
visualization.savefig(basename=FILENAME, filepath=PLOT_DIR)
visualization.plot_confusion_matrix(y_test, y_pred, "Random Forest", figsize=(4, 4))

print(classification_report(y_true=y_test, y_pred=y_pred_rf))

# %% [markdown]
# ## Interpretation with Means of Permutation Importance
# Random Forests can be interpreted with impurity-based feature importance, but this approach
# has some downsides.
# I will therefore use permutation feature importance to analyze the model. For this, I will
# calculate the feature importance weights for both the training and the test set and compare them.
# Those features that show a high # difference # between the calculated values for training and
# test set are considered to be causal for overfitting.

# %%
# The permutation performance takes a while to compute.
r_train = permutation_importance(
    random_forest_clf,
    X_train,
    y_train,
    n_repeats=N_PERMUTATION_REPETITIONS,
    random_state=0,
)
r_test = permutation_importance(
    random_forest_clf,
    X_test,
    y_test,
    n_repeats=N_PERMUTATION_REPETITIONS,
    random_state=0,
)

# %%
importances_mean_df = pd.DataFrame(index=feature_columns)
importances_std_df = pd.DataFrame(index=feature_columns)

# Mean Values
importances_mean_df["train"] = r_train.importances_mean
importances_mean_df["test"] = r_test.importances_mean

importances_mean_df["train_test_diff"] = abs(
    importances_mean_df["test"] - importances_mean_df["train"]
)
importances_mean_df.sort_values(by="train_test_diff", ascending=False, inplace=True)
importances_mean_df.drop(columns=["train_test_diff"], inplace=True)

# Standard Deviation
importances_std_df["train"] = r_train.importances_std
importances_std_df["test"] = r_test.importances_std
importances_std_df = importances_std_df.reindex_like(importances_mean_df)

# %%
importances_mean_df[["train", "test"]].head(N_PLOT).plot(
    kind="barh", capsize=2, xerr=importances_std_df.head(N_PLOT)
)
plt.title("Features with High Difference in Importance between Train and Test Set")
plt.xlabel("")
plt.ylabel("feature")

visualization.savefig(basename="rf_feature_importance", filepath=PLOT_DIR)

# %%
importances_mean_df.sort_values("train", ascending=False, inplace=True)
importances_std_df = importances_std_df.reindex_like(importances_mean_df)
sns.barplot(
    data=importances_mean_df.head(N_PLOT),
    x="train",
    y=importances_mean_df.head(N_PLOT).index.values,
    xerr=importances_std_df["train"].head(N_PLOT),
    capsize=1.0,
    ecolor="white",
    palette="Dark2",
)
