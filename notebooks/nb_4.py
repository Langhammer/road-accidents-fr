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
# <img src="../images/headers/nb_4.svg"  width="1080" height="220">
#
# The objective of this notebook is to apply artificial neural network models to the
# tabular data used in notebook 3.
#
# First, a simple dense nn will be trained. The task will be treated as classification.
# This means, that the model will ignore that we have ordered categories.
# Second, a dense nn will be trained using an ordinal layer, an ordinal loss function and
# ordinal metrics provided by the coral_ordinal package.
#
# The models will be compared with regard to their accuracy.
#

# %% [markdown]
# # Import Packages and Data

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import coral_ordinal as coral
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from roaf import visualization

# %%
plt.style.use("dark_background")
plt.set_cmap("Dark2")
sns.set_palette("Dark2")

# %% tags=["parameters"]
FAST_EXECUTION = False
N_EPOCHS = 20

# %%
TRAIN_FILENAME = "Xy_train"
TEST_FILENAME = "Xy_test"

if FAST_EXECUTION:
    TRAIN_FILENAME = "TESTING_" + TRAIN_FILENAME
    TEST_FILENAME = "TESTING_" + TEST_FILENAME

train = pd.read_parquet("../data/processed/" + TRAIN_FILENAME + ".parquet")
test = pd.read_parquet("../data/processed/" + TEST_FILENAME + ".parquet")

# %%
X_train = train.drop(columns="severity")
y_train = train["severity"]
X_test = test.drop(columns="severity")
y_test = test["severity"]

# %% [markdown]
# # ANN Models with Keras

# %%
models_df = pd.DataFrame(  # pylint: disable=C0103
    columns=["model", "history", "i_color", "metric"]
).rename_axis(index="model_name")

# %% [markdown]
# ## Simple Dense Layer Network Classifier

# %%
MODEL_ID = "ann"
DROPOUT_RATE = 0.3
models_df.loc[MODEL_ID, "model"] = keras.Sequential(
    [
        keras.layers.Dense(
            units=32, activation="gelu", input_shape=(X_train.shape[1],)
        ),
        keras.layers.Dropout(rate=DROPOUT_RATE),
        keras.layers.Dense(units=3, activation="softmax"),
    ],
    name="Dense_ANN",
)

# get index fo
models_df.loc[MODEL_ID, "model"].compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", coral.MeanAbsoluteErrorLabels()],
)

reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
    factor=0.75, patience=6, cooldown=10, min_lr=0.0001
)
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)
callbacks = [reduce_lr_callback, early_stopping]

# %%
models_df.loc[MODEL_ID, "history"] = models_df.loc[MODEL_ID, "model"].fit(
    X_train, y_train, epochs=N_EPOCHS, validation_split=0.1, callbacks=callbacks
)

# %%
models_df.loc[MODEL_ID, "model"]
test_pred = models_df.loc[MODEL_ID, "model"].predict(X_test)
test_pred_class = np.argmax(test_pred, axis=1)

print(classification_report(y_pred=test_pred_class, y_true=y_test))

# %%
visualization.plot_confusion_matrix(
    y_true=y_test,
    y_pred=test_pred_class,
    model_name=models_df.loc[MODEL_ID, "model"].name.replace("_", " "),
)

# %% [markdown]
# ## Ordinal Regression with Coral

# %%
MODEL_ID = "coral_ann"

NUM_CLASSES = 3
DROPOUT_RATE = 0.4

models_df.loc[MODEL_ID, "model"] = keras.Sequential(
    [
        keras.layers.Dense(128, activation="gelu"),
        keras.layers.Dropout(rate=DROPOUT_RATE),
        keras.layers.Dense(32, activation="gelu"),
        keras.layers.Dropout(rate=DROPOUT_RATE),
        coral.CoralOrdinal(num_classes=NUM_CLASSES),
    ],
    name="Coral_Ordinal_ANN",
)

models_df.loc[MODEL_ID, "model"].compile(
    loss=coral.OrdinalCrossEntropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    metrics=["accuracy", coral.MeanAbsoluteErrorLabels()],
)

# %%
models_df.loc[MODEL_ID, "history"] = models_df.loc[MODEL_ID, "model"].fit(
    X_train, y_train, epochs=N_EPOCHS, validation_split=0.1, callbacks=callbacks
)

# %%
test_pred = coral.ordinal_softmax(models_df.loc["coral_ann", "model"].predict(X_test))
test_pred_class = np.argmax(test_pred, axis=1)

print(classification_report(y_pred=test_pred_class, y_true=y_test))


# %%
def plot_training_history(
    model, history, metric="loss", training_options=None, validation_options=None
):
    """Plots the selected metric over the training history."""
    plt.plot(history.history[metric], label="training", **training_options)
    plt.plot(history.history["val_" + metric], label="validation", **validation_options)

    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize() + " Value")
    plt.legend(
        title=metric.capitalize(),
        frameon=True,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
    )
    plt.title(model.name.replace("_", " ") + " Training History")


# %%
models_df.loc["ann", "metric"] = "accuracy"
models_df.loc["coral_ann", "metric"] = "mean_absolute_error_labels"

# %%
models_df.apply(func=lambda x: x.name, axis=1)

# %%
colors_ids = list(range(len(models_df)))
models_df.iloc[colors_ids]["i_color"] = colors_ids

# %%
cmap = matplotlib.colormaps["Dark2"]


def plot_history_from_df_row(row):
    """Plot the history with the parameters from a row in a DataFrame"""
    plt.figure()
    plot_training_history(
        model=row["model"],
        history=row["history"],
        metric=row["metric"],
        training_options={"linestyle": "--", "color": cmap(row["i_color"])},
        validation_options={"color": cmap(row["i_color"])},
    )


models_df.apply(plot_history_from_df_row, axis=1)

# %% [markdown]
# # Conclusion
# The applied ann do not provide better predictions than the conventional models from notebook 2.
# It is noticeable that the training hardly provides any improvement over the different epochs,
# as seen in the validation loss.
