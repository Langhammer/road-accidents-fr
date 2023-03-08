from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from bokeh.io import output_file
from bokeh.models import ColumnDataSource, HoverTool, WheelZoomTool
from bokeh.plotting import figure, show, curdoc
from bokeh.embed import file_html
from bokeh.resources import CDN


def plot_confusion_matrix(y_true, y_pred, model_name, normalize=None, figsize=(4, 4)):
    """Plots the confusion matrix as a heatmap"""
    matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=normalize)

    plt.figure(figsize=figsize)
    
    # Title
    title_string = "Confusion Matrix of the " + model_name + " Predictions"
    if normalize is not None:
        title_string = "Normalized " + title_string   
    plt.title(title_string)

    sns.heatmap(matrix, cmap="viridis", annot=True, fmt=".2%")

    severity_categories = ("Unharmed", "Injured", "Killed")
    plt.xticks(np.array(range(3)) + 0.5, labels=severity_categories, rotation=45)
    plt.yticks(np.array(range(3)) + 0.5, labels=severity_categories, rotation=0)
    plt.xlabel('Predictions')
    plt.ylabel('Observations')

def plot_geodata(
    df, plot_date, output_path, n_plot_max=10_000, figsize=None, return_html=False
):
    """Plot gps data on map"""
    output_file(output_path)

    tooltips = [
        ("index", "@accident_id"),
        ("(lat, lon)", "(@lat, @lon)"),
        ("Unharmed", "@unharmed"),
        ("Injured", "@injured"),
        ("Killed", "@killed"),
    ]

    fig = figure(
        x_range=(-750_000, 1_125_000),
        y_range=(5_755_000, 5_955_000),
        x_axis_type="mercator",
        y_axis_type="mercator",
        tooltips=tooltips,
    )

    if figsize is not None:
        if isinstance(figsize, int):
            fig.width = figsize
            fig.height = figsize
        elif len(figsize) == 2:
            fig.width = figsize[0]
            fig.height = figsize[1]
        else:
            fig.width = figsize = [0]
            fig.height = figsize = [0]

    fig.add_tile("STAMEN_TONER")

    # Size of sample of data points to plot.
    # More than 10_000 data points can become very slow
    plot_cols = [
        "accident_id",
        "longitude",
        "latitude",
        "severity_0",
        "severity_1",
        "severity_2",
    ]
    plot_df = df[df["date"].apply(datetime.date) == plot_date][plot_cols]

    if len(plot_df) > n_plot_max:
        plot_df = plot_df.sample(n=n_plot_max)

    severity = np.zeros(shape=len(plot_df))
    for i_accident in range(len(plot_df)):
        if plot_df["severity_2"].iloc[i_accident]:
            severity[i_accident] = 2
        else:
            severity[i_accident] = 1

    colors = pd.Series(severity).replace({2: "red", 1: "orange"})

    source = ColumnDataSource(
        data={
            "accident_id": plot_df["accident_id"],
            "lat": plot_df["latitude"],
            "lon": plot_df["longitude"],
            "unharmed": plot_df["severity_0"],
            "injured": plot_df["severity_1"],
            "killed": plot_df["severity_2"],
            "colors": list(colors),
        }
    )

    # Add accident locations
    circles = fig.circle(
        x="lat",
        y="lon",
        size=15,
        fill_alpha=0.8,
        fill_color="colors",
        line_color="grey",
        line_width=1,
        source=source,
    )

    HoverTool(tooltips=tooltips, renderers=[circles])
    fig.toolbar.active_scroll = fig.select_one(WheelZoomTool)

    # Change bokeh theme
    curdoc().theme = "dark_minimal"

    if return_html:
        curdoc().add_root(fig)
        return file_html(fig, CDN, "map")
    else:
        show(fig)


def plot_continuous_variable_overview(
    df,
    variable_name,
    smooth_bandwith=1,
    rough_bandwith=0.2,
    filter_percentile=0.05,
    figsize=(10, 5),
    sample_frac=None,
):
    """Plot an overview for the specified variable in the dataframe."""
    if sample_frac is not None:
        df = df.sample(frac=sample_frac)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"Overview for Variable\n'{variable_name}'",
        bbox={"boxstyle": "square", "alpha": 0.5},
    )
    axs = fig.subplot_mosaic("AABB;CCDE")
    fig.tight_layout(h_pad=4)

    ax = axs["A"]
    plt.sca(ax)
    sns.kdeplot(data=df, x=variable_name, legend=True, bw_adjust=smooth_bandwith, ax=ax)
    sns.kdeplot(data=df, x=variable_name, legend=True, bw_adjust=rough_bandwith, ax=ax)
    sns.rugplot(data=df, x=variable_name, ax=ax)
    plt.legend(["Smooth", "Rough"])
    ax.set_title("Distribution of " + variable_name)

    ax = axs["B"]
    percentile_lower = df[variable_name].quantile(filter_percentile)
    percentile_upper = df[variable_name].quantile(1 - filter_percentile)
    df_filtered = df[
        (df[variable_name] < percentile_upper) & (df[variable_name] > percentile_lower)
    ]
    plt.sca(ax)
    sns.kdeplot(data=df_filtered, x=variable_name, bw_adjust=smooth_bandwith, ax=ax)
    sns.kdeplot(
        data=df_filtered, x=variable_name, legend=True, bw_adjust=rough_bandwith, ax=ax
    )
    sns.rugplot(data=df_filtered, x=variable_name, ax=ax)
    plt.legend(["Smooth", "Rough"])
    ax.set_title(
        f"Robust Distribution of {variable_name}\n({filter_percentile}--{1-filter_percentile})"
    )

    ax = axs["C"]
    plt.sca(ax)
    sns.boxplot(data=df, x=variable_name, ax=ax)
    sns.rugplot(data=df, x=variable_name, ax=ax)
    ax.set_title("Boxplot of " + variable_name)

    ax = axs["D"]
    plt.sca(ax)
    plot_data = pd.DataFrame(df[variable_name].isna().value_counts() / len(df)).rename(
        index={True: "Missing", False: "Present"}
    )
    plot_data.plot(kind="pie", y=variable_name, ax=ax, legend=False, autopct="%.02f%%")
    plt.ylabel(None)
    ax.set_title(f"Missing Values:\n{df[variable_name].isna().sum()}")

    ax = axs["E"]
    plt.sca(ax)

    text = f"Name: {variable_name}\n" f"Data Type: {str(df[variable_name].dtype)}"
    ax.axis("off")
    props = dict(boxstyle="round", alpha=0.5)
    plt.text(
        x=0,
        y=1,
        s=text,
        bbox=props,
        verticalalignment="top",
        horizontalalignment="left",
    )


def savefig(basename, filepath=None, formats=None, verbose=0):
    """Save the current figure in the specified formats."""
    if formats is None:
        formats = ["png", "svg"]
    filenames = []
    for suffix in formats:
        filenames.append(
            filepath.rstrip("/") + "/" + basename + "." + suffix.lstrip(".")
        )
        plt.savefig(filenames[-1])
    if verbose:
        print(filenames)
    return filenames
