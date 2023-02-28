from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.io import output_file
from bokeh.models import ColumnDataSource, HoverTool, WheelZoomTool
from bokeh.plotting import figure, show, curdoc
from bokeh.embed import file_html
from bokeh.resources import CDN


def plot_confusion_matrix(y_true, y_pred, model_name, figsize=(4, 4)):
    """Plots the confusion matrix as a heatmap"""
    confusion_matrix = pd.crosstab(
        y_true, y_pred, rownames=["Observations"], colnames=["Predictions"]
    )
    severity_categories = ("Unharmed", "Injured", "Killed")
    plt.figure(figsize=figsize)
    plt.title("Confusion Matrix of the " + model_name)
    sns.heatmap(confusion_matrix / len(y_true), cmap="viridis", annot=True, fmt=".2%")
    plt.xticks(np.array(range(3)) + 0.5, labels=severity_categories, rotation=45)
    plt.yticks(np.array(range(3)) + 0.5, labels=severity_categories, rotation=0)


def plot_geodata(
    df, plot_date, output_path, n_plot_max=10_000, figsize=None, return_html=False
):
    """Plot gps data on map"""
    output_file(output_path)

    tooltips = [
        ("index", "@accident_id"),
        ("(lat, lon)", "(@lat, @lon)"),
        ("severity", "@severity_label"),
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
    plot_cols = ["accident_id", "longitude", "latitude", "severity"]
    plot_df = df[df["date"].apply(datetime.date) == plot_date][plot_cols]

    if len(plot_df) > n_plot_max:
        plot_df = plot_df.sample(n=n_plot_max)

    colors = plot_df["severity"].replace({1: "blue", 2: "orangered", 3: "red"})
    severity_labels = plot_df["severity"].replace(
        {0: "Unharmed", 1: "Injured", 2: "Killed"}
    )
    markers = plot_df["severity"].replace({1: "circle", 2: "square", 3: "triangle"})

    source = ColumnDataSource(
        data={
            "accident_id": plot_df["accident_id"],
            "lat": plot_df["latitude"],
            "lon": plot_df["longitude"],
            "severity": plot_df["severity"],
            "color": colors,
            "severity_label": severity_labels,
            "marker": markers,
        }
    )

    # Add accident locations
    circles = fig.circle(
        x="lat",
        y="lon",
        size=15,
        fill_alpha=0.8,
        fill_color="color",
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
):
    """Plot an overview for the specified variable in the dataframe."""
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
