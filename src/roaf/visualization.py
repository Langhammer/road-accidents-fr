from datetime import datetime
from bokeh.io import output_file
from bokeh.models import ColumnDataSource, HoverTool, WheelZoomTool
from bokeh.plotting import figure, show, curdoc
from bokeh.embed import file_html
from bokeh.resources import CDN


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
        {1: "Unharmed", 2: "Injured", 3: "Killed"}
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
