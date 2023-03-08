"""Streamlit App for Road Accidents in France"""
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.roaf import visualization

st.title("Road accidents")

df = pd.read_parquet("./data/processed/df_by_accident.parquet")
PLOT_START_SIZE = 500


def init_key(key, value=None):
    """Initialize sessions states if not already set"""
    if key not in st.session_state:
        st.session_state[key] = value
    return st.session_state

st.session_state = init_key("lethal_only", False)
st.session_state = init_key("plot_zoom", 100)
st.session_state = init_key("plot_max_number", 10_000)
st.session_state = init_key("plot_start_date", datetime(2019, 1, 1, 0, 10))
st.session_state = init_key("plot_end_date", datetime(2021, 12, 31, 23, 59))

col_0, col_1 = st.columns(2)


def plot_map():
    """Plot locations of filtered accidents on a map with bokeh"""
    plot_size = PLOT_START_SIZE * st.session_state.plot_zoom * 0.01
    components.html(
        html=visualization.plot_geodata(
            df,
            date_range=pd.date_range(start=st.session_state.plot_start_date,
                                     end=st.session_state.plot_end_date,
                                     freq='min'),
            n_plot_max=st.session_state.plot_max_number,
            figsize=int(plot_size),
            return_html=True,
            output_path="./plot.html",
            lethal_only=st.session_state.lethal_only
        ),
        width=plot_size,
        height=plot_size,
    )


with col_1:
    plot_map()

with col_0:
    st.header("Filter Accidents")
    st.date_input(
        "Choose Start Date",
        value=st.session_state.plot_start_date,
        min_value=datetime(2019, 1, 1, 0, 0),
        max_value=datetime(2021, 12, 31, 23, 59),
        key="plot_start_date",
    )
    st.date_input(
        "Choose End Date",
        value=st.session_state.plot_end_date,
        min_value=datetime(2019, 1, 1, 0, 0),
        max_value=datetime(2021, 12, 31, 23, 59),
        key="plot_end_date",
    )
    st.checkbox(label="show only lethal accidents", value=False, key="lethal_only")

    st.slider("plot size in %", min_value=50, max_value=150, value=100, key="plot_zoom")
    st.slider("max number of accidents to plot", min_value=1, max_value=len(df), key='plot_max_number')