"""Streamlit App for Road Accidents in France"""
import os
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

from src.roaf import data, visualization

st.title("Road accidents")

os.system("notebooks/nb_1.py")

df = data.df_from_pickle("./data/processed/df.p")
PLOT_START_SIZE = 500


def init_key(key, value=None):
    """Initialize sessions states if not already set"""
    if key not in st.session_state:
        st.session_state[key] = value


init_key("plot_zoom", 100)
init_key("plot_date", datetime(2019, 1, 1, 0, 10))

col_0, col_1 = st.columns(2)


def plot_map():
    """Plot locations of filtered accidents on a map with bokeh"""
    plot_size = PLOT_START_SIZE * st.session_state.plot_zoom * 0.01
    components.html(
        html=visualization.plot_geodata(
            df,
            st.session_state.plot_date,
            n_plot_max=1_000,
            figsize=int(plot_size),
            return_html=True,
            output_path="./plot.html",
        ),
        width=plot_size,
        height=plot_size,
    )


with col_1:
    plot_map()

with col_0:
    st.date_input(
        "Choose Date",
        value=st.session_state.plot_date,
        min_value=datetime(2019, 1, 1, 0, 0),
        max_value=datetime(2021, 12, 31, 23, 59),
        key="plot_date",
    )

    st.slider("plot size in %", min_value=50, max_value=150, value=100, key="plot_zoom")
