import streamlit as st
import xarray as xr
import numpy as np
from pathlib import Path
import acr
import acr.units as au
import os
import pandas as pd
import matplotlib.pyplot as plt
import kdephys.pd.df_methods
import matplotlib
import kdephys.plot as kp

# -------------------------------------- Get Basic Info Needed --------------------------------------
plt.style.use("acr_plots.mplstyle")

subject = root = Path("/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/")
subjects = []
for f in os.listdir(root):
    full_path = os.path.join(root, f)
    if np.logical_and("ACR_" in f, os.path.isdir(full_path)):
        subjects.append(f)

subject = st.sidebar.selectbox("Subject", subjects)
exp = st.sidebar.text_input("Experiment", "exp_name")
st.markdown(f"# Unit Exploration - {subject}")
st.markdown(f"Note: Currently designed to look at one experiment at a time")

unit_df_dir = f"/Volumes/opto_loc/Data/{subject}/sorting_data/spike_dataframes/"
available_sortings = []
for f in os.listdir(unit_df_dir):
    if exp in f:
        available_sortings.append(f.split(".")[0])

sortings_to_load = st.sidebar.multiselect("Choose your sortings", available_sortings)
st.write("Sortings to be loaded:", sortings_to_load, "from experiment", exp)

# ------------------------------------ Functions ------------------------------------------
@st.cache()
def load_spike_dfs(subject, sort_id=None):
    """
    Load sorted spike dataframes
    if sort_id is specified, only load that one
    if sort_id is not specified, load all in sorting_data/spike_dataframes folder

    Args:
        subject (str): subject name
        sort_id (optional): specific sort_id to load. Defaults to None.

    Returns:
        spikes_df: spike dataframe or dictionary of spike dataframes, depending on sort_id
    """
    path = f"/Volumes/opto_loc/Data/{subject}/sorting_data/spike_dataframes/"
    if sort_id:
        key = sort_id + ".parquet"
        spike_dfs = pd.read_parquet(path + key)
    else:
        spike_dfs = {}
        for f in os.listdir(path):
            sort_id = f.split(".")[0]
            spike_dfs[sort_id] = pd.read_parquet(path + f)
    return spike_dfs


@st.cache()
def concat_dfs(df_dict):
    return pd.concat(df_dict.values(), ignore_index=True)


@st.cache()
def get_unique_note_ids(df):
    notes = df.note.values
    note_vals = np.unique(notes)
    all_vals = [n.split("/") for n in note_vals]
    unique_notes = np.unique([i for sl in all_vals for i in sl])
    return unique_notes


@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def simple_fr_plot(df, sort_ids, t1, t2, window, xnote=None, hypno=None):
    f, ax = plt.subplots(figsize=(10, 5))
    exp = sort_ids[0].split("N")[0]
    if xnote:
        for sort_id in sort_ids:
            ax.plot(
                df.sid(sort_id).ts(t1, t2).xnote(xnote).datetime,
                df.sid(sort_id)
                .ts(t1, t2)
                .xnote(xnote)
                .rolling(window, on="datetime")
                .count()
                .cluster_id.values,
                label=sort_id,
            )
    else:
        for sort_id in sort_ids:
            ax.plot(
                df.sid(sort_id).ts(t1, t2).datetime,
                df.sid(sort_id)
                .ts(t1, t2)
                .rolling(window, on="datetime")
                .count()
                .cluster_id.values,
                label=sort_id,
            )
    if hypno:
        kp.shade_hypno_for_me(hypno, ax)
    duration_hrs = (pd.Timestamp(t2) - pd.Timestamp(t1)).total_seconds() / 3600
    ax.set_title(
        f"Simple FR Plot for {exp} ({window} rolling window spike count), Notes Excluded = {xnote}, Duration = {duration_hrs} hrs"
    )
    return f


# -------------------------------------- Main Streamlit App --------------------------------------
if st.checkbox("Load Units"):
    unit_dfs = {}
    for sort_id in sortings_to_load:
        unit_dfs[sort_id] = load_spike_dfs(subject, sort_id=sort_id)
    st.write("Unit Dataframes Loaded")
    df = concat_dfs(unit_dfs)
    st.write("Unit Dataframes Concatenated")


_plot_options = ["Simple FR Comparison", "Filter FR by Notes"]
plot_options = st.sidebar.multiselect("Choose Exploration Plots", _plot_options)


if "Simple FR Comparison" in plot_options:
    st.markdown("## Simple FR Comparison")
    st.markdown(
        "Plot the simple firing rate of units over time using a rolling window, with options to filter by notes, add a hypnogram"
    )
    filter_notes = st.checkbox("Filter by Notes")
    add_hypno = st.checkbox("Add Hypnogram")
    with st.form("Simple FR Comparison"):
        time_span_covered = df.datetime.values.max() - df.datetime.values.min()
        st.write(
            "time span covered: ", df.datetime.values.min(), df.datetime.values.max()
        )
        _t1 = df.datetime.values.min()
        _t2 = _t1 + pd.to_timedelta(1, unit="h")
        t1 = st.text_input("t1", str(_t1))
        t2 = st.text_input("t2", str(_t2))
        rolling_window = st.text_input("Rolling Window", "60s")
        if filter_notes:
            note_ids = get_unique_note_ids(df)
            notes_to_exclude = st.multiselect("Notes to Exclude", note_ids)
        else:
            notes_to_exclude = None

        if add_hypno:
            hypno_to_load = st.text_input("Recording for hypno loading", "rec_name")
        sub = st.form_submit_button("Generate Simple FR Plot")
    if sub:
        hypno = acr.io.load_hypno(subject, hypno_to_load) if add_hypno else None
        f = simple_fr_plot(
            df, sortings_to_load, t1, t2, rolling_window, notes_to_exclude, hypno
        )
        st.pyplot(f)
