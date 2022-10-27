import streamlit as st
import xarray as xr
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path
import dask

sub = st.text_input("Subject", "ACR_14")
exp = st.text_input("Enter experiment for loading of field potentials", "short2-iso")
store = st.text_input("Enter field potential data store", "LFP_")
fp_path = f"/Volumes/opto_loc/Data/{sub}/{exp}-{store}.nc"


up = st.text_input("Enter path to unit data", "/nvme/sorting/tdt/")
unit_path = Path(up)


def get_sorted_data(root=Path(f"/nvme/sorting/tdt/")):
    root = Path(root)
    files = [str(p) for p in root.glob("*")]
    return files


folders = get_sorted_data()
up_options = st.multiselect("Choose unit data", folders)

paths = [folder + "/ks2_5_nblocks=1_8s-batches/" for folder in up_options]

fp = xr.open_dataset(fp_path)
fp = fp.swap_dims({"datetime": "time"})


def load_units(paths):
    unit_dfs = []
    for path in paths:
        unit_ex = eu.io.load_sorting_extractor(path)
        unit_df = eu.spikeinterface_sorting_to_dataframe(unit_ex)
        unit_dfs.append(unit_df)
    return unit_dfs


# @st.cache(hash_funcs={xr.core.dataarray.DataArray: xr_hash_func})
def load_data(fp_path, unit_path):
    fp = xr.load_dataarray(fp_path)
    fp = fp.swap_dims({"datetime": "time"})
    unit_ex = eu.io.load_sorting_extractor(unit_path)
    unit_df = eu.spikeinterface_sorting_to_dataframe(unit_ex)
    return fp, unit_df


fp, unit_df = load_data(fp_path, unit_path)


channel = st.selectbox("Select channel", fp.channel.values)
t1 = st.slider("Start time", 0, int(fp.time.values.max()), 0)
t2 = st.slider("End time", 0, int(fp.time.values.max()), 5)


def prep_plotting_data(fp, unit_df, channel, t1, t2):
    fp = fp.sel(channel=channel, time=slice(t1, t2))
    unit_df = unit_df.loc[np.logical_and(unit_df["t"] > t1, unit_df["t"] < t2)]
    return fp, unit_df


# @st.cache()
def plot_inspector(fp, unit_df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=fp.time, y=fp, name="LFP"), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=unit_df["t"], y=unit_df["cluster_id"], mode="markers", name="Units"
        ),
        row=2,
        col=1,
    )
    return fig


def plot_insp(fp, unit_df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=fp.time, y=fp, name="LFP"), secondary_y=False)
    fig.add_trace(
        go.Scatter(
            x=unit_df["t"], y=unit_df["cluster_id"], mode="markers", name="Units"
        ),
        secondary_y=True,
    )
    fig.update_yaxes(showgrid=False)
    return fig


fp, unit_df = prep_plotting_data(fp, unit_df, channel, t1, t2)
fig = plot_insp(fp, unit_df)
st.plotly_chart(fig)
