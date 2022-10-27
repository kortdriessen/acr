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

paths = [Path(folder + "/ks2_5_nblocks=1_8s-batches/") for folder in up_options]


def load_units(paths):
    unit_dfs = []
    for path in paths:
        unit_ex = eu.io.load_sorting_extractor(path)
        unit_df = eu.spikeinterface_sorting_to_dataframe(unit_ex)
        unit_dfs.append(unit_df)
    return unit_dfs


fp = xr.open_dataarray(fp_path)
fp = fp.swap_dims({"datetime": "time"})

unit_dfs = load_units(paths)

channel = st.selectbox("Select channel", fp.channel.values)
t1 = st.slider("Start time", 0, int(fp.time.values.max()), 0)
t2 = st.slider("End time", 0, int(fp.time.values.max()), 5)

fp = fp.sel(channel=channel, time=slice(t1, t2))
plot_dfs = []
for unit_df in unit_dfs:
    unit_df = unit_df.loc[np.logical_and(unit_df.t > t1, unit_df.t < t2)]
    plot_dfs.append(unit_df)


fig = go.Figure()
fig.add_trace(go.Scatter(x=fp.time.values, y=fp.values, mode="lines", name="LFP"))
fig.add_trace(
    go.Scatter(
        x=plot_dfs[0].t.values,
        y=plot_dfs[0].cluster_id.values,
        mode="markers",
        name="Units1",
        yaxis="y2",
    )
)
fig.add_trace(
    go.Scatter(
        x=plot_dfs[1].t.values,
        y=plot_dfs[1].cluster_id.values,
        mode="markers",
        name="Units2",
        yaxis="y2",
    )
)

fig.update_layout(
    yaxis=dict(
        title="y1", titlefont=dict(color="#1f77b4"), tickfont=dict(color="#1f77b4")
    ),
    yaxis2=dict(
        title="y2",
        titlefont=dict(color="#ff7f0e"),
        tickfont=dict(color="#ff7f0e"),
        anchor="free",
        overlaying="y",
        side="left",
    ),
    yaxis3=dict(
        title="y3",
        titlefont=dict(color="#d62728"),
        tickfont=dict(color="#d62728"),
        anchor="free",
        overlaying="y",
        side="right",
    ),
)

st.plotly_chart(fig)
