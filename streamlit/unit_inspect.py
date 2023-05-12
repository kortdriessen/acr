import cudf_tools as ct
import cudf_tools.cudf_flavor as cf
import cudf_tools.unit_methods
from kdephys.xr import xr_flavor_da, xr_flavor_ds
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cudf_tools.unit_analysis import *
from kdephys.plot.main import *
from kdephys.plot.units import *
from acr.stim import get_pulse_train_times, get_total_spike_rate
from acr.units import get_cluster_notes, get_fr_by_cluster, get_fr_suppression_by_cluster
import streamlit as st
plt.style.use('fast')
plt.style.use('/home/kdriessen/github_t2/kdephys/kdephys/plot/acr_plots.mplstyle')

@st.cache_data
def load_hypnogram(subject, exp):
    return acr.io.load_hypno_full_exp(subject, exp)

@st.cache_data
def load_spike_cudf(subject, sort_id=None, test=False, refresh_state=False):
    """
    Load sorted spike dataframes as cuda dataframes
    if sort_id is specified, load those and concatenate into one dataframe
    if sort_id is not specified, load all in sorting_data/spike_dataframes folder, as a dictionary of dataframes

    Args:
        subject (str): subject name
        sort_id (optional, list): specific sort_ids to load. Defaults to None.

    Returns:
        spikes_df: spike dataframe or dictionary of spike dataframes, depending on sort_id
    """
    path = f"/Volumes/opto_loc/Data/{subject}/sorting_data/spike_dataframes/"
    if sort_id:
        sdfs = []
        for sid in sort_id:
            key = sid + ".parquet"
            cuda_df = cudf.read_parquet(path + key)
            if refresh_state:
                recs = au.get_sorting_recs(subject, sid)
                cuda_df = add_hypno_cudf(cuda_df, subject, recs)
            sdfs.append(cuda_df)
            spike_dfs = cudf.concat(sdfs)
    else:
        spike_dfs = {}
        for f in os.listdir(path):
            sort_id = f.split(".")[0]
            spike_dfs[sort_id] = cudf.read_parquet(path + f)
            if refresh_state:
                recs = au.get_sorting_recs(subject, sort_id)
                spike_dfs[sort_id] = add_hypno_cudf(spike_dfs[sort_id], subject, recs)
    if test:
        spike_dfs = spike_dfs.iloc[::10000] # for testing
    return spike_dfs


def plot_all_clusters(df, probe, t1, t2, window, hypno=None):
    n_clust = len(df.prb(probe).cid_un())
    f, ax = plt.subplots(n_clust, 1, figsize=(25, 6*n_clust)) 
    for i, cluster in enumerate(df.prb(probe).cid_un()):
        df_to_plot = df_rel(df.ts(t1, t2).prb(probe).cid(cluster), window)
        ax[i].plot(df_to_plot.index, df_to_plot.rel_rate)
        ax[i].set_title(f"Cluster {cluster} | {probe} | {df.prb(probe).cid(cluster).note.values_host[0]}")
        if hypno:
            shade_hypno_for_me(hypno, ax[i])
        del df_to_plot
    return f, ax

root = Path("/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/")
subjects = []
for f in os.listdir(root):
    full_path = os.path.join(root, f)
    if np.logical_and("ACR_" in f, os.path.isdir(full_path)):
        subjects.append(f)

subject = st.sidebar.selectbox("Subject", subjects)
exp = st.sidebar.text_input("Experiment", "exp_name")
st.markdown(f"# Unit Inspection - {subject}")

unit_df_dir = f"/Volumes/opto_loc/Data/{subject}/sorting_data/spike_dataframes/"
available_sortings = []
for f in os.listdir(unit_df_dir):
    if exp in f:
        available_sortings.append(f.split(".")[0])

sortings_to_load = st.sidebar.multiselect("Choose your sortings", available_sortings)

if st.checkbox("Load Units"):
    df = load_spike_cudf(subject, sortings_to_load, test=False)

if st.checkbox("Load Hypnogram"):
    h = load_hypnogram(subject, exp)

st.markdown(f'# Cluster Notes')
if st.checkbox("Display Cluster Notes - NNXr"):
    notes = get_cluster_notes(df.prb('NNXr'))
    st.write(notes)

if st.checkbox("Display Cluster Notes - NNXo"):
    notes = get_cluster_notes(df.prb('NNXo'))
    st.write(notes)

st.markdown(f'# All Individual Clusters')
probe = st.selectbox("Probe", ["NNXr", "NNXo"])
if st.checkbox("Plot Clusters"):
    t1 = pd.Timestamp(df.prb(probe).datetime.min())
    t2 = pd.Timestamp(df.prb(probe).datetime.max())
    f, ax = plot_all_clusters(df, probe, t1, t2, window='600s')
    st.pyplot(f)

