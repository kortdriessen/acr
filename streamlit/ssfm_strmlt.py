import numpy as np
import pandas as pd
import tdt
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import xarray as xr

import kdephys.hypno as kh
import kdephys.pd as kp
import kdephys.xr as kx
import kdephys.utils as ku
import kdephys.ssfm as ss
import ipywidgets as wd
import acr

pio.templates.default = "plotly_dark"
import streamlit as st
import acr.subjects as asub

# Information we need to load the data:

sub_info = asub.a12_info

exp_key = st.selectbox("Select experiment", sub_info["complete_key_list"])
start_default = sub_info["load_times"][exp_key][0]
start = st.sidebar.number_input(
    "Start time", value=start_default, min_value=0, max_value=86400
)
length_default = sub_info["load_times"][exp_key][1]
length_in_seconds = st.sidebar.number_input(
    "Length in seconds", value=length_default, min_value=0, max_value=86400
)


@st.cache(allow_output_mutation=True)
def load_scoring_data(t1, t2, exp_key=exp_key):
    sub = sub_info["subject"]
    paths = acr.io.get_acr_paths(sub, sub_info["complete_key_list"])

    eeg = kx.io.load_sev_store(
        paths[exp_key], t1=t1, t2=t2, channel=[1, 2], store="EEGr"
    )
    emg = kx.io.load_sev_store(
        paths[exp_key], t1=t1, t2=t2, channel=[1, 2], store="EMGr"
    )
    emg = emg.sel(channel=1)
    return eeg, emg


eeg, emg = load_scoring_data(t1=start, t2=start + length_in_seconds)

# @st.cache(allow_output_mutation=True)
def ss(eeg, emg, chan=1):
    return ss.ssfm_v4(eeg, emg, chan)


hypno, fig = ss(eeg, emg, 1)
st.pyplot(fig)

hypno_name = st.text_input("Name of hypno file", "")
hyp_write = st.button("Save hypno")
if hyp_write:
    path = (
        "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/"
        + sub_info["subject"]
        + "/hypnograms/hypno_"
        + hypno_name
        + ".txt"
    )
    hypno.write(path)
