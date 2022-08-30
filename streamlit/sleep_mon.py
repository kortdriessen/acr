import sleep_score_for_me.v4 as ssfm
import kd_analysis.main.kd_utils as kd
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import dask
import xarray

def xr_hash_func(xr_obj):
    hash = dask.base.tokenize(xr_obj)
    return hash

# Get the subject and block-id from the user
subject = st.text_input('Enter the subject ID:')
block_id = st.text_input('Enter the block ID:')
block_path  = Path('/Volumes/opto_loc/Data/' + subject + '/' + subject + '-' + block_id)

#load the data needed for SSFM
@st.cache(allow_output_mutation=True)
def load_scoring_data(emg_chan=1):
    eeg = kd.get_data(block_path, store='EEGr', channel=[1,2])
    emg = kd.get_data(block_path, store='EMGr', channel = [1,2], sel_chan=emg_chan)
    return eeg, emg

eeg, emg = load_scoring_data()

#run SSFM
eeg_chan = st.selectbox('Select EEG channel', list(eeg.channel.values))

def ss(eeg, emg, chan=eeg_chan):
    return ssfm.ssfm_v4(eeg, emg, chan)

hypno, fig = ss(eeg, emg)

#select a day to view
min = eeg.datetime.values.min()
max = eeg.datetime.values.max()
total_days = np.arange(min.astype('datetime64[D]'), max.astype('datetime64[D]')+ np.timedelta64(1, 'D'), np.timedelta64(1, 'D'))
"Range of all loaded data: " + str(min.astype('datetime64[m]')) + " to " + str(max.astype('datetime64[m]'))

day = st.selectbox('Select a day', total_days)

trim_start = day
trim_end = day + np.timedelta64(24, 'h')

trim_start = min if trim_start < min else trim_start
trim_end = max if trim_end > max else trim_end

fig.axes[0].set_xlim(trim_start, trim_end)
fig.axes[1].set_xlim(trim_start, trim_end)
fig.axes[2].set_xlim(trim_start, trim_end)
st.pyplot(fig)

st.session_state