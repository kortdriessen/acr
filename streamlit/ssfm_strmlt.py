import sleep_score_for_me.v4 as ssfm
import kd_analysis.main.kd_utils as kd
import kd_analysis.main.kd_utils_pandas as kup
import pandas as pd
import plotly.express as px
import plotly.io as pio
import kd_analysis.ACR.acr_utils as acu
import kd_analysis.ACR.acr_info as ai
import numpy as np
pio.templates.default = "plotly_dark"
import streamlit as st

# Information we need to load the data: 
sub_info = {}
sub_info['subject'] = 'ACR_9'
sub_info['complete_key_list'] = ['control1', 'laser1']
sub_info['stores'] = ['EEGr', 'LFP', 'EMGr']

exp_key = st.selectbox('Select experiment', sub_info['complete_key_list'])
times = ai.a9_times[exp_key]
start_default = times['bl_sleep_start']-30
start = st.sidebar.number_input('Start time', value=start_default, min_value=0, max_value=86400)
length_in_seconds = st.sidebar.number_input('Length in seconds', value=14400, min_value=0, max_value=86400)

@st.cache(allow_output_mutation=True)
def load_scoring_data(t1, t2, exp_key=exp_key):
    sub = sub_info['subject']
    paths = acu.get_paths(sub, sub_info['complete_key_list'])

    eeg = kd.load_sev_store(paths[exp_key], t1=t1, t2=t2, channel=[1,2], store='EEGr')
    emg = kd.load_sev_store(paths[exp_key], t1=t1, t2=t2, channel=[1,2], store='EMGr')
    emg = emg.sel(channel=1)
    return eeg, emg

eeg, emg = load_scoring_data(t1=start, t2=start+length_in_seconds)

#@st.cache(allow_output_mutation=True)
def ss(eeg, emg, chan=1):
    return ssfm.ssfm_v4(eeg, emg, chan)

hypno, fig = ss(eeg, emg, 1)
st.pyplot(fig)

hypno_name = st.text_input('Name of hypno file', '')
hyp_write = st.button('Save hypno')
if hyp_write:
    path = '/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/'+sub_info['subject']+'/dt_hypnograms/hypno_'+hypno_name+'.txt'
    hypno.write(path)