import kd_analysis.main.kd_utils as kd
import pandas as pd
import plotly.express as px
import plotly.io as pio
import kd_analysis.main.kd_pandas as kpd
import kd_analysis.main.kd_plotting as kp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
pio.templates.default = "plotly_dark"
import streamlit as st
from scipy.ndimage import gaussian_filter
import kd_analysis.ACR.acr_info as ai

t = ai.a9_times

control_laser_on = t['control1']['stim_on_dt']
control_laser_off = t['control1']['stim_off_dt']
laser_laser_on = t['laser1']['stim_on_dt']
laser_laser_off = t['laser1']['stim_off_dt']
cds = ['white', 'royalblue']
bands = ['delta1', 'delta2', 'delta', 'theta', 'alpha', 'sigma', 'beta', 'low_gamma', 'high_gamma']

sub_info = {}
sub_info['subject'] = 'ACR_9'
sub_info['complete_key_list'] = ['control1', 'laser1']
sub_info['stores'] = ['EEGr', 'LFP']

cond_list = ['control1-eeg', 'control1-lfp', 'laser1-eeg', 'laser1-lfp']
path_root = '/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/ACR_9/analysis-data/'

@st.cache(allow_output_mutation=True)
def load_data(path_root, cond_list, type):
    return kpd.load_dataset(path_root, cond_list, type)
@st.cache()
def load_hypnos(path_root, cond_list=['control1', 'laser1']):
    return kpd.load_hypnos(path_root, cond_list)

@st.cache()
def get_smoothed_column(df, col, ss=8):
    return gaussian_filter(df[col].values, ss)

bp = load_data(path_root, cond_list, '-bp')
hyp = load_hypnos(path_root)



st.markdown('# NREM-Only Bandpower over Defined Time Intervals, Relative to Baseline Period')
@st.cache()
def get_rel_bp_set(df, laser_on):
    bl_period = slice(df.datetime.min(), laser_on)
    bl_df = df.ts(bl_period)
    df = df.set_index(['datetime', 'channel'])
    bl_df.set_index(['datetime', 'channel'], inplace=True)
    means = bl_df.groupby(['channel']).mean()
    for col in list(means.columns):
        df[col] = df[col]/means[col]
    df = df.reset_index()
    return df

@st.cache()
def get_rel_bp_set_all(bp_set):
    bp_rel = {}
    for key in list(bp_set.keys()):
        on_time = control_laser_on if 'control1' in key else laser_laser_on
        bp_rel[key] = get_rel_bp_set(bp_set[key], on_time)
    return bp_rel
bp_rel = get_rel_bp_set_all(bp)
dtype = st.selectbox('Data Type', ['-eeg', '-lfp'])
df_time_int = pd.concat([bp_rel[key+dtype] for key in sub_info['complete_key_list']])
channel = st.selectbox('Channel', range(1, df_time_int.channel.max()+1))
band = st.selectbox('BandPower', bands)
fig = px.box(df_time_int.ch(channel).filt_state(), x='time_class', y=band, color='condition', color_discrete_sequence=cds, title=band+' Bandpower Relative to Baseline Period, '+dtype+' Channel-'+str(channel))
st.plotly_chart(fig)

st.markdown('# Bandpower/Hypno Overview')
def plot_bp_with_hypno(bp, hyp, chan=1, dtype='-eeg'):
    n = len(hyp.keys())
    f, axes = plt.subplots(n, 1, figsize=(24, 10))
    for i, key in enumerate(hyp.keys()):
        df = bp[key+dtype].ch(chan)
        delta_smooth = get_smoothed_column(df, 'delta')
        axes[i].plot(df.datetime.values, delta_smooth)
        kp.shade_hypno_for_me(hyp[key], axes[i])
        axes[i].set_title(key+dtype+' channel-'+str(chan))
    return f, axes
f,a = plot_bp_with_hypno(bp_rel, hyp, chan=15, dtype='-lfp')
a[0].axvline(control_laser_on, color='r', linestyle='--')
a[0].axvline(control_laser_off, color='r', linestyle='--')
a[1].axvline(laser_laser_on, color='r', linestyle='--')
a[1].axvline(laser_laser_off, color='r', linestyle='--')

st.pyplot(f)