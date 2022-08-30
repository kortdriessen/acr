import kd_analysis.main.kd_utils as kd
import kd_analysis.main.kd_plotting as kp

import pandas as pd
import xarray
import dask

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import kd_analysis.ACR.a9_utils as a9u
import kd_analysis.main.kd_hypno as kh
pio.templates.default = "plotly_dark"

import streamlit as st


def xr_hash_func(xr_obj):
    hash = dask.base.tokenize(xr_obj)
    return hash

@st.cache(hash_funcs={xarray.core.dataarray.DataArray:xr_hash_func})
def acr_bp_rel(spg, hyp, times, state=['NREM'], type='df', key=''):
    #Time values that we will need
    start = spg.datetime.values[0]
    t1 = times['stim_on_dt']
    t2 = times['stim_off_dt']
    
    #Calculate the bandpower values, then cut out only the desired states 
    bp = kd.get_bp_set2(spg, bp_def)
    bp = kh.keep_states(bp, hyp, state)

    #Gets the average bandpowers over the peak period (for the given state)
    avg_period = slice(start, t1)
    avgs = bp.sel(datetime=avg_period).mean(dim='datetime')

    #This expresses everything relative to that mean value over the peak period
    bp = bp/avgs

    #This selects out only the stim period
    bp = bp.sel(datetime=slice(t1, t2))
    
    # NOW HAVE: Stim period bandpower values, from only the desired state(s), relative to their mean value during the peak period

    #This outputs the data in the desired format:
    if type == 'xr':
        return bp
    elif type == 'df':
        bp_df = bp.to_dataframe()
        bp_df = bp_df.reset_index()
        bp_df['key'] = key
        return bp_df

bp_def = dict(delta1=(0.75, 1.75), delta2=(2.5, 3.5), delta=(0.75, 4), theta=(4, 8), alpha = (8, 13), sigma = (11, 16), beta = (13, 30), low_gamma = (30, 55), high_gamma = (65, 90), hz40 = [39, 41])

acr9_info = {}
acr9_info['subject'] = 'ACR_9'
acr9_info['complete_key_list'] = ['control1', 'laser1']


a9, h9, a9_times = a9u.load_data(acr9_info, add_time=None)

ct_lfp = acr_bp_rel(a9['control1-f-s'], h9['control1'], a9_times['control1'], key='Control')
ls_lfp = acr_bp_rel(a9['laser1-f-s'], h9['laser1'], a9_times['laser1'], key='Laser')
rel_stim_bp_lfp = pd.concat([ct_lfp, ls_lfp])

chn = st.radio(
     "select channel to plot",
     (2, 8, 15))

band = st.text_input(label="Bandpower to Plot", value='delta')
nb = st.slider('Number of Bins for histogram')

title = 'Superficial-LFP Delta Values During Sinusoidal Laser Stimulation vs Control, Normalized to Baseline Period | NREM Only'
f = px.histogram(
    rel_stim_bp_lfp.loc[(rel_stim_bp_lfp['channel']==chn)],
    x=band,
    color='key', 
    barmode='overlay',
    opacity=0.6,
    marginal='box',
    color_discrete_sequence=['firebrick', 'cornflowerblue'],
    title=title,
    nbins=nb,
    height=600, 
    width=1600)

f.update_xaxes(title='Normalized Delta Power', range=[0, 2.5])

f

#f, ax = plt.subplots()
#kp.plot_shaded_bp(a9['laser1-e-s'], chn, bp_def, band, h9['laser1'], ax=ax)
#f
