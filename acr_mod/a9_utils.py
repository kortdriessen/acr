import numpy as np
import pandas as pd
import tdt
from pathlib import Path
import seaborn as sns
import xarray

import kd_analysis.main.kd_utils as kd
import kd_analysis.main.kd_hypno as kh
import kd_analysis.ACR.acr_utils as acu
import streamlit as st
import dask

def xr_hash_func(xr_obj):
    hash = dask.base.tokenize(xr_obj)
    return hash

bp_def = dict(delta1=(0.75, 1.75), delta2=(2.5, 3.5), delta=(0.75, 4), theta=(4, 8), alpha = (8, 13), sigma = (11, 16), beta = (13, 30), low_gamma = (30, 55), high_gamma = (65, 90), hz40 = [39, 41])

kd_ref = {}
kd_ref['echans'] = [1,2]
kd_ref['fchans']=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
kd_ref['analysis_root'] = Path('/Volumes/opto_loc/Data/ACHR_PROJECT_MATERIALS/ACHR_3/ACHR_3-analysis-data')
kd_ref['tank_root'] = Path('/Volumes/opto_loc/Data/ACHR_3/ACHR_3_TANK')

import plotly.io as pio

pio.templates.default = "plotly_dark"

def ss_times(sub, exp):
    def acr_get_times(sub, exp):
        block_path = '/Volumes/opto_loc/Data/'+sub+'/'+sub+'-'+exp
        ep = tdt.read_block(block_path, t1=0, t2=0, evtype=['epocs'])
        times = {}
        times['bl_sleep_start'] = ep.epocs.Bttn.onset[0]
        times['stim_on'] = ep.epocs.Wdr_.onset[-1]
        times['stim_off'] = ep.epocs.Wdr_.offset[-1]

        dt_start = pd.to_datetime(ep.info.start_date)

        on_sec = pd.to_timedelta(times['stim_on'], unit='S')
        off_sec = pd.to_timedelta(times['stim_off'], unit='S')

        times['stim_on_dt'] = dt_start+on_sec
        times['stim_off_dt'] = dt_start+off_sec
        return times
    
    times = acr_get_times(sub, exp)

    start1 = times['bl_sleep_start'] - 30
    end1 = start1 + 7200
    print('FILE #1'), print(start1), print(end1)

    start2 = end1
    end2 = times['stim_off']
    print('FILE #2'), print(start2), print(end2)
    return times

@st.cache(hash_funcs={xarray.core.dataarray.DataArray:xr_hash_func})
def load_data(sub_info, exp_list=None, add_time=None):
    sub = sub_info['subject']
    times = {}

    if exp_list == None:
        exp_list = sub_info['complete_key_list']
    
    for condition in exp_list:
        times[condition] = ss_times(sub, condition)
    paths = acu.get_paths(sub, sub_info['complete_key_list'])   

    a = {}
    h={}
    for condition in exp_list:
       if add_time is not None:
           a[condition+'-e-d'], a[condition+'-e-s'] = kd.get_data_spg(paths[condition], store='EEGr', t1=times[condition]['bl_sleep_start']-30, t2=times[condition]['stim_off']+add_time, channel=[1,2])
           a[condition+'-f-d'], a[condition+'-f-s'] = kd.get_data_spg(paths[condition], store='LFP_', t1=times[condition]['bl_sleep_start']-30, t2=times[condition]['stim_off']+add_time, channel=[2, 8, 15])
       else:
           a[condition+'-e-d'], a[condition+'-e-s'] = kd.get_data_spg(paths[condition], store='EEGr', t1=times[condition]['bl_sleep_start']-30, t2=times[condition]['stim_off'], channel=[1,2])
           a[condition+'-f-d'], a[condition+'-f-s'] = kd.get_data_spg(paths[condition], store='LFP_', t1=times[condition]['bl_sleep_start']-30, t2=times[condition]['stim_off'], channel=[2, 8, 15])
       start_time = a[condition+'-e-d'].datetime.values[0]
       h[condition] = acu.load_hypno_set(sub, condition, scoring_start_time=start_time)
    return a, h, times

def acr_rel2peak(spg, hyp, times, band='delta', ylim=None):
    """
    spg --> xarray.dataarray
    hyp --> hypnogram object
    times --> dictionary (make sure to select the condition)
     """
    
    bp = kd.get_bp_set2(spg, bp_def)
    smooth_bp = kd.get_smoothed_ds(bp, smoothing_sigma=6)
    smooth_nrem_bp = kh.keep_states(smooth_bp, hyp, ['NREM'])

    nrem_spg = kh.keep_states(spg, hyp, ['NREM'])
    nrem_bp = kd.get_bp_set2(nrem_spg, bands=bp_def)
    rel_time_index = np.arange(0, len(smooth_nrem_bp.datetime.values))

    t1 = smooth_nrem_bp.datetime.values[0]
    t2 = times['stim_on_dt']
    avg_period = slice(t1, t2)
    avgs = smooth_nrem_bp.sel(datetime=avg_period).mean(dim='datetime')

    bp_nrem_rel2peak = smooth_nrem_bp/avgs

    bp_nrem_rel2peak = bp_nrem_rel2peak.assign_coords(time_rel=('datetime', rel_time_index))

    return bp_nrem_rel2peak.to_dataframe().reset_index()

@st.cache()
def acr_rel_allstates(spg, hyp, times, band='delta', ylim=None):
    """
    spg --> xarray.dataarray
    hyp --> hypnogram object
    times --> dictionary (make sure to select the condition)
     """
    
    bp = kd.get_bp_set2(spg, bp_def)
    smooth_bp = kd.get_smoothed_ds(bp, smoothing_sigma=6)
    smooth_nrem_bp = kh.keep_states(smooth_bp, hyp, ['NREM'])

    nrem_spg = kh.keep_states(spg, hyp, ['NREM'])
    nrem_bp = kd.get_bp_set2(nrem_spg, bands=bp_def)
    rel_time_index = np.arange(0, len(smooth_bp.datetime.values))

    t1 = smooth_nrem_bp.datetime.values[0]
    t2 = times['stim_on_dt']
    avg_period = slice(t1, t2)
    avgs = smooth_nrem_bp.sel(datetime=avg_period).mean(dim='datetime')

    bp_rel2peak = smooth_bp/avgs

    bp_rel2peak = bp_rel2peak.assign_coords(time_rel=('datetime', rel_time_index))

    return bp_rel2peak.to_dataframe().reset_index()

def acr(spg, hyp, times, band='delta', ylim=None):
    """
    spg --> xarray.dataarray
    hyp --> hypnogram object
    times --> dictionary (make sure to select the condition)
     """
    
    bp = kd.get_bp_set2(spg, bp_def)
    smooth_bp = kd.get_smoothed_ds(bp, smoothing_sigma=6)
    smooth_nrem_bp = kh.keep_states(smooth_bp, hyp, ['NREM'])

    nrem_spg = kh.keep_states(spg, hyp, ['NREM'])
    nrem_bp = kd.get_bp_set2(nrem_spg, bands=bp_def)
    rel_time_index = np.arange(0, len(smooth_nrem_bp.datetime.values))

    t1 = smooth_nrem_bp.datetime.values[0]
    t2 = times['stim_on_dt']
    avg_period = slice(t1, t2)
    avgs = smooth_nrem_bp.sel(datetime=avg_period).mean(dim='datetime')

    bp_nrem_rel2peak = smooth_nrem_bp/avgs

    bp_nrem_rel2peak = bp_nrem_rel2peak.assign_coords(time_rel=('datetime', rel_time_index))

    df4plot = bp_nrem_rel2peak.to_dataframe()
    df4plot.reset_index(inplace=True)
    
    g = sns.FacetGrid(df4plot, row='channel', ylim=ylim, height=3, aspect=6)
    g.map(sns.lineplot, 'time_rel', band)

    return g


    