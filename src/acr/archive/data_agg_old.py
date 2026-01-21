import kdephys as kde
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import acr
import os
from acr.utils import swi_subs_exps, sub_probe_locations, sub_exp_types
from kdephys.hypno.ecephys_hypnogram import trim_hypnogram
import acr.hypnogram_utils as ahu
import xarray as xr

# IMPORTANT PARAMETERS
# --------------------
REBOUND_LENGTH = '3600s'
REL_STATE = 'NREM'

data_path_root = '/home/kdriessen/gh_master/acr/pub/data'

def check_df(subject, exp, data_folder):
    path = f'{data_path_root}/{data_folder}/{subject}--{exp}.parquet'
    return os.path.exists(path)

def get_sd_df(subject, exp, reprocess_existing=False, save=True):
    
    """Generate a dataframe of bandpower values during the stimulation period at the end of sleep deprivation

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    reprocess_existing : bool, optional
        whether to reprocess data that is already in the dataframe, by default False
    save : bool, optional
        whether to save the data, by default True
    """
    
    already_processed = check_df(subject, exp, type='sd')
    if already_processed == True:
        if reprocess_existing == False:
            return
        else:
            os.remove(f'./sd_data/{subject}--{exp}--sd.parquet')
    
    params = acr.info_pipeline.subject_params(subject)
    stores = params['time_stores']
    recordings = acr.info_pipeline.get_exp_recs(subject, exp)
    sd_start, stim_start, stim_end, reb_start, exp_start = acr.info_pipeline.get_sd_exp_landmarks(subject, exp)
    
    # Load the BANDPOWER DATA
    #-------------------------------
    bp = acr.io.load_concat_bandpower(subject, recordings, stores, hypno=True, update_hyp=False);
    bp_rel = kde.xr.utils.rel_by_store(bp, state=REL_STATE, t1=None, t2=None);

    # Get the SD values during the experimental manipulation:
    sd_bp = bp_rel.ts(sd_start, stim_end) 
    bp_df = kde.xr.spectral.bp_melt(sd_bp.to_dataframe().reset_index()) #convert to dataframe
    bp_df['region'] = sub_probe_locations[subject]
    bp_df['exp_type'] = sub_exp_types[subject]
    bp_df['subject'] = subject
    bp_df['exp'] = exp
    stim_df = add_layer_info_to_df(bp_df, subject)
    if save:
        stim_df.to_parquet(f'./sd_data/{subject}--{exp}--sd.parquet', version="2.6") 
    return

def add_layer_info_to_df(df, subject):
    if 'layer' not in df.columns:
        df['layer'] = 'no_layer_info'
    chan_map = acr.info_pipeline.subject_info_section(subject, 'channel_map')
    if chan_map == None:
        print(f'No channel map for {subject}')
        return df
    if len(chan_map) == 0:
        print(f'No channel map for {subject}')
        return df
    
    #check if layer information has already been added:
    stores = df.sbj(subject)['store'].unique()
    chans = df.sbj(subject).prb(stores[0])['channel'].unique()
    test_chan1 = chans[0]
    test_chan2 = chans[-1]
    layer1 = chan_map[stores[0]][str(test_chan1)]['layer']
    layer2 = chan_map[stores[0]][str(test_chan2)]['layer']
    
    if (df.sbj(subject).prb(stores[0]).chnl(test_chan1)['layer'].values[0] == layer1) & (df.sbj(subject).prb(stores[0]).chnl(test_chan2)['layer'].values[0] == layer2):
        print(f'Layer information already added for {subject}')
        return df

    #add layer information:
    for store in stores:
        for chan in df.sbj(subject).prb(store)['channel'].unique():
            layer = chan_map[store][str(chan)]['layer']
            df.loc[(df.subject==subject) & (df.prb(store)['channel']==chan), 'layer'] = layer
    return df



def get_rebound_df(subject, exp, data_folder=None, reprocess_existing=False, save=True, ref_to='full-bl', cumulative=False, norm_method='mean'):
    """Generate a dataframe of bandpower values during the rebound period following sleep deprivation

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    reprocess_existing : bool, optional
        whether to reprocess data that is already in the dataframe, by default False
    save : bool, optional
        whether to save the data, by default True
    ref_to : str, optional
        - full-bl: reference to the full 12-hour baseline period 
        (i.e. the full 12 hour light period of the day before the experiment)
        - circ: after getting the rebound hypnogram, reference to the rebound period of the previous day
        - early: reference to the first 2 hours of the baseline period (previous day)
        - uni-reb: reference to the period between 2 and 5pm on the baseline day.
    cumulative : bool, optional
        - if False, the rebound will last for exactly REBOUND_LENGTH after the reb_start datetime.
        - if True, the rebound will consist of the first REBOUND_LENGTH of cumulative NREM sleep after the reb_start datetime.
    norm_method : str, optional
        - 'mean': all data gets referenced to the baseline mean
        - 'median': all data gets referenced to the baseline median
    """
    if data_folder == None:
        raise ValueError('Must provide a data folder, do not want to accidentally overwrite!')
    already_processed = check_df(subject, exp, data_folder)
    if already_processed == True:
        if reprocess_existing == False:
            print(f'{subject}--{exp} already processed, returning')
            return
        else:
            os.remove(f'{data_path_root}/{data_folder}/{subject}--{exp}.parquet')

    params = acr.info_pipeline.subject_params(subject)
    stores = params['time_stores']
    recordings = acr.info_pipeline.get_exp_recs(subject, exp)
    sd_start, stim_start, stim_end, reb_start, exp_start = acr.info_pipeline.get_sd_exp_landmarks(subject, exp, update=False)
    h = acr.io.load_hypno_full_exp(subject, exp, corrections=True, update=False, float=False)
    
    #Get the rebound hypnogram, so we know how to reference
    if cumulative==True:
        reb_hypno = ahu.get_cumulative_rebound_hypno(h, reb_start, cum_dur=REBOUND_LENGTH, states=[REL_STATE])
    if cumulative==False:
        reb_hypno = trim_hypnogram(h._df, reb_start, reb_start+pd.Timedelta(REBOUND_LENGTH))
    
    # Get t1 and t2 for the baseline normalization, depending on the ref_to parameter
    if ref_to == 'full-bl':
        t1 = None
        t2 = None
    elif ref_to == 'circ':
        t1, t2 = ahu.get_circadian_match_of_rebound(reb_hypno)
    elif ref_to == 'early':
        t1, t2 = ahu.get_previous_day_times(reb_start, t1='09:00:00', t2='12:00:00')
    elif ref_to == 'uni-reb':
        t1, t2 = ahu.get_previous_day_times(reb_start, t1='14:00:00', t2='17:00:00')
    else:
        raise ValueError('ref_to parameter not recognized')
    print(t1, t2)
    if t1 != None and t2 != None:
        if h.hts(t1, t2).st(REL_STATE)['duration'].sum() == 0:
            print(f'No {REL_STATE} sleep in the baseline period, extending t2 by 90 minutes')
            t2 = t2 + pd.Timedelta('90m')
            if h.hts(t1, t2).st(REL_STATE)['duration'].sum() == 0:
                print(f'Still no {REL_STATE} sleep in the baseline period, returning')
                return
        if h.hts(t1, t2).empty:
            print(f'NO states at all in the baseline period, extending t2 by 90 minutes')
            t2 = t2 + pd.Timedelta('90m')
            if h.hts(t1, t2).empty:
                print(f'Still no states in the baseline period, returning')
                return
    
    
    # Load the RAW Bandpower Data
    #-------------------------------
    bp = acr.io.load_concat_bandpower(subject, recordings, stores, hypno=True, update_hyp=False, exclude_bad_channels=False);
    
    # Reference the data to baseline
    bp_rel = kde.xr.utils.rel_by_store(bp, state=REL_STATE, t1=t1, t2=t2, method=norm_method);

    # Get the rebound values:
    rebound_dur = pd.Timedelta(REBOUND_LENGTH)
    reb_bp = bp_rel.ts(reb_start, reb_hypno['end_time'].max()) 
    bp_df = kde.xr.spectral.bp_melt(reb_bp.to_dataframe().reset_index()) #convert to dataframe
    bp_df['region'] = sub_probe_locations[subject]
    bp_df['exp_type'] = sub_exp_types[subject]
    bp_df['subject'] = subject
    bp_df['exp'] = exp
    bp_df = add_layer_info_to_df(bp_df, subject)
    if save:
        bp_df.to_parquet(f'{data_path_root}/{data_folder}/{subject}--{exp}.parquet', version="2.6") 
    else:
        return bp_df



def _get_rebound_df(subject, exp, ref_to='full-bl', cumulative=True, norm_method='mean'):
    """Generate a dataframe of bandpower values during the rebound period following sleep deprivation

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    reprocess_existing : bool, optional
        whether to reprocess data that is already in the dataframe, by default False
    save : bool, optional
        whether to save the data, by default True
    ref_to : str, optional
        - full-bl: reference to the full 12-hour baseline period 
        (i.e. the full 12 hour light period of the day before the experiment)
        - circ: after getting the rebound hypnogram, reference to the rebound period of the previous day
        - early: reference to the first 2 hours of the baseline period (previous day)
        - uni-reb: reference to the period between 2 and 5pm on the baseline day.
    cumulative : bool, optional
        - if False, the rebound will last for exactly REBOUND_LENGTH after the reb_start datetime.
        - if True, the rebound will consist of the first REBOUND_LENGTH of cumulative NREM sleep after the reb_start datetime.
    norm_method : str, optional
        - 'mean': all data gets referenced to the baseline mean
        - 'median': all data gets referenced to the baseline median
    """
    assert subject in sub_probe_locations.keys(), f'{subject} not in sub_probe_locations'
    params = acr.info_pipeline.subject_params(subject)
    stores = params['time_stores']
    recordings = acr.info_pipeline.get_exp_recs(subject, exp)
    sd_start, stim_start, stim_end, reb_start, exp_start = acr.info_pipeline.get_sd_exp_landmarks(subject, exp, update=False)
    h = acr.io.load_hypno_full_exp(subject, exp, corrections=True, update=False, float=False)
    hd = acr.hypnogram_utils.create_acr_hyp_dict(subject, exp)
    #Get the rebound hypnogram, so we know how to reference
    if cumulative==True:
        reb_hypno = ahu.get_cumulative_rebound_hypno(h, reb_start, cum_dur=REBOUND_LENGTH, states=[REL_STATE])
    if cumulative==False:
        reb_hypno = trim_hypnogram(h._df, reb_start, reb_start+pd.Timedelta(REBOUND_LENGTH))
    
    # Get t1 and t2 for the baseline normalization, depending on the ref_to parameter
    if ref_to == 'full-bl':
        t1 = None
        t2 = None
    elif ref_to == 'circ':
        t1 = hd['circ_bl']['start_time'].min()
        t2 = hd['circ_bl']['end_time'].max()
    elif ref_to == 'early':
        t1 = hd['early_bl']['start_time'].min()
        t2 = hd['early_bl']['end_time'].max()
    elif ref_to == 'uni-reb':
        t1, t2 = ahu.get_previous_day_times(reb_start, t1='14:00:00', t2='17:00:00')
    else:
        raise ValueError('ref_to parameter not recognized')
    print(t1, t2)
    if t1 != None and t2 != None:
        if h.hts(t1, t2).st(REL_STATE)['duration'].sum() == 0:
            print(f'No {REL_STATE} sleep in the baseline period, extending t2 by 90 minutes')
            t2 = t2 + pd.Timedelta('90m')
            if h.hts(t1, t2).st(REL_STATE)['duration'].sum() == 0:
                print(f'Still no {REL_STATE} sleep in the baseline period, returning')
                return
        if h.hts(t1, t2).empty:
            print(f'NO states at all in the baseline period, extending t2 by 90 minutes')
            t2 = t2 + pd.Timedelta('90m')
            if h.hts(t1, t2).empty:
                print(f'Still no states in the baseline period, returning')
                return
    
    
    # Load the RAW Bandpower Data
    #-------------------------------
    bp = acr.io.load_concat_bandpower(subject, recordings, stores, hypno=True, update_hyp=False, exclude_bad_channels=False);
    
    # Reference the data to baseline
    bp_rel = kde.xr.utils.rel_by_store(bp, state=REL_STATE, t1=t1, t2=t2, method=norm_method);

    # Get the rebound values:
    reb_bp = bp_rel.ts(reb_start, reb_hypno['end_time'].max()) 
    bp_df = kde.xr.spectral.bp_melt(reb_bp.to_dataframe().reset_index()) #convert to dataframe
    bp_df['region'] = sub_probe_locations[subject]
    bp_df['exp_type'] = sub_exp_types[subject]
    bp_df['subject'] = subject
    bp_df['exp'] = exp
    try:
        bp_df = add_layer_info_to_df(bp_df, subject)
    except:
        print(f'No layer information for {subject}')
    return bp_df
