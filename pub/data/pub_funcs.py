import kdephys as kde
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import acr
import os
from acr.utils import swi_subs_exps, sub_probe_locations, sub_exp_types


# IMPORTANT PARAMETERS
# --------------------
REBOUND_LENGTH = '1h'
REL_STATE = 'NREM'
data_path_root = '/home/kdriessen/gh_master/acr/pub/data'

def check_df(subject, exp, type='reb'):
    if type == 'reb':
        path = f'./rebound_data_1h/{subject}--{exp}--reb.parquet'
    elif type == 'sd':
        path = f'./sd_data/{subject}--{exp}--sd.parquet'
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



def get_rebound_df(subject, exp, reprocess_existing=False, save=True):
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
    """
    already_processed = check_df(subject, exp, type='reb')
    if already_processed == True:
        if reprocess_existing == False:
            return
        else:
            os.remove(f'./rebound_data_1h/{subject}--{exp}--reb.parquet')

    params = acr.info_pipeline.subject_params(subject)
    stores = params['time_stores']
    recordings = acr.info_pipeline.get_exp_recs(subject, exp)
    sd_start, stim_start, stim_end, reb_start, exp_start = acr.info_pipeline.get_sd_exp_landmarks(subject, exp)
    
    # Load the BANDPOWER DATA
    #-------------------------------
    bp = acr.io.load_concat_bandpower(subject, recordings, stores, hypno=True, update_hyp=False, exclude_bad_channels=True);
    bp_rel = kde.xr.utils.rel_by_store(bp, state=REL_STATE, t1=None, t2=None);

    # Get the rebound values:
    rebound_dur = pd.Timedelta(REBOUND_LENGTH)
    reb_bp = bp_rel.ts(reb_start, reb_start+rebound_dur) 
    bp_df = kde.xr.spectral.bp_melt(reb_bp.to_dataframe().reset_index()) #convert to dataframe
    bp_df['region'] = sub_probe_locations[subject]
    bp_df['exp_type'] = sub_exp_types[subject]
    bp_df['subject'] = subject
    bp_df['exp'] = exp
    bp_df = add_layer_info_to_df(bp_df, subject)
    if save:
        bp_df.to_parquet(f'./rebound_data_1h/{subject}--{exp}--reb.parquet', version="2.6") 
    return