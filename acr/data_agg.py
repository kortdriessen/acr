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
from typing import Union, Tuple, Dict, List, Optional, Any

# IMPORTANT PARAMETERS
# --------------------
REBOUND_LENGTH = '3600s'
REL_STATE = 'NREM'


# ----------------------------------------------------------------------------------------------------------
# ============================= Updated/Best Functions =====================================================
# ----------------------------------------------------------------------------------------------------------

def _label_12_hr_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Label a 12-hour baseline period in the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to label with baseline information.
        
    Returns
    -------
    pd.DataFrame
        The dataframe with a 'full_bl' column added, marking the 12-hour baseline period.
    """
    df['full_bl'] = 'False'
    bl_day = df['datetime'].min()
    bl_day = df['datetime'].min()
    bl_9am = pd.Timestamp(bl_day.date())+pd.Timedelta('9h')
    bl_9pm = bl_9am+pd.Timedelta('12h')
    df.loc[(df['datetime'] >= bl_9am) & (df['datetime'] <= bl_9pm), 'full_bl'] = 'True'
    return df

def gen_bp_df(subject: str, exp: str, save_to: str = 'return', reprocess_existing: bool = True, 
              add_conditions: bool = True, update_hyp: bool = False) -> Optional[pd.DataFrame]:
    """Generate a bandpower dataframe for a subject and experiment.
    
    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.
    save_to : str, optional
        Path to save the dataframe to, or 'return' to return the dataframe, by default 'return'.
    reprocess_existing : bool, optional
        Whether to reprocess existing data, by default True.
    add_conditions : bool, optional
        Whether to add condition labels to the dataframe, by default True.
    update_hyp : bool, optional
        Whether to update the hypnogram, by default False.
        
    Returns
    -------
    Optional[pd.DataFrame]
        The bandpower dataframe if save_to is 'return', otherwise None.
    """
    if os.path.exists(save_to) == True:
        if reprocess_existing == False:
            print(f'{save_to} already exists, skipping')
            return
        else:
            os.system(f'rm -rf {save_to}')

    recs = acr.info_pipeline.get_exp_recs(subject, exp)
    stores = ['NNXo', 'NNXr']
    bp = acr.io.load_concat_bandpower(subject, recs, stores, hypno=True, update_hyp=update_hyp, exclude_bad_channels=False);
    bp_df = bp.to_dataframe().reset_index().drop(columns=['time', 'timedelta', 'recording'])
    bp_df['subject'] = subject
    bp_df['exp'] = exp
    bp_df = bp_df.melt(id_vars=['datetime', 'channel', 'store', 'state', 'subject', 'exp'], 
                    var_name='band', value_name='bandpower')
    if add_conditions == True:
        hd = acr.hypnogram_utils.create_acr_hyp_dict(subject, exp, update=False)
        bp_df = acr.hypnogram_utils.label_df_with_hypno_conditions(bp_df, hd)
        bp_df = _label_12_hr_baseline(bp_df)
    
    if save_to != 'return':
        bp_df.to_parquet(save_to, version="2.6")
    else:
        return bp_df


def load_raw_fr_df(subject: str, exp: str) -> pl.DataFrame:
    """Load a raw firing rate dataframe for a subject and experiment.
    
    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.
        
    Returns
    -------
    pl.DataFrame
        The raw firing rate dataframe.
    """
    return pl.read_parquet(f'./combo_data/raw_fr/{subject}--{exp}.parquet')

def gen_fr_df(subject: str, exp: str, every: int = 2, save_to: str = 'return', 
              reprocess_existing: bool = False, update: bool = False) -> Optional[pl.DataFrame]:
    """Generate a firing rate dataframe for a subject and experiment.
    
    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.
    every : int, optional
        The time bin size in seconds, by default 2.
    save_to : str, optional
        Path to save the dataframe to, or 'return' to return the dataframe, by default 'return'.
    reprocess_existing : bool, optional
        Whether to reprocess existing data, by default False.
    update : bool, optional
        Whether to update the hypnogram, by default False.
        
    Returns
    -------
    Optional[pl.DataFrame]
        The firing rate dataframe if save_to is 'return', otherwise None.
    """
    if os.path.exists(save_to) == True:
        if reprocess_existing == False:
            print(f'{save_to} already exists, skipping')
            return
        else:
            os.system(f'rm -rf {save_to}')
    
    h = acr.io.load_hypno_full_exp(subject, exp, update=update)
    hd = acr.hypnogram_utils.create_acr_hyp_dict(subject, exp, update=False)
    mua = acr.mua.load_concat_peaks_df(subject, exp)
    fr = acr.mua.get_dynamic_fr_df(mua, every=every)
    fr = acr.mua.add_states_to_df(fr, h)
    fr = acr.hypnogram_utils.label_df_with_hypno_conditions(fr, hd)
    fr = fr.with_columns(subject=pl.lit(subject), exp=pl.lit(exp))
    if save_to != 'return':
        fr.write_parquet(save_to)
    else:
        return fr

def _pldf_bp_relative_to_baseline(pldf: pl.DataFrame, col_to_use: str = 'full_bl', 
                                 value_to_use: str = 'True', state_to_use: str = 'NREM', 
                                 method: str = 'mean') -> pl.DataFrame:
    """Make a polars dataframe relative to baseline.
    
    Parameters
    ----------
    pldf : pl.DataFrame
        The polars dataframe to make relative.
    col_to_use : str, optional
        The column to use for filtering baseline data, by default 'full_bl'.
    value_to_use : str, optional
        The value in col_to_use to use for filtering baseline data, by default 'True'.
    state_to_use : str, optional
        The state to use for filtering baseline data, by default 'NREM'.
    method : str, optional
        The method to use for calculating baseline values, by default 'mean'.
        
    Returns
    -------
    pl.DataFrame
        The dataframe with bandpower values made relative to baseline.
    """
    bl_df = pldf.filter(pl.col(col_to_use) == value_to_use).filter(pl.col('state') == state_to_use)
    bl_vals = bl_df.group_by(['store', 'channel', 'band']).mean() if method == 'mean' else bl_df.group_by(['store', 'channel', 'band']).median()
    bl_vals = bl_vals.drop(['datetime', 'state', 'subject', 'exp', 'condition', 'full_bl'])
    rel_df = pldf.join(bl_vals, on=['store', 'channel', 'band'], suffix='_bl')
    rel_vals = rel_df['bandpower'] / rel_df['bandpower_bl']
    rel_df = rel_df.with_columns(bandpower_rel=rel_vals)
    return rel_df

def make_raw_bp_df_relative_to_baseline(df: Union[pd.DataFrame, pl.DataFrame], col_to_use: str = 'full_bl', 
                                       value_to_use: str = 'True', state_to_use: str = 'NREM', 
                                       method: str = 'mean') -> Union[pd.DataFrame, pl.DataFrame]:
    """Make a bandpower dataframe relative to baseline.
    
    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        The dataframe to make relative.
    col_to_use : str, optional
        The column to use for filtering baseline data, by default 'full_bl'.
    value_to_use : str, optional
        The value in col_to_use to use for filtering baseline data, by default 'True'.
    state_to_use : str, optional
        The state to use for filtering baseline data, by default 'NREM'.
    method : str, optional
        The method to use for calculating baseline values, by default 'mean'.
        
    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        The dataframe with bandpower values made relative to baseline.
    """
    if type(df) == pl.DataFrame:
        return _pldf_bp_relative_to_baseline(df, col_to_use, value_to_use, state_to_use, method)
    elif type(df) == pd.DataFrame:
        bl_df = df.loc[df[col_to_use] == value_to_use].loc[df['state'] == state_to_use]
        bl_vals = bl_df.groupby(['store', 'channel', 'band']).mean() if method == 'mean' else bl_df.groupby(['store', 'channel', 'band']).median()
        df_merged = df.merge(bl_vals, on=['store', 'channel', 'band'], suffixes=['', '_bl'])
        df_merged['bandpower_rel'] = df_merged['bandpower'] / df_merged['bandpower_bl']
        return df_merged

def load_raw_bp_df(subject: str, exp: str, method: str = 'pl') -> Union[pd.DataFrame, pl.DataFrame]:
    """Load a raw bandpower dataframe for a subject and experiment.
    
    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.
    method : str, optional
        The method to use for loading the dataframe, either 'pd' for pandas or 'pl' for polars, by default 'pl'.
        
    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        The raw bandpower dataframe.
    """
    if method == 'pd':
        return pd.read_parquet(f'./combo_data/raw_bp/{subject}--{exp}.parquet')
    elif method == 'pl':
        return pl.read_parquet(f'./combo_data/raw_bp/{subject}--{exp}.parquet')
    
def norm_single_exp_reb_to_contra(df: pl.DataFrame, method: str = 'median') -> pl.DataFrame:
    """Requires a df with a single experiment, only one band, in the rebound (or any other unitary condition) only.

    Parameters
    ----------
    df : pl.DataFrame
        The dataframe to normalize.
    method : str, optional
        The method to use to normalize the dataframe, by default 'median'

    Returns
    -------
    pl.DataFrame
        The normalized dataframe.
    """
    keep = ['channel', 'bandpower_rel']
    drop = [col for col in df.columns if col not in keep]
    bl_vals = df.group_by(['store', 'channel']).median().prb('NNXr').drop(drop) if method == 'median' else df.group_by(['store', 'channel']).mean().prb('NNXr').drop(drop)
    df = df.join(bl_vals, on=['channel'], suffix='_contra')
    df = df.with_columns(bandpower_relcc=pl.col('bandpower_rel') / pl.col('bandpower_rel_contra'))
    return df

def read_sc_mask(subject: str, exp: str) -> xr.DataArray:
    """Read a single-channel OFF period mask for a subject and experiment.
    
    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.
        
    Returns
    -------
    xr.DataArray
        The single-channel OFF period mask as an xarray DataArray.
    """
    zmsk = xr.open_zarr(f'./combo_data/sc_full_masks/{subject}--{exp}.zarr')
    return zmsk['sc_mask']

def create_hybrid_off_df(subject: str, exp: str, chan_threshold: int = 9) -> pd.DataFrame:
    """Creates a hybrid-off_df, which is a dataframe of all the OFF periods in the experiment.

    Parameters
    ----------
    subject : str
        The subject to create the hybrid-off_df for.
    exp : str
        The experiment to create the hybrid-off_df for.
    chan_threshold : int, optional
        The threshold for the number of channels needed to be considered an OFF period, by default 9

    Returns
    -------
    pd.DataFrame
        A dataframe of all the OFF periods in the experiment.
    """
    sc_mask = read_sc_mask(subject, exp)
    h = acr.io.load_hypno_full_exp(subject, exp, update=False)
    hd = acr.hypnogram_utils.create_acr_hyp_dict(subject, exp)
    
    bl_start, bl_end = acr.info_pipeline.get_bl_bookends(subject, exp)
    exp_start, exp_end = acr.info_pipeline.get_exp_bookends(subject, exp)
    hd_offs = {}
    hd_offs['full_bl'] = h.keep_between_datetime(bl_start, bl_end)
    hd_offs['full_exp'] = h.keep_between_datetime(exp_start, exp_end)
    hdf_main = pd.DataFrame()
    for condition in hd_offs.keys():
        for probe in ['NNXo', 'NNXr']:
            start_times = []
            end_times = []
            durations = []
            channel_avgs = []
            print(f'Processing {condition} | {probe}')
            start_time = hd_offs[condition]['start_time'].min()
            end_time = hd_offs[condition]['end_time'].max()
            vals = sc_mask.sel(store=probe).ts(start_time, end_time).values
            dtvals = sc_mask.sel(store=probe).ts(start_time, end_time).datetime.values
            counts = vals.sum(axis=0)
            off_ixs, off_lens = acr.onoffmua.find_consecutive_runs(counts, threshold=chan_threshold, min_length=50)
            print(f'Found {len(off_ixs)} OFF periods')
            i=0
            for off_ix, off_len in zip(off_ixs, off_lens):
                print(f'Processing OFF period {i}')
                i+=1
                if off_ix+off_len >= len(counts):
                    off_len = off_len-1
                start_datetime = dtvals[off_ix]
                end_datetime = dtvals[off_ix+off_len]
                off_dur = off_len/1000
                channel_avg = counts[off_ix:off_ix+off_len].mean()
                start_times.append(start_datetime)
                end_times.append(end_datetime)
                durations.append(off_dur)
                channel_avgs.append(channel_avg)
            
            off_df = pd.DataFrame({'probe':probe, 'start_datetime':start_times, 'end_datetime':end_times, 'condition_full':condition, 'duration':durations, 'channel_avg':channel_avgs, 'status':'off', 'chan_threshold':chan_threshold}, index=np.arange(len(start_times)))
            hdf_main = pd.concat([hdf_main, off_df])

    hdf_main = hdf_main.sort_values(by=['start_datetime'])
    hdf_main['subject'] = subject
    hdf_main['exp'] = exp
    hdf_main['area'] = hdf_main['duration']*hdf_main['channel_avg']
    hdf_main = acr.hypnogram_utils.label_df_with_hypno_conditions(hdf_main, hd, col='start_datetime')
    hd_bl = {}
    hd_bl['full_bl'] = h.keep_states(['NREM']).keep_between_datetime(bl_start, bl_end)
    hdf_main = acr.hypnogram_utils.label_df_with_hypno_conditions(hdf_main, hd_bl, label_col='condition_bl', col='start_datetime')
    return hdf_main

def make_hdf_rel(hdf: pd.DataFrame, rel_cond: str = 'full_bl', method: str = 'mean') -> pd.DataFrame:
    """Makes the hybrid off df relative to a baseline condition.

    Parameters
    ----------
    hdf : pd.DataFrame
        The hybrid-off_df to make the relative dataframe from.
    rel_cond : str, optional
        The condition to make the relative dataframe to, by default 'full_bl'
    method : str, optional
        The method to use to make the relative dataframe, by default 'mean'

    Returns
    -------
    pd.DataFrame
        The relative dataframe.
    """
    
    hdf_bl = hdf.loc[hdf['condition_bl']==rel_cond]
    hdf_rel_vals = hdf_bl.groupby('probe').mean().reset_index() if method=='mean' else hdf_bl.groupby('probe').median().reset_index()
    hdf = hdf.merge(hdf_rel_vals.drop(columns=['chan_threshold']), on=['probe'], suffixes=['', '_bl'])
    hdf['area_rel'] = hdf['area']/hdf['area_bl']
    hdf['duration_rel'] = hdf['duration']/hdf['duration_bl']
    hdf['channel_avg_rel'] = hdf['channel_avg']/hdf['channel_avg_bl']
    return hdf

