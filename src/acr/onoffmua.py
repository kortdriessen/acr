import acr
import numpy as np
import pandas as pd
import polars as pl
import os
from acr.utils import raw_data_root
import matplotlib.pyplot as plt
import xarray as xr
import dask.array as da

import numba as nb
from datetime import datetime, timedelta
from pathlib import Path
from acr.utils import pub_data_root

#____________________________________________________________________________________________________________________________________
#=====================================         ACR MUA-BASED OFF PERIOD FUNCITONS        ============================================
# Generally used for working with mua-based on-off detection functions. For improved utils created with the spike-bin-based development, see the oo_utils module.      
# ___________________________________________________________________________________________________________________________________

def label_oodf_with_hypno_conditions(oodf, hd):
    print('acr.oo_utils.label_oodf_hyp_dict_conditions is better and faster, use that!')
    if 'condition' not in oodf.columns:
        oodf = oodf.with_columns(condition=pl.lit('None'))
    oodf_pd = oodf.to_pandas()
    for key in hd.keys():
        for bout in hd[key].itertuples():
            oodf_pd.loc[((oodf_pd['start_datetime'] >= bout.start_time) & (oodf_pd['end_datetime'] <= bout.end_time)), 'condition'] = key
    return pl.DataFrame(oodf_pd)

def make_oodf_relative(oodf, by='channel', ref_to='early_bl', avg_method='mean', make_rel_col='duration'):
    print('acr.oo_utils.relativize_oodf is better (unless operating on single channels!), use that where possible!')
    assert 'condition' in oodf.columns, 'oodf must be labeled with condition'
    assert ref_to in oodf['condition'].unique(), 'ref_to condition not found in oodf'
    refdf = oodf.filter(pl.col('condition')==ref_to)
    
    if by == 'channel':
        ref_avgs = refdf.groupby(['probe', 'channel', 'status']).median() if avg_method == 'median' else refdf.groupby(['probe', 'channel', 'status']).mean()
        oodf = oodf.with_columns(pl.lit(101).alias(f'rel_{make_rel_col}'))
        for probe in oodf['probe'].unique():
            for channel in oodf['channel'].unique():
                for status in oodf['status'].unique():
                    bl_val = ref_avgs.filter(pl.col('probe')==probe).filter(pl.col('channel')==channel).filter(pl.col('status')==status)[make_rel_col].to_numpy()[0]
                    oodf = oodf.with_columns(pl.when(((pl.col('probe')==probe) & (pl.col('channel')==channel) & (pl.col('status')==status)))
                                            .then(pl.col(make_rel_col)/bl_val)
                                            .otherwise(pl.col(f'rel_{make_rel_col}'))
                                            .alias(f'rel_{make_rel_col}'))
        return oodf
    elif by == 'probe':
        ref_avgs = refdf.groupby(['probe', 'status']).median() if avg_method == 'median' else refdf.groupby(['probe', 'status']).mean()
        oodf = oodf.with_columns(pl.lit(101).alias(f'rel_{make_rel_col}'))
        for probe in oodf['probe'].unique():
            for status in oodf['status'].unique():
                bl_val = ref_avgs.filter(pl.col('probe')==probe).filter(pl.col('status')==status)[make_rel_col].to_numpy()[0]
                oodf = oodf.with_columns(pl.when(((pl.col('probe')==probe) & (pl.col('status')==status)))
                                            .then(pl.col(make_rel_col)/bl_val)
                                            .otherwise(pl.col(f'rel_{make_rel_col}'))
                                            .alias(f'rel_{make_rel_col}'))
        return oodf

    
def get_oodf_freqs(oodf, hd):
    if type(oodf) != pl.DataFrame:
        oodf = pl.DataFrame(oodf)
    if 'channel' in oodf.columns:
        return _oodf_freqs_by_chan(oodf, hd)
    elif 'probe' in oodf.columns and 'channel' not in oodf.columns:
        return _oodf_freqs_by_probe(oodf, hd)

def _oodf_freqs_by_chan(oodf, hd):
    freqs = {}
    for freq_cond in hd.keys():
        freqs[freq_cond] = []
        cdn_hyp = hd[freq_cond]
        condoo = oodf.filter(pl.col('condition')==freq_cond)
        for bout in cdn_hyp.itertuples():
            bout_dur = (bout.end_time - bout.start_time).total_seconds()
            counts = condoo.oots(bout.start_time, bout.end_time).groupby(['probe', 'channel', 'status']).agg(pl.count('duration').alias('pure_count'))
            counts = counts.with_columns(pl.lit(bout_dur).alias('bout_duration'))
            counts = counts.with_columns(condition = pl.lit(freq_cond))
            freqs[freq_cond].append(counts)
    full_freq_dfs = []
    for b in freqs.keys():
        if len(freqs[b]) == 0:
            continue
        full_cond_df = pl.concat(freqs[b])
        full_freq_dfs.append(full_cond_df)
    full_freq_df = pl.concat(full_freq_dfs)
    return full_freq_df.sort(['probe', 'channel', 'status', 'condition'])


def _oodf_freqs_by_probe(oodf, hd):
    freqs = {}
    for freq_cond in hd.keys():
        freqs[freq_cond] = []
        cdn_hyp = hd[freq_cond]
        condoo = oodf.filter(pl.col('condition')==freq_cond)
        for bout in cdn_hyp.itertuples():
            bout_dur = (bout.end_time - bout.start_time).total_seconds()
            counts = condoo.oots(bout.start_time, bout.end_time).group_by(['probe', 'status']).agg(pl.count('duration').alias('pure_count'))
            counts = counts.with_columns(pl.lit(bout_dur).alias('bout_duration'))
            counts = counts.with_columns(condition = pl.lit(freq_cond))
            freqs[freq_cond].append(counts)
    full_freq_dfs = []
    for b in freqs.keys():
        if len(freqs[b]) == 0:
            continue
        full_cond_df = pl.concat(freqs[b])
        full_freq_dfs.append(full_cond_df)
    full_freq_df = pl.concat(full_freq_dfs)
    return full_freq_df.sort(['probe', 'status', 'condition'])


def __rel_avg_freq_df(total_freqs, rel_to='early_bl'):
    total_freqs = total_freqs.with_columns(rel_freq=pl.lit(101))
    bl_freqs = total_freqs.filter(pl.col('condition')==rel_to)
    for probe in total_freqs['probe'].unique():
        for channel in total_freqs['channel'].unique():
            for status in total_freqs['status'].unique():
                bl_val = bl_freqs.filter(pl.col('probe')==probe).filter(pl.col('channel')==channel).filter(pl.col('status')==status)['freq'].to_numpy()[0]
                total_freqs = total_freqs.with_columns(pl.when(((pl.col('probe')==probe) & (pl.col('channel')==channel) & (pl.col('status')==status)))
                                        .then(pl.col('freq')/bl_val)
                                        .otherwise(pl.col('rel_freq')).alias('rel_freq'))
    return total_freqs


def compute_oodf_from_t_array(t, off_min_dur=.050000, synth_dur=2.0):
    diffs = t.diff().dt.total_seconds().values

    msk = np.ma.masked_where(diffs >= (off_min_dur), diffs)
    msk = msk.mask
    if type(msk) == np.bool_:
        offs = pd.DataFrame({'start_datetime': t[0], 'end_datetime': t[0], 'status': 'off'}, index=[0])
        ons = pd.DataFrame({'start_datetime': t[0], 'end_datetime': t[0], 'status': 'on'}, index=[0])
        oodf = pd.concat([ons, offs])
        oodf['duration'] = (oodf['end_datetime'] - oodf['start_datetime']).dt.total_seconds()
        return oodf

    # Convert the mask to integers (False = 0, True = 1) and compute the difference
    diff = np.diff(msk.astype(int))

    # Step 2: Find the start indices (where diff == 1, indicating a switch from False to True)
    start_indices = np.where(diff == 1)[0] + 1

    # Step 3: Find the end indices (where diff == -1, indicating a switch from True to False)
    end_indices = np.where(diff == -1)[0] + 1

    # Handle edge cases where the mask starts or ends with True
    if msk[0]:
        start_indices = np.r_[0, start_indices]  # Add the first index if it starts with True
    if msk[-1]:
        end_indices = np.r_[end_indices, len(msk)]  # Add the last index if it ends with True
    
    valid_lengths = (end_indices - start_indices)>=2
    all_lengths = end_indices - start_indices
    print(f'Found {len(all_lengths)} off periods, {sum(valid_lengths)} of which are valid')
    good_starts = start_indices[valid_lengths]
    good_ends = end_indices[valid_lengths]
    true_lengths = good_ends - good_starts

    # ADD THE SYNTHETIC SPIKES
    # Initialize an empty list to store all the starts
    all_starts = []
    starts_to_add = true_lengths-2
    # Iterate through good_starts and starts_to_add simultaneously
    for start, add_count in zip(good_starts, starts_to_add):
        all_starts.append(start)
        # Add additional starts if needed
        for i in range(1, add_count + 1):
            all_starts.append(start + i)

    # Convert the list to a numpy array
    spikes_to_synth = np.array(all_starts)
    spike_times_to_be_synthed = t[spikes_to_synth]
    synth_befores = spike_times_to_be_synthed - pd.Timedelta(milliseconds=synth_dur)
    synth_afters = spike_times_to_be_synthed + pd.Timedelta(milliseconds=synth_dur)
    
    # now we integrate the synthetic spikes into the original t array
    t_synthed = t.copy()
    t_synthed = np.append(t_synthed, synth_befores)
    t_synthed = np.append(t_synthed, synth_afters)
    t_synthed = np.sort(t_synthed)
    
    # NOW WE PROCEED WITH THE ACTUAL OFF DETECTION AND OODF CREATION
    diffs = pd.Series(t_synthed).diff().dt.total_seconds().values
    msk = np.ma.masked_where(diffs >= (off_min_dur), diffs)
    start_times_mask = msk.mask
    #start_times_mask = np.append(start_times_mask, False)
    start_times_indices = np.where(start_times_mask == True)[0]
    start_times_indices = start_times_indices-1
    end_times_indices = start_times_indices + 1
    start_times = t_synthed[start_times_indices]
    end_times = t_synthed[end_times_indices]
    off_det = pd.DataFrame(
                    {
                    "start_datetime": start_times,
                    "end_datetime": end_times,
                    "status": "off",
                }
            )
    off_det['duration'] = (off_det['end_datetime'] - off_det['start_datetime']).dt.total_seconds()
    
    on_starts = end_times
    _on_ends = start_times
    on_ends = np.append(_on_ends[1:], t_synthed[-1])
    on_starts = np.insert(on_starts, 0, t_synthed[0])
    on_ends = np.insert(on_ends, 0, _on_ends[0])

    on_det = pd.DataFrame(
        {
            'start_datetime': on_starts,
            'end_datetime': on_ends,
            'status': 'on',
        }
    )
    on_det['duration'] = (on_det['end_datetime'] - on_det['start_datetime']).dt.total_seconds()
    oodf = (
        pd.concat([on_det, off_det])
        .sort_values(by="start_datetime")
        .reset_index(drop=True)
    )
    return oodf

def _get_full_oodf_by_channel_slow(mua, off_min_dur=.050, synth_dur=2.0):
    oodfs = []
    subject = mua['subject'].unique()[0]
    for probe in mua['probe'].unique():
        for rec in mua.prb(probe)['recording'].unique():
            for chnl in mua.prb(probe)['channel'].unique():
                rec_mua = mua.filter((pl.col('recording')==rec)&(pl.col('channel')==chnl)&(pl.col('probe')==probe))
                t = rec_mua['datetime'].to_pandas()
                oodf = compute_oodf_from_t_array(t, off_min_dur=off_min_dur, synth_dur=synth_dur)
                oodf['subject'] = subject
                oodf['probe'] = probe
                oodf['recording'] = rec
                oodf['channel'] = chnl
                oodfs.append(oodf)
    return pd.concat(oodfs)

def compute_full_oodf_by_channel(mua, off_min_dur=.050, synth_dur=2.0):
    oodfs = []
    subject = mua['subject'].unique()[0]
    chan_times = mua.groupby(['recording', 'probe', 'channel']).agg(pl.col('datetime')).sort(['recording', 'probe', 'channel'])
    recs = chan_times['recording'].unique().to_list()
    recs.sort()
    probes = chan_times['probe'].unique().to_list()
    print(probes)
    probes.sort()
    print(probes)
    channels = chan_times['channel'].unique().to_list()
    channels.sort()
    assert len(chan_times) == len(recs)*len(probes)*len(channels), 'This function relies on an exact match of recording, probe, and channel across the mua dataframe. All recordings must have both probes, both of which must have all channels. Use the _get_full_oodf_by_channel_slow function if this is not the case.'
    chan_t_arrays = chan_times['datetime'].to_numpy()
    count = 0
    for rec in recs:
        for probe in probes:
            for chnl in channels:
                print(f'{rec} {probe} {chnl}')
                t = pd.Series(chan_t_arrays[count])
                oodf = compute_oodf_from_t_array(t, off_min_dur=off_min_dur, synth_dur=synth_dur)
                oodf['subject'] = subject
                oodf['probe'] = probe
                oodf['recording'] = rec
                oodf['channel'] = chnl
                oodfs.append(pl.DataFrame(oodf))
                count += 1
    return pl.concat(oodfs)

def compute_full_oodf_by_probe(mua, off_min_dur=.050, synth_dur=2.0):
    oodfs = []
    subject = mua['subject'].unique()[0]
    for probe in mua['probe'].unique():
        for rec in mua.prb(probe)['recording'].unique():
            rec_mua = mua.filter((pl.col('recording')==rec)&(pl.col('probe')==probe)).sort('datetime')
            t = rec_mua['datetime'].to_pandas()
            oodf = compute_oodf_from_t_array(t, off_min_dur=off_min_dur, synth_dur=synth_dur)
            oodf['subject'] = subject
            oodf['probe'] = probe
            oodf['recording'] = rec
            oodfs.append(oodf)
    fdf = pd.concat(oodfs)
    return pl.DataFrame(fdf)

def get_hypno_off_durs(df, h):
    """Loops through every bout in a hypnogram, counts the number of spikes and total duration for each probe-channel pair, then adds all of those to a new dataframe. 

    Parameters
    ----------
    df : _type_
        _description_
    h : _type_
        _description_
    """
    h_off_durs = []
    bout_num = 1
    for bout in h.itertuples():
        bout_start = bout.start_time
        bout_end = bout.end_time
        duration = (bout_end - bout_start).total_seconds()
        bout_df = df.ts(bout_start, bout_end)
        bout_df = bout_df.groupby(['probe', 'channel']).count()
        bout_df = bout_df.with_columns(bout_duration=pl.lit(duration))
        bout_df = bout_df.with_columns(bout_number=pl.lit(bout_num))
        h_off_durs.append(bout_df)
        bout_num += 1
    full_df = pl.concat(h_off_durs)
    return full_df


def get_reb_and_bl_off_durs(df, bh, rh):
    bl_off_durs = acr.mua.get_hypno_frs(df, bh)
    reb_frs = acr.mua.get_hypno_frs(df, rh)
    bl_frs = bl_frs.with_columns(cond=pl.lit('baseline'))
    reb_frs = reb_frs.with_columns(cond=pl.lit('rebound'))
    return pl.concat([bl_frs, reb_frs])


def add_off_spans_by_chan(oodf, ax):
    "needs to be a single probe oodf, with time already selected"
    if type(oodf) == pl.DataFrame:
        oodf = oodf.to_pandas()
    num_colors = 16
    cmap = plt.get_cmap('viridis_r')
    color_indices = np.linspace(0, 1, num_colors)
    colors = [cmap(index) for index in color_indices]
    xmins = {}
    xmaxs = {}
    yvals = {}
    starts = oodf.groupby('channel')['start_datetime'].apply(list)
    ends = oodf.groupby('channel')['end_datetime'].apply(list)
    for chan in starts.keys():
        yvals = np.full(len(starts[chan]), chan*-1)
        ax.hlines(y=yvals, xmin=starts[chan], xmax=ends[chan], color=colors[chan-1], linewidth=12, alpha=0.8)
    return ax

def true_strictify_oodf(oodf, min_on=.015):
    oodf = oodf.with_columns([
            pl.col('duration').shift(1).alias('prev_duration'),
            pl.col('duration').shift(-1).alias('next_duration'),])
    oodf = oodf.with_columns(pl.when(
        (pl.col('prev_duration')<=min_on)&(pl.col('next_duration')<=min_on)&(pl.col('status')=='off'))
                    .then(pl.lit('on'))
                    .otherwise(pl.col('status'))
                    .alias('status'))
    return oodf



def mask_full_oodf(oodf, start=None, end=None, resolution=1e-3):
    offs = oodf.offs()
    channel_masks = {}
    offs = offs.to_pandas()
    if start==None and end==None:
        start = oodf['start_datetime'].to_pandas().min()
        end = oodf['end_datetime'].to_pandas().max()
    mask_dur = (end-start).total_seconds()
    tds = np.arange(0, mask_dur, resolution)
    tds = pd.to_timedelta(tds, unit='s')
    times_to_mask = tds+start
    for probe in offs['probe'].unique():
        channel_masks[probe] = {}
        for channel in offs.prb(probe)['channel'].unique():
            print(f'Masking {probe} {channel}')
            chdf = offs.prb(probe).chnl(channel)
            chdf = chdf.sort_values('start_datetime')
            off_starts = chdf['start_datetime'].values
            chdf = chdf.sort_values('end_datetime')
            off_ends = chdf['end_datetime'].values
            start_indices = np.searchsorted(off_starts, times_to_mask, side='right') - 1
            end_indices = np.searchsorted(off_ends, times_to_mask, side='left')
            mask = (start_indices==end_indices)
            channel_masks[probe][channel] = mask
    return channel_masks, times_to_mask

def _gen_oodf_mask(oodf):
    mask, mask_times = mask_full_oodf(oodf)
    arrs = {}
    for probe in mask.keys():
        arr = np.array(list(mask[probe].values()))
        arrs[probe] = da.from_array(arr)
    full_mask = xr.Dataset({})
    for probe in arrs.keys():
        full_mask[probe] = xr.DataArray(arrs[probe], dims=['channel', 'datetime'], coords={'channel': range(1, 17), 'datetime': mask_times})
    return full_mask

def gen_oodf_mask(subject, exp, save=False):
    mua = acr.mua.load_concat_peaks_df(subject, exp)
    oodf = compute_full_oodf_by_channel(mua, off_min_dur=.05, synth_dur=2.50)
    oodf = true_strictify_oodf(oodf, min_on=.020)
    mask, mask_times = mask_full_oodf(oodf)
    arrs = {}
    for probe in mask.keys():
        arr = np.array(list(mask[probe].values()))
        arrs[probe] = da.from_array(arr)
    full_mask = xr.Dataset({})
    for probe in arrs.keys():
        full_mask[probe] = xr.DataArray(arrs[probe], dims=['channel', 'datetime'], coords={'channel': range(1, 17), 'datetime': mask_times})
    if save == True:
        _save_oodf_mask(full_mask, subject, exp)
    else: 
        return full_mask
    
def _save_oodf_mask(mask, subject, exp, overwrite=False):
    fold = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/oodf_masks'
    if os.path.exists(fold) == False:
        os.mkdir(fold)
    path = os.path.join(fold, f'{exp}_oodf_mask.zarr')
    if os.path.exists(path) and overwrite == True:
        os.system(f'rm -rf {path}')
    mask.to_zarr(path)
    
def load_oodf_mask(subject, exp):
    fold = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/oodf_masks'
    path = os.path.join(fold, f'{exp}_oodf_mask.zarr')
    return xr.open_zarr(path)


def make_oofreqs_relative(freqs, by='channel', ref_to='early_bl'):
    if type(freqs) != pl.DataFrame:
        freqs = pl.DataFrame(freqs)
    assert 'condition' in freqs.columns, 'freqs must be labeled with condition'
    assert ref_to in freqs['condition'].unique(), 'ref_to condition not found in freqs'

    refdf = freqs.filter(pl.col('condition')==ref_to)
    
    if by == 'channel':
        ref_avgs = refdf.group_by(['probe', 'channel', 'status']).agg((pl.sum('pure_count')/pl.sum('bout_duration')).alias('freq_total'))
        freqs = freqs.with_columns(rel_freq=pl.lit(101))
        for probe in freqs['probe'].unique():
            for channel in freqs['channel'].unique():
                for status in freqs['status'].unique():
                    bl_val = ref_avgs.filter(pl.col('probe')==probe).filter(pl.col('channel')==channel).filter(pl.col('status')==status)['freq_total'].to_numpy()[0]
                    freqs = freqs.with_columns(pl.when(((pl.col('probe')==probe) & (pl.col('channel')==channel) & (pl.col('status')==status)))
                                            .then(pl.col('freq')/bl_val)
                                            .otherwise(pl.col('rel_freq'))
                                            .alias('rel_freq'))
        return freqs

    elif by == 'probe':
        ref_avgs = refdf.group_by(['probe', 'status']).agg((pl.sum('pure_count')/pl.sum('bout_duration')).alias('freq_total'))
        freqs = freqs.with_columns(rel_freq=pl.lit(101))
        for probe in freqs['probe'].unique():
            for status in freqs['status'].unique():
                bl_val = ref_avgs.filter(pl.col('probe')==probe).filter(pl.col('status')==status)['freq_total'].to_numpy()[0]
                freqs = freqs.with_columns(pl.when(((pl.col('probe')==probe) & (pl.col('status')==status)))
                                            .then(pl.col('freq')/bl_val)
                                            .otherwise(pl.col('rel_freq'))
                                            .alias('rel_freq'))
        return freqs


def make_oofreqs_relative_bouts(freqs, by='channel', ref_to='early_bl', avg_method='mean'):
    
    if type(freqs) != pl.DataFrame:
        freqs = pl.DataFrame(freqs)
    assert 'condition' in freqs.columns, 'freqs must be labeled with condition'
    assert ref_to in freqs['condition'].unique(), 'ref_to condition not found in freqs'
    
    print(ref_to)
    refdf = freqs.filter(pl.col('condition')==ref_to)
    
    if by == 'channel':
        ref_avgs = refdf.groupby(['probe', 'channel', 'status']).median() if avg_method=='median' else refdf.groupby(['probe', 'channel', 'status']).mean()
        freqs = freqs.with_columns(rel_freq=pl.lit(101))
        for probe in freqs['probe'].unique():
            for channel in freqs['channel'].unique():
                for status in freqs['status'].unique():
                    bl_val = ref_avgs.filter(pl.col('probe')==probe).filter(pl.col('channel')==channel).filter(pl.col('status')==status)['freq'].to_numpy()[0]
                    freqs = freqs.with_columns(pl.when(((pl.col('probe')==probe) & (pl.col('channel')==channel) & (pl.col('status')==status)))
                                            .then(pl.col('freq')/bl_val)
                                            .otherwise(pl.col('rel_freq'))
                                            .alias('rel_freq'))
        return freqs

    elif by == 'probe':
        ref_avgs = refdf.groupby(['probe', 'status']).median() if avg_method=='median' else refdf.groupby(['probe', 'status']).mean()
        freqs = freqs.with_columns(rel_freq=pl.lit(101))
        for probe in freqs['probe'].unique():
            for status in freqs['status'].unique():
                bl_val = ref_avgs.filter(pl.col('probe')==probe).filter(pl.col('status')==status)['freq'].to_numpy()[0]
                print(probe, status, bl_val)
                freqs = freqs.with_columns(pl.when(((pl.col('probe')==probe) & (pl.col('status')==status)))
                                            .then(pl.col('freq')/bl_val)
                                            .otherwise(pl.col('rel_freq'))
                                            .alias('rel_freq'))
        return freqs


def find_consecutive_runs(arr, threshold=10, min_length=50):
    """
    Find starting indices and lengths of consecutive runs in a 1D NumPy array
    where the values are greater than a specified threshold.

    Parameters:
    - arr (np.ndarray): 1D array of integers.
    - threshold (int, optional): The threshold value. Defaults to 10.
    - min_length (int, optional): Minimum length of the run. Defaults to 50.

    Returns:
    - start_indices (List[int]): List of starting indices of qualifying runs.
    - run_lengths (List[int]): List of lengths corresponding to each run.
    """
    # Ensure input is a NumPy array
    arr = np.asarray(arr)
    
    # Create a boolean array where True indicates the value is greater than the threshold
    condition = arr >= threshold
    
    # Find the edges where the condition changes
    # Pad with False at both ends to capture runs that start or end at the array boundaries
    padded = np.pad(condition, (1, 1), 'constant', constant_values=(False, False))
    diff = np.diff(padded.astype(int))
    
    # Start indices are where the difference is 1
    run_starts = np.where(diff == 1)[0]
    
    # End indices are where the difference is -1
    run_ends = np.where(diff == -1)[0]
    
    # Calculate the lengths of each run
    run_lengths = run_ends - run_starts
    
    # Filter runs that meet the minimum length requirement
    valid_runs = run_lengths >= min_length
    start_indices = run_starts[valid_runs].tolist()
    run_lengths = run_lengths[valid_runs].tolist()
    
    return start_indices, run_lengths


# =====================================================================================================================================
# =====================================         SPIKE-BIN-BASED OFF PERIOD DETECTION        ===========================================
# =====================================================================================================================================

def bin_spikes(
    spikes_per_chan,          # list[ndarray] of spike times, dtype=float64 seconds OR datetime64[ns]
    start_time,               # datetime64[ns]
    end_time,                 # datetime64[ns]
    mmap_path: Path | None = None,
    dtype=np.uint8           # pick uint32 if you expect >65535 spikes in any 1 ms bin
):
    """
    Parameters
    ----------
    spikes_per_chan : list of 1‑D numpy arrays (N_spikes, )
        One array per channel, sorted ascending.
    start_time, end_time : numpy.datetime64[ns]
        Recording limits (inclusive start, exclusive end).
    mmap_path : Path | None
        If given, a .npy file is memory‑mapped on disk instead of allocating in RAM.
    dtype : numpy dtype
        Integer storage for spike counts.
    Returns
    -------
    xr.DataArray
        dims=('channel', 'time'), coordinate 'time' is datetime64[ns].
    """
    start_ns = start_time.astype('datetime64[ns]').astype(np.int64)
    end_ns   = end_time.astype('datetime64[ns]').astype(np.int64)
    ns_per_bin = 1_000_000  # 1 ms

    n_bins = (end_ns - start_ns + ns_per_bin - 1) // ns_per_bin
    n_ch   = len(spikes_per_chan)

    # allocate output (RAM or on‑disk memmap)
    if mmap_path is None:
        binned = np.zeros((n_ch, n_bins), dtype=dtype)
    else:
        binned = np.lib.format.open_memmap(
            mmap_path, mode='w+', dtype=dtype, shape=(n_ch, n_bins)
        )

    @nb.njit(parallel=True, fastmath=True, error_model="numpy")
    def _fill(spikes, out, start_ns, ns_per_bin):
        for c in nb.prange(len(spikes)):
            # convert to ns int64 once to avoid python datetime arithmetic
            idx = ((spikes[c].view(np.int64) - start_ns) // ns_per_bin).astype(np.int64)
            valid = (idx >= 0) & (idx < out.shape[1])
            if valid.any():
                bincount = np.bincount(idx[valid], minlength=out.shape[1])
                out[c, :] = bincount.astype(out.dtype)

    # ensure all spike arrays are datetime64[ns] for unified view casting
    spikes_ns = [
        s if np.issubdtype(s.dtype, np.datetime64)
        else (start_time + (s * 1_000_000_000).astype('timedelta64[ns]'))
        for s in spikes_per_chan
    ]

    _fill(spikes_ns, binned, start_ns, ns_per_bin)

    # build xarray view without copying
    time_index = start_time + np.arange(n_bins) * np.timedelta64(1, 'ms')

    return xr.DataArray(
        binned,
        dims=('channel', 'datetime'),
        coords={'datetime': time_index}
    )
    
def calculate_off_df_on_spike_bins(bin_counts, subject, exp, probe, chan_threshold=15, n_consecutive_bins=50):
    """
    Calculate the OFF DataFrame from binned spike counts.

    Parameters
    ----------
    bin_counts : np.ndarray
        2D array of binned spike counts with shape (n_channels, n_bins).
    chan_threshold : int
        Minimum number of channels that need to be off for a bin to be considered "off".
    n_consecutive_bins : int
        Minimum number of consecutive bins that need to be off for a run to be considered "off".

    Returns
    -------
    OODF : pl.DataFrame
        DataFrame containing the detected OFF periods.
    """
    bc = bin_counts.data
    om = bc == 0
    off_counts = np.sum(om, axis=0)
    
    # Find the starting indices and lengths of runs where the number of channels off is greater than or equal to chan_threshold
    start_indices, lengths = find_consecutive_runs(off_counts, threshold=chan_threshold, min_length=n_consecutive_bins)
    start_indices = np.array(start_indices[:-1])
    lengths = np.array(lengths[:-1])
    dt_vals = bin_counts.datetime.values
    
    start_times = dt_vals[start_indices]

    end_times = dt_vals[start_indices+lengths]

    off_durations = (end_times - start_times)/np.timedelta64(1, 's')

    off_df = pl.DataFrame({
        'subject': subject,
        'exp': exp,
        'probe': probe,
        'start_datetime': start_times,
        'end_datetime': end_times,
        'duration': off_durations,
        'status': 'off',
        'chan_threshold': chan_threshold
    })
    return off_df