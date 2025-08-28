import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

def compute_intra_off_activity(counts, start_times, end_times):
    """
    counts:        an xarray.DataArray with dims ('datetime', 'channel')
    start_times:   array‐like of datetime64 or pandas.Timestamp
    end_times:     array‐like of datetime64 or pandas.Timestamp
    """
    # 2) pull out the raw numpy arrays
    times = counts.coords['datetime'].values         # shape (T,)
    data  = counts.values.T                            # shape (T, C)   
    
    # 3) find for each event the start‐ and end‐bin indices
    #    - start at first bin >= end_time
    #    - end   at last  bin <= right_edge
    idx_start = np.searchsorted(times, start_times.values, side='left')
    idx_end   = np.searchsorted(times, end_times.values, side='right') - 1
    C = data.shape[1]
    cs = np.concatenate([
        np.zeros((1, C), dtype=data.dtype),
        data.cumsum(axis=0)
    ], axis=0)  # shape (T+1, C)
    
    
    # 5) compute per-channel sums for each window
    sums = cs[idx_end + 1, :] - cs[idx_start, :]   # shape (N_windows, C)

    # 6) summary stats
    act_mag  = sums.sum(axis=1)                   # total spikes per window
    return act_mag

def add_intra_off_activity_to_oodf(oodf, counts_dict):
    probe_oodfs = []
    for probe in ['NNXr', 'NNXo']:
        counts = counts_dict[probe]
        probe_oodf = oodf.prb(probe)
        start_times = probe_oodf['start_datetime'].to_pandas()
        end_times = probe_oodf['end_datetime'].to_pandas()
        act_mag = compute_intra_off_activity(counts, start_times, end_times)
        probe_oodf = probe_oodf.with_columns(intra_off_sum=pl.lit(act_mag))
        probe_oodfs.append(probe_oodf)
    df =pl.concat(probe_oodfs)
    durs = df['duration']
    intras = df['intra_off_sum']
    normed_intras = intras / durs
    df = df.with_columns(normed_intras=pl.lit(normed_intras/1000))
    return df
    



def compute_termination_activity(counts, end_times, post_term_duration='10ms'):
    """
    counts:        an xarray.DataArray with dims ('datetime', 'channel')
    end_times:     array‐like of datetime64 or pandas.Timestamp
    post_term_duration: string parseable by pd.Timedelta, e.g. '10ms'
    
    Returns the same three lists (pct_act, act_mag, act_mean).
    """

    # 1) compute right‐edge times
    end_times = pd.to_datetime(end_times)
    dr = pd.Timedelta(post_term_duration)
    right_edges = end_times + dr

    # 2) pull out the raw numpy arrays
    times = counts.coords['datetime'].values         # shape (T,)
    data  = counts.values.T                            # shape (T, C)

    # 3) find for each event the start‐ and end‐bin indices
    #    - start at first bin >= end_time
    #    - end   at last  bin <= right_edge
    idx_start = np.searchsorted(times, end_times.values, side='left')
    idx_end   = np.searchsorted(times, right_edges.values, side='right') - 1

    # 4) build a zero-padded cumsum so that
    #    window sum from i→j inclusive is cs[j+1] - cs[i]
    C = data.shape[1]
    cs = np.concatenate([
        np.zeros((1, C), dtype=data.dtype),
        data.cumsum(axis=0)
    ], axis=0)  # shape (T+1, C)

    # 5) compute per-channel sums for each window
    sums = cs[idx_end + 1, :] - cs[idx_start, :]   # shape (N_windows, C)

    # 6) summary stats
    act_mag  = sums.sum(axis=1)                   # total spikes per window
    act_mean = sums.mean(axis=1)                  # mean per-channel
    pct_act  = (sums > 0).sum(axis=1) / C         # fraction of channels with any spike

    # 7) return as lists to match the original signature
    return pct_act, act_mag, act_mean


def compute_intitiation_activity(counts, start_times, pre_init_duration='10ms'):
    """
    counts:        an xarray.DataArray with dims ('datetime', 'channel')
    start_times:     array‐like of datetime64 or pandas.Timestamp
    pre_init_duration: string parseable by pd.Timedelta, e.g. '10ms'
    
    Returns the same three lists (pct_act, act_mag, act_mean).
    """

    # 1) compute right‐edge times
    start_times = pd.to_datetime(start_times)
    dr = pd.Timedelta(pre_init_duration)
    left_edges = start_times - dr

    # 2) pull out the raw numpy arrays
    times = counts.coords['datetime'].values         # shape (T,)
    data  = counts.values.T                            # shape (T, C)

    # 3) find for each event the start‐ and end‐bin indices
    #    - start at first bin >= end_time
    #    - end   at last  bin <= right_edge
    idx_start = np.searchsorted(times, left_edges.values, side='left')
    idx_end   = np.searchsorted(times, start_times.values, side='right') - 1

    # 4) build a zero-padded cumsum so that
    #    window sum from i→j inclusive is cs[j+1] - cs[i]
    C = data.shape[1]
    cs = np.concatenate([
        np.zeros((1, C), dtype=data.dtype),
        data.cumsum(axis=0)
    ], axis=0)  # shape (T+1, C)

    # 5) compute per-channel sums for each window
    sums = cs[idx_end + 1, :] - cs[idx_start, :]   # shape (N_windows, C)

    # 6) summary stats
    act_mag  = sums.sum(axis=1)                   # total spikes per window
    act_mean = sums.mean(axis=1)                  # mean per-channel
    pct_act  = (sums > 0).sum(axis=1) / C         # fraction of channels with any spike

    # 7) return as lists to match the original signature
    return pct_act, act_mag, act_mean

def add_init_term_activity_to_oodf(oodf, counts_dict, surrounding_duration='10ms', method='termination', chans=(1, 16)):
    probe_oodfs = []
    for probe in ['NNXr', 'NNXo']:
        probe_oodf = oodf.prb(probe)
        
        # termination activity
        end_times = probe_oodf['end_datetime'].to_pandas()
        pct, mag, mns = compute_termination_activity(counts_dict[probe].sel(channel=slice(chans[0], chans[1])), end_times, surrounding_duration)
        probe_oodf = probe_oodf.with_columns(term_pct_act=pl.lit(pct))
        probe_oodf = probe_oodf.with_columns(term_act_mag=pl.lit(mag))
        probe_oodf = probe_oodf.with_columns(term_act_mean=pl.lit(mns))
        
        # initiation activity
        start_times = probe_oodf['start_datetime'].to_pandas()
        pct, mag, mns = compute_intitiation_activity(counts_dict[probe].sel(channel=slice(chans[0], chans[1])), start_times, surrounding_duration)
        probe_oodf = probe_oodf.with_columns(init_pct_act=pl.lit(pct))
        probe_oodf = probe_oodf.with_columns(init_act_mag=pl.lit(mag))
        probe_oodf = probe_oodf.with_columns(init_act_mean=pl.lit(mns))
        probe_oodfs.append(probe_oodf)
    return pl.concat(probe_oodfs)