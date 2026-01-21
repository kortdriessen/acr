import polars as pl
import numpy as np
import kdephys as kde
import acr
import pandas as pd
import xarray as xr

# =====================================================================================================================================
# =====================================         Improved Utils for On/Off dataframes       ===========================================
# =====================================================================================================================================
def label_oodf_with_states(oodf, h):
    start_times = oodf['start_datetime'].to_numpy()
    end_times = oodf['end_datetime'].to_numpy()
    start_states = kde.hypno.hypno.get_states_fast(h, start_times)
    end_states = kde.hypno.hypno.get_states_fast(h, end_times)
    full_states = start_states.copy()
    # Mark states as 'UNDETERMINED' where start and end states don't match
    full_states = np.where(start_states == end_states, full_states, 'UNDETERMINED')
    return oodf.with_columns(state=pl.lit(full_states))




def label_oodf_full_bl(oodf: pl.DataFrame, state: str = 'NREM') -> pl.DataFrame:
    """Label a 12-hour baseline period in the off periods dataframe.

    Parameters
    ----------
    oodf : pl.DataFrame
        The off periods dataframe to label with baseline information.
    state : str
        The state to select in the full baseline period. Set to 'None' to not select a state.

    Returns
    -------
    pl.DataFrame
        The off periods dataframe with a 'full_bl' column added, marking the 12-hour baseline period.
    """
    
    
    
    oodf = oodf.with_columns(full_bl=pl.lit('False'))
    bl_day = pd.Timestamp(oodf['start_datetime'].min().date())
    bl_9am = bl_day + pd.Timedelta('9h')
    bl_9pm = bl_9am + pd.Timedelta('12h')
    oodf = oodf.with_columns(
        full_bl=pl.when((pl.col('start_datetime') >= bl_9am) & (pl.col('start_datetime') <= bl_9pm))
        .then(pl.lit('True'))
        .otherwise(pl.lit('False'))
    )
    
    if state != 'None':
        oodf = oodf.with_columns(
            full_bl=pl.when(pl.col('state') != state)
            .then(pl.lit('False'))
            .otherwise(pl.col('full_bl'))
            .alias('full_bl'))
    
    return oodf

def label_oodf_hyp_dict_conditions(oodf: pl.DataFrame, hd: dict, label_col='condition') -> pl.DataFrame:
    """label oodf with all hypnogram conditions in the hypnogram dictionary

    Parameters
    ----------
    oodf : pl.DataFrame
        _description_
    hd : dict
        _description_

    Returns
    -------
    pl.DataFrame
        _description_
    """
    oodf = oodf.with_columns(pl.lit('None').alias(label_col))
    for key in hd.keys():
        for bout in hd[key].itertuples():
            oodf = oodf.with_columns(
                pl.when((pl.col('start_datetime') >= bout.start_time) & (pl.col('end_datetime') <= bout.end_time))
                .then(pl.lit(key))
                .otherwise(pl.col(label_col))
                .alias(label_col)
            )
    return oodf

def relativize_oodf(oodf, ref_to_col: str = 'full_bl', ref_to_val: str = 'True', avg_method: str = 'mean', col_to_relativize: str = 'duration'):
    assert ref_to_col in oodf.columns, 'oodf must be labeled with ref_to_col'
    assert ref_to_val in oodf[ref_to_col].unique(), 'ref_to_val condition not found in oodf'
    
    # Drop existing bl and rel columns if they exist
    if f'{col_to_relativize}_bl' in oodf.columns:
        oodf.drop_in_place(f'{col_to_relativize}_bl')
    if f'{col_to_relativize}_rel' in oodf.columns:
        oodf.drop_in_place(f'{col_to_relativize}_rel')
    
    ref_df = oodf.filter(pl.col(ref_to_col)==ref_to_val)
    
    bl_vals = ref_df.group_by(['probe', 'status']).mean() if avg_method == 'mean' else ref_df.group_by(['probe', 'status']).median()
    bl_vals = bl_vals.select(['probe', 'status', col_to_relativize])

    oodf = oodf.join(bl_vals, on=['probe', 'status'], how='left', suffix='_bl')
    rel_values = oodf[col_to_relativize]/oodf[f'{col_to_relativize}_bl']
    oodf = oodf.with_columns(pl.lit(rel_values).alias(f'{col_to_relativize}_rel'))
    return oodf


def calc_off_freq(oodf, window='5s', probes=['NNXr', 'NNXo']):
    """calculate the off frequency for each probe in the oodf. The oodf must be a single subject, with only OFFs or only ONs.

    Parameters
    ----------
    oodf : _type_
        _description_
    window : str, optional
        _description_, by default '5s'
    probes : list, optional
        _description_, by default ['NNXr', 'NNXo']

    Returns
    -------
    pl.DataFrame
        a dataframe with the off frequency for each probe
    """
    
    ofpd = oodf.to_pandas()
    subject = oodf['subject'][0]
    exp = oodf['exp'][0] if 'exp' in oodf.columns else oodf['recording'][0]
    rates = []
    status = oodf['status'].to_numpy()[0]
    assert len(oodf['status'].unique()) == 1, 'oodf must have only OFFs or only ONs'
    assert len(oodf['subject'].unique()) == 1, 'oodf must have only one subject'
    for probe in probes:
        df = ofpd.sort_values('start_datetime').set_index('start_datetime').prb(probe)
        freq = df.resample(window).size()            # Series indexed every 5 s
        rate = freq.div(5).rename('off_freq_per_s')
        rate = rate.to_frame()
        rate.reset_index(inplace=True)
        rate['probe'] = probe
        rate['subject'] = subject
        rate['exp'] = exp
        rate['status'] = status
        sdt = rate['start_datetime']
        edt = sdt+pd.Timedelta(seconds=5)
        rate['end_datetime'] = edt
        rate = pl.DataFrame(rate)
        #rate = rate.rename({'start_datetime': 'datetime'})
        rates.append(rate)

    return pl.concat(rates)

def enhance_oodf(oodf, full_hyp, hyp_dict, ref_to_col='full_bl', ref_to_val='True', avg_method='mean', col_to_relativize='duration'):
    """Adds states, full bl, and hyp_dict conditions to oodf, and relativizes one column to baseline.

    Parameters
    ----------
    oodf : _type_
        _description_
    full_hyp : _type_
        _description_
    hyp_dict : _type_
        _description_
    ref_to_col : str, optional
        _description_, by default 'full_bl'
    ref_to_val : str, optional
        _description_, by default 'True'
    avg_method : str, optional
        _description_, by default 'mean'
    col_to_relativize : str, optional
        _description_, by default 'duration'

    Returns
    -------
    _type_
        _description_
    """
    oodf = acr.oo_utils.label_oodf_with_states(oodf, full_hyp)
    oodf = acr.oo_utils.label_oodf_full_bl(oodf)
    oodf = acr.oo_utils.label_oodf_hyp_dict_conditions(oodf, hyp_dict)
    oodf = acr.oo_utils.relativize_oodf(oodf, ref_to_col=ref_to_col, ref_to_val=ref_to_val, avg_method=avg_method, col_to_relativize=col_to_relativize)
    return oodf

def sum_spikes_in_windows(
    spike_counts: xr.DataArray,
    on_starts: np.ndarray,
    on_ends:   np.ndarray,
    collapse_channels: bool = True,
):
    """
    This function is for use with the 1ms BINNED spike counts! For using with raw spike times (mua data), see the 
    Parameters
    ----------
    spike_counts
        xarray.DataArray with dims ('channel', 'time'); each value = spikes in one 1 ms bin.
    on_starts, on_ends
        1-D arrays (same length) of np.datetime64[ns] marking the beginning and
        end **inclusive** of each window.
    collapse_channels
        True  ➜ return shape (n_windows,)  totals across *all* channels  
        False ➜ return shape (n_channels, n_windows)  per-channel totals

    Returns
    -------
    np.ndarray
        Vector (or matrix) of spike sums in each window.
    """

    # -------- fast path: pull raw data & pre-compute cumulative sums --------
    counts = spike_counts.data            # (C, T)
    times  = spike_counts["datetime"].values  # (T,)  np.datetime64[ns]

    C, T = counts.shape

    #   pad with a leading 0 so we can use  cs[:, e+1] - cs[:, s]
    cs = np.empty((C, T + 1), dtype=np.int64)
    cs[:, 0] = 0
    np.cumsum(counts, axis=1, dtype=np.int64, out=cs[:, 1:])

    # -------- convert wall-clock boundaries ➜ integer bin indices --------
    t0      = times[0].astype("datetime64[ns]")
    idx0    = ((on_starts.astype("datetime64[ns]") - t0)
               // np.timedelta64(1, "ms")).astype(np.int64)
    idx1    = ((on_ends.astype("datetime64[ns]")   - t0)
               // np.timedelta64(1, "ms")).astype(np.int64)

    #   clip in case a window strays outside the recording
    np.clip(idx0, 0, T,  out=idx0)
    np.clip(idx1, -1, T-1, out=idx1)

    # -------- vectorised gather & difference --------
    #   cs[:, idx1+1] - cs[:, idx0]   →  shape (C, n_windows)
    sums = cs.take(idx1 + 1, axis=1) - cs.take(idx0, axis=1)

    return sums.sum(axis=0) if collapse_channels else sums

def add_rate_to_oodf(oodf, spks, use_ends=True):
    start_times = oodf['start_datetime'].to_numpy()
    end_times = oodf['end_datetime'].to_numpy()
    if use_ends != True:
        end_times = start_times + pd.to_timedelta(use_ends, unit='s')
    summed_spikes = sum_spikes_in_windows(spks, start_times, end_times)
    oodf = oodf.with_columns(spikes_count = pl.lit(summed_spikes))
    spk_rate = oodf['spikes_count'].to_numpy()/oodf['duration'].to_numpy()
    oodf = oodf.with_columns(spk_rate = pl.lit(spk_rate/1000))
    return oodf
    
#def add_global_spk_rate(on, spks, use_ends=True):
#    on_starts = on['start_datetime'].to_numpy()
#    on_ends = on['end_datetime'].to_numpy()
#    if use_ends != True:
#        on_ends = on_starts + pd.to_timedelta(use_ends, unit='s')
#    summed_spikes = sum_spikes_in_windows(spks, on_starts, on_ends)
#    on = on.with_columns(spikes_count = pl.lit(summed_spikes))
#    spk_rate = on['spikes_count'].to_numpy()/on['duration'].to_numpy()
#    on = on.with_columns(spk_rate = pl.lit(spk_rate))
#    return on


def count_mua_spikes_by_window(
    spike_times: np.ndarray,
    start_times: np.ndarray,
    end_times:   np.ndarray,
    *,
    left_inclusive:  bool = True,
    right_inclusive: bool = True,
) -> np.ndarray:
    """
    Parameters
    ----------
    spike_times
        1-D sorted array of np.datetime64 (or anything monotonic) – one entry per spike.
    start_times, end_times
        1-D arrays (same length) of np.datetime64 window boundaries.
    left_inclusive, right_inclusive
        Whether to count spikes exactly at `start_times` / `end_times`.

    Returns
    -------
    np.ndarray
        1-D integer array `counts[i]` = number of spikes in the i-th window.
    """

    spike_times  = np.asarray(spike_times)
    start_times  = np.asarray(start_times)
    end_times    = np.asarray(end_times)

    if start_times.shape != end_times.shape:
        raise ValueError("start_times and end_times must have the same shape")

    # ---------------------------------------------------------------------
    # Map each boundary to its *position* in the sorted spike array
    # ---------------------------------------------------------------------
    #   side='left': first index >= boundary
    #   side='right': first index  > boundary
    #   tweak sides to get the desired inclusivity
    left_side  = 'left'  if left_inclusive  else 'right'
    right_side = 'right' if right_inclusive else 'left'

    idx_left  = np.searchsorted(spike_times, start_times, side=left_side)
    idx_right = np.searchsorted(spike_times, end_times,   side=right_side)

    # counts = (# spikes ≤ right boundary) − (# spikes  < left boundary)
    return idx_right - idx_left


def add_mua_spike_info_to_oodf(oodf, mua_df, pre_window='50ms', post_window='50ms'):
    prb_oodfs = []
    for probe in ['NNXr', 'NNXo']:
        prb_df = mua_df.prb(probe)
        prb_oodf = oodf.prb(probe)
        spike_times = prb_df['datetime'].to_numpy()
        off_starts = prb_oodf['start_datetime'].to_numpy()
        off_ends = prb_oodf['end_datetime'].to_numpy()
        intra_off_counts = count_mua_spikes_by_window(spike_times, off_starts, off_ends, left_inclusive=True, right_inclusive=True)
        prb_oodf = prb_oodf.with_columns(intra_off_mua_count = pl.lit(intra_off_counts))
        
        pre_off_starts = off_starts - pd.Timedelta(pre_window)
        pre_off_counts = count_mua_spikes_by_window(spike_times, pre_off_starts, off_starts, left_inclusive=True, right_inclusive=False)
        prb_oodf = prb_oodf.with_columns(pre_off_mua_count = pl.lit(pre_off_counts))
        
        post_off_ends = off_ends + pd.Timedelta(post_window)
        post_off_counts = count_mua_spikes_by_window(spike_times, off_ends, post_off_ends, left_inclusive=False, right_inclusive=True)
        prb_oodf = prb_oodf.with_columns(post_off_mua_count = pl.lit(post_off_counts))
        
        prb_oodfs.append(prb_oodf)
        
    return pl.concat(prb_oodfs)









def add_decile_column(
    numbers: pl.Series | np.ndarray | list[float],
    df: pl.DataFrame,
    target_col: str,
    decile_col: str = "decile",
) -> pl.DataFrame:
    """
    Compute deciles from *numbers* and tag each row of *df* with the
    decile (1‒10) into which its *target_col* value falls.

    Parameters
    ----------
    numbers : polars.Series | array-like
        Data used to derive the decile thresholds.  
        If you want to use `df[target_col]` itself, just pass that series.
    df : polars.DataFrame
        The DataFrame to receive the new column.
    target_col : str
        Name of the numeric column whose values will be binned.
    decile_col : str, default "decile"
        Name of the new integer column (1–10).

    Returns
    -------
    pl.DataFrame
        A copy of *df* with an added *decile_col*.
    """
    # ── 1. Convert to a clean NumPy array (drop nulls for break-point calc) ──
    if isinstance(numbers, pl.Series):
        arr = numbers.drop_nulls().to_numpy()
    else:
        arr = np.asarray(numbers, dtype=float)

    if arr.size == 0:
        raise ValueError("`numbers` must contain at least one non-null value.")

    # ── 2. Compute decile edges: 0 %, 10 %, … 100 % ──
    edges = np.quantile(arr, q=np.linspace(0.0, 1.0, 11))

    # Ensure edges are strictly increasing (duplicate quantiles ⇒ zero-width bins)
    eps = np.finfo(arr.dtype).eps
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps

    # ── 3. Build a vectorised Polars expression to assign 1‒10 ──
    expr = pl.when(pl.col(target_col) < edges[1]).then(1)
    for k in range(1, 9):  # middle deciles
        expr = expr.when(
            (pl.col(target_col) >= edges[k]) & (pl.col(target_col) < edges[k + 1])
        ).then(k + 1)
    expr = expr.otherwise(10).alias(decile_col)  # uppermost 10 %

    # ── 4. Attach the new column ──
    return df.with_columns(expr)

def filter_binned_oodf(oodf, decile_filter=8, duration_filter=0.06, state_filter='NREM'):
    probe_dfs = []
    for probe in oodf['probe'].unique():
        probe_df = oodf.filter(pl.col('probe')==probe)
        ref_vals = probe_df.filter(pl.col('state')=='NREM')['spk_rate']
        probe_df = acr.oo_utils.add_decile_column(ref_vals, probe_df, 'spk_rate', 'decile')
        probe_df = probe_df.filter(~((pl.col('decile')>decile_filter)& (pl.col('duration')<duration_filter)&(pl.col('state')==state_filter)))
        probe_dfs.append(probe_df)
    return pl.concat(probe_dfs)

def add_sw_peak_to_oodf(oodf, fp, ch=15, buffer_dur=0.07, single_chan=None):
    prb_oodfs = []
    for probe in ['NNXr', 'NNXo']:
        prb_oodf = oodf.filter(pl.col('probe') == probe)
        data = fp.prb(probe)
        off_starts = prb_oodf['start_datetime'].to_pandas()
        off_durs = prb_oodf['duration'].to_pandas()
        
        peak_data = acr.sync.select_data_for_peaks(off_starts, off_durs, data, buffer=buffer_dur)
        
        peaks = np.mean(np.max(peak_data, axis=1), axis=0)
        
        if single_chan:
            peaks = np.max(peak_data[ch-1, :, :], axis=0)
        
        peaks = pl.Series(peaks, dtype=pl.Float64)
        prb_oodf = prb_oodf.with_columns(sw_peak = pl.lit(peaks))
        prb_oodfs.append(prb_oodf)
    
    return pl.concat(prb_oodfs)