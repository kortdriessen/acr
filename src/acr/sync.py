import pandas as pd
import numpy as np
import polars as pl
import acr
import pickle
from numba import njit, prange
from typing import List, Tuple


def select_data_around_times(times, data, buffer=0.05):
    """returns a list of arrays that are [buffer*fs*2 x n_channels], each element centered around one of the times

    Parameters
    ----------
    times : _type_
        _description_
    data : xr.Array, dimensions are 'datetime' and 'channel'
        _description_
    buffer : float, optional
        time in seconds on either side of each timepoint in times
    """
    total_times = data.datetime.values
    raw_dat = data.values.T  # so shape is nchannels X ntimepoints
    n_samps = int(round(buffer * data.fs, ndigits=0))
    timepoint_indices = np.searchsorted(total_times, times)
    starts = timepoint_indices - n_samps
    ends = timepoint_indices + n_samps
    # get all of the samples between each start and end index
    arrays = [raw_dat[:, s:e] for s, e in zip(starts, ends)]

    req_shape = (16, n_samps * 2)
    for i, arr in enumerate(arrays):
        if arr.shape != req_shape:
            arrays[i] = np.zeros(req_shape)
    return np.stack(arrays, axis=2)


def compute_peak_df(peak_array):
    peak_mean = peak_array.mean(axis=2)
    off_midpoint = peak_mean.shape[1] / 2
    peak_indices = np.argmax(peak_mean, axis=1)
    peak_values = peak_mean[np.arange(peak_mean.shape[0]), peak_indices]
    rel_peak_indices = peak_indices - off_midpoint
    peak_df = pd.DataFrame(
        {
            "channel": np.arange(peak_mean.shape[0]) + 1,
            "peak": peak_values,
            "peak_position": rel_peak_indices,
        }
    )
    return peak_df


def compute_trough_df(fp_array, fs, window_length=15, polyorder=5):
    middle = int(fp_array.shape[1] / 2)

    fp_mean = fp_array.mean(axis=2)
    fp_mean = fp_mean[:, middle:]
    trough_indices = np.argmin(fp_mean, axis=1)
    trough_values = fp_mean[np.arange(fp_mean.shape[0]), trough_indices]
    trough_df = pd.DataFrame(
        {
            "channel": np.arange(fp_mean.shape[0]) + 1,
            "trough": trough_values,
            "trough_position": trough_indices,
        }
    )
    return trough_df


def select_data_for_peaks(start_times, durations, data, buffer=0.05):
    """returns a list of arrays that are [buffer*fs*2 x n_channels], each element centered around one of the times

    Parameters
    ----------
    times : _type_
        _description_
    data : xr.Array, dimensions are 'datetime' and 'channel'
        _description_
    buffer : float, optional
        time in seconds on either side of each timepoint in times
    """
    total_times = data.datetime.values
    raw_dat = data.values.T  # so shape is nchannels X ntimepoints
    mid_durs = durations / 2
    peak_samps = np.array(
        [int(round(mid_dur * data.fs, ndigits=0)) for mid_dur in mid_durs]
    )
    buffer_samps = int(round(buffer * data.fs, ndigits=0))
    start_time_indices = np.searchsorted(total_times, start_times)

    peak_indices = start_time_indices + peak_samps

    starts = peak_indices - buffer_samps
    ends = peak_indices + buffer_samps

    # get all of the samples between each start and end index
    arrays = [raw_dat[:, s:e] for s, e in zip(starts, ends)]
    req_shape = (16, buffer_samps * 2)
    for i, arr in enumerate(arrays):
        if arr.shape != req_shape:
            arrays[i] = np.zeros(req_shape)
    return np.stack(arrays, axis=2)


def compute_slope_df(
    fp_array, fs, find="min", window_length=15, polyorder=5, search_buffer=10
):
    fp_mean = fp_array.mean(axis=2)
    slopes = []
    for ch in range(fp_mean.shape[0]):
        slopes.append(
            acr.fp.compute_instantaneous_slope_savgol(
                fp_mean[ch, :],
                sampling_rate=fs,
                window_length=window_length,
                polyorder=polyorder,
            )
        )
    slopes_array = np.array(slopes)
    middle = int(fp_array.shape[1] / 2)
    range_start = middle - search_buffer
    range_end = middle + search_buffer
    if find == "min":
        extrema = np.argmin(slopes_array[:, range_start:range_end], axis=1)
        extrema = extrema + range_start
    elif find == "max":
        extrema = np.argmax(slopes_array[:, range_start:range_end], axis=1)
        extrema = extrema + range_start
    else:
        raise ValueError(f"Invalid value for find: {find}")
    slopes_at_extrema = slopes_array[np.arange(slopes_array.shape[0]), extrema]
    extrema_rel_to_middle = extrema - middle
    slope_df = pd.DataFrame(
        {
            "channel": np.arange(slopes_array.shape[0]) + 1,
            "extrema": extrema,
            "extrema_rel_position": extrema_rel_to_middle,
            "slope": slopes_at_extrema,
        }
    )
    return slope_df


def relativize_slope_df_to_condition(
    slope_df, condition, col_to_rel="slope", on=["subject", "probe", "channel"]
):
    if f"{col_to_rel}_rel" in slope_df.columns:
        slope_df = slope_df.drop(columns=[f"{col_to_rel}_rel"])
    if f"{col_to_rel}_bl" in slope_df.columns:
        slope_df = slope_df.drop(columns=[f"{col_to_rel}_bl"])
    bl_df = slope_df.cdn(condition)
    select_from_bl = on + [col_to_rel]
    merged = slope_df.merge(
        bl_df[select_from_bl], on=on, how="inner", suffixes=("", "_bl")
    )
    merged[f"{col_to_rel}_rel"] = merged[col_to_rel] / merged[col_to_rel + "_bl"]
    return merged


def convert_spike_trains_to_seconds(
    spike_trains_datetime: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Given a list of spike-time arrays in np.datetime64[ns], return a list of
    1D float arrays (seconds since epoch), sorted.
    """
    sec_trains = []
    for train in spike_trains_datetime:
        # Convert datetime64[ns] to int64 ns, then to float seconds
        float_seconds = train.astype("datetime64[ns]").astype(np.int64) * 1e-9
        # Ensure sorted order
        sec_trains.append(np.sort(float_seconds))
    return sec_trains


# ---------------------------------------------------------------------
# 1. Coincidence fraction: no double-count, purely two-pointer, O(Na+Nb)
# ---------------------------------------------------------------------
@njit
def _fraction_with_partner(a: np.ndarray, b: np.ndarray, delta: float) -> float:
    """
    Fraction of spikes in 'a' that have ≥1 partner in 'b' within ±delta.
    Arrays must be sorted; counts each spike at most once (no double-count).
    """
    na, nb = a.size, b.size
    if na == 0 or nb == 0:
        return 0.0

    j = 0
    coinc = 0
    for i in range(na):
        t = a[i]

        # advance j until b[j] >= t - delta
        while j < nb and b[j] < t - delta:
            j += 1

        # check the closer of b[j-1] and b[j] (if they exist)
        nearest = delta + 1.0  # larger than delta
        if j < nb:
            d = abs(b[j] - t)
            if d < nearest:
                nearest = d
        if j > 0:
            d = abs(b[j - 1] - t)
            if d < nearest:
                nearest = d

        if nearest <= delta:
            coinc += 1

    return coinc / na


# ---------------------------------------------------------------------
# 2. Proportion of time within ±Δ of ANY spike (O(N))
# ---------------------------------------------------------------------
@njit
def _prop_time_within_delta(
    spikes: np.ndarray, rec_start: float, rec_end: float, delta: float
) -> float:
    """
    Fraction of [rec_start, rec_end] that lies within ±delta of any spike.
    Uses the gap-sum formula; arrays must be sorted.
    """
    n = spikes.size
    if n == 0:
        return 0.0

    total = rec_end - rec_start
    if total <= 0.0:
        return 0.0

    # Core coverage without edge clipping
    covered = 2.0 * delta * n
    if n > 1:
        gaps = np.diff(spikes)
        overlap = 0.0
        for g in gaps:
            tmp = 2.0 * delta - g
            if tmp > 0.0:
                overlap += tmp
        covered -= overlap

    # Clip first and last intervals to recording window
    left_overhang = max(0.0, (rec_start - (spikes[0] - delta)))
    right_overhang = max(0.0, ((spikes[-1] + delta) - rec_end))
    covered -= left_overhang + right_overhang

    # Normalise
    if covered < 0.0:
        covered = 0.0
    elif covered > total:
        covered = total

    return covered / total


# ---------------------------------------------------------------------
# 3. One-pair STTC (unchanged formula, faster helpers)
# ---------------------------------------------------------------------
@njit
def _sttc_pair(
    a: np.ndarray, b: np.ndarray, rec_start: float, rec_end: float, delta: float
) -> float:
    TA = _fraction_with_partner(a, b, delta)
    TB = _fraction_with_partner(b, a, delta)
    PA = _prop_time_within_delta(a, rec_start, rec_end, delta)
    PB = _prop_time_within_delta(b, rec_start, rec_end, delta)

    def part(T, P):
        denom = 1.0 - T * P
        return 0.0 if denom == 0.0 else (T - P) / denom

    return 0.5 * (part(TA, PA) + part(TB, PB))


# ---------------------------------------------------------------------
# 4. Full STTC matrix, parallel over channels
# ---------------------------------------------------------------------
@njit(parallel=True)
def sttc_matrix_fast(
    spike_trains: List[np.ndarray], rec_start: float, rec_end: float, delta: float
) -> np.ndarray:
    """
    Upper-triangular K×K STTC matrix (diag & lower = NaN).
    spike_trains: list of K sorted float seconds arrays.
    """
    K = len(spike_trains)
    M = np.full((K, K), np.nan, dtype=np.float64)

    for i in prange(K):
        ai = spike_trains[i]
        if ai.size < 2:  # optional skip for empty/sparse channels
            continue
        for j in range(i + 1, K):
            bj = spike_trains[j]
            if bj.size < 2:
                continue
            M[i, j] = _sttc_pair(ai, bj, rec_start, rec_end, delta)

    return M


def sttc_matrix_slow(
    spike_trains: List[np.ndarray], rec_start: float, rec_end: float, delta: float
) -> np.ndarray:
    """
    Upper-triangular K×K STTC matrix (diag & lower = NaN).
    spike_trains: list of K sorted float seconds arrays.
    """
    K = len(spike_trains)
    M = np.full((K, K), np.nan, dtype=np.float64)

    for i in range(K):
        ai = spike_trains[i]
        if ai.size < 2:  # optional skip for empty/sparse channels
            continue
        for j in range(i + 1, K):
            bj = spike_trains[j]
            if bj.size < 2:
                continue
            M[i, j] = _sttc_pair(ai, bj, rec_start, rec_end, delta)

    return M


def process_nrem_epochs_fast(
    all_spike_trains_dt: List[List[np.ndarray]],
    nrem_epochs: List[Tuple[np.datetime64, np.datetime64]],
    delta_ms: float = 5.0,
) -> List[np.ndarray]:
    """
    Compute STTC matrices for multiple NREM epochs for a single probe's spike data.
    - all_spike_trains_dt: list of K spike trains, each np.datetime64[ns]
    - nrem_epochs: list of (start, end) np.datetime64[ns] tuples
    Returns a list of K×K STTC matrices (float64) in the same order as nrem_epochs.
    """
    # Pre-convert all channels to float seconds once
    sec_trains_full = acr.sync.convert_spike_trains_to_seconds(all_spike_trains_dt)
    delta = delta_ms * 1e-3  # convert ms to seconds

    sttc_matrices = []
    for t0_dt, t1_dt in nrem_epochs:
        # print(t0_dt)
        # Convert epoch boundaries to seconds
        t0_sec = t0_dt.astype("datetime64[ns]").astype(np.int64) * 1e-9
        t1_sec = t1_dt.astype("datetime64[ns]").astype(np.int64) * 1e-9

        # Extract spikes within the epoch for each channel efficiently
        seg_sec = []
        for train in sec_trains_full:
            # Find indices of the window boundary via searchsorted
            idx0 = np.searchsorted(train, t0_sec, side="left")
            idx1 = np.searchsorted(train, t1_sec, side="right")
            seg = train[idx0:idx1]
            seg_sec.append(seg)

        # JIT-compiled STTC matrix
        M = sttc_matrix_fast(seg_sec, t0_sec, t1_sec, delta)
        sttc_matrices.append(M)

    return sttc_matrices


def process_nrem_epochs_fast_seconds(
    all_spike_trains_sec: List[List[np.ndarray]],
    nrem_epochs: List[Tuple[float, float]],
    delta_ms: float = 5.0,
) -> List[np.ndarray]:
    """
    Compute STTC matrices for multiple NREM epochs for a single probe's spike data.
    - all_spike_trains_sec: list of K spike trains, each already in float seconds
    - nrem_epochs: list of (start, end) tuples already in float seconds
    Returns a list of K×K STTC matrices (float64) in the same order as nrem_epochs.
    """
    delta = delta_ms * 1e-3  # convert ms to seconds

    sttc_matrices = []
    for t0_sec, t1_sec in nrem_epochs:
        # Extract spikes within the epoch for each channel efficiently
        seg_sec = []
        for train in all_spike_trains_sec:
            # Find indices of the window boundary via searchsorted
            idx0 = np.searchsorted(train, t0_sec, side="left")
            idx1 = np.searchsorted(train, t1_sec, side="right")
            seg = train[idx0:idx1]
            seg_sec.append(seg)

        # JIT-compiled STTC matrix
        M = sttc_matrix_fast(seg_sec, t0_sec, t1_sec, delta)
        sttc_matrices.append(M)

    return sttc_matrices


def dual_probe_sttc(full_mua, hypno_to_use, delta_ms=5.0):
    epocs = [
        (np.datetime64(bout.start_time), np.datetime64(bout.end_time))
        for bout in hypno_to_use.itertuples()
    ]
    sttc_mats = {}
    for probe in ["NNXr", "NNXo"]:
        mua_to_use = full_mua.filter(pl.col("probe") == probe).filter(
            (pl.col("datetime") >= hypno_to_use.start_time.min())
            & (pl.col("datetime") <= hypno_to_use.end_time.max())
        )
        strains = [
            mua_to_use.filter(pl.col("channel") == i)["datetime"].to_numpy()
            for i in range(1, 17)
        ]
        sttc_mats[probe] = process_nrem_epochs_fast(strains, epocs, delta_ms=delta_ms)
    return sttc_mats


def dual_probe_sttc_single_units(full_mua, hypno_to_use, unit_ids, delta_ms=5.0):
    epocs = [
        (np.datetime64(bout.start_time), np.datetime64(bout.end_time))
        for bout in hypno_to_use.itertuples()
    ]
    sttc_mats = {}
    for probe in ["NNXr", "NNXo"]:
        mua_to_use = full_mua.filter(pl.col("probe") == probe).filter(
            (pl.col("datetime") >= hypno_to_use.start_time.min())
            & (pl.col("datetime") <= hypno_to_use.end_time.max())
        )
        strains = [
            mua_to_use.filter(pl.col("unit_id") == unit_ids[probe][i])[
                "datetime"
            ].to_numpy()
            for i in range(len(unit_ids[probe]))
        ]
        sttc_mats[probe] = process_nrem_epochs_fast(strains, epocs, delta_ms=delta_ms)
    return sttc_mats


import numpy as np
from typing import List


def average_sttc_matrices(sttc_list: List[np.ndarray]) -> np.ndarray:
    """
    Given a list of K×K STTC matrices (one per NREM bout), return the
    element-wise average across all bouts, ignoring NaNs.

    sttc_list: List of NumPy arrays of shape (K, K).  All must share the same shape.
    Returns:   A single (K, K) array where entry (i,j) is the nanmean of sttc_list[b][i,j].
    """
    # Stack into shape (K, K, n_bouts)
    arr = np.stack(sttc_list, axis=2)
    # Compute element-wise mean, ignoring NaNs
    mean_mat = np.nanmean(arr, axis=2)
    return mean_mat


import numpy as np
from typing import List, Sequence


def mask_bad_channels(
    sttc_matrices: List[np.ndarray], bad_channels: Sequence[int]
) -> List[np.ndarray]:
    """
    Replace all (row, col) entries involving *any* bad channel with NaN.

    Parameters
    ----------
    sttc_matrices : list of (K, K) arrays
        One STTC matrix per NREM bout (already computed).
    bad_channels  : iterable of ints
        Zero-based indices of channels to exclude, e.g. [3, 5].

    Returns
    -------
    masked_mats : list of (K, K) arrays
        Same objects left-in-place (for memory economy) but with NaNs
        written into the rows and columns of every bad channel.
    """
    if not bad_channels:
        return sttc_matrices  # nothing to do

    for M in sttc_matrices:  # loop over bouts
        M[np.ix_(bad_channels, range(M.shape[1]))] = np.nan
        M[np.ix_(range(M.shape[0]), bad_channels)] = np.nan
        # optional: also NaN the diagonal of those channels if it
        # wasn’t already (our STTC code leaves the whole diag NaN).
    return sttc_matrices
