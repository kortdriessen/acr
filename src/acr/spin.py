from __future__ import annotations

import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
import yasa
from scipy import signal

import acr
import kdephys as kde


@dataclass(frozen=True)
class DownsampleInfo:
    fs_in_hz: float
    fs_out_hz: float
    method: str
    factor_or_ratio: str
    dt_median_s: float
    dt_jitter_rel: float  # robust jitter estimate / median dt


def _infer_fs_from_datetime_coord(t: np.ndarray) -> tuple[float, float, float]:
    """
    Infer sampling rate from a 1D numpy datetime64 coordinate.

    Returns
    -------
    fs_hz : float
        Inferred sampling frequency (1 / median_dt).
    median_dt_s : float
        Median sample period in seconds.
    jitter_rel : float
        Robust relative jitter estimate: MAD(dt) / median_dt.
    """
    if t.ndim != 1:
        raise ValueError("Time coordinate must be 1D.")
    if not np.issubdtype(t.dtype, np.datetime64):
        raise TypeError(f"Expected datetime64 coordinate, got {t.dtype}.")

    dt = np.diff(t).astype("timedelta64[ns]").astype(np.int64)  # ns
    if dt.size < 2:
        raise ValueError("Need at least 3 time points to infer a stable sampling rate.")

    med_ns = np.median(dt)
    if med_ns <= 0:
        raise ValueError(
            "Non-positive median dt detected; timestamps may be unsorted or duplicated."
        )

    # Robust jitter estimate (median absolute deviation)
    mad_ns = np.median(np.abs(dt - med_ns))
    jitter_rel = float(mad_ns / med_ns)

    median_dt_s = float(med_ns) * 1e-9
    fs_hz = 1.0 / median_dt_s
    return fs_hz, median_dt_s, jitter_rel


def downsample_lfp_dataarray(
    lfp: xr.DataArray,
    fs_hz: Optional[float] = None,
    target_fs_hz: float = 100.0,
    time_dim: str = "datetime",
    prefer_integer_decimation: bool = True,
    integer_tol: float = 1e-3,
    max_denominator: int = 2000,
    ftype: str = "fir",
    zero_phase: bool = True,
    jitter_warn_rel: float = 1e-3,
) -> Tuple[xr.DataArray, np.ndarray, DownsampleInfo]:
    """
    Downsample an LFP xarray.DataArray from ~400 Hz to ~100 Hz (or any target),
    returning both the downsampled DataArray and matching timestamps.

    Parameters
    ----------
    lfp : xr.DataArray
        LFP with dims including `time_dim` and (typically) 'channel'.
    fs_hz : float, optional
        Sampling rate of `lfp` in Hz. If None, inferred from timestamps.
    target_fs_hz : float
        Desired output sampling rate (e.g., 100.0).
    time_dim : str
        Name of the time dimension (default: 'datetime').
    prefer_integer_decimation : bool
        If True, use scipy.signal.decimate when fs/target_fs is close to an integer.
    integer_tol : float
        Relative tolerance for considering fs/target_fs to be an integer.
    max_denominator : int
        Max denominator when approximating a rational resampling ratio for resample_poly.
    ftype : str
        Filter type for decimate ('fir' recommended here).
    zero_phase : bool
        Use zero-phase correction in decimate (recommended) :contentReference[oaicite:3]{index=3}.
    jitter_warn_rel : float
        If robust timestamp jitter exceeds this fraction of dt, raise a warning.

    Returns
    -------
    lfp_ds : xr.DataArray
        Downsampled LFP with updated time coordinate.
    t_ds : np.ndarray
        Downsampled timestamps (datetime64).
    info : DownsampleInfo
        Metadata about inference and resampling choices.
    """
    if time_dim not in lfp.dims:
        raise ValueError(f"`time_dim`='{time_dim}' not found in dims: {lfp.dims}")

    # Pull time coordinate as numpy datetime64
    t = lfp[time_dim].values
    fs_in, med_dt_s, jitter_rel = _infer_fs_from_datetime_coord(t)

    if fs_hz is not None:
        # Trust user-provided fs but keep inferred values for diagnostics
        fs_in_use = float(fs_hz)
    else:
        fs_in_use = float(fs_in)

    if jitter_rel > jitter_warn_rel:
        # For true DSP resampling, you want (nearly) uniform sampling.
        # If timestamps are irregular, consider re-timestamping/interpolating first.
        import warnings

        warnings.warn(
            f"Timestamps appear irregular: robust relative jitter ~ {jitter_rel:.3g}. "
            "DSP-style resampling assumes (near) uniform sampling.",
            RuntimeWarning,
        )

    axis = lfp.get_axis_num(time_dim)
    x = np.asarray(
        lfp.data
    )  # will error if this is a dask array; intentional for DSP ops

    ratio = fs_in_use / float(target_fs_hz)

    # --- Path A: clean integer decimation (your ~400 -> ~100 is typically q=4) ---
    if prefer_integer_decimation:
        q = int(np.round(ratio))
        rel_err = abs(ratio - q) / ratio
        if q >= 2 and rel_err < integer_tol:
            y = signal.decimate(x, q=q, ftype=ftype, axis=axis, zero_phase=zero_phase)
            # For decimate, output corresponds to taking every q-th sample after filtering,
            # so timestamps can be subselected directly.
            t_ds = t[::q]

            lfp_ds = lfp.isel({time_dim: slice(None, None, q)}).copy(deep=False)
            lfp_ds.data = y
            lfp_ds = lfp_ds.assign_coords({time_dim: t_ds})
            lfp_ds.attrs = dict(lfp.attrs)
            lfp_ds.attrs.update(
                {
                    "fs_hz": float(target_fs_hz),
                    "downsample_method": f"decimate(q={q}, ftype={ftype})",
                }
            )

            info = DownsampleInfo(
                fs_in_hz=fs_in_use,
                fs_out_hz=float(target_fs_hz),
                method="decimate",
                factor_or_ratio=f"q={q} (ratio={ratio:.6f}, rel_err={rel_err:.3g})",
                dt_median_s=med_dt_s,
                dt_jitter_rel=jitter_rel,
            )
            return lfp_ds, t_ds, info

    # --- Path B: rational polyphase resampling for non-integer ratios ---
    # resample_poly changes the number of samples; we create a fresh uniform timestamp axis.
    frac = Fraction(target_fs_hz / fs_in_use).limit_denominator(max_denominator)
    up, down = frac.numerator, frac.denominator

    y = signal.resample_poly(
        x, up=up, down=down, axis=axis
    )  # :contentReference[oaicite:4]{index=4}
    n_out = y.shape[axis]

    # Build new timestamps anchored to the original start time
    t0 = t[0].astype("datetime64[ns]")
    step_ns = int(
        np.round(1e9 / float(target_fs_hz))
    )  # exact for 100 Hz (10,000,000 ns)
    t_ds = t0 + (np.arange(n_out, dtype=np.int64) * np.timedelta64(step_ns, "ns"))

    # Replace time coordinate and data
    lfp_ds = lfp.copy(deep=False)
    # Construct new coordinate safely: drop old then assign
    lfp_ds = lfp_ds.drop_vars(time_dim).assign_coords({time_dim: t_ds})
    # Update sizes along time_dim to match y
    # Easiest: rebuild via transpose to align, then assign data with correct shape
    lfp_ds = lfp_ds.isel({time_dim: slice(0, n_out)}).copy(deep=False)
    lfp_ds.data = y
    lfp_ds.attrs = dict(lfp.attrs)
    lfp_ds.attrs.update(
        {
            "fs_hz": float(target_fs_hz),
            "downsample_method": f"resample_poly(up={up}, down={down})",
        }
    )

    info = DownsampleInfo(
        fs_in_hz=fs_in_use,
        fs_out_hz=float(target_fs_hz),
        method="resample_poly",
        factor_or_ratio=f"up/down={up}/{down} (~{float(frac):.8f})",
        dt_median_s=med_dt_s,
        dt_jitter_rel=jitter_rel,
    )
    return lfp_ds, t_ds, info


def convert_states_array_to_int(st=["NREM"]):
    mapping = {
        "Wake": 0,
        "NREM": 1,
        "Transition-to-REM": 2,
        "REM": 4,
    }
    return np.array([mapping.get(s, -2) for s in st])


def generate_spindle_detections(
    subject,
    exp,
    threshes={"rel_pow": 0.2, "corr": 0.65, "rms": 1.5},
    frange=(10, 15),
    write=True,
    overwrite=True,
    tag="base",
):
    h = acr.io.load_hypno_full_exp(subject, exp, update=False)
    recs = acr.info_pipeline.get_exp_recs(subject, exp)
    h_ints = {}
    for rec in recs:
        if "-post" in rec or "-sd" in rec:
            continue
        for store in ["NNXo", "NNXr"]:
            print(f"{subject} {rec} {store}")
            lfp = acr.io.load_raw_data(subject, rec, store=store)
            lfp_100, t_100, info = downsample_lfp_dataarray(
                lfp, fs_hz=lfp.fs, target_fs_hz=100.0
            )
            lfp_start = pd.Timestamp(lfp_100.datetime.values.min())
            lfp_end = pd.Timestamp(lfp_100.datetime.values.max())
            h_rec = h.trim_select(lfp_start, lfp_end)
            raw_dat = lfp_100.data
            raw_dat = raw_dat.swapaxes(0, 1)
            if rec not in h_ints:
                st = kde.hypno.hypno.get_states_fast(h_rec, lfp_100.datetime.values)
                h_int = convert_states_array_to_int(st)
                h_ints[rec] = h_int
            ch_names = [str(i) for i in range(1, 17)]
            detres = yasa.spindles_detect(
                data=raw_dat,
                sf=lfp_100.fs_hz,
                ch_names=ch_names,
                hypno=h_ints[rec],
                include=(1),
                freq_sp=frange,
                thresh=threshes,
            )
            spdf = detres.summary()
            spdf["rec"] = rec
            spdf["probe"] = store
            spdf["subject"] = subject
            spdf = pl.from_pandas(spdf)
            idx_chan = spdf["IdxChannel"].to_numpy()
            idx_chan = idx_chan + 1
            spdf = spdf.with_columns((pl.col("IdxChannel") + 1).alias("IdxChannel_1"))
            spdf = spdf.drop("IdxChannel")
            spdf = spdf.drop("Channel")
            spdf = spdf.rename({"IdxChannel_1": "Channel"})

            starts = spdf["Start"].to_numpy()
            starts_delta = pd.to_timedelta(starts, unit="s")
            starts_dt = lfp_start + starts_delta
            starts_dt = np.array(starts_dt)
            ends = spdf["End"].to_numpy()
            ends_delta = pd.to_timedelta(ends, unit="s")
            ends_dt = lfp_start + ends_delta
            ends_dt = np.array(ends_dt)
            peaks = spdf["Peak"].to_numpy()
            peaks_delta = pd.to_timedelta(peaks, unit="s")
            peaks_dt = lfp_start + peaks_delta
            peaks_dt = np.array(peaks_dt)
            spdf = spdf.with_columns(pl.lit(starts_dt).alias("start_time"))
            spdf = spdf.with_columns(pl.lit(ends_dt).alias("end_time"))
            spdf = spdf.with_columns(pl.lit(peaks_dt).alias("time"))

            if write:
                sp_dir = f"{acr.utils.raw_data_root}/spindles/{subject}"
                if not os.path.exists(sp_dir):
                    os.makedirs(sp_dir)
                sp_path = f"{sp_dir}/{rec}--{store}--{tag}.parquet"
                if not os.path.exists(sp_path):
                    spdf.write_parquet(sp_path)
                else:
                    if overwrite:
                        os.remove(sp_path)
                        spdf.write_parquet(sp_path)
                    else:
                        print(f"{sp_path} already exists, skipping")


def load_spindle_df(subject, exp, tag="base"):
    sp_dir = f"{acr.utils.raw_data_root}/spindles/{subject}"
    recs = acr.info_pipeline.get_exp_recs(subject, exp)
    dfs = []
    for rec in recs:
        if "-post" in rec or "-sd" in rec:
            continue
        for store in ["NNXo", "NNXr"]:
            sp_path = f"{sp_dir}/{rec}--{store}--{tag}.parquet"
            spdf = pl.read_parquet(sp_path)
            dfs.append(spdf)
    return pl.concat(dfs)
