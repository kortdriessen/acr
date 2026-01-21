# Functions for mountain-sort based single unit pipelines

import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import spikeinterface as si
from spikeinterface.qualitymetrics import (
    compute_drift_metrics,
    compute_quality_metrics,
)
from spikeinterface.sorters import run_sorter

import acr

ms5_params = dict(
    # Use scheme 3 for long, drifting recordings
    scheme="3",
    # ---- Detection ----
    detect_sign=-1,  # negative-going spikes (standard for cortex)
    detect_threshold=5.5,  # in SD of whitened noise
    detect_time_radius_msec=0.5,  # suppress double detections within 0.5 ms
    # ---- Snippet window ----
    snippet_T1=20,  # samples before peak  (~0.82 ms at 24.4 kHz)
    snippet_T2=20,  # samples after peak   (~0.82 ms)
    # ---- PCA dimensionality ----
    npca_per_channel=3,  # 3 PCs per channel  -> 48D for 16-ch probe
    npca_per_subdivision=10,  # PCA dim used inside subdivision method
    # ---- Spatial radii (geometry units, e.g. microns) ----
    snippet_mask_radius=200.0,  # center + a few neighbors
    scheme1_detect_channel_radius=150.0,  # phase-1 detection neighborhood
    scheme2_phase1_detect_channel_radius=200.0,
    scheme2_detect_channel_radius=50.0,  # smaller radius in final phase
    # ---- Scheme-2 training (applied within each scheme-3 block) ----
    scheme2_max_num_snippets_per_training_batch=200,
    scheme2_training_duration_sec=300.0,  # 5 min training per block
    scheme2_training_recording_sampling_mode="uniform",
    # ---- Scheme-3 drift handling ----
    scheme3_block_duration_sec=1800.0,  # 30-min blocks across the 36 h
    # ---- Sorter-internal preprocessing (we already did this) ----
    filter=False,  # do NOT re-filter inside MS5
    whiten=False,  # do NOT re-whiten inside MS5
    freq_min=300.0,
    freq_max=12000.0,  # irrelevant if filter=False, but explicit
    # ---- Runtime / parallelization (optional) ----
    delete_temporary_recording=True,
    pool_engine="process",  # default
    n_jobs=32,  # MS5 internal parallelization; you can
    chunk_duration="1s",  # leave as default and parallelize at SI level
    progress_bar=True,
    mp_context=None,
    max_threads_per_worker=1,
)

from spikeinterface.sorters import get_default_sorter_params

ms5_params2 = get_default_sorter_params("mountainsort5")
ms5_params2.update(
    {
        "scheme": "2",  # use scheme 2
        # Detection
        "detect_sign": -1,  # negative spikes (very likely your case)
        "detect_threshold": 5.0,  # good starting value on whitened traces
        "detect_time_radius_msec": 0.5,
        # Waveform window (these defaults are fine)
        "snippet_T1": 20,  # ~0.82 ms pre at 24.4 kHz
        "snippet_T2": 20,  # ~0.82 ms post
        # PCA features (defaults are solid)
        "npca_per_channel": 3,
        "npca_per_subdivision": 10,
        # Spatial radii – tuned to your 50 µm geometry but basically defaults
        "snippet_mask_radius": 250,  # ≈ ±5 chans → 11-ch neighborhood
        "scheme1_detect_channel_radius": 150,  # ≈ ±3 chans
        "scheme2_phase1_detect_channel_radius": 200,  # ≈ ±4 chans
        "scheme2_detect_channel_radius": 50,  # ≈ ±1 chan; keeps detection very local
        # Training parameters
        "scheme2_max_num_snippets_per_training_batch": 500,
        # (bump up from 200; with 16 chans this is still cheap and gives nicer classifiers)
        "scheme2_training_duration_sec": 300,  # 5 minutes of data
        "scheme2_training_recording_sampling_mode": "uniform",
        # Preprocessing flags – **critical** given your offline preprocessing
        "filter": False,
        "whiten": False,
        "freq_min": 300,  # ignored when filter=False but OK to leave
        "freq_max": 6000,
        # Runtime / parallelism
        "n_jobs": 64,  # or whatever fits your machine
        "pool_engine": "process",
        "chunk_duration": "1s",
        "delete_temporary_recording": True,
        "progress_bar": True,
    }
)

sing_unit_rec_root = (
    "/Volumes/neuropixel_archive/Data/acr_archive/mua_data/single_unit_sortings"
)
sing_unit_root = "/nvme/sorting/acr_single_units"


def spikes_dataframe_from_analyzer(
    analyzer_folder,
    good_unit_ids=None,
):
    """
    Load a SortingAnalyzer and return a dataframe with one row per spike.

    Columns:
      - 'unit_id'
      - 'segment_index'
      - 'time'          (seconds)
      - 'channel'       (extremum channel id for that unit)
      - 'unit_location' (depth / y-position of that unit in probe coordinates)
    """
    # Load analyzer & underlying objects
    analyzer = si.load_sorting_analyzer(Path(analyzer_folder))
    sorting = analyzer.sorting

    # Optionally restrict to a subset of units
    if good_unit_ids is not None:
        analyzer = analyzer.select_units(list(good_unit_ids))
        sorting = analyzer.sorting

    unit_ids = np.array(sorting.unit_ids)
    recording = analyzer.recording
    chan_ids = np.array(recording.channel_ids)

    # ---------- 1) per-unit main channel (peak template channel) ----------
    if not analyzer.has_extension("templates"):
        raise RuntimeError(
            "SortingAnalyzer is missing 'templates' extension. "
            "Compute it first with sorting_analyzer.compute('templates')."
        )

    templates_ext = analyzer.get_extension("templates")
    templates = np.asarray(templates_ext.get_data())
    # Expected shape: (n_units, n_samples, n_channels)

    if templates.ndim != 3 or templates.shape[0] != len(unit_ids):
        raise RuntimeError(
            f"Unexpected templates shape {templates.shape}; "
            "expected (n_units, n_samples, n_channels)."
        )

    # Take the minimum (most negative) amplitude over time for each channel
    # → shape (n_units, n_channels)
    per_channel_min = templates.min(axis=1)

    # For negative-going spikes, the most negative channel is the "main" one
    main_chan_idx = np.argmin(per_channel_min, axis=1)  # (n_units,)

    # Map unit_id → channel_id
    unit_main_channel = {
        uid: chan_ids[idx] for uid, idx in zip(unit_ids, main_chan_idx)
    }

    # ---------- 2) per-unit location/depth ----------
    if not analyzer.has_extension("unit_locations"):
        raise RuntimeError(
            "SortingAnalyzer is missing 'unit_locations' extension. "
            "Compute it with sorting_analyzer.compute('unit_locations', ...)."
        )

    ul_ext = analyzer.get_extension("unit_locations")
    ul_data = np.asarray(ul_ext.get_data())
    # Typical shape: (n_units, 2) or (n_units, 3) for (x, y[, z])

    if ul_data.shape[0] != len(unit_ids):
        raise RuntimeError(
            f"unit_locations length {ul_data.shape[0]} does not match "
            f"number of units {len(unit_ids)}."
        )

    # For a linear laminar probe we usually care about depth (y coordinate).
    if ul_data.ndim == 2 and ul_data.shape[1] >= 2:
        # Use second coordinate as depth / cortical location
        unit_location_map = {
            uid: float(ul_data[i, 1]) for i, uid in enumerate(unit_ids)
        }
    else:
        # Fallback: 1D positions
        unit_location_map = {uid: float(ul_data[i]) for i, uid in enumerate(unit_ids)}

    # ---------- 3) build spike table ----------
    n_segments = sorting.get_num_segments()
    rows = []

    for seg in range(n_segments):
        for uid in unit_ids:
            spike_times = sorting.get_unit_spike_train(
                unit_id=uid,
                segment_index=seg,
                return_times=True,  # times in seconds
            )
            if spike_times.size == 0:
                continue

            chan = unit_main_channel[uid]
            loc = unit_location_map[uid]

            # One row per spike
            for t in spike_times:
                rows.append((uid, seg, float(t), chan, loc))

    df = pd.DataFrame(
        rows,
        columns=["unit_id", "segment_index", "time", "channel", "unit_location"],
    )
    return pl.from_pandas(df)


def _run_ms5_sorting_pipeline(
    path_to_si_rec,
    path_to_sorting_folder,
    path_to_analyzer,
    qm_path,
    spike_df_path,
    remove_existing_folder=True,
    n_jobs=32,
    progress_bar=True,
    chunk_duration="1s",
    recording=None,
    m_params=ms5_params,
):
    if recording is None:
        recording = si.read_zarr(path_to_si_rec)
    sorting = run_sorter(
        sorter_name="mountainsort5",
        recording=recording,  # bandpassed + refed + whitened rec
        folder=path_to_sorting_folder,
        remove_existing_folder=remove_existing_folder,
        **m_params,
    )
    job_kwargs = dict(
        n_jobs=n_jobs, progress_bar=progress_bar, chunk_duration=chunk_duration
    )
    sorting_analyzer = si.create_sorting_analyzer(
        sorting,
        recording,
        format="binary_folder",
        folder=path_to_analyzer,
        overwrite=remove_existing_folder,
        **job_kwargs,
    )
    sorting_analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
    sorting_analyzer.compute("waveforms", **job_kwargs)
    sorting_analyzer.compute("templates", **job_kwargs)
    sorting_analyzer.compute("noise_levels")
    sorting_analyzer.compute("unit_locations", method="monopolar_triangulation")
    sorting_analyzer.compute("isi_histograms")
    sorting_analyzer.compute("correlograms", window_ms=100, bin_ms=5.0)
    sorting_analyzer.compute(
        "principal_components",
        n_components=3,
        mode="by_channel_global",
        whiten=True,
        **job_kwargs,
    )
    sorting_analyzer.compute("quality_metrics", metric_names=["snr", "firing_rate"])
    sorting_analyzer.compute("template_similarity")
    sorting_analyzer.compute("spike_amplitudes", **job_kwargs)
    sorting_analyzer.compute(
        "spike_locations",
        ms_before=0.5,
        ms_after=0.5,
        method="center_of_mass",  # fast and good enough for drift
        spike_retriver_kwargs=dict(
            channel_from_template=True,
            radius_um=150.0,  # ~3 neighbors on either side for 50 µm spacing
            peak_sign="neg",
        ),
        **job_kwargs,
    )
    sorting_analyzer.compute(
        "template_metrics", include_multi_channel_metrics=True, **job_kwargs
    )

    # ---------- Quality Metrics ---------- #
    metric_names = [
        "firing_rate",
        "presence_ratio",
        "snr",
        "isi_violation",
        "amplitude_cutoff",
    ]

    qm = compute_quality_metrics(
        sorting_analyzer=sorting_analyzer,
        metric_names=metric_names,
        load_if_exists=False,  # reuse if already computed
        skip_pc_metrics=True,  # avoids computing PCA-based metrics, saves time
    )
    drift_ptp_dict, drift_std_dict, drift_mad_dict = compute_drift_metrics(
        sorting_analyzer=sorting_analyzer,
        interval_s=60,  # 1-min bins
        min_spikes_per_interval=100,  # default; can tweak later
        direction="y",  # vertical axis of your laminar probe
        min_fraction_valid_intervals=0.5,
        min_num_bins=2,
    )

    # Convert dicts -> Series aligned with qm index, and attach as columns
    qm["drift_ptp"] = qm.index.to_series().map(drift_ptp_dict)
    qm["drift_std"] = qm.index.to_series().map(drift_std_dict)
    qm["drift_mad"] = qm.index.to_series().map(drift_mad_dict)
    all_unit_ids = qm.index.to_numpy()
    qm.to_parquet(qm_path)

    all_spikes_df = spikes_dataframe_from_analyzer(path_to_analyzer, all_unit_ids)
    all_spikes_df.write_parquet(spike_df_path)
    return


def run_ms5_sorting_pipeline_scheme2(
    path_to_si_rec,
    path_to_sorting_folder,
    path_to_analyzer,
    qm_path,
    spike_df_path,
    remove_existing_folder=True,
    n_jobs=112,
    progress_bar=True,
    chunk_duration="1s",
    recording=None,
):
    if recording is None:
        recording = si.read_zarr(path_to_si_rec)
    sorting = run_sorter(
        sorter_name="mountainsort5",
        recording=recording,  # bandpassed + refed + whitened rec
        folder=path_to_sorting_folder,
        remove_existing_folder=remove_existing_folder,
        **ms5_params2,
    )
    job_kwargs = dict(
        n_jobs=n_jobs, progress_bar=progress_bar, chunk_duration=chunk_duration
    )
    sorting_analyzer = si.create_sorting_analyzer(
        sorting,
        recording,
        format="binary_folder",
        folder=path_to_analyzer,
        overwrite=remove_existing_folder,
        **job_kwargs,
    )
    sorting_analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
    sorting_analyzer.compute("waveforms", **job_kwargs)
    sorting_analyzer.compute("templates", **job_kwargs)
    sorting_analyzer.compute("noise_levels")
    sorting_analyzer.compute("unit_locations", method="monopolar_triangulation")
    sorting_analyzer.compute("isi_histograms")
    sorting_analyzer.compute("correlograms", window_ms=100, bin_ms=5.0)
    sorting_analyzer.compute(
        "principal_components",
        n_components=3,
        mode="by_channel_global",
        whiten=True,
        **job_kwargs,
    )
    sorting_analyzer.compute("quality_metrics", metric_names=["snr", "firing_rate"])
    sorting_analyzer.compute("template_similarity")
    sorting_analyzer.compute("spike_amplitudes", **job_kwargs)
    sorting_analyzer.compute(
        "spike_locations",
        ms_before=0.5,
        ms_after=0.5,
        method="center_of_mass",  # fast and good enough for drift
        spike_retriver_kwargs=dict(
            channel_from_template=True,
            radius_um=150.0,  # ~3 neighbors on either side for 50 µm spacing
            peak_sign="neg",
        ),
        **job_kwargs,
    )
    sorting_analyzer.compute(
        "template_metrics", include_multi_channel_metrics=True, **job_kwargs
    )

    # ---------- Quality Metrics ---------- #
    metric_names = [
        "firing_rate",
        "presence_ratio",
        "snr",
        "isi_violation",
        "amplitude_cutoff",
    ]

    qm = compute_quality_metrics(
        sorting_analyzer=sorting_analyzer,
        metric_names=metric_names,
        load_if_exists=False,  # reuse if already computed
        skip_pc_metrics=True,  # avoids computing PCA-based metrics, saves time
    )
    drift_ptp_dict, drift_std_dict, drift_mad_dict = compute_drift_metrics(
        sorting_analyzer=sorting_analyzer,
        interval_s=60,  # 1-min bins
        min_spikes_per_interval=100,  # default; can tweak later
        direction="y",  # vertical axis of your laminar probe
        min_fraction_valid_intervals=0.5,
        min_num_bins=2,
    )

    # Convert dicts -> Series aligned with qm index, and attach as columns
    qm["drift_ptp"] = qm.index.to_series().map(drift_ptp_dict)
    qm["drift_std"] = qm.index.to_series().map(drift_std_dict)
    qm["drift_mad"] = qm.index.to_series().map(drift_mad_dict)
    all_unit_ids = qm.index.to_numpy()
    qm.to_parquet(qm_path)

    all_spikes_df = spikes_dataframe_from_analyzer(path_to_analyzer, all_unit_ids)
    all_spikes_df.write_parquet(spike_df_path)
    return


def run_ms5_pipeline(
    subject,
    exp,
    probes=["NNXo", "NNXr"],
    sorting_id="matched_cbl-reb",
    n_jobs=32,
    m_params=ms5_params,
    single_unit_root=None,
):
    if single_unit_root is None:
        single_unit_root = "/nvme/sorting/acr_single_units"
    sub_dir = os.path.join(single_unit_root, subject)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    exp_dir = os.path.join(sub_dir, exp)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    sorting_id_dir = os.path.join(exp_dir, sorting_id)
    if not os.path.exists(sorting_id_dir):
        os.mkdir(sorting_id_dir)

    for probe in probes:
        path_to_si_rec = os.path.join(
            sing_unit_rec_root,
            subject,
            exp,
            sorting_id,
            f"si_rec--{probe}.zarr",
        )
        path_to_sorting_folder = os.path.join(
            single_unit_root,
            subject,
            exp,
            sorting_id,
            f"sorting--{probe}",
        )
        path_to_analyzer = os.path.join(
            single_unit_root,
            subject,
            exp,
            sorting_id,
            f"analyzer--{probe}",
        )
        path_to_qm = os.path.join(
            single_unit_root,
            subject,
            exp,
            sorting_id,
            f"qm--{probe}.parquet",
        )
        path_to_spike_df = os.path.join(
            single_unit_root,
            subject,
            exp,
            sorting_id,
            f"spike_df--{probe}.parquet",
        )
        _run_ms5_sorting_pipeline(
            path_to_si_rec,
            path_to_sorting_folder,
            path_to_analyzer,
            path_to_qm,
            path_to_spike_df,
            n_jobs=n_jobs,
            m_params=m_params,
        )
    return


def check_spike_df_exists(subject, exp, probe, sorting_id):
    path_to_spike_df = os.path.join(
        sing_unit_root,
        subject,
        exp,
        sorting_id,
        f"spike_df--{probe}.parquet",
    )
    if os.path.exists(path_to_spike_df):
        return True
    else:
        return False


def load_quality_metrics_df(subject, exp, probe, sorting_id):
    path_to_qm = os.path.join(
        sing_unit_root,
        subject,
        exp,
        sorting_id,
        f"qm--{probe}.parquet",
    )
    qm = pl.read_parquet(path_to_qm)
    return qm


def label_ebl_cbl_reb_df(df):
    df = df.with_columns(pl.lit("no_cond").alias("condition"))
    df = df.with_columns(
        pl.when(pl.col("time") <= 3600.0)
        .then(pl.lit("early-bl"))
        .otherwise(pl.col("condition"))
        .alias("condition")
    )
    df = df.with_columns(
        pl.when((pl.col("time") <= 7200.0) & (pl.col("time") > 3600.0))
        .then(pl.lit("circ-bl"))
        .otherwise(pl.col("condition"))
        .alias("condition")
    )
    df = df.with_columns(
        pl.when((pl.col("time") <= 10800.0) & (pl.col("time") > 7200.0))
        .then(pl.lit("rebound"))
        .otherwise(pl.col("condition"))
        .alias("condition")
    )
    return df


def load_spike_df_with_qm(
    subject, exp, sorting_id, ebl_cbl_reb_labels=True, return_analyzers=True
):
    spike_dfs = []
    qms = []
    analyzer_paths = {}
    analyzers = {}
    for probe in ["NNXr", "NNXo"]:
        if acr.ms.check_spike_df_exists(subject, exp, probe, sorting_id):
            analyzer_paths[probe] = os.path.join(
                sing_unit_root,
                subject,
                exp,
                sorting_id,
                f"analyzer--{probe}",
            )
            qm = load_quality_metrics_df(subject, exp, probe, sorting_id)
            qm = qm.rename({"__index_level_0__": "unit_id"})
            all_unit_ids = qm["unit_id"].to_numpy()
            spike_df = spikes_dataframe_from_analyzer(
                analyzer_paths[probe], all_unit_ids
            )
            spike_df = spike_df.with_columns(pl.lit(probe).alias("probe"))
            qm = qm.with_columns(pl.lit(probe).alias("probe"))
            spike_dfs.append(spike_df)
            qms.append(qm)
    full_spike_df = pl.concat(spike_dfs)
    full_qm = pl.concat(qms)
    if ebl_cbl_reb_labels:
        full_spike_df = label_ebl_cbl_reb_df(full_spike_df)
    if return_analyzers:
        for probe in analyzer_paths.keys():
            analyzers[probe] = si.load_sorting_analyzer(analyzer_paths[probe])
        return full_spike_df, full_qm, analyzers
    else:
        return full_spike_df, full_qm


def load_saved_dfs(
    subject,
    exp,
    sorting_id,
):
    spike_dfs = []
    qms = []
    for probe in ["NNXr", "NNXo"]:
        if check_spike_df_exists(subject, exp, probe, sorting_id):
            spike_df_path = os.path.join(
                sing_unit_root, subject, exp, sorting_id, f"spike_df--{probe}.parquet"
            )
            qm_path = os.path.join(
                sing_unit_root, subject, exp, sorting_id, f"qm--{probe}.parquet"
            )
            qm = pl.read_parquet(qm_path)
            qm = qm.rename({"__index_level_0__": "unit_id"})
            spike_df = pl.read_parquet(spike_df_path)
            spike_df = spike_df.with_columns(pl.lit(probe).alias("probe"))
            qm = qm.with_columns(pl.lit(probe).alias("probe"))
            spike_dfs.append(spike_df)
            qms.append(qm)
    full_spike_df = pl.concat(spike_dfs)
    full_qm = pl.concat(qms)
    return full_spike_df, full_qm


def select_good_units(
    qm,
    snr_min=0.0,
    amp_cutoff_max=0.1,
    isi_viol_max=0.2,
    presence_min=0.8,
    fr_min=0.1,
    drift_ptp_max=100.0,  # µm
    full_mask=False,
):
    mask_full = (
        (qm["snr"] >= snr_min)
        & (qm["amplitude_cutoff"] <= amp_cutoff_max)
        & (qm["isi_violations_ratio"] <= isi_viol_max)
        & (qm["presence_ratio"] >= presence_min)
        & (qm["firing_rate"] >= fr_min)
        & (qm["drift_ptp"] <= drift_ptp_max)
    )
    mask_presence = qm["presence_ratio"] > presence_min

    if full_mask:
        mask = mask_full
    else:
        mask = mask_presence

    good_units = qm["unit_id"][mask].to_numpy()
    return good_units, mask
