import acr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl


def get_individual_pulse_times(subject, recording, store):
    """get the pulse times for all individual pulses in an experiment

    Parameters
    ----------
    subject : str
        subject name
    recording : str
        recording name
    store : str
        stim store, e.g. 'Pu1_'
    """
    sub_info = acr.info_pipeline.load_subject_info(subject)
    pulse_ons = sub_info["stim_info"][recording][store]["onsets"]
    pulse_offs = sub_info["stim_info"][recording][store]["offsets"]
    pulse_ons = pd.to_datetime(pulse_ons)
    pulse_ons = pulse_ons.to_numpy()
    pulse_offs = pd.to_datetime(pulse_offs)
    pulse_offs = pulse_offs.to_numpy()
    return pulse_ons, pulse_offs


def get_pulse_train_times(subject, recording, store):
    """Get the index values of the pulse train onsets and offsets

    Args:
        subject (str): subject name
        recording (str): recording name
        store (str): store name under which the pulses are stored
    """
    sub_info = acr.info_pipeline.load_subject_info(subject)
    pulse_ons = sub_info["stim_info"][recording][store]["onsets"]
    pulse_offs = sub_info["stim_info"][recording][store]["offsets"]
    pulse_ons = pd.to_datetime(pulse_ons)
    pulse_offs = pd.to_datetime(pulse_offs)
    train_ons = [0]
    train_offs = []
    for i in np.arange(0, (len(pulse_ons) - 1)):
        diff = pulse_ons[i + 1] - pulse_ons[i]
        if diff.total_seconds() > 5:
            train_ons.append(i + 1)
            train_offs.append(i)
    train_offs.append(len(pulse_ons) - 1)
    return train_ons, train_offs


def get_sorting_stim_start(subject, exp):
    """gives the time at which a stimulation started, in TOTAL time, as it would appear in a kilosort sorting.

    Args:
        subject (str): subject name
        exp (str): experiment name
    """
    sub_info = acr.info_pipeline.load_subject_info(subject)
    stim_store = sub_info["stim-exps"][exp]
    stim_start = pd.Timestamp(sub_info["stim_info"][exp][stim_store]["onsets"][0])
    exp_start = pd.Timestamp(sub_info["rec_times"][exp]["start"])
    recs = acr.info_pipeline.get_exp_recs(subject, exp)
    time_to_stim = (stim_start - exp_start).total_seconds()
    for rec in recs:
        rec_start = pd.Timestamp(sub_info["rec_times"][rec]["start"])
        if rec_start < exp_start:
            time_to_stim += sub_info["rec_times"][rec]["duration"]
    return time_to_stim


def stim_bookends(subject, exp):
    """gives the starting and ending datetimes that a stimulation happened for a given experiment.
    Returns:
    ---------
    stim_start, stim_end : pd.Timestamp
    """

    sub_info = acr.info_pipeline.load_subject_info(subject)
    stim_store = sub_info["stim-exps"][exp]
    stim_start = pd.Timestamp(sub_info["stim_info"][exp][stim_store]["onsets"][0])
    stim_end = pd.Timestamp(sub_info["stim_info"][exp][stim_store]["offsets"][-1])
    return stim_start, stim_end


def get_sorting_time_from_dt(subject, exp, dt, dt_rec):
    sub_info = acr.info_pipeline.load_subject_info(subject)
    dt_rec_start = pd.Timestamp(sub_info["rec_times"][dt_rec]["start"])
    recs = acr.info_pipeline.get_exp_recs(subject, exp)
    time_to_dt = (dt - dt_rec_start).total_seconds()
    for rec in recs:
        rec_start = pd.Timestamp(sub_info["rec_times"][rec]["start"])
        if rec_start < dt_rec_start:
            time_to_dt += sub_info["rec_times"][rec]["duration"]
    return time_to_dt


def get_total_spike_rate(df, pons, poffs):
    """get the total spike rate during pulse-ON, and during pulse-OFF, for each cluster in each probe

    Parameters
    ----------
    df : polars dataframe
        spike dataframe
    pons : np.array
        pulse onsets
    poffs : np.array
        pulse offsets

    Returns
    -------
    on_spike_rate, off_spike_rate : polars dataframe
        spike rates for each cluster in each probe during pulse-ON and pulse-OFF
    """
    on_spike_counts = pl.DataFrame()
    total_on_time = 0
    off_spike_counts = pl.DataFrame()
    total_off_time = 0

    # First count all of the spikes and total time for pulse-ON
    for i in np.arange(0, len(pons)):
        pulse_on_count = (
            df.ts(pons[i], poffs[i]).groupby(["probe", "cluster_id"]).count()
        )
        on_spike_counts = pl.concat([on_spike_counts, pulse_on_count])
        total_on_time += (poffs[i] - pons[i]) / np.timedelta64(1, "s")

    # Then count all of the spikes and total time for pulse-OFF
    for i in np.arange(0, len(pons) - 1):
        pulse_off_count = (
            df.ts(poffs[i], pons[i + 1]).groupby(["probe", "cluster_id"]).count()
        )
        off_spike_counts = pl.concat([off_spike_counts, pulse_off_count])
        total_off_time += (pons[i + 1] - poffs[i]) / np.timedelta64(1, "s")

    # Calculate the spike rate for each cluster
    on_spike_rate = on_spike_counts.groupby(["probe", "cluster_id"]).agg(
        (pl.col("count").sum() / total_on_time).alias("pulse_on_rate")
    )
    off_spike_rate = off_spike_counts.groupby(["probe", "cluster_id"]).agg(
        (pl.col("count").sum() / total_off_time).alias("pulse_off_rate")
    )

    return on_spike_rate, off_spike_rate
