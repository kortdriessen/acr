import numpy as np
import pandas as pd
import polars as pl

import acr


def get_individual_pulse_times(subject, recording, store=None, pandas=False):
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
    if store is None:
        store = sub_info["stim-exps"][recording]
    pulse_ons = sub_info["stim_info"][recording][store]["onsets"]
    pulse_offs = sub_info["stim_info"][recording][store]["offsets"]
    pulse_ons = pd.to_datetime(pulse_ons)
    pulse_offs = pd.to_datetime(pulse_offs)
    if not pandas:
        pulse_offs = pulse_offs.to_numpy()
        pulse_ons = pulse_ons.to_numpy()
    return pulse_ons, pulse_offs


def get_pulse_train_times(pulse_ons, pulse_offs, times=False):
    """Get the index values of the pulse train onsets and offsets

    Args:
        subject (str): subject name
        recording (str): recording name
        store (str): store name under which the pulses are stored
    """
    train_ons = [0]
    train_offs = []
    if times == True:
        train_ons = [pulse_ons[0]]
    for i in np.arange(0, (len(pulse_ons) - 1)):
        diff = pulse_ons[i + 1] - pulse_ons[i]
        diff = pd.Timedelta(diff)
        if diff.total_seconds() > 5:
            if times == True:
                train_ons.append(pd.Timestamp(pulse_ons[i + 1]))
                train_offs.append(pd.Timestamp(pulse_offs[i]))
            else:
                train_ons.append(i + 1)
                train_offs.append(i)
    if times == True:
        train_offs.append(pd.Timestamp(pulse_offs[len(pulse_ons) - 1]))
    else:
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


def stim_bookends(subject, exp, store=None):
    """gives the starting and ending datetimes that a stimulation happened for a given experiment.
    Returns:
    ---------
    stim_start, stim_end : pd.Timestamp
    """

    sub_info = acr.info_pipeline.load_subject_info(subject)
    stim_store = sub_info["stim-exps"][exp]
    if (
        type(stim_store) == list
    ):  # this should be the Pu1_ store in the few cases where there are multiple stim stores
        stim_store = stim_store[0]
    if store != None:
        stim_store = store
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
        print(len(on_spike_counts["cluster_id"].unique()))

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


def add_zero_count(df1):
    assert len(df1) == 1
    probe = "NNXo" if df1["probe"][0] == "NNXr" else "NNXr"
    dur = df1["duration"][0]
    train_num = df1["train_number"][0]
    new_df = pl.DataFrame(
        {"probe": probe, "count": 0, "duration": dur, "train_number": train_num}
    )
    new_df = new_df.with_columns(pl.col("count").cast(pl.UInt32))
    new_df = new_df.with_columns(pl.col("train_number").cast(pl.Int32))
    return pl.concat([df1, new_df])


def pulse_cal_calcs(df, pons, poffs, trn_starts, trn_ends):
    """get the total spike counts during pulse-ON, and during pulse-OFF, for each probe

    Parameters
    ----------
    df : polars dataframe
        spike dataframe
    pons : np.array
        pulse onsets
    poffs : np.array
        pulse offsets
    trn_starts : np.array
        pulse train onsets
    trn_ends : np.array
        pulse train offsets

    Returns
    -------
    on_spike_rate, off_spike_rate : polars dataframe
        spike rates for each each probe during pulse-ON and pulse-OFF
    """

    # iterate through each pulse train, calculate the spike rate during pulse-ON and pulse-OFF, and express it as a ratio of the baseline spike rate
    trn_number = 0
    on_spike_counts = pl.DataFrame()
    off_spike_counts = pl.DataFrame()

    for trn_start, trn_end in zip(trn_starts, trn_ends):
        pulse_ons = pons[trn_start : trn_end + 1]
        pulse_offs = poffs[trn_start : trn_end + 1]

        off_interval = pulse_ons[1] - pulse_offs[0]
        off_duration = off_interval / np.timedelta64(1, "s")

        for onset, offset in zip(pulse_ons, pulse_offs):
            on_spike_count = df.ts(onset, offset).groupby(["probe"]).agg(pl.count())
            on_duration = (offset - onset) / np.timedelta64(1, "s")
            on_spike_count = on_spike_count.with_columns(duration=pl.lit(on_duration))
            on_spike_count = on_spike_count.with_columns(
                train_number=pl.lit(trn_number)
            )
            if len(on_spike_count) < 2:
                on_spike_count = add_zero_count(on_spike_count)
            on_spike_counts = pl.concat([on_spike_counts, on_spike_count])

            off_spike_count = (
                df.ts(offset, offset + off_interval).groupby(["probe"]).count()
            )
            off_spike_count = off_spike_count.with_columns(
                duration=pl.lit(off_duration)
            )
            off_spike_count = off_spike_count.with_columns(
                train_number=pl.lit(trn_number)
            )
            if len(off_spike_count) < 2:
                off_spike_count = add_zero_count(off_spike_count)
            off_spike_counts = pl.concat([off_spike_counts, off_spike_count])
        trn_number += 1

    on_spike_counts = on_spike_counts.to_pandas()
    off_spike_counts = off_spike_counts.to_pandas()

    on_spike_counts = (
        on_spike_counts.groupby(["probe", "train_number"]).sum().reset_index()
    )
    off_spike_counts = (
        off_spike_counts.groupby(["probe", "train_number"]).sum().reset_index()
    )

    on_spike_counts["fr"] = on_spike_counts["count"] / on_spike_counts["duration"]
    off_spike_counts["fr"] = off_spike_counts["count"] / off_spike_counts["duration"]

    return on_spike_counts, off_spike_counts


def sincal_calculation(df, pons, poffs):
    """get the total spike counts during pulse-ON, and during pulse-OFF, for each probe

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
    # iterate through each pulse train, calculate the spike rate during pulse-ON and pulse-OFF, and express it as a ratio of the baseline spike rate
    trn_number = 0

    on_spike_counts = pl.DataFrame()
    off_spike_counts = pl.DataFrame()

    for onset, offset in zip(pons, poffs):
        on_spike_count = df.ts(onset, offset).groupby(["probe"]).agg(pl.count())
        on_duration = (offset - onset) / np.timedelta64(1, "s")
        on_spike_count = on_spike_count.with_columns(duration=pl.lit(on_duration))
        on_spike_count = on_spike_count.with_columns(train_number=pl.lit(trn_number))
        if len(on_spike_count) < 2:
            on_spike_count = add_zero_count(on_spike_count)
        on_spike_counts = pl.concat([on_spike_counts, on_spike_count])
        trn_number += 1

    return on_spike_counts.to_pandas()


def clus_check(subject, exp, probe, clus):
    sid = f"{exp}-{probe}"
    ex = acr.pl_units.get_units_to_exclude(subject, sid)
    if ex == None:
        return True
    else:
        if clus in ex:
            return False


def get_all_stim_info(subject, exp, stim_store=None, trn_idx=False):
    """Gets the stim start, stim_end, and pulse train times for a given subject and experiment.
    returns:
    --------
    stim_start, stim_end, pon, poff : pd.Timestamp"""
    stim_start, stim_end = acr.stim.stim_bookends(subject, exp)
    pon, poff = get_individual_pulse_times(subject, exp, store=stim_store)
    if trn_idx:
        ton, toff = acr.stim.get_pulse_train_times(pon, poff, times=False)
        return stim_start, stim_end, pon, poff, ton, toff
    else:
        ton, toff = acr.stim.get_pulse_train_times(pon, poff)
        return (
            stim_start,
            stim_end,
            pon,
            poff,
            pd.to_datetime(ton),
            pd.to_datetime(toff),
        )


def assign_train_times_to_frdf(tons, toffs, frdf, pretrain=True):
    frdf = frdf.with_columns(stim_train=pl.lit(None))
    ptrain = 1
    for ton, toff in zip(tons, toffs):
        frdf = frdf.with_columns(
            pl.when(pl.col("datetime").is_between(ton, toff))
            .then(ptrain)
            .otherwise(pl.col("stim_train"))
            .alias("stim_train")
        )
        ptrain += 1
    if pretrain == True:
        ptrain_time = tons[0] - pd.Timedelta("1h")
        frdf = frdf.with_columns(
            pl.when(
                ((pl.col("datetime") < tons[0]) & (pl.col("datetime") > ptrain_time))
            )
            .then(0)
            .otherwise(pl.col("stim_train"))
            .alias("stim_train")
        )
    return frdf


def add_stim_times_to_df(df, subject, exp, pre_stim="1h"):
    """Will Add stim train times to any dataframe with a datetime column

    Parameters
    ----------
    df : _type_
        _description_
    subject : _type_
        _description_
    exp : _type_
        _description_
    """
    return_pl = False
    if type(df) == pl.DataFrame:
        df = df.to_pandas()
        return_pl = True
    pon, poff = acr.stim.get_individual_pulse_times(subject, exp)
    ton, toff = acr.stim.get_pulse_train_times(pon, poff, times=True)
    df["stim"] = "None"
    df.loc[
        (df["datetime"] < pon[0])
        & (df["datetime"] > (pon[0] - pd.to_timedelta(pre_stim))),
        "stim",
    ] = "pre-stim"
    for i, (t1, t2) in enumerate(zip(ton, toff)):
        df.loc[(df["datetime"] >= t1) & (df["datetime"] <= t2), "stim"] = f"train-{i}"
    return df if return_pl == False else pl.DataFrame(df)


def compute_fr_decline_during_stim_by_probe(mua, ton, toff, tbefore="60min"):
    t1 = ton[0] - pd.Timedelta(tbefore)
    t2 = ton[0]
    pre_counts = mua.ts(t1, t2).group_by("probe").count().to_pandas()
    stim_dfs = pd.DataFrame()
    for i, (on, off) in enumerate(zip(ton, toff)):
        t1 = pd.Timestamp(on)
        t2 = pd.Timestamp(off)
        stim_df = mua.ts(t1, t2).group_by("probe").count().to_pandas()
        stim_df["duration"] = (t2 - t1).total_seconds()
        stim_df["stim_bout"] = f"stim-{i}"
        stim_dfs = pd.concat([stim_dfs, stim_df])
    for probe in pre_counts["probe"].unique():
        bl_rate = pre_counts.loc[pre_counts["probe"] == probe]["count"].values[0] / 3600
        stim_dfs.loc[stim_dfs["probe"] == probe, "bl_rate"] = bl_rate
    stim_dfs["rate"] = stim_dfs["count"] / stim_dfs["duration"]
    stim_dfs["decline"] = (stim_dfs["rate"] - stim_dfs["bl_rate"]) / stim_dfs["bl_rate"]
    stim_dfs["rel_rate"] = stim_dfs["rate"] / stim_dfs["bl_rate"]
    nnxo_rel_rate = stim_dfs.prb("NNXo").sort_values("stim_bout")["rel_rate"].values
    nnxr_rel_rate = stim_dfs.prb("NNXr").sort_values("stim_bout")["rel_rate"].values
    rel_decline = stim_dfs.prb("NNXo").sort_values("stim_bout")
    rel_decline["rel_to_cc"] = nnxo_rel_rate / nnxr_rel_rate
    return stim_dfs, rel_decline
