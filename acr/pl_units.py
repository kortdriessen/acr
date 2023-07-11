import polars as pl
import pandas as pd
import acr
from acr.units import *
import seaborn as sns
import matplotlib.pyplot as plt
import kdephys.plot.main as kp


def load_spikes_polars(
    subject, sort_id, cols2drop=None, info=True, exclude_bad_units=True
):
    """load polars spike dataframe from parquet files, eager mode

    Parameters
    ----------
    subject : str
        subject name
    sort_id : list
        list of sort ids
    cols2drop : list, optional
        columns to drop, if default is used then many columns are dropped; DEFAULT = None
    info : bool, optional
        whether or not to load info_dataframe; DEFAULT = True
    exclude_bad_units : bool, optional
        whether or not to exclude bad units; DEFAULT = True

    Returns
    -------
    spike_df : polars dataframe
    info_df: pandas dataframe
        spike and info dataframes
    """
    if cols2drop == None:
        cols2drop = [
            "group",
            "note",
            "channel",
            "exp",
            "recording",
            "state",
            "amp",
            "Amplitude",
        ]
    if cols2drop == 0:
        cols2drop = []
    path = f"/Volumes/opto_loc/Data/{subject}/sorting_data/spike_dataframes/"
    sdfs = []
    for sid in sort_id:
        key = sid + ".parquet"
        polars_df = pl.read_parquet(path + key)
        if exclude_bad_units == True:
            units2ex = get_units_to_exclude(subject, sid)
            polars_df = polars_df.filter(~pl.col("cluster_id").is_in(units2ex))
        sdfs.append(polars_df)
    spike_df = pl.concat(sdfs)
    for col in cols2drop:
        if col in spike_df.columns:
            spike_df = spike_df.drop(col)
    if info == True:
        idf = load_info_df(subject, sort_id, exclude_bad_units=exclude_bad_units)
        return spike_df, idf
    else:
        return spike_df


def load_spikes_polars_lazy(subject, sort_id, cols2drop=None):
    """load polars spike dataframe from parquet files, lazy-style (scan_parquet)

    Parameters
    ----------
    subject : str
        subject name
    sort_id : list
        list of sort ids
    cols2drop : list, optional
        columns to drop from saved dataframe, if default is used, many columns are removed; DEFAULT = None

    Returns
    -------
    spike_df : polars dataframe
        spike dataframe with spike times, clusters, and probes
    """
    if cols2drop == None:
        cols2drop = [
            "group",
            "note",
            "channel",
            "exp",
            "recording",
            "state",
            "amp",
            "Amplitude",
        ]
    path = f"/Volumes/opto_loc/Data/{subject}/sorting_data/spike_dataframes/"
    sdfs = []
    for sid in sort_id:
        key = sid + ".parquet"
        polars_df = pl.scan_parquet(path + key)
        sdfs.append(polars_df)
    spike_df = pl.concat(sdfs)
    spike_df = spike_df.drop(cols2drop)
    return spike_df


def bout_duration_similarity_check(df, col="bout_duration"):
    """checks whether all values in the 'bout_duration' column are the same. If not, it checks for how different they are and if the difference is very small, it rectifies them so all values are the same.

    Parameters
    ----------
    df : pl.DataFrame
        polars dataframe with 'bout_duration' column
    """
    if all(
        df[col] == df[col][0]
    ):  # if all values are already equal, just return the dataframe
        return df
    else:
        mask = (
            df[col] == df[col][0]
        )  # if all vals not equal, check first how different they are; if the diff is small, rectify.
        for dur in df[
            col
        ]:  # this is a check that the difference between the bout_durations is very small.
            if dur != df[col][0]:
                diff = abs(dur - df[col][0])
                if diff < 1e-7:
                    continue
                else:
                    raise ValueError(
                        f"difference between bouts is too large! Diff = {diff}"
                    )
        new_val = df[col][
            0
        ]  # given that the difference between bouts is very small, we can just set all bouts to the first bout duration
        df = df.with_columns(
            pl.when(~mask).then(new_val).otherwise(pl.col(col)).alias(col)
        )
        assert all(df[col] == df[col][0])  # check that all values are now equal
        return df


def get_state_fr(df, hyp, t1=None, t2=None, state="NREM", min_duration=30):
    """gets the firing rate for each cluster during a specified state.

    Args:
    ----------
        - df (dataframe): spike dataframe
        - hyp (dataframe): hypnogram
        - t1 (datetime, optional): if not specified, the first 12 hours from the start of the first spike are used. Defaults to None.
        - t2 (datetime, optional): if not specified, the first 12 hours from the start of the first spike are used. Defaults to None.
        - state (str, optional): state to use. Defaults to 'NREM'.

    Returns:
    -------------------------
        fr_master: dataframe that has the number of spikes and total duration of each bout for each cluster. These can then be summed to get the firing rate across all bouts.
    """
    if t1 is None and t2 is None:
        t1 = df["datetime"].to_pandas().min()
        t2 = t1 + pd.Timedelta("12h")

    hyp_state = hyp.loc[hyp.state == state]
    hyp_state = hyp_state.loc[hyp_state.end_time < t2]

    fr_master = pl.DataFrame()

    for bout in hyp_state.itertuples():
        start = bout.start_time
        end = bout.end_time
        bout_duration = (end - start).total_seconds()
        if bout_duration < min_duration:
            continue
        spikes = df.ts(start, end).groupby(["probe", "cluster_id"]).count()
        fr_bout = spikes.with_columns(pl.lit(bout_duration).alias("bout_duration"))
        fr_master = pl.concat([fr_master, fr_bout])

    return fr_master


def get_state_firing_rates(df, hyp):
    """get the firring rates for wake, nrem, and rem, for each bout of each covered by the hypnogram.
    **NOTE** this function loops through every single bout of each of the three states for the entire time covered by df (max_datetime - min_datetime).
    If this is not desired, trim the hypnogram to the desired time period before passing it to this function. The hypno will be passed to get_state_fr, which just loops through the bouts of a given state.
    The default is set this way to maximize the amount of data used, but it is not always desirable.

    Args:
    -----------
        df (pl.dataframe): polars spike dataframe
        hyp (dataframe): hypnogram dataframe

    Returns:
    ----------------
        wake_frs, nrem_frs, rem_frs : polars dataframes with the firing rates for each cluster in each bout of each state
    """
    min_t = df["datetime"].to_pandas().min()
    max_t = df["datetime"].to_pandas().max()
    _total_t = int((max_t - min_t).total_seconds())
    total_t = pd.Timedelta(f"{_total_t}s")
    t1 = min_t
    t2 = min_t + total_t

    rem_frs = get_state_fr(df, hyp, t1, t2, state="REM")
    wake_frs = get_state_fr(df, hyp, t1, t2, state="Wake")
    nrem_frs = get_state_fr(df, hyp, t1, t2, state="NREM")

    wake_frs = wake_frs.with_columns(pl.lit("wake").alias("state"))
    rem_frs = rem_frs.with_columns(pl.lit("rem").alias("state"))
    nrem_frs = nrem_frs.with_columns(pl.lit("nrem").alias("state"))

    return wake_frs, nrem_frs, rem_frs


def get_state_specific_unit_df(df, hyp, state):
    hyp = hyp.st(state)
    final_df = pl.DataFrame()
    for bout in hyp.itertuples():
        bout_df = df.ts(bout.start_time, bout.end_time)
        final_df = pl.concat([final_df, bout_df])
    return final_df


def get_rel_fr_df(
    df,
    hyp,
    rel_state="NREM",
    window="120s",
    by="probe",
    t1=None,
    t2=None,
    over_bouts=False,
    return_early=False,
):
    """gets the firing rate (either by probe or cluster) relative to the baseline firing rate in a specified state

    ***NOTE*** the baseline is defined as 9am-9pm on the day of the first spike in df, unless t1 and t2 are specified!

    Args:
        df (_type_): polars spike dataframe
        hyp (_type_): hypnogram
        rel_state (str, optional): state to get the reference firing rates from. Defaults to 'NREM'.
        window (str, optional): window size for polars groupby_dynamic. Defaults to '120s'.
        by (str, optional): get relative firing rate by cluster or by probe. Defaults to 'probe'.
        t1 (_type_, optional): If not provided, 9am on the day of the first spike in df is used. Defaults to None.
        t2 (_type_, optional): if not provided, 9pm on the day of the first spike in df is used. Defaults to None.

    Returns:
        fr_rel: relative firing rate dataframe, with columns: probe, datetime, fr_rel, and cluster_id if by='cluster'
    """
    bl_date = str(df["datetime"].to_pandas().min()).split(" ")[0]
    bl_start = pd.Timestamp(f"{bl_date} 09:00:00")
    if t1 == None and t2 == None:
        t1 = bl_start
        t2 = bl_start + pd.Timedelta("12h")
    bl_frs = get_state_fr(df, hyp, t1=t1, t2=t2, state=rel_state)
    window_time = int(window.strip("s"))
    if by == "probe":
        if over_bouts == True:
            bl_frs_by_probe = bl_frs.frates().groupby(["probe", "cluster_id"]).mean()
        else:
            bl_frs_by_probe = bl_frs.groupby(["probe", "cluster_id"]).sum()
            if return_early == True:
                return bl_frs_by_probe
            bl_frs_by_probe = bout_duration_similarity_check(
                bl_frs_by_probe, col="bout_duration"
            )
            bl_frs_by_probe = bl_frs_by_probe.groupby(["probe", "bout_duration"]).sum()
            bl_frs_by_probe = bl_frs_by_probe.with_columns(
                (pl.col("count") / pl.col("bout_duration")).alias("fr")
            ).drop("count", "bout_duration")
        fr_window = df.groupby_dynamic(
            "datetime", every=window, start_by="datapoint", closed="left", by="probe"
        ).agg(pl.col("cluster_id").count())
        fr_raw = fr_window.with_columns(
            (pl.col("cluster_id") / window_time).alias("fr")
        ).drop("cluster_id")
        fr_rel = fr_raw.with_columns(
            pl.when(pl.col("probe") == "NNXr")
            .then(pl.col("fr") / bl_frs_by_probe.prb("NNXr")["fr"][0])
            .otherwise(pl.col("fr") / bl_frs_by_probe.prb("NNXo")["fr"][0])
            .alias("fr_rel")
        )
        return fr_rel.drop("fr")
    elif by == "cluster":
        bl_frs_by_cluster = bl_frs.groupby(["probe", "cluster_id"]).sum()
        bl_frs_by_cluster = bl_frs_by_cluster.with_columns(
            (pl.col("count") / pl.col("bout_duration")).alias("fr")
        ).drop("count", "bout_duration")
        if return_early == True:
            return bl_frs_by_cluster
        fr_window = df.groupby_dynamic(
            "datetime",
            every=window,
            start_by="datapoint",
            closed="left",
            by=["probe", "cluster_id"],
        ).agg(pl.col("cluster_id").count().alias("count"))
        fr_rel = fr_window.with_columns(
            (pl.col("count") / window_time).alias("fr_rel")
        ).drop(
            "count"
        )  # not really relative yet
        for probe in fr_rel["probe"].unique():
            for cluster_id in fr_rel.prb(probe).cid_un():
                fr_rel = fr_rel.with_columns(
                    pl.when(
                        (pl.col("probe") == probe)
                        & (pl.col("cluster_id") == cluster_id)
                    )
                    .then(
                        pl.col("fr_rel")
                        / bl_frs_by_cluster.pclus(probe, cluster_id)["fr"][0]
                    )
                    .otherwise(pl.col("fr_rel"))
                )
        return fr_rel


def fr_arbitrary_bout(df, t1, t2, by="cluster_id"):
    """Calculate the total number of spikes and total duration of an arbitrary period of time (t1 to t2), by cluster or probe.

    Parameters
    ----------
    df : pl.Dataframe
        spike dataframe
    t1 : datetime
        start time
    t2 : datetime
        end time
    by : str, optional
        calculate firing rate by cluster_id or by probe; DEFAULT = 'cluster_id'

    Returns
    -------
    fr_bout : pl.Dataframe
        dataframe with firing rate for each cluster or probe over the specified time period
    """
    bout_duration = (t2 - t1).total_seconds()

    if by == "cluster_id":
        spikes = df.ts(t1, t2).groupby(["probe", "cluster_id"]).count()
        fr_bout = spikes.with_columns((pl.lit(bout_duration)).alias("bout_duration"))
    elif by == "probe":
        spikes = df.ts(t1, t2).groupby(["probe"]).count()
        fr_bout = spikes.with_columns((pl.lit(bout_duration)).alias("bout_duration"))
    else:
        raise ValueError("by parameter must be either 'cluster_id' or 'probe'")

    return fr_bout


def time_zones_to_unit_df(df, t1, t2, label):
    assert type(df) == pl.DataFrame
    df = df.to_pandas()
    df.loc[(df.datetime >= t1) & (df.datetime < t2), "time_zone"] = label
    return pl.from_pandas(df)


### ---------------------------------------------- Specialized Plotting Functions --------------------------------------------------- ###


def plot_fr_by_probe(fr_rel, hyp, ax=None, color=False):
    ax = kp.check_ax(ax)
    if color == False:
        sns.lineplot(data=fr_rel, x="datetime", y="fr_rel", hue="probe", ax=ax)
    else:
        sns.lineplot(
            data=fr_rel,
            x="datetime",
            y="fr_rel",
            hue="probe",
            palette=["blue", "black"],
            ax=ax,
        )
    ax = kp.shade_hypno_for_me(hyp, ax)
    ax = kp.add_light_schedule(fr_rel.light_schedule(), ax=ax)
    return ax


def plot_fr_by_cluster(fr_rel, hyp):
    for probe in fr_rel.prbs():
        color = "blue" if probe == "NNXo" else "black"
        for cluster_id in fr_rel.prb(probe).cid_un():
            df2plot = fr_rel.pclus(probe, cluster_id)
            f, ax = plt.subplots()
            ax = sns.lineplot(
                data=df2plot, x="datetime", y="fr_rel", ax=ax, color=color
            )
            kp.shade_hypno_for_me(hyp, ax)
            ax.set_title(f"{probe} | Cluster = {cluster_id}")
            kp.add_light_schedule(fr_rel.light_schedule(), ax=ax)
    return


def plot_state_fr_distributions(wake_frs, nrem_frs, rem_frs):
    for probe in wake_frs.prbs():
        for cluster in wake_frs.prb(probe).cid_un():
            wake_clus = wake_frs.pclus(probe, cluster)
            rem_clus = rem_frs.pclus(probe, cluster)
            nrem_clus = nrem_frs.pclus(probe, cluster)

            wake_fr = wake_clus.frate()["fr"][0]
            rem_fr = rem_clus.frate()["fr"][0]
            nrem_fr = nrem_clus.frate()["fr"][0]
            wake_rem_ratio = wake_fr / rem_fr
            wake_nrem_ratio = wake_fr / nrem_fr
            nrem_rem_ratio = nrem_fr / rem_fr

            wake_clus = wake_clus.filter(pl.col("bout_duration") > 15).frates()
            rem_clus = rem_clus.filter(pl.col("bout_duration") > 10).frates()
            nrem_clus = nrem_clus.filter(pl.col("bout_duration") > 15).frates()
            df2plot = pl.concat([wake_clus, rem_clus, nrem_clus])
            f, ax = plt.subplots()
            ax = sns.histplot(
                data=df2plot,
                x="fr",
                hue="state",
                palette=["lightgreen", "magenta", "cornflowerblue"],
                binwidth=0.25,
                kde=True,
                ax=ax,
                stat="density",
            )
            ax.set_title(
                f"{probe} Cluster = {cluster} | Wake/REM = {wake_rem_ratio:.2f} | Wake/NREM = {wake_nrem_ratio:.2f} | NREM/REM = {nrem_rem_ratio:.2f}"
            )
    return
