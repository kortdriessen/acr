import acr
import numpy as np
import pandas as pd
import polars as pl
import os


def on_off_detection_basic(df, off_period_min=0.05):
    """Basic on/off detection algorithm.  Detects on/off periods based on a minimum period of complete neural silence.
    This is a very basic algorithm that just looks at differences between adjacent spike times in a dataframe.

    Parameters
    ----------
    df : polars.DataFrame
        polars spike dataframe, must have 'time' and 'probe' columns
    off_period_min : float, optional
        minimum duration for an off period. Any gap of at least this long without any spikes will be defined as an OFF-period; DEFAULT = .05

    Returns
    -------
    oodf: pd.DataFrame
        Final dataframe covering all time points in the spike dataframe, with ON and OFF periods labeled.
    """
    oodf = pd.DataFrame()
    for probe in df.prbs():
        times = df.prb(probe)["time"].to_numpy()
        times_diff = np.diff(times)
        # gets us the off-detection information
        msk = np.ma.masked_where(times_diff >= off_period_min, times_diff)
        start_times_mask = msk.mask
        start_times_mask = np.append(start_times_mask, False)
        start_times_indices = np.where(start_times_mask == True)[0]
        end_times_indices = start_times_indices + 1
        start_times = times[start_times_indices]
        end_times = times[end_times_indices]
        off_det = pd.DataFrame(
            {
                "start_time": start_times,
                "end_time": end_times,
                "status": "off",
                "probe": probe,
            }
        )
        # gets us the on-detection information
        on_starts = []
        on_ends = []
        for i in np.arange(0, len(start_times) + 1):
            if i == 0:
                on_starts.append(0)
                on_ends.append(start_times[i])
            elif i == len(start_times):
                on_starts.append(end_times[i - 1])
                on_ends.append(times[-1])
            else:
                on_starts.append(end_times[i - 1])
                on_ends.append(start_times[i])
        on_starts = np.array(on_starts)
        on_ends = np.array(on_ends)
        on_det = pd.DataFrame(
            {
                "start_time": on_starts,
                "end_time": on_ends,
                "status": "on",
                "probe": probe,
            }
        )

        # final dataframe
        oo_new = (
            pd.concat([on_det, off_det])
            .sort_values(by="start_time")
            .reset_index(drop=True)
        )
        oo_new["duration"] = oo_new["end_time"] - oo_new["start_time"]
        oodf = pd.concat([oodf, oo_new])
    return oodf


def save_oodf(oodf, subject, oodf_id):
    """save an on-off dataframe to the subject's on-off_dataframes folder

    Parameters
    ----------
    oodf : pd.DataFrame
        on-off dataframe
    subject : str
        subject name
    oodf_id : str
        name for the dataframe; typically, this should be named as "{exp-name}-oodf" e.g. "swi-oodf"
    """
    # First we make sure the on-off_dataframes folder exists
    sort_root = f"/Volumes/opto_loc/Data/{subject}/sorting_data/"
    if "on-off_dataframes" not in os.listdir(sort_root):
        os.mkdir(sort_root + "on-off_dataframes")
    # Then we save the dataframe
    path = f"/Volumes/opto_loc/Data/{subject}/sorting_data/on-off_dataframes/{oodf_id}.parquet"
    oodf.to_parquet(path, version="2.6")
    return


def load_oodf(subject, oodf_id):
    """load an on-off dataframe from the subject's on-off_dataframes folder

    Parameters
    ----------
    subject : str
        subject name
    oodf_id : str
        name for the dataframe; typically, this should be named as "{exp-name}-oodf" e.g. "swi-oodf"

    Returns
    -------
    oodf : pd.DataFrame
        on-off dataframe
    """
    path = f"/Volumes/opto_loc/Data/{subject}/sorting_data/on-off_dataframes/{oodf_id}.parquet"
    oodf = pl.read_parquet(path)
    return oodf


def add_datetime_to_oodf(oodf, subject, exp):
    probe = oodf.probe.unique()[0]
    sort_id = f"{exp}-{probe}"
    recs, starts, durations = acr.units.get_time_info(subject, sort_id)
    start_times = oodf.start_time.values
    end_times = oodf.end_time.values

    zero_time = 0
    for i in range(len(starts)):
        rec_start_dt = starts[i]
        rec_start_times = start_times[
            np.logical_and(
                start_times >= zero_time, start_times < zero_time + durations[i]
            )
        ]
        rec_end_times = end_times[
            np.logical_and(end_times >= zero_time, end_times < zero_time + durations[i])
        ]
        rec_starts_rel = rec_start_times - zero_time
        rec_ends_rel = rec_end_times - zero_time
        rec_start_timedeltas = pd.to_timedelta(rec_starts_rel, unit="s")
        rec_end_timedeltas = pd.to_timedelta(rec_ends_rel, unit="s")
        rec_start_datetimes = rec_start_dt + rec_start_timedeltas
        rec_end_datetimes = rec_start_dt + rec_end_timedeltas
        oodf.loc[
            np.logical_and(
                start_times >= zero_time, start_times < zero_time + durations[i]
            ),
            "start_datetime",
        ] = rec_start_datetimes
        oodf.loc[
            np.logical_and(
                end_times >= zero_time, end_times < zero_time + durations[i]
            ),
            "end_datetime",
        ] = rec_end_datetimes
        zero_time += durations[i]
    return oodf


def states_to_oodf(oodf, hyp):
    for bout in hyp.itertuples():
        t1 = bout.start_time
        t2 = bout.end_time
        oodf.loc[
            (oodf.start_datetime > t1) & (oodf.end_datetime < t2), "state"
        ] = bout.state
    return oodf


def oodf_durations_rel2bl(oodf, tz_name="baseline", state="NREM"):
    oodf = oodf.with_columns(pl.lit(None).alias("rel_duration"))
    avgs = oodf.tz(tz_name).st(state).groupby(["probe", "status"]).mean()
    for probe in oodf.prbs():
        oodf = oodf.with_columns(
            pl.when((pl.col("probe") == probe) & (pl.col("status") == "on"))
            .then(
                (pl.col("duration") / avgs.prb(probe).ons()["duration"][0]).alias(
                    "rel_duration"
                )
            )
            .otherwise(pl.col("rel_duration"))
        )
        oodf = oodf.with_columns(
            pl.when((pl.col("probe") == probe) & (pl.col("status") == "off"))
            .then(
                (pl.col("duration") / avgs.prb(probe).offs()["duration"][0]).alias(
                    "rel_duration"
                )
            )
            .otherwise(pl.col("rel_duration"))
        )
    return oodf


def time_zones_to_oodf(oodf, t1, t2, label):
    if type(oodf) == pd.core.frame.DataFrame:
        oodf.loc[
            (oodf.start_datetime > t1) & (oodf.end_datetime < t2), "time_zone"
        ] = label
        return oodf
    else:
        oodf = oodf.to_pandas()
        oodf.loc[
            (oodf.start_datetime > t1) & (oodf.end_datetime < t2), "time_zone"
        ] = label
        return pl.from_pandas(oodf)
