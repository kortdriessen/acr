import pandas as pd
import yaml
import acr
import tdt
import numpy as np


def add_time_class(df, times):
    if "control1" in df.condition[0]:
        stim_on = times["control1"]["stim_on_dt"]
        stim_off = times["control1"]["stim_off_dt"]
    elif "laser1" in df.condition[0]:
        stim_on = times["laser1"]["stim_on_dt"]
        stim_off = times["laser1"]["stim_off_dt"]
    df["time_class"] = "NA"
    df.loc[df.datetime.between(df.datetime.min(), stim_on), "time_class"] = "Baseline"
    stim_mid = stim_on + pd.Timedelta("2H")
    df.loc[df.datetime.between(stim_on, stim_mid), "time_class"] = "Photostim, 0-2Hr"
    df.loc[df.datetime.between(stim_mid, stim_off), "time_class"] = "Photostim, 2-4Hr"
    c1 = stim_off + pd.Timedelta("2H")
    c2 = stim_off + pd.Timedelta("4H")
    c3 = stim_off + pd.Timedelta("6H")
    df.loc[df.datetime.between(stim_off, c1), "time_class"] = "Post Stim, 0-2Hr"
    df.loc[df.datetime.between(c1, c2), "time_class"] = "Post Stim, 2-4Hr"
    df.loc[df.datetime.between(c2, c3), "time_class"] = "Post Stim, 4-6Hr"
    return df


def gen_config(
    subject="",
    chunks=6,
    chunk_length=7200,
    start_from=0,
    exp="",
    chans=["EMGr-1", "EEG_-1", "EEG_-2", "LFP_-2", "LFP_-6", "LFP_-10", "LFP_-14"],
):
    template_name = "ACR_X_sleepscore-config_experiment-chunk.yml"
    config_directory = (
        f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/" + subject + "/config-files/"
    )
    template_path = config_directory + template_name
    data_path = f"L:\Data\{subject}\{subject}-{exp}"
    for chunk in range(1, chunks + 1):
        # First load the template file
        with open(template_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        # set the binPath
        data["datasets"][0]["binPath"] = data_path
        # Set the channels
        data["datasets"][0]["chanList"] = chans
        # Set the times
        tend = (chunk * chunk_length) + start_from
        data["tStart"] = tend - chunk_length
        data["tEnd"] = tend
        # Save the file with a new name based on the actual subject, experiment, and chunk
        config_path = (
            config_directory + f"{subject}_sleepscore-config_{exp}-chunk{chunk}.yml"
        )
        with open(config_path, "w") as f:
            yaml.dump(data, f)


def tdt_to_dt(info, data, time_key, slc=True):
    """converts tdt time to datetime"""

    start = data.datetime.values.min()
    tmin = data.time.values.min()
    dt_start = pd.to_timedelta((info["times"][time_key][0] - tmin), unit="s") + start
    dt_end = pd.to_timedelta((info["times"][time_key][1] - tmin), unit="s") + start
    if slc:
        return slice(dt_start, dt_end)
    else:
        return (dt_start, dt_end)


def get_rec_times(si):
    sub = si["subject"]
    times = {}
    for exp in si["complete_key_list"]:
        p = acr.io.acr_path(sub, exp)
        d = tdt.read_block(p, t1=0, t2=1, evtype=["scalars"])
        i = d.info
        start = np.datetime64(i.start_date)
        end = np.datetime64(i.stop_date)
        d = (end - start) / np.timedelta64(1, "s")
        times[exp] = (start, end, d)
    return times
