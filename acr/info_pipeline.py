import numpy as np
import pandas as pd
import tdt
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import xarray as xr
import yaml

import kdephys.xr as kx

import acr

# plt.style.use(Path("acr_plots.mplstyle"))

import os
from itertools import cycle


def load_subject_info(subject):
    path = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/subject_info.yml"
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def get_all_tank_keys(root, sub):
    tanks = []
    for f in os.listdir(root):
        full_path = os.path.join(root, f)
        if np.logical_and(os.path.isdir(full_path), sub in f):
            full_path = Path(full_path)
            tanks.append(full_path)
    tank_keys = []
    for tp in tanks:
        tp = os.path.split(tp)[1]
        tps = tp.split("-")
        tps = tps[1:]
        tps = "-".join(tps)
        tank_keys.append(tps)
    return tank_keys


def get_time_full_load(path, store):
    data = tdt.read_block(path, evtype=["streams"], channel=[1], store=store)
    start = np.datetime64(data.info.start_date)
    num_samples = len(data.streams[store].data)
    fs = data.streams[store].fs
    total_time = float(num_samples / fs)
    end = start + pd.Timedelta(total_time, unit="s")
    times = {}
    times["start"] = str(start)
    times["duration"] = total_time
    times["end"] = str(end)
    return times


def time_sanity_check(time):
    return time > np.datetime64("2018-01-01")


def get_rec_times(sub, exps, time_stores=["NNXo", "NNXr"], backup_store=["EMGr"]):
    """Gets durations, starts, and ends of recording times for set of experiments.

    Args:
        sub (str) : subject name
        exps (list): experiment names
        time_stores (list, optional): stores to use to calculate start and end times. Defaults to ['NNXo', 'NNXr'].

    Returns:
        times: dictionary, keys are experiment names,
    """
    times = {}
    for exp in exps:
        p = acr.io.acr_path(sub, exp)
        d = tdt.read_block(p, t1=0, t2=1, evtype=["streams"])
        streams = list(d.streams.keys())
        print(f"streams for {exp} are {streams}")
        i = d.info
        start = np.datetime64(i.start_date)
        end = np.datetime64(i.stop_date)
        block_duration = float((end - start) / np.timedelta64(1, "s"))
        new_t1 = int(block_duration - 30)

        if not time_sanity_check(end):
            times[exp] = get_time_full_load(p, backup_store[0])
            continue

        times[exp] = {}
        times[exp]["start"] = str(start)

        if all([s in streams for s in time_stores]):
            time_stores_to_use = time_stores
        else:
            time_stores_to_use = backup_store
        print(f"using {time_stores_to_use} to get end times for {exp}")
        data = tdt.read_block(p, store=time_stores_to_use, channel=[1], t1=new_t1, t2=0)

        for ts in time_stores_to_use:
            num_samples = len(data.streams[ts].data)
            fs = data.streams[ts].fs
            samples_before = new_t1 * fs
            total_samples = int(np.ceil(num_samples + samples_before))
            total_time = float(total_samples / fs)
            times[exp][ts + "-duration"] = total_time
            times[exp][ts + "-end"] = str(start + pd.Timedelta(total_time, unit="s"))
    return times


def get_rec_times_deprecated(sub, exps):
    times = {}
    for exp in exps:
        p = acr.io.acr_path(sub, exp)
        d = tdt.read_block(p, t1=0, t2=1, evtype=["scalars"])
        i = d.info
        start = np.datetime64(i.start_date)
        end = np.datetime64(i.stop_date)
        d = (end - start) / np.timedelta64(1, "s")
        times[exp] = [str(start.astype(str)), str(end.astype(str)), float(d)]
        # TODO add something to correct the end times that are happening in 1969 for some reason
    return times


def rec_time_comparitor(subject):
    """
    Looks the rec_times field of the subject_info file, and where there are competing ends or competing durations
    (because the time_stores had a different number of samples),
    it chooses the shortest of each and assigns that to the general 'end' and 'duration' fields.

    Args:
        subject (str): subject name

    """
    path = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/subject_info.yml"
    with open(path, "r") as f:
        info = yaml.safe_load(f)
    recs = info["recordings"]
    for rec in recs:
        durations = []
        ends = []
        keys = list(info["rec_times"][rec].keys())
        for key in keys:
            if "-duration" in key:
                durations.append(info["rec_times"][rec][key])
            if "-end" in key:
                ends.append(info["rec_times"][rec][key])
        if np.logical_and(len(durations) == 1, len(ends) == 1):
            info["rec_times"][rec]["duration"] = durations[0]
            info["rec_times"][rec]["end"] = ends[0]
        if np.logical_and(len(durations) > 1, len(ends) > 1):
            info["rec_times"][rec]["duration"] = min(durations)
            end_dts = [np.datetime64(end) for end in ends]
            info["rec_times"][rec]["end"] = str(min(end_dts))
        else:
            continue

    with open(path, "w") as f:
        yaml.dump(info, f)


def subject_info_gen(params):
    """Params is a dictionary with the following keys:
    subject: str
    raw_stores: list of important data stores, to be used in preprocessing of important recordings (all channels from all raw stores)
    lite_stores: list of less important data stores, to heavily downsample and save only a subset of channels
    channels: a dictionary whose keys are values from stores, and values are the channels to use for each store
    preprocess-list: list of important recordings that need to be downsampled/processed. Should be a subset of all available recordings
    stim-exps: dictionary where the keys are stim experiments, and the values are stores to get the onsets/offsets from (e.g. 'Wav2' or 'Pu1_')
    time_stores: list of stores to use to calculate start and end times. Defaults to ['NNXo', 'NNXr'].
    """

    subject = params["subject"]
    channels = params["channels"]
    ds_list = params["preprocess-list"]
    path = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/subject_info.yml"
    root = f"/Volumes/opto_loc/Data/{subject}/"
    recordings = get_all_tank_keys(root, subject)

    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    data = {}
    data["subject"] = subject
    times = get_rec_times(
        subject,
        recordings,
        time_stores=params["time_stores"],
        backup_store=[params["lite_stores"][0]],
    )

    data["rec_times"] = times

    data["channels"] = channels
    data["paths"] = acr.io.get_acr_paths(subject, recordings)
    data["raw_stores"] = params["raw_stores"]
    data["lite_stores"] = params["lite_stores"]
    data["recordings"] = recordings
    data["preprocess-list"] = ds_list
    data["stim-exps"] = params["stim-exps"]
    with open(path, "w") as f:
        yaml.dump(data, f)


def preprocess_and_save_exp(subject, rec, fs_target=400, t1=0, t2=0):
    """
    Preprocesses (downsample via decimate) and saves timeseries data as xarray objects.
    Takes a single experiment ID loads the relevant stores from raw_stores and saves that.
    Channels are specified in the subject_info.yml file.
    Stores to use are defined by raw_stores in the subject_info.yml file.
    """

    # Load raw data from 'preprocess-list' in subject_info.yml
    path = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/subject_info.yml"
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    raw_data = {}
    path = data["paths"][rec]
    for store in data["raw_stores"]:
        raw_data[rec + "-" + store] = kx.io.get_data(
            path, store, channel=data["channels"][store], t1=t1, t2=t2
        )

    # Decimate raw data
    dec_data = {}
    for key in raw_data.keys():
        print("decimating: " + key)
        dec_q = int(raw_data[key].fs / fs_target)
        if dec_q > 1:
            dec_data[key] = kx.utils.decimate(raw_data[key], dec_q)
        else:
            print(
                "fs_target is higher than original fs, skipping decimation for " + key
            )
            print("fs_target: " + str(fs_target))
            print("original fs: " + str(raw_data[key].fs))
            dec_data[key] = raw_data[key]
    # Save decimated data
    save_root = f"/Volumes/opto_loc/Data/{subject}/"
    for key in dec_data.keys():
        print("saving: " + key)
        save_path = save_root + key + ".nc"
        kx.io.save_dataarray(dec_data[key], save_path)
    return


def prepro_lite(subject, rec, fs_target=100, t1=0, t2=0):
    """
    Preprocess and save all experiments which were not included in preprocess-list in subject_info.yml.
    """
    # Load all data that was not included in preprocess-list
    path = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/subject_info.yml"
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    raw_data = {}
    path = data["paths"][rec]

    try:
        for store in data["lite_stores"]:
            raw_data[rec + "-" + store] = kx.io.get_data(
                path, store, channel=data["channels"][store], t1=t1, t2=t2
            )
    except:
        print("no lite stores for " + rec)
        print("using raw_stores for " + rec)
        for store in data["raw_stores"]:
            raw_data[rec + "-" + store] = kx.io.get_data(
                path, store, channel=data["channels"][store], t1=t1, t2=t2
            )
    # Decimate the data
    dec_data = {}
    for key in raw_data.keys():
        print("decimating: " + key)
        dec_q = int(raw_data[key].fs / fs_target)
        if dec_q > 1:
            dec_data[key] = kx.utils.decimate(raw_data[key], dec_q)
        else:
            print(
                "fs_target is higher than original fs, skipping decimation for " + key
            )
            print("fs_target: " + str(fs_target))
            print("original fs: " + str(raw_data[key].fs))
            dec_data[key] = raw_data[key]

    # Save decimated data
    save_root = f"/Volumes/opto_loc/Data/{subject}/"
    for key in dec_data.keys():
        print("saving: " + key)
        save_path = save_root + key + ".nc"
        kx.io.save_dataarray(dec_data[key], save_path)
    return


def data_range_plot(subject):
    """Need properly configured subject_info.yml file"""
    data = load_subject_info(subject)
    starts = []
    ends = []
    for rec in data["recordings"]:
        s = np.datetime64(data["rec_times"][rec]["start"])
        e = np.datetime64(data["rec_times"][rec]["end"])
        s_correct = s > np.datetime64("2020-01-01")
        e_correct = e > np.datetime64("2020-01-01")

        if np.logical_and(s_correct, e_correct):
            starts.append(np.datetime64(data["rec_times"][rec]["start"]))
            ends.append(np.datetime64(data["rec_times"][rec]["end"]))
        else:
            print(f"{rec} has bad start/end date")
    start = min(starts)
    end = max(ends)
    dr = pd.date_range(start=start, end=end, freq="1H")
    f, ax = plt.subplots(figsize=(25, 10))
    ax.plot(dr, np.ones(len(dr)), "o", color="moccasin", markersize=1)

    starts.sort()
    ends.sort()
    it = cycle(range(2, 7, 1))
    colors = [
        "turquoise",
        "blue",
        "lightgrey",
        "dodgerblue",
        "slategrey",
        "turquoise",
        "blue",
        "lightgrey",
        "dodgerblue",
        "slategrey",
        "turquoise",
        "blue",
        "lightgrey",
        "dodgerblue",
        "slategrey",
        "turquoise",
        "blue",
        "lightgrey",
        "dodgerblue",
        "slategrey",
        "turquoise",
        "blue",
        "lightgrey",
        "dodgerblue",
        "slategrey",
        "turquoise",
        "blue",
        "lightgrey",
        "dodgerblue",
    ]
    for s, e, c in zip(starts, ends, colors):
        duration = str(np.timedelta64((e - s), "h"))
        ax.axvspan(s, e, color=c, alpha=0.5, label=duration)

        for rec in data["recordings"]:
            if s == np.datetime64(data["rec_times"][rec]["start"]):
                y = next(it)
                ax.text(s, y, rec, fontsize=10, color="blue")
                ax.text(s, y - 0.2, duration, fontsize=10, color="k")
    ax.set_ylim(1.1, 8)
    return f, ax


def epoc_extractor(subject, recording, epoc_store, t1=0, t2=0):
    # get the subject info file
    info = load_subject_info(subject)
    # get the start time of the recording
    rec_start = np.datetime64(info["rec_times"][recording]["start"])
    rec_end = np.datetime64(info["rec_times"][recording]["end"])
    # load the epocs
    path = info["paths"][recording]
    epocs = tdt.read_block(path, evtype=["epocs"])
    # get the epoc data
    ep = epocs.epocs[epoc_store]
    # get the onsets
    onsets = ep.onset
    # get the offsets
    offsets = ep.offset
    # convert to datetime
    onsets = rec_start + (onsets * 1e9).astype("timedelta64[ns]")
    offsets = rec_start + (offsets * 1e9).astype("timedelta64[ns]")
    # plot the onsets and offsets:
    dt_range = pd.DatetimeIndex([rec_start, rec_end])
    f, ax = plt.subplots(figsize=(20, 5))
    ax.plot(dt_range, [0, 0], "k")
    for on, off in zip(onsets, offsets):
        ax.axvline(on, color="green")
        ax.axvline(off, color="red")
        ax.axvspan(on, off, color="blue", alpha=0.2)
    ax.set_xlim(onsets[0] - pd.Timedelta(1, "s"), offsets[-1] + pd.Timedelta(1, "s"))

    # return the onsets and offsets
    return onsets, offsets


def get_wav2_up_data(subject, exp, t1=0, t2=0, store="Wav2", thresh=1.1e6):
    """returns the times where Wav2 store is greater than 1.5, which should equal the laser on times"""
    info = load_subject_info(subject)
    w = kx.io.get_data(info["paths"][exp], store, t1=t1, t2=t2)
    w_on = w.where(w > thresh, drop=True)

    return w_on


def get_wav2_on_and_off(wav2_up):
    times = wav2_up.datetime.values
    ons = []
    offs = []
    time_int = times[1] - times[0]
    for i in range(len(times)):
        if i == 0:
            ons.append(times[i])
        elif i == (len(times) - 1):
            offs.append(times[i])
        else:
            interval = times[i] - times[i - 1]
            if interval > (time_int * 5):
                ons.append(times[i])
                offs.append(times[i - 1])
    f, ax = plt.subplots(figsize=(15, 5))
    ax.plot(wav2_up.datetime, wav2_up.data)

    dt_range = pd.DatetimeIndex([ons[0], offs[-1]])
    ax.plot(dt_range, [0, 0], "k")
    for on, off in zip(ons, offs):
        ax.axvline(on, color="green")
        ax.axvline(off, color="red")
        ax.axvspan(on, off, color="blue", alpha=0.2)
    ax.set_xlim(ons[0] - pd.Timedelta(1, "s"), offs[-1] + pd.Timedelta(1, "s"))
    return ons, offs


def stim_info_to_yaml(subject, exps):
    """
    subject = subject name (string)
    exps = should be the 'stim-exps' key from the params dict given to subject_info_gen
        - Keys should be experiment names, values should be stim stores to use (Wav2, Bttn, etc.)
    """

    path = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/subject_info.yml"
    info = load_subject_info(subject)

    stim_info = {}
    for exp in exps:
        if exps[exp] != "Wav2":
            pulse_store = exps[exp]
            on, off = epoc_extractor(subject, exp, exps[exp])
            on_list = list(on)
            on_str = [str(x) for x in on_list]
            off_list = list(off)
            off_str = [str(x) for x in off_list]
            stim_info[exp] = {}
            stim_info[exp][pulse_store] = {}
            stim_info[exp][pulse_store]["onsets"] = on_str
            stim_info[exp][pulse_store]["offsets"] = off_str
        elif exps[exp] == "Wav2":
            pulse_store = exps[exp]
            wav2_up = get_wav2_up_data(subject, exp)
            on, off = get_wav2_on_and_off(wav2_up)
            on_list = list(on)
            on_str = [str(x) for x in on_list]
            off_list = list(off)
            off_str = [str(x) for x in off_list]
            stim_info[exp] = {}
            stim_info[exp][pulse_store] = {}
            stim_info[exp][pulse_store]["onsets"] = on_str
            stim_info[exp][pulse_store]["offsets"] = off_str
    info["stim_info"] = stim_info
    with open(path, "w") as f:
        yaml.dump(info, f)
    return


def prepro_test(subject, target=400, t1=0, t2=10, type="full"):
    try:
        if type == "full":
            preprocess_and_save_exp(subject, target, t1, t2)
        elif type == "lite":
            prepro_lite(subject, target, t1, t2)
        else:
            print("type must be full or lite")
            return False
        return True
    except:
        print(f"prepro_test failed for {subject}")
        return False
