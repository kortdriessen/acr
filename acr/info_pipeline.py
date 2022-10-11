import numpy as np
import pandas as pd
import tdt
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import xarray as xr
import yaml

import kdephys.hypno as kh
import kdephys.pd as kpd
import kdephys.xr as kx
import kdephys.utils as ku
import kdephys.ssfm as ss

import acr.subjects as subs
import acr
import acr.utils as acu
import plotly.express as px

plt.style.use("acr_plots.mplstyle")
import os
from itertools import cycle


def load_subject_info(subject):
    path = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/subject_info.yml"
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data


def get_all_tank_keys(root):
    tanks = []
    for f in os.listdir(root):
        full_path = os.path.join(root, f)
        if os.path.isdir(full_path):
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


def get_rec_times(sub, exps):
    times = {}
    for exp in exps:
        p = acr.io.acr_path(sub, exp)
        d = tdt.read_block(p, t1=0, t2=1, evtype=["scalars"])
        i = d.info
        start = np.datetime64(i.start_date)
        end = np.datetime64(i.stop_date)
        d = (end - start) / np.timedelta64(1, "s")
        times[exp] = [str(start.astype(str)), str(end.astype(str)), float(d)]
    return times


def subject_info_gen(params):
    """Params is a dictionary with the following keys:
    subject: str
    raw_stores: list of important data stores, to be used in preprocessing of important recordings (all channels from all raw stores)
    lite-stores: list of less important data stores, to heavily downsample and save only a subset of channels
    channels: a dictionary whose keys are values from stores, and values are the channels to use for each store
    preprocess-list: list of important recordings that need to be downsampled/processed. Should be a subset of all available recordings
    stim-exps: dictionary where the keys are stim experiments, and the values are stores to get the onsets/offsets from (e.g. 'Wav2' or 'Pu1_')
    """

    subject = params["subject"]
    channels = params["channels"]
    ds_list = params["preprocess-list"]
    path = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/subject_info.yml"
    root = f"/Volumes/opto_loc/Data/{subject}/"
    recordings = get_all_tank_keys(root)

    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    data = {}
    data["subject"] = subject
    times = get_rec_times(subject, recordings)
    data["rec_times"] = {}
    for rec in recordings:
        data["rec_times"][rec] = {}
        data["rec_times"][rec]["start"] = times[rec][0]
        data["rec_times"][rec]["end"] = times[rec][1]
        data["rec_times"][rec]["duration"] = times[rec][2]
    data["channels"] = channels
    data["paths"] = acr.io.get_acr_paths(subject, recordings)
    data["raw_stores"] = params["raw_stores"]
    data["lite_stores"] = params["lite_stores"]
    data["recordings"] = recordings
    data["preprocess-list"] = ds_list
    data["stim-exps"] = params["stim-exps"]
    with open(path, "w") as f:
        yaml.dump(data, f)


def preprocess_and_save_exps(subject, fs_target=400, t1=0, t2=0):
    """
    Preprocesses (downsample via decimate) and saves timeseries data as xarray objects.
    Goes through all important recordings in in the subject's preprocess-list in the subject_info.yml file.
    Channels are specified in the subject_info.yml file.
    Stores to use are defined by raw_stores in the subject_info.yml file.
    """

    # Load raw data from 'preprocess-list' in subject_info.yml
    path = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/subject_info.yml"
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    raw_data = {}
    for rec in data["preprocess-list"]:
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


def prepro_lite(subject, fs_target=100, t1=0, t2=0):
    """
    Preprocess and save all experiments which were not included in preprocess-list in subject_info.yml.
    """
    # Load all data that was not included in preprocess-list
    path = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/subject_info.yml"
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    raw_data = {}
    for rec in data["recordings"]:
        if rec not in data["preprocess-list"]:
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
    path = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/subject_info.yml"
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    starts = []
    ends = []
    for rec in data["recordings"]:
        starts.append(np.datetime64(data["rec_times"][rec]["start"]))
        ends.append(np.datetime64(data["rec_times"][rec]["end"]))
    start = min(starts)
    end = max(ends)
    dr = pd.date_range(start=start, end=end, freq="1H")
    f, ax = plt.subplots(figsize=(25, 10))
    ax.plot(dr, np.ones(len(dr)), "o", color="moccasin", markersize=1)

    starts.sort()
    ends.sort()
    it = cycle(range(2, 4, 1))
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
    ]
    for s, e, c in zip(starts, ends, colors):
        duration = str(np.timedelta64((e - s), "h"))
        ax.axvspan(s, e, color=c, alpha=0.5, label=duration)

        for rec in data["recordings"]:
            if s == np.datetime64(data["rec_times"][rec]["start"]):
                y = next(it)
                ax.text(s, y, rec, fontsize=20, color="b")
                ax.text(s, y - 0.3, duration, fontsize=20, color="k")
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
            preprocess_and_save_exps(subject, target, t1, t2)
        elif type == "lite":
            prepro_lite(subject, target, t1, t2)
        else:
            print("type must be full or lite")
            return False
        return True
    except:
        print(f"prepro_test failed for {subject}")
        return False
