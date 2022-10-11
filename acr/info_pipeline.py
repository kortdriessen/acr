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
    stores: list of relevant data stores, to be used in preprocessing of raw data
    channels: a dictionary whose keys are values from stores, and values are the channels to use for each store
    ds-list = list of recordings that need to be downsampled/processed. Should be a subset of all available recordings
    """
    subject = params["subject"]
    stores = params["stores"]
    channels = params["channels"]
    ds_list = params["ds-list"]
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
    data["stores"] = stores
    data["recordings"] = recordings
    data["ds-list"] = ds_list
    with open(path, "w") as f:
        yaml.dump(data, f)


def preprocess_and_save_timeseries(subject, fs_target=400):
    """Preprocesses (downsample via decimate) and saves timeseries data for each experiment in subject_info.yml"""
    # Load raw data from 'stores' in subject_info.yml
    path = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/subject_info.yml"
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    raw_data = {}
    for exp in data["exps"]:
        path = data["paths"][exp]
        for store in data["stores"]:
            raw_data[exp + "-" + store] = kx.io.get_data(
                path, store, channel=data["channels"][store], t1=0, t2=0
            )
    # Decimate raw data
    dec_data = {}
    for key in raw_data.keys():
        print("decimating: " + key)
        dec_q = int(raw_data[key].fs / fs_target)
        dec_data[key] = kx.utils.decimate(raw_data[key], dec_q)
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
