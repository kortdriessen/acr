import numpy as np
import pandas as pd
import tdt
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import yaml
import kdephys.xr as kx
import acr
import os
from itertools import cycle
import math
from importlib.machinery import SourceFileLoader
from benedict import benedict
import datetime
from acr.utils import raw_data_root, materials_root, opto_loc_root

def subject_params(subject):
    path = f"{materials_root}{subject}/subject_params.py"
    sub_params = SourceFileLoader("sub_params", path).load_module()
    from sub_params import params
    return params

def load_subject_info(subject):
    path = f"{materials_root}{subject}/subject_info.yml"
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return benedict(data)

def load_dup_info(subject, rec, store):
    path = f"{materials_root}duplication_info.yaml"
    dup_info = yaml.safe_load(open(path, 'r'))
    return benedict(dup_info[subject][rec][store])

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

def get_duration_from_store(path, store):
    data = tdt.read_block(path, evtype=["streams"], channel=[1], store=store)
    num_samples = len(data.streams[store].data)
    fs = data.streams[store].fs
    return float(num_samples / fs)

def time_sanity_check(time):
    return time > np.datetime64("2018-01-01")

def no_end_check(block):
    """returns True if block.info.stop_date is nan, returns False otherwise

    Args:
        block (_type_): tdt block
    """
    if type(block.info.stop_date) == datetime.datetime:
        return False
    else:
        return math.isnan(block.info.stop_date)

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
        
        if no_end_check(d):
            block_duration = get_duration_from_store(p, 'NNXr')
            end = start + pd.Timedelta(block_duration, 's')
            end = np.datetime64(end)
        elif no_end_check(d) == False:
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

def rec_time_comparitor(subject):
    """
    Looks the rec_times field of the subject_info file, and where there are competing ends or competing durations
    (because the time_stores had a different number of samples),
    it chooses the shortest of each and assigns that to the general 'end' and 'duration' fields.

    Args:
        subject (str): subject name

    """
    path = f"{materials_root}{subject}/subject_info.yml"
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


def subject_info_gen(subject):
    """
    This function will load a subject's params dictionary from subject_params.py, and then GENERATE the subject_info.yml file (thus wiping anything already there).
    The params dictionary should have the following properties:
    subject:
        subject name
    raw_stores: 
        list of important data stores, to be used in preprocessing of important recordings (all channels from all raw stores)
    lite_stores: 
        list of less important data stores, to heavily downsample and save only a subset of channels
    channels: 
        a dictionary whose keys are values from stores, and values are the channels to use for each store
    stim-exps: 
        dictionary where the keys are stim experiments, and the values are stores to get the onsets/offsets from (e.g. 'Wav2' or 'Pu1_')
    time_stores: 
        list of stores to use to calculate start and end times. Defaults to ['NNXo', 'NNXr'].
    """
    params = subject_params(subject)
    path = f"{materials_root}{subject}/subject_info.yml"
    root = f"{raw_data_root}{subject}/"
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
    data["channels"] = params["channels"]
    data["paths"] = acr.io.get_acr_paths(subject, recordings)
    data["raw_stores"] = params["raw_stores"]
    data["lite_stores"] = params["lite_stores"]
    data["recordings"] = recordings
    
    data["stim-exps"] = params["stim-exps"]
    with open(path, "w") as f:
        yaml.dump(data, f)
    rec_time_comparitor(subject)
    stim_info_to_yaml(subject, params["stim-exps"])
    return

def update_subject_info(subject, impt_only=True):
    """
    This function will load a subject's params dictionary from subject_params.py, and then update the subject_info.yml file.
    The params dictionary should have the following properties:
    subject:
        subject name
    raw_stores: 
        list of important data stores, to be used in preprocessing of important recordings (all channels from all raw stores)
    lite_stores: 
        list of less important data stores, to heavily downsample and save only a subset of channels
    channels: 
        a dictionary whose keys are values from stores, and values are the channels to use for each store
    stim-exps: 
        dictionary where the keys are stim experiments, and the values are stores to get the onsets/offsets from (e.g. 'Wav2' or 'Pu1_')
    time_stores: 
        list of stores to use to calculate start and end times. Defaults to ['NNXo', 'NNXr'].
    """

    params = subject_params(subject)
    path = f"{materials_root}{subject}/subject_info.yml"
    root = f"{raw_data_root}{subject}/"
    recordings = get_all_tank_keys(root, subject)
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    impt_recs = get_impt_recs(subject)
    new_recs = []
    for rec in recordings:
        if rec not in data['recordings']:
            if impt_only==True:
                if rec in impt_recs:
                    print(f'adding important rec: {rec}')
                    new_recs.append(rec)
            if impt_only == False:
                print(f'adding NON-important rec {rec}')
                new_recs.append(rec)
    
    times = get_rec_times(
        subject,
        new_recs,
        time_stores=params["time_stores"],
        backup_store=[params["lite_stores"][0]],
    )
    for time_key in list(times.keys()):
        data['rec_times'][time_key] = times[time_key]
    
    paths = acr.io.get_acr_paths(subject, recordings)
    data['paths'] = paths
    data["raw_stores"] = params["raw_stores"]
    data["lite_stores"] = params["lite_stores"]
    data["subject"] = subject 
    data["channels"] = params["channels"]
    data["recordings"] = recordings
    data["stim-exps"] = params["stim-exps"]

    with open(path, "w") as f:
        yaml.dump(data, f) # write the new subject_info file
    rec_time_comparitor(subject) # assess the rec_times for all recordings
    stim_info_to_yaml(subject, params["stim-exps"]) # add the stim_info (automatically only loads new recordings, old ones are kept)
    return

def preprocess_and_save_recording(subject, rec, fs_target=400):
    """
    * Requires an updated subject_info file *
    Preprocesses (downsample via decimate) and saves a single recording as xarray object.
    Takes a single recording ID, loads the relevant stores from raw_stores (in subject_info), and saves that.
    Channels are specified in the subject_info.yml file.
    Stores to use are defined by raw_stores in the subject_info.yml file.
    """

    path = f"{materials_root}{subject}/subject_info.yml"
    with open(path) as f:
        info = yaml.load(f, Loader=yaml.FullLoader)
    raw_data = {}
    path = info["paths"][rec]
    t2 = info["rec_times"][rec]["duration"]
    

    for store in info["raw_stores"]:
        raw_data[rec + "-" + store] = kx.io.get_data(
            path, store, channel=info["channels"][store], t1=0, t2=t2
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
    save_root = f"{opto_loc_root}{subject}/"
    for key in dec_data.keys():
        print("saving: " + key)
        save_path = save_root + key + ".nc"
        kx.io.save_dataarray(dec_data[key], save_path)
    return

def preprocess_and_save_all_recordings(subject, fs_target=400):
    all_recs = get_impt_recs(subject)
    already_done = current_processed_recordings(subject)
    for rec in all_recs:
        if rec not in already_done:
            print("preprocessing: " + rec)
            preprocess_and_save_recording(subject, rec, fs_target=fs_target)
    return

def prepro_lite(subject, rec, fs_target=100, t1=0, t2=0):
    """
    * Requires an updated subject_info file *
    Preprocesses (downsample via decimate) and saves a single recording as xarray object.
    Takes a single recording ID, loads the relevant stores from lite_stores (in subject_info), and saves that.
    Channels are specified in the subject_info.yml file.
    Stores to use are defined by lite_stores in the subject_info.yml file.
    """
    
    path = f"{materials_root}{subject}/subject_info.yml"
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
    save_root = f"{opto_loc_root}{subject}/"
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

    path = f"{materials_root}{subject}/subject_info.yml"
    #info = load_subject_info(subject)
    with open(path) as f:
        info = yaml.load(f, Loader=yaml.FullLoader)
    stim_info = {}

    if "stim_info" not in info.keys():
        info["stim_info"] = {}
    
    for exp in exps:
        exp = str(exp)
        if exp in info['stim_info']: # check if recording has already been processed. 
            print(f'stim info for {exp} already processed. Skipping...')
            continue
        stim_info[exp] = {}
        assert type(exps[exp]) == list, f"stores for each experiment must be a list of stores to use. {exp} is not a list."
        for store in exps[exp]:
            if store == "Wav2":
                wav2_up = get_wav2_up_data(subject, exp)
                on, off = get_wav2_on_and_off(wav2_up)
                on_list = list(on)
                on_str = [str(x) for x in on_list]
                off_list = list(off)
                off_str = [str(x) for x in off_list]
                stim_info[exp][store] = {}
                stim_info[exp][store]["onsets"] = on_str
                stim_info[exp][store]["offsets"] = off_str
            elif store == 'Wavt':
                wavt_up = get_wavt_up_data(subject, exp)
                on, off = get_wavt_on_and_off(wavt_up)
                on_list = list(on)
                on_str = [str(x) for x in on_list]
                off_list = list(off)
                off_str = [str(x) for x in off_list]
                stim_info[exp][store] = {}
                stim_info[exp][store]["onsets"] = on_str
                stim_info[exp][store]["offsets"] = off_str
            elif store in ['Bttn', 'Pu1_', 'Pu2_', 'Pu3_']:
                on, off = epoc_extractor(subject, exp, store)
                on_list = list(on)
                on_str = [str(x) for x in on_list]
                off_list = list(off)
                off_str = [str(x) for x in off_list]
                stim_info[exp][store] = {}
                stim_info[exp][store]["onsets"] = on_str
                stim_info[exp][store]["offsets"] = off_str
    
    if 'stim_info' not in info: # make sure stim_info sectio of subject_info.yml exists
        info['stim_info'] = {}
    for exp in stim_info:
        info["stim_info"][exp] = stim_info[exp] # add the new stim info to subject_info.yml
    with open(path, "w") as f:
        yaml.dump(info, f) # save subject_info.yml
    return


def get_wavt_up_data(subject, exp, t1=0, t2=0, store="Wavt", thresh=1e6):
    """returns the times where Wavt store is greater than 1.5, which should equal the laser on times"""
    info = load_subject_info(subject)
    w = kx.io.get_data(info["paths"][exp], store, t1=t1, t2=t2)
    w_on = w.where(w > thresh, drop=True)

    return w_on


def get_wavt_on_and_off(wavt_up):
    times = wavt_up.datetime.values
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
    ax.plot(wavt_up.datetime, wavt_up.data)

    dt_range = pd.DatetimeIndex([ons[0], offs[-1]])
    ax.plot(dt_range, [0, 0], "k")
    for on, off in zip(ons, offs):
        ax.axvline(on, color="green")
        ax.axvline(off, color="red")
        ax.axvspan(on, off, color="blue", alpha=0.2)
    ax.set_xlim(ons[0] - pd.Timedelta(1, "s"), offs[-1] + pd.Timedelta(1, "s"))
    return ons, offs

def prepro_test(subject, target=400, t1=0, t2=10, type="full"):
    try:
        if type == "full":
            preprocess_and_save_recording(subject, target, t1, t2)
        elif type == "lite":
            prepro_lite(subject, target, t1, t2)
        else:
            print("type must be full or lite")
            return False
        return True
    except:
        print(f"prepro_test failed for {subject}")
        return False


def current_processed_recordings(subject):
    processed = []
    root = f"{opto_loc_root}{subject}/"
    for f in os.listdir(root):
        if f.endswith(".nc"):
            name = f.split('-')[:-1]
            rec_name = '-'.join(name)
            processed.append(rec_name)
    return np.unique(processed)

def get_impt_recs(subject):
    """returns a list of all recordings (under all experiments) from the important_recs.yaml file"""
    important_recs = yaml.safe_load(open(f'{materials_root}important_recs.yaml', 'r'))
    impt_recs = important_recs[subject]
    recs = []
    for exp in impt_recs:
        if exp == 'stores':
            continue
        for rec in impt_recs[exp]:
            recs.append(rec)
    return recs

def get_exp_recs(subject, exp):
    """returns a list of all recordings (under a single experiment) from the important_recs.yaml file"""
    important_recs = yaml.safe_load(open(f'{materials_root}important_recs.yaml', 'r'))
    impt_recs = important_recs[subject]
    recs = []
    for rec in impt_recs[exp]:
        recs.append(rec)
    return recs