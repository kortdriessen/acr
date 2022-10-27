import pandas as pd
from pathlib import Path
import yaml
import tdt

import kdephys.hypno as kh
from kdephys.pd.ecdata import ecdata
import kdephys.xr as kx

import kdephys.utils as ku
from kdephys.hypnogram import DatetimeHypnogram
import numpy as np
import acr.info_pipeline as aip
import os
import xarray as xr

bands = ku.spectral.bands


class data_dict(dict):
    def __init__(self, *args, **kwargs):
        super(data_dict, self).__init__(*args, **kwargs)

    @property
    def _contructor(self):
        return data_dict

    def af(self, method, args):
        nd = {}
        for key in list(self.keys()):
            nd[key] = getattr(self[key], method)(args)
        return data_dict(nd)


def acr_path(sub, x):
    path = "/Volumes/opto_loc/Data/" + sub + "/" + sub + "-" + x
    return path


def get_acr_paths(sub, xl):
    paths = {}
    for x in xl:
        path = acr_path(sub, x)
        paths[x] = path
    return paths


def load_subject_data(info, stores=["EEGr", "LFP_"]):
    data = {}
    for exp in info["complete_key_list"]:
        t1 = info["load_times"][exp][0]
        t2 = t1 + (info["load_times"][exp][1] * 3600)
        for store in stores:
            chans = info["channels"][store]
            data[exp + "-" + store] = kx.io.get_data(
                info["paths"][exp], store, t1=t1, t2=t2, channel=chans
            )
    return data


def xarray_to_pandas(xarr, name=None):
    if name is None:
        return xarr.to_dataframe()
    else:
        return xarr.to_dataframe(name=name)


def redo_timdelta(df):
    start = df.index.get_level_values(0)[0]
    df.drop(columns=["timedelta"], inplace=True)
    df["timedelta"] = df.index.get_level_values(0) - start
    return df


def check_hypno(subject, condition):
    hypnograms_yaml_file = (
        "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/acr-hypno-paths.yaml"
    )
    with open(hypnograms_yaml_file) as fp:
        yaml_data = yaml.safe_load(fp)
    root = Path(yaml_data[subject]["hypno-root"])
    if condition in yaml_data[subject].keys():
        return True
    else:
        return False


def _load_hypno(subject, condition, start_time):
    "critical - this only works for loading continuous hypnograms"
    # TODO: make this more flexible for loading discontinuous hypnograms from the same experiment
    hypnograms_yaml_file = (
        "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/acr-hypno-paths.yaml"
    )
    with open(hypnograms_yaml_file) as fp:
        yaml_data = yaml.safe_load(fp)
    root = Path(yaml_data[subject]["hypno-root"])
    if check_hypno(subject, condition):
        hypnogram_fnames = yaml_data[subject][condition]
        hypnogram_paths = [root / (fname + ".txt") for fname in hypnogram_fnames]
        hypnos = range(1, len(hypnogram_paths) + 1)
        hyp_list = []
        for num, hp in zip(hypnos, hypnogram_paths):
            if num == 1:
                h = kh.load_hypno_file(hp, start_time)
                hyp_list.append(h)
            if num > 1:
                h = kh.load_hypno_file(hp, h.end_time.max())
                hyp_list.append(h)
        hh = pd.concat(hyp_list).reset_index(drop=True)
        return DatetimeHypnogram(hh)
    else:
        return None


def load_hypno_dep(info, data, data_tag):
    """
    info --> subject info dictionary
    data --> data dictionary
    Note: The minimum datetime of the data will be taken as the start time for the hypnogram.
    """
    hyp = {}
    subject = info["subject"]

    for cond in info["complete_key_list"]:
        if check_hypno(subject, cond):
            cond_data = cond + "-" + data_tag
            for d in data.keys():
                if cond_data in d:
                    start_time = data[d].datetime.values.min()
                    break
            hyp[cond] = _load_hypno(subject, cond, start_time)
        else:
            print("No hypnogram for condition: ", cond)
    return hyp


def add_hypnograms_to_dataset(dataset, hypno_set):
    for key in hypno_set.keys():
        for exp in dataset.keys():
            if key in exp:
                dataset[exp] = kh.add_states(dataset[exp], hypno_set[key])
    return dataset


def get_spectral(data):
    spg = kx.spectral.get_spg_from_dataset(data)
    bp = kx.spectral.get_bp_from_dataset(spg)
    # for key in list(spg.keys()):
    # bp[key] = kx.spectral.get_bp_set(spg[key], bp_def)
    return spg, bp


def dataset_to_pandas(dataset, index=["datetime", "channel"], name=None):
    for key in list(dataset.keys()):
        dataset[key] = xarray_to_pandas(dataset[key], name=name)
        dataset[key] = redo_timdelta(dataset[key])
        dataset[key]["condition"] = key
        dataset[key] = dataset[key].reset_index()
        dataset[key] = dataset[key].set_index(index)
        dataset[key] = ecdata(dataset[key])
    return dataset


def load_saved_dataset(si, type, data_tags):
    """
    data_tage --> e.g. '-EEGr'
    type --> e.g. '-spg'
    """
    # TODO: should probably import the data as my pandas ecdata class

    cond_list = si["complete_key_list"]
    sub = si["subject"]
    path_root = (
        "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/" + sub + "/" + "analysis-data/"
    )
    ds = {}
    if type == "-hypno":
        for cond in cond_list:
            try:
                path = path_root + cond + type + ".parquet"
                ds[cond] = pd.read_parquet(path)
                ds[cond] = DatetimeHypnogram(ds[cond])
            except:
                print("No hypnogram for condition: ", cond)
        return ds

    else:
        for cond in cond_list:
            for tag in data_tags:
                key = cond + tag
                path = path_root + key + type + ".parquet"
                ds[key] = pd.read_parquet(path)
                ds[key] = ecdata(ds[key])
        return ds


def save_xarray(data_dict, si, type="-bp"):
    sub = si["subject"]
    save_path = (
        "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/" + sub + "/" + "analysis-data/"
    )
    for key in list(data_dict.keys()):
        data_dict[key].to_pickle(save_path + key + type + ".parquet")
        print(f"{key} saved")
    return None


def save_dataframes(data_dict, si, type="-bp"):
    sub = si["subject"]
    save_path = (
        "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/" + sub + "/" + "analysis-data/"
    )
    for key in list(data_dict.keys()):
        data_dict[key].to_pickle(save_path + key + type + ".parquet")
        print(f"{key} saved")
    return None


def ss_times(sub, exp, print_=False):
    # TODO: this may be deprecated, and may not be needed, I think it was only used for ACR_9?

    # Load the relevant times:
    def acr_get_times(sub, exp):
        block_path = "/Volumes/opto_loc/Data/" + sub + "/" + sub + "-" + exp
        ep = tdt.read_block(block_path, t1=0, t2=0, evtype=["epocs"])
        times = {}
        times["bl_sleep_start"] = ep.epocs.Bttn.onset[0]
        times["stim_on"] = ep.epocs.Wdr_.onset[-1]
        times["stim_off"] = ep.epocs.Wdr_.offset[-1]
        dt_start = pd.to_datetime(ep.info.start_date)

        # This get us the datetime values of the stims for later use:
        on_sec = pd.to_timedelta(times["stim_on"], unit="s")
        off_sec = pd.to_timedelta(times["stim_off"], unit="s")
        times["stim_on_dt"] = dt_start + on_sec
        times["stim_off_dt"] = dt_start + off_sec
        return times

    times = acr_get_times(sub, exp)

    # Start time for scoring is 30 seconds before the button signal was given to inidicate the start of bl peak period.
    start1 = times["bl_sleep_start"] - 30
    end1 = start1 + 7200

    # End time for the second scoring file is when the stim/laser signal is turned off.
    start2 = end1
    end2 = times["stim_off"]
    if print_:
        print("FILE #1"), print(start1), print(end1)
        print("FILE #2"), print(start2), print(end2)
    print("Done loading times")
    return times


def acr_load_master(info, type="xarray", stores=["EEGr", "LFP_"], hyp=True):
    """
    returns: dataset, spg_set, bp_set, hypno_set.
    hypno_sets is only the mannually scored hypnograms.

    Parameters:
    -----------
    info --> subject info dictionary, ALL RELEVANT INFORMATION IS DEFINED HERE!;
    type --> 'pandas' or 'xarray'
    stores:
        the data stores to load, e.g. ['EEGr', 'LFP_']
    hyp:
        whether or not to load the hypnograms, if false, only data, spg, and bp are returned.
    """

    # Load the data
    data = load_subject_data(info, stores=stores)

    if hyp:
        # Load the hypnograms
        hyp = load_hypno(info, data, stores[0])

    # Calculate the spectral data
    spg, bp = get_spectral(data)

    if hyp:
        # add hypnogram to bandpower set and spectrogram
        # spg = add_hypnograms_to_dataset(spg, hyp)
        bp = add_hypnograms_to_dataset(bp, hyp)

    if type == "xarray":
        spg = data_dict(spg)
        bp = data_dict(bp)
        data = data_dict(data)
        if hyp:
            return data, spg, bp, hyp
        else:
            return data, spg, bp

    # We can do the conversion to pandas here...
    elif type == "pandas":
        data = dataset_to_pandas(data, name="data")
        spg = dataset_to_pandas(spg, name="spg")
        bp = dataset_to_pandas(bp)
        if hyp:
            return data, spg, bp, hyp
        else:
            return data, spg, bp


def save_dataset(data, si, type="-bp"):
    sub = si["subject"]
    save_path = (
        "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/" + sub + "/" + "analysis-data/"
    )
    for key in list(data.keys()):
        data[key].to_parquet(save_path + key + type + ".parquet", version="2.6")
        print(f"{key} saved")
    return None


def save_bandpower_data(subject, dataset):
    """save a dictionary of xarray bandpower datasets to netcdf files

    Args:
        subject (str): subject name
        dataset (_type_): dictionary, where keys will be used as key-type for saving
    Returns:

    """
    root = f"/Volumes/opto_loc/Data/{subject}/bandpower_data/"
    for key in dataset.keys():
        save_key = f"{key}.nc"
        dataset[key].to_netcdf(root + save_key)
        print(f"{key} saved")


def load_bandpower_data(subject):
    """Returns a dictionary with every bandpower set in the subject's bandpower_data folder

    Args:
        subject (str): subject name

    Returns:
        bp_dataset: dictionary of xarray datasets, consisting of all bandpower sets in the subject's bandpower_data folder
    """
    path = f"/Volumes/opto_loc/Data/{subject}/bandpower_data/"
    bp_dataset = {}
    for file in os.listdir(path):
        full_path = path + file
        key = file.split(".")[0]
        bp_dataset[key] = xr.open_dataset(full_path)
    return bp_dataset


# -------------- New Hypnogram Functions ------------------#
def check_for_hypnos(subject, recording):
    hypno_file = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/acr-hypno-paths.yaml"
    with open(hypno_file, "r") as f:
        hypno_info = yaml.load(f, Loader=yaml.FullLoader)
    if recording in hypno_info[subject]:
        return True
    else:
        return False


def update_hypno_yaml(subject):
    hypno_root = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/hypnograms/"
    info = aip.load_subject_info(subject)
    recs = info["recordings"]
    hypno_file = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/acr-hypno-paths.yaml"
    with open(hypno_file, "r") as f:
        hypno_info = yaml.load(f, Loader=yaml.FullLoader)
    if not type(hypno_info[subject]) == dict:
        hypno_info[subject] = {}
        hypno_info[subject]["hypno-root"] = hypno_root
    if not hypno_info[subject]["hypno-root"] == hypno_root:
        hypno_info[subject]["hypno-root"] = hypno_root

    for rec in recs:
        rec_hypnos = []
        for f in os.listdir(hypno_root):
            if f"{rec}_chunk" in f:
                rec_hypnos.append(f)
        if len(rec_hypnos) == 0:
            continue
        rec_hypnos = sorted(rec_hypnos)
        hypno_info[subject][rec] = rec_hypnos
    with open(hypno_file, "w") as f:
        yaml.dump(hypno_info, f)


def load_hypno(subject, recording):
    update_hypno_yaml(subject)
    hypno_file = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/acr-hypno-paths.yaml"
    hypno_root = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/hypnograms/"
    hypno_info = yaml.load(open(hypno_file, "r"), Loader=yaml.FullLoader)
    hypno_paths = hypno_info[subject][recording]
    hypno_paths = [f"{hypno_root}{hp}" for hp in hypno_paths]
    sub_info = aip.load_subject_info(subject)
    rec_start = np.datetime64(sub_info["rec_times"][recording]["start"])

    all_hypnos = []
    for hp in hypno_paths:
        if "chunk1" in str(hp):
            config_path = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/config-files/{subject}_sleepscore-config_{recording}-chunk1.yml"
            config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
            start = config["tStart"]
            hypno_start = pd.to_timedelta(start, unit="s")
            start_time = np.datetime64(rec_start + hypno_start)
            h = kh.load_hypno_file(hp, start_time)
            all_hypnos.append(h)
            end = h.end_time.max()
        else:
            h = kh.load_hypno_file(hp, end)
            all_hypnos.append(h)
            end = h.end_time.max()
    hypno = pd.concat(all_hypnos)
    return DatetimeHypnogram(hypno)


# -------------- New Loading Functions ------------------#
def load_xr_exp(subject, recordings, stores=["NNXo", "NNXr"]):
    data = {}
    concat_data = {}
    for store in stores:
        for recording in recordings:
            path = f"/Volumes/opto_loc/Data/{subject}/{recording}-{store}.nc"
            data[f"{recording}-{store}"] = xr.open_dataarray(path)
        keys = data.keys()
        to_concat = [data[key] for key in keys]
        concat_data[f"{store}"] = xr.concat(to_concat, dim="datetime")
    return concat_data


def load_xr(
    subject, recordings, stores=["NNXo", "NNXr"], channels=None, add_rec_coord=False
):
    data = {}
    for store in stores:
        for recording in recordings:
            path = f"/Volumes/opto_loc/Data/{subject}/{recording}-{store}.nc"
            data[f"{recording}-{store}"] = (
                xr.open_dataarray(path).sel(channel=channels)
                if channels
                else xr.open_dataarray(path)
            )
            if add_rec_coord:
                data[f"{recording}-{store}"] = data[
                    f"{recording}-{store}"
                ].assign_coords(recording=recording)
    return data
