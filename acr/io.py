import pandas as pd
from pathlib import Path
import yaml
import tdt
import kdephys.hypno.hypno as kh
from kdephys.pd.ecdata import ecdata
import kdephys.xr as kx

import kdephys.utils as ku
from kdephys.hypno.ecephys_hypnogram import DatetimeHypnogram
import numpy as np
import acr.info_pipeline as aip
import os
import xarray as xr

from acr.utils import materials_root, opto_loc_root, raw_data_root
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
    return f"{raw_data_root}{sub}/{sub}-{x}"


def get_acr_paths(sub, xl):
    paths = {}
    for x in xl:
        path = acr_path(sub, x)
        paths[x] = path
    return paths


# ------------------------------------------ Hypnogram io -------------------------------------#
def check_for_hypnos(subject, recording):
    hypno_file = f"{materials_root}acr-hypno-paths.yaml"
    with open(hypno_file, "r") as f:
        hypno_info = yaml.load(f, Loader=yaml.FullLoader)
    if recording in hypno_info[subject]:
        return True
    else:
        return False


def update_hypno_yaml(subject):
    hypno_root = f"{materials_root}{subject}/hypnograms/"
    info = aip.load_subject_info(subject)
    recs = info["recordings"]
    hypno_file = f"{materials_root}acr-hypno-paths.yaml"
    with open(hypno_file, "r") as f:
        hypno_info = yaml.load(f, Loader=yaml.FullLoader)
    if subject not in hypno_info:
        hypno_info[subject] = {}
        hypno_info[subject]["hypno-root"] = hypno_root
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
    return


def load_hypno(subject, recording):
    update_hypno_yaml(subject)
    hypno_file = f"{materials_root}acr-hypno-paths.yaml"
    hypno_root = f"{materials_root}{subject}/hypnograms/"
    hypno_info = yaml.load(open(hypno_file, "r"), Loader=yaml.FullLoader)
    if recording not in list(hypno_info[subject].keys()):
        print(f"No hypnogram for {recording}")
        return None
    hypno_paths = hypno_info[subject][recording]
    
    hypno_paths = [f"{hypno_root}{hp}" for hp in hypno_paths]
    sub_info = aip.load_subject_info(subject)
    rec_start = np.datetime64(sub_info["rec_times"][recording]["start"])

    all_hypnos = []
    for hp in hypno_paths:
        if "chunk1" in str(hp):
            config_path = f"{materials_root}{subject}/config-files/{subject}_sleepscore-config_{recording}-chunk1.yml"
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

def load_hypno_full_exp(subject, exp):
    """loads every hypnogram across all recordings of an experiment, and concatenates them

    Args:
        subject (str): subject name
        exp (str): experiment name
    """
    h = {}
    update_hypno_yaml(subject)
    hypno_file = f"{materials_root}acr-hypno-paths.yaml"
    hypno_info = yaml.load(open(hypno_file, "r"), Loader=yaml.FullLoader)
    recs = [x for x in list(hypno_info[subject].keys()) if exp in x]
    for rec in recs:
        h[rec] = load_hypno(subject, rec)
    return DatetimeHypnogram(pd.concat(h.values()))

# ---------------------------------------------------- Data + Spectral io --------------------------------------
def calc_and_save_bandpower_sets(subject, recordings, stores, window_length=4, overlap=2):
    """
    NOTE: ONLY USE THIS IF BANDPOWER SETS NOT ALREADY CALCULATED
    NOTE: MUST HAVE DATA STORED IN SUBJECT FOLDER AS NETCDF FILE
    loads and calculates a bandpower dataset for each recording-store combindation, 
    then adds the recording and store information to their coordinates, and saves them to the bandpower_data folder
    
    Args:
        subject (str): subject name
        recordings (list): recordings
        stores (list): data stores to use
    Returns:
        Nothing - only used to save bandpower datasets to bandpower_data folder
    """
    
    for store in stores:
        for recording in recordings:
            data_root = f"{opto_loc_root}{subject}"
            bp_root = f'{data_root}/bandpower_data/'
            data = load_raw_data(subject, recording, store)
            spg = kx.spectral.get_spextrogram(data, window_length=window_length, overlap=overlap)
            bp = kx.spectral.get_bp_set(spg)
            bp = bp.assign_coords(recording=recording, store=store)
            bp.to_netcdf(f'{bp_root}{recording}-{store}.nc')
    return None

def load_raw_data(subject, recording, store, select=None, hypno=None):
    """loads the xr.dataarray of raw data for a single recording-store combination.

    Args:
        subject (str): subject name
        recording (str): recording name
        store (str): store name
        select(dict): dictionary to pass to .sel() method of xr.dataarray, keys are dimentions, values are values to select

    Returns:
        xr.dataarray of raw data for recording-store combination
    """
    path = f"{opto_loc_root}{subject}/{recording}-{store}.nc"
    
    data = xr.open_dataarray(path)
    if select:
        data = data.sel(select)
    if np.logical_and(recording not in list(data.coords.keys()), store not in list(data.coords.keys())):
        data = data.assign_coords({'recording': recording, 'store': store})
        print(f'{recording} was missing recording and store coordinates, added them')
    
    if hypno:
        h = load_hypno(subject, recording)
        if h is not None:
            data = kh.add_states(data, h)
        elif h is None:
            states = kh.no_states_array(data.datetime.values)
            data = data.assign_coords(state=("datetime", states))
            print(f'{recording} has no hypnogram, added no_states array dataset')
    return data



def load_bandpower_file(subject, recording, store, hypno=True, select=None):
    """loads the xr.dataset of bandpower data for a single recording-store combination.

    Args:
        subject (str): subject name
        recording (str): recording name
        store (str): store name
        hypno (bool, optional): if True, loads the hypnogram for the recording and adds it as a state coordindate
        select(dict, optional): dictionary to pass to .sel() method of xr.dataset, keys are dimentions, values are values to select

    Returns:
        xr.dataset of bandpower data for recording-store combination
    """
    
    path = f"{opto_loc_root}{subject}/bandpower_data/{recording}-{store}.nc"
    data = xr.open_dataset(path)
    if select:
        data = data.sel(select)
    
    if np.logical_and(recording not in list(data.coords.keys()), store not in list(data.coords.keys())):
        data = data.assign_coords({'recording': recording, 'store': store})
        print(f'{recording} was missing recording and store coordinates, added them')
    if hypno:
        h = load_hypno(subject, recording)
        if h is not None:
            data = kh.add_states(data, h)
        elif h is None:
            states = kh.no_states_array(data.datetime.values)
            data = data.assign_coords(state=("datetime", states))
            print(f'{recording} has no hypnogram, added no_states array dataset')
    return data

def load_concat_bandpower(subject, recordings, stores, hypno=True, select=None):
    """loads and concatenates bandpower data for a list of recordings and stores

    Args:
        subject (str): subject name
        recordings (list): recordings
        stores (list): data stores to use
        select(dict): dictionary to pass to .sel() method of concatenated xr.dataset, keys are dimentions, values are values to select
        hypno (bool, optional): passed to load_bandpower_file, if True adds a state coordinate to the bandpower data
    Returns:
        concatenated xr.dataset of bandpower data for all recording-store combinations
    """
    
    bp_stores = []
    for store in stores:
        bp_recs = []
        for recording in recordings:
            bp = load_bandpower_file(subject, recording, store, hypno=hypno)
            bp_recs.append(bp)
        bp_cx_store = xr.concat(bp_recs, dim='datetime')
        bp_stores.append(bp_cx_store)
    bp = xr.concat(bp_stores, dim='store')
    return bp

def load_concat_raw_data(subject, recordings, stores, hypno=True, select=None):
    """loads and concatenates bandpower data for a list of recordings and stores

    Args:
        subject (str): subject name
        recordings (list): recordings
        stores (list): data stores to use
        select(dict): dictionary to pass to .sel() method of concatenated xr.dataset, keys are dimentions, values are values to select
        hypno (bool, optional): passed to load_bandpower_file, if True adds a state coordinate to the bandpower data
    Returns:
        concatenated xr.dataset of bandpower data for all recording-store combinations
    """
    
    data_stores = []
    for store in stores:
        data_recs = []
        for recording in recordings:
            data_rec = load_raw_data(subject, recording, store, select=select)
            data_recs.append(data_rec)
        data_cx_store = xr.concat(data_recs, dim='datetime')
        data_stores.append(data_cx_store)
    
    data = xr.concat(data_stores, dim='store')
    # occasionally because of mismatching total lengths between recordings, the time dimens
    return data
