import pandas as pd
from pathlib import Path
import yaml
import tdt
import kdephys.hypno.hypno as kh
from kdephys.pd.ecdata import ecdata
import kdephys.xr as kx

import kdephys.utils as ku
from kdephys.hypno.ecephys_hypnogram import DatetimeHypnogram, FloatHypnogram
import numpy as np
import acr.info_pipeline as aip
import acr.hypnogram_utils as hu
import os
import xarray as xr
import acr
import polars as pl
import spikeinterface as si
from kdephys.units.utils import set_probe_and_channel_locations_on_si_rec

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

def get_channels_to_exclude(subject, experiment, which='both', probe=None):
    """Takes a subject and experiment, finds which channels are no good based on the bad_channels.xlsx file.

    Parameters
    ----------
    subject : _type_
        _description_
    experiment : _type_
        _description_
    which:
        - dead = only exclude dead channels
        - all_ex = only exclude 'all_exclude' channels (which are excluded from all states, but maybe only for bandpower, etc.)
        - both = exclude both dead and all_exclude channels
    probe : str
        if this is passed, the channels to exclude will be restricted to the given probe, and only channels on that probe will be excluded.
    """
    ex_path = f'{materials_root}bad_channels.xlsx'
    ex = pd.read_excel(ex_path)
    if probe == None:
        deads = ex.loc[(ex['subject']==subject) & (ex['exp']==experiment)]['dead_channels'].values
        all_exs = ex.loc[(ex['subject']==subject) & (ex['exp']==experiment)]['all_exclude'].values
    elif 'NNX' in probe:
        deads = ex.loc[(ex['subject']==subject) & (ex['exp']==experiment) & (ex['store']==probe)]['dead_channels'].values
        all_exs = ex.loc[(ex['subject']==subject) & (ex['exp']==experiment) & (ex['store']==probe)]['all_exclude'].values
    else:
        raise ValueError(f'Probe: {probe} is not recognized')
    total_exclusion = []
    if which == 'dead' or which == 'both':
        for dead in deads:
            if type(dead) != str:
                if type(dead) == int:
                    total_exclusion.append(int(dead))
                else:
                    continue
            elif dead == '-':
                continue
            else:
                if ',' in dead:
                    dead = dead.split(',')
                    for d in dead:
                        total_exclusion.append(int(d))
                else:
                    total_exclusion.append(int(dead))
    if which == 'all_ex' or which == 'both':
        for ex in all_exs:
            if type(ex) != str:
                if type(ex) == int:
                    total_exclusion.append(int(ex))
                else:
                    continue
            elif ex == '-':
                continue
            else:
                if ',' in ex:
                    ex = ex.split(',')
                    for e in ex:
                        total_exclusion.append(int(e))
                else:
                    total_exclusion.append(int(ex))
    return np.unique(total_exclusion)

def nuke_bad_chans_from_df(subject, experiment, df, which='dead', probe=None):
    """Takes a subject and experiment, finds which channels are no good based on the bad_channels.xlsx file, then completely removes them from any dataframe.

    Parameters
    ----------
    subject : _type_
        _description_
    experiment : _type_
        _description_
    df : _type_
        _description_
    which:
        - dead = only exclude dead channels
        - all_ex = only exclude 'all_exclude' channels (which are excluded from all states, but maybe only for bandpower, etc.)
        - both = exclude both dead and all_exclude channels
    probe : str
        if this is passed, the channels to exclude will be restricted to the given probe, and only channels on that probe will be excluded.
    """
    chans_to_nuke = get_channels_to_exclude(subject, experiment, which=which, probe=probe)
    if len(chans_to_nuke) == 0:
        return df

    store_col = 'store' if 'store' in df.columns else 'probe'
    if type(df) == pd.DataFrame:
        if type(probe) == str:
            if 'NNX' in probe:
                return df[~((df[store_col] == probe) & (df['channel'].isin(chans_to_nuke)))]
        elif probe==None:
            return df[~df['channel'].isin(chans_to_nuke)]
        else:
            raise ValueError(f'Probe: {probe} not recognized')
    elif type(df) == pl.DataFrame:
        if type(probe) == str:
            if 'NNX' in probe:
                return df.filter(~((pl.col(store_col) == probe) & (pl.col('channel').is_in(chans_to_nuke))))
        elif probe==None:
            return df.filter(~(pl.col('channel').is_in(chans_to_nuke)))
        else:
            raise ValueError(f'Probe: {probe} not recognized')


def nuke_bad_chans_from_xrds(ds, subject, exp, which='both', probe=None):
    bad_chans = get_channels_to_exclude(subject, exp, which=which, probe=probe)
    if len(bad_chans) == 0:
        return ds
    if type(probe) == str:
        assert 'NNX' in probe, 'Probe must be a valid probe name'
        if type(ds) == xr.Dataset:
            for var_name in ds.data_vars:
                for channel in bad_chans['NNX']:
                    if 'store' in ds.dims:
                        ds[var_name].loc[{'channel': channel, 'store': probe}] = np.nan
                    elif 'store' not in ds.dims:
                        ds[var_name].loc[{'channel': channel}] = np.nan
        elif type(ds) == xr.DataArray:
            for channel in bad_chans['NNX']:
                if 'store' in ds.dims:
                    ds.loc[{'channel': channel, 'store': probe}] = np.nan
                elif 'store' not in ds.dims:
                    ds.loc[{'channel': channel}] = np.nan
    elif probe == None:
        if type(ds) == xr.Dataset:
            for var_name in ds.data_vars:
                for channel in bad_chans['NNX']:
                        ds[var_name].loc[{'channel': channel}] = np.nan
        elif type(ds) == xr.DataArray:
            for channel in bad_chans['NNX']:
                ds.loc[{'channel': channel}] = np.nan
    else:
        raise ValueError('Probe not recognized')
    return ds


# ------------------------------------------ Hypnogram io -------------------------------------#
def get_chunk_num(name):
    return int(name.split('nk')[1].split('.txt')[0])

def check_for_hypnos(subject, recording):
    hypno_file = f"{materials_root}acr-hypno-paths.yaml"
    with open(hypno_file, "r") as f:
        hypno_info = yaml.load(f, Loader=yaml.FullLoader)
    if subject not in hypno_info:
        hypno_info[subject] = {}
        hypno_info[subject]["hypno-root"] = f'{materials_root}{subject}/hypnograms/'
        with open(hypno_file, "w") as f:
            yaml.dump(hypno_info, f)
    if recording in hypno_info[subject]:
        return True
    else:
        return False

def update_hypno_yaml(subject):
    hypno_root = f"{materials_root}{subject}/hypnograms/"
    recs = aip.subject_info_section(subject, 'recordings')
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
        rec_hypnos.sort(key=get_chunk_num)
        hypno_info[subject][rec] = rec_hypnos
    with open(hypno_file, "w") as f:
        yaml.dump(hypno_info, f)
    return


def load_hypno(subject, recording, corrections=False, update=True, float=False):
    if update == True:
        update_hypno_yaml(subject)
    hypno_file = f"{materials_root}acr-hypno-paths.yaml"
    hypno_root = f"{materials_root}{subject}/hypnograms/"
    hypno_info = yaml.load(open(hypno_file, "r"), Loader=yaml.FullLoader)
    if recording not in list(hypno_info[subject].keys()):
        print(f"No hypnogram for {recording}")
        return None
    hypno_paths = hypno_info[subject][recording]
    
    hypno_paths = [f"{hypno_root}{hp}" for hp in hypno_paths]
    rec_times = aip.subject_info_section(subject, "rec_times")
    rec_start = np.datetime64(rec_times[recording]["start"])

    all_hypnos = []
    for hp in hypno_paths:
        if "chunk1.txt" in str(hp):
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
    hypno = hypno.reset_index(drop=True)
    hypno = DatetimeHypnogram(hypno)
    if float:
        hypno = hypno.as_float()
    if corrections:
        return hu.standard_hypno_corrections(hypno)
    else:
        return hypno

def get_float_hypno_dict(subject, exp):
    recs = acr.info_pipeline.get_exp_recs(subject, exp)
    hd = {}
    for rec in recs:
        hd[rec] = load_hypno(subject, rec, corrections=True, update=False, float=True)
    return hd

def load_hypno_full_exp(subject, exp, corrections=True, float=False, update=True):
    """loads every hypnogram across all recordings of an experiment, and concatenates them

    Args:
        subject (str): subject name
        exp (str): experiment name
        float (bool, optional): if True, each recording's hypnogram is FIRST converted to float BEFORE concatenating, so each individual hypnogram will start at zero. Defaults to False.
    """
    h = {}
    if update == True:
        update_hypno_yaml(subject)
    hypno_file = f"{materials_root}acr-hypno-paths.yaml"
    hypno_info = yaml.load(open(hypno_file, "r"), Loader=yaml.FullLoader)
    recs = [x for x in list(hypno_info[subject].keys()) if exp in x]
    recs = acr.info_pipeline.get_exp_recs(subject, exp)
    for rec in recs:
        if rec in list(hypno_info[subject].keys()):
            h[rec] = load_hypno(subject, rec, update=update)
    if float: 
        for rec in h.keys():
            h[rec] = h[rec].as_float()
        hypno_final =  pd.concat(h.values())
        if corrections == True:
            hypno_final = hu.standard_hypno_corrections(hypno_final.reset_index(drop=True))
        return FloatHypnogram(hypno_final)
    else:
        hypno_final =  pd.concat(h.values())
        if corrections == True:
            hypno_final = hu.standard_hypno_corrections(hypno_final.reset_index(drop=True))
        return DatetimeHypnogram(hypno_final)



# ---------------------------------------------------- Data + Spectral io --------------------------------------

def calc_and_save_bandpower_sets(subject, stores=['NNXo', 'NNXr'], recordings=None, window_length=4, overlap=2, redo=False, folder='bandpower_data'):
    """
    NOTE: THIS IS THE OLD FUNCTION FOR CALCULATING BANDPOWER SETS, USE MT_CALC_AND_SAVE_BANDPOWER_SETS INSTEAD; this computes the non-multitaper version!
    NOTE: ONLY USE THIS IF BANDPOWER SETS NOT ALREADY CALCULATED
    NOTE: MUST HAVE DATA STORED IN SUBJECT FOLDER AS NETCDF FILE
    loads and calculates a bandpower dataset for each recording-store combindation, 
    then adds the recording and store information to their coordinates, and saves them to the bandpower_data folder
    
    Args:
        subject (str): subject name
        stores (list): data stores to use
        recordings (list): recordings, defaults to none, in which case all processed recordings in important recordings list are used
    Returns:
        Nothing - only used to save bandpower datasets to bandpower_data folder
    """
    raise DeprecationWarning("This function is deprecated, use MT_CALC_AND_SAVE_BANDPOWER_SETS instead")
    pp_recs = aip.current_processed_recordings(subject)
    impt_recs = aip.get_impt_recs(subject)

    if recordings is None:
        recordings = [r for r in pp_recs if r in impt_recs]

    # make sure the bandpower_data folder exists
    root = f'/Volumes/opto_loc/Data/{subject}/'
    dirs = os.listdir(root)
    dirs = [d for d in dirs if os.path.isdir(root+d)]
    if folder not in dirs:
        os.makedirs(root+folder, exist_ok=False)

    bp_root = f'/Volumes/opto_loc/Data/{subject}/{folder}/'
    bp_recs = os.listdir(bp_root)
    for store in stores:
        for recording in recordings:
            if f'{recording}-{store}.nc' in bp_recs:
                if redo == False:
                    print(f'{recording}-{store} already calculated')
                    continue
            data_root = f"{opto_loc_root}{subject}"
            bp_root = f'{data_root}/{folder}/'
            data = load_raw_data(subject, recording, store)
            spg = kx.spectral.get_spextrogram(data, window_length=window_length, overlap=overlap)
            bp = kx.spectral.get_bp_set(spg)
            bp = bp.assign_coords(recording=recording, store=store)
            bp.to_netcdf(f'{bp_root}{recording}-{store}.nc')
    return None


def MT_calc_and_save_bandpower_sets(subject, stores=['NNXo', 'NNXr'], recordings=None, seg_length=2, overlap=1, NW=4, redo=False):
    """
    NOTE: ONLY USE THIS IF BANDPOWER SETS NOT ALREADY CALCULATED
    NOTE: MUST HAVE DATA STORED IN SUBJECT FOLDER AS NETCDF FILE
    loads and calculates a bandpower dataset for each recording-store combindation, 
    then adds the recording and store information to their coordinates, and saves them to the bandpower_data folder
    
    Args:
        subject (str): subject name
        stores (list): data stores to use
        recordings (list): recordings, defaults to none, in which case all processed recordings in important recordings list are used
    Returns:
        Nothing - only used to save bandpower datasets to bandpower_data folder
    """
    pp_recs = aip.current_processed_recordings(subject)
    impt_recs = aip.get_impt_recs(subject)

    if recordings is None:
        recordings = [r for r in pp_recs if r in impt_recs]

    # make sure the bandpower_data folder exists
    root = f'/Volumes/opto_loc/Data/{subject}/'
    dirs = os.listdir(root)
    dirs = [d for d in dirs if os.path.isdir(root+d)]
    if 'mt_bandpower_data' not in dirs:
        os.makedirs(root+'mt_bandpower_data', exist_ok=False)

    bp_root = f'/Volumes/opto_loc/Data/{subject}/mt_bandpower_data/'
    bp_recs = os.listdir(bp_root)
    for store in stores:
        for recording in recordings:
            if f'{recording}-{store}.nc' in bp_recs:
                if redo == False:
                    print(f'{recording}-{store} already calculated')
                    continue
            print(f'calculating {recording}-{store} MT bandpower set')
            data_root = f"{opto_loc_root}{subject}"
            bp_root = f'{data_root}/mt_bandpower_data/'
            data = load_raw_data(subject, recording, store)
            spg = kx.spectral.get_mt_spextrogram(data, subject=subject, rec=recording, seg_length=seg_length, overlap=overlap, NW=NW)
            bp = kx.spectral.get_bp_set(spg)
            bp = bp.assign_coords(recording=recording, store=store)
            bp.to_netcdf(f'{bp_root}{recording}-{store}.nc')
    return None


def save_raw_data(data, subject, recording, store, overwrite=False, new_path=False):
    path = f"{opto_loc_root}{subject}/{recording}-{store}.nc"
    if new_path:
        path = f"{opto_loc_root}{subject}/{recording}-{store}--NEW.nc"
    if os.path.exists(path) and overwrite == False:
        print(f'{path} already exists, skipping -- use overwrite == True to overwrite')
        return
    if os.path.exists(path) and overwrite == True:
        os.system(f'rm -rf {path}')
        data.to_netcdf(path)
    if os.path.exists(path) == False:
        data.to_netcdf(path)
    return

def _save_bp_set(data, subject, recording, store, overwrite=False, folder='mt_bandpower_data'):
    path = f"{opto_loc_root}{subject}/{folder}/{recording}-{store}.nc"
    if os.path.exists(path) and overwrite == False:
        print(f'{path} already exists, skipping -- use overwrite == True to overwrite')
        return
    data.to_netcdf(path)
    return

def load_raw_data(subject, recording, store, select=None, hypno=None, exclude_bad_channels=False, update_hypno=True):
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
    
    if hypno:
        h = load_hypno(subject, recording, update=update_hypno)
        if h is not None:
            data = kh.add_states(data, h)
        elif h is None:
            states = kh.no_states_array(data.datetime.values)
            data = data.assign_coords(state=("datetime", states))
            print(f'{recording} has no hypnogram, added no_states array dataset')
    if exclude_bad_channels:
        exp = aip.get_exp_from_rec(subject, recording)
        data = nuke_bad_chans_from_xrds(data, subject, exp)
                
    return data



def load_bandpower_file(subject, recording, store, hypno=True, update_hyp=True, select=None, exclude_bad_channels=False, folder='mt_bandpower_data'):
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
    
    path = f"{opto_loc_root}{subject}/{folder}/{recording}-{store}.nc"
    if os.path.exists(path) == False:
        #calc_and_save_bandpower_sets(subject, stores=[store], recordings=[recording])
        raise FileNotFoundError(f'{path} does not exist, run calc_and_save_bandpower_sets(subject, stores=[store], recordings=[recording]) to calculate')
    data = xr.open_dataset(path)
    if select:
        data = data.sel(select)
    
    if np.logical_and(recording not in list(data.coords.keys()), store not in list(data.coords.keys())):
        data = data.assign_coords({'recording': recording, 'store': store})
    if hypno:
        h = load_hypno(subject, recording, update=update_hyp)
        if h is not None:
            data = kh.add_states(data, h)
        elif h is None:
            states = kh.no_states_array(data.datetime.values)
            data = data.assign_coords(state=("datetime", states))
    if exclude_bad_channels:
        exp = aip.get_exp_from_rec(subject, recording)
        data = nuke_bad_chans_from_xrds(data, subject, exp)
    return data

def load_concat_bandpower(subject, recordings, stores, hypno=True, update_hyp=True, select=None, exclude_bad_channels=False, folder='mt_bandpower_data'):
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
            bp = load_bandpower_file(subject, recording, store, hypno=hypno, update_hyp=update_hyp, exclude_bad_channels=exclude_bad_channels, folder=folder)
            bp_recs.append(bp)
        bp_cx_store = xr.concat(bp_recs, dim='datetime')
        bp_stores.append(bp_cx_store)
    bp = xr.concat(bp_stores, dim='store')
    return bp.sel(select)

def load_concat_raw_data(subject, recordings, stores=['NNXo', 'NNXr'], select=None, exclude_bad_channels=False):
    """loads and concatenates raw data for a list of recordings and stores

    Args:
        subject (str): subject name
        recordings (list): recordings
        stores (list): data stores to use
        select(dict): dictionary to pass to .sel() method of concatenated xr.dataset, keys are dimentions, values are values to select
    Returns:
        concatenated xr.dataset of bandpower data for all recording-store combinations
    """
    
    data_stores = []
    for store in stores:
        data_recs = []
        for recording in recordings:
            data_rec = load_raw_data(subject, recording, store, exclude_bad_channels=exclude_bad_channels)
            data_recs.append(data_rec)
        data_cx_store = xr.concat(data_recs, dim='datetime')
        data_stores.append(data_cx_store)
    
    data = xr.concat(data_stores, dim='store')
    if select:
        data = data.sel(select)
    return data


def _interp_raw_fp_data(data, chans_to_interp, sigma_um=50, p=1.3):
    """Takes a xr.dataarray of raw data and interpolates the bad channels, returns the interpolated dataarray

    Parameters
    ----------
    data : _type_
        _description_
    chans_to_interp : _type_
        _description_
    sigma_um : int, optional
        _description_, by default 50
    p : float, optional
        _description_, by default 1.3

    Returns
    -------
    _type_
        _description_
    """
    raw_data = data.values
    assert len(data.channel.values) == 16, 'this function is written for 16 channel probes'
    chan_ids = np.arange(1, 17)
    si_rec = si.extractors.NumpyRecording(raw_data, sampling_frequency=data.fs, channel_ids=chan_ids)
    si_rec = set_probe_and_channel_locations_on_si_rec(si_rec)
    intp_rec = si.preprocessing.interpolate_bad_channels(si_rec, chans_to_interp, sigma_um=sigma_um, p=p)
    intp_data = intp_rec.get_traces()
    data.data = intp_data
    return data

def interpol_and_save_fp_data(subject, rec, probe, redo=False, folder='mt_bandpower_data', new_path=False, data=None):
    """Loads the raw FP data, interpolates the bad channels, recalculates the bandpower set, then resaves the fp data and bandpower set.

    Parameters
    ----------
    subject : _type_
        _description_
    rec : _type_
        _description_
    probe : _type_
        _description_
    """
    if data is None:
        data = load_raw_data(subject, rec, probe, exclude_bad_channels=False, hypno=False)
    else:
        data = data
    chans_to_interp = acr.info_pipeline.get_interpol_info(subject, probe)
    if len(chans_to_interp) == 0:
        return
    if redo == False:
        if acr.info_pipeline.read_interpol_done(subject, rec, probe, version='lfp') is not None:
            return
    data = _interp_raw_fp_data(data, chans_to_interp)
    spg = kx.spectral.get_mt_spextrogram(data, subject=subject, rec=rec, seg_length=2, overlap=1, NW=4)
    bp = kx.spectral.get_bp_set(spg)
    save_raw_data(data, subject, rec, probe, overwrite=True, new_path=new_path)
    _save_bp_set(bp, subject, rec, probe, overwrite=True, folder=folder)
    acr.info_pipeline.write_interpol_done(subject, rec, probe, chans=chans_to_interp, version='lfp')
    return

def interpolate_exp_lfp_data(subject, exp, probes=['NNXo', 'NNXr'], redo=False, folder='mt_bandpower_data'):
    recs_to_interp = acr.info_pipeline.get_exp_recs(subject, exp)
    for rec in recs_to_interp:
        for probe in probes:
            if redo == False:
                if acr.info_pipeline.read_interpol_done(subject, rec, probe, version='lfp') is not None:
                    continue
            print(f'interpolating {rec}, {probe}')
            interpol_and_save_fp_data(subject, rec, probe, folder=folder)
    return

# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS FOR WORKING WITH THE PUB/DATA FOLDER 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
from acr.utils import swi_subs_exps, sub_probe_locations, sub_exp_types

def read_full_df(folder='reb-bp__1h-cum__mean-rel__full-bl', subs_exp=True):
    dataframes = []  # List to hold all dataframes
    
    if subs_exp == False:
        data_directory = f'/home/kdriessen/gh_master/acr/pub/data/{folder}'
        parquet_files = [f for f in os.listdir(data_directory) if f.endswith('.parquet')]
        for file in parquet_files:
            subject = file.split('--')[0]
            exp = file.split('--')[1].split('.parquet')[0]
            df = read_subject_rebdf(subject, exp, folder=folder, exclude_bad_chans=True)
            dataframes.append(df)
        reb_df = pd.concat(dataframes, ignore_index=True)
    
    else:
        for subject in swi_subs_exps.keys():
            for exp in swi_subs_exps[subject]:
                df = read_subject_rebdf(subject, exp, folder=folder, exclude_bad_chans=True)
                dataframes.append(df)
        reb_df = pd.concat(dataframes, ignore_index=True)
    
    if 'Unnamed: 0' in reb_df.columns:
        reb_df = reb_df.drop(columns=['Unnamed: 0'])
    return reb_df

def read_subject_rebdf(sub, exp, folder='reb-bp__1h-cum__mean-rel__full-bl', exclude_bad_chans=True):
    data_path = f'/home/kdriessen/gh_master/acr/pub/data/{folder}/{sub}--{exp}.parquet'
    df = pd.read_parquet(data_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if exclude_bad_chans:
        df = nuke_bad_chans_from_df(sub, exp, df, which='both', probe=None)
    return df