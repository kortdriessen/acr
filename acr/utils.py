import pandas as pd
import yaml
import acr
import tdt
import numpy as np
import os
import matplotlib.pyplot as plt

raw_data_root = "/Volumes/neuropixel_archive/Data/acr_archive/"
materials_root = "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/"
opto_loc_root = "/Volumes/opto_loc/Data/"

NNXR_GRAY = "#4b4e4d"
NNXO_BLUE = "dodgerblue"
SOM_BLUE = "#4508ff"
BAND_ORDER = ['delta1', 'delta2', 'delta', 'theta', 'alpha', 'sigma', 'beta', 'low_gamma', 'high_gamma']

swi_subs_exps = {
    "ACR_14": ["swi"],
    "ACR_16": ["swi2"],
    "ACR_17": ["swi"],
    "ACR_18": ["swi", "swisin"],
    "ACR_19": ["swi2", "swisin"],
    "ACR_20": ["swi", "swisin"],
    "ACR_21": ["swi2", "swisin"],
    "ACR_23": ["swi2", "swisin"],
    "ACR_25": ["swi", "swisin"],
    "ACR_26": ["swi", "swisin2"],
    "ACR_27": ["swi"],
    "ACR_28": ["swi", "swisin"],
    "ACR_29": ["swi"],
    "ACR_30": ["swi2"],
    "ACR_31": ["swisin2", "swi2"],
    "ACR_33": ["swisin", "swi"],
    "ACR_34": ["swisin2", "swi"],
    "ACR_35": ["swi", "swisin"],
}

sub_probe_locations = {
    "ACR_14": "frontal",
    "ACR_16": "frontal",
    "ACR_17": "parietal",
    "ACR_18": "frontal",
    "ACR_19": "frontal",
    "ACR_20": "frontal",
    "ACR_21": "frontal",
    "ACR_23": "parietal",
    "ACR_25": "frontal",
    "ACR_26": "parietal",
    "ACR_27": "parietal",
    "ACR_28": "parietal",
    "ACR_29": "parietal",
    "ACR_30": "parietal",
    "ACR_31": "parietal",
    "ACR_33": "frontal",
    "ACR_34": "frontal",
    "ACR_35": "frontal",
}

sub_exp_types = {
    "ACR_14": "acr",
    "ACR_16": "acr",
    "ACR_17": "acr",
    "ACR_18": "acr",
    "ACR_19": "acr",
    "ACR_20": "acr",
    "ACR_21": "control",
    "ACR_23": "acr",
    "ACR_25": "som",
    "ACR_26": "som",
    "ACR_27": "control",
    "ACR_28": "som",
    "ACR_29": "som", 
    "ACR_30": "som",
    "ACR_31": "acr",
    "ACR_33": "acr",
    "ACR_34": "acr",
    "ACR_35": "som",
}


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

def dt_to_tdt(subject, rec, dt):
    """converts datetime to tdt time for a given recording"""

    rec_times = acr.info_pipeline.subject_info_section(subject, 'rec_times')
    rec_start = pd.Timestamp(rec_times[rec]['start'])
    if type(dt) == str:
        dt = pd.Timestamp(dt)
    return (dt - rec_start).total_seconds()
    

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

def elimate_bad_channels(subject, exp, stores=['NNXo', 'NNXr'], fp=True, bp=True, unit_df=True):
    """

    Parameters
    ----------
    subject : _type_
        _description_
    exp : _type_
        _description_
    stores : list, optional
        _description_, by default ['NNXo', 'NNXr']
    fp : bool, optional
        _description_, by default True
    bp : bool, optional
        _description_, by default True
    unit_df : bool, optional
        _description_, by default True
    """
    path = '/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/channel_exclusion.yaml'
    params = acr.info_pipeline.subject_params(subject)
    all_chans = params['channels'][stores[0]]
    with open(path, 'r') as f:
        channel_exclusion = yaml.safe_load(f)
    chans_to_exclude = channel_exclusion[subject][exp]
    chans_to_keep = [chan for chan in all_chans if chan not in chans_to_exclude]
    recordings = acr.info_pipeline.get_exp_recs(subject, exp)
    
def get_recording_end_time(subject, recording, store):
    end_info = yaml.safe_load(
        open("/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/end_times.yaml", "r")
    )
    return end_info[subject][recording][store]['zero_period_start'][0]


def _resample_numpy(signal, desired_length):
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, desired_length, endpoint=False),  # where to interpolate
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return(resampled_signal)

def save_sub_exp_fig(subject, exp, filename):
    save_root = f'/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/plots_presentations_etc/PLOTS_MASTER/{subject}/{exp}'
    if os.path.exists(save_root) == False:
        os.mkdir(save_root)
    plt.savefig(f'{save_root}/{filename}.svg', dpi=300, bbox_inches='tight')