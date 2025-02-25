import pandas as pd
import yaml
import acr
import tdt
import numpy as np
import os
import matplotlib.pyplot as plt
import polars

raw_data_root = "/Volumes/neuropixel_archive/Data/acr_archive/"
materials_root = "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/"
opto_loc_root = "/Volumes/opto_loc/Data/"

PAPER_FIGURE_ROOT = "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/plots_presentations_etc/paper_figures"


#SOM_BLUE = '#0046B8'
SOM_BLUE = '#0528b2'
ACR_BLUE = '#54A3FF'

NNXR_GRAY = "#4b4e4d"
NNXO_BLUE = "dodgerblue"

BAND_ORDER = ['delta1', 'delta2', 'delta', 'theta', 'alpha', 'sigma', 'beta', 'low_gamma', 'high_gamma']
COND_ORDER = ['early_bl', 'circ_bl', 'early_sd', 'late_sd', 'stim', 'rebound', 'late_rebound']
SUB_GROUPS = ['acr-frontal', 'acr-parietal', 'som-frontal', 'som-parietal', 'control-frontal', 'control-parietal']
SLEEP_CONDS = ['early_bl', 'circ_bl', 'rebound', 'late_rebound']
WAKE_CONDS = ['early_sd', 'late_sd', 'stim']

probe_ord = ['NNXr', 'NNXo']
probe_pal = ["#4b4e4d",
             '#475ED1']

pal_full = ['#475ED1',
            '#F97A1F',
            '#1DC9A4', 
            '#F6423C', 
            '#F9C31F',
            '#bf56dc',
            '#adb8eb',
            '#fcb583',]

sub_swi_exps = {
    "ACR_14": ["swi"],
    "ACR_16": ["swi2"],
    "ACR_17": ["swi"],
    "ACR_18": ["swi"],
    "ACR_19": ["swi2"],
    "ACR_20": ["swi"],
    "ACR_21": ["swi2"],
    "ACR_23": ["swi2"],
    "ACR_25": ["swi"],
    "ACR_26": ["swi"],
    "ACR_27": ["swi"],
    #"ACR_28": ["swi", "swisin"], don't think the data quality was good enough to use this mouse
    "ACR_29": ["swi"],
    "ACR_30": ["swi2"],
    "ACR_31": ["swi2"],
    "ACR_33": ["swi"],
    "ACR_34": ["swi"],
    "ACR_35": ["swi"], #don't think I can include the TBS experiment here...
    "ACR_37": ["swi2"],
    "ACR_39": ["swi"],
    "ACR_40": ["swi"],
    "ACR_41": ["swi"],
    "ACR_42": ["swi"],
    "ACR_44": ["swi"],
    "ACR_45": ["swi2"],
}

sub_swisin_exps = {
    "ACR_18": ["swisin"],
    "ACR_19": ["swisin"],
    "ACR_20": ["swisin"],
    "ACR_21": ["swisin"],
    "ACR_23": ["swisin"],
    "ACR_25": ["swisin"],
    "ACR_31": ["swisin2"],
    "ACR_33": ["swisin"],
    "ACR_34": ["swisin2"],
    "ACR_37": ["swisin3"],
    "ACR_39": ["swisin"],
    "ACR_40": ["swisin"],
    "ACR_41": ["swisin"],
    "ACR_42": ["swisin"],
    "ACR_44": ["swisin"],
    "ACR_45": ["swisin"],
}

sub_swinat_exps = {
    "ACR_44": ["swinat"],
    "ACR_45": ["swinat"],
}



sub_ctrl_exps = {
    'ACR_19': ['controlsd'],
    'ACR_23': ['controlsd'],
    'ACR_31': ['controlsd'],
    'ACR_33': ['controlsd'],
    'ACR_34': ['controlsd'],
    'ACR_35': ['controlsd'],
    'ACR_37': ['controlsd'],
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
    "ACR_37": "parietal",
    "ACR_39": "frontal",
    "ACR_40": "parietal",
    "ACR_41": "parietal",
    "ACR_42": "parietal",
    "ACR_44": "parietal",
    "ACR_45": "parietal",
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
    #"ACR_28": "som",
    "ACR_29": "som", 
    "ACR_30": "som",
    "ACR_31": "acr",
    "ACR_33": "acr",
    "ACR_34": "acr",
    "ACR_35": "som",
    "ACR_37": "som",
    "ACR_38": "control",
    "ACR_39": "som",
    "ACR_40": "som",
    "ACR_41": "som",
    "ACR_42": "control",
    "ACR_44": "som",
    "ACR_45": "som",
}

som_animals = [sub for sub in sub_exp_types if sub_exp_types[sub] == 'som']
acr_animals = [sub for sub in sub_exp_types if sub_exp_types[sub] == 'acr']
control_animals = [sub for sub in sub_exp_types if sub_exp_types[sub] == 'control']


def get_acr_sub_groups():
    acr_sub_groups = {}
    acr_sub_groups['acr'] = {}
    acr_sub_groups['som'] = {}
    acr_sub_groups['control'] = {}
    for exp_type in acr_sub_groups:
        acr_sub_groups[exp_type]['frontal'] = []
        acr_sub_groups[exp_type]['parietal'] = []
    for subject in swi_subs_exps:
        exp_type = sub_exp_types[subject]
        pl = sub_probe_locations[subject]
        acr_sub_groups[exp_type][pl].append(subject)
    return acr_sub_groups
        

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


def dt_from_tdt(subject, rec, tdt_time):
    rec_times = acr.info_pipeline.subject_info_section(subject, 'rec_times')
    rec_start = pd.Timestamp(rec_times[rec]['start'])
    tdt_time = pd.Timedelta(tdt_time, unit='s')
    return rec_start + tdt_time
    
def dt_to_tdt(subject, rec=None, dt=None):
    """converts datetime to tdt time for a given recording"""

    if dt==None:
        print("Please provide a datetime")
        return None
    if rec==None:
        rec, _ = get_rec_from_dt(subject, dt)
    rec_times = acr.info_pipeline.subject_info_section(subject, 'rec_times')
    rec_start = pd.Timestamp(rec_times[rec]['start'])
    if type(dt) == str:
        dt = pd.Timestamp(dt)
    return (dt - rec_start).total_seconds()
    
def get_rec_from_dt(sub, dt):
    rt = acr.info_pipeline.subject_info_section(sub, 'rec_times')
    for rec in rt.keys():
        if pd.Timestamp(rt[rec]['start']) < dt < pd.Timestamp(rt[rec]['end'] ):
            return rec, pd.Timestamp(rt[rec]['start'])

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

def save_single_sub_exp_fig(subject, exp, filename):
    sub_root = f'/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/plots_presentations_etc/PLOTS_MASTER/single_subject_plots/{subject}'
    exp_root = f'{sub_root}/{exp}'
    if os.path.exists(sub_root) == False:
        os.mkdir(sub_root)
    if os.path.exists(exp_root) == False:
        os.mkdir(exp_root)
    plt.savefig(f'{exp_root}/{filename}.svg', dpi=300, bbox_inches='tight')
    plt.savefig(f'{exp_root}/{filename}.png', dpi=300, bbox_inches='tight')

def add_time_quartiles_to_rebound_df(rebdf):
    rebdf['time_quartile'] = pd.qcut(rebdf['datetime'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    return rebdf

def normalize_bp_df_to_contra(df):
    """Takes a rebound df. Gets the average bp value for each band on each channel of NNXr. Then divides all the other bandpower values for each channel and band by the average NNXr value for that channel and band. 
    
    For whatever you want to look at, essentially sets NNXr to 1.

    Parameters
    ----------
    df : pd.Dataframe
        rebound dataframe.
    """
    df = df.sort_values('channel')
    band_averages = {}
    for band in df['Band'].unique():
        band_averages[band] = df.loc[df['Band']==band].prb('NNXr').groupby('channel')['Bandpower'].mean().to_frame().reset_index().sort_values('channel')
    for band in df['Band'].unique():
        bdf = band_averages[band]
        for channel in df['channel'].unique():
            norm_vals = df.loc[(df['Band']==band) & (df['channel']==channel)]['Bandpower'] / bdf.loc[bdf['channel']==channel]['Bandpower'].values[0]
            df.loc[((df['Band']==band) & (df['channel']==channel)), 'Bandpower'] = norm_vals
    return df


def normalize_fr_df_to_contra(df):
    """Takes a rebound df. Gets the average bp value for each band on each channel of NNXr. Then divides all the other bandpower values for each channel and band by the average NNXr value for that channel and band. 
    
    For whatever you want to look at, essentially sets NNXr to 1.

    Parameters
    ----------
    df : pd.Dataframe
        rebound dataframe.
    """
    df = df.sort_values('channel')
    fr_averages = df.prb('NNXr').groupby('channel')['fr_rel'].mean().to_frame().reset_index().sort_values('channel')
    for channel in df['channel'].unique():
        norm_vals = df.loc[df['channel']==channel]['fr_rel'] / fr_averages.loc[fr_averages['channel']==channel]['fr_rel'].values[0]
        df.loc[df['channel']==channel, 'fr_rel'] = norm_vals
    return df.sort_values(['probe', 'channel'], ascending=False)

def normalize_bp_df_to_contra_chan_group(df):
    """Takes a rebound df. Gets the average bp value for each band on each chan_group of NNXr. Then divides all the other bandpower values for each chan_group and band by the average NNXr value for that chan_group and band. 
    
    For whatever you want to look at, essentially sets NNXr to 1.

    Parameters
    ----------
    df : pd.Dataframe
        rebound dataframe.
    """
    df = df.sort_values('chan_group')
    band_averages = {}
    for band in df['Band'].unique():
        band_averages[band] = df.loc[df['Band']==band].prb('NNXr').groupby('chan_group')['Bandpower'].mean().to_frame().reset_index().sort_values('chan_group')
    for band in df['Band'].unique():
        bdf = band_averages[band]
        for chan_group in df['chan_group'].unique():
            norm_vals = df.loc[(df['Band']==band) & (df['chan_group']==chan_group)]['Bandpower'] / bdf.loc[bdf['chan_group']==chan_group]['Bandpower'].values[0]
            df.loc[((df['Band']==band) & (df['chan_group']==chan_group)), 'Bandpower'] = norm_vals
    return df

def get_sub_exp_combos():
    """
    Returns a LIST of tuples, where each tuple is a 
    (subject, exp) combination from swi_subs_exps.
    """
    sub_exp_combos = []
    for subject in swi_subs_exps:
        for exp in swi_subs_exps[subject]:
            sub_exp_combos.append((subject, exp))
    return sub_exp_combos

def assign_sub_groups_to_df(df):
    """Assigns subject groups (e.g. acr-frontal) to a dataframe with many subjects. Adds sub_group column to df.

    Parameters
    ----------
    df : dataframe
        Dataframe with subject column
    """
    if type(df) == polars.DataFrame:
        df = df.to_pandas()
    sg = get_acr_sub_groups()
    df['sub_group'] = 'None'
    for subject in df['subject'].unique():
        for exp_group in sg.keys():
            for pl in sg[exp_group].keys():
                if subject in sg[exp_group][pl]:
                    subs_group = f'{exp_group}-{pl}'
                    df.loc[df['subject'] == subject, 'sub_group'] = subs_group
                    continue
    return df

def add_chan_groups_to_df(df):
    df['chan_group'] = 101
    chan_groups = [[1, 2, 3, 4], 
                    [5, 6, 7, 8], 
                    [9, 10, 11, 12], 
                    [13, 14, 15, 16]]
    for i, cg in enumerate(chan_groups):
        df.loc[df['channel'].isin(cg), 'chan_group'] = i
    return df

def unify_exp_names_in_df(df):
    df.loc[df['exp']=='swi2', 'exp'] = 'swi'
    df.loc[df['exp']=='swisin2', 'exp'] = 'swisin'
    return df



#==============================================================================================================
### _________________________________ SPOFF QUANTILE THRESHOLDS ___________________________________________ ###
#==============================================================================================================

WAKE_QUANTILE_THRESHOLDS = {
    ("ACR_14", "NNXo", None): 0.005, # Unsure
    ("ACR_14", "NNXr", None): 0.005, # Unsure
    ("ACR_16", "NNXo", None): 0.005, # Unsure
    ("ACR_16", "NNXr", None): 0.005, # Unsure
    ("ACR_17", "NNXo", None): 0.005, # UNSURE
    ("ACR_17", "NNXr", None): 0.005, # UNSURE
    ("ACR_18", "NNXo", None): 0.005, # OK
    ("ACR_18", "NNXr", None): 0.005, # Exclude? Weird activity (as in NREM)
    ("ACR_19", "NNXo", None): 0.005, # OK. Check for possible NREM incursions?
    ("ACR_19", "NNXr", None): 0.005, # OK. Check for possible NREM incursions?
    ("ACR_20", "NNXo", None): 0.005, # OK
    ("ACR_20", "NNXr", None): 0.0075, # OK
    ("ACR_21", "NNXo", None): 0.0075, # OK
    ("ACR_21", "NNXr", None): 0.0075, # OK
    ("ACR_23", "NNXo", None): 0.005, # OK
    ("ACR_23", "NNXr", None): 0.005, # 0.0075
    ("ACR_25", "NNXo", None): 0.0025, # 0K
    ("ACR_25", "NNXr", None): 0.005, # OK
    ("ACR_26", "NNXo", None): 0.0075, # OK
    ("ACR_26", "NNXr", None): 0.0075, # UNSURE
    ("ACR_27", "NNXo", None): 0.005, # UNSURE
    ("ACR_27", "NNXr", None): 0.005, # UNSURE
    ("ACR_28", "NNXo", None): 0.005, # UNSURE
    ("ACR_28", "NNXr", None): 0.005, # UNSURE
    ("ACR_29", "NNXo", None): 0.005, # UNSURE
    ("ACR_29", "NNXr", None): 0.005, # UNSURE
    ("ACR_30", "NNXo", None): 0.005, # UNSURE
    ("ACR_30", "NNXr", None): 0.005, # UNSURE
    ("ACR_31", "NNXo", None): 0.005, # UNSURE
    ("ACR_31", "NNXr", None): 0.005, # UNSURE
    ("ACR_33", "NNXo", None): 0.005, # UNSURE
    ("ACR_33", "NNXr", None): 0.005, # UNSURE
    ("ACR_34", "NNXo", None): 0.005, # UNSURE
    ("ACR_34", "NNXr", None): 0.005, # UNSURE
    ("ACR_35", "NNXr", None): 0.005, # UNSURE
    ("ACR_35", "NNXo", None): 0.005, # UNSURE
}

NREM_QUANTILE_THRESHOLDS = {
    ("ACR_14", "NNXo", None): 0.25, # Unsure
    ("ACR_14", "NNXr", None): 0.25, # Unsure
    ("ACR_16", "NNXo", None): 0.25, # Unsure
    ("ACR_16", "NNXr", None): 0.25, # Unsure
    ("ACR_17", "NNXo", None): 0.15, # UNSURE
    ("ACR_17", "NNXr", None): 0.15, # UNSURE
    ("ACR_18", "NNXo", None): 0.175, # OK
    ("ACR_18", "NNXr", None): 0.25, # Exclude?. Hard.
    ("ACR_19", "NNXo", None): 0.20, # OK
    ("ACR_19", "NNXr", None): 0.175, # OK
    ("ACR_20", "NNXo", None): 0.175, # OK
    ("ACR_20", "NNXr", None): 0.20, # OK
    ("ACR_21", "NNXo", None): 0.175, # OK
    ("ACR_21", "NNXr", None): 0.175, # OK
    ("ACR_23", "NNXo", None): 0.15, # OK
    ("ACR_23", "NNXr", None): 0.15, # OK
    ("ACR_25", "NNXo", None): 0.15, # OK
    ("ACR_25", "NNXr", None): 0.15, # 9/9/24
    ("ACR_26", "NNXo", None): 0.25, # 9/9/24
    ("ACR_26", "NNXr", None): 0.25, # UNSURE
    ("ACR_27", "NNXo", None): 0.25, # UNSURE
    ("ACR_27", "NNXr", None): 0.25, # UNSURE
    ("ACR_28", "NNXo", None): 0.25, # UNSURE
    ("ACR_28", "NNXr", None): 0.25, # UNSURE
    ("ACR_29", "NNXo", None): 0.15, # UNSURE
    ("ACR_29", "NNXr", None): 0.15, # UNSURE
    ("ACR_30", "NNXo", None): 0.25, # UNSURE
    ("ACR_30", "NNXr", None): 0.25, # UNSURE
    ("ACR_31", "NNXo", None): 0.15, # UNSURE
    ("ACR_31", "NNXr", None): 0.15, # UNSURE
    ("ACR_33", "NNXo", None): 0.25, # UNSURE
    ("ACR_33", "NNXr", None): 0.25, # UNSURE
    ("ACR_34", "NNXo", None): 0.25, # UNSURE
    ("ACR_34", "NNXr", None): 0.25, # UNSURE
    ("ACR_35", "NNXr", None): 0.2, # UNSURE
    ("ACR_35", "NNXo", None): 0.2, # UNSURE
    ("ACR_37", "NNXr", None): 0.15, # 9/9/24
    ("ACR_37", "NNXo", None): 0.15, # 9/9/24
    ("ACR_39", "NNXr", None): 0.25, # 10/1/24
    ("ACR_39", "NNXo", None): 0.25, # 10/1/24
    ("ACR_40", "NNXr", None): 0.15, # 10/15/24 (?)
    ("ACR_40", "NNXo", None): 0.15, # 10/15/24 (?)
    ("ACR_41", "NNXr", None): 0.15, # 10/15/24
    ("ACR_41", "NNXo", None): 0.15, # 10/15/24
}

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
    "ACR_26": ["swi"],
    "ACR_27": ["swi"],
    #"ACR_28": ["swi", "swisin"], don't think the data quality was good enough to use this mouse
    "ACR_29": ["swi"],
    "ACR_30": ["swi2"],
    "ACR_31": ["swisin2", "swi2"],
    "ACR_33": ["swisin", "swi"],
    "ACR_34": ["swisin2", "swi"],
    "ACR_35": ["swi"], #don't think I can include the TBS experiment here...
    "ACR_37": ["swi2", "swisin3"],
    "ACR_39": ["swi", "swisin"],
    "ACR_40": ["swi", "swisin"],
    "ACR_41": ["swi", "swisin"],
    "ACR_42": ["swi", "swisin"],
    "ACR_44": ["swi"],
    "ACR_45": ["swi2"],
}

other_exps_of_interest = {
    'ACR_35': ['swisin2'],
    'ACR_37': ['swisin', 'swisin2'],
}

sub_nap_exps = {
    'ACR_39': ['swinap', 'swisinnap'],
    'ACR_40': ['swinap', 'swisinnap'],
    'ACR_41': ['swinap'],
}