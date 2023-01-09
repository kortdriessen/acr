import pandas as pd
import numpy as np
import acr
import acr.info_pipeline as aip
import kdephys.hypno.hypno as kh
import kdephys.units as ku

import pandas as pd
import os

def save_spike_df(subject, spike_df, sort_id):
    path = f"/Volumes/opto_loc/Data/{subject}/sorting_data/spike_dataframes/{sort_id}.parquet"
    spike_df.to_parquet(path, version="2.6")

def load_and_save_spike_dfs(subject, sort_ids, drop_noise=True, stim=False):
    for si in sort_ids:
        df = single_probe_spike_df(subject, si, drop_noise=drop_noise, stim=stim)
        save_spike_df(subject, df, si)

def save_all_spike_dfs(subject, drop_noise=True, stim=True):
    """Looks through the subject's sorting_data folder and generates + saves all spike dataframes to the spike_dataframes folder which are not already present there."""
    sort_root = f"/Volumes/opto_loc/Data/{subject}/sorting_data/"
    if 'spike_dataframes' not in os.listdir(sort_root):
        os.mkdir(sort_root + 'spike_dataframes')
    df_path = sort_root + 'spike_dataframes/'
    for f in os.listdir(sort_root):
        if os.path.isdir(sort_root + f): #ensure it is a folder
            if subject in f: #ensure there is no subject name in the folder name (so that folder name is sort_id)
                print(f'{f} is misnamed')
                continue
            if f == 'spike_dataframes': #ensure it is not the spike_dataframes folder
                continue
            if f'{f}.parquet' in os.listdir(df_path): # ensure it is not already saved
                print(f'{f} already saved')
                continue
            if f'{f}.parquet' not in os.listdir(df_path): # if not saved, generate and save it
                print(f'loading {f}')
                stim = False if 'swi' in f else True
                df = single_probe_spike_df(subject, f, drop_noise=drop_noise, stim=stim)
                save_spike_df(subject, df, f)
                print(f'saved {f}')

def get_sorting_recs(subject, sort_id):
    ss = pd.read_excel("/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/spikesorting.xlsx")
    ss_narrowed = ss.loc[np.logical_and(ss.subject == subject, ss.sort_id == sort_id)]
    if type( ss_narrowed.recording_end_times.values[0] ) == int:
        recs = [ss_narrowed.recordings.values[0]]
    else:
        recs = ss_narrowed.recordings.values[0].split(",")
        recs = [r.strip() for r in recs]
    return recs

def load_spike_dfs(subject, sort_id=None):
    """
    Load sorted spike dataframes
    if sort_id is specified, only load that one
    if sort_id is not specified, load all in sorting_data/spike_dataframes folder

    Args:
        subject (str): subject name
        sort_id (optional): specific sort_id to load. Defaults to None.

    Returns:
        spikes_df: spike dataframe or dictionary of spike dataframes, depending on sort_id
    """
    path = f"/Volumes/opto_loc/Data/{subject}/sorting_data/spike_dataframes/"
    if sort_id:
        key = sort_id + ".parquet"
        spike_dfs = pd.read_parquet(path + key)
    else:
        spike_dfs = {}
        for f in os.listdir(path):
            sort_id = f.split(".")[0]
            spike_dfs[sort_id] = pd.read_parquet(path + f)
    return spike_dfs


def sorting_path(subject, sort_id):
    for n in os.listdir(f"/Volumes/opto_loc/Data/{subject}/sorting_data/{sort_id}/"):
        if 'batch' in n:
            return f"/Volumes/opto_loc/Data/{subject}/sorting_data/{sort_id}/{n}/"

def info_to_spike_df(spike_df, info, sort_id):
    for cluster_id in info.cluster_id.values:
        group = info[info.cluster_id == cluster_id].group.values[0]
        note = info[info.cluster_id == cluster_id].note.values[0]
        channel = info[info.cluster_id == cluster_id].ch.values[0]
        spike_df.loc[spike_df["cluster_id"] == cluster_id, "group"] = group
        spike_df.loc[spike_df["cluster_id"] == cluster_id, "note"] = note
        spike_df.loc[spike_df["cluster_id"] == cluster_id, "channel"] = channel
    spike_df.note.fillna("", inplace=True)
    spike_df['exp'] = '-'.join(sort_id.split('-')[:-1])
    spike_df['probe'] = sort_id.split('-')[-1]
    return spike_df


def get_time_info(subject, sort_id):
    ss = pd.read_excel("/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/spikesorting.xlsx")
    ss_narrowed = ss.loc[np.logical_and(ss.subject == subject, ss.sort_id == sort_id)]
    if type( ss_narrowed.recording_end_times.values[0] ) == int:
        times = [ss_narrowed.recording_end_times.values[0]]
        recs = [ss_narrowed.recordings.values[0]]
    else:
        times = ss_narrowed.recording_end_times.values[0].split(",")
        times = [int(t.strip()) for t in times]
        recs = ss_narrowed.recordings.values[0].split(",")
        recs = [r.strip() for r in recs]
    info = aip.load_subject_info(subject)
    times_new = []
    
    # TODO fix this!!
    probe = sort_id.split("-")[-1]
    assert "NNX" in probe
    

    starts = []
    for t, r in zip(times, recs):
        if t != 0:
            times_new.append(t)
        if t == 0:
            duration = info["rec_times"][r][f"{probe}-duration"]
            times_new.append(duration)
        starts.append(np.datetime64(info["rec_times"][r]["start"]))
    return recs, starts, times_new


def assign_recordings_to_spike_df(spike_df, recordings, durations):
    prev_split = 0
    for r, d in zip(recordings, durations):
        border1 = prev_split
        border2 = prev_split + d
        spike_df.loc[
            np.logical_and(spike_df.time > border1, spike_df.time <= border2),
            "recording",
        ] = r
        prev_split += d
    total_duration = np.sum(durations)
    spike_df.loc[spike_df.time > total_duration, "recording"] = recordings[-1]
    return spike_df


def assign_datetimes_to_spike_df(spike_df, recordings, start_times):
    dti = pd.DatetimeIndex([])
    for r, s in zip(recordings, start_times):
        rec_df = spike_df.loc[spike_df.recording == r]
        times = rec_df.time.values
        assert np.min(times) == times[0]
        times_rel = times - times[0]
        timedeltas = pd.to_timedelta(times_rel, unit="s")
        datetimes = s + timedeltas
        dti = dti.append(datetimes)
    spike_df["datetime"] = dti
    return spike_df


def add_stim_info(spike_df, subject):
    info = aip.load_subject_info(subject)
    stim_exps = list(info["stim-exps"].keys())
    stim_info = info["stim_info"]

    for rec in np.unique(spike_df.recording.values):
        if rec in stim_exps:
            stim_type = info["stim-exps"][rec]
            if len(stim_type) == 1:
                stim_type = stim_type[0]
            elif len(stim_type) > 1:
                print("More than one stim type for this recording")
                continue
            onsets = stim_info[rec][stim_type]["onsets"]
            onsets = [np.datetime64(o) for o in onsets]
            offsets = stim_info[rec][stim_type]["offsets"]
            offsets = [np.datetime64(o) for o in offsets]
            for on, off in zip(onsets, offsets):
                spike_df.loc[
                    np.logical_and(spike_df.datetime > on, spike_df.datetime < off),
                    "stim",
                ] = 1
    return spike_df


def add_hypno(spike_df, subject, recordings):
    states = pd.Series()
    for rec in recordings:
        if not acr.io.check_for_hypnos(subject, rec):
            dt = spike_df.loc[spike_df.recording == rec].datetime.values
            state_values = kh.no_states_array(dt)
            states = pd.concat([states, state_values])
        else:
            hyp = acr.io.load_hypno(subject, rec)
            dt = spike_df.loc[spike_df.recording == rec].datetime.values
            state_values = kh.get_states(hyp, dt)
            states = pd.concat([states, state_values])
    spike_df["state"] = states.values
    return spike_df


def single_probe_spike_df(
    subject,
    sort_id,
    drop_noise=True,
    stim=True,
):
    path = acr.units.sorting_path(subject, sort_id)
    sort_extractor, info = ku.io.load_sorting_extractor(path, drop_noise=drop_noise)
    spike_df = ku.io.spikeinterface_sorting_to_dataframe(sort_extractor)
    spike_df = info_to_spike_df(spike_df, info, sort_id)
    recordings, start_times, durations = get_time_info(subject, sort_id)
    print("assigning recordings")
    spike_df = assign_recordings_to_spike_df(spike_df, recordings, durations)
    print("assigning datetimes")
    spike_df = assign_datetimes_to_spike_df(spike_df, recordings, start_times)
    if stim:
        print("adding stim info")
        spike_df = add_stim_info(spike_df, subject)
    print("adding hypno")
    spike_df = add_hypno(spike_df, subject, recordings)
    return spike_df
