import pandas as pd
import numpy as np
import acr
import acr.info_pipeline as aip
import kdephys.hypno.hypno as kh
import kdephys.units as ku
from acr.hypnogram_utils import standard_hypno_corrections

import pandas as pd
import os

from on_off_detection import OnOffModel
import on_off_detection as ood

def sorting_path(subject, sort_id):
    for n in os.listdir(f"/Volumes/opto_loc/Data/{subject}/sorting_data/{sort_id}/"):
        if 'ks2_5' in n:
            return f"/Volumes/opto_loc/Data/{subject}/sorting_data/{sort_id}/{n}/"

def save_spike_df(subject, spike_df, sort_id):
    path = f"/Volumes/opto_loc/Data/{subject}/sorting_data/spike_dataframes/{sort_id}.parquet"
    spike_df.to_parquet(path, version="2.6")

def load_and_save_spike_dfs(subject, sort_ids, drop_noise=True, stim=False):
    for si in sort_ids:
        df = single_probe_spike_df(subject, si, drop_noise=drop_noise, stim=stim)
        save_spike_df(subject, df, si)

def update_data_path_for_phy(subject, sort_id):
    output_dir_path = sorting_path(subject, sort_id)
    params_dot_py_path = os.path.join(output_dir_path, 'params.py')
    temp_wh_dot_dat_path = os.path.join(output_dir_path, 'temp_wh.dat')
    with open(params_dot_py_path, 'r+') as file:
        # Read in all the lines of the file
        lines = file.readlines()

        # Loop through the lines and find the one that defines dat_path
        for i, line in enumerate(lines):
            if 'dat_path' in line:
                # Modify the line to set dat_path to a new value
                lines[i] = f"dat_path = '{temp_wh_dot_dat_path}'\n"
                break

        # Move the file pointer to the beginning of the file and overwrite the file with the modified lines
        file.seek(0)
        file.writelines(lines)

    # Close the file
    file.close()

def save_all_spike_dfs(subject, drop_noise=True, stim=False):
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
                df = single_probe_spike_df(subject, f, drop_noise=drop_noise, stim=stim)
                save_spike_df(subject, df, f)
                update_data_path_for_phy(subject, f)
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

def load_single_sorting_df(subject, sort_id, drop=None):
    """
    Load a single saved sorting (which has already been saved as a parquet file)
    

    Args:
        - subject (str): subject name. 
        - sort_id (str): specific sort_id to load.
        - drop (list): list of columns to drop from the dataframe. Defaults to None.

    Returns:
        spike_df: spike dataframe (pandas)
    """
    df_root_path = f"/Volumes/opto_loc/Data/{subject}/sorting_data/spike_dataframes"
    df_path = f'{df_root_path}/{sort_id}.parquet'
    spike_df = pd.read_parquet(df_path)
    if drop is not None:
        spike_df = spike_df.drop(drop, axis=1)
    return spike_df


def info_to_spike_df(spike_df, info, sort_id):
    for cluster_id in info.cluster_id.values:
        group = info[info.cluster_id == cluster_id].group.values[0]
        note = info[info.cluster_id == cluster_id].note.values[0]
        channel = info[info.cluster_id == cluster_id].ch.values[0]
        amp = info[info.cluster_id == cluster_id].amp.values[0]
        amplitude = info[info.cluster_id == cluster_id].Amplitude.values[0]
        spike_df.loc[spike_df["cluster_id"] == cluster_id, "group"] = group
        spike_df.loc[spike_df["cluster_id"] == cluster_id, "note"] = note
        spike_df.loc[spike_df["cluster_id"] == cluster_id, "channel"] = channel
        spike_df.loc[spike_df["cluster_id"] == cluster_id, "amp"] = amp
        spike_df.loc[spike_df["cluster_id"] == cluster_id, "Amplitude"] = amplitude
    spike_df.note.fillna("", inplace=True)
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
    #spike_df = add_hypno(spike_df, subject, recordings)
    return spike_df

def get_cluster_notes(df):
    """returns a dataframe with the associated notes and group label for each cluster in a unit dataframe

    Args:
        df (_type_): the unit dataframe, note: should be probe-specific!
    """
    notes_df = pd.DataFrame()
    notes_dic = {}
    for i in df.cid_un():
        note_val = df.cid(i).note.values_host[0]
        notes_dic[str(i)] = note_val
    notes_df = pd.DataFrame.from_dict(notes_dic, orient='index')
    notes_df.columns = ['note']
    group_labels = []
    for i in df.cid_un():
        group_labels.append(df.cid(i).group.values_host[0])
    notes_df['group'] = group_labels
    notes_df.index.name = 'cluster_id'
    return notes_df

def get_fr_by_cluster(df):
    """loops through each cluster on each probe in the df, and returns a dictionary of the firing rate for each cluster

    Args:
        df (_type_): spike dataframe
    """
    fr = {}
    probes = list(df.probe.unique().values_host)
    for probe in probes:
        fr[probe] = {}
        clusters = df.prb(probe).cid_un()
        start = pd.Timestamp(df.prb(probe).datetime.min())
        end = pd.Timestamp(df.prb(probe).datetime.max())
        total_time = (end - start).total_seconds()
        for clus in clusters:
            clus_name = str(clus)
            fr[probe][clus_name] = len(df.prb(probe).cid(clus))/total_time
    return fr

def get_fr_suppression_by_cluster(df, pons, poffs, probes=['NNXr', 'NNXo']):
    total_spike_rate = {}
    total_pulse_on_time = 0
    
    for on, off in zip(pons, poffs): # this should get us an accurate measure for the total time that the pulses were on, which shouldn't vary by cluster or probe
        total_pulse_on_time += (off - on).total_seconds()
    
    for probe in probes:
        clusters = df.prb(probe).cid_un()
        total_spike_rate[probe] = {}
        for clus in clusters:
            new_df = df.prb(probe).cid(clus)
            clus_name = str(clus)
            total_spike_rate[probe][clus_name] = 0
            
            for i in np.arange(0, len(pons)):
                total_spike_rate[probe][clus_name] += new_df.ts(pons[i], poffs[i]).prb(probe).cluster_id.count() #this loops through every pulse, and adds the number of spikes during that pulse to the total spike count for that cluster
            
            total_spike_rate[probe][clus_name] = total_spike_rate[probe][clus_name]/total_pulse_on_time
            del new_df

    return total_spike_rate


# ----------------------------------------- ON-OFF DETECTION ----------------------------------------------------------------------------------------
def load_hypno_for_ood(subject, sort_id, state='NREM'):
    recs, st, durs = acr.units.get_time_info(subject, sort_id)
    hyps = {}
    running_dur = 0
    for i, rec in enumerate(recs):
        hyp = acr.io.load_hypno(subject, rec)
        if hyp is None:
            hyps[rec] = hyp
            running_dur += durs[i-1]
            continue
        hyp = hyp.as_float()
        if i == 0:
            hyps[rec] = hyp
            continue
        running_dur += durs[i-1]
        hyp['start_time'] = hyp.start_time.values + running_dur
        hyp['end_time'] = hyp.end_time.values + running_dur
        hyps[rec] = hyp
    df = pd.concat(hyps)
    df = df.reset_index(drop=True)
    df = standard_hypno_corrections(df)
    df_nrem = df.loc[df.state == state]
    return df_nrem.reset_index(drop=True)
    
def load_spike_trains_for_ood(subject, sort_id):
    #TODO: choose only units with certain note values...??
    path = acr.units.sorting_path(subject, sort_id)
    sort_extractor, info = ku.io.load_sorting_extractor(path, drop_noise=True)
    cluster_ids = sort_extractor.unit_ids
    trains = []
    for cluster in cluster_ids: #gets list of spike trains for each unit
        trn = sort_extractor.get_unit_spike_train(cluster)
        trn = trn / 24414.0625 # gives the spike times in seconds instead of sample numbers
        trains.append(trn)
    return trains, info, cluster_ids


def run_ood(trains, clust_ids, hypno, tmax=None, save=False):
    ood.HMMEM_PARAMS["init_state_estimate_method"] = "conservative"
    if tmax is None:
        tmax = int(hypno.end_time.max()+1)
    GLOBAL_ON_OFF_DETECTION_PARAMS = {
    "binsize": 0.01,  # (s) (Discrete algorithm)
    "history_window_nbins": 3,  # Size of history window IN BINS
    "n_iter_EM": 200,  # Number of iterations for EM
    "n_iter_newton_ralphson": 100,
    "init_A": np.array(
        [[0.1, 0.9], [0.01, 0.99]]
    ),  # Initial transition probability matrix
    "init_mu": None,  # ~ OFF rate. Fitted to data if None
    "init_alphaa": None,  # ~ difference between ON and OFF rate. Fitted to data if None
    "init_betaa": None,  # ~ Weight of recent history firing rate. Fitted to data if None,
    "gap_threshold": .04,  # Merge active states separated by less than gap_threhsold
    }

    mod = OnOffModel(trains, tmax, clust_ids, method='hmmem', params=GLOBAL_ON_OFF_DETECTION_PARAMS, bouts_df=hypno)
    oodf = mod.run()
    return oodf

def assign_recordings_to_oodf(oodf, recordings, durations):
    prev_split = 0
    for r, d in zip(recordings, durations):
        border1 = prev_split
        border2 = prev_split + d
        oodf.loc[
            np.logical_and(oodf.end_time > border1, oodf.end_time <= border2),
            "recording",
        ] = r
        prev_split += d
    total_duration = np.sum(durations)
    oodf.loc[oodf.end_time > total_duration, "recording"] = recordings[-1]
    return oodf

def assign_datetimes_to_oodf(oodf, recordings, start_times):
    dti_starts = pd.DatetimeIndex([])
    dti_ends = pd.DatetimeIndex([])
    for r, s in zip(recordings, start_times):
        rec_df = oodf.loc[oodf.recording == r]
        if rec_df.empty:
            print(f"Nothing for {r}, skipping")
            continue
        starts = rec_df.start_time.values
        ends = rec_df.end_time.values
        
        start_times_rel = starts - starts[0]
        end_times_rel = ends - ends[0]
        start_timedeltas = pd.to_timedelta(start_times_rel, unit="s")
        end_timedeltas = pd.to_timedelta(end_times_rel, unit="s")
        start_datetimes = s + start_timedeltas
        end_datetimes = s + end_timedeltas
        dti_starts = dti_starts.append(start_datetimes)
        dti_ends = dti_ends.append(end_datetimes)
    oodf["start_datetime"] = dti_starts
    oodf["end_datetime"] = dti_ends
    return oodf

def process_oodf(oodf, subject, sort_id, load=False):
    if load:
        oodf = acr.units.load_oodf(subject=subject, sort_id=sort_id)
    recs, starts, durations = acr.units.get_time_info(subject, sort_id)
    oodf = assign_recordings_to_oodf(oodf, recs, durations)
    oodf = assign_datetimes_to_oodf(oodf, recs, starts)
    oodf['probe'] = sort_id.split('-')[-1]
    if load:
        acr.units.save_oodf(subject, sort_id, oodf)
    else:
        return oodf

def save_oodf(subject, sort_id, oodf):
    # First we make sure the on-off_dataframes folder exists
    sort_root = f"/Volumes/opto_loc/Data/{subject}/sorting_data/"
    if 'on-off_dataframes' not in os.listdir(sort_root):
        os.mkdir(sort_root+'on-off_dataframes')
    # Then we save the dataframe
    path = f"/Volumes/opto_loc/Data/{subject}/sorting_data/on-off_dataframes/{sort_id}.json"
    oodf.to_json(path, orient='records', lines=True)

def load_oodf(subject, sort_id=[]):
    """_summary_

    Args:
        subject (str): subject name
        sort_id (list, optional): list of sort_id's Defaults to [].
    """
    path = f"/Volumes/opto_loc/Data/{subject}/sorting_data/on-off_dataframes/"
    sdfs = []
    for sid in sort_id:
        key = sid + ".json"
        oodf = pd.read_json(path + key, lines=True)
        oodf.drop(columns=['start_datetime', 'end_datetime'], inplace=True)
        oodf = process_oodf(oodf, subject, sid)
        sdfs.append(oodf)
    oodfs = pd.concat(sdfs)
    return oodfs