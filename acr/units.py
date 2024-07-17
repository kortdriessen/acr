import pandas as pd
import numpy as np
import acr
import acr.info_pipeline as aip
import kdephys.hypno.hypno as kh
import kdephys.units as ku
from acr.hypnogram_utils import standard_hypno_corrections

import pandas as pd
import os
import yaml
import polars as pl
from acr.utils import raw_data_root, materials_root, opto_loc_root


def get_units_to_exclude(subject, sort_id):
    unit_exclusion = yaml.safe_load(open(f"{materials_root}unit_exclusion.yaml", "r"))
    if subject not in unit_exclusion.keys():
        return None
    if sort_id not in unit_exclusion[subject].keys():
        return None
    return unit_exclusion[subject][sort_id]


def sorting_path(subject, sort_id):
    for n in os.listdir(f"/Volumes/opto_loc/Data/{subject}/sorting_data/{sort_id}/"):
        if "ks2_5" in n:
            return f"/Volumes/opto_loc/Data/{subject}/sorting_data/{sort_id}/{n}/"


def save_spike_df(subject, spike_df, sort_id):
    path = f"/Volumes/opto_loc/Data/{subject}/sorting_data/spike_dataframes/{sort_id}.parquet"
    spike_df.to_parquet(path, version="2.6")


def load_and_save_spike_dfs(subject, sort_ids, drop_noise=True, stim=False):
    for si in sort_ids:
        df = single_probe_spike_df(subject, si, drop_noise=drop_noise, stim=stim)
        save_spike_df(subject, df, si)


def save_info_df(info_df, subject, exp):
    """save an info dataframe to the sorting output directory, even if the info dataframe has multiple probes (hence multiple sortings)

    Parameters
    ----------
    info_df : pd.DataFrame
        info dataframe
    subject : str
        subject name
    exp : str
        experiment name
    """
    group_by_probe = info_df.groupby("probe")
    for probe in info_df.prbs():
        probe_df = group_by_probe.get_group(probe)
        sort_id = f"{exp}-{probe}"
        output_dir = sorting_path(subject, sort_id)
        info_df_path = os.path.join(output_dir, "info_df.parquet")
        probe_df.to_parquet(info_df_path, version="2.6")
    return


def generate_info_dataframe(subject, sort_id):
    """generates info_df.parquet for a given subject and sort_id, and if metrics.csv is present in the sorting output directory, it adds quality metrics to the info_df.
    Saves info_df.parquet to the sorting output directory. Should generally only be used when initial spike dataframe is generated.

    Parameters
    ----------
    subject : string
        subject name
    sort_id : string
        sort_id
    """
    output_dir = sorting_path(subject, sort_id)
    info_df = load_raw_info_df_util(subject, sort_id)
    qm_path = os.path.join(output_dir, "metrics.csv")
    if os.path.isfile(qm_path):
        qm_cols = [
            "firing_rate",
            "isi_violations_ratio",
            "isi_violations_rate",
            "isi_violations_count",
            "snr",
        ]
        if all(qm_col in info_df.columns for qm_col in qm_cols):
            pass  # we already have the quality metric columns in the info dataframe, trying to merge them will cause an error (not sure how this happens...)
        else:
            qm_df = pd.read_csv(
                qm_path
            )  # if they are not already there, we add them here.
            info_df = add_qm_to_info_dataframe(info_df, qm_df)
    info_df_path = os.path.join(output_dir, "info_df.parquet")
    info_df.to_parquet(info_df_path, version="2.6")
    return


def load_info_df(subject, sort_ids, exclude_bad_units=True):
    """loads info dataframes for list of sort_ids and concatenates them into one dataframe

    Parameters
    ----------
    subject : str
        subject name
    sort_ids : list or str
        sort_id(s) to use
    exclude_bad_units : bool, optional
        whether to exclude units that are marked as bad in the unit_exclusion.yaml file; DEFAULT = True

    Returns
    -------
    info_df : pandas dataframe
        info dataframe for all sort_ids
    """
    idfs = []
    if type(sort_ids) == str:
        sort_ids = [sort_ids]
    for sort_id in sort_ids:
        output_dir = sorting_path(subject, sort_id)
        info_df_path = os.path.join(output_dir, "info_df.parquet")
        info_df = pd.read_parquet(info_df_path)
        # if exclude_bad_units == True:
        # units_to_exclude = get_units_to_exclude(subject, sort_id)
        # info_df = info_df.loc[~info_df["cluster_id"].isin(units_to_exclude)]
        idfs.append(info_df)
    info_df = pd.concat(idfs)
    return info_df


def load_raw_info_df_util(subject, sort_id):
    """loads a single info dataframe from a sorting directory and does some basic formatting

    Parameters
    ----------
    subject : str
        subject name
    sort_id : str
        sort id

    Returns
    -------
    info_df : pd.DataFrame
        info dataframe
    """
    probe = sort_id.split("-")[-1]
    assert probe == "NNXr" or probe == "NNXo"
    path = sorting_path(subject, sort_id)
    info_path = os.path.join(path, "cluster_info.tsv")
    idf = pd.read_csv(info_path, sep="\t")
    idf = idf.loc[idf.group != "noise"]
    idf.note.fillna("", inplace=True)
    new_note = idf["note"].str.split("/")
    idf["note"] = new_note
    idf["probe"] = probe
    return idf


def add_qm_to_info_dataframe(info_df, qm):
    """merges info dataframe and quality metrics dataframe

    Parameters
    ----------
    info_df : pd.DataFrame
        info dataframe
    qm : pd.DataFrame
        quality metrics dataframe

    Returns
    -------
    info_df : pd.DataFrame
        info dataframe, with snr and isi_violations_rate added
    """
    info_df = info_df.join(qm.set_index("cluster_id"), on="cluster_id")
    info_df.reset_index(inplace=True)
    info_df.drop(
        columns=["isi_violations_ratio", "isi_violations_count", "firing_rate"],
        inplace=True,
    )
    return info_df


def update_data_path_for_phy(subject, sort_id):
    output_dir_path = sorting_path(subject, sort_id)
    params_dot_py_path = os.path.join(output_dir_path, "params.py")
    temp_wh_dot_dat_path = os.path.join(output_dir_path, "temp_wh.dat")
    with open(params_dot_py_path, "r+") as file:
        # Read in all the lines of the file
        lines = file.readlines()

        # Loop through the lines and find the one that defines dat_path
        for i, line in enumerate(lines):
            if "dat_path" in line:
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
    if "spike_dataframes" not in os.listdir(sort_root):
        os.mkdir(sort_root + "spike_dataframes")
    df_path = sort_root + "spike_dataframes/"
    for f in os.listdir(sort_root):
        if os.path.isdir(sort_root + f):  # ensure it is a folder
            if (
                subject in f
            ):  # ensure there is no subject name in the folder name (so that folder name is sort_id)
                print(f"{f} is misnamed")
                continue
            if f == "spike_dataframes":  # ensure it is not the spike_dataframes folder
                continue
            if f"{f}.parquet" in os.listdir(df_path):  # ensure it is not already saved
                print(f"{f} already saved")
                continue
            if f"{f}.parquet" not in os.listdir(
                df_path
            ):  # if not saved, generate and save it
                print(f"loading {f}")
                df = single_probe_spike_df(subject, f, drop_noise=drop_noise, stim=stim)
                save_spike_df(subject, df, f)
                update_data_path_for_phy(subject, f)
                generate_info_dataframe(subject, f)
                print(f"saved {f}")


def get_sorting_recs(subject, sort_id):
    ss = pd.read_excel("/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/spikesorting.xlsx")
    ss_narrowed = ss.loc[np.logical_and(ss.subject == subject, ss.sort_id == sort_id)]
    if type(ss_narrowed.recording_end_times.values[0]) == int:
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
    df_path = f"{df_root_path}/{sort_id}.parquet"
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
    spike_df["probe"] = sort_id.split("-")[-1]
    return spike_df


def get_time_info(subject, sort_id):
    ss = pd.read_excel("/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/spikesorting.xlsx")
    ss_narrowed = ss.loc[np.logical_and(ss.subject == subject, ss.sort_id == sort_id)]
    if type(ss_narrowed.recording_end_times.values[0]) == int:
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
    if "swi-N" in sort_id:
        stim = False
    if stim:
        print("adding stim info")
        spike_df = add_stim_info(spike_df, subject)
    print("adding hypno")
    # spike_df = add_hypno(spike_df, subject, recordings)
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
    notes_df = pd.DataFrame.from_dict(notes_dic, orient="index")
    notes_df.columns = ["note"]
    group_labels = []
    for i in df.cid_un():
        group_labels.append(df.cid(i).group.values_host[0])
    notes_df["group"] = group_labels
    notes_df.index.name = "cluster_id"
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
            fr[probe][clus_name] = len(df.prb(probe).cid(clus)) / total_time
    return fr


def get_fr_suppression_by_cluster(df, pons, poffs, probes=["NNXr", "NNXo"]):
    total_spike_rate = {}
    total_pulse_on_time = 0

    for on, off in zip(
        pons, poffs
    ):  # this should get us an accurate measure for the total time that the pulses were on, which shouldn't vary by cluster or probe
        total_pulse_on_time += (off - on).total_seconds()

    for probe in probes:
        clusters = df.prb(probe).cid_un()
        total_spike_rate[probe] = {}
        for clus in clusters:
            new_df = df.prb(probe).cid(clus)
            clus_name = str(clus)
            total_spike_rate[probe][clus_name] = 0

            for i in np.arange(0, len(pons)):
                total_spike_rate[probe][clus_name] += (
                    new_df.ts(pons[i], poffs[i]).prb(probe).cluster_id.count()
                )  # this loops through every pulse, and adds the number of spikes during that pulse to the total spike count for that cluster

            total_spike_rate[probe][clus_name] = (
                total_spike_rate[probe][clus_name] / total_pulse_on_time
            )
            del new_df

    return total_spike_rate
