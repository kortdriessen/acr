import pandas as pd
import numpy as np
import acr
import acr.info_pipeline as aip
import kdephys.hypno as kh
import kdephys.units as ku


def sorting_path(subject, sort_id, analysis_name="ks2_5_nblocks=1_8s-batches"):
    return f"/Volumes/opto_loc/Data/{subject}/sorting_data/{sort_id}/{analysis_name}/"


def info_to_spike_df(spike_df, info, sort_id):
    for cluster_id in info.cluster_id.values:
        group = info[info.cluster_id == cluster_id].group.values[0]
        note = info[info.cluster_id == cluster_id].note.values[0]
        spike_df.loc[spike_df["cluster_id"] == cluster_id, "group"] = group
        spike_df.loc[spike_df["cluster_id"] == cluster_id, "note"] = note
    spike_df["sort_id"] = sort_id
    return spike_df


def get_time_info(subject, sort_id):
    ss = pd.read_excel("/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/spikesorting.xlsx")
    ss_narrowed = ss.loc[np.logical_and(ss.subject == subject, ss.sort_id == sort_id)]
    times = ss_narrowed.recording_end_times.values[0].split(",")
    times = [int(t.strip()) for t in times]
    recs = ss_narrowed.recordings.values[0].split(",")
    recs = [r.strip() for r in recs]
    info = aip.load_subject_info(subject)
    times_new = []
    probe = sort_id.split("-")[1]

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
    subject, sort_id, analysis_name="ks2_5_nblocks=1_8s-batches", drop_noise=True
):
    path = acr.units.sorting_path(subject, sort_id, analysis_name)
    sort_extractor, info = ku.io.load_sorting_extractor(path, drop_noise=drop_noise)
    spike_df = ku.io.spikeinterface_sorting_to_dataframe(sort_extractor)
    spike_df = info_to_spike_df(spike_df, info, sort_id)
    recordings, start_times, durations = get_time_info(subject, sort_id)
    spike_df = assign_recordings_to_spike_df(spike_df, recordings, durations)
    spike_df = assign_datetimes_to_spike_df(spike_df, recordings, start_times)
    spike_df = add_stim_info(spike_df, subject)
    spike_df = add_hypno(spike_df, subject, recordings)
    return spike_df
