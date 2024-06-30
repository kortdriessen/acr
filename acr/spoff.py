#from offproj import core, tdt_core
import acr
import numpy as np
import pandas as pd
from acr.utils import swi_subs_exps, sub_probe_locations, sub_exp_types
import matplotlib.pyplot as plt
import seaborn as sns

def assign_recordings_to_off_df(off_df, recordings, durations):
    prev_split = 0
    for r, d in zip(recordings, durations):
        border1 = prev_split
        border2 = prev_split + d
        off_df.loc[
            np.logical_and(off_df['start_time'] > border1, off_df['end_time'] <= border2),
            "recording",
        ] = r
        prev_split += d
    total_duration = np.sum(durations)
    off_df.loc[off_df['start_time'] > total_duration, "recording"] = recordings[-1]
    return off_df

def assign_datetimes_to_off_df(off_df, recordings, start_times, durations):
    dti_starts = pd.DatetimeIndex([])
    dti_ends = pd.DatetimeIndex([])
    current_rec_zero = 0
    for r, s, d in zip(recordings, start_times, durations):
        
        if r not in off_df['recording'].unique():
            continue
        
        rec_df = off_df.loc[off_df.recording == r]
        start_times = rec_df['start_time'].values
        end_times = rec_df['end_time'].values
        
        assert np.min(start_times) == start_times[0]
        assert np.min(end_times) == end_times[0]
        
        
        start_times_rel = start_times - current_rec_zero
        end_times_rel = end_times - current_rec_zero
        
        current_rec_zero += d
        
        start_timedeltas = pd.to_timedelta(start_times_rel, unit="s")
        end_timedeltaa = pd.to_timedelta(end_times_rel, unit="s")
        start_datetimes = s + start_timedeltas
        end_datetimes = s + end_timedeltaa
        
        dti_starts = dti_starts.append(start_datetimes)
        dti_ends = dti_ends.append(end_datetimes)

    off_df["start_datetime"] = dti_starts
    off_df["end_datetime"] = dti_ends
    return off_df

def complete_exp_off(subject, exp, probes=['NNXr', 'NNXo'], sort_id=None, structure=None, which='ap', sensible_filters=True):
    if sort_id is None:
        sort_id = f'{exp}-{probes[0]}'
    
    recs, starts, durations = acr.units.get_time_info(subject, sort_id)
    
    off_dfs_by_probe = {}
    for probe in probes:
        off = core.get_offs(subject=subject, prb=probe, structure=structure, which=which, experiment=exp)
        off['probe'] = probe
        off = assign_recordings_to_off_df(off, recs, durations)
        off = assign_datetimes_to_off_df(off, recs, starts, durations)
        off_dfs_by_probe[probe] = off
    offdf = pd.concat(off_dfs_by_probe.values())
    if sensible_filters:
        offdf = offdf.loc[(offdf['median_duration']>.04) & (offdf['median_duration']<0.8)]
    return offdf

def get_current_off_processing():
    cur_offs = pd.DataFrame(columns=['subject', 'exp', 'result_nnxo', 'result_nnxr'])
    for sub in swi_subs_exps:
        for exp in swi_subs_exps[sub]:
            off_df_nnxo = core.get_offs(subject=sub, prb='NNXo', structure=None, which='ap', experiment=exp)
            if not off_df_nnxo.empty:
                cur_offs = cur_offs.append({'subject': sub, 'exp': exp, 'result_nnxo': 'present'}, ignore_index=True)
            else:
                cur_offs = cur_offs.append({'subject': sub, 'exp': exp, 'result_nnxo': 'absent'}, ignore_index=True)
        
    for sub in swi_subs_exps:
        for exp in swi_subs_exps[sub]:
            off_df_nnxr = core.get_offs(subject=sub, prb='NNXr', structure=None, which='ap', experiment=exp)
            if not off_df_nnxr.empty:
                cur_offs.loc[(cur_offs['subject'] == sub) & (cur_offs['exp'] == exp), 'result_nnxr'] = 'present'
            else:
                cur_offs.loc[(cur_offs['subject'] == sub) & (cur_offs['exp'] == exp), 'result_nnxr'] = 'absent'
    f, ax = plt.subplots(1, 1, figsize=(5, 10))
    sns.scatterplot(data=cur_offs, x='exp', y='subject', hue='result_nnxo', hue_order=['present', 'absent'], palette=['green', 'red'], s=80, ax=ax)
    return f, ax