from offproj import core, tdt_core
import acr
import numpy as np
import pandas as pd
from acr.utils import swi_subs_exps, sub_probe_locations, sub_exp_types
import matplotlib.pyplot as plt
import seaborn as sns
import os

def assign_datetimes_to_off_df(offs, sorting_start):
    starts = offs['start_time'].values
    ends = offs['end_time'].values
    starts_td = pd.to_timedelta(starts, unit='s')
    ends_td = pd.to_timedelta(ends, unit='s')
    starts_dt = sorting_start + starts_td
    ends_dt = sorting_start + ends_td
    offs['start_datetime'] = starts_dt
    offs['end_datetime'] = ends_dt
    return offs

def assign_recordings_to_off_df(df, recs, starts, durations):
    for rec, start, duration in zip(recs, starts, durations):
        start = pd.Timestamp(start)
        end = start + pd.Timedelta(duration, unit='s')
        df.loc[df['start_datetime'].between(start, end), 'rec'] = rec
    return df


def load_complete_exp_off(subject, exp, probes=['NNXr', 'NNXo'], sort_id=None, structure=None, which='ap', sensible_filters=True, relative=False, rel_method='mean'):
    if sort_id is None:
        sort_id = f'{exp}-{probes[0]}'
    
    recs, starts, durations = acr.units.get_time_info(subject, sort_id)
    
    off_dfs_by_probe = {}
    for probe in probes:
        off = core.get_offs(subject=subject, prb=probe, structure=structure, which=which, experiment=exp)
        off['probe'] = probe
        sorting_start = pd.Timestamp(starts[0])
        off = assign_datetimes_to_off_df(off, sorting_start)
        off = assign_recordings_to_off_df(off, recs, starts, durations)
        
        off_dfs_by_probe[probe] = off
    offdf = pd.concat(off_dfs_by_probe.values())
    offdf.sort_values('start_datetime', inplace=True)
    if sensible_filters:
        offdf = offdf.loc[(offdf['median_duration']>.04) & (offdf['median_duration']<0.8)]
        offdf = offdf.loc[(offdf['duration']>.04) & (offdf['duration']<0.8)]
    if relative:
        offdf = make_odf_relative(offdf, method=rel_method)
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

def make_odf_relative(odf, method='mean'):
    if method == 'mean':
        avgs = odf.groupby(['probe', 'descriptor']).mean().reset_index()
    elif method == 'median':
        avgs = odf.groupby(['probe', 'descriptor']).median().reset_index()
    else:
        raise ValueError('method must be mean or median')
    avgs = avgs.loc[avgs['descriptor']=='Early_Baseline_NREM']
    odf.loc[odf['probe']=='NNXo', 'median_duration'] = odf.prb('NNXo')['median_duration']/avgs.prb('NNXo')['median_duration'].values[0]
    odf.loc[odf['probe']=='NNXo', 'duration'] = odf.prb('NNXo')['duration']/avgs.prb('NNXo')['duration'].values[0]
    odf.loc[odf['probe']=='NNXr', 'median_duration'] = odf.prb('NNXr')['median_duration']/avgs.prb('NNXr')['median_duration'].values[0]
    odf.loc[odf['probe']=='NNXr', 'duration'] = odf.prb('NNXr')['duration']/avgs.prb('NNXr')['duration'].values[0]
    return odf

def check_if_off_detection_is_done(subject, exp, probe, dates=['2024-07-29', '2024-07-30']):
    data_folder = f'/Volumes/npx_nfs/nobak/offproj/{exp}/{subject}'
    txt_files = []
    if not os.path.exists(data_folder):
        return []
    for f in os.listdir(data_folder):
        if f.endswith('.txt'):
            if 'SUCCESS' in f:
                if probe in f:
                    txt_files.append(f)
    processed_conditions = []
    for f in txt_files:
        cond = f.split('.')[2]
        date = f.split('--')[1].split('.')[0]
        if date in dates:
            processed_conditions.append(cond)        
    return processed_conditions