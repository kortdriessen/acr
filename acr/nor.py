import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import os
import yaml
import numpy as np
#plt.style.use('/home/kdriessen/gh_master/acr/acr/plot_styles/master_plots.mplstyle')
plt.style.use('fast')

stim_subjects = [
    #"NOR_37", # had to exclude because acquisition was greater than 2/3 on one side 
    #"NOR_38", had to exclude, bad ferrules so stim didn't work 
    "NOR_39", 
    "NOR_40", 
    "NOR_41", 
    #"NOR_42", # had to exclude because acquisition was greater than 2/3 on one side 
    #"NOR_44", # had to exclude because acquisition was greater than 2/3 on one side 
    #"NOR_45", # had to exclude because acquisition was greater than 2/3 on one side 
    #"NOR_46", excluding because ferrule was loose during stim 
    "NOR_47", 
    "NOR_48", 
    "NOR_49", 
    "NOR_50", 
    #"NOR_51", # had to exclude because acquisition was greater than 2/3 on one side 
    #"NOR_53", # had to exclude because acquisition was greater than 2/3 on one side 
    "NOR_54",
]


sd_subjects = [
    "NOR_11",
    #"NOR_12",  # excluded because acquisition was greater than 2/3 on one side
    "NOR_15",
    "NOR_16",
    "NOR_19",
    "NOR_20",
    "NOR_23",
    "NOR_24",
    "NOR_27",
    "NOR_28",
    "NOR_31",
    "NOR_32",
    "NOR_35",
    "NOR_36",  
]

sleep_subjects = [
    "NOR_9",
    "NOR_10",
    "NOR_13",
    #"NOR_14", # excluded because < 50% sleep
    "NOR_17",
    "NOR_18",
    "NOR_21",
    "NOR_22",
    "NOR_25",
    #"NOR_26", # excluded because acquisition was greater than 2/3 on one side 
    #"NOR_29", # excluded because < 50% sleep
    "NOR_30",
    #"NOR_33", # excluded because < 50% sleep
    #"NOR_34", # excluded because < 50% sleep 
]

def get_all_subjects():
    return stim_subjects + sd_subjects + sleep_subjects

def get_subject_type(subject):
    if subject in stim_subjects:
        return 'stim'
    elif subject in sd_subjects:
        return 'sd'
    elif subject in sleep_subjects:
        return 'sleep'
    else:
        raise ValueError(f'Subject {subject} not found')

arena_nodes = [
    'midline_upper',
    'midline_mid',
    'midline_lower',
    'left_object_center',
    'left_object_top',
    'left_object_right',
    'left_object_bottom',
    'left_object_left',
    'right_object_center',
    'right_object_top',
    'right_object_right',
    'right_object_bottom',
    'right_object_left',
    'arena_upper_left',
    'arena_upper_right',
    'arena_lower_left',
    'arena_lower_right',]

all_nodes = [
    'nose',
    'head',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'mid_body',
    'left_hip',
    'right_hip',
    'tail_base',
    'midline_upper',
    'midline_mid',
    'midline_lower',
    'left_object_center',
    'left_object_top',
    'left_object_right',
    'left_object_bottom',
    'left_object_left',
    'right_object_center',
    'right_object_top',
    'right_object_right',
    'right_object_bottom',
    'right_object_left',
    'arena_upper_left',
    'arena_upper_right',
    'arena_lower_left',
    'arena_lower_right',]


def load_nor_info():
    path = '/Volumes/neuropixel_archive/Data/acr_archive/NOR_videos/subject_info.yaml'
    with open(path, 'r') as f:
        nor_info = yaml.safe_load(f)
    return nor_info

def clean_full_df(df):
    new_dfs = []
    for node in all_nodes:
        new_dfs.append(clean_df(df.loc[:, df.loc[0] == node]))
    return pd.concat(new_dfs)
    

def clean_df(df, new_names=['x', 'y', 'likelihood']):
    label = df.iloc[0][0]
    df.columns = new_names
    df['node'] = label
    df = df.iloc[2:].reset_index(drop=True)
    df['frame'] = df.index
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    return df

def add_side_col(sub_df, novel=''):
    for cond in sub_df.condition.unique():
        cond_df = sub_df[sub_df['condition'] == cond]
        grid_df = cond_df.loc[cond_df['node'].isin(arena_nodes)]
        grid_df = grid_df.groupby(['subject', 'condition', 'node']).median(numeric_only=True).reset_index()
        midline_x = (grid_df.loc[grid_df.node == 'midline_upper']['x'].values[0] + grid_df.loc[grid_df.node == 'midline_lower']['x'].values[0]) / 2
        if novel == 'left':
            cond_df.loc[cond_df['x'] < midline_x, 'side'] = 'novel'
            cond_df.loc[cond_df['x'] >= midline_x, 'side'] = 'familiar'
        elif novel == 'right':
            cond_df.loc[cond_df['x'] < midline_x, 'side'] = 'familiar'
            cond_df.loc[cond_df['x'] >= midline_x, 'side'] = 'novel'
        else:
            raise ValueError('novel side not recognized')
        sub_df.loc[sub_df['condition'] == cond, 'side'] = cond_df['side']
    return sub_df


def create_arena_plot(arena_positions, condition='acq'):
    f, ax = plt.subplots(figsize=(16, 9))
    midline_x = (arena_positions[condition].loc['midline_upper']['x'] + arena_positions[condition].loc['midline_lower']['x']) / 2
    upper_arena_y = arena_positions[condition].loc['midline_upper']['y']
    lower_arena_y = arena_positions[condition].loc['midline_lower']['y']
    left_arena_x = (arena_positions[condition].loc['arena_upper_left']['x'] + arena_positions[condition].loc['arena_lower_left']['x']) / 2
    right_arena_x = (arena_positions[condition].loc['arena_upper_right']['x'] + arena_positions[condition].loc['arena_lower_right']['x']) / 2
    ax.axvline(midline_x, color='r', linestyle='--')
    ax.axhline(upper_arena_y, color='r', linestyle='--')
    ax.axhline(lower_arena_y, color='r', linestyle='--')
    ax.axvline(left_arena_x, color='r', linestyle='--')
    ax.axvline(right_arena_x, color='r', linestyle='--')
    sns.scatterplot(x='x', y='y', data=arena_positions[condition], ax=ax, s=100, c='r')
    return f, ax

def create_arena_plot_streamlined(ap, buffer_lr=0, buffer_ud=0, return_bounds=False):
    """creates arena plot with a buffer around marked edges. If return_bounds is True, return the values of left, right, upper, and lower bounds.

    Parameters
    ----------
    ap : pd.DataFrame
        dataframe of arena positions
    buffer : int, optional
        buffer around marked edges, by default 0
    return_bounds : bool, optional
        if True, return the values of left, right, upper, and lower bounds, by default False

    Returns
    -------
    f, ax, bounds : tuple
        figure, axis, and bounds of the arena
    """
    ap = ap[~ap['node'].str.contains('object')]
    f, ax = plt.subplots(figsize=(30, 18))
    midline_x = (ap.loc[ap['node']=='midline_upper']['x'].values[0] + ap.loc[ap['node']=='midline_lower']['x'].values[0]) / 2
    upper_arena_y = ap.loc[ap['node']=='midline_upper']['y'].values[0]
    lower_arena_y = ap.loc[ap['node']=='midline_lower']['y'].values[0]
    left_arena_x = (ap.loc[ap['node']=='arena_upper_left']['x'].values[0] + ap.loc[ap['node']=='arena_lower_left']['x'].values[0]) / 2
    right_arena_x = (ap.loc[ap['node']=='arena_upper_right']['x'].values[0] + ap.loc[ap['node']=='arena_lower_right']['x'].values[0]) / 2
    ax.axvline(midline_x, color='r', linestyle='--', linewidth=7)
    ax.axhline(upper_arena_y-buffer_ud, color='r', linestyle='--', linewidth=7)
    ax.axhline(lower_arena_y+buffer_ud, color='r', linestyle='--', linewidth=7)
    ax.axvline(left_arena_x-buffer_lr, color='r', linestyle='--', linewidth=7)
    ax.axvline(right_arena_x+buffer_lr, color='r', linestyle='--', linewidth=7)
    if return_bounds:
        return f, ax, (left_arena_x-buffer_lr, right_arena_x+buffer_lr, upper_arena_y-buffer_ud, lower_arena_y+buffer_ud)
    return f, ax

# =====================================================================================================
#======================================== HOME CAGE PLOTS =============================================
# =====================================================================================================



def get_sub_timing(subject):
    sprdsheet_file = '/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/NOR_subjects_analysis.xlsx'
    df = pd.read_excel(sprdsheet_file)
    box_on_time = df.loc[df['subject_id'] == subject].box_light_on.values[0]
    acq_date = df.loc[df['subject_id'] == subject].acq_date.values[0]
    return box_on_time, acq_date

def load_diffs_csv(subject, smoothing_sigma=None):
    sub_folder = f'/Volumes/neuropixel_archive/Data/acr_archive/NOR_videos/home_boxes/{subject}'
    diffs_files = glob.glob(f'{sub_folder}/*.csv')
    diffs_list = [pd.read_csv(f) for f in diffs_files if 'pos' not in f]
    diffs_df = pd.concat(diffs_list)
    diffs_df = diffs_df.rename(columns={'Unnamed: 0': 'datetime'})
    diffs_df['datetime'] = pd.to_datetime(diffs_df['datetime'])
    diffs_df.sort_values(by='datetime', inplace=True)
    diffs_df.reset_index(inplace=True, drop=True)
    if smoothing_sigma is not None:
        sm_data = gaussian_smooth(diffs_df['diff'].values, smoothing_sigma, 1)
        diffs_df['sm_diff'] = sm_data
    return diffs_df


def gaussian_smooth(data, sigma, sampling_frequency, axis=0, truncate=8):
    """1D convolution of the data with a Gaussian.

    The standard deviation of the gaussian is in the units of the sampling
    frequency. The function is just a wrapper around scipy's
    `gaussian_filter1d`, The support is truncated at 8 by default, instead
    of 4 in `gaussian_filter1d`

    Parameters
    ----------
    data : array_like
    sigma : float
    sampling_frequency : int
    axis : int, optional
    truncate : int, optional

    Returns
    -------
    smoothed_data : array_like

    """

    return gaussian_filter1d(
        data, sigma * sampling_frequency, truncate=truncate, axis=axis, mode="constant"
    )

def plot_diffs(diffs, box_time, acq_day, smoothing_sigma=None, limit_x=True):
    if smoothing_sigma is not None:
        sm_data = gaussian_smooth(diffs['diff'].values, smoothing_sigma, 1)
    else:
        sm_data = diffs['diff'].values
    diffs['sm_diff'] = sm_data
    f, ax = plt.subplots(1, 1, figsize=(60, 10))
    ax.plot(diffs['datetime'], diffs['sm_diff'])
    acq_start = pd.Timestamp(f'{acq_day} {box_time}')
    sd_start = acq_start + pd.Timedelta('18m')
    sd_end = sd_start + pd.Timedelta('1h')
    ax.axvspan(acq_start, acq_start+pd.Timedelta('17m'), color='green', alpha=0.2)
    ax.axvspan(sd_start, sd_end, color='red', alpha=0.2)
    ax.axvline(sd_start, color='red', linewidth=1)
    ax.axvline(sd_end, color='red', linewidth=1)
    test_start = acq_start + pd.Timedelta('24h')
    ax.axvline(test_start, color='green', linewidth=4)
    
    if limit_x:
        ax.set_xlim(acq_start-pd.Timedelta('13h'), test_start+pd.Timedelta('20m'))
    return f, ax


def threshold_actigraph(df, threshold=.1, col='sm_diff'):
    quant = df[col].quantile(threshold)
    df['moving'] = 1
    df.loc[df[col] < quant, 'moving'] = 0
    return df

def get_sd_sleep_proportion(subject):
    box_time, acq_day = get_sub_timing(subject)
    

def check_for_manual_labels(subject, condition):
    label_path = f'/Volumes/neuropixel_archive/Data/acr_archive/NOR_videos/scoring_files/{subject}--{condition}.csv'
    if os.path.exists(label_path):
        return pd.read_csv(label_path)
    else:
        return False

def get_exlcusion_frames(df):
    starts = df.loc[df['Behavior type'] == 'START']['Time'].values
    stops = df.loc[df['Behavior type'] == 'STOP']['Time'].values
    start_frames = starts*20
    stop_frames = stops*20
    start_frames = start_frames.astype(int)
    stop_frames = stop_frames.astype(int)
    return start_frames, stop_frames

def label_bad_frames(df, starts, stops):
    df['label'] = 'good'
    for start, stop in zip(starts, stops):
        df.loc[((df['frame']>start)&(df['frame']<=stop)), 'label'] = 'exclude'
    return df


def sum_exclusion_and_good(df, first=300):
    if first is not None:
        df = df.loc[df['Time']<first]
    # Check if the dataframe has an even number of rows
    if len(df) % 2 != 0:
        print(f"Warning: DataFrame has an odd number of rows ({len(df)})")
        print(df['Time'].iloc[-1])
        df = df.iloc[:-1]
    starts = df.loc[df['Behavior type'] == 'START']['Time'].values
    stops = df.loc[df['Behavior type'] == 'STOP']['Time'].values
    durations = stops - starts
    total_bad = sum(durations)
    total_t = df['Time'].max()
    total_good = total_t - total_bad
    return total_good, total_bad


def load_subject_node(subject, node, conds=['acq', 'test'], test_dur=6000, type=None, min_ex_time=30, consider_first=300):
    ni = load_nor_info()
    data_root = '/Volumes/neuropixel_archive/Data/acr_archive/NOR_videos/DLC_project_files/NOR_ROUND_2--analysis_results'
    dats = []

    for cond in conds:
        dat = pd.read_csv(f"{data_root}/{subject}-{cond}DLC_Resnet50_nor_single_subJun25shuffle0_snapshot_250_filtered.csv")
        df = clean_full_df(dat)
        df['subject'] = subject
        df['condition'] = cond
        df = add_side_col(df, ni[subject]['novel'])
        df = df.loc[df['node'] == node]
        
        if check_for_manual_labels(subject, cond) is not False:
            labels = check_for_manual_labels(subject, cond)
            total_good, total_bad = sum_exclusion_and_good(labels, consider_first)
            if total_bad >= min_ex_time:
                print(f'{subject} {cond} has {total_bad} seconds of exclusion, EXCLUDING BAD FRAMES')
                starts, stops = get_exlcusion_frames(labels)
                df = label_bad_frames(df, starts, stops)
                df = df.loc[(df['label']=='good')&(df['node']==node)]
                df = df.reset_index(drop=True)
        if cond == 'test':
            df = df.iloc[0:test_dur]
        dats.append(df)
    
    data = pd.concat(dats)
    
    if type is not None:
        data['type'] = type
    return data

def get_arena_df(subject, cond='test'):
    data_root = '/Volumes/neuropixel_archive/Data/acr_archive/NOR_videos/DLC_project_files/NOR_ROUND_2--analysis_results'
    dat = pd.read_csv(f"{data_root}/{subject}-{cond}DLC_Resnet50_nor_single_subJun25shuffle0_snapshot_250_filtered.csv")
    df = clean_full_df(dat)
    df['subject'] = subject
    df['condition'] = cond
    grid_df = df.loc[df['node'].isin(arena_nodes)]
    grid_df = grid_df.groupby(['subject', 'condition', 'node']).median(numeric_only=True).reset_index()
    return grid_df

def DEPRICATED_add_side_col(sub_df, novel=''):
    grid_df = sub_df.loc[sub_df['node'].isin(arena_nodes)]
    grid_df = grid_df.groupby(['subject', 'condition', 'node']).median(numeric_only=True).reset_index()
    midline_x = (grid_df.loc[grid_df.node == 'midline_upper']['x'].values[0] + grid_df.loc[grid_df.node == 'midline_lower']['x'].values[0]) / 2
    if novel == 'left':
        cond_df.loc[cond_df['x'] < midline_x, 'side'] = 'novel'
        cond_df.loc[cond_df['x'] >= midline_x, 'side'] = 'familiar'
    elif novel == 'right':
        cond_df.loc[cond_df['x'] < midline_x, 'side'] = 'familiar'
        cond_df.loc[cond_df['x'] >= midline_x, 'side'] = 'novel'
    else:
        raise ValueError('novel side not recognized')
    sub_df.loc[sub_df['condition'] == cond, 'side'] = cond_df['side']
    return sub_df

def sum_and_plot_scored_sleep(path, time_scored=14400):
    df = pd.read_csv(path)
    starts = df.loc[df['Behavior type'] == 'START']['Time'].values
    stops = df.loc[df['Behavior type'] == 'STOP']['Time'].values
    durations = stops - starts
    total_sleep = sum(durations)
    sleep_fraction = total_sleep/time_scored
    x_vals = np.linspace(0, time_scored)
    y_vals = np.zeros(len(x_vals))
    f, ax = plt.subplots(1, 1, figsize=(25, 5))
    ax.plot(x_vals, y_vals)
    for start, stop in zip(starts, stops):
        x_seg = np.arange(start, stop, step=0.5)
        y_seg = np.ones(len(x_seg))
        ax.plot(x_seg, y_seg, linewidth=1, color='red')
    ax.set_title(f'SLEEP FRACTION: {sleep_fraction}')
    ax.set_ylim(0.95, 1.05)
    return f, ax, sleep_fraction

def threshold_post_dep_sleep(subject, smoothing_sigma=2, quantile=0.75, epoc_duration=3600, col_to_use='diff'):
    df = load_diffs_csv(subject, smoothing_sigma=smoothing_sigma)
    box_time, acq_day = get_sub_timing(subject)
    acqday_start = pd.Timestamp(f'{acq_day} {box_time}')
    acqday_end = acqday_start + pd.Timedelta('11.5h')
    dfday = df.loc[(df['datetime']>acqday_start) & (df['datetime']<acqday_end)]
    box_ts = pd.Timestamp(f'{acq_day} {box_time}')
    
    
    
    min_sleep_start = box_ts + pd.Timedelta('40min')
    sleep_start_real = dfday.query('datetime > @min_sleep_start')['datetime'].values.min()
    sleep_start = sleep_start_real + pd.Timedelta('3min')
    
    if quantile > 1:
        threshold = quantile
    else:
        threshold = np.quantile(dfday[col_to_use].values, quantile)
    dfday['state'] = ['wake' if x >= threshold else 'sleep' for x in dfday[col_to_use].values]


    df_sleep = dfday.loc[(dfday['datetime']>=sleep_start) & (dfday['datetime']<sleep_start+pd.Timedelta(f'{epoc_duration}s'))]
    counts = df_sleep.groupby('state').count().reset_index()
    sleep_proportion = counts.loc[counts['state']=='sleep'][col_to_use].values[0]/epoc_duration
    
    
    # Simple plot to show results
    f, ax = plt.subplots(1, 1, figsize=(30, 8))
    ax.plot(dfday['datetime'], dfday[col_to_use])
    ax.axhline(threshold, color='red', linestyle='--', linewidth=3)
    ax.axvspan(sleep_start, sleep_start+pd.Timedelta(f'{epoc_duration}s'), color='orange', alpha=0.7)
    ax.set_title(f'{subject} | Sleep Proportion = {sleep_proportion}')
    ax.set_ylim(0, threshold*2)
    return f, ax, dfday, sleep_proportion
    
def get_scoring_starts_and_stops(path):
    df = pd.read_csv(path)
    starts = df.loc[df['Behavior type'] == 'START']['Time'].values
    stops = df.loc[df['Behavior type'] == 'STOP']['Time'].values
    return starts, stops