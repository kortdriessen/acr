import pandas as pd
import numpy as np
import polars as pl
import acr
import pickle

def _check_for_all_chans_in_mask_bout(all_chans, bout_df):
    bout_chans = bout_df['channel'].unique()
    return np.all([ch in bout_chans for ch in all_chans])



def compute_channel_off_masks(oodf, hyp_t1, hyp_t2, all_chans, resolution=1e-3):
    if type(oodf) is pl.DataFrame:
        oodf = oodf.to_pandas()
    bout_dur = ( hyp_t2-hyp_t1 ).total_seconds()
    tds = np.arange(0, bout_dur, resolution)
    tds = pd.to_timedelta(tds, unit='s')
    times_to_mask  = tds+hyp_t1
    channel_masks = {}
    for ch in all_chans:
        if ch not in oodf['channel'].unique():
            channel_masks[ch] = np.zeros(len(times_to_mask), dtype=bool)
            continue
        chdf = oodf[oodf['channel']==ch]
        off_starts = chdf['start_datetime'].values
        off_ends = chdf['end_datetime'].values
        start_indices = np.searchsorted(off_starts, times_to_mask, side='right') - 1
        end_indices = np.searchsorted(off_ends, times_to_mask, side='left')
        mask = (start_indices==end_indices)
        channel_masks[ch] = mask
    return channel_masks, times_to_mask

def get_off_masks_by_bout(oodf, hd):
    assert len(oodf['probe'].unique())==1, 'Only works for single probe'
    offs = oodf.offs()
    bout_off_masks = {}
    bout_mask_times = {}
    all_chans = offs['channel'].unique()
    for cond in hd.keys():
        bout_off_masks[cond] = {}
        bout_mask_times[cond] = []
    for cond in bout_off_masks.keys():
        hyp = hd[cond]
        bout_num = 0
        print(f'Working on {cond}')
        for bout in hyp.itertuples():
            bout_num+=1
            t1 = bout.start_time
            t2 = bout.end_time
            bout_offs = offs.oots(t1, t2)
            masks, mask_times = compute_channel_off_masks(bout_offs, t1, t2, all_chans)
            bout_off_masks[cond][bout_num] = masks
            bout_mask_times[cond].append(mask_times)
    return bout_off_masks, bout_mask_times

def concatenate_mask_times(cond_mask_times):
    full_mask_times = {}
    for cond in cond_mask_times.keys():
        full_mask_times[cond] = np.concatenate(cond_mask_times[cond])
    return full_mask_times

def concatenate_bout_masks(bomasks):
    conditions = bomasks.keys()
    channel_concat_masks = {}
    for cond in conditions:
        channel_concat_masks[cond] = {}
        # concatenate bout masks
        for bout in bomasks[cond].keys():
            for channel in bomasks[cond][bout].keys():
                if channel not in channel_concat_masks[cond].keys():
                    channel_concat_masks[cond][channel] = []
                channel_concat_masks[cond][channel].append(bomasks[cond][bout][channel])
        for channel in channel_concat_masks[cond].keys():
            channel_concat_masks[cond][channel] = np.concatenate(channel_concat_masks[cond][channel])
    return channel_concat_masks


def process_bout_off_masks_dict(bomasks):
    relative_overlaps = {}
    conditions = bomasks.keys()
    for cond in conditions:
        # concatenate bout masks
        channel_concat_masks = {}
        for bout in bomasks[cond].keys():
            for channel in bomasks[cond][bout].keys():
                if channel not in channel_concat_masks.keys():
                    channel_concat_masks[channel] = []
                channel_concat_masks[channel].append(bomasks[cond][bout][channel])
        for channel in channel_concat_masks.keys():
            channel_concat_masks[channel] = np.concatenate(channel_concat_masks[channel])
        #overlap masks
        overlap_masks = {}
        for comp_channel in channel_concat_masks.keys():
            overlap_masks[comp_channel] = {}
            for other_channel in channel_concat_masks.keys():
                print(f'computing overlap for {comp_channel} relative to {other_channel}')
                overlap_masks[comp_channel][other_channel] = (channel_concat_masks[comp_channel] & channel_concat_masks[other_channel])
        
        # relative overlaps
        relative_overlaps[cond] = {}
        for comp_channel in overlap_masks.keys():
            relative_overlaps[cond][comp_channel] = {}
            total_off_sum = channel_concat_masks[comp_channel].sum()
            for other_channel in overlap_masks[comp_channel].keys():
                shared_off_sum = overlap_masks[comp_channel][other_channel].sum()
                shared_off_pct = shared_off_sum / total_off_sum
                relative_overlaps[cond][comp_channel][other_channel] = shared_off_pct

    return relative_overlaps

def compute_overlap_by_channel_distances(rels):
    channel_distances = {}
    share_pcts = {}
    for probe in rels.keys():
        for cond in rels[probe].keys():
            for comp_channel in rels[probe][cond].keys():
                for other_channel in rels[probe][cond][comp_channel].keys():
                    if rels[probe][cond][comp_channel][other_channel] == 1:
                        rels[probe][cond][comp_channel][other_channel] = np.nan
        
        channel_distances[probe] = {}
        share_pcts[probe] = {}
        for cond in rels[probe].keys():
            channel_distances[probe][cond] = {}
            share_pcts[probe][cond] = {}
            for comp_channel in rels[probe][cond].keys():
                channel_distances[probe][cond][comp_channel] = []
                share_pcts[probe][cond][comp_channel] = []
                for other_channel in rels[probe][cond][comp_channel].keys():
                    channel_distances[probe][cond][comp_channel].append(comp_channel-other_channel)
                    share_pcts[probe][cond][comp_channel].append(rels[probe][cond][comp_channel][other_channel])
    master_dvc = pd.DataFrame(columns=['probe', 'condition', 'channel', 'comp_distance', 'share_pct', 'cond'])
    for probe in channel_distances.keys():
        for cond in channel_distances[probe].keys():
            for comp_channel in channel_distances[probe][cond].keys():
                dvc = pd.DataFrame({'probe': probe, 
                                    'condition': cond, 
                                    'channel': comp_channel, 
                                    'comp_distance': channel_distances[probe][cond][comp_channel], 
                                    'share_pct': share_pcts[probe][cond][comp_channel], 
                                    'cond': cond})
                master_dvc = pd.concat([master_dvc, dvc])
    master_dvc.dropna(inplace=True)
    master_dvc['comp_distance_abs'] = master_dvc['comp_distance'].abs()
    
    return master_dvc



# Functions Related to ONSET/OFFSET Synchrony
def find_transitions(channel_data):
    # Find transitions from False to True (start of OFF period)
    starts = np.where(np.diff(channel_data.astype(int)) == 1)[0] + 1
    
    # Find transitions from True to False (end of OFF period)
    ends = np.where(np.diff(channel_data.astype(int)) == -1)[0]
    
    if len(starts) - len(ends) == 1:
        #add the final index to the ends
        ends = np.append(ends, len(channel_data))
    return starts, ends

def get_off_transitions(full_masks):
    transitions = {}
    for probe in full_masks.keys():
        transitions[probe] = {}
        for condition in full_masks[probe].keys():
            transitions[probe][condition] = {}
            mask_array = np.array(list(full_masks[probe][condition].values()))
            for channel, channel_data in enumerate(mask_array):
                starts, ends = find_transitions(channel_data)
                transitions[probe][condition][channel+1] = {'starts': starts, 'ends': ends}
    return transitions

def find_overlapping_offs_slow(ch1_starts, ch1_ends, ch2_starts, ch2_ends):
    """this is the slower mode, but is technically for sure correct if needed to double check results"""
    overlaps = []
    
    for i, (ch1_start, ch1_end) in enumerate(zip(ch1_starts, ch1_ends)):
        for j, (ch2_start, ch2_end) in enumerate(zip(ch2_starts, ch2_ends)):
            # Check for overlap
            if ch1_start < ch2_end and ch2_start < ch1_end:
                overlaps.append((i, j))
            
            # If we've passed the end of the current ch1 OFF period, move to next
            if ch2_start > ch1_end:
                break
    
    return overlaps

def find_overlapping_offs(ch1_starts, ch1_ends, ch2_starts, ch2_ends):
    # if channel-1 is shorter than channel-2, we swap them
    if len(ch1_starts) < len(ch2_starts):
        placehold_starts = ch1_starts.copy()
        placehold_ends = ch1_ends.copy()
        ch1_starts, ch1_ends = ch2_starts, ch2_ends
        ch2_starts, ch2_ends = placehold_starts, placehold_ends
    
    overlaps = []
    j = 0
    ch2_len = len(ch2_starts)
    
    for i, (ch1_start, ch1_end) in enumerate(zip(ch1_starts, ch1_ends)):
        # Skip ch2 OFF periods that end before ch1_start
        while j < ch2_len-1 and ch2_ends[j] <= ch1_start:
            j += 1
        
        # Check for overlaps
        k = j
        while k < ch2_len and ch2_starts[k] < ch1_end:
            overlaps.append((i, k))
            k += 1
        
        # If we've reached the end of ch2, we're done
        if j >= ch2_len:
            break
    
    # if we had to swap the channels, we need to reverse the order of the overlaps
    # check if the placerhold_starts variable exists
    if 'placehold_starts' in locals():
        overlaps = [(j, i) for i, j in overlaps]
    return overlaps

def _find_overlapping_offs(transition_dic):
    overlapping_offs = {}
    for channel in transition_dic.keys():
        overlapping_offs[channel] = {}
        for other_channel in transition_dic.keys():
            if channel != other_channel:
                print(f'comparing {channel} to {other_channel}')
                overlapping_offs[channel][other_channel] = find_overlapping_offs(transition_dic[channel]['starts'], transition_dic[channel]['ends'], transition_dic[other_channel]['starts'], transition_dic[other_channel]['ends'])
    return overlapping_offs
def get_all_overlapping_offs(transitions):
    overlapping_offs = {}
    for probe in transitions.keys():
        overlapping_offs[probe] = {}
        for condition in transitions[probe].keys():
            overlapping_offs[probe][condition] = _find_overlapping_offs(transitions[probe][condition])
    return overlapping_offs

def _get_onset_offset_diffs(transitions, overlapping_offs):
    onset_diffs = {}
    offset_diffs = {}
    for channel in transitions.keys():
        onset_diffs[channel] = {}
        offset_diffs[channel] = {}
        ovlp = overlapping_offs[channel]
        for other_channel in ovlp.keys():
            onset_diffs[channel][other_channel] = []
            offset_diffs[channel][other_channel] = []
            print(f'processing {channel} and {other_channel}')
            for ov in ovlp[other_channel]:
                onset_diffs[channel][other_channel].append(transitions[channel]['starts'][ov[0]] - transitions[other_channel]['starts'][ov[1]])
                offset_diffs[channel][other_channel].append(transitions[channel]['ends'][ov[0]] - transitions[other_channel]['ends'][ov[1]])
    return onset_diffs, offset_diffs

def get_full_onset_offset_diffs(transitions, overlapping_offs):
    onset_diffs = {}
    offset_diffs = {}
    for probe in transitions.keys():
        onset_diffs[probe] = {}
        offset_diffs[probe] = {}
        for condition in transitions[probe].keys():
            onset_diffs[probe][condition], offset_diffs[probe][condition] = _get_onset_offset_diffs(transitions[probe][condition], overlapping_offs[probe][condition])
    return onset_diffs, offset_diffs

def generate_onset_offset_diff_df(onset_diffs, offset_diffs):
    dfs = []
    for probe in onset_diffs.keys():
        for cond in onset_diffs[probe].keys():
            for chan in onset_diffs[probe][cond].keys():
                for comp_chan in onset_diffs[probe][cond][chan].keys():
                    dfs.append(pd.DataFrame({'onset_diff': onset_diffs[probe][cond][chan][comp_chan], 'offset_diff': offset_diffs[probe][cond][chan][comp_chan], 'comp_chan': comp_chan, 'cond': cond, 'probe': probe, 'channel': chan}))
    master_df = pd.concat(dfs)
    master_df['abs_onset_diff'] = master_df['onset_diff'].abs()
    master_df['abs_offset_diff'] = master_df['offset_diff'].abs()
    return master_df

def calculate_global_off_synchrony(mua, oodf):
    "Global Synchrony -- Not really useful probably"
    sync = pd.DataFrame(columns = ['start_datetime', 'end_datetime', 'channel', 'onset_spike', 'offset_spike', 'opdur', 'probe'])
    if type(oodf) != pd.DataFrame:
        oodf = oodf.to_pandas()
    if type(mua) != pd.DataFrame:
        mua = mua.to_pandas()
    for probe in oodf.prbs():
        for off in oodf.prb(probe).offs().itertuples():
            off_begin = off.start_datetime
            off_end = off.end_datetime
            opdur = (off_end - off_begin).total_seconds()
            
            for channel in mua['channel'].unique():
                ch_times = mua.prb(probe).chnl(channel)['datetime']
                onset_diffs = off_begin-ch_times
                offset_diffs = ch_times - off_end
                onset_diffs = onset_diffs[onset_diffs > pd.Timedelta('0s')]
                offset_diffs = offset_diffs[offset_diffs > pd.Timedelta('0s')]
                onset_spike = onset_diffs.min()
                offset_spike = offset_diffs.min()
                chdf = pd.DataFrame({'start_datetime': off_begin, 'end_datetime': off_end, 'channel': channel, "probe": probe, 'onset_spike': onset_spike, 'offset_spike': offset_spike, "opdur":opdur}, index=[0])
                sync = pd.concat([sync, chdf])
    return sync

def save_off_masks_pipeline(subject, exp, probes=['NNXo', 'NNXr']):
    hd = acr.hypnogram_utils.create_acr_hyp_dict(subject, exp)
    #osc = acr.onoffmua.load_oodf(subject, exp)
    mua = acr.mua.load_concat_peaks_df(subject, exp)
    osc = acr.onoffmua.compute_full_oodf_by_channel(mua)
    osc_st = acr.onoffmua.true_strictify_oodf(osc)
    offs = osc.offs()
    offs_st = osc_st.offs()
    
    #First the non-strict masks
    bout_off_masks = {}
    for probe in probes:
        bout_off_masks[probe], bout_mask_times = acr.sync.get_off_masks_by_bout(offs.prb(probe), hd)
    mask_path = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/off_masks/{exp}--off_masks_by_bout.pkl'
    mask_times_path = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/off_masks/{exp}--off_masks_by_bout_times.pkl'
    with open(mask_path, 'wb') as f:
        pickle.dump(bout_off_masks, f)
    with open(mask_times_path, 'wb') as f:
        pickle.dump(bout_mask_times, f)

    #Now the strict masks
    bout_off_masks_st = {}
    for probe in probes:
        bout_off_masks_st[probe], bout_mask_times_st = acr.sync.get_off_masks_by_bout(offs_st.prb(probe), hd)
    mask_path = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/off_masks/{exp}--off_masks_by_bout_strict.pkl'
    with open(mask_path, 'wb') as f:
        pickle.dump(bout_off_masks_st, f)
    return

def load_off_masks(subject, exp, probes=['NNXo', 'NNXr'], strict=False, times=False):
    mask_path = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/off_masks/{exp}--off_masks_by_bout.pkl'
    if strict:
        mask_path = mask_path.replace('off_masks_by_bout', 'off_masks_by_bout_strict')
    with open(mask_path, 'rb') as f:
        bout_off_masks = pickle.load(f)
    if times:
        time_path = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/off_masks/{exp}--off_masks_by_bout_times.pkl'
        with open(time_path, 'rb') as f:
            bout_mask_times = pickle.load(f)
            return bout_off_masks, bout_mask_times
    elif times==False:
        return bout_off_masks
    
def morpho_clean_mask(off_mask, horz_remove=50, vert_remove=2, vert_connect=2, horz_connect=6, morpho_ops=None):
    import dask_image.ndmorph
    from dask_image import ndmorph
    import dask.array as dska
    if morpho_ops is not None:
        horz_remove = morpho_ops['horz_remove']
        vert_remove = morpho_ops['vert_remove']
        vert_connect = morpho_ops['vert_connect']
        horz_connect = morpho_ops['horz_connect']

    om = off_mask.copy()
    
    # # # vertical : Remove few-channel epochs
    struct = np.ones((1, vert_remove))
    print(f"Binary open: {struct.shape}")
    om.data = dask_image.ndmorph.binary_opening(om.data, structure=struct, iterations=1)
    
     # # Horizontal  Remove shorter blobs
    struct = np.ones((horz_remove, 1))
    print(f"Binary open: {struct.shape}")
    om.data = dask_image.ndmorph.binary_opening(om.data, structure=struct, iterations=1)
    
     # # vertical : Connect distant blobs vertically
     # vertical : Connect distant blobs vertically
    struct = np.ones((1, vert_connect))
    print(f"Binary close: {struct.shape}")
    temp_pad = dska.pad(om.data, ((0, 0), (vert_connect, vert_connect)), mode='edge')
    temp_pad_cleaned = dask_image.ndmorph.binary_closing(temp_pad, structure=struct, iterations=1)
    # remove the padding
    om.data = temp_pad_cleaned[:, vert_connect:-vert_connect]
    
    #horizontal closing: connect across small gaps
    struct = np.ones((horz_connect, 1))
    print(f"Binary close: {struct.shape}")
    om.data = dask_image.ndmorph.binary_closing(om.data, structure=struct, iterations=1)
    
    # # Horizontal  Remove shorter blobs
    struct = np.ones((10, 2))
    print(f"Binary open: {struct.shape}")
    om.data = dask_image.ndmorph.binary_opening(om.data, structure=struct, iterations=1)
    
    return om


# ----------------- WHOLE-PROBE Synchrony -----------------
# =========================================================

def find_onset_diffs(spike_times, onset_times):
    """
    For each datetime in onset_times, find the index in spike_times of the largest datetime
    that is <= the datetime in onset_times.

    Parameters:
    - spike_times: List[datetime], sorted in ascending order
    - onset_times: List[datetime], any order

    Returns:
    - List[int]: Indices in spike_times corresponding to each datetime in onset_times
    """
    # Convert datetime lists to numpy datetime64 for efficient computation
    spike_array = np.array(spike_times).astype('datetime64[ns]')
    onset_array = np.array(onset_times).astype('datetime64[ns]')

    # check that both arrays are sorted
    #assert np.all(np.diff(spike_times) >= 0)
    #assert np.all(np.diff(onset_times) >= 0)
    
    # Use searchsorted to find insertion points
    # side='right' returns the index where the element should be inserted to maintain order
    insertion_indices = np.searchsorted(spike_array, onset_array, side='right') - 1

    # Handle cases where no element in spike_times is <= the short datetime
    # Set such indices to -1 or any sentinel value as per your requirement
    insertion_indices[insertion_indices < 0] = -1

    onset_spikes = spike_array[insertion_indices.tolist()]
    onset_diffs = (pd.DatetimeIndex(onset_array) - pd.DatetimeIndex(onset_spikes)).total_seconds()
    
    return np.array(onset_diffs)

def find_offset_diffs(spike_times, offset_times):
    """
    For each datetime in offset_times, find the index in spike_times of the smallest datetime
    that is >= the datetime in offset_times.

    Parameters:
    - spike_times: List[datetime], sorted in ascending order
    - offset_times: List[datetime], any order

    Returns:
    - List[int]: Indices in spike_times corresponding to each datetime in offset_times
    """
    # Convert datetime lists to numpy datetime64 for efficient computation
    spike_array = np.array(spike_times).astype('datetime64[ns]')
    offset_array = np.array(offset_times).astype('datetime64[ns]')

    # check that both arrays are sorted
    #assert np.all(np.diff(spike_times) >= 0)
    #assert np.all(np.diff(offset_times) >= 0)
    
    # Use searchsorted to find insertion points
    # side='right' returns the index where the element should be inserted to maintain order
    insertion_indices = np.searchsorted(spike_array, offset_array, side='left')

    # Handle cases where the short datetime is greater than all in long_list
    insertion_indices = np.where(insertion_indices < len(spike_array), insertion_indices, -1)

    offset_spikes = spike_array[insertion_indices.tolist()]
    offset_diffs = (pd.DatetimeIndex(offset_spikes) - pd.DatetimeIndex(offset_array)).total_seconds()
    
    
    return np.array(offset_diffs)


# ----------------- Field Potentials / Slope -----------------
# =========================================================

def compute_full_slope_df(off_means, fs):
    off_means = off_means.sort_values(['channel', 't'])
    slope_df = pd.DataFrame()
    for channel in off_means['channel'].unique():
        channel_df = off_means[off_means['channel']==channel]
        x = channel_df['t'].values
        y = channel_df['d'].values
        drvtv = acr.fp.compute_instantaneous_slope_savgol(y, sampling_rate=fs, window_length=15, polyorder=5)
        drvtv2 = acr.fp.compute_instantaneous_slope_savgol(drvtv, sampling_rate=fs, window_length=15, polyorder=5)
        chan_df = pd.DataFrame({'channel':channel, 't':x, 'deriv':drvtv, 'deriv2':drvtv2})
        slope_df = pd.concat([slope_df, chan_df])
    slope_df.reset_index(drop=True, inplace=True)
    slope_df = slope_df.sort_values(['channel', 't'])
    off_means['deriv'] = slope_df['deriv']
    off_means['deriv2'] = slope_df['deriv2']
    return off_means        