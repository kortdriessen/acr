import spikeinterface as si
import pandas as pd
import numpy as np
import zarr
import kdephys as kde
from acr.utils import raw_data_root
from acr.info_pipeline import subject_info_section
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
import os
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.core.recording_tools as rt
import tdt
import acr
import polars as pl

def v_to_uv(data, gain=1e6):
    """Converts data from volts to microvolts.

    Parameters
    ----------
    data : np.array
        data, of shape (n_channels, n_samples), should already be filtered.
    gain : int, optional
        the gain of the amplifier, by default 1000
    """
    data *= gain
    return data


def check_for_preprocessed_mua_data(subject, recording, probe):
    save_folder = f'{raw_data_root}mua_data/{subject}'
    save_path = f'{save_folder}/MUA--{subject}--{recording}--{probe}.zarr'
    if os.path.exists(save_path):
        if os.path.exists(f'{save_path}/properties'):
            if os.path.exists(f'{save_path}/properties/noise_level_mad_scaled'):
                return True
    else:
        return False

def prepro_np_array_via_si(np_array, fs, chans_to_interp=None):
    nchans = np_array.shape[0]
    if nchans != 16:
        raise ValueError('This function is only for 16 channel probes')
    chan_ids = np.arange(1, nchans+1)
    si_raw = se.NumpyRecording(np_array.T, sampling_frequency=fs, channel_ids=chan_ids)
    si_filt = sp.bandpass_filter(si_raw, freq_min=300, freq_max=12000)
    si_filt_cmr = sp.common_reference(si_filt, reference='global', operator='median')
    
    # now we should set the offset and gain (to get to uV) to 0 and 1 respectively, since we've already converted the incoming array.
    num_chans = si_filt_cmr.get_num_channels()
    si_filt_cmr.set_property(key="gain_to_uV", values=np.ones(num_chans, dtype="float32"))
    si_filt_cmr.set_property(key="offset_to_uV", values=np.zeros(num_chans, dtype="float32"))
    
    # we will also set the noise levels here which will be used later in detection
    rand_kwargs = dict(chunk_size=int(si_filt_cmr.sampling_frequency*2), num_chunks_per_segment=100)
    new_nlvs = rt.get_noise_levels(si_filt_cmr, method='mad', force_recompute=True, **rand_kwargs)
    if 'noise_level_mad_scaled' not in si_filt_cmr.get_property_keys():
        si_filt_cmr.set_property(key="noise_level_mad_scaled", values=new_nlvs)
    
    si_filt_cmr_with_probe = kde.units.utils.set_probe_and_channel_locations_on_si_rec(si_filt_cmr)
    
    #TODO: could zscore with --> si_filt_cmr_z = sp.zscore(si_filt_cmr)
    #TODO: importantly, add bad channel interpolation!! Should probably go here! Or could possibly be done in a separate function.
    return si_filt_cmr_with_probe 

def save_preprocessed_mua_data(data, subject, recording, probe, njobs=16, chunk_duration='100s', progress_bar=True, overwrite=False):
    subject_dir = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}'
    if os.path.exists == False:
        os.mkdir(subject_dir)
    save_path = f'{subject_dir}/MUA--{subject}--{recording}--{probe}.zarr'
    data.save(folder=save_path, overwrite=overwrite, format="zarr", n_jobs=njobs, chunk_duration=chunk_duration, progress_bar=progress_bar)

def preprocess_data_for_mua(subject, exp_sort_id, probes=['NNXo', 'NNXr'], chans_to_interp=None, overwrite=False, njobs=16):
    for probe in probes:
        sort_id = f'{exp_sort_id}-{probe}'
        recs, starts, durations = acr.units.get_time_info(subject, sort_id)
        for rec, dur in zip(recs, durations):
            exists = check_for_preprocessed_mua_data(subject, rec, probe)
            if exists==True:
                if overwrite == True:
                    print(f'Overwriting {subject}--{rec}--{probe}')
                else:
                    print(f'Preprocessed data already exists for {subject}--{rec}--{probe}, set overwrite=True to overwrite')
                    continue
            else:
                print(f'Preprocessing {subject}--{rec}--{probe}')
            block_path = acr.io.acr_path(subject, rec)
            dat = tdt.read_block(block_path, t1=0, t2=dur, store=probe)
            data = dat.streams[probe].data
            data = v_to_uv(data)
            fs = dat.streams[probe].fs
            preprocessed_data = prepro_np_array_via_si(data, fs, chans_to_interp=chans_to_interp)
            save_preprocessed_mua_data(preprocessed_data, subject, rec, probe, overwrite=overwrite, njobs=njobs)
    return

def load_processed_mua_signal(subject, recording, probe, version='xr'):
    path = f'{raw_data_root}mua_data/{subject}/MUA--{subject}--{recording}--{probe}.zarr'
    if version == 'xr':
        data = kde.xr.utils.load_processed_zarr_as_xarray(path)
        return data
    else:
        data = si.read_zarr(path)
        return data

def check_for_mua_spikes_df(subject, recording, probe):
    save_folder = f'{raw_data_root}mua_data/{subject}/si_peak_dfs'
    save_path = f'{save_folder}/PEAKS--{subject}--{recording}--{probe}.parquet'
    if os.path.exists(save_path):
        return True
    else:
        return False

def detect_mua_spikes_si(subject, recording, probe, save=True, chunk_duration='1s', n_jobs=224, progress_bar=True, threshold=4, overwrite=False):
    # load the raw mua data (zarr) as a spikeinterface recording extractor
    raw_mua_exists = check_for_preprocessed_mua_data(subject, recording, probe)
    if raw_mua_exists == False:
        print(f'Preprocessed data does not exist for {subject}--{recording}--{probe}, run preprocess_data_for_mua first')
        return None
    if (overwrite==False) and (save==True):
        if check_for_mua_spikes_df(subject, recording, probe):
            print(f'Peaks already detected for {subject}--{recording}--{probe}, set overwrite=True to overwrite')
            return None
    rec = load_processed_mua_signal(subject, recording, probe, version='si')
    job_kwargs = dict(chunk_duration=chunk_duration, n_jobs=n_jobs, progress_bar=progress_bar)
    peaks = detect_peaks(rec, method='by_channel', detect_threshold=threshold, peak_sign='neg', **job_kwargs)
    dtype = [('sample', 'int64'), ('channel', 'int64'), ('amplitude', 'float64'), ('segment', 'int64')]
    peaks_array = peaks.view(dtype).reshape(-1)
    peaks_df = pd.DataFrame(peaks_array)
    peaks_df.drop(columns='segment', inplace=True)
    peaks_df['time'] = peaks_df['sample'] / rec.get_sampling_frequency()
    peaks_df['subject'] = subject
    peaks_df['recording'] = recording
    peaks_df['probe'] = probe
    if save==True:
        save_folder = f'{raw_data_root}mua_data/{subject}/si_peak_dfs'
        save_path = f'{save_folder}/PEAKS--{subject}--{recording}--{probe}.parquet'
        if os.path.exists(save_folder) == False:
            os.mkdir(save_folder)
        peaks_df.to_parquet(save_path)
        return peaks_df
    else:
        return peaks_df


def detect_all_subject_mua_spikes(subject, threshold=4, chunk_duration='1s', n_jobs=224, progress_bar=True, save=True, overwrite=False):
    mua_folder = f'{raw_data_root}mua_data/{subject}'
    stems = []
    for f in os.listdir(mua_folder):
        if ('MUA--' in f) and ('zarr' in f):
            stems.append(f.split('.zarr')[0].split('MUA--')[1])
    for stem in stems:
        rec = stem.split('--')[1]
        probe = stem.split('--')[2]
        print(f'Detecting spikes for {subject}--{rec}--{probe}')
        detect_mua_spikes_si(subject, rec, probe, threshold=threshold, chunk_duration=chunk_duration, n_jobs=n_jobs, progress_bar=progress_bar, save=save, overwrite=overwrite)
    return

def list_all_preprocessed_mua_data(subject):
    processed = []
    mua_folder = f'{raw_data_root}mua_data/{subject}'
    if os.path.exists(mua_folder) == False:
        return processed
    for f in os.listdir(mua_folder):
        if ('MUA--' in f) and ('zarr' in f):
            sub = f.split('--')[1]
            rec = f.split('--')[2]
            probe = f.split('--')[3].split('.zarr')[0]
            processed.append((sub, rec, probe))
    return processed

def load_detected_mua_spikes(subject, rec, probe):
    path = f'{raw_data_root}mua_data/{subject}/si_peak_dfs/PEAKS--{subject}--{rec}--{probe}.parquet' 
    df = pl.read_parquet(path)
    raw_times = df['time'].to_numpy()
    rec_times = subject_info_section(subject, 'rec_times')
    start_dt = pd.Timestamp(rec_times[rec]['start'])
    times_td = pd.to_timedelta(raw_times, unit='s')
    times_in_dt = start_dt + times_td
    df = df.with_columns(pl.Series('datetime', times_in_dt))
    df = df.with_columns(channel=pl.col('channel') + 1)
    return df

def load_mua_full_exp(subject, exp, probes=['NNXo', 'NNXr']):
    all_dfs = []
    for probe in probes:
        sort_id = f'{exp}-{probe}'
        recs, starts, durations = acr.units.get_time_info(subject, sort_id)
        for rec, dur in zip(recs, durations):
            peaks = load_detected_mua_spikes(subject, rec, probe)
            all_dfs.append(peaks)
    return pl.concat(all_dfs)