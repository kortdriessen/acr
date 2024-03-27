import os
import acr
import pandas as pd
import numpy as np
import tdt
import spikeinterface as si

def read_and_save_tdt_recording(subject, recording, probe, t_start, t_end, dtype='float32', save_directory=None):
    if save_directory is None:
        save_directory = f'/Volumes/neuropixel_archive/Data/acr_archive/OFF_data/{subject}'
    if not os.path.exists(save_directory):
            os.makedirs(save_directory)
    
    for file in os.listdir(save_directory):
        if f'{recording}_{probe}_{t_start}_{t_end}' in file:
            print(f'{recording}_{probe}_{t_start}_{t_end} already exists in {save_directory}')
            return
    
    block_path = acr.io.acr_path(subject, recording)
    blk = tdt.read_block(block_path, store=probe, evtype=['streams'], t1=t_start, t2=t_end) #assumes the use of all channels (16), all channels have same sampling rate and same number of samples.
    srate = blk.streams[probe].fs
    srate = str(srate).replace('.', '-')
    data = blk.streams[probe].data # ndarray of shape (n_channels, n_samples)
    tmp_bin_path = f'{save_directory}/{recording}_{probe}_{t_start}_{t_end}_{srate}'
    data.astype(dtype).tofile(f'{tmp_bin_path}')
    return

def load_si_recording(subject, recording, probe, num_channels=16, dtype='float32', gen_data=True):
    subject_dir = f'/Volumes/neuropixel_archive/Data/acr_archive/OFF_data/{subject}'
    file_exists = False
    for file in os.listdir(subject_dir):
        if f'{recording}_{probe}' in file:
            file_path = f'{subject_dir}/{file}'
            file_exists = True
            print(f'File found at {file_path}')
            break
    if file_exists == False:
        if gen_data == True:
            print('File Not found, running read_save_tdt_recording')
            end_time = acr.utils.get_recording_end_time(subject, recording, probe)
            assert end_time is not None, f'No end time found for {subject} {recording} {probe}'
            if type(end_time) != int:
                end_time = int(end_time)
            read_and_save_tdt_recording(subject, recording, probe, 0, end_time)
            file_path = f'{subject_dir}/{recording}_{probe}_0_{end_time}_24414-0625'
            assert os.path.exists(file_path), f'File not found at {file_path}'
        else:
            print('File Not found, gen_data=False')
            return None
    srate = file.split('_')[-1]
    srate = srate.replace('-', '.')
    srate = float(srate)
    return si.core.BinaryRecordingExtractor(
        file_path, 
        srate, 
        dtype=dtype, 
        num_channels=num_channels, 
        time_axis=1, 
        gain_to_uV=1, 
        is_filtered=False
    )

def load_complete_si_recording(subject, sort_id):
    si_complete_recordings = {}
    recs_in_sorting, start_times, end_times = acr.units.get_time_info(subject, sort_id)
    info_times = acr.info_pipeline.subject_info_section(subject, 'rec_times')
    for probe in ['NNXo', 'NNXr']:
        si_recs = []
        for rec, end_time in zip(recs_in_sorting, end_times):
            t1 = 0 #This should be the case for all sortings so far...
            if (info_times[rec]['duration'] - end_time) < .0001:
                t2 = 0
            else:
                t2 = end_time
            assert type(t2) == int, f't2 is not an integer, it is {t2}'
            si_rec = load_si_recording(subject, rec, probe, gen_data=False)
            si_recs.append(si_rec)
        si_concat = si.core.concatenate_recordings(si_recs)
        si_complete_recordings[probe] = si_concat
    return si_complete_recordings

def get_sorting_start_time(subject, sort_id):
    recs_in_sorting, start_times, end_times = acr.units.get_time_info(subject, sort_id)
    info_times = acr.info_pipeline.subject_info_section(subject, 'rec_times')
    sorting_start_dt = pd.Timestamp(info_times[recs_in_sorting[0]]['start'])
    return sorting_start_dt

def get_relative_rec_start(subject, rec, sort_id, probe='NNXo'):
    sorting_start_time = get_sorting_start_time(subject, sort_id)
    info_times = acr.info_pipeline.subject_info_section(subject, 'rec_times')
    rec_start = pd.Timestamp(info_times[rec]['start'])
    return (rec_start - sorting_start_time).total_seconds()

def get_rec_time_array(subject, rec, sort_id):
    sir = load_si_recording(subject, rec, 'NNXo')
    rel_start = get_relative_rec_start(subject, rec, sort_id)
    times = sir.get_times()
    return times + rel_start


def get_full_time_array(subject, sort_id, overwrite=False):
    corrected_times = []
    recs_in_sorting, start_times, end_times = acr.units.get_time_info(subject, sort_id)
    for rec in recs_in_sorting:
        print(rec)
        corrected_times.append(get_rec_time_array(subject, rec, sort_id))
    
    exp_name = sort_id.split('-')[0]
    if not os.path.exists(f'/Volumes/neuropixel_archive/Data/acr_archive/OFF_data/{subject}/time_arrays'): #make sure time_arrays directory exists
        os.makedirs(f'/Volumes/neuropixel_archive/Data/acr_archive/OFF_data/{subject}/time_arrays')
    
    for file in os.listdir(f'/Volumes/neuropixel_archive/Data/acr_archive/OFF_data/{subject}/time_arrays'):
        if f'times_{subject}_{exp_name}' in file:
            if overwrite == True:
                print(f'times_{subject}_{exp_name} already exists')
                print('Overwriting...')
                np.save(f'/Volumes/neuropixel_archive/Data/acr_archive/OFF_data/{subject}/time_arrays/times_{subject}_{exp_name}', np.hstack(corrected_times))
                return
            elif overwrite == False:
                print(f'times_{subject}_{exp_name} already exists, overwrite=False')
                return
    np.save(f'/Volumes/neuropixel_archive/Data/acr_archive/OFF_data/{subject}/time_arrays/times_{subject}_{exp_name}', np.hstack(corrected_times))
    return

def load_saved_time_array(subject, sort_id, generate=True, exp_name=None):
    if exp_name is None:
        exp_name = sort_id.split('-')[0]
    if os.path.exists(f'/Volumes/neuropixel_archive/Data/acr_archive/OFF_data/{subject}/time_arrays/times_{subject}_{exp_name}.npy'):
        return np.load(f'/Volumes/neuropixel_archive/Data/acr_archive/OFF_data/{subject}/time_arrays/times_{subject}_{exp_name}.npy')
    else:
        if generate==True:
            get_full_time_array(subject, sort_id)
            return np.load(f'/Volumes/neuropixel_archive/Data/acr_archive/OFF_data/{subject}/time_arrays/times_{subject}_{exp_name}.npy')
        else:
            return None

def hypno_times_to_sorting_times(hypno, subject, rec, sort_id):
    rel_start = get_relative_rec_start(subject, rec, sort_id)
    hypno['start_time'] = hypno['start_time'] + rel_start
    hypno['end_time'] = hypno['end_time'] + rel_start
    return hypno

def load_full_hypnogram_for_sorting(subject, sort_id):
    hypnograms = []
    recs_in_sorting, start_times, end_times = acr.units.get_time_info(subject, sort_id)
    for rec in recs_in_sorting:
        h = acr.io.load_hypno(subject, rec, corrections=True, update=True, float=True)
        hc = hypno_times_to_sorting_times(h, subject, rec, sort_id)
        hypnograms.append(hc)
    return pd.concat(hypnograms)