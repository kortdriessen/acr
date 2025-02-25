from scipy.signal import savgol_filter
import pandas as pd
import numpy as np

def get_off_samples(fp, off_starts, off_ends, duration_range=(.100, .105), nbefore=20, nafter=80):
    start_times = []
    end_times = []
    for s, e in zip(off_starts, off_ends):
        if (e-s).total_seconds() > duration_range[0] and (e-s).total_seconds() < duration_range[1]:
            start_times.append(s)
            end_times.append(e)
    time_vals = fp.datetime.values
    dat = fp.values
    off_samples = []
    for st, en in zip(start_times, end_times):
        start_time = fp.ts(st, en).datetime.values[0]
        start_index = np.searchsorted(time_vals, start_time)
        pre_index = start_index - nbefore
        post_index = start_index + nafter
        off_data = dat[pre_index:post_index, :]
        off_samples.append(off_data)
    return off_samples

def get_off_samples_offset(fp, off_starts, off_ends, duration_range=(.100, .105), nbefore=20, nafter=20):
    start_times = []
    end_times = []
    for s, e in zip(off_starts, off_ends):
        if (e-s).total_seconds() > duration_range[0] and (e-s).total_seconds() < duration_range[1]:
            start_times.append(s)
            end_times.append(e)
    time_vals = fp.datetime.values
    dat = fp.values
    off_samples = []
    for st, en in zip(start_times, end_times):
        offset_time = fp.ts(en, en+pd.to_timedelta('5s')).datetime.values[0]
        offset_index = np.searchsorted(time_vals, offset_time)
        pre_index = offset_index - nbefore
        post_index = offset_index + nafter
        off_data = dat[pre_index:post_index, :]
        off_samples.append(off_data)
    return off_samples

def get_onset_offset_samples(fp, off_starts, off_ends, duration_range=(.070, .130), duration_epoc=.005, nbefore=40, nafter=40):
    
    time_vals = fp.datetime.values
    dat = fp.values
    onset_samples = {}
    offset_samples = {}
    
    for i in np.arange(duration_range[0], duration_range[1], duration_epoc):
        start_times = []
        end_times = []
        
        i = round(i, 3)
        print(i)
        
        dur_low = i
        dur_high = i + duration_epoc
        dur_low = round(dur_low, 3)
        dur_high = round(dur_high, 3)
        for s, e in zip(off_starts, off_ends):
            if (e-s).total_seconds() > dur_low and (e-s).total_seconds() < dur_high:
                start_times.append(s)
                end_times.append(e)
        onset_samples[i] = []
        offset_samples[i] = []
        for st, en in zip(start_times, end_times):
            start_time = fp.ts(st, en).datetime.values[0]
            start_index = np.searchsorted(time_vals, start_time)
            pre_index = start_index - nbefore
            post_index = start_index + nafter
            onset_data = dat[pre_index:post_index, :]
            onset_samples[i].append(onset_data)
            offset_time = fp.ts(en, en+pd.to_timedelta('5s')).datetime.values[0]
            offset_index = np.searchsorted(time_vals, offset_time)
            pre_index = offset_index - nbefore
            post_index = offset_index + nafter
            offset_data = dat[pre_index:post_index, :]
            offset_samples[i].append(offset_data)
    return onset_samples, offset_samples
    
def get_slope_df(onsamps, offsamps):
    # average all of the samples for each key
    onset_avg = {}
    offset_avg = {}
    for i, epoc in enumerate(onsamps.keys()):
        epoc_name = round(epoc, 3)
        onset_avg[epoc_name] = np.mean(onsamps[epoc], axis=0)
        offset_avg[epoc_name] = np.mean(offsamps[epoc], axis=0)
    df = pd.DataFrame()
    for epoc in onset_avg.keys():
        for chan in range(16):
            time = np.arange(-40, 40, 1)
            chan_df = pd.DataFrame({"data": onset_avg[epoc][:, chan], "time":time, "onoff":"onset", "channel": chan+1, "epoc": epoc})
            df = pd.concat([df, chan_df])
    for epoc in offset_avg.keys():
        for chan in range(16):
            time = np.arange(-40, 40, 1)
            chan_df = pd.DataFrame({"data": offset_avg[epoc][:, chan], "time":time, "onoff":"offset", "channel": chan+1, "epoc": epoc})
            df = pd.concat([df, chan_df])
    return df

def create_off_samp_df(off_samps, off_range='0.8'):
    dfs = []
    for noff, op in enumerate(off_samps):
        t = np.arange(op.shape[0])
        for i in range(op.shape[1]):
            dfs.append(pd.DataFrame(dict(t=t, d=op[:, i], channel=i+1, off_num=noff+1, off_range=off_range)))
    return pd.concat(dfs, ignore_index=True)

def compute_instantaneous_slope_savgol(voltage, sampling_rate=400, window_length=11, polyorder=3):
    """
    Computes the instantaneous slope of the voltage signal using Savitzky-Golay filter.
    
    Parameters:
    - voltage (np.ndarray): 1D array of voltage samples.
    - sampling_rate (float): Sampling rate in Hz. Default is 400 Hz.
    - window_length (int): The length of the filter window (must be odd). Default is 11.
    - polyorder (int): The order of the polynomial used to fit the samples. Default is 3.
    
    Returns:
    - slope (np.ndarray): 1D array of instantaneous slope values.
    """
    
    dt = 1.0 / sampling_rate
    
    # Compute the first derivative
    slope = savgol_filter(voltage, window_length=window_length, polyorder=polyorder, deriv=1, delta=dt)
    return slope

def compute_instantaneous_slope_numpy(voltage, sampling_rate=400):
    """
    Computes the instantaneous slope of the voltage signal using numpy's gradient.
    
    Parameters:
    - voltage (np.ndarray): 1D array of voltage samples.
    - sampling_rate (float): Sampling rate in Hz. Default is 400 Hz.
    
    Returns:
    - slope (np.ndarray): 1D array of instantaneous slope values.
    """
    dt = 1.0 / sampling_rate
    slope = np.gradient(voltage, dt)
    return slope