import acr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_pulse_train_times(subject, recording, store):
    """Get the index values of the pulse train onsets and offsets

    Args:
        subject (str): subject name
        recording (str): recording name
        store (str): store name under which the pulses are stored
    """
    sub_info = acr.info_pipeline.load_subject_info(subject)
    pulse_ons = sub_info['stim_info'][recording][store]['onsets']
    pulse_offs = sub_info['stim_info'][recording][store]['offsets']
    pulse_ons = pd.to_datetime(pulse_ons)
    pulse_offs = pd.to_datetime(pulse_offs)
    train_ons = [0]
    train_offs = []
    for i in np.arange(0, (len(pulse_ons)-1)):
        diff = pulse_ons[i+1] - pulse_ons[i]
        if diff.total_seconds() > 5:
            train_ons.append(i+1)
            train_offs.append(i)
    train_offs.append(len(pulse_ons)-1)
    return train_ons, train_offs

def get_sorting_stim_start(subject, exp):
    """gives the time at which a stimulation started, in TOTAL time, as it would appear in a kilosort sorting. 

    Args:
        subject (str): subject name
        exp (str): experiment name
    """
    sub_info = acr.info_pipeline.load_subject_info(subject)
    stim_store = sub_info['stim-exps'][exp]
    stim_start = pd.Timestamp( sub_info['stim_info'][exp][stim_store]['onsets'][0] )
    exp_start = pd.Timestamp( sub_info['rec_times'][exp]['start'] )
    recs = acr.info_pipeline.get_exp_recs(subject, exp)
    time_to_stim = (stim_start - exp_start).total_seconds()
    for rec in recs:
        rec_start = pd.Timestamp( sub_info['rec_times'][rec]['start'] )
        if rec_start < exp_start:
            time_to_stim += sub_info['rec_times'][rec]['duration']
    return time_to_stim

def get_total_spike_rate(df, pons, poffs, probes=['NNXr', 'NNXo']):
    """Get the total spike rate (during both pulse-ON and pulse-OFF) for each probe in a dataframe, given a list of pulse on and off times

    Args:
        df (_type_): unit dataframe with spike times
        pons (_type_): pulse-ON times, in seconds
        poffs (_type_): pulse-OFF times, in seconds
        probes (list, optional): list of probe names. Defaults to ['NNXr', 'NNXo'].

    Returns:
        df_rate, f, ax
    """
    total_spike_rate = {}
    total_time = {}
    for probe in probes:
        #pulse-OFF rate
        total_spike_rate[probe] = {}
        total_spike_rate[probe]['pulse-OFF'] = 0
        total_time[f'{probe}-OFF'] = 0
        for i in np.arange(0, len(pons)-1):
            total_spike_rate[probe]['pulse-OFF'] += df.ts(poffs[i], pons[i+1]).prb(probe).cluster_id.count()
            total_time[f'{probe}-OFF'] += (pons[i+1] - poffs[i]).total_seconds()
        total_spike_rate[probe]['pulse-OFF'] = total_spike_rate[probe]['pulse-OFF']/total_time[f'{probe}-OFF']

        # pulse-ON rate
        
        total_spike_rate[probe]['pulse-ON'] = 0
        total_time[f'{probe}-ON'] = 0
        for i in np.arange(0, len(pons)):
            total_spike_rate[probe]['pulse-ON'] += df.ts(pons[i], poffs[i]).prb(probe).cluster_id.count()
            total_time[f'{probe}-ON'] += (poffs[i] - pons[i]).total_seconds()
        total_spike_rate[probe]['pulse-ON'] = total_spike_rate[probe]['pulse-ON']/total_time[f'{probe}-ON']

    f, ax = plt.subplots()
    df_rate = pd.DataFrame.from_dict(total_spike_rate).T
    df_rate.rename(index={'NNXr': 'Probe', 'NNXo': 'Optrode'}, inplace=True)
    df_rate.plot(kind='bar', ax=ax)
    ax.set_ylabel('Spike Rate (spikes/second), All Channels')
    return df_rate, f, ax