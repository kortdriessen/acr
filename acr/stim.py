import acr
import numpy as np
import pandas as pd



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