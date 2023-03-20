import numpy as np
import acr
import yaml
import matplotlib.pyplot as plt

seq_len = 245760
fs = 24414.0625

def find_all_duplicate_start_indexes(data, seq_len=245760):
    starts = []
    for i in range(0, len(data), 50):
        a1 = data[i:i+49]
        a2 = data[i+seq_len:i+seq_len+49]
        if len(a1) != len(a2):
            return np.array(starts)
        if all(a1 == a2):
            for bk in range(51):
                v1 = data[i-bk]
                v2 = data[i+seq_len-bk]
                if v1 == v2:
                    continue
                else:
                    starts.append( i-bk+1 )
                    break
    return np.array(starts)

def find_duplicate_end_index(start_index, data, seq_len=245760):
    scan = len(data) - start_index
    for i in range(scan-1):
        if data[start_index+i] != data[start_index+seq_len+i]:
            return start_index+(i-1)
    return None

def plot_duplicate(data, s, e, ds, de):
    f, ax = plt.subplots()
    dup_len = len(data[s:e+1])
    dup_spacing = ds-s
    d1 = data[s-dup_len:e+dup_len]
    d2 = data[ds-dup_len:de+dup_len]
    if len(d1) != len(d2):
        d1 = data[s-dup_len:e+1]
        d2 = data[ds-dup_len:de+1]
    tix = np.arange(0, len(d1))
    ax.plot(tix, d1, color='darkgreen')
    ax.plot(tix, d2, color='blue', alpha=0.6)
    ax.axvspan(dup_len, dup_len+dup_len, color='red', alpha=0.2)
    
    return f, ax, dup_len

def check_dup_info_yaml(subject, rec, store):
    dup_info = yaml.safe_load(open('/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/duplication_info.yaml', 'r'))
    if subject in dup_info.keys():
        if rec in dup_info[subject].keys():
            if store in dup_info[subject][rec].keys():
                if 'ends' and 'starts' in dup_info[subject][rec][store].keys():
                    return True
    return False

def _check_for_dups(subject, rec, store):
    di = acr.info_pipeline.load_dup_info(subject, rec, store)
    if di['starts']!=[] and di['ends']!=[]:
        return True
    else:
        return False

def dup_lengths(di):
    lens = []
    for s, e in zip(di['starts'], di['ends']):
        lens.append((e-s)+1)
    return lens

def _confirm_dup_position(start, end, data):
    return all(data[start:end+1] == data[start+seq_len:end+seq_len+1])

def starts_and_ends_standard(data):
    starts = find_all_duplicate_start_indexes(data)
    ends = []
    starts_new = []
    for start in starts:
        end = find_duplicate_end_index(start, data)
        if end-start == 245759:
            if end is not None:
                starts_new.append(start)
                ends.append(end)
    return starts_new, ends

def starts_and_ends_nonst(data):
    starts = find_all_duplicate_start_indexes(data)
    ends = []
    starts_new = []
    for start in starts:
        end = find_duplicate_end_index(start, data)
        if end-start != 245759:
            if end is not None:
                starts_new.append(start)
                ends.append(end)
    return starts_new, ends