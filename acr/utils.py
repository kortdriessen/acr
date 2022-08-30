import xarray as xr
import pandas as pd
import tdt
import hypnogram as hp
import kd_analysis.main.kd_utils as kd
import kd_analysis.main.kd_hypno as kh
import kd_analysis.main.kd_pandas as kpd
import yaml
from pathlib import Path

bp_def = dict(delta1=(0.75, 1.75), delta2=(2.5, 3.5), delta=(0.75, 4), theta=(4, 8), alpha = (8, 13), sigma = (11, 16), beta = (13, 30), low_gamma = (30, 55), high_gamma = (65, 90), hz40 = [39, 41])

def achr_path(sub, x):
    path = '/Volumes/opto_loc/Data/'+sub+'/'+sub+'-'+x
    return path

def get_paths(sub, xl):
    paths = {}
    for x in xl:
        path = achr_path(sub, x)
        paths[x] = path
    return paths

def load_dataset(si, type):
    cond_list = si['complete_key_list']
    sub = si['subject']
    path_root = '/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/'+sub+'/'+'analysis-data/'
    if type != '-hyp':
        ds = kpd.load_dataset(path_root, cond_list, type)
    else:
        ds = {}
        for cond in cond_list:
            path = path_root + cond + type + '.pkl'
            ds[cond] = pd.read_pickle(path)
    return ds

def ss_times(sub, exp, print_=False):
    #Load the relevant times:
    def acr_get_times(sub, exp):
        block_path = '/Volumes/opto_loc/Data/'+sub+'/'+sub+'-'+exp
        ep = tdt.read_block(block_path, t1=0, t2=0, evtype=['epocs'])
        times = {}
        times['bl_sleep_start'] = ep.epocs.Bttn.onset[0]
        times['stim_on'] = ep.epocs.Wdr_.onset[-1]
        times['stim_off'] = ep.epocs.Wdr_.offset[-1]
        dt_start = pd.to_datetime(ep.info.start_date)

        #This get us the datetime values of the stims for later use:
        on_sec = pd.to_timedelta(times['stim_on'], unit='S')
        off_sec = pd.to_timedelta(times['stim_off'], unit='S')
        times['stim_on_dt'] = dt_start+on_sec
        times['stim_off_dt'] = dt_start+off_sec
        return times 
    times = acr_get_times(sub, exp)

    #Start time for scoring is 30 seconds before the button signal was given to inidicate the start of bl peak period. 
    start1 = times['bl_sleep_start'] - 30
    end1 = start1 + 7200
    
    #End time for the second scoring file is when the stim/laser signal is turned off. 
    start2 = end1
    end2 = times['stim_off']
    if print_:
        print('FILE #1'), print(start1), print(end1)
        print('FILE #2'), print(start2), print(end2)
    print('Done loading times')
    return times
