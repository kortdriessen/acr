#---------------- Adjust Parameters Here -----------------# 
subject = "ACR_#"
exp = 'EXP'
stores = ['NNXo', 'NNXr']
rel_state='NREM'
#---------------------------------------------------------#


# ----------------------------------------- subject_info + Hypno -----------------------------------------
h = acr.io.load_hypno_full_exp(subject, exp)
si = acr.info_pipeline.load_subject_info(subject)
sort_ids = [f'{exp}-{store}' for store in stores]
recordings = acr.info_pipeline.get_exp_recs(subject, exp)
#---------------------------------------------------------------------------------------------------------


# ----------------------------------------- Load Basic Info -----------------------------------------
stim_start, stim_end = acr.stim.stim_bookends(subject, exp)
reb_start = h.hts(stim_end-pd.Timedelta('15min'), stim_end+pd.Timedelta('1h')).st('NREM').iloc[0].start_time
if reb_start < stim_end:
    stim_end_hypno = h.loc[(h.start_time<stim_end)&(h.end_time>stim_end)] # if stim time is in the middle of a nrem bout, then it can be the start of the rebound
    if stim_end_hypno.state.values[0] == 'NREM':
        reb_start = stim_end
    else:
        raise ValueError('Rebound start time is before stim end time, need to inspect')

assert reb_start >= stim_end, 'Rebound start time is before stim end time'

bl_start_actual = si["rec_times"][f'{exp}-bl']["start"]
bl_day = bl_start_actual.split("T")[0]
bl_start = pd.Timestamp(bl_day + "T09:00:00")

if f'{exp}-sd' in si['rec_times'].keys():
    sd_rec = f'{exp}-sd'
    sd_end = pd.Timestamp(si['rec_times'][sd_rec]['end'])
else:
    sd_rec = exp
    sd_end = stim_start
sd_start_actual = pd.Timestamp(si['rec_times'][sd_rec]['start'])
sd_day = si['rec_times'][sd_rec]['start'].split("T")[0]
sd_start = pd.Timestamp(sd_day + "T09:00:00")