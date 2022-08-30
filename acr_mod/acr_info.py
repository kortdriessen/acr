import pandas as pd
acr9_info = {}
acr9_info['subject'] = 'ACR_9'
acr9_info['complete_key_list'] = ['control1', 'laser1']


a9_times_c1 = {}
a9_times_c1['bl_sleep_start'] = 5753
a9_times_c1['stim_on'] = 6574
a9_times_c1['stim_off'] = 20974
a9_times_c1['stim_on_dt'] = pd.Timestamp('2022-06-14 10:40:27.383717400')
a9_times_c1['stim_off_dt'] = pd.Timestamp('2022-06-14 14:40:27.385437720')
a9_times_l1 = {}
a9_times_l1['bl_sleep_start'] = 5753
a9_times_l1['stim_on'] = 9921
a9_times_l1['stim_off'] = 24321
a9_times_l1['stim_on_dt'] = pd.Timestamp('2022-06-17 11:18:33.631271960')
a9_times_l1['stim_off_dt'] = pd.Timestamp('2022-06-17 15:18:33.632992280')
a9_times = {}
a9_times['control1'] = a9_times_c1
a9_times['laser1'] = a9_times_l1