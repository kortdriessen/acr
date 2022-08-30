from . import utils as acu

a10_info = {}
a10_info['subject'] = 'ACR_10'
a10_info['complete_key_list'] = ['laser1', 'laser1-bl']
a10_info['paths'] = acu.get_paths(a10_info['subject'], a10_info['complete_key_list'])
a10_info['start_times'] = {}
a10_info['start_times']['laser1'] = 4158
a10_info['start_times']['laser1-bl'] = 0


a11_info = {}
a11_info['subject'] = 'ACR_11'
a11_info['complete_key_list'] = ['laser1', 'laser1-bl']
a11_info['paths'] = acu.get_paths(a11_info['subject'], a11_info['complete_key_list'])
a11_info['start_times'] = {}
a11_info['start_times']['laser1'] = 5974
a11_info['start_times']['laser1-bl'] = 0

