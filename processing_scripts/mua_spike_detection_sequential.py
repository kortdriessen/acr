from acr.mua import preprocess_data_for_mua
from acr.utils import swi_subs_exps
import acr
import os
import argparse

parser = argparse.ArgumentParser(
    description=('set option reverse to run the script in reverse order'),
    epilog='',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument("--reverse", action="store_true", help="If present, run the subjects in reverse.")
args = parser.parse_args()

# Prepare the sub-rec-probe list
sub_rec_probe_list = []
for subject in swi_subs_exps:
    prepro_list = acr.mua.list_all_preprocessed_mua_data(subject)
    for tup in prepro_list:
        sub_rec_probe_list.append(tup)
        

#Reverse the list if reverse option present (allows paralellization)
if args.reverse:
    sub_rec_probe_list = sub_rec_probe_list[::-1]
    
for i, val in enumerate(sub_rec_probe_list):
    subject = val[0]
    recording = val[1]
    probe = val[2]
    try:
        print(f'Running Detection for subject: {subject}, recording: {recording}, probe: {probe}')
        os.system(f'python sub-rec-probe_mua-detection.py {subject}--{recording}--{probe}')
    except Exception as e:
        print(e)
        print('Error in subject: ' + subject + ' recording: ' + recording + ' probe: ' + probe)