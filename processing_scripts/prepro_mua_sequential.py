from acr.mua import preprocess_data_for_mua
from acr.utils import swi_subs_exps
import argparse
import subprocess
import time
import os

parser = argparse.ArgumentParser(
    description=('set option reverse to run the script in reverse order'),
    epilog='',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument("--reverse", action="store_true", help="If present, run the subjects in reverse.")
args = parser.parse_args()

# Prepare the sub-exp list
subject_experiment_list = []
for subject in swi_subs_exps:
    for experiment in swi_subs_exps[subject]:
        subject_experiment_list.append((subject, experiment))

#Reverse the list if reverse option present (allows paralellization)
if args.reverse:
    subject_experiment_list = subject_experiment_list[::-1]


for i, val in enumerate(subject_experiment_list):
    subject = val[0]
    experiment = val[1]
    try:
        os.system(f'python prepro_mua_single_sub-exp.py {subject}--{experiment}')
    except Exception as e:
        print(f"Error in {subject}--{experiment}: {e}")