from acr.mua import detect_mua_spikes_si
import argparse

parser = argparse.ArgumentParser(
    description=(
        f"Create one pane per completed subject/probe within "
        f"an existing tmux session and type/run `<command_prefix>+'<subj>,<prb>'`"
        f"in each pane.\n\n"
    ),
    epilog='blah',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument(
    "sub_rec_probe",
    type=str,
    help="should be in the form of subject--recording--probe",
)

args = parser.parse_args()

subject = args.sub_rec_probe.split('--')[0]
recording = args.sub_rec_probe.split('--')[1]
probe = args.sub_rec_probe.split('--')[2]

# preprocess the mua data
try:
    detect_mua_spikes_si(subject, recording, probe, overwrite=False, n_jobs=224, threshold=4, chunk_duration='1s')
except Exception as e:
    print(f'Error for {subject}--{recording}--{probe}: {e}')