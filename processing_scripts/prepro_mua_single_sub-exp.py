from acr.mua import preprocess_data_for_mua
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
    "subexp",
    type=str,
    help="should be in the form of subject--experiment",
)

args = parser.parse_args()

subject = args.subexp.split('--')[0]
experiment = args.subexp.split('--')[1]


# preprocess the mua data
try:
    preprocess_data_for_mua(subject, experiment, probes=['NNXo', 'NNXr'], overwrite=False, njobs=16)
except Exception as e:
    print(f'Error for {subject}--{experiment}: {e}')