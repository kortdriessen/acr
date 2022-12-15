from pipeline_tdt import run_pipeline_tdt
import os
import yaml
from sort_utils import check_sorting_thresholds, check_recs_and_times, check_probe_spacing

subject = 'ACR_13'
experiment = 'sdpi'
recordings = ['sdpi-bl', 'sdpi']
STORES = ["NNXr", "NNXo"]
location = 'archive'

NCHANS = 16
T_END = [85654, 42853]

threshhold_params = [4, 8, 2]

probe_spacing = 100

# ------------------------------------------------------------------------
#Run some checks
check_sorting_thresholds(threshhold_params)
check_probe_spacing(probe_spacing)
for store in STORES:
    sort_id = f"{experiment}-{store}"
    times, recs = check_recs_and_times(subject, sort_id)
    if times != T_END:
        print(f"WARNING: times {times} do not match T_END {T_END}, chanding T_END to {times}")
        T_END = times
    if recs != recordings:
        print(f"WARNING: recordings {recs} do not match recordings {recordings}, chanding recordings to {recs}")
        recordings = recs

# Main Pipeline
paths_to_concat = []
for rec in recordings:
    if location == 'opto_loc':
        paths_to_concat.append(f'/Volumes/opto_loc/Data/{subject}/{subject}-{rec}')
    elif location == 'archive':
        paths_to_concat.append(f'/Volumes/neuropixel_archive/Data/acr_archive/{subject}/{subject}-{rec}')


prepro_analysis_name = "prepro_df"
bad_channel_ids = None
artifact_frames_list = None
hyp_artifactual_states = None
sorting_analysis_name = ("ks2_5_nblocks=1_64s-batches")
rerun_existing = True
dry_run = False
hyp_paths = []

for STORE in STORES:
    output_dir = f'/ssd-raid0/analysis/acr_sorting/{subject}-{experiment}-{STORE}/'
    tdt_folder_paths_and_sorting_output_dir_list = [(paths_to_concat, output_dir)]
    for (tdt_folder_paths, output_dir) in tdt_folder_paths_and_sorting_output_dir_list:
        run_pipeline_tdt(
            tdt_folder_paths,
            output_dir,
            store=STORE,
            nchans=NCHANS,
            t_end=T_END,
            prepro_analysis_name=prepro_analysis_name,
            bad_channel_ids=bad_channel_ids,
            artifact_frames_list=artifact_frames_list,
            hyp_paths=hyp_paths,
            hyp_artifactual_states=hyp_artifactual_states,
            sorting_analysis_name=sorting_analysis_name,
            # postpro_analysis_name=postpro_analysis_name,
            rerun_existing=rerun_existing,
            dry_run=dry_run,
        )
    # delete whitened data
    white_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    for wf in white_files:
        full_path = os.path.join(output_dir, wf)
        os.system(f'rm -rf {full_path}')
    os.system(f'touch {output_dir}success.txt')