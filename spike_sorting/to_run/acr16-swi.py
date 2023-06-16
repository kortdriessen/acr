from pipeline_tdt import run_pipeline_tdt
import os
import yaml
import pandas as pd
from sort_utils import (
    check_sorting_thresholds,
    check_recs_and_times,
    check_probe_spacing,
)

subject = "ACR_16"
experiment = "swi"
recordings = ["swi-bl", "swi-sd", "swi", "swi-post"]
STORES = ["NNXr", "NNXo"]

NCHANS = 16
T_START = [0, 0, 0, 0]
T_END = [0, 0, 0, 0]


threshhold_params = [4, 8, 2]

probe_spacing = 50
analysis_version = "ks2_5_no-drift-correction"
out_dir = "ssd-raid0"
tag = None

CHECK_SPREADSHEET = "ON"
CHECK_DATA_QUALITY = "ON"
# ------------------------------------------------------------------------
# Run some checks
check_sorting_thresholds(threshhold_params)
check_probe_spacing(probe_spacing)
if CHECK_SPREADSHEET == "ON":
    for store in STORES:
        sort_id = f"{experiment}-{store}"
        times, recs = check_recs_and_times(subject, sort_id)
        if times != T_END:
            print(
                f"WARNING: times {times} do not match T_END {T_END}, chanding T_END to {times}"
            )
            T_END = times
        if recs != recordings:
            print(
                f"WARNING: recordings {recs} do not match recordings {recordings}, chanding recordings to {recs}"
            )
            recordings = recs

if CHECK_DATA_QUALITY == "ON":
    _rc = pd.read_excel(
        "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/master_rec_quality.xlsx"
    )
    for rec in recordings:
        for store in STORES:
            rc = info = (
                _rc.loc[_rc.subject == subject]
                .loc[_rc.recording == rec]
                .loc[_rc.store == store]
            )
            assert (
                rc.empty == False
            ), f"ERROR: {subject}-{rec}-{store} not found in rec_quality spreadsheet"
            assert (
                len(rc) == 1
            ), f"ERROR: could not get exactly one row of rec_quality spreadsheet for {subject}-{rec}-{store}"
            assert (
                rc.duration_match.values[0] == 0
            ), f"ERROR: {subject}-{rec}-{store} duration_match is not zero!"
            if rc.duplicate_found.values[0] != "No":
                assert (
                    rc.duplicates_corrected.values[0] == "yes" or "Yes"
                ), f"ERROR: {subject}-{rec}-{store} has duplicates that are not corrected!"

# Main Pipeline
paths_to_concat = []
for rec in recordings:
    paths_to_concat.append(
        f"/Volumes/neuropixel_archive/Data/acr_archive/{subject}/{subject}-{rec}"
    )


prepro_analysis_name = "prepro_df"
bad_channel_ids = None
artifact_frames_list = None
hyp_artifactual_states = None
sorting_analysis_name = analysis_version
rerun_existing = True
dry_run = False
hyp_paths = []
ks_output_dir_list = []

for STORE in STORES:
    if out_dir == "ssd-raid0":
        output_dir = (
            f"/ssd-raid0/analysis/acr_sorting/{subject}-{experiment}-{STORE}/"
            if tag is None
            else f"/ssd-raid0/analysis/acr_sorting/{subject}-{experiment}-{STORE}-{tag}/"
        )
    elif out_dir == "nvme":
        output_dir = (
            f"/nvme/sorting/{subject}-{experiment}-{STORE}/"
            if tag is None
            else f"/nvme/sorting/{subject}-{experiment}-{STORE}-{tag}/"
        )
    else:
        raise ValueError(f"out_dir {out_dir} not recognized")

    true_output_dir = f"{output_dir}{sorting_analysis_name}"
    ks_output_dir_list.append(true_output_dir)
    tdt_folder_paths_and_sorting_output_dir_list = [(paths_to_concat, output_dir)]
    for tdt_folder_paths, output_dir in tdt_folder_paths_and_sorting_output_dir_list:
        run_pipeline_tdt(
            tdt_folder_paths,
            output_dir,
            store=STORE,
            nchans=NCHANS,
            t_start=T_START,
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
    white_files = [
        f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))
    ]
    for wf in white_files:
        full_path = os.path.join(output_dir, wf)
        os.system(f"rm -rf {full_path}")
    os.system(f"touch {output_dir}success.txt")


# ------------------------------------ Run the Full Quality Metrics ------------------------------------
from pipeline_tdt.postprocessing import run_postprocessing_tdt

postprocessing_analysis_name = "metrics_df"

metrics_names = [
    "firing_rate",
    "isi_violation",
    "snr",
    # 'nn_hit_rate',  # Not implemented yet
    # 'nn_miss_rate',  # not implemented yet
]

n_jobs = 60

if __name__ == "__main__":
    # for (tdt_folder_paths, output_dir) in tdt_folder_paths_and_sorting_output_dir_list:
    for ks_output_dir in ks_output_dir_list:
        run_postprocessing_tdt(
            ks_output_dir,
            metrics_names,
            postprocessing_analysis_name=postprocessing_analysis_name,
            n_jobs=n_jobs,
        )
