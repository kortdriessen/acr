from pipeline_tdt import run_pipeline_tdt

"""Copy, modify and run to run the whole pipeline for several datasets."""

tdt_folder_paths_and_sorting_output_dir_list = [
    # (
    #     [
    #         <tdt_folder_to_concat_1>,
    #         <tdt_folder_to_concat_2>,
    #     ],
    #     <output_dir_1>,
    # ),
    (
        [
            "/Volumes/opto_loc/Data/ACR_13/ACR_13-isolated_test/",
        ],
        "/nvme/sorting/tdt/ACR_13-isolated_test/",
    ),
]

# Preprocessing
prepro_analysis_name = (
    "prepro_df"  # Must be in 'preprocessing' doc in pipeline_tdt/params/params.yml
)
bad_channel_ids = None  #  TODO!! eg ["NNXr-2"]. Applied to all datasets.
artifact_frames_list = None  # Sample/frame indices. ms_before and ms_after params pulled from params.yaml. eg [10000, 110000]. Applied to all datasets. TODO if concatenating


# Sorting
sorting_analysis_name = (
    "ks2_5_no-drift-correction"  # Must be in 'sorting' doc in analysis_cfg.yaml
)

# Misc
rerun_existing = True
dry_run = False

# Constants
STORE = "NNXr"
NCHANS = 16
T_END = 7200

assert bad_channel_ids is None  # TODO check that works fine

if __name__ == "__main__":

    for (tdt_folder_paths, output_dir) in tdt_folder_paths_and_sorting_output_dir_list:

        run_pipeline_tdt(
            tdt_folder_paths,
            output_dir,
            store=STORE,
            nchans=NCHANS,
            t_end=T_END,
            prepro_analysis_name=prepro_analysis_name,
            sorting_analysis_name=sorting_analysis_name,
            # postpro_analysis_name=postpro_analysis_name,
            rerun_existing=rerun_existing,
            dry_run=dry_run,
        )
