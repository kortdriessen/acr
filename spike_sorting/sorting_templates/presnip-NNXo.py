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
            "/Volumes/opto_loc/Data/ACR_15/ACR_15-test-pre-snip",
        ],
        "/nvme/sorting/tdt/ACR_15-test-pre-snip/",
    ),
]

# Preprocessing, Must be in 'preprocessing' doc in pipeline_tdt/params/params.yml
prepro_analysis_name = "prepro_df"
bad_channel_ids = None
artifact_frames_list = None

hyp_paths = [
    # "/path/to/block1/hyp.csv",
    # "/path/to/block2/hyp.csv",
]

hyp_artifactual_states = None  # hyp_artifactual_states = ['Art', 'A', 'Artifact', 'artifact'] # Requires hypnograms if specified


# Sorting
sorting_analysis_name = (
    "ks2_5_nblocks=1_8s-batches"  # Must be in 'sorting' doc in analysis_cfg.yaml
)

# Misc
rerun_existing = True
dry_run = False

# Constants
# T_END should be a list where each value corresponds to a path in tdt_folder_paths at the same index
STORE = "NNXo"
NCHANS = 16
T_END = [69500]

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
            bad_channel_ids=bad_channel_ids,
            artifact_frames_list=artifact_frames_list,
            hyp_paths=hyp_paths,
            hyp_artifactual_states=hyp_artifactual_states,
            sorting_analysis_name=sorting_analysis_name,
            # postpro_analysis_name=postpro_analysis_name,
            rerun_existing=rerun_existing,
            dry_run=dry_run,
        )
