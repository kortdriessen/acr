from pipeline_tdt.postprocessing import run_postprocessing_tdt


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
            "/Volumes/opto_loc/Data/ACR_14/ACR_14-short2",
        ],
        "/nvme/sorting/tdt/ACR_14-short-plug-NNXo/",
    ),
]  # as in run_sorting script


sorting_analysis_name = "ks2_5_nblocks=1_8s-batches"  # Kilsort output is in `output_dir/sorting_analysis_name``

postprocessing_analysis_name = "metrics_df"

metrics_names = [
    "firing_rate",
    "isi_violation",
    "snr",
    # 'nn_hit_rate',  # Not implemented yet
    # 'nn_miss_rate',  # not implemented yet
]

n_jobs = 60

# Constants
STORE = "NNXo"
NCHANS = 16
T_END = 520

if __name__ == "__main__":

    for (tdt_folder_paths, output_dir) in tdt_folder_paths_and_sorting_output_dir_list:

        run_postprocessing_tdt(
            tdt_folder_paths,
            output_dir,
            metrics_names,
            sorting_analysis_name=sorting_analysis_name,
            postprocessing_analysis_name=postprocessing_analysis_name,
            store="NNXr",
            nchans=16,
            t_end=T_END,
            n_jobs=n_jobs,
        )
