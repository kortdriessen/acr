from pipeline_tdt.postprocessing import run_postprocessing_tdt


ks_output_dir_list = [
    '/Volumes/opto_loc/Data/ACR_18/sorting_data/swi-NNXo/ks2_5_no-drift-correction/',
    '/Volumes/opto_loc/Data/ACR_18/sorting_data/swi-NNXr/ks2_5_no-drift-correction/'
    # "/nvme/neuropixels/sorting/tdt/tmp/ACR_14-laser1-NNXr/ks2_5_nblocks=1_8s-batches",
]  # Output_dir as in run_sorting script

postprocessing_analysis_name = 'metrics_df'

metrics_names = [
    'firing_rate',
    'isi_violation',
    'snr',
    # 'nn_hit_rate',  # Not implemented yet
    # 'nn_miss_rate',  # not implemented yet
]

n_jobs=60

if __name__ == "__main__":

    # for (tdt_folder_paths, output_dir) in tdt_folder_paths_and_sorting_output_dir_list:
    for ks_output_dir in ks_output_dir_list:

        run_postprocessing_tdt(
            ks_output_dir,
            metrics_names,
            postprocessing_analysis_name=postprocessing_analysis_name,
            n_jobs=n_jobs
        )