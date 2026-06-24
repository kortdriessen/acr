import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import xarray as xr
path = "/Volumes/opto_loc/Data/ACR_39/swi-bl-NNXo.nc"
da = xr.open_dataarray(path)
#-------------------------- Standard Imports --------------------------#

import matplotlib.pyplot as plt

import acr

from acr.utils import SOM_BLUE, ACR_BLUE, LASER_BLUE, NNXR_GRAY, NNXO_BLUE, EMG_SLATE, BACKUP_RED


#-------------------------- adjust here --------------------------#
subject = 'ACR_57'
exp = 'mart'
interpol=False
#----------------------------------------------------------------#


#acr.info_pipeline.preprocess_and_save_recording(subject, f'{exp}-bl', stores=['NNXo', 'NNXr'], interpol=interpol)
#acr.info_pipeline.preprocess_and_save_recording(subject, f'{exp}', stores=['NNXo', 'NNXr'], interpol=interpol)
# Process Bandpower
stores = ['NNXo', 'NNXr']
recs = acr.info_pipeline.get_exp_recs(subject, exp)
acr.io.MT_calc_and_save_bandpower_sets(
        subject, stores=stores, recordings=recs,
    )
# Process MUA Data
acr.mua.full_mua_pipeline_for_subject(
            subject,
            list_of_exps=[exp],
            overwrite=False,
            interpol=interpol,
            df_version="concat",
            detect_jobs=32,
        )