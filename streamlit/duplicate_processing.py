import os
import matplotlib.pyplot as plt
import numpy as np
from cudf_tools.unit_analysis import *
from kdephys.plot.main import *
from kdephys.plot.units import *
import acr
import tdt
plt.style.use('fast')
plt.style.use('/home/kdriessen/github_t2/kdephys/kdephys/plot/acr_plots.mplstyle')
import shutil
from acr.duplication_functions import *

seq_len = 245760
fs = 24414.0625
subject = st.sidebar.text_input('Subject', '')
important_recs = acr.info_pipeline.get_impt_recs(subject)
recs = st.sidebar.multiselect('Recordings', important_recs, None)
stores = st.sidebar.multiselect('Choose stores to process', ['NNXr', 'NNXo'], ['NNXr', 'NNXo'])

if st.button('Check for duplicates'):

    for rec in recs:
        for store in stores:
            # First thing we do is check duplication_info.yaml for whether duplicates actually exist
            if _check_for_dups(subject, rec, store) == False:
                    print(f'ZERO duplicates found for store {store} of recording {rec}')
                    continue
            elif _check_for_dups(subject, rec, store) == True:
                print(f'duplicates found for store {store} of recording {rec}, continuing with pipeline')