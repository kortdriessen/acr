import os
import matplotlib.pyplot as plt
import numpy as np
from kdephys.plot.main import *
from kdephys.plot.utils import *
import acr
import tdt
import streamlit as st

plt.style.use("fast")
plt.style.use("/home/kdriessen/gh_master/kdephys/kdephys/plot/acr_plots.mplstyle")
import shutil
import acr.duplication_functions as dpf
import yaml


seq_len = 245760
fs = 24414.0625
subject = st.sidebar.text_input("Subject", "")

important_recs = acr.info_pipeline.get_impt_recs(subject)
dup_info = yaml.safe_load(
    open("/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/duplication_info.yaml", "r")
)
recq_path = "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/master_rec_quality.xlsx"
rec_quality = pd.read_excel(recq_path)

recs = st.sidebar.multiselect("Recordings", important_recs, None)
store = st.sidebar.selectbox("Choose store to process", ["NNXr", "NNXo"])

channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


if st.button("Remove end chunk"):
    for rec in recs:
        for channel in channels:
            diff = (
                rec_quality.loc[rec_quality.subject == subject]
                .loc[rec_quality.recording == rec]
                .loc[rec_quality.store == store]
                .duration_match.values[0]
            )
            diff = round(diff, 5)
            assert diff == 0.08389

            path = acr.io.acr_path(subject, rec)
            for fl in os.listdir(
                path
            ):  # this also gets us the requiste .sev file names for when we write data
                if ".sev" and store in fl:
                    file = fl
                    break
            base = file.split(f"_{subject}-")[0]
            _post = file.split(f"_{subject}-")[1]
            post = _post.split(f"_{store}")[0]
            sev_file = f"{path}/{base}_{subject}-{post}_{store}_Ch{channel}.sev"
            assert os.path.isfile(sev_file) == True
            old_sev_name = f"{path}/{base}_{subject}-{post}_{store}_Ch{channel}.old"

            # reads the data
            data_raw = tdt.read_sev(sev_file)
            data = data_raw[store].data
            data = data[:-1]  # this is to remove the last sample, which is always 0
            st.write(f"Channel {channel} loaded, Length of raw data: {len(data)}")

            new_end = len(data) - 2048
            corrected_data = data[:new_end]
            st.write(f"Length of data after end chunk removal: {len(corrected_data)}")

            # ------------- Here begins the part where we write corrected_data ------------------------

            # we already have the sev file names from when we loaded the data, so we just need to rename the old file and write the new one
            # some stuff from mark hanus @ TDT
            old_file_size = np.fromfile(sev_file, dtype=np.uint64, count=1)[0]
            with open(sev_file, "rb") as f:
                header = bytearray(f.read(40))

            # read the data array, replace it with corrected_data
            rawsev = tdt.read_sev(sev_file)
            # assert corrected_data[-1] == 0.0, 'last value of corrected_data is not 0.0, unclear whether to remove it'
            st.write(
                f"Length of corrected data just before writing to rawsev[store]: {len(corrected_data)}"
            )
            rawsev[store].data = corrected_data

            # some more stuff from mark hanus (update the new file size in the header)
            st.write(
                f"Length of corrected data just before writing to header: {rawsev[store].data.size}"
            )
            new_file_size = np.uint64(
                rawsev[store].data.size * rawsev[store].data.itemsize + 40
            )
            header[:8] = new_file_size.tobytes()

            # This renames the original (uncorrected) .sev file so it has a .old extension
            print(f"making a copy of channel {channel}")
            shutil.move(sev_file, old_sev_name)

            # this writes the corrected data to the original .sev file name
            with open(sev_file, "wb") as f:
                # write header to new file
                f.write(header)
                # write data array
                rawsev[store].data.tofile(f)

if st.button("Redo Subject info for all Recordings"):
    acr.info_pipeline.redo_subject_info(subject, recs)
