import os
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from kdephys.plot.main import *
from kdephys.plot.units import *
import acr
import tdt

plt.style.use("fast")
plt.style.use("/home/kdriessen/gh_t2/kdephys/kdephys/plot/acr_plots.mplstyle")
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
stores = st.sidebar.multiselect(
    "Choose stores to process", ["NNXr", "NNXo"], ["NNXr", "NNXo"]
)
channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

end_chunk_combos = [f"{rec}-{store}" for rec in recs for store in stores]
end_chunk_removal = st.sidebar.multiselect(
    "Choose recordings and stores to remove end chunks from", end_chunk_combos, None
)

if st.button("Process duplicates"):
    for rec in recs:
        for store in stores:
            # First thing we do is check duplication_info.yaml for whether duplicates actually exist
            if dpf._check_for_dups(subject, rec, store) == False:
                Print(f"No duplicates found for {subject} {rec} {store}")
                continue

            assert dpf._check_for_dups(subject, rec, store) == True
            st.write(f"Processing duplicates for {subject}, {rec}, {store}")
            di = acr.info_pipeline.load_dup_info(subject, rec, store)
            si = acr.info_pipeline.load_subject_info(subject)

            # This block tells us, for each listed duplicate, whether the duplicate at that index is standard or non-standard
            dup_lens = dpf.dup_lengths(di)
            st_dups = []
            nonst_dups = []
            for i, dup in enumerate(dup_lens):
                if dup_lens[i] == 245760:
                    st_dups.append(i)
                if dup_lens[i] < 245760:
                    nonst_dups.append(i)

            # This is where we start actually loading and manipulating data
            for channel in channels:
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

                # correct non-standard duplicates first, because they don't mess with the indeces of other duplicates
                if len(nonst_dups) >= 1:
                    for i, dup_pos in enumerate(nonst_dups):
                        start = di["starts"][dup_pos]
                        end = di["ends"][dup_pos]
                        if (
                            dpf._confirm_dup_position(start, end, data) == False
                        ):  # if the duplicate position is not confirmed, we need to find it (since we only log the duplicate position for one channel, almost always we will have to use this)
                            start, end = dpf.starts_and_ends_nonst(data)
                            start = start[
                                0
                            ]  # can do this because we should always be going after the first non-standard duplicate
                            end = end[0]
                            assert (
                                start - di["starts"][dup_pos] <= 10
                            )  # these are just to ensure that we are taking out the correct duplicate
                            assert end - di["ends"][dup_pos] <= 10

                        assert dpf._confirm_dup_position(start, end, data) == True
                        data[start + seq_len : end + seq_len + 1] = 1e-9

                    if len(st_dups) == 0:
                        corrected_data = data
                    if len(st_dups) >= 1:
                        pass
                    # corrected_data = data if len(st_dups) == 0 else None # we want to end up with the clean data in this variable
                    st.write(
                        "after non-standard duplicates removed, length of data is: ",
                        len(data),
                    )
                    # st.write('after non-standard duplicates removed, length of corrected_data is: ', len(corrected_data))
                # standard duplicate correction (slicing) #TODO: this should and could probably be a lot better... (written on plane 3/4/23)
                if len(st_dups) >= 1:
                    for i, dup_pos in enumerate(st_dups):
                        start = di["starts"][dup_pos]
                        end = di["ends"][dup_pos]

                        if (
                            dpf._confirm_dup_position(start, end, data) == False
                            and i == 0
                        ):
                            print(
                                f"indexes recorded in dup_info did not match, finding actual coordinates now"
                            )
                            start, end = dpf.starts_and_ends_standard(data)
                            start = start[0]
                            end = end[0]
                            assert (
                                start - di["starts"][dup_pos] <= 10
                            )  # these are just to ensure that we are taking out the correct duplicate
                            assert end - di["ends"][dup_pos] <= 10
                        if i == 0:
                            assert dpf._confirm_dup_position(start, end, data) == True
                            slice1 = data[: start + seq_len]
                            slice2 = data[end + seq_len + 1 :]
                            corrected_data = np.concatenate([slice1, slice2])

                        if i > 0 and len(corrected_data) > 1000:
                            print(
                                f"standard duplicate #{i}, need to find duplicate again as original coordinates are no longer valid, will use corrected data"
                            )
                            start, end = dpf.starts_and_ends_standard(corrected_data)
                            start = start[0]
                            end = end[0]

                        if i > 0:
                            assert (
                                dpf._confirm_dup_position(start, end, corrected_data)
                                == True
                            )
                            slice1 = corrected_data[: start + seq_len]
                            slice2 = corrected_data[end + seq_len + 1 :]
                            corrected_data = np.concatenate([slice1, slice2])
                    assert (
                        len(corrected_data) == len(data) - len(st_dups) * seq_len
                    )  # ensures that we end with data of the correct length

                # Finally, we need to check if there are any end chunks to remove, which it makes sense to do here.
                if f"{rec}-{store}" in end_chunk_removal:
                    st.write(
                        f"Length of data before end chunk removal: {len(corrected_data)}"
                    )
                    new_end = len(corrected_data) - 2048
                    corrected_data = data[:new_end]
                    st.write(
                        f"Length of data after end chunk removal: {len(corrected_data)}"
                    )

                # ------------- Here begins the part where we write corrected_data ------------------------

                # we already have the sev file names from when we loaded the data, so we just need to rename the old file and write the new one
                st.write(
                    f"Duplicate correction complete for channel {channel}, writing corrected data || {subject}, {rec}, {store}"
                )

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

st.markdown("---")
st.markdown("## Redo subject info for all recordings")
st.markdown(
    "This removes each of the recordings already selected from all fields of subject_info.yaml, and then runs update_subject_info()"
)

if st.button("Redo subject info for all recordings"):
    acr.info_pipeline.redo_subject_info(subject, recs)
