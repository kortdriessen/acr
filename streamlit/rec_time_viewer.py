import acr
import acr.info_pipeline as aip
import streamlit as st
import numpy as np

subject = st.sidebar.text_input("Subject:", "ACR_14")

f, ax = aip.data_range_plot(subject)
st.pyplot(f)

info = aip.load_subject_info(subject)
all_recs = info["recordings"]

sort_recs = st.sidebar.multiselect(
    "Consecutive Recordings for SpikeSorting Reference:", all_recs
)

durations = [info["rec_times"][rec]["duration"] for rec in sort_recs]

st.write(durations)

n = len(durations)

count = np.arange(0, n)
tracker = 0


for c in count:
    tracker = tracker + durations[c]
    st.write("After element", c, "the total duration is", tracker)
