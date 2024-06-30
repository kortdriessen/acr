import streamlit as st
from kdephys.plot.main import *
import pandas as pd
import kdephys.hypno.hypno as kh
import numpy as np


@st.cache()
def hypno_info(hypno, title=""):
    fo = hypno.fractional_occupancy()
    x = np.arange(0, 7200, 100)
    f, ax = plt.subplots(figsize=(30, 10))
    ax.plot(x, np.ones_like(x), color="k", alpha=0.5)
    ax.set_title(title)
    shade_hypno_for_me(hypno, ax=ax)
    return fo, f


subject = st.text_input("Enter a subject", "ACR_X")
recording = st.text_input("Enter a recording", "sdpi")
chunks = st.text_input(
    "Enter hypno tag (usually chunk number)", "chunk1/chunk2"
)  # min, max, default

chunks = chunks.split("/")  # split into list

hd = {}
standard = chunks[0]
comp = chunks[1]

path1 = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/hypnograms/hypno_{recording}_{standard}.txt"
hd[standard] = kh.load_hypno_file(path1, st=None, dt=False)

path2 = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/hypnograms/PRACTICE_HYPNOS/hypno_{recording}_{comp}.txt"
hd[comp] = kh.load_hypno_file(path2, st=None, dt=False)

for hyp_name in hd.keys():
    st.markdown(f"## {hyp_name}")
    fo = hd[hyp_name].fractional_occupancy()
    fo = fo*100
    st.write(fo)

f, ax = plt.subplots(2, 1, figsize=(30, 10))
for i, hyp_name in enumerate(hd.keys()):
    x = np.arange(0, 7200, 100)
    ax[i].plot(x, np.ones_like(x), color="k", alpha=0.5)
    ax[i].set_title(hyp_name)
    shade_hypno_for_me(hd[hyp_name], ax=ax[i], alpha=0.8)
st.pyplot(f)