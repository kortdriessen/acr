import streamlit as st
from kdephys.plot.main import *
import pandas as pd
import kdephys.hypno.hypno as kh

@st.cache()
def hypno_info(hypno, title=''):
    fo = hypno.fractional_occupancy()
    x = np.arange(0, 7200, 100)
    f, ax = plt.subplots(figsize=(30, 10))
    ax.plot(x, np.ones_like(x), color='k', alpha=0.5)
    ax.set_title(title)
    shade_hypno_for_me(hypno, ax=ax)
    return fo, f

subject = st.text_input("Enter a subject", "ACR_X")
recording = st.text_input("Enter a recording", "sdpi")
chunks = st.text_input("Enter chunks", 'chunk1/chunk2') # min, max, default

chunks = chunks.split('/') # split into list

hd={}
for chunk in chunks:
    path = f'/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/hypnograms/hypno_{recording}_{chunk}.txt'
    hd[chunk] = kh.load_hypno_file(path, st=None, dt=False)

for hyp_name in hd.keys():
    st.markdown(f'## {hyp_name}')
    fo = hd[hyp_name].fractional_occupancy()
    st.write(fo)
    
f, ax = plt.subplots(2, 1, figsize=(30, 10))
for i, hyp_name in enumerate(hd.keys()):
    x = np.arange(0, 7200, 100)
    ax[i].plot(x, np.ones_like(x), color='k', alpha=0.5)
    ax[i].set_title(hyp_name)
    shade_hypno_for_me(hd[hyp_name], ax=ax[i])
st.pyplot(f)