import streamlit as st
import tdt
import kd_analysis.main.kd_utils_pandas as kup

path = '/Volumes/opto_loc/Data/ACR_9/ACR_9-control1'
data = kup.tdt_to_pandas(path, t1=0, t2=1200, channel=[1,2], store='EEGr')

avg = data['1'].mean(axis=0)
st.write(avg)