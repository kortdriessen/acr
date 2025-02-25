import streamlit as st
import acr
import os

subject = st.text_input('Subject Name: ', 'ACR_')
rec = st.text_input('Recording Name: ', '')
probe = st.selectbox('Probe: ', ['NNXr', 'NNXo'])

t1 = st.number_input('Start Time (s): ', 0.0)
t2 = st.number_input('End Time (s): ', 0.0)

exp = acr.info_pipeline.get_exp_from_rec(subject, rec)

data_type = st.selectbox('Data Type: ', ['tdt', 'lfp', 'mua'])

t1 = int(t1)
t2 = int(t2)

if st.button('Launch Viewer'):
    #os.system(f'python /home/kdriessen/gh_master/data_viewer/data_viewer_full.py {subject}--{rec}--{probe} {exp} {t1} {t2} {data_type}')
    st.write(f'python /home/kdriessen/gh_master/data_viewer/data_viewer_full.py {subject}--{rec}--{probe} {exp} {t1} {t2} {data_type}')
