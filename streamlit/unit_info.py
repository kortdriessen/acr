import streamlit as st
import acr
import pandas as pd
import os

subject = st.sidebar.text_input('Subject', 'ACR')

unit_df_dir = f"/Volumes/opto_loc/Data/{subject}/sorting_data/spike_dataframes/"
available_sortings = []
for f in os.listdir(unit_df_dir):
    available_sortings.append(f.split(".")[0])

sort_ids = st.sidebar.multiselect('Sort_IDs', available_sortings)

if st.button('Load Info Dataframe'):
    idf = acr.units.load_info_df(subject, sort_ids)
    st.write(idf)
