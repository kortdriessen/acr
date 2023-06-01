import streamlit as st

st.set_page_config(layout="wide")
import acr
import pandas as pd
import os


def get_all_notes(idf):
    all_notes = []
    for row in idf.itertuples():
        for n in row.note:
            if n not in all_notes:
                all_notes.append(n)
            else:
                continue
    return all_notes


subject = st.sidebar.text_input("Subject", "ACR")

unit_df_dir = f"/Volumes/opto_loc/Data/{subject}/sorting_data/spike_dataframes/"
available_sortings = []
for f in os.listdir(unit_df_dir):
    available_sortings.append(f.split(".")[0])

sort_ids = st.sidebar.multiselect("Sort_IDs", available_sortings)

idf = acr.units.load_info_df(subject, sort_ids, exclude_bad_units=False)

all_notes = get_all_notes(idf)

note = st.sidebar.multiselect("Only clusters with these Notes:", all_notes)


if len(note) != 0:
    idf = idf[idf["note"].apply(lambda lst: any(item in lst for item in note))]

st.write(idf)
