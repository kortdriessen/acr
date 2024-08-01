from pathlib import Path
from turtle import width
import pandas as pd
import plotly.express as px
import streamlit as st
import seaborn as sns
import os
import numpy as np

root = st.text_input(
    "Choose root folder", "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/ferrules/"
)

mode = st.selectbox("Choose mode", ["comma", "semicolon"])

def read_ferrule_data(path, mode=mode):
    """
    Reads the ferrule data from the specified path (single .csv file).
    """

    path = Path(path)

    if mode == 'semicolon':
        # Read the ferrule data
        ferrule_data = pd.read_csv(path, header=18, sep=";")
        ferrule_data = ferrule_data.replace(",", ".", regex=True).astype(float)
    elif mode == 'comma':
        # Read the ferrule data
        ferrule_data = pd.read_csv(path, header=18, sep=",")
        #ferrule_data = ferrule_data.replace(",", ".", regex=True).astype(float)

    # Convert to seconds and microWatts
    ferrule_data["Power (W)"] = ferrule_data["Power (W)"] * 1e6
    ferrule_data["Time (ms)"] = ferrule_data["Time (ms)"] / 1e3
    # compute the temporal resolution:
    temporal_resolution = np.diff(ferrule_data["Time (ms)"].values).mean()
    

    # rename the columns with the correct units
    ferrule_data.rename(
        columns={"Time (ms)": "Time (s)", "Power (W)": "Power (uW)"}, inplace=True
    )

    # add a column for the ferrule ID
    # ferrule_data["Ferrule"] = path.stem.split("-")[0]

    # add a column for the patch-cord identifier (or fiber split)
    # ferrule_data["Fiber"] = path.stem.split("-")[1]

    # add a column for the knob value
    # ferrule_data["Knob"] = path.stem.split("-")[-1]

    return ferrule_data, temporal_resolution


def get_ferrule_names(root):
    root = Path(root)
    files = [str(p) for p in root.glob("*.csv")]
    ferrule_names = set()
    for f in files:
        f = Path(f)
        ferrule_names.add(os.path.split(f)[1])
    return ferrule_names


ferrule_options = get_ferrule_names(root)

ms = st.multiselect("choose ferrules to view", ferrule_options)

for m in ms:
    fd, tr = read_ferrule_data(os.path.join(root, m))
    diff = np.diff(fd["Time (s)"].values).mean()
    tr = np.mean(diff)*1e3
    tr_min = np.min(diff)*1e3
    tr_max = np.max(diff)*1e3
    tr_std = np.std(diff)*1e3
    fig = px.line(fd, x="Time (s)", y="Power (uW)", width=1500, height=600, title=f'{m} || temporal resolution: {tr:.2f} ms, min: {tr_min:.2f} ms, max: {tr_max:.2f} ms, std: {tr_std:.2f} ms')
    st.plotly_chart(fig)
