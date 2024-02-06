import marimo

__generated_with = "0.1.82"
app = marimo.App(width="full")


@app.cell
def __():
    import streamlit as st
    import acr.info_pipeline as aip

    from pathlib import Path
    import os
    from openpyxl import load_workbook
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import matplotlib.pyplot as plt
    import yaml

    import kdephys.hypno.hypno as kh
    import kdephys.plot.main as kp

    import acr  
    import acr.info_pipeline as aip
    return Path, acr, aip, kh, kp, load_workbook, np, os, pd, plt, st, yaml


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    subject = mo.ui.text(placeholder='enter subject', label='subject')
    subject
    return subject,


@app.cell
def __(mo):
    recording = mo.ui.text(placeholder='enter recording', label='recording')
    recording
    return recording,


@app.cell
def __(acr, recording, suject):
    hypnogram = acr.io.load_hypno(suject, recording, update=False)
    return hypnogram,


@app.cell
def __(acr, mo, recording, subject):
    rec_times = acr.info_pipeline.subject_info_section(subject.value, 'rec_times')
    duration = rec_times[recording.value]['duration']
    chunk_length = mo.ui.number(start=0, stop=100000, value=7200, label='Chunk Length (default 7200 sec.)')
    chunk_length
    return chunk_length, duration, rec_times


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
