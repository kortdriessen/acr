import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kdephys as kde
import acr

plt.style.use("fast")
plt.style.use("/home/kdriessen/github_t2/kdephys/kdephys/plot/acr_plots.mplstyle")

subject = st.sidebar.text_input("Enter Subject", "ACR_")
recording = st.sidebar.text_input("Enter Recording", "")
eeg_chan = st.sidebar.number_input("Enter EEG Channel", 1)
lfp_chan = st.sidebar.number_input("Enter LFP Channel", 9)

block_path = acr.io.acr_path(subject, recording)


if st.button("Load and Plot"):
    eeg = kde.xr.io.get_data(block_path, store="EEG_", channel=eeg_chan)
    emg = kde.xr.io.get_data(block_path, store="EMGr", channel=1)
    lfp = kde.xr.io.get_data(block_path, store="LFP_", channel=lfp_chan)

    espg = kde.xr.spectral.get_spextrogram(eeg, f_range=(0, 40))
    lspg = kde.xr.spectral.get_spextrogram(lfp, f_range=(0, 40))
    mspg = kde.xr.spectral.get_spextrogram(emg, f_range=(0, 40))

    mus_bp = kde.xr.spectral.get_bandpower(mspg, f_range=(20, 60))
    eswa = kde.xr.spectral.get_bandpower(espg, f_range=(0.75, 4.5))
    lswa = kde.xr.spectral.get_bandpower(lspg, f_range=(0.75, 4.5))

    f, ax = plt.subplots(3, 1)
    ax[0] = kde.plot.main.spectro_plotter(mspg, ax=ax[0])
    ax[0].set_title("EMG Spectrogram")
    ax[1] = kde.plot.main.spectro_plotter(espg, ax=ax[1])
    ax[1].set_title("EEG Spectrogram")
    ax[2] = kde.plot.main.spectro_plotter(lspg, ax=ax[2])
    ax[2].set_title("LFP Spectrogram")
    st.pyplot(f)

    f2, ax2 = plt.subplots(3, 1, sharex=True)
    ax2[0] = kde.plot.main.bp_plot(mus_bp.smooth(12), ax=ax2[0])
    ax2[0].set_title("EMG total Power")
    ax2[1] = kde.plot.main.bp_plot(eswa.smooth(12), ax=ax2[1])
    ax2[1].set_title("EEG SWA")
    ax2[2] = kde.plot.main.bp_plot(lswa.smooth(12), ax=ax2[2])
    ax2[2].set_title("LFP SWA")
    st.pyplot(f2)
