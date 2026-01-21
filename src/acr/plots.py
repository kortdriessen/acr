import os

import kdephys.hypno as kh
import kdephys.plot as kp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import acr
import kdephys as kde
from acr.utils import BAND_ORDER, NNXO_BLUE, NNXR_GRAY, SOM_BLUE, materials_root

bp_def = dict(
    sub_delta=(0.5, 2),
    delta=(0.5, 4),
    theta=(4, 8),
    alpha=(8, 13),
    sigma=(11, 16),
    beta=(13, 30),
    low_gamma=(30, 55),
    high_gamma=(65, 90),
    omega=(300, 700),
)


def pub(xticks=False, spine_width=2.5):
    current_path = os.path.dirname(os.path.abspath(__file__))
    plt.style.use(os.path.join(current_path, "plot_styles/acr_pub_large.mplstyle"))
    plt.rcParams["xtick.bottom"] = xticks
    plt.rcParams["axes.linewidth"] = spine_width
    return


def lrg(xticks=False, spine_width=2.5):
    current_path = os.path.dirname(os.path.abspath(__file__))
    plt.style.use(os.path.join(current_path, "plot_styles/acr_pub_large.mplstyle"))
    plt.rcParams["xtick.bottom"] = xticks
    plt.rcParams["axes.linewidth"] = spine_width
    return plt.style.use(
        os.path.join(current_path, "plot_styles/acr_pub_large.mplstyle")
    )


def supl(xticks=False, spine_width=2.5):
    current_path = os.path.dirname(os.path.abspath(__file__))
    plt.style.use(os.path.join(current_path, "plot_styles/acr_pub_supplement.mplstyle"))
    plt.rcParams["xtick.bottom"] = xticks
    plt.rcParams["axes.linewidth"] = spine_width
    return plt.style.use(
        os.path.join(current_path, "plot_styles/acr_pub_supplement.mplstyle")
    )


def simple_bp_lineplot(bp, ax, ss=12, color="k", linewidth=2, hyp=None):
    """
    This is just a plotting function, does not do any calculation except, if ss
    is specified, smooth a copy of the bandpower array for display purposes
    bp --> single channel bandpower data, xr.dataarray

    """
    if ss:
        bp = kde.xr.utils.get_smoothed_da(bp, smoothing_sigma=ss)
    ax = sns.lineplot(x=bp.datetime, y=bp, color=color, linewidth=linewidth, ax=ax)
    if hyp is not None:
        kp.shade_hypno_for_me(hypnogram=hyp, ax=ax)
    return ax


"""
MAIN PLOT #1 --> DELTA-BP AS % OF BASELINE OVER COURSE OF ENTIRE REBOUND
------------------------------------------------------------------------
"""


def simple_shaded_bp(bp, hyp, ax, ss=12, color="k", linewidth=2):
    """
    This is just a plotting function, does not do any calculation except, if ss
    is specified, smooth a copy of the bandpower array for display purposes"""
    if ss:
        bp = kde.xr.utils.get_smoothed_da(bp, smoothing_sigma=ss)
    ax = sns.lineplot(x=bp.datetime, y=bp, color=color, linewidth=linewidth, ax=ax)
    kp.shade_hypno_for_me(hypnogram=hyp, ax=ax)
    return ax


def get_bp_rel(data, comp, comp_hyp, comp_state):
    if comp_state is not None:
        comp = kh.keep_states(comp, comp_hyp, comp_state)

    data_bp = kde.xr.spectral.get_bp_set(data, bp_def)
    comp_bp = kde.xr.spectral.get_bp_set(comp, bp_def)

    comp_mean = comp_bp.mean(dim="datetime")

    data_rel = (data_bp / comp_mean) * 100

    return data_rel


def bp_pair_plot(bp1, bp2, h1, h2, names=["name1", "name2"], ylim=False):
    chans = bp1.channel.values
    fig_height = (10 / 3) * len(chans)
    fig, axes = plt.subplots(
        nrows=len(chans), ncols=2, figsize=(40, fig_height), sharex="col", sharey="row"
    )

    lx1 = fig.add_subplot(111, frameon=False)
    lx1.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    lx1.set_ylabel(
        "Delta Power as % of Baseline NREM Mean", fontsize=16, fontweight="bold"
    )

    for chan in chans:
        simple_shaded_bp(bp1.sel(channel=chan), h1, axes[chan - 1, 0])
        axes[chan - 1, 0].set_title(names[0] + ", Ch-" + str(chan), fontweight="bold")
        axes[chan - 1, 0].set_ylabel(" ")
        axes[chan - 1, 0].set_ylim(0, 500)
        axes[chan - 1, 0].axhline(y=100, linestyle="--", linewidth=1.5, color="k")
        if ylim is not False:
            axes[chan - 1, 0].set_ylim(0, ylim)
        simple_shaded_bp(
            bp2.sel(channel=chan), h2, axes[chan - 1, 1], color="royalblue"
        )
        axes[chan - 1, 1].set_title(
            names[1] + ", Ch-" + str(chan), fontweight="bold", color="darkblue"
        )
        axes[chan - 1, 1].axhline(
            y=100, linestyle="--", linewidth=1.5, color="royalblue"
        )

        # plt.subplots_adjust(wspace=0, hspace=0.25)
    return fig, axes


def bp_plot_set(x, spg, hyp, time="4-Hour", ylim=False):
    # define names
    bl1 = x[0] + "-bl"
    bl2 = x[1] + "-bl"

    # Get relative bandpower states
    bp_rel1 = get_bp_rel(spg[x[0]], spg[bl1], hyp[bl1], ["NREM"]).delta
    bp_rel2 = get_bp_rel(spg[x[1]], spg[bl2], hyp[bl2], ["NREM"]).delta

    # Plot
    fig, axes = bp_pair_plot(bp_rel1, bp_rel2, hyp[x[0]], hyp[x[1]], names=x, ylim=ylim)
    plt.tight_layout(pad=2, w_pad=0)
    fig.suptitle(
        spg["sub"]
        + ", "
        + x[2]
        + " | "
        + spg["dtype"]
        + " | Sleep Rebound as % of Baseline | Delta Bandpower (0.5-4Hz) | "
        + time
        + " Rebound",
        x=0.52,
        y=1,
        fontsize=20,
        fontweight="bold",
    )
    # plt.savefig('/Volumes/paxilline/Data/paxilline_project_materials/fin_plots_all/DELTA_BP-'+spg['sub']+'--'+x[0]+x[1]+'--'+spg['dtype']+'--'+spg['x-time']+'.png', dpi=200)


"------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"


def get_state_spectrogram(spg, hyp, state):
    return spg.sel(datetime=hyp.keep_states(state).covers_time(spg.datetime))


def get_state_psd(spg, hyp, state):
    return get_state_spectrogram(spg, hyp, state).median(dim="datetime")


def get_psd_rel(x, spg, hyp):
    # define names
    bl1 = x[0] + "-bl"
    bl2 = x[1] + "-bl"

    # calc the PSD's
    x1_psd = get_state_psd(spg[x[0]], hyp[x[0]], ["NREM"])
    x2_psd = get_state_psd(spg[x[1]], hyp[x[1]], ["NREM"])

    bl1_psd = get_state_psd(spg[bl1], hyp[bl1], ["NREM"])
    bl2_psd = get_state_psd(spg[bl2], hyp[bl2], ["NREM"])

    rel_psd1 = (x1_psd / bl1_psd) * 100
    rel_psd2 = (x2_psd / bl2_psd) * 100

    return rel_psd1, rel_psd2


def n_freq_bins(da, f_range):
    return da.sel(frequency=slice(*f_range)).frequency.size


def pax_scatter_quantal(df, chan, ax):
    ax.plot(df.index, df, color="k", linewidth=2, linestyle="--")
    ax.scatter(df.index, df, s=100, c="royalblue")
    ax.axhspan(ax.get_ylim()[0] - 5, ymax=0, color="royalblue", alpha=0.2)
    ax.axhspan(ymin=0, ymax=ax.get_ylim()[1] + 5, color="k", alpha=0.2)
    ax.axhline(y=0, color="k", linewidth=2)
    # ax.set_xlabel('Frequency Band')
    # ax.set_ylabel('Paxilline AUC - Saline AUC | Relative to Baselines')
    ax.set_title("Ch-" + str(chan), fontweight="bold")
    return ax


def plot_probe_locations(subject):
    """
    Visualize channel locations for e4ach probe

    Parameters
    ----------
    hist_data : dict
        Dictionary containing probe channel locations with structure:
        {probe_name: {channel_num: {'region': str, 'layer': str}}}
    """
    hist_data = acr.info_pipeline.get_channel_map(subject)
    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot each probe
    for probe in ["NNXr", "NNXo"]:
        channels = hist_data[probe]
        if probe == "NNXo":
            xpos = 0.1
        else:
            xpos = -0.1
        # Get unique regions and layers
        regions = list(set([ch["region"] for ch in channels.values()]))
        layers = list(set([ch["layer"] for ch in channels.values()]))

        # Plot each channel
        for ch_num, ch_data in channels.items():
            y_pos = int(ch_num)  # Channel number determines y position
            # Color by layer
            if ch_data["layer"] == "2/3":
                color = acr.utils.pal_full[0]
            elif ch_data["layer"] == "4":
                color = acr.utils.pal_full[1]
            elif ch_data["layer"] == "5":
                color = acr.utils.pal_full[2]
            elif ch_data["layer"] == "6a":
                color = acr.utils.pal_full[3]
            elif ch_data["layer"] == "6b":
                color = acr.utils.pal_full[4]
            elif ch_data["layer"] == "1":
                color = acr.utils.pal_full[5]
            elif ch_data["layer"] == "7":
                color = "black"
            else:
                color = "gray"

            ax.scatter(xpos, y_pos, c=color, s=300)

        # Customize plot
        ax.set_title(f"{subject} Channel Locations")
        ax.set_ylim(17, 0)  # Inverted y-axis
        ax.set_xlim(-0.5, 0.5)
        ax.set_yticks(range(1, 17))
        ax.set_xticks([-0.25, 0.25])
        ax.set_xticklabels(["NNXr", "NNXo"])
        ax.set_ylabel("Channel Number")

    # Add legend
    legend_elements = [
        plt.scatter([], [], c=acr.utils.pal_full[0], s=300, label="Layer 2/3"),
        plt.scatter([], [], c=acr.utils.pal_full[1], s=300, label="Layer 4"),
        plt.scatter([], [], c=acr.utils.pal_full[2], s=300, label="Layer 5"),
        plt.scatter([], [], c=acr.utils.pal_full[3], s=300, label="Layer 6a"),
        plt.scatter([], [], c=acr.utils.pal_full[4], s=300, label="Layer 6b"),
        plt.scatter([], [], c=acr.utils.pal_full[5], s=300, label="Layer 1"),
        plt.scatter([], [], c="black", s=300, label="White Matter"),
        plt.scatter([], [], c="gray", s=300, label="Unknown"),
    ]
    ax.legend(handles=legend_elements, loc="center right", bbox_to_anchor=(1.3, 0.5))

    plt.tight_layout()
    return fig


def psd_comp_quantal(psd1, psd2, keys, ax):
    df = pd.concat(
        [psd1.to_dataframe("power"), psd2.to_dataframe("power")], keys=keys
    ).rename_axis(index={None: "Condition"})
    ax = sns.lineplot(
        data=df,
        x="frequency",
        y="power",
        hue="Condition",
        palette=["k", "royalblue"],
        ax=ax,
    )
    return ax


def auc_master_plot(x, spg, hyp, time="4-Hour"):
    r1, r2 = get_psd_rel(x, spg, hyp)
    auc_df = auc_bandpowers(r1, r2, bp_def)
    r1_comp = r1.sel(frequency=slice(0, 40))
    r2_comp = r2.sel(frequency=slice(0, 40))
    chans = r1.channel.values
    fig_height = (25 / 6) * len(chans)

    fig, axes = plt.subplots(
        figsize=(30, fig_height), nrows=len(chans), ncols=2, sharex="col"
    )
    for chan in chans:
        psd1 = r1_comp.sel(channel=chan)
        psd2 = r2_comp.sel(channel=chan)
        ax = axes[chan - 1, 0]
        ax = psd_comp_quantal(psd1, psd2, x, ax=ax)
        ax.set(ylabel=" ", xlabel=" ")
        ax.set_title("Ch-" + str(chan), fontweight="bold")
    for chan in auc_df.index:
        pax_scatter_quantal(auc_df.iloc[chan - 1], chan, axes[chan - 1, 1])

    # This block sets the ylabels for both columns
    lx1 = fig.add_subplot(121, frameon=False)
    lx1.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    lx1.set_ylabel("PSD as % of BL", fontsize=16, fontweight="bold")
    lx1.set_xlabel("Frequency", fontsize=16, fontweight="bold")
    lx2 = fig.add_subplot(122, frameon=False)
    lx2.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    lx2.set_ylabel(
        "Paxilline AUC - Saline AUC | Relative to Baselines",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(pad=1.5, w_pad=2)
    fig.suptitle(
        spg["sub"]
        + ", "
        + x[2]
        + " | "
        + spg["dtype"]
        + " | PSD as % of Baseline | "
        + time
        + " Rebound",
        x=0.52,
        y=1,
        fontsize=20,
        fontweight="bold",
    )
    # plt.savefig('/Volumes/paxilline/Data/paxilline_project_materials/fin_plots_all/AUC-'+spg['sub']+'--'+x[0]+x[1]+'--'+spg['dtype']+'--'+spg['x-time']+'.png', dpi=200)


def _gen_rms_plot_lfp(lfp_data, chunk_dur=5, probes=["NNXo", "NNXr"], vmin=0, vmax=500):
    f, ax = plt.subplots(2, 1, figsize=(45, 20))
    for i, probe in enumerate(probes):
        data = lfp_data.sel(store=probe).values
        data = data.T
        fs = lfp_data.fs
        chunk_size = int(chunk_dur * fs)
        num_chunks = data.shape[1] // chunk_size
        data_chunked = data[:, : num_chunks * chunk_size].reshape(
            data.shape[0], num_chunks, chunk_size
        )
        rms_chunked = np.sqrt(np.mean(data_chunked**2, axis=2))
        sns.heatmap(rms_chunked, cmap="inferno", ax=ax[i], vmin=vmin, vmax=vmax)
        new_t = ax[i].get_xticks() * chunk_dur
        ax[i].set_xticks(ax[i].get_xticks())
        ax[i].set_xticklabels(new_t, rotation=45, fontsize=16)
        ax[i].set_yticklabels(range(1, 17))
        ax[i].set_title(f"LFP RMS | {probe}")
    return f, ax


def gen_and_save_rms_plot_lfp(subject, rec, probes=["NNXo", "NNXr"], chunk_dur=5):
    for probe in probes:
        dat = acr.io.load_raw_data(subject, rec, probe, exclude_bad_channels=False)
        data = dat.values
        data = data.T
        fs = dat.fs
        chunk_size = int(chunk_dur * fs)  # 5 seconds * sampling rate
        num_chunks = data.shape[1] // chunk_size
        data_chunked = data[:, : num_chunks * chunk_size].reshape(
            data.shape[0], num_chunks, chunk_size
        )
        # perform rms calculation on each chunk
        rms_chunked = np.sqrt(np.mean(data_chunked**2, axis=2))
        f, ax = plt.subplots(figsize=(45, 20))
        sns.heatmap(rms_chunked, cmap="inferno", ax=ax)
        new_t = ax.get_xticks() * chunk_dur
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(new_t, rotation=45, fontsize=16)
        ax.set_yticklabels(range(1, 17))
        ax.set_title(f"RMS | {subject} | {rec} | {probe}")
        plt.savefig(
            f"{materials_root}plots_presentations_etc/PLOTS_MASTER/single_subject_plots/rms/LFP__{subject}--{rec}--{probe}.png"
        )
        plt.close()
    return


import dask.array as da


def gen_rms_plot_ap(subject, rec, probes=["NNXo", "NNXr"], spacing=300):
    for probe in probes:
        sig = acr.mua.load_processed_mua_signal(subject, rec, probe, version="zarr")
        dx = da.from_zarr(sig.traces_seg0)
        fs = int(sig.attrs["sampling_frequency"])
        total_t = int(sig.traces_seg0.shape[0] / fs)
        dx = dx.rechunk((fs, dx.shape[1]))
        snips_to_grab_starts = np.arange(0, total_t, spacing)
        snips_to_grab_ends = snips_to_grab_starts + 1
        snips_to_grab_starts = snips_to_grab_starts * fs
        snips_to_grab_ends = snips_to_grab_ends * fs

        sliced_dx = da.stack(
            [
                dx[start:end]
                for start, end in zip(snips_to_grab_starts, snips_to_grab_ends)
            ]
        )
        sample_dat = sliced_dx.compute(scheduler="threads")
        rms_values = np.sqrt(np.mean(sample_dat**2, axis=1))
        rms_values = rms_values.T
        f, ax = plt.subplots(figsize=(45, 20))
        sns.heatmap(rms_values, cmap="inferno", ax=ax)
        new_t = ax.get_xticks() * spacing
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(new_t, rotation=75, fontsize=24)
        ax.set_yticklabels(range(1, 17))
        ax.set_title(f"RMS | {subject} | {rec} | {probe}", fontsize=32)
    return f, ax


def gen_and_save_rms_plot_ap(subject, rec, probes=["NNXo", "NNXr"], spacing=300):
    for probe in probes:
        sig = acr.mua.load_processed_mua_signal(subject, rec, probe, version="zarr")
        dx = da.from_zarr(sig.traces_seg0)
        fs = int(sig.attrs["sampling_frequency"])
        total_t = int(sig.traces_seg0.shape[0] / fs)
        dx = dx.rechunk((fs, dx.shape[1]))
        snips_to_grab_starts = np.arange(0, total_t, spacing)
        snips_to_grab_ends = snips_to_grab_starts + 1
        snips_to_grab_starts = snips_to_grab_starts * fs
        snips_to_grab_ends = snips_to_grab_ends * fs

        sliced_dx = da.stack(
            [
                dx[start:end]
                for start, end in zip(snips_to_grab_starts, snips_to_grab_ends)
            ]
        )
        sample_dat = sliced_dx.compute(scheduler="threads")
        rms_values = np.sqrt(np.mean(sample_dat**2, axis=1))
        rms_values = rms_values.T
        f, ax = plt.subplots(figsize=(45, 20))
        sns.heatmap(rms_values, cmap="inferno", ax=ax)
        new_t = ax.get_xticks() * spacing
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(new_t, rotation=75, fontsize=24)
        ax.set_yticklabels(range(1, 17))
        ax.set_title(f"RMS | {subject} | {rec} | {probe}", fontsize=32)
        plt.savefig(
            f"{materials_root}plots_presentations_etc/PLOTS_MASTER/single_subject_plots/rms/AP__{subject}--{rec}--{probe}.png",
            bbox_inches="tight",
        )
        plt.close()
    return


def bp_fan_plot(rebound_df, subject=None, exp=None, title_add=None, ylim=None):
    HUEORD = ["NNXr", "NNXo"]
    PAL = [NNXR_GRAY, NNXO_BLUE]

    # plt.style.use('/home/kdriessen/gh_master/kdephys/kdephys/plot/acr_plots_dark.mplstyle')
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["axes.spines.top"] = True

    nrem_norm = acr.utils.normalize_bp_df_to_contra(rebound_df)
    probe_chan_means = (
        nrem_norm.groupby(["store", "channel", "Band"])["Bandpower"]
        .mean()
        .to_frame()
        .reset_index()
    )
    probe_chan_means.sort_values("channel", ascending=True, inplace=True)
    probe_chan_means.sort_values("store", ascending=False, inplace=True)
    g = sns.relplot(
        probe_chan_means,
        x="store",
        y="Bandpower",
        kind="line",
        col="Band",
        hue="channel",
        palette="viridis_r",
        col_order=BAND_ORDER,
        linewidth=4,
        height=10,
        aspect=0.4,
    )

    g.map_dataframe(
        sns.barplot,
        x="store",
        y="Bandpower",
        hue="store",
        fill=True,
        alpha=0.2,
        hue_order=HUEORD,
        palette=PAL,
    )
    g.set_titles("{col_name}", fontsize=14)
    g.set_xlabels("")
    g.set_xticklabels([" ", " "], fontsize=24, color="cornflowerblue")
    g.set_ylabels("", fontsize=24)
    g.figure.supxlabel("Probe", fontsize=24, color="white")
    tit_text = (
        f"{subject} | {exp} | All Values Relative to Contra. Control Mean | {title_add}"
    )
    g.figure.suptitle(tit_text, fontsize=24, y=1.04, color="white")
    ylim_max = probe_chan_means["Bandpower"].max() + 0.03
    ylim_min = probe_chan_means["Bandpower"].min() - 0.03
    # ylim_max = math.ceil(probe_chan_means['Bandpower'].max()*10)/10
    # ylim_min = math.floor(probe_chan_means['Bandpower'].min()*10)/10
    for ax in g.axes:
        if ylim:
            ax[0].set_ylim(ylim)
        else:
            ax[0].set_ylim(ylim_min, ylim_max)
    plt.show()
    return g


def fr_fan_plot(normdf, subject=None, exp=None, title_add=None):
    HUEORD = ["NNXr", "NNXo"]
    PAL = [NNXR_GRAY, NNXO_BLUE]

    plt.style.use(
        "/home/kdriessen/gh_master/kdephys/kdephys/plot/acr_plots_dark.mplstyle"
    )
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["axes.spines.top"] = True

    probe_chan_means = (
        normdf.groupby(["probe", "channel"])["fr_rel"].mean().to_frame().reset_index()
    )
    probe_chan_means.sort_values("channel", ascending=True, inplace=True)
    probe_chan_means.sort_values("probe", ascending=False, inplace=True)
    g = sns.relplot(
        probe_chan_means,
        x="probe",
        y="fr_rel",
        kind="line",
        hue="channel",
        palette="viridis_r",
        col_order=[1, 2, 3, 4],
        linewidth=4,
        height=10,
        aspect=0.7,
    )

    g.map_dataframe(
        sns.barplot,
        x="probe",
        y="fr_rel",
        hue="probe",
        fill=True,
        alpha=0.2,
        hue_order=HUEORD,
        palette=PAL,
    )
    g.set_titles("{col_name}", fontsize=14)
    g.set_xlabels("")
    g.set_xticklabels([" ", " "], fontsize=24, color="cornflowerblue")
    g.set_ylabels("", fontsize=24)
    g.figure.supxlabel("Probe", fontsize=24, color="white")
    tit_text = (
        f"{subject} | {exp} | All Values Relative to Contra. Control Mean | {title_add}"
    )
    g.figure.suptitle(tit_text, fontsize=24, y=1.04, color="white")
    ylim_max = probe_chan_means["fr_rel"].max() + 0.03
    ylim_min = probe_chan_means["fr_rel"].min() - 0.03
    # ylim_max = math.ceil(probe_chan_means['Bandpower'].max()*10)/10
    # ylim_min = math.floor(probe_chan_means['Bandpower'].min()*10)/10
    for ax in g.axes:
        ax[0].set_ylim(ylim_min, ylim_max)
    plt.show()
    return g


def fr_fan_plot_quantile(normdf, subject=None, exp=None, title_add=None):
    HUEORD = ["NNXr", "NNXo"]
    PAL = [NNXR_GRAY, NNXO_BLUE]

    plt.style.use(
        "/home/kdriessen/gh_master/kdephys/kdephys/plot/acr_plots_dark.mplstyle"
    )
    plt.rcParams["axes.spines.right"] = True
    plt.rcParams["axes.spines.top"] = True

    probe_chan_means = (
        normdf.groupby(["probe", "channel", "quantile"])["fr_rel"]
        .mean()
        .to_frame()
        .reset_index()
    )
    probe_chan_means.sort_values("channel", ascending=True, inplace=True)
    probe_chan_means.sort_values("probe", ascending=False, inplace=True)
    g = sns.relplot(
        probe_chan_means,
        x="probe",
        y="fr_rel",
        kind="line",
        col="quantile",
        hue="channel",
        palette="viridis_r",
        col_order=[1, 2, 3, 4],
        linewidth=4,
        height=10,
        aspect=0.4,
    )

    g.map_dataframe(
        sns.barplot,
        x="probe",
        y="fr_rel",
        hue="probe",
        fill=True,
        alpha=0.2,
        hue_order=HUEORD,
        palette=PAL,
    )
    g.set_titles("{col_name}", fontsize=14)
    g.set_xlabels("")
    g.set_xticklabels([" ", " "], fontsize=24, color="cornflowerblue")
    g.set_ylabels("", fontsize=24)
    g.figure.supxlabel("Probe", fontsize=24, color="white")
    tit_text = (
        f"{subject} | {exp} | All Values Relative to Contra. Control Mean | {title_add}"
    )
    g.figure.suptitle(tit_text, fontsize=24, y=1.04, color="white")
    ylim_max = probe_chan_means["fr_rel"].max() + 0.03
    ylim_min = probe_chan_means["fr_rel"].min() - 0.03
    # ylim_max = math.ceil(probe_chan_means['Bandpower'].max()*10)/10
    # ylim_min = math.floor(probe_chan_means['Bandpower'].min()*10)/10
    for ax in g.axes:
        ax[0].set_ylim(ylim_min, ylim_max)
    plt.show()
    return g


def _gen_box_onesided(
    data1,
    data2,
    colors=[NNXR_GRAY, SOM_BLUE],
    ax=None,
    fsize=(4, 6),
    alpha=0.8,
    lineplot_width=4.5,
    one_sided=False,
    means=True,
    mean_color="gold",
    mean_linewidth=3.5,
    mean_linestyle="--",
    mean_dashes=(1.2, 1.2),
    widths=0.06,
    xlim=(0.35, 0.7),
    median_linewidth=4,
    whisker_linewidth=4,
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=fsize)
    else:
        f = ax.get_figure()

    # boxplot - only NNXo
    box_o = ax.boxplot(
        data2,
        positions=[0.65],
        widths=widths,
        patch_artist=True,
        capprops=dict(color="none", linewidth=0),
        whiskerprops=dict(color="black", linewidth=whisker_linewidth),
        medianprops=dict(color="k", linewidth=median_linewidth, zorder=101),
        showfliers=False,
        showmeans=means,
        meanline=True,
        meanprops=dict(
            color=mean_color,
            linestyle=mean_linestyle,
            linewidth=mean_linewidth,
            dashes=mean_dashes,
            zorder=200,
        ),
    )
    box_o["boxes"][0].set_facecolor(colors[1])
    box_o["boxes"][0].set_alpha(alpha)
    box_o["boxes"][0].set_linewidth(0)
    box_o["boxes"][0].set_zorder(100)

    # line plots
    for i in range(len(data1)):
        ax.plot(
            [0.4, 0.6],
            [data1[i], data2[i]],
            color=colors[1],
            alpha=0.85,
            linewidth=lineplot_width,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    # scatter plots for individual points
    for i in range(len(data1)):
        ax.scatter(0.4, data1[i], color=colors[0], alpha=0.7, s=110, zorder=202)
        ax.scatter(0.6, data2[i], color=colors[1], alpha=0.7, s=110, zorder=203)
    ax.set_xlim(xlim)
    ax.set_xticks([0.4, 0.65])
    return f, ax


def gen_paired_boxplot(
    data1,
    data2,
    colors=[NNXR_GRAY, SOM_BLUE],
    ax=None,
    fsize=(3.5, 4),
    lineplot_width=4.5,
    lineplot_alpha=0.85,
    scatter_alpha=0.7,
    one_sided=False,
    alphas=[0.8, 0.8],
    means=True,
    mean_color="gold",
    mean_linewidth=3.5,
    mean_linestyle="--",
    mean_dashes=(1.2, 1.2),
    widths=0.06,
    xlim=(0.3, 0.7),
    dot_size=110,
    median_linewidth=4,
    whisker_linewidth=4,
):
    if one_sided:
        return _gen_box_onesided(
            data1,
            data2,
            colors,
            ax,
            fsize,
            alphas[1],
            lineplot_width,
            one_sided,
            means,
            mean_color,
            mean_linewidth,
            mean_linestyle,
            mean_dashes,
            widths,
            median_linewidth=median_linewidth,
            whisker_linewidth=whisker_linewidth,
        )

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=fsize)
    else:
        f = ax.get_figure()

    # boxplots
    box = ax.boxplot(
        data1,
        positions=[0.35],
        widths=widths,
        patch_artist=True,
        capprops=dict(color="none", linewidth=0),
        whiskerprops=dict(color="k", linewidth=whisker_linewidth),
        medianprops=dict(color="k", linewidth=median_linewidth, zorder=101),
        showfliers=False,
        showmeans=means,
        meanline=True,
        meanprops=dict(
            color=mean_color,
            linestyle=mean_linestyle,
            linewidth=mean_linewidth,
            dashes=mean_dashes,
            zorder=200,
        ),
    )

    box["boxes"][0].set_facecolor(colors[0])
    box["boxes"][0].set_alpha(alphas[0])
    box["boxes"][0].set_linewidth(0)
    box["boxes"][0].set_zorder(100)

    box_o = ax.boxplot(
        data2,
        positions=[0.65],
        widths=widths,
        patch_artist=True,
        capprops=dict(color="none", linewidth=0),
        whiskerprops=dict(color="black", linewidth=whisker_linewidth),
        medianprops=dict(color="k", linewidth=median_linewidth, zorder=101),
        showfliers=False,
        showmeans=means,
        meanline=True,
        meanprops=dict(
            color=mean_color,
            linestyle=mean_linestyle,
            linewidth=mean_linewidth,
            dashes=mean_dashes,
            zorder=200,
        ),
    )
    box_o["boxes"][0].set_facecolor(colors[1])
    box_o["boxes"][0].set_alpha(alphas[1])
    box_o["boxes"][0].set_linewidth(0)
    box_o["boxes"][0].set_zorder(100)

    # line plots
    for i in range(len(data1)):
        ax.plot(
            [0.40, 0.6],
            [data1[i], data2[i]],
            color=colors[1],
            alpha=lineplot_alpha,
            linewidth=lineplot_width,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    # scatter plots for individual points
    for i in range(len(data1)):
        ax.scatter(
            0.4, data1[i], color=colors[0], alpha=scatter_alpha, s=dot_size, zorder=202
        )
        ax.scatter(
            0.6, data2[i], color=colors[1], alpha=scatter_alpha, s=dot_size, zorder=203
        )
    ax.set_xlim(xlim)
    return f, ax


def add_boxplot(
    ax,
    data,
    positions=[0.5],
    widths=0.06,
    color=NNXR_GRAY,
    means=True,
    mean_color="gold",
    mean_linewidth=3.5,
    mean_linestyle="--",
    mean_dashes=(1.2, 1.2),
    alpha=0.85,
    median_linewidth=3,
    whisker_linewidth=3,
):
    box = ax.boxplot(
        data,
        positions=positions,
        widths=widths,
        showfliers=False,
        patch_artist=True,
        capprops=dict(color="none", linewidth=0),
        whiskerprops=dict(color="k", linewidth=whisker_linewidth),
        medianprops=dict(color="k", linewidth=median_linewidth, zorder=101),
        showmeans=means,
        meanline=True,
        meanprops=dict(
            color=mean_color,
            linestyle=mean_linestyle,
            linewidth=mean_linewidth,
            dashes=mean_dashes,
            zorder=200,
        ),
    )
    box["boxes"][0].set_facecolor(color)
    box["boxes"][0].set_alpha(alpha)
    box["boxes"][0].set_linewidth(0)
    box["boxes"][0].set_zorder(100)
    return ax, box


def add_data_points(ax, data, x_pos=0.4, color=NNXR_GRAY, alpha=0.8, s=110, zorder=202):
    for i in range(len(data)):
        ax.scatter(x_pos, data[i], color=color, alpha=alpha, s=s, zorder=zorder)
    return ax


def add_vertical_kde(
    ax,
    data,
    x_pos=0.3,
    color=NNXR_GRAY,
    alpha=0.4,
    width=0.08,
    n_points=200,
    fill=True,
    linewidth=2,
    zorder=50,
    bw_method=None,
    flip=False,
):
    """
    Add a vertical KDE (kernel density estimate) plot to show data distribution.

    Parameters
    ----------
    ax : matplotlib axis
        The axis to plot on
    data : array-like
        The data to compute the KDE for
    x_pos : float
        The x position where the KDE will be centered
    color : str
        Color for the KDE fill/line
    alpha : float
        Transparency of the fill
    width : float
        The width of the KDE plot (max extent from x_pos)
    n_points : int
        Number of points to evaluate the KDE at
    fill : bool
        Whether to fill the KDE area
    linewidth : float
        Width of the KDE outline
    zorder : int
        Z-order for layering
    bw_method : str, scalar, or callable, optional
        Bandwidth method for gaussian_kde (see scipy.stats.gaussian_kde)
    flip : bool
        If False (default), KDE extends to the left of x_pos.
        If True, KDE extends to the right of x_pos.

    Returns
    -------
    ax : matplotlib axis
    """
    from scipy.stats import gaussian_kde

    data = np.asarray(data)
    data = data[~np.isnan(data)]  # remove NaNs

    if len(data) < 2:
        return ax

    # Compute KDE
    kde = gaussian_kde(data, bw_method=bw_method)

    # Create y-values spanning the data range with some padding
    y_min = data.min()
    y_max = data.max()
    y_range = y_max - y_min
    y_pad = y_range * 0.1
    y_vals = np.linspace(y_min - y_pad, y_max + y_pad, n_points)

    # Evaluate KDE at y values
    density = kde(y_vals)

    # Normalize density to fit within the specified width
    density_norm = (density / density.max()) * width

    # Determine x positions based on flip
    if flip:
        x_edge = x_pos + density_norm
    else:
        x_edge = x_pos - density_norm

    # Plot the KDE as a horizontal fill (which appears vertical)
    if fill:
        ax.fill_betweenx(
            y_vals,
            x_pos,
            x_edge,
            alpha=alpha,
            color=color,
            zorder=zorder,
        )

    # Add outline
    ax.plot(
        x_edge,
        y_vals,
        color=color,
        linewidth=linewidth,
        alpha=min(alpha + 0.3, 1.0),
        zorder=zorder + 1,
    )

    return ax


def add_paired_lines(
    ax, data1, data2, x1, x2, colors=[NNXR_GRAY, SOM_BLUE], alpha=0.85, linewidth=4.5
):
    for i in range(len(data1)):
        ax.plot(
            [x1, x2],
            [data1[i], data2[i]],
            color=colors[1],
            alpha=alpha,
            linewidth=linewidth,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    return ax


def add_paired_scatter_points(ax, data1, data2, x1, x2, colors=[NNXR_GRAY, SOM_BLUE]):
    # scatter plots for individual points
    for i in range(len(data1)):
        ax.scatter(x1, data1[i], color=colors[0], alpha=0.7, s=110, zorder=202)
        ax.scatter(x2, data2[i], color=colors[1], alpha=0.7, s=110, zorder=203)
    return ax


def triple_boxplot(
    left,
    right,
    colors=[NNXR_GRAY, SOM_BLUE],
    ax=None,
    fsize=(6, 4),
    lineplot_width=4.5,
    one_sided=False,
    alphas=[0.8, 0.8],
    means=True,
    mean_color="gold",
    mean_linewidth=3.5,
    mean_linestyle="--",
    mean_dashes=(1.2, 1.2),
    widths=0.06,
    xlim=(0.25, 0.75),
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=fsize)
    else:
        f = ax.get_figure()

    # boxplots
    box = ax.boxplot(
        left,
        positions=[0.3],
        widths=widths,
        patch_artist=True,
        capprops=dict(color="none", linewidth=0),
        whiskerprops=dict(color="k", linewidth=4),
        medianprops=dict(color="k", linewidth=4, zorder=101),
        showfliers=False,
        showmeans=means,
        meanline=True,
        meanprops=dict(
            color=mean_color,
            linestyle=mean_linestyle,
            linewidth=mean_linewidth,
            dashes=mean_dashes,
            zorder=200,
        ),
    )

    box["boxes"][0].set_facecolor(colors[0])
    box["boxes"][0].set_alpha(alphas[0])
    box["boxes"][0].set_linewidth(0)
    box["boxes"][0].set_zorder(100)

    box_o = ax.boxplot(
        right,
        positions=[0.7],
        widths=widths,
        patch_artist=True,
        capprops=dict(color="none", linewidth=0),
        whiskerprops=dict(color="black", linewidth=4),
        medianprops=dict(color="k", linewidth=4, zorder=101),
        showfliers=False,
        showmeans=means,
        meanline=True,
        meanprops=dict(
            color=mean_color,
            linestyle=mean_linestyle,
            linewidth=mean_linewidth,
            dashes=mean_dashes,
            zorder=200,
        ),
    )
    box_o["boxes"][0].set_facecolor(colors[1])
    box_o["boxes"][0].set_alpha(alphas[1])
    box_o["boxes"][0].set_linewidth(0)
    box_o["boxes"][0].set_zorder(100)

    # line plots
    left_mids = np.ones_like(left)
    for i in range(len(left)):
        ax.plot(
            [0.50, 0.35],
            [left_mids[i], left[i]],
            color=colors[0],
            alpha=alphas[0],
            linewidth=4.5,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
    right_mids = np.ones_like(right)
    for i in range(len(right)):
        ax.plot(
            [0.50, 0.65],
            [right_mids[i], right[i]],
            color=colors[1],
            alpha=alphas[1],
            linewidth=4.5,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    # scatter plots for individual points
    for i in range(len(left)):
        ax.scatter(0.5, left_mids[i], color="gray", alpha=0.7, s=110, zorder=202)
        ax.scatter(0.35, left[i], color=colors[0], alpha=0.7, s=110, zorder=203)
    for i in range(len(right)):
        ax.scatter(0.5, right_mids[i], color="gray", alpha=0.7, s=110, zorder=204)
        ax.scatter(0.65, right[i], color=colors[1], alpha=0.7, s=110, zorder=205)
    ax.set_xlim(xlim)
    return f, ax
