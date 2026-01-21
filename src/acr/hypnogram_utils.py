from openpyxl import load_workbook
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

import kdephys.hypno.hypno as kh
import kdephys.plot as kp


import acr
import acr.info_pipeline as aip
from kdephys.hypno.ecephys_hypnogram import trim_hypnogram
from kdephys.hypno.ecephys_hypnogram import Hypnogram, DatetimeHypnogram
import polars as pl
import kdephys
import math


def gen_config(
    subject="",
    chunks=6,
    chunk_length=7200,
    start_from=1,
    exp="",
    chans=["EMGr-1", "EEG_-1", "EEG_-2", "LFP_-2", "LFP_-6", "LFP_-10", "LFP_-14"],
    downsample=400.0,
):
    """Generates config files for sleep scoring, saves them in config-files directory of subject's materials folder.

    Requires:
        - subject_info.yml needs to be updated
        - template config file needs to be in config-files directory

    Args:
        subject (str, optional): subject name. Defaults to "".
        chunks (int, optional): number of chunks to generate. Defaults to 6.
        chunk_length (int, optional): length in seconds of each chunk. Defaults to 7200.
        start_from (int, optional): the chunk number (not time) to start from. Requires preceding chunks already in place. Defaults to 1.
        exp (str, optional): recording name. Defaults to "".
        chans (list, optional): channels to be used for scoring in form of store-channel. Defaults to ["EMGr-1", "EEG_-1", "EEG_-2", "LFP_-2", "LFP_-6", "LFP_-10", "LFP_-14"].
    """
    config_directory = (
        f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/" + subject + "/config-files/"
    )
    template_name = "ACR_X_sleepscore-config_experiment-chunk.yml"
    template_path = config_directory + template_name
    data_path = f"A:\\Data\\acr_archive\\{subject}\\{subject}-{exp}"
    if start_from == 1:
        start_time = 0
    elif start_from == 0:
        print("Start from 0 is not allowed. Please use 1 instead.")
        return
    else:
        cfg_info = get_config_info(subject)
        start_time = cfg_info[exp]["ends"][start_from - 2]
        print("the start time being used is: ", start_time)
    for rel_num, chunk in enumerate(range(start_from, start_from + chunks)):
        # First load the template file
        with open(template_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        # set the binPath
        data["datasets"][0]["binPath"] = data_path
        # Set the channels
        data["datasets"][0]["chanList"] = chans
        # set the downsample rate
        data["downSample"] = downsample
        # Set the times
        tend = ((rel_num + 1) * chunk_length) + start_time
        data["tStart"] = tend - chunk_length
        data["tEnd"] = tend
        # Save the file with a new name based on the actual subject, experiment, and chunk
        config_path = (
            config_directory + f"{subject}_sleepscore-config_{exp}-chunk{chunk}.yml"
        )
        with open(config_path, "w") as f:
            yaml.dump(data, f)


def get_hypno_times(subject):
    info = aip.load_subject_info(subject)
    root = Path(f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/hypnograms/")
    hyp_info = {}
    for rec in info["recordings"]:
        name_root = f"hypno_{rec}_chunk"
        hyp_info[rec] = {}
        hyp_info[rec]["starts"] = []
        hyp_info[rec]["ends"] = []
        for file in root.glob(f"{name_root}*"):
            file = str(file)
            recd = file.split("/")[-1].split(".")[0].split("_")[1:][0]
            chunk = file.split("/")[-1].split(".")[0].split("_")[1:][1]
            config_path = Path(
                f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/config-files/{subject}_sleepscore-config_{recd}-{chunk}.yml"
            )
            config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
            hyp_info[rec]["starts"].append(config["tStart"])
            hyp_info[rec]["ends"].append(config["tEnd"])
    recs = list(hyp_info.keys())
    for rec in recs:
        if not hyp_info[rec]["starts"]:
            hyp_info.pop(rec)
        else:
            hyp_info[rec]["starts"] = sorted(hyp_info[rec]["starts"])
            hyp_info[rec]["ends"] = sorted(hyp_info[rec]["ends"])
            s = hyp_info[rec]["starts"]
            e = hyp_info[rec]["ends"]

            # contiguity check - critcal!
            assert len(s) == len(e), f"Starts and ends of {rec} are not the same length"
            i = np.arange(0, len(s) - 1)
            for i in i:
                assert (
                    e[i] == s[i + 1]
                ), f"End of chunk-{i+1} does not match start of chunk-{i+2} in {rec}!!"
    return hyp_info


def hypno_coverage(subject):
    hypno_times = get_hypno_times(subject)
    info = aip.load_subject_info(subject)
    for rec in hypno_times.keys():
        h = acr.io.load_hypno(subject, rec)
        rec_start = np.datetime64(info["rec_times"][rec]["start"])
        rec_end = np.datetime64(info["rec_times"][rec]["end"])
        hypno_start = hypno_times[rec]["starts"][0]
        hypno_start_td = pd.to_timedelta(hypno_start, unit="s")
        hypno_start_dt = rec_start + hypno_start_td
        hypno_end = hypno_times[rec]["ends"][-1]
        hypno_end_td = pd.to_timedelta(hypno_end, unit="s")
        hypno_end_dt = rec_start + hypno_end_td
        f, ax = plt.subplots(figsize=(12, 3))
        datetime_ix = pd.date_range(rec_start, rec_end, freq="5min")
        ax.plot(datetime_ix, np.ones(len(datetime_ix)), "k")
        kp.shade_hypno_for_me(h, ax=ax)
        # ax.axvspan(hypno_start_dt, hypno_end_dt, color='green', alpha=0.5)
        ax.set_title(rec + " Hypnogram Coverage")
    return


def get_config_info(subject):
    info = aip.load_subject_info(subject)
    root = Path(f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/config-files/")
    cfg_info = {}
    for rec in info["recordings"]:
        name_root = f"{subject}_sleepscore-config_{rec}-chunk"
        cfg_info[rec] = {}
        cfg_info[rec]["starts"] = []
        cfg_info[rec]["ends"] = []
        cfg_info[rec]["chunk_num"] = []

        for file in root.glob(f"{name_root}*"):
            file = str(file)
            chunk_number = int(
                file.split("/")[-1]
                .split(".")[0]
                .split("_")[-1]
                .split("-")[-1]
                .split("k")[-1]
            )
            assert (
                type(chunk_number) == int
            ), f"Chunk number is not an integer for {file}"
            cfg_info[rec]["chunk_num"].append(chunk_number)
            config = yaml.load(open(file, "r"), Loader=yaml.FullLoader)
            cfg_info[rec]["starts"].append(config["tStart"])
            cfg_info[rec]["ends"].append(config["tEnd"])
    for key in list(cfg_info.keys()):
        if not cfg_info[key]["starts"]:
            cfg_info.pop(key)
        else:
            cfg_info[key]["starts"] = sorted(cfg_info[key]["starts"])
            cfg_info[key]["ends"] = sorted(cfg_info[key]["ends"])
            cfg_info[key]["chunk_num"] = sorted(cfg_info[key]["chunk_num"])
    return cfg_info


def config_visualizer(subject):
    cfg_info = get_config_info(subject)
    for key in list(cfg_info.keys()):
        x1 = 0
        x2 = cfg_info[key]["ends"][-1]
        x_axis = np.arange(x1, x2, 10)
        f, ax = plt.subplots(figsize=(12, 3))
        ax.plot(x_axis, np.ones(len(x_axis)), "k")
        for i in np.arange(0, len(cfg_info[key]["starts"])):
            ax.axvspan(
                cfg_info[key]["starts"][i],
                cfg_info[key]["ends"][i],
                color="green",
                alpha=0.5,
            )
            ax.axvline(cfg_info[key]["starts"][i], color="k", linestyle="--")
            ax.axvline(cfg_info[key]["ends"][i], color="k", linestyle="--")
            chunk_distance = cfg_info[key]["ends"][i] - cfg_info[key]["starts"][i]
            chunk_num = str(cfg_info[key]["chunk_num"][i])
            ax.text(cfg_info[key]["starts"][i], 2.0, f"Chunk-{chunk_num}", fontsize=8)
            ax.text(cfg_info[key]["starts"][i], 1.5, f"{chunk_distance}s", fontsize=8)
        ax.set_title(key)
        ax.set_ylim(-3, 3)
        plt.show()
        return f


def update_hypno_info(subject, kd=False, overwrite=False):
    ss_path = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/hypnogram_tracker.xlsx"
    hypno_info_ss = pd.read_excel(ss_path, sheet_name=subject)
    hypno_book = load_workbook(
        f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/hypnogram_tracker.xlsx"
    )
    hypno_info = get_hypno_times(subject)
    recs_list = []
    starts_ends = []
    chunks = []
    scored = []
    for rec in list(hypno_info.keys()):
        for i in np.arange(0, len(hypno_info[rec]["starts"])):
            recs_list.append(rec)
            starts_ends.append(
                (hypno_info[rec]["starts"][i], hypno_info[rec]["ends"][i])
            )
            chunk_num = i + 1
            chunks.append(chunk_num)
            path = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/hypnograms/hypno_{rec}_chunk{chunk_num}.txt"
            hyp = kh.load_hypno_file(path, st=None, dt=False)
            fo = hyp.fractional_occupancy()
            if "Unsure" in list(fo.keys()):
                if fo["Unsure"] < 0.02:
                    scored.append("yes")
                else:
                    scored.append("No")
            else:
                scored.append("yes")
    hyp_inf = pd.DataFrame(
        {
            "recording": recs_list,
            "covers_time": starts_ends,
            "chunk": chunks,
            "scoring_complete": scored,
        }
    )
    hyp_inf["subject"] = subject
    if kd:
        hyp_inf["scored_by"] = "KD"

    hypno_info_combined = pd.concat([hypno_info_ss, hyp_inf], axis=0)

    df_to_save = hyp_inf if overwrite else hypno_info_combined
    df_to_save.drop_duplicates(inplace=True, keep=False)

    with pd.ExcelWriter(
        ss_path,
        mode="a",
        engine="openpyxl",
        if_sheet_exists="replace",
    ) as writer:
        df_to_save.to_excel(writer, sheet_name=subject, index=False)
    return


def change_config_path_to_archive(path_to_config):
    cfg = yaml.load(open(path_to_config, "r"), Loader=yaml.FullLoader)
    current_path = cfg["datasets"][0]["binPath"]
    if "L:\\Data" in current_path:
        if "ACR_X-EXPERIMENT" in current_path:
            print("Skipping template config file")
            return
        else:
            new_path = current_path.replace("L:\\Data", "A:\\Data\\acr_archive")
            cfg["datasets"][0]["binPath"] = new_path
            with open(path_to_config, "w") as f:
                yaml.dump(cfg, f)
            print(f"Changed {path_to_config} to {new_path}")
            return
    else:
        print(f"No change to {path_to_config}")
        return


def all_configs_to_archive(subject):
    root = f"/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/{subject}/config-files/"
    for f in Path(root).glob("*.yml"):
        change_config_path_to_archive(f)
    return


def drop_empty_bouts(hyp):
    if type(hyp["duration"][0]) == pd.Timedelta:
        zd = hyp.loc[hyp.duration == pd.to_timedelta(0)]
    elif type(hyp["duration"][0]) == np.float64:
        zd = hyp.loc[hyp.duration == 0]
    else:
        print("duration field is not a float or timedelta, using float64")
        zd = hyp.loc[hyp.duration == 0]
    for ixp in zd.index:
        zp = hyp.index.get_loc(ixp)
        hyp.drop(hyp.index[zp], inplace=True)
    hyp.reset_index(inplace=True, drop=True)
    return hyp


def remove_unsure_bouts(hyp):
    # iterate through all bouts
    flt = True if type(hyp.iloc[0].start_time) == np.float64 else False
    for bout in hyp.loc[hyp.state == "Unsure"].itertuples():
        ix = bout.Index  # gets the index label
        pos = hyp.index.get_loc(
            ix
        )  # gets the actual position in the dataframe using the label

        # if the bout is the last one in the entire hypnogram, remove it
        if ix == hyp.index.max():
            hyp.drop(hyp.index[pos], inplace=True)
            continue

        # assert that neither neighboring bout is unsure #TODO: do I actually need these next two lines?
        # assert hyp.iloc[pos-1].state != 'Unsure', f'bout before Unsure is Unsure - Not allowed. index = {ix}, position = {pos}'
        # assert hyp.iloc[pos+1].state != 'Unsure', f'bout after Unsure is Unsure - Not allowed. index = {ix}, position = {pos}'

        # if the duration of the unsure state is greater than 4 secconds, keep it
        bout_duration = (
            (bout.end_time - bout.start_time)
            if flt
            else (bout.end_time - bout.start_time).total_seconds()
        )
        if bout_duration > 4:
            continue

        # if both states surrounding the unsure are the same, remove the unsure and merge into the previous bout. #TODO: this is dangerous!
        if hyp.iloc[pos - 1].state == hyp.iloc[pos + 1].state:
            prev = False
            nxt = False
            if hyp.iloc[pos].start_time == hyp.iloc[pos - 1].end_time:
                prev = True
            if hyp.iloc[pos].end_time == hyp.iloc[pos + 1].start_time:
                nxt = True
            assert (
                prev == True
            ), f"previous bout is not adjacent to unsure bout. Unsure bout starts at {bout.start_time} and previous bout ends at {hyp.iloc[pos-1].end_time}. Index = {ix}, position = {pos}"
            new_end = hyp.iloc[pos]["end_time"]
            hyp.at[hyp.index[pos - 1], "end_time"] = new_end
            hyp.at[hyp.index[pos - 1], "duration"] = (
                hyp.iloc[pos - 1]["end_time"] - hyp.iloc[pos - 1]["start_time"]
            )  # reset duration field
            row = hyp.index[pos]
            # next_row = h.index[pos+1]
            hyp.drop(row, inplace=True)
            continue

        # if the states surrounding the unsure are different, need to figure out if the merge should happen with the previous or next bout
        if hyp.iloc[pos - 1].state != hyp.iloc[pos + 1].state:
            prev = False
            nxt = False
            if hyp.iloc[pos].start_time == hyp.iloc[pos - 1].end_time:
                prev = True
            if hyp.iloc[pos].end_time == hyp.iloc[pos + 1].start_time:
                nxt = True

            assert (
                prev or nxt == True
            ), "neither previous nor next bout is adjacent to unsure bout"

            if prev:  # merge with previous bout, even if nxt is also True
                new_end = hyp.iloc[pos]["end_time"]
                hyp.at[hyp.index[pos - 1], "end_time"] = new_end
                hyp.at[hyp.index[pos - 1], "duration"] = (
                    hyp.iloc[pos - 1]["end_time"] - hyp.iloc[pos - 1]["start_time"]
                )  # reset duration field
                hyp.drop(hyp.index[pos], inplace=True)
                continue
            if prev == False and nxt == True:  # merge with next bout
                new_start = hyp.iloc[pos]["start_time"]
                hyp.at[hyp.index[pos + 1], "start_time"] = new_start
                hyp.at[hyp.index[pos + 1], "duration"] = (
                    hyp.iloc[pos + 1]["end_time"] - hyp.iloc[pos + 1]["start_time"]
                )  # reset duration field
                hyp.drop(hyp.index[pos], inplace=True)
                continue

    hyp.reset_index(drop=True, inplace=True)
    return hyp


def remove_duplicate_bouts(hyp):
    for bout in hyp.itertuples():
        ix = bout.Index  # gets the index label
        pos = hyp.index.get_loc(
            ix
        )  # gets the actual position in the dataframe using the label

        if ix == hyp.index.min():
            continue
        if bout.state == hyp.iloc[pos - 1].state:
            if bout.start_time == hyp.iloc[pos - 1].end_time:
                new_start = hyp.iloc[pos - 1]["start_time"]
                hyp.at[hyp.index[pos], "start_time"] = new_start
                hyp.at[hyp.index[pos], "duration"] = (
                    hyp.iloc[pos]["end_time"] - hyp.iloc[pos]["start_time"]
                )  # reset duration field
                row = hyp.index[pos - 1]
                hyp.drop(row, inplace=True)
            else:
                continue
    hyp.reset_index(drop=True, inplace=True)
    return hyp


def standard_hypno_corrections(hyp):
    """Applies the standard hypnogram corrections, in order, which is currently:
    1. Remove empty bouts - anything with duration of 0 seconds
    2. Remove unsure bouts - if unsure bout is less than 4 seconds, merge with the previous or next bout
    3. Remove duplicate bouts - if a two bouts are identical and adjacent, merge them into one bout


    Args:
        hyp (Hypnogram DF): Any hypnogram with either Datetime or float64 timestamps
    """
    hyp = drop_empty_bouts(hyp)
    hyp = remove_unsure_bouts(hyp)
    hyp = remove_duplicate_bouts(hyp)
    return hyp


def get_cumulative_rebound_hypno(hypno, reb_start, cum_dur="3600s", states=["NREM"]):
    reb_hypno = trim_hypnogram(
        hypno._df, reb_start, reb_start + pd.Timedelta("6h"), ret_hyp=True
    )
    cum_reb_hypno = reb_hypno.keep_states(states).keep_first(cum_dur)
    return cum_reb_hypno


def get_circadian_match_of_rebound(reb_hypno):
    t1 = reb_hypno["start_time"].min()
    t2 = reb_hypno["end_time"].max()
    blt1 = t1 - pd.Timedelta("24h")
    blt2 = t2 - pd.Timedelta("24h")
    return blt1, blt2


def get_previous_day_times(ref_time, t1="09:00:00", t2="21:00:00"):
    """
    takes a ref_time, and gives you the times t1 and t2 of the day previous to ref_time
    """

    prev_day_ref_time = ref_time - pd.Timedelta("24h")
    prev_day_string = prev_day_ref_time.strftime("%Y-%m-%d %H:%M:%S").split(" ")[0]
    t1 = pd.to_datetime(f"{prev_day_string} {t1}")
    t2 = pd.to_datetime(f"{prev_day_string} {t2}")
    return t1, t2


def get_bl_times(reb_hypno, mode="full"):
    """Get the times for the day before the first time of the rebound hypnogram, according to mode.

    Parameters
    ----------
    reb_hypno : hypno
        The rebound hypnogram.
    mode : str, optional
        default = 'full'
        - full: return the full light period hypnogram of the previous day (9am-9pm)
        - circ: return the circadian-matched hypnogram of the previous day
    """
    reb_start = reb_hypno.start_time.min()
    prev_day = (reb_start - pd.Timedelta("24h")).strftime("%Y-%m-%d").split(" ")[0]
    if mode == "full":
        blt1 = pd.Timestamp(f"{prev_day} 09:00:00")
        blt2 = pd.Timestamp(f"{prev_day} 21:00:00")
    elif mode == "circ":
        t1 = reb_hypno["start_time"].min()
        t2 = reb_hypno["end_time"].max()
        blt1 = t1 - pd.Timedelta("24h")
        blt2 = t2 - pd.Timedelta("24h")
    else:
        raise ValueError('mode must be either "full" or "circ"')
    return blt1, blt2


def _get_hd_as_float(subject, exp, update=False, duration="3600s"):
    hd = acr.hypnogram_utils.create_acr_hyp_dict(
        subject, exp, update=update, duration=duration
    )
    recs, starts, durations = acr.units.get_time_info(subject, f"{exp}-NNXr")
    full_start_time = pd.Timestamp(starts[0])
    float_hd = {}
    for key in hd.keys():
        h = hd[key].copy()
        h["start_time"] = (h["start_time"] - full_start_time).dt.total_seconds()
        h["end_time"] = (h["end_time"] - full_start_time).dt.total_seconds()
        h["duration"] = h["end_time"] - h["start_time"]
        float_hd[key] = Hypnogram(h)
    return float_hd


def get_true_stim_hyp(subject, exp):
    pon, poff = acr.stim.get_individual_pulse_times(subject, exp)
    ton, toff = acr.stim.get_pulse_train_times(pon, poff, times=True)
    durations = [(t_end - t_start).total_seconds() for t_start, t_end in zip(ton, toff)]
    hyp = pd.DataFrame(
        {"start_time": ton, "end_time": toff, "state": "Wake", "duration": durations}
    )
    return DatetimeHypnogram(hyp)


def create_acr_hyp_dict(
    subject,
    exp,
    duration="3600s",
    update=False,
    float_hd=False,
    true_stim=False,
    extra_rebounds=False,
):
    if float_hd:
        return _get_hd_as_float(subject, exp, update=update, duration=duration)
    h = acr.io.load_hypno_full_exp(subject, exp, update=update)
    hd = {}
    sd_true_start, stim_start, stim_end, rebound_start, full_exp_start = (
        acr.info_pipeline.get_sd_exp_landmarks(subject, exp, update=False)
    )
    reb_hypno = acr.hypnogram_utils.get_cumulative_rebound_hypno(
        h, rebound_start, cum_dur=duration
    )
    hd["rebound"] = reb_hypno
    reb_end = reb_hypno.end_time.max()
    xday = pd.Timestamp(rebound_start).strftime("%Y-%m-%d")
    xday_end = pd.Timestamp(xday + " 21:00:00")
    full_bl_t1, full_bl_t2 = acr.hypnogram_utils.get_bl_times(reb_hypno, mode="full")
    circ_bl_t1, circ_bl_t2 = acr.hypnogram_utils.get_bl_times(reb_hypno, mode="circ")
    hd["early_bl"] = (
        h.trim_select(full_bl_t1, full_bl_t2).keep_states(["NREM"]).keep_first(duration)
    )
    hd["circ_bl"] = (
        h.trim_select(circ_bl_t1, full_bl_t2).keep_states(["NREM"]).keep_first(duration)
    )

    if true_stim:
        hd["stim"] = get_true_stim_hyp(subject, exp)
    else:
        hd["stim"] = h.trim_select(stim_start, stim_end)
    hd["early_sd"] = (
        h.trim_select(sd_true_start, stim_start)
        .keep_states(["Wake", "Wake-Good"])
        .keep_first(duration)
    )
    hd["late_sd"] = (
        h.trim_select(sd_true_start, stim_start)
        .keep_states(["Wake", "Wake-Good"])
        .keep_last(duration)
    )
    infinite_end = xday_end + pd.Timedelta("36h")
    if extra_rebounds:
        hd["reb2"] = (
            h.trim_select(reb_end, infinite_end)
            .keep_states(["NREM"])
            .keep_first(duration)
        )
        reb2_end = hd["reb2"].end_time.max()
        hd["reb3"] = (
            h.trim_select(reb2_end, infinite_end)
            .keep_states(["NREM"])
            .keep_first(duration)
        )
        reb3_end = hd["reb3"].end_time.max()
        hd["reb4"] = (
            h.trim_select(reb3_end, infinite_end)
            .keep_states(["NREM"])
            .keep_first(duration)
        )
        reb4_end = hd["reb4"].end_time.max()
        hd["reb5"] = (
            h.trim_select(reb4_end, infinite_end)
            .keep_states(["NREM"])
            .keep_first(duration)
        )
    if extra_rebounds == False:
        hd["late_rebound"] = (
            h.trim_select(reb_end, xday_end).keep_states(["NREM"]).keep_last(duration)
        )
    return hd


def create_hyp_dict_fast(
    subject,
    exp,
    h=None,
    duration="3600s",
    update=False,
    float_hd=False,
    true_stim=False,
    extra_rebounds=False,
):
    """Optimized version of create_acr_hyp_dict with identical functionality.

    Optimizations:
    - Inlines get_bl_times logic to avoid duplicate computation of reb_start_time
    - Caches SD period hypnogram since it's used for both early_sd and late_sd
    - For extra_rebounds, does single trim + keep_states call then subsets from result
    """
    if float_hd:
        return _get_hd_as_float(subject, exp, update=update, duration=duration)

    if h is None:
        h = acr.io.load_hypno_full_exp(subject, exp, update=update)
    hd = {}

    sd_true_start, stim_start, stim_end, rebound_start, full_exp_start = (
        acr.info_pipeline.get_sd_exp_landmarks(subject, exp, update=False, h=h)
    )

    reb_hypno = get_cumulative_rebound_hypno(h, rebound_start, cum_dur=duration)
    hd["rebound"] = reb_hypno
    reb_end = reb_hypno.end_time.max()

    xday = pd.Timestamp(rebound_start).strftime("%Y-%m-%d")
    xday_end = pd.Timestamp(xday + " 21:00:00")

    # Inline get_bl_times logic to avoid computing reb_start_time twice
    reb_start_time = reb_hypno.start_time.min()
    prev_day = (reb_start_time - pd.Timedelta("24h")).strftime("%Y-%m-%d")
    full_bl_t1 = pd.Timestamp(f"{prev_day} 09:00:00")
    full_bl_t2 = pd.Timestamp(f"{prev_day} 21:00:00")
    circ_bl_t1 = reb_start_time - pd.Timedelta("24h")
    # Note: circ_bl_t2 not used in original code (uses full_bl_t2 instead)

    hd["early_bl"] = (
        h.trim_select(full_bl_t1, full_bl_t2).keep_states(["NREM"]).keep_first(duration)
    )
    hd["circ_bl"] = (
        h.trim_select(circ_bl_t1, full_bl_t2).keep_states(["NREM"]).keep_first(duration)
    )

    if true_stim:
        hd["stim"] = get_true_stim_hyp(subject, exp)
    else:
        hd["stim"] = h.trim_select(stim_start, stim_end)

    # Cache SD period hypnogram - used for both early_sd and late_sd
    sd_wake_hypno = h.trim_select(sd_true_start, stim_start).keep_states(
        ["Wake", "Wake-Good"]
    )
    hd["early_sd"] = sd_wake_hypno.keep_first(duration)
    hd["late_sd"] = sd_wake_hypno.keep_last(duration)

    infinite_end = xday_end + pd.Timedelta("36h")

    if extra_rebounds:
        # Single trim + keep_states call, then subset for each subsequent rebound
        post_rebound_nrem = h.trim_select(reb_end, infinite_end).keep_states(["NREM"])

        hd["reb2"] = post_rebound_nrem.keep_first(duration)
        reb2_end = hd["reb2"].end_time.max()

        hd["reb3"] = post_rebound_nrem.trim_select(reb2_end, infinite_end).keep_first(
            duration
        )
        reb3_end = hd["reb3"].end_time.max()

        hd["reb4"] = post_rebound_nrem.trim_select(reb3_end, infinite_end).keep_first(
            duration
        )
        reb4_end = hd["reb4"].end_time.max()

        hd["reb5"] = post_rebound_nrem.trim_select(reb4_end, infinite_end).keep_first(
            duration
        )
    else:
        hd["late_rebound"] = (
            h.trim_select(reb_end, xday_end).keep_states(["NREM"]).keep_last(duration)
        )

    return hd


def add_states_to_dataframe(df, h):
    print("DEPRECATED: use label_df_with_states instead!!")
    start_times = h["start_time"].values
    end_times = h["end_time"].values
    states = h["state"].values
    times = df["datetime"].values

    # Find the indices in the start_times where each time in `times` would be inserted
    indices = np.searchsorted(start_times, times, side="right") - 1

    # Ensure the found index is within bounds and the time is within the bout
    indices = np.clip(indices, 0, len(start_times) - 1)
    valid_mask = (times >= start_times[indices]) & (times <= end_times[indices])

    # Create an array of the same shape as `times` filled with the corresponding states
    state_array = np.empty(times.shape, dtype=states.dtype)
    state_array[valid_mask] = states[indices[valid_mask]]

    # If there are times outside of the hypnogram ranges (unlikely in your case)
    # you may want to handle them, e.g., by setting a default state like 'unknown'
    state_array[~valid_mask] = "unlabelled"  # or whatever is appropriate
    df["state"] = state_array
    return df


import kdephys as kde
from kdephys.hypno.hypno import get_states_fast


def label_df_with_states(df, h, col="datetime"):
    times = df[col].to_numpy()
    states = get_states_fast(h, times)
    states = np.array(states)
    return df.with_columns(state=pl.lit(states))


def label_df_with_full_bl(
    df: pl.DataFrame, state: str = "NREM", col: str = "datetime"
) -> pl.DataFrame:
    """Label a 12-hour baseline period in a dataframe.

    Parameters
    ----------
    df : pl.DataFrame
        The dataframe to label with baseline information.
    state : str
        The state to select in the full baseline period. Set to 'None' to not select a state.

    Returns
    -------
    pl.DataFrame
        The dataframe with a 'full_bl' column added, marking the 12-hour baseline period.
    """

    df = df.with_columns(full_bl=pl.lit("False"))
    bl_day = pd.Timestamp(df[col].min().date())
    bl_9am = bl_day + pd.Timedelta("9h")
    bl_9pm = bl_9am + pd.Timedelta("12h")
    df = df.with_columns(
        full_bl=pl.when((pl.col(col) >= bl_9am) & (pl.col(col) <= bl_9pm))
        .then(pl.lit("True"))
        .otherwise(pl.lit("False"))
    )

    if state != "None":
        df = df.with_columns(
            full_bl=pl.when(pl.col("state") != state)
            .then(pl.lit("False"))
            .otherwise(pl.col("full_bl"))
            .alias("full_bl")
        )

    return df


def get_full_bl_hypno(hypno, state=["NREM"]):
    """given the full experimental hypnogram, return the times between 9am and 9pm on the day of the earliest start_time.

    Parameters
    ----------
    hypno : _type_
        _description_
    """
    start_time = hypno["start_time"].min()
    bl_day = pd.Timestamp(start_time.date())
    bl_9am = bl_day + pd.Timedelta("9h")
    bl_9pm = bl_9am + pd.Timedelta("12h")
    bl_hypno = (
        hypno.trim_select(bl_9am, bl_9pm).keep_states(state)
        if type(state) == list
        else hypno.trim_select(bl_9am, bl_9pm).keep_states([state])
    )
    return bl_hypno


def label_df_with_hypno_conditions(
    df, hd, col=None, label_col="condition", max_bouts=1000
):
    if col == None:
        col = "datetime"
    if type(df) == pl.DataFrame:
        return _label_polars_df(df, hd, col, max_bouts=max_bouts, label_col=label_col)
    else:
        df[label_col] = "None"
        for key in hd.keys():
            for bout in hd[key].itertuples():
                df.loc[
                    ((df[col] >= bout.start_time) & (df[col] <= bout.end_time)),
                    label_col,
                ] = key
        return df


def _label_polars_df(df, hd, col, label_col="condition", max_bouts=1000):
    """Vectorised interval labelling for Polars DataFrame.

    This re-implementation is **orders of magnitude faster** than the original
    loop-based version.  It works by:

    1.  Collapsing all hypnogram dictionaries (`hd`) into a *single* Pandas
        DataFrame containing `start_time`, `end_time`, and the desired
        `label_col` (condition) for every bout of interest.
    2.  Converting that table into a `DatetimeHypnogram` and using the highly
        optimised `get_states_fast` routine (numpy-backed searchsorted) to map
        every timestamp in ``df[col]`` to its corresponding condition in **one
        vectorised pass**.

    The behaviour is identical to the original function:
      • Any sample outside all bouts is labelled 'None'.
      • Only the first ``max_bouts + 1`` bouts per key are considered, matching
        the original "``i > max_bouts``" break condition.
    """

    # ------------------------------------------------------------------
    # 1. Build a combined interval table (as Pandas for Hypnogram support)
    # ------------------------------------------------------------------
    interval_tables = []
    for key, hyp in hd.items():
        # Retrieve the underlying DataFrame regardless of Hypnogram wrapper
        bouts_df = getattr(hyp, "_df", hyp)

        # Respect the original inclusive max_bouts logic (i > max_bouts → break)
        if max_bouts is not None:
            bouts_df = bouts_df.iloc[: max_bouts + 1]

        if bouts_df.empty:
            continue

        tmp = pd.DataFrame(
            {
                "start_time": bouts_df["start_time"].values,
                "end_time": bouts_df["end_time"].values,
                "state": key,  # store desired label in the `state` column
            }
        )
        tmp["duration"] = tmp["end_time"] - tmp["start_time"]
        interval_tables.append(tmp)

    # If no intervals were provided, just return the original df with 'None'
    if not interval_tables:
        return df.with_columns(pl.lit("None").alias(label_col))

    intervals_df = pd.concat(interval_tables, ignore_index=True, sort=False)

    # Make sure the intervals are sorted for numpy searchsorted logic
    intervals_df = intervals_df.sort_values("start_time").reset_index(drop=True)

    # Choose the appropriate Hypnogram class based on dtype
    if np.issubdtype(intervals_df["start_time"].dtype, np.floating):
        combined_hyp = Hypnogram(intervals_df)
    else:
        combined_hyp = DatetimeHypnogram(intervals_df)

    # ------------------------------------------------------------------
    # 2. Vectorised lookup of condition for every timestamp in df[col]
    # ------------------------------------------------------------------
    time_values = df[col].to_numpy()
    labels_series = get_states_fast(combined_hyp, time_values)

    # get_states_fast returns 'no_state' for gaps – map this to 'None'
    labels_arr = labels_series.to_numpy()
    labels_arr = np.where(labels_arr == "no_state", "None", labels_arr)

    # ------------------------------------------------------------------
    # 3. Attach the computed labels back to the Polars DataFrame
    # ------------------------------------------------------------------
    df = df.with_columns(pl.Series(name=label_col, values=labels_arr))

    return df


def sel_random_bouts_for_plotting(hd, key, window_size=5, num_times=8, state="NREM"):
    """takes a hypno dict, selects a key. Then finds bouts of NREM at least 30 seconds long and randomly selects a 5 second window from each bout.

    Parameters
    ----------
    hd : _type_
        _description_
    key : _type_
        _description_
    """
    hyp = hd[key].keep_states([state])
    bouts = hyp.loc[hyp.duration > pd.Timedelta(seconds=30)]
    bout_starts = bouts.start_time.values
    bout_ends = bouts.end_time.values
    starts = []
    ends = []
    random_bouts_to_plot = np.random.choice(len(bout_starts), num_times, replace=False)
    for bout in random_bouts_to_plot:
        start, end = pd.Timestamp(bout_starts[bout]), pd.Timestamp(bout_ends[bout])
        duration = (end - start).total_seconds()
        start_time = np.random.choice(int(duration - window_size))
        starts.append(start + pd.Timedelta(seconds=start_time))
        ends.append(start + pd.Timedelta(seconds=start_time + window_size))
    return starts, ends


def get_light_schedule(hypno):
    start = pd.Timestamp(hypno["start_time"].min())
    end = pd.Timestamp(hypno["end_time"].max())
    chunks = (end - start).total_seconds() / 3600 / 12
    chunks = math.ceil(chunks)  # round up to nearest integer
    begin = pd.Timestamp(f'{start.date().strftime("%Y-%m-%d")} 09:00:00')
    times = []
    for i in np.arange(chunks + 1):
        if i == 0:
            times.append(begin)
        else:
            time = times[-1] + pd.Timedelta("12h")
            times.append(time)
    return times
