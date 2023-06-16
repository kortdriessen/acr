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


def gen_config(
    subject="",
    chunks=6,
    chunk_length=7200,
    start_from=1,
    exp="",
    chans=["EMGr-1", "EEG_-1", "EEG_-2", "LFP_-2", "LFP_-6", "LFP_-10", "LFP_-14"],
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
