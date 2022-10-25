from openpyxl import load_workbook
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

import kdephys.hypno as kh
import kdephys.plot as kp


import acr
import acr.info_pipeline as aip


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
