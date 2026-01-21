import os

import numpy as np
import pandas as pd
import polars as pl
import scipy
import yaml
from scipy.ndimage import gaussian_filter

import acr
import kdephys as kde

nodes = [
    "nose",
    "mid_ears",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "mouse_center",
    "left_hip",
    "right_hip",
    "tail_base",
]


def smooth_df(df, col, sigma=12):
    smoothed_data = gaussian_filter(df[col].to_numpy(), sigma)
    new_col = f"{col}_smoothed"
    df = df.with_columns(pl.lit(smoothed_data).alias(new_col))
    return df


def compute_diffs_single_node(df, node, absolute=True):
    ndf = df.loc[df["node"] == node]
    node_xdiffs = np.diff(ndf["x"].values)
    if absolute:
        node_xdiffs = np.abs(node_xdiffs)
    node_ydiffs = np.diff(ndf["y"].values)
    if absolute:
        node_ydiffs = np.abs(node_ydiffs)
    frame_index = np.arange(1, len(node_xdiffs) + 1)
    ydf = pd.DataFrame(
        {"diff": node_ydiffs, "node": node, "axis": "y", "frame": frame_index}
    )
    xdf = pd.DataFrame(
        {"diff": node_xdiffs, "node": node, "axis": "x", "frame": frame_index}
    )
    return pd.concat([ydf, xdf])


def assign_dt_to_diff_df(diff_df, frame_zero_time, fps=1):
    frames = diff_df["frame"].to_pandas()
    frame_timedelta = frames * (1 / fps)
    frames_timedelta = pd.to_timedelta(frame_timedelta, unit="s")

    frames_dt = frame_zero_time + frames_timedelta
    diff_df = diff_df.with_columns(datetime=pl.lit(frames_dt.values))
    return diff_df


def load_probe_diff_df(subject, rec, filtered=False, method="sum", nodes=nodes):
    h = acr.io.load_hypno(subject, rec)
    if filtered:
        path = f"/Volumes/ceph-tononi/acr_archived_subjects/dlc_home_cage_model/inference_results/{subject}--{rec}DLC_Resnet50_home_cagesJun3shuffle0_snapshot_200_filtered.csv"
    else:
        path = f"/Volumes/ceph-tononi/acr_archived_subjects/dlc_home_cage_model/inference_results/{subject}--{rec}DLC_Resnet50_home_cagesJun3shuffle0_snapshot_200.csv"
    df = pd.read_csv(path)
    cdf = acr.nor.clean_full_df(df, nodes=nodes)
    node_diffs = []
    for node in nodes:
        node_diffs.append(compute_diffs_single_node(cdf, node, absolute=True))
    _diff = pd.concat(node_diffs)
    _diff = _diff.reset_index(drop=True)
    difpl = pl.from_pandas(_diff)
    if method == "mean":
        diff = difpl.group_by(["node", "frame"]).agg(pl.col("diff").mean())
    elif method == "sum":
        diff = difpl.group_by(["node", "frame"]).agg(pl.col("diff").sum())

    diff = diff.sort(["node", "frame"])
    # assign dt to diffs
    dt_start = acr.info_pipeline.subject_info_section(subject, "rec_times")

    rec_start = pd.Timestamp(dt_start[rec]["start"])
    diffs = assign_dt_to_diff_df(diff, frame_zero_time=rec_start, fps=1)
    return diffs, h


def load_probe_points_df(subject, rec, filtered=False):
    if filtered:
        path = f"/Volumes/ceph-tononi/acr_archived_subjects/dlc_home_cage_model/inference_results/{subject}--{rec}DLC_Resnet50_home_cagesJun3shuffle0_snapshot_200_filtered.csv"
    else:
        path = f"/Volumes/ceph-tononi/acr_archived_subjects/dlc_home_cage_model/inference_results/{subject}--{rec}DLC_Resnet50_home_cagesJun3shuffle0_snapshot_200.csv"
    cdf = load_points_df(path)
    dt_start = acr.info_pipeline.subject_info_section(subject, "rec_times")
    rec_start = pd.Timestamp(dt_start[rec]["start"])
    cdf = assign_dt_to_diff_df(cdf, frame_zero_time=rec_start, fps=1)
    return cdf


def threshold_sleep_on_diff_df(df, thresh, col="speed"):
    df = df.with_columns(state=pl.lit("wake"))
    df = df.with_columns(
        state=pl.when(pl.col(col) < thresh)
        .then(pl.lit("sleep"))
        .otherwise(pl.col("state"))
    )
    sleep_prop = df.group_by(["state"]).agg(pl.col(col).count()).sort(["state"])[col][
        0
    ] / len(df)
    return sleep_prop


def load_ucms_rec_info():
    rec_info_path = "/Volumes/em_storage/Electron_Microscopy/UCMS/rec_info.yaml"
    with open(rec_info_path, "r") as file:
        rec_info = yaml.load(file, Loader=yaml.FullLoader)
    return rec_info


rec_name_map = {
    "STRESS5": "STRESS_5_6_7_8",
    "STRESS6": "STRESS_5_6_7_8",
    "STRESS7": "STRESS_5_6_7_8",
    "STRESS8": "STRESS_5_6_7_8",
    "STRESS9": "STRESS_9_10_11_12",
    "STRESS10": "STRESS_9_10_11_12",
    "STRESS11": "STRESS_9_10_11_12",
    "STRESS12": "STRESS_9_10_11_12",
    "STRESS13": "STRESS_13_14",
    "STRESS14": "STRESS_13_14",
}


def load_points_df(path, nodes=nodes):
    df = pd.read_csv(path)
    cdf = acr.nor.clean_full_df(df, nodes=nodes)
    cdf["likelihood"] = cdf["likelihood"].astype(float)
    cdf = pl.from_pandas(cdf)
    return cdf


def compute_full_velocity_df(points_df, cm_per_px=1):
    node_dfs = []
    for node in points_df["node"].unique():
        node_df = points_df.filter(pl.col("node") == node)
        node_dfs.append(compute_velocity_single_node(node_df, cm_per_px))
    full_df = pl.concat(node_dfs)
    return full_df.sort(["node", "datetime"])


def load_ucms_hypno(subject, rec, chunk_dur=21600, num_chunks=4):
    hypno_root = (
        "/Volumes/em_storage/Electron_Microscopy/UCMS/Sleepscoring_data_2/hypnograms"
    )
    # _sub = subject.split('_')
    # _subject = ''.join(_sub)
    # TODO - not really good practice here... subjects should always be named in the same way!
    # TODO - also I had to rename any hypnogram containing 'baseline' to 'Baseline' so that the capitalization is consistent
    hyp = pd.DataFrame()
    for i in range(num_chunks):
        chunk = i + 1
        added_dur = i * chunk_dur
        path = f"{hypno_root}/hypno_{subject}_{rec}_chunk{chunk}_JM.txt"
        chunk_hyp = kde.hypno.hypno.load_hypno_file(path, st=None, dt=False)
        chunk_hyp["start_time"] = chunk_hyp["start_time"] + added_dur
        chunk_hyp["end_time"] = chunk_hyp["end_time"] + added_dur
        hyp = pd.concat([hyp, chunk_hyp])
    rec_info = load_ucms_rec_info()
    rec_block_name = f"{rec_name_map[subject]}-{rec}"
    rec_start_time = pd.Timestamp(rec_info[rec_block_name]["start_time"])
    hyp = kde.hypno.hypno.to_datetime(hyp, rec_start_time)
    return hyp


ucms_video_starts = {}
ucms_video_starts["STRESS_10_baseline"] = pd.Timestamp("2024-10-04 09:02:59")
ucms_video_starts["STRESS_20_baseline"] = pd.Timestamp("2025-05-24 09:02:50")
ucms_video_starts["STRESS_22_baseline"] = pd.Timestamp("2025-05-24 09:02:54")
ucms_video_starts["STRESS_12_baseline"] = pd.Timestamp("2024-10-04 09:03:01")
ucms_video_starts["STRESS_8_baseline"] = pd.Timestamp("2024-09-30 09:01:00")


def load_ucms_points_df(subject, rec, filtered=False):
    if filtered:
        path = f"/Volumes/ceph-tononi/acr_archived_subjects/dlc_home_cage_model/inference_results/{subject}_{rec}DLC_Resnet50_home_cagesJun3shuffle0_snapshot_200_filtered.csv"
    else:
        path = f"/Volumes/ceph-tononi/acr_archived_subjects/dlc_home_cage_model/inference_results/{subject}_{rec}DLC_Resnet50_home_cagesJun3shuffle0_snapshot_200.csv"

    df = pd.read_csv(path)
    cdf = acr.nor.clean_full_df(df, nodes=nodes)
    cdf["likelihood"] = cdf["likelihood"].astype(float)
    cdf = pl.from_pandas(cdf)
    rec_start = ucms_video_starts[f"{subject}_{rec}"]
    cdf = assign_dt_to_diff_df(cdf, frame_zero_time=rec_start, fps=1)
    return cdf


def load_ucms_diff_df(subject, rec, filtered=False, method="sum"):
    if filtered:
        path = f"/Volumes/ceph-tononi/acr_archived_subjects/dlc_home_cage_model/inference_results/{subject}_{rec}DLC_Resnet50_home_cagesJun3shuffle0_snapshot_200_filtered.csv"
    else:
        path = f"/Volumes/ceph-tononi/acr_archived_subjects/dlc_home_cage_model/inference_results/{subject}_{rec}DLC_Resnet50_home_cagesJun3shuffle0_snapshot_200.csv"
    df = pd.read_csv(path)
    cdf = acr.nor.clean_full_df(df, nodes=nodes)
    node_diffs = []
    for node in nodes:
        node_diffs.append(compute_diffs_single_node(cdf, node, absolute=True))
    _diff = pd.concat(node_diffs)
    _diff = _diff.reset_index(drop=True)
    difpl = pl.from_pandas(_diff)
    if method == "mean":
        diff = difpl.group_by(["node", "frame"]).agg(pl.col("diff").mean())
    elif method == "sum":
        diff = difpl.group_by(["node", "frame"]).agg(pl.col("diff").sum())

    diff = diff.sort(["node", "frame"])
    # assign dt to diffs
    rec_start = ucms_video_starts[f"{subject}_{rec}"]
    diffs = assign_dt_to_diff_df(diff, frame_zero_time=rec_start, fps=1)
    return diffs


def get_hypno_percent_asleep(
    h,
    sleep_states=[
        "NREM",
        "REM",
        "Transition-to-REM",
        "Transition-to-NREM",
        "Transition-to-Wake",
    ],
):
    return (
        h.keep_states(sleep_states).duration.sum().total_seconds()
        / h.duration.sum().total_seconds()
    )


def get_nor_start_from_csv(path):
    s1 = path.split("DLC_Resnet")[0]
    name = s1.split("/")[-1]
    st = name.split("-")[-1]
    start_date = st.split("_")[0]
    start_time = st.split("_")[1]
    full_dt = pd.Timestamp(f"{start_date}T{start_time}")
    return full_dt


def get_all_nor_paths(
    subject,
    root_dir="/Volumes/ceph-tononi/acr_archived_subjects/dlc_home_cage_model/inference_results",
    filtered=False,
):
    paths = []
    if filtered:
        for f in os.listdir(root_dir):
            if subject in f and "filtered" in f and ".csv" in f:
                paths.append(os.path.join(root_dir, f))
    else:
        for f in os.listdir(root_dir):
            if subject in f and "filtered" not in f and ".csv" in f:
                paths.append(os.path.join(root_dir, f))

    return paths


def load_nor_single_file(path):
    nodes = [
        "nose",
        "mid_ears",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "mouse_center",
        "left_hip",
        "right_hip",
        "tail_base",
    ]

    cdf = load_points_df(path, nodes=nodes)
    start_time = get_nor_start_from_csv(path)
    vel_df = compute_full_velocity_df(cdf, start_time)
    return vel_df


def load_nor_concat_positions(
    subject,
    root_dir="/Volumes/ceph-tononi/acr_archived_subjects/dlc_home_cage_model/inference_results",
):
    paths = get_all_nor_paths(subject, root_dir)
    position_dfs = []
    for p in paths:
        point_df = load_points_df(p)
        point_df = point_df.sort(["node", "frame"])
        start_dt = get_nor_start_from_csv(p)
        point_df = assign_dt_to_diff_df(point_df, start_dt, fps=1)
        position_dfs.append(point_df)
    pos_df = pl.concat(position_dfs)
    return pos_df


def load_nor_actigraphy(
    subject,
    root_dir="/Volumes/ceph-tononi/acr_archived_subjects/dlc_home_cage_model/inference_results",
):
    pos_df = load_nor_concat_positions(subject, root_dir)
    vel_df = compute_full_velocity_df(pos_df)
    return vel_df


def compute_velocity_single_node(node_df, cm_per_px=1):
    node = node_df["node"].unique()[0]
    dt = node_df["datetime"].to_numpy()
    dt = dt[1:]
    x = node_df["x"].to_numpy()
    y = node_df["y"].to_numpy()
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    disp = np.hypot(dx, dy)  # pixels / frame
    speed = disp * cm_per_px
    med = np.median(speed)
    mad = scipy.stats.median_abs_deviation(speed, scale="normal")  # ≈1.4826*MAD
    robust_z = (speed - med) / mad
    zpl = pl.DataFrame(
        {"node": node, "datetime": dt, "robust_z": robust_z, "speed": speed}
    )
    return zpl


def rob_z_col(df, col="speed"):
    med = np.median(df[col])
    mad = scipy.stats.median_abs_deviation(df[col], scale="normal")  # ≈1.4826*MAD
    robust_z = (df[col] - med) / mad
    return df.with_columns(pl.lit(robust_z).alias(f"{col}_robust_z"))


def load_sd_recovery_info(velocity, subject, recovery_duration="4h", buffer="0min"):
    box_time, acq_day = acr.nor.get_sub_timing(subject)
    acqday_start = pd.Timestamp(f"{acq_day} {box_time}")
    min_sleep_start = acqday_start + pd.Timedelta("55min")
    sleep_period_border = velocity.filter(pl.col("datetime") > min_sleep_start)[
        "datetime"
    ].min()
    recovery_start = pd.Timestamp(sleep_period_border + pd.Timedelta(buffer))
    recovery_end = recovery_start + pd.Timedelta(recovery_duration)
    return acqday_start, recovery_start, recovery_end


def load_sleep_recovery_info(velocity, subject, recovery_duration="1h", buffer="0min"):
    box_time, acq_day = acr.nor.get_sub_timing(subject)
    acqday_start = pd.Timestamp(f"{acq_day} {box_time}")
    min_sleep_start = acqday_start + pd.Timedelta("5min")
    sleep_period_border = velocity.filter(pl.col("datetime") > min_sleep_start)[
        "datetime"
    ].min()
    recovery_start = pd.Timestamp(sleep_period_border + pd.Timedelta(buffer))
    recovery_end = recovery_start + pd.Timedelta(recovery_duration)
    return acqday_start, recovery_start, recovery_end


def label_vel_df_with_conds(vel, subject, recovery_duration="4h", buffer="0min"):
    acq, recovery_start, recovery_end = acr.dlc.load_sd_recovery_info(
        vel, subject, recovery_duration, buffer
    )
    bl_start = recovery_start - pd.Timedelta("24h")
    bl_end = recovery_end - pd.Timedelta("24h")
    cond_hyp = {}
    cond_hyp["bl"] = pd.DataFrame(
        {
            "start_time": bl_start,
            "end_time": bl_end,
            "duration": bl_end - bl_start,
            "state": "NREM",
        },
        index=[0],
    )
    cond_hyp["recovery"] = pd.DataFrame(
        {
            "start_time": recovery_start,
            "end_time": recovery_end,
            "duration": recovery_end - recovery_start,
            "state": "NREM",
        },
        index=[0],
    )
    vel = acr.hypnogram_utils.label_df_with_hypno_conditions(vel, cond_hyp)
    return vel
