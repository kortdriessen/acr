# I am using this file to store old ON-OFF detection code, which was using Tom's pipeline (i.e. on_off_detection repository @ https://github.com/CSC-UW/on_off_detection.git)
# I am not using this code anymore, but I am keeping it here for reference.


# ----------------------------------------- ON-OFF DETECTION ----------------------------------------------------------------------------------------
def load_hypno_for_ood(subject, sort_id, state="NREM"):
    recs, st, durs = acr.units.get_time_info(subject, sort_id)
    hyps = {}
    running_dur = 0
    for i, rec in enumerate(recs):
        hyp = acr.io.load_hypno(subject, rec)
        if hyp is None:
            hyps[rec] = hyp
            running_dur += durs[i - 1]
            continue
        hyp = hyp.as_float()
        if i == 0:
            hyps[rec] = hyp
            continue
        running_dur += durs[i - 1]
        hyp["start_time"] = hyp.start_time.values + running_dur
        hyp["end_time"] = hyp.end_time.values + running_dur
        hyps[rec] = hyp
    df = pd.concat(hyps)
    df = df.reset_index(drop=True)
    df = standard_hypno_corrections(df)
    df_nrem = df.loc[df.state == state]
    return df_nrem.reset_index(drop=True)


def load_spike_trains_for_ood(subject, sort_id, exlcude_bad_units=True):
    path = acr.units.sorting_path(subject, sort_id)
    sort_extractor, info = ku.io.load_sorting_extractor(path, drop_noise=True)
    cluster_ids = sort_extractor.unit_ids
    if exlcude_bad_units == True:
        cids_2_ex = get_units_to_exclude(subject, sort_id)
        cluster_ids = [x for x in cluster_ids if x not in cids_2_ex]
    trains = []
    for cluster in cluster_ids:  # gets list of spike trains for each unit
        trn = sort_extractor.get_unit_spike_train(cluster)
        trn = (
            trn / 24414.0625
        )  # gives the spike times in seconds instead of sample numbers
        trains.append(trn)
    return trains, info, cluster_ids


def run_ood(trains, clust_ids, hypno, tmax=None, save=False):
    ood.HMMEM_PARAMS["init_state_estimate_method"] = "conservative"
    if tmax is None:
        tmax = int(hypno.end_time.max() + 1)
    GLOBAL_ON_OFF_DETECTION_PARAMS = {
        "binsize": 0.01,  # (s) (Discrete algorithm)
        "history_window_nbins": 3,  # Size of history window IN BINS
        "n_iter_EM": 200,  # Number of iterations for EM
        "n_iter_newton_ralphson": 100,
        "init_A": np.array(
            [[0.1, 0.9], [0.01, 0.99]]
        ),  # Initial transition probability matrix
        "init_mu": None,  # ~ OFF rate. Fitted to data if None
        "init_alphaa": None,  # ~ difference between ON and OFF rate. Fitted to data if None
        "init_betaa": None,  # ~ Weight of recent history firing rate. Fitted to data if None,
        "gap_threshold": 0.05,  # Merge active states separated by less than gap_threhsold
    }

    mod = OnOffModel(
        trains,
        tmax,
        clust_ids,
        method="hmmem",
        params=GLOBAL_ON_OFF_DETECTION_PARAMS,
        bouts_df=hypno,
    )
    oodf = mod.run()
    return oodf


def assign_recordings_to_oodf(oodf, recordings, durations):
    prev_split = 0
    for r, d in zip(recordings, durations):
        border1 = prev_split
        border2 = prev_split + d
        oodf.loc[
            np.logical_and(oodf.end_time > border1, oodf.end_time <= border2),
            "recording",
        ] = r
        prev_split += d
    total_duration = np.sum(durations)
    oodf.loc[oodf.end_time > total_duration, "recording"] = recordings[-1]
    return oodf


def assign_datetimes_to_oodf(oodf, recordings, start_times):
    dti_starts = pd.DatetimeIndex([])
    dti_ends = pd.DatetimeIndex([])
    for r, s in zip(recordings, start_times):
        rec_df = oodf.loc[oodf.recording == r]
        if rec_df.empty:
            print(f"Nothing for {r}, skipping")
            continue
        starts = rec_df.start_time.values
        ends = rec_df.end_time.values

        start_times_rel = starts - starts[0]
        end_times_rel = ends - ends[0]
        start_timedeltas = pd.to_timedelta(start_times_rel, unit="s")
        end_timedeltas = pd.to_timedelta(end_times_rel, unit="s")
        start_datetimes = s + start_timedeltas
        end_datetimes = s + end_timedeltas
        dti_starts = dti_starts.append(start_datetimes)
        dti_ends = dti_ends.append(end_datetimes)
    oodf["start_datetime"] = dti_starts
    oodf["end_datetime"] = dti_ends
    return oodf


def process_oodf(oodf, subject, sort_id, load=False):
    if load:
        oodf = acr.units.load_oodf(subject=subject, sort_id=sort_id)
    recs, starts, durations = acr.units.get_time_info(subject, sort_id)
    oodf = assign_recordings_to_oodf(oodf, recs, durations)
    oodf = assign_datetimes_to_oodf(oodf, recs, starts)
    oodf["probe"] = sort_id.split("-")[-1]
    if load:
        acr.units.save_oodf(subject, sort_id, oodf)
    else:
        return oodf


def save_oodf(subject, sort_id, oodf):
    # First we make sure the on-off_dataframes folder exists
    sort_root = f"/Volumes/opto_loc/Data/{subject}/sorting_data/"
    if "on-off_dataframes" not in os.listdir(sort_root):
        os.mkdir(sort_root + "on-off_dataframes")
    # Then we save the dataframe
    path = f"/Volumes/opto_loc/Data/{subject}/sorting_data/on-off_dataframes/{sort_id}.json"
    oodf.to_json(path, orient="records", lines=True)


def load_oodf(subject, sort_id=[]):
    """_summary_

    Args:
        subject (str): subject name
        sort_id (list, optional): list of sort_id's Defaults to [].
    """
    if type(sort_id) == str:
        sort_id = [sort_id]
    path = f"/Volumes/opto_loc/Data/{subject}/sorting_data/on-off_dataframes/"
    sdfs = []
    for sid in sort_id:
        key = sid + ".json"
        oodf = pd.read_json(path + key, lines=True)
        # oodf.drop(columns=["start_datetime", "end_datetime"], inplace=True)
        oodf = process_oodf(oodf, subject, sid)
        sdfs.append(oodf)
    oodfs = pd.concat(sdfs)
    return oodfs


def load_oodf_hack(subject, sort_id=[]):
    """_summary_

    Args:
        sort_id (list, optional): list of sort_id's Defaults to [].
    """
    if type(sort_id) == str:
        sort_id = [sort_id]
    sdfs = []
    for sid in sort_id:
        path = sid + ".json"
        oodf = pd.read_json(path, lines=True)
        # oodf.drop(columns=['start_datetime', 'end_datetime'], inplace=True)
        oodf = acr.units.process_oodf(oodf, subject, sid)
        sdfs.append(oodf)
    oodfs = pd.concat(sdfs)
    return oodfs
