def load_hypnograms(subject_info, subtract_sd=False):
    h = {}
    subject = subject_info["subject"]
    root = (
        "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/"
        + subject
        + "/"
        + "hypnograms-"
        + subject
        + "/"
    )
    for key in subject_info["hypnos"]:
        bp = achr_path(subject, key)
        block = tdt.read_block(bp, store="EEGr", t1=0, t2=1)
        start_time = pd.to_datetime(block.info.start_date)

        if subtract_sd is not False:
            t = pd.to_timedelta(subtract_sd, "h")
            start_time = start_time + t

        path = root + "hypno_" + key + ".txt"
        h[key] = kh.load_hypno_file(path, st=start_time)
    return h


def load_hypno_set(
    subject,
    condition,
    scoring_start_time,
    hypnograms_yaml_file="/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/acr-hypno-paths.yaml",
):
    """Do not use this if hynograms are not exactly 2 hours long"""

    with open(hypnograms_yaml_file) as fp:
        yaml_data = yaml.safe_load(fp)

    root = Path(yaml_data[subject]["hypno-root"])
    hypnogram_fnames = yaml_data[subject][condition]
    hypnogram_paths = [root / (fname + ".txt") for fname in hypnogram_fnames]

    hypnogram_start_times = pd.date_range(
        start=scoring_start_time, periods=len(hypnogram_paths), freq="7200S"
    )
    hypnograms = [
        hp.load_visbrain_hypnogram(path).as_datetime(start_time)
        for path, start_time in zip(hypnogram_paths, hypnogram_start_times)
    ]

    return pd.concat(hypnograms).reset_index(drop=True)


def load_complete_dataset_from_blocks(info_dict, store, chans, start_at=0, time=4):
    """
    start_at --> number of hours into the recording to start
    time --> number of hours total to load
    """
    data_dict = {}
    key_list = info_dict["complete_key_list"]
    path_dict = get_paths(info_dict["subject"], key_list)

    for key in key_list:
        if key.find("bl") != -1:
            start = 0
            stop = 43200
        else:
            start = start_at * 3600
            stop = start + (time * 3600)
        data_dict[key] = kd.get_data(
            path_dict[key], store=store, t1=start, t2=stop, channel=chans, sev=True
        )
    else:
        data_dict["x-time"] = str(time) + "-Hour"
        data_dict["sub"] = info_dict["subject"]
        data_dict["dtype"] = "EEG-Data" if store == "EEGr" else "LFP-Data"
        if store == "EMG_" or store == "EMGr":
            data_dict["dtype"] = "EMG-Data"
            for key in key_list:
                data_dict[key] = data_dict[key].sel(channel=1)
            return data_dict
        else:
            return data_dict


def save_dataset(ds, name, key_list=None, folder=None):
    """saves each component of an experimental
    dataset dictionary (i.e. xr.arrays of the raw data and of the spectrograms),
    as its own separate .nc file. All can be loaded back in as an experimental dataset dictionary
    using fetch_xset
    """
    keys = kd.get_key_list(ds) if key_list == None else key_list
    analysis_root = (
        "/Volumes/opto_loc/Data/ACHR_PROJECT_MATERIALS/analysis_data/" + folder + "/"
        if folder is not None
        else "/Volumes/opto_loc/Data/ACHR_PROJECT_MATERIALS/analysis_data/"
    )

    for key in keys:
        if type(ds[key]) == str:
            pass
        path = analysis_root + (name + "_" + key + ".nc")
        ds[key].to_netcdf(path)


def save_hypnoset(ds, name, key_list=None, folder=None):
    keys = kd.get_key_list(ds) if key_list == None else key_list
    analysis_root = (
        "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/analysis_data_complete/"
        + folder
        + "/"
        if folder is not None
        else "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/analysis_data/"
    )
    for key in keys:
        path = analysis_root + (name + "_" + key + ".tsv")
        ds[key].write(path)


def load_saved_dataset(subject_info, set_name, folder=None, spg=False):
    """
    Used to load a dataset, can calculate hypnogram automatically also
    -------------------------------------------------------------------------------------
    """
    data_set = {}
    subject = subject_info["subject"]
    path_root = (
        "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/analysis_data/" + folder + "/"
        if folder is not None
        else "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/analysis_data/"
    )

    if set_name.find("h") != -1:
        for key in subject_info["complete_key_list"]:
            path = path_root + set_name + "_" + key + ".tsv"
            data_set[key] = hp.load_datetime_hypnogram(path)
        data_set["name"] = set_name

    else:
        for key in subject_info["complete_key_list"]:
            path = path_root + set_name + "_" + key + ".nc"
            data_set[key] = xr.load_dataarray(path)
        data_set["name"] = set_name
    if spg == True:
        spg_set = kd.get_spg_from_dataset(data_set)
        spg_set["sub"] = subject_info["subject"]
        spg_set["dtype"] = "EEG-Data" if set_name.find("de") != -1 else "LFP-Data"
        return data_set, spg_set
    else:
        return data_set

def check_hypno(subject, condition):
    hypnograms_yaml_file = (
        "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/acr-hypno-paths.yaml"
    )
    with open(hypnograms_yaml_file) as fp:
        yaml_data = yaml.safe_load(fp)
    root = Path(yaml_data[subject]["hypno-root"])
    if condition in yaml_data[subject].keys():
        return True
    else:
        return False


def _load_hypno(subject, condition, start_time):
    "critical - this only works for loading continuous hypnograms"
    # TODO: make this more flexible for loading discontinuous hypnograms from the same experiment
    hypnograms_yaml_file = (
        "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/acr-hypno-paths.yaml"
    )
    with open(hypnograms_yaml_file) as fp:
        yaml_data = yaml.safe_load(fp)
    root = Path(yaml_data[subject]["hypno-root"])
    if check_hypno(subject, condition):
        hypnogram_fnames = yaml_data[subject][condition]
        hypnogram_paths = [root / (fname + ".txt") for fname in hypnogram_fnames]
        hypnos = range(1, len(hypnogram_paths) + 1)
        hyp_list = []
        for num, hp in zip(hypnos, hypnogram_paths):
            if num == 1:
                h = kh.load_hypno_file(hp, start_time)
                hyp_list.append(h)
            if num > 1:
                h = kh.load_hypno_file(hp, h.end_time.max())
                hyp_list.append(h)
        hh = pd.concat(hyp_list).reset_index(drop=True)
        return DatetimeHypnogram(hh)
    else:
        return None
    
def xarray_to_pandas(xarr, name=None):
    if name is None:
        return xarr.to_dataframe()
    else:
        return xarr.to_dataframe(name=name)


def redo_timdelta(df):
    start = df.index.get_level_values(0)[0]
    df.drop(columns=["timedelta"], inplace=True)
    df["timedelta"] = df.index.get_level_values(0) - start
    return df


def load_hypno_dep(info, data, data_tag):
    """
    info --> subject info dictionary
    data --> data dictionary
    Note: The minimum datetime of the data will be taken as the start time for the hypnogram.
    """
    hyp = {}
    subject = info["subject"]

    for cond in info["complete_key_list"]:
        if check_hypno(subject, cond):
            cond_data = cond + "-" + data_tag
            for d in data.keys():
                if cond_data in d:
                    start_time = data[d].datetime.values.min()
                    break
            hyp[cond] = _load_hypno(subject, cond, start_time)
        else:
            print("No hypnogram for condition: ", cond)
    return hyp


def add_hypnograms_to_dataset(dataset, hypno_set):
    for key in hypno_set.keys():
        for exp in dataset.keys():
            if key in exp:
                dataset[exp] = kh.add_states(dataset[exp], hypno_set[key])
    return dataset


def get_spectral(data):
    spg = kx.spectral.get_spg_from_dataset(data)
    bp = kx.spectral.get_bp_from_dataset(spg)
    # for key in list(spg.keys()):
    # bp[key] = kx.spectral.get_bp_set(spg[key], bp_def)
    return spg, bp

def load_subject_data(info, stores=["EEGr", "LFP_"]):
    data = {}
    for exp in info["complete_key_list"]:
        t1 = info["load_times"][exp][0]
        t2 = t1 + (info["load_times"][exp][1] * 3600)
        for store in stores:
            chans = info["channels"][store]
            data[exp + "-" + store] = kx.io.get_data(
                info["paths"][exp], store, t1=t1, t2=t2, channel=chans
            )
    return data




def dataset_to_pandas(dataset, index=["datetime", "channel"], name=None):
    for key in list(dataset.keys()):
        dataset[key] = xarray_to_pandas(dataset[key], name=name)
        dataset[key] = redo_timdelta(dataset[key])
        dataset[key]["condition"] = key
        dataset[key] = dataset[key].reset_index()
        dataset[key] = dataset[key].set_index(index)
        dataset[key] = ecdata(dataset[key])
    return dataset


def load_saved_dataset(si, type, data_tags):
    """
    data_tage --> e.g. '-EEGr'
    type --> e.g. '-spg'
    """
    # TODO: should probably import the data as my pandas ecdata class

    cond_list = si["complete_key_list"]
    sub = si["subject"]
    path_root = (
        "/Volumes/opto_loc/Data/ACR_PROJECT_MATERIALS/" + sub + "/" + "analysis-data/"
    )
    ds = {}
    if type == "-hypno":
        for cond in cond_list:
            try:
                path = path_root + cond + type + ".parquet"
                ds[cond] = pd.read_parquet(path)
                ds[cond] = DatetimeHypnogram(ds[cond])
            except:
                print("No hypnogram for condition: ", cond)
        return ds

    else:
        for cond in cond_list:
            for tag in data_tags:
                key = cond + tag
                path = path_root + key + type + ".parquet"
                ds[key] = pd.read_parquet(path)
                ds[key] = ecdata(ds[key])
        return ds


def ss_times(sub, exp, print_=False):
    # TODO: this may be deprecated, and may not be needed, I think it was only used for ACR_9?

    # Load the relevant times:
    def acr_get_times(sub, exp):
        block_path = "/Volumes/opto_loc/Data/" + sub + "/" + sub + "-" + exp
        ep = tdt.read_block(block_path, t1=0, t2=0, evtype=["epocs"])
        times = {}
        times["bl_sleep_start"] = ep.epocs.Bttn.onset[0]
        times["stim_on"] = ep.epocs.Wdr_.onset[-1]
        times["stim_off"] = ep.epocs.Wdr_.offset[-1]
        dt_start = pd.to_datetime(ep.info.start_date)

        # This get us the datetime values of the stims for later use:
        on_sec = pd.to_timedelta(times["stim_on"], unit="s")
        off_sec = pd.to_timedelta(times["stim_off"], unit="s")
        times["stim_on_dt"] = dt_start + on_sec
        times["stim_off_dt"] = dt_start + off_sec
        return times

    times = acr_get_times(sub, exp)

    # Start time for scoring is 30 seconds before the button signal was given to inidicate the start of bl peak period.
    start1 = times["bl_sleep_start"] - 30
    end1 = start1 + 7200

    # End time for the second scoring file is when the stim/laser signal is turned off.
    start2 = end1
    end2 = times["stim_off"]
    if print_:
        print("FILE #1"), print(start1), print(end1)
        print("FILE #2"), print(start2), print(end2)
    print("Done loading times")
    return times

def pulse_cal_calculation(df, pons, poffs, interval=12):
    """get the total spike counts during pulse-ON, and during pulse-OFF, for each probe

    Parameters
    ----------
    df : polars dataframe
        spike dataframe
    pons : np.array
        pulse onsets
    poffs : np.array
        pulse offsets
    interval : int, optional
        number of pulses in each pulse train, by default 12

    Returns
    -------
    on_spike_rate, off_spike_rate : polars dataframe
        spike rates for each cluster in each probe during pulse-ON and pulse-OFF
    """

    # iterate through each pulse train, calculate the spike rate during pulse-ON and pulse-OFF, and express it as a ratio of the baseline spike rate
    trn_number = 0
    trains = np.arange(0, len(pons), 12)
    on_spike_counts = pl.DataFrame()
    off_spike_counts = pl.DataFrame()

    for i in trains:
        pulse_ons = pons[i : i + 12]
        pulse_offs = poffs[i : i + 12]

        off_interval = pulse_ons[1] - pulse_offs[0]
        off_duration = off_interval / np.timedelta64(1, "s")

        for onset, offset in zip(pulse_ons, pulse_offs):
            on_spike_count = df.ts(onset, offset).groupby(["probe"]).agg(pl.count())
            on_duration = (offset - onset) / np.timedelta64(1, "s")
            on_spike_count = on_spike_count.with_columns(duration=pl.lit(on_duration))
            on_spike_count = on_spike_count.with_columns(
                train_number=pl.lit(trn_number)
            )
            if len(on_spike_count) < 2:
                on_spike_count = add_zero_count(on_spike_count)
            on_spike_counts = pl.concat([on_spike_counts, on_spike_count])

            off_spike_count = (
                df.ts(offset, offset + off_interval).groupby(["probe"]).count()
            )
            off_spike_count = off_spike_count.with_columns(
                duration=pl.lit(off_duration)
            )
            off_spike_count = off_spike_count.with_columns(
                train_number=pl.lit(trn_number)
            )
            if len(off_spike_count) < 2:
                off_spike_count = add_zero_count(off_spike_count)
            off_spike_counts = pl.concat([off_spike_counts, off_spike_count])
        trn_number += 1

    return on_spike_counts.to_pandas(), off_spike_counts.to_pandas()
