def load_oodf(subject, rec, probe, flavor='single-chan'):
    oodf_path = f'{raw_data_root}mua_data/{subject}/oodfs/{rec}--{probe}__{flavor}.parquet'
    oodf = pl.read_parquet(oodf_path)
    oodf = oodf.with_columns(rec=pl.lit(rec))
    oodf = oodf.with_columns(subject=pl.lit(subject))
    oodf = oodf.with_columns(probe=pl.lit(probe))
    return oodf

def load_oodf_full_exp(subject, exp, flavor='single-chan', probes=['NNXr', 'NNXo']):
    recs, _sts, _tms = acr.units.get_time_info(subject, f'{exp}-{probes[0]}')
    oodfs = []
    for rec in recs:
        for probe in probes:
            oodf = load_oodf(subject, rec, probe, flavor)
            oodfs.append(oodf)
    full_oodf = pl.concat(oodfs)
    return full_oodf

def compute_and_save_oodf(subject, rec, probe, repeat=False, flavor='single-chan'):
    if flavor == 'single-chan':
        by = 'channel'
    elif flavor == 'whole-probe':
        by='probe'
    else:
        raise ValueError('flavor must be single-chan or whole-probe')
    already_done = check_for_oodf(subject, rec, probe, flavor)
    if already_done and not repeat:
        return
    mua_df = acr.mua.load_detected_mua_spikes(subject, rec, probe, dt=True)
    oodf = calculate_oodf(mua_df, off_min_dur=.05, by=by)
    exp = acr.info_pipeline.get_exp_from_rec(subject, rec)
    hd = acr.hypnogram_utils.create_acr_hyp_dict(subject, exp, duration='3600s')
    oodf = label_oodf_with_hypno_conditions(oodf, hd)
    save_oodf(oodf, subject, rec, probe, flavor)
    return

def save_all_oodfs(subject, repeat=False, flavor='single-chan'):
    peaks_path = f'{raw_data_root}mua_data/{subject}/si_peak_dfs'
    for f in os.listdir(peaks_path):
        if 'PEAKS' not in f:
            continue
        rec = f.split('--')[2]
        probe = f.split('--')[3]
        probe = probe.split('.')[0]
        print(f'Computing and saving oodf for {subject}--{rec}--{probe}')
        compute_and_save_oodf(subject, rec, probe, repeat=repeat, flavor=flavor)
    return

def save_oodf(oodf, subject, rec, probe, name):
    oodf_path = f'{raw_data_root}mua_data/{subject}/oodfs'
    if os.path.exists(oodf_path) == False:
        os.mkdir(oodf_path)
    oodf.write_parquet(f'{oodf_path}/{rec}--{probe}__{name}.parquet')
    
def check_for_oodf(subject, rec, probe, name):
    oodf_path = f'{raw_data_root}mua_data/{subject}/oodfs'
    if os.path.exists(oodf_path) == False:
        return False
    if f'{rec}--{probe}__{name}.parquet' in os.listdir(oodf_path):
        return True
    else:
        return False
    
    
    
### MAY ACTUALLY WANT SOME OF THIS STUFF, FROM 09/05/2024

def get_pure_mua_times_by_chan(df):
    times = {}
    for probe in df['probe'].unique():
        times[probe] = {}
        for chan in df['channel'].unique():
            raw_times = df.prb(probe).ch(chan)['datetime'].to_pandas().values
            times[probe][str(chan)] = raw_times
    return times

def get_pure_mua_times_by_probe(df):
    times = {}
    for probe in df['probe'].unique():
        raw_times = df.prb(probe)['datetime'].to_pandas().values
        times[probe] = raw_times
    return times

def calculate_oodf(df, off_min_dur=0.05, by='channel'):
    if by == 'channel':
        return oodf_by_chan(df, off_min_dur)
    elif by == 'probe':
        return oodf_by_probe(df, off_min_dur)


def oodf_by_chan(df, off_min_dur=.05):
    all_oodfs = []
    pure_mua_times = get_pure_mua_times_by_chan(df)
    for probe in pure_mua_times.keys():
        for chan in pure_mua_times[probe].keys():
            chan_times = pure_mua_times[probe][chan]
            diffs = np.diff(chan_times)
            diffs = diffs.astype(float)
            msk = np.ma.masked_where(diffs >= (off_min_dur*1e9), diffs)
            start_times_mask = msk.mask
            start_times_mask = np.append(start_times_mask, False)
            start_times_indices = np.where(start_times_mask == True)[0]
            end_times_indices = start_times_indices + 1
            start_times = chan_times[start_times_indices]
            end_times = chan_times[end_times_indices]
            off_det = pd.DataFrame(
                {
                    "start_datetime": start_times,
                    "end_datetime": end_times,
                    "status": "off",
                    "probe": probe,
                    'channel': int(chan),
                }
            )
            # gets us the on-detection information
            on_starts = []
            on_ends = []
            for i in np.arange(0, len(start_times) + 1):
                if i == 0:
                    on_starts.append(chan_times[0])
                    on_ends.append(start_times[i])
                elif i == len(start_times):
                    on_starts.append(end_times[i - 1])
                    on_ends.append(chan_times[-1])
                else:
                    on_starts.append(end_times[i - 1])
                    on_ends.append(start_times[i])
            on_starts = np.array(on_starts)
            on_ends = np.array(on_ends)
            on_det = pd.DataFrame(
                {
                    "start_datetime": on_starts,
                    "end_datetime": on_ends,
                    "status": "on",
                    "probe": probe,
                    "channel": int(chan),
                }
            )
            oodf = (
                pd.concat([on_det, off_det])
                .sort_values(by="start_datetime")
                .reset_index(drop=True)
            )

            durs = oodf['end_datetime'] - oodf['start_datetime']
            durs_secs = durs.values.astype(float)/1e9
            oodf['duration'] = durs_secs
            all_oodfs.append(oodf)
    full_final_oodf = pd.concat(all_oodfs)
    return pl.DataFrame(full_final_oodf)

def oodf_by_probe(df, off_min_dur=.05):
    all_oodfs = []
    pure_mua_times = get_pure_mua_times_by_probe(df)
    for probe in pure_mua_times.keys():
        probe_times = pure_mua_times[probe]
        diffs = np.diff(probe_times)
        diffs = diffs.astype(float)
        msk = np.ma.masked_where(diffs >= (off_min_dur*1e9), diffs)
        start_times_mask = msk.mask
        start_times_mask = np.append(start_times_mask, False)
        start_times_indices = np.where(start_times_mask == True)[0]
        end_times_indices = start_times_indices + 1
        start_times = probe_times[start_times_indices]
        end_times = probe_times[end_times_indices]
        off_det = pd.DataFrame(
            {
                "start_datetime": start_times,
                "end_datetime": end_times,
                "status": "off",
                "probe": probe,
            }
        )
        # gets us the on-detection information
        on_starts = []
        on_ends = []
        for i in np.arange(0, len(start_times) + 1):
            if i == 0:
                on_starts.append(probe_times[0])
                on_ends.append(start_times[i])
            elif i == len(start_times):
                on_starts.append(end_times[i - 1])
                on_ends.append(probe_times[-1])
            else:
                on_starts.append(end_times[i - 1])
                on_ends.append(start_times[i])
        on_starts = np.array(on_starts)
        on_ends = np.array(on_ends)
        on_det = pd.DataFrame(
            {
                "start_datetime": on_starts,
                "end_datetime": on_ends,
                "status": "on",
                "probe": probe,
            }
        )
        oodf = (
            pd.concat([on_det, off_det])
            .sort_values(by="start_datetime")
            .reset_index(drop=True)
        )

        durs = oodf['end_datetime'] - oodf['start_datetime']
        durs_secs = durs.values.astype(float)/1e9
        oodf['duration'] = durs_secs
        all_oodfs.append(oodf)
    full_final_oodf = pd.concat(all_oodfs)
    return pl.DataFrame(full_final_oodf)


def compute_and_save_oodf(subject, exp):
    """
    Compute and save the concatenated oodf for a given subject and experiment. Requires that the concat mua is available.
    """
    #if _check_if_oodf_exists(subject, exp):
    #    return
    mua = acr.mua.load_concat_peaks_df(subject, exp)
    hd = acr.hypnogram_utils.create_acr_hyp_dict(subject, exp, duration='3600s')
    oodfs = []
    for probe in mua.prbs():
        for rec in mua['recording'].unique():
            rec_mua = mua.prb(probe).filter(pl.col('recording')==rec)
            oodf = acr.onoffmua.calculate_oodf(rec_mua, by='channel')
            oodf = oodf.with_columns(rec = pl.lit(rec))
            oodf = oodf.with_columns(probe = pl.lit(probe))
            oodf = oodf.with_columns(subject = pl.lit(subject))
            oodfs.append(oodf)
    oodf = pl.concat(oodfs)
    oodf = acr.onoffmua.label_oodf_with_hypno_conditions(oodf, hd)
    _save_concat_oodf(oodf, subject, exp)
    return

def _save_concat_oodf(oodf, subject, exp):
    """
    Save the concatenated oodf for a given subject and experiment.
    """
    path = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/oodfs/{exp}--oodf.parquet'
    oodf.write_parquet(path)
    return

def load_oodf(subject, exp):
    path = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/oodfs/{exp}--oodf.parquet'
    oodf = pl.read_parquet(path)
    return oodf

def _check_if_oodf_exists(subject, exp):
    path = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/oodfs/{exp}--oodf.parquet'
    return os.path.exists(path)

def strictify_oodf_polars_fast(oodf, min_on_duration=0.01):
    s_oodfs = []
    for probe in oodf['probe'].unique():
        for channel in oodf.prb(probe)['channel'].unique():
            oodf_probe_ch = oodf.filter(pl.col('probe')==probe).filter(pl.col('channel')==channel)
            strict_oodf_probe_ch = _strictify_oodf_polars_fast(oodf_probe_ch, min_on_duration)
            s_oodfs.append(strict_oodf_probe_ch)
    return pl.concat(s_oodfs)

def _strictify_oodf_polars_fast(oodf, min_on_duration=0.01):
    # Sort the dataframe
    sorted_df = oodf.sort('start_datetime')
    
    # Create shifted columns for previous and next rows
    result = (
        sorted_df
        .with_columns([
            pl.col('status').shift(1).alias('prev_status'),
            pl.col('status').shift(-1).alias('next_status'),
            pl.col('duration').shift(1).alias('prev_duration'),
            pl.col('duration').shift(-1).alias('next_duration')
        ])
        .with_columns(
            pl.when(
                (pl.col('status') == 'off') &
                (pl.col('prev_status') == 'on') &
                (pl.col('next_status') == 'on') &
                (pl.col('prev_duration') >= min_on_duration) &
                (pl.col('next_duration') >= min_on_duration)
            ).then(True).otherwise(False).alias('keep')
        )
    )

    # Keep all 'on' periods and only the valid 'off' periods
    filtered = result.filter(
        (pl.col('status') == 'on') | 
        ((pl.col('status') == 'off') & pl.col('keep'))
    ).select(oodf.columns)  # Select only original columns

    # Merge consecutive 'on' periods
    merged = (
        filtered
        .with_columns(
            (pl.col('status') != pl.col('status').shift(1)).cumsum().alias('group')
        )
        .groupby('group')
        .agg([
            pl.col('start_datetime').first().alias('start_datetime'),
            pl.col('end_datetime').last().alias('end_datetime'),
            pl.col('status').first().alias('status'),
            pl.col('probe').first().alias('probe'),
            pl.col('channel').first().alias('channel'),
            (pl.col('end_datetime').last() - pl.col('start_datetime').first()).dt.total_seconds().alias('duration'),
            pl.col('condition').first().alias('condition'),
            pl.col('rec').first().alias('rec'),
            pl.col('subject').first().alias('subject')
        ])
        .sort('start_datetime')
        .drop('group')
    )
    durations = (merged['end_datetime'] - merged['start_datetime']).to_pandas().dt.total_seconds().values
    merged = merged.with_columns(duration = pl.lit(durations))
    return merged


#OLD STUFF FROM SPOFF 
#--------------------------------------------------------------------------------------------------------------------------------------

def load_complete_exp_off(subject, exp, probes=['NNXr', 'NNXo'], sort_id=None, structure=None, which='ap', sensible_filters=True, relative=True, rel_method='median', max_only=False):
    if sort_id is None:
        sort_id = f'{exp}-{probes[0]}'
    
    recs, starts, durations = acr.units.get_time_info(subject, sort_id)
    
    off_dfs_by_probe = {}
    for probe in probes:
        off = core.get_offs(subject=subject, prb=probe, structure=structure, which=which, experiment=exp)
        off['probe'] = probe
        sorting_start = pd.Timestamp(starts[0])
        off = assign_datetimes_to_off_df(off, sorting_start)
        off = assign_recordings_to_off_df(off, recs, starts, durations)
        
        off_dfs_by_probe[probe] = off
    offdf = pd.concat(off_dfs_by_probe.values())
    offdf.sort_values('start_datetime', inplace=True)
    if sensible_filters:
        offdf = offdf.loc[(offdf['median_duration']>.04) & (offdf['median_duration']<0.8)]
        offdf = offdf.loc[(offdf['duration']>.04) & (offdf['duration']<0.8)]
    if relative:
        offdf = make_odf_relative(offdf, method=rel_method)
    if max_only:
        span_min_val = offdf['span'].max()
        offdf = offdf.loc[offdf['span']>=span_min_val]
    return offdf

def get_current_off_processing():
    cur_offs = pd.DataFrame(columns=['subject', 'exp', 'result_nnxo', 'result_nnxr'])
    for sub in swi_subs_exps:
        for exp in swi_subs_exps[sub]:
            off_df_nnxo = core.get_offs(subject=sub, prb='NNXo', structure=None, which='ap', experiment=exp)
            if not off_df_nnxo.empty:
                cur_offs = cur_offs.append({'subject': sub, 'exp': exp, 'result_nnxo': 'present'}, ignore_index=True)
            else:
                cur_offs = cur_offs.append({'subject': sub, 'exp': exp, 'result_nnxo': 'absent'}, ignore_index=True)
        
    for sub in swi_subs_exps:
        for exp in swi_subs_exps[sub]:
            off_df_nnxr = core.get_offs(subject=sub, prb='NNXr', structure=None, which='ap', experiment=exp)
            if not off_df_nnxr.empty:
                cur_offs.loc[(cur_offs['subject'] == sub) & (cur_offs['exp'] == exp), 'result_nnxr'] = 'present'
            else:
                cur_offs.loc[(cur_offs['subject'] == sub) & (cur_offs['exp'] == exp), 'result_nnxr'] = 'absent'
    f, ax = plt.subplots(1, 1, figsize=(5, 10))
    sns.scatterplot(data=cur_offs, x='exp', y='subject', hue='result_nnxo', hue_order=['present', 'absent'], palette=['green', 'red'], s=80, ax=ax)
    return f, ax

def make_odf_relative(odf, method='median', span_min='max'):
    if span_min == 'max':
        span_min_val = odf['span'].max()
        odf_span = odf.loc[odf['span']>=span_min_val]
    elif span_min==None:
        odf_span = odf
    if method == 'mean':
        avgs = odf_span.groupby(['probe', 'descriptor']).mean(numeric_only=True).reset_index()
    elif method == 'median':
        avgs = odf_span.groupby(['probe', 'descriptor']).median(numeric_only=True).reset_index()
    else:
        raise ValueError('method must be mean or median')
    avgs = avgs.loc[avgs['descriptor']=='Early_Baseline_NREM']
    odf['median_duration_rel'] = 0
    odf['duration_rel'] = 0
    odf.loc[odf['probe']=='NNXo', 'median_duration_rel'] = odf.prb('NNXo')['median_duration']/avgs.prb('NNXo')['median_duration'].values[0]
    odf.loc[odf['probe']=='NNXo', 'duration_rel'] = odf.prb('NNXo')['duration']/avgs.prb('NNXo')['duration'].values[0]
    odf.loc[odf['probe']=='NNXr', 'median_duration_rel'] = odf.prb('NNXr')['median_duration']/avgs.prb('NNXr')['median_duration'].values[0]
    odf.loc[odf['probe']=='NNXr', 'duration_rel'] = odf.prb('NNXr')['duration']/avgs.prb('NNXr')['duration'].values[0]
    return odf

def check_if_off_detection_is_done(subject, exp, probe, min_date='2024-07-28'):
    data_folder = f'/Volumes/npx_nfs/nobak/offproj/{exp}/{subject}'
    txt_files = []
    if not os.path.exists(data_folder):
        return []
    for f in os.listdir(data_folder):
        if f.endswith('.txt'):
            if 'SUCCESS' in f:
                if probe in f:
                    txt_files.append(f)
    processed_conditions = []
    for f in txt_files:
        cond = f.split('.')[2]
        date = f.split('--')[1].split('.')[0]
        if pd.Timestamp(date) > pd.Timestamp(min_date):
            processed_conditions.append(cond)
    return processed_conditions

def check_preprocessing(subject, experiment, probe):
    path = f'/Volumes/npx_nfs/nobak/offproj/{experiment}/{subject}/{probe}.processed_ap.zarr'
    if os.path.exists(path):
        return True
    else:
        return False