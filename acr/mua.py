import spikeinterface as si
import pandas as pd
import numpy as np
import zarr
import kdephys as kde
from acr.utils import raw_data_root
from acr.info_pipeline import subject_info_section
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
import os
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.core.recording_tools as rt
import tdt
import acr
import polars as pl

def v_to_uv(data, gain=1e6):
    """Converts data from volts to microvolts.

    Parameters
    ----------
    data : np.array
        data, of shape (n_channels, n_samples), should already be filtered.
    gain : int, optional
        the gain of the amplifier, by default 1000
    """
    data *= gain
    return data


def check_for_preprocessed_mua_data(subject, recording, probe):
    save_folder = f'{raw_data_root}mua_data/{subject}'
    save_path = f'{save_folder}/MUA--{subject}--{recording}--{probe}.zarr'
    if os.path.exists(save_path):
        if os.path.exists(f'{save_path}/properties'):
            if os.path.exists(f'{save_path}/properties/gain_to_uV'):
                return True
    return False

def check_mua_prepro_full_exp(subject, exp, probes=['NNXo', 'NNXr']):
    for probe in probes:
        try:
            recs, starts, durations = acr.units.get_time_info(subject, f'{exp}-{probe}')
        except:
            return False
        for rec in recs:
            exists = acr.mua.check_for_preprocessed_mua_data(subject, rec, probe)
            if exists == False:
                return False
    # if all of this passes without returning False, then we can return True
    return True

def _interp_mua_data(data, chans_to_interp, sigma_um=50, p=1.3):
    return sp.interpolate_bad_channels(data, chans_to_interp, sigma_um=sigma_um, p=p)

def prepro_np_array_via_si(np_array, fs):
    nchans = np_array.shape[0]
    if nchans != 16:
        raise ValueError('This function is only for 16 channel probes')
    chan_ids = np.arange(1, nchans+1)
    si_raw = se.NumpyRecording(np_array.T, sampling_frequency=fs, channel_ids=chan_ids)
    si_filt = sp.bandpass_filter(si_raw, freq_min=300, freq_max=12000)
    si_filt_cmr = sp.common_reference(si_filt, reference='global', operator='median')
    
    # now we should set the offset and gain (to get to uV) to 0 and 1 respectively, since we've already converted the incoming array.
    num_chans = si_filt_cmr.get_num_channels()
    si_filt_cmr.set_property(key="gain_to_uV", values=np.ones(num_chans, dtype="float32"))
    si_filt_cmr.set_property(key="offset_to_uV", values=np.zeros(num_chans, dtype="float32"))
    
    # we will also set the noise levels here which will be used later in detection
    rand_kwargs = dict(chunk_size=int(si_filt_cmr.sampling_frequency*2), num_chunks_per_segment=100)
    new_nlvs = rt.get_noise_levels(si_filt_cmr, method='mad', force_recompute=True, **rand_kwargs)
    if 'noise_level_mad_scaled' not in si_filt_cmr.get_property_keys():
        si_filt_cmr.set_property(key="noise_level_mad_scaled", values=new_nlvs)
        si_filt_cmr.set_property(key="noise_level_scaled", values=new_nlvs)
    
    si_filt_cmr_with_probe = kde.units.utils.set_probe_and_channel_locations_on_si_rec(si_filt_cmr)
    
    return si_filt_cmr_with_probe 

def save_preprocessed_mua_data(data, subject, recording, probe, njobs=16, chunk_duration='100s', progress_bar=True, overwrite=False):
    subject_dir = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}'
    if os.path.exists == False:
        os.mkdir(subject_dir)
    save_path = f'{subject_dir}/MUA--{subject}--{recording}--{probe}.zarr'
    data.save(folder=save_path, overwrite=overwrite, format="zarr", n_jobs=njobs, chunk_duration=chunk_duration, progress_bar=progress_bar)

def preprocess_data_for_mua(subject, exp_sort_id, probes=['NNXo', 'NNXr'], interpol=False, overwrite=False, njobs=16):
    for probe in probes:
        sort_id = f'{exp_sort_id}-{probe}'
        recs, starts, durations = acr.units.get_time_info(subject, sort_id)
        for rec, dur in zip(recs, durations):
            exists = check_for_preprocessed_mua_data(subject, rec, probe)
            if exists==True:
                if overwrite == True:
                    print(f'Overwriting {subject}--{rec}--{probe}')
                else:
                    print(f'Preprocessed data already exists for {subject}--{rec}--{probe}, set overwrite=True to overwrite')
                    continue
            else:
                print(f'Preprocessing {subject}--{rec}--{probe}')
            block_path = acr.io.acr_path(subject, rec)
            dat = tdt.read_block(block_path, t1=0, t2=dur, store=probe)
            data = dat.streams[probe].data
            data = v_to_uv(data)
            fs = dat.streams[probe].fs
                
            preprocessed_data = prepro_np_array_via_si(data, fs)
            if interpol == True:
                chans_to_interp = acr.info_pipeline.get_interpol_info(subject, probe)
                preprocessed_data = _interp_mua_data(preprocessed_data, chans_to_interp, sigma_um=50, p=1.3)
            save_preprocessed_mua_data(preprocessed_data, subject, rec, probe, overwrite=overwrite, njobs=njobs)
            if interpol == True:
                acr.info_pipeline.write_interpol_done(subject, rec, probe, chans=chans_to_interp, version='ap')
    return

def load_processed_mua_signal(subject, recording, probe, version='xr'):
    path = f'{raw_data_root}mua_data/{subject}/MUA--{subject}--{recording}--{probe}.zarr'
    if version == 'xr':
        data = kde.xr.utils.load_processed_zarr_as_xarray(path)
        return data
    elif version == 'si':
        data = si.read_zarr(path)
        return data
    elif version == 'zarr':
        data = kde.xr.utils.load_raw_processed_zarr(path)
        return data
    else:
        raise ValueError("version must be 'xr', 'si', or 'zarr'")

def check_for_mua_spikes_df(subject, recording, probe):
    save_folder = f'{raw_data_root}mua_data/{subject}/si_peak_dfs'
    save_path = f'{save_folder}/PEAKS--{subject}--{recording}--{probe}.parquet'
    if os.path.exists(save_path):
        return True
    else:
        return False

def detect_mua_spikes_si(subject, recording, probe, save=True, chunk_duration='1s', n_jobs=224, progress_bar=True, threshold=4, overwrite=False):
    # load the raw mua data (zarr) as a spikeinterface recording extractor
    raw_mua_exists = check_for_preprocessed_mua_data(subject, recording, probe)
    if raw_mua_exists == False:
        print(f'Preprocessed data does not exist for {subject}--{recording}--{probe}, run preprocess_data_for_mua first')
        return None
    if (overwrite==False) and (save==True):
        if check_for_mua_spikes_df(subject, recording, probe):
            print(f'Peaks already detected for {subject}--{recording}--{probe}, set overwrite=True to overwrite')
            return None
    rec = load_processed_mua_signal(subject, recording, probe, version='si')
    job_kwargs = dict(chunk_duration=chunk_duration, n_jobs=n_jobs, progress_bar=progress_bar)
    noise_levs = rec.get_property('noise_level_mad_scaled')
    peaks = detect_peaks(rec, method='by_channel', detect_threshold=threshold, noise_levels=noise_levs, peak_sign='neg', **job_kwargs)
    dtype = [('sample', 'int64'), ('channel', 'int64'), ('amplitude', 'float64'), ('segment', 'int64')]
    peaks_array = peaks.view(dtype).reshape(-1)
    peaks_df = pd.DataFrame(peaks_array)
    peaks_df.drop(columns='segment', inplace=True)
    peaks_df['time'] = peaks_df['sample'] / rec.get_sampling_frequency()
    peaks_df['subject'] = subject
    peaks_df['recording'] = recording
    peaks_df['probe'] = probe
    if save==True:
        save_folder = f'{raw_data_root}mua_data/{subject}/si_peak_dfs'
        save_path = f'{save_folder}/PEAKS--{subject}--{recording}--{probe}.parquet'
        if os.path.exists(save_folder) == False:
            os.mkdir(save_folder)
        peaks_df.to_parquet(save_path)
        return peaks_df
    else:
        return peaks_df


def detect_all_subject_mua_spikes(subject, threshold=4, chunk_duration='1s', n_jobs=224, progress_bar=True, save=True, overwrite=False):
    mua_folder = f'{raw_data_root}mua_data/{subject}'
    stems = []
    for f in os.listdir(mua_folder):
        if ('MUA--' in f) and ('zarr' in f):
            stems.append(f.split('.zarr')[0].split('MUA--')[1])
    for stem in stems:
        rec = stem.split('--')[1]
        probe = stem.split('--')[2]
        print(f'Detecting spikes for {subject}--{rec}--{probe}')
        detect_mua_spikes_si(subject, rec, probe, threshold=threshold, chunk_duration=chunk_duration, n_jobs=n_jobs, progress_bar=progress_bar, save=save, overwrite=overwrite)
    return

def list_all_preprocessed_mua_data(subject):
    processed = []
    mua_folder = f'{raw_data_root}mua_data/{subject}'
    if os.path.exists(mua_folder) == False:
        return processed
    for f in os.listdir(mua_folder):
        if ('MUA--' in f) and ('zarr' in f):
            sub = f.split('--')[1]
            rec = f.split('--')[2]
            probe = f.split('--')[3].split('.zarr')[0]
            processed.append((sub, rec, probe))
    return processed

def load_detected_mua_spikes(subject, rec, probe, dt=True):
    path = f'{raw_data_root}mua_data/{subject}/si_peak_dfs/PEAKS--{subject}--{rec}--{probe}.parquet' 
    df = pl.read_parquet(path)
    df = df.with_columns(channel=pl.col('channel') + 1) #because initially the channel index was zero-based.
    if dt == True:
        raw_times = df['time'].to_numpy()
        rec_times = subject_info_section(subject, 'rec_times')
        start_dt = pd.Timestamp(rec_times[rec]['start'])
        times_td = pd.to_timedelta(raw_times, unit='s')
        times_in_dt = start_dt + times_td
        df = df.with_columns(pl.Series('datetime', times_in_dt))
    return df.with_columns(negchan=pl.col('channel')*-1)

def load_mua_full_exp(subject, exp, probes=['NNXo', 'NNXr'], exclude_dead_chans=True):
    all_dfs = []
    for probe in probes:
        sort_id = f'{exp}-{probe}'
        recs, starts, durations = acr.units.get_time_info(subject, sort_id)
        for rec, dur in zip(recs, durations):
            peaks = load_detected_mua_spikes(subject, rec, probe)
            all_dfs.append(peaks)
    full_df = pl.concat(all_dfs)
    if exclude_dead_chans:
        #if any channel is dead on either probe, that channel is excluded from the full df
        return acr.io.nuke_bad_chans_from_df(subject, exp, full_df, which='dead', probe=None)
    else:
        return full_df



def get_state_fr(df, hyp, t1=None, t2=None, state="NREM", min_duration=30, prb=False):
    """gets the firing rate for each channel during a specified state.

    Args:
    ----------
        - df (dataframe): spike dataframe
        - hyp (dataframe): hypnogram
        - t1 (datetime, optional): if not specified, the first 12 hours from the start of the first spike are used. Defaults to None.
        - t2 (datetime, optional): if not specified, the first 12 hours from the start of the first spike are used. Defaults to None.
        - state (str, optional): state to use. Defaults to 'NREM'.

    Returns:
    -------------------------
        fr_master: dataframe that has the number of spikes and total duration of each bout for each channel. These can then be summed to get the firing rate across all bouts.
    """
    if t1 is None and t2 is None:
        t1 = df["datetime"].to_pandas().min()
        t2 = t1 + pd.Timedelta("12h")

    hyp_state = hyp.loc[hyp.state == state]
    hyp_state = hyp_state.loc[hyp_state.end_time < t2]

    fr_master = pl.DataFrame()

    for bout in hyp_state.itertuples():
        start = bout.start_time
        end = bout.end_time
        bout_duration = (end - start).total_seconds()
        if bout_duration < min_duration:
            continue
        if prb == True:
            spikes = df.ts(start, end).groupby(["probe"]).count()
        else:
            spikes = df.ts(start, end).groupby(["probe", "channel"]).count()
        fr_bout = spikes.with_columns(pl.lit(bout_duration).alias("bout_duration"))
        fr_master = pl.concat([fr_master, fr_bout])
    return fr_master




def fr_arbitrary_bout(df, t1, t2, by="channel"):
    """Calculate the total number of spikes and total duration of an arbitrary period of time (t1 to t2), by channel or probe.

    Parameters
    ----------
    df : pl.Dataframe
        spike dataframe
    t1 : datetime
        start time
    t2 : datetime
        end time
    by : str, optional
        calculate firing rate by channel or by probe; DEFAULT = 'channel'

    Returns
    -------
    fr_bout : pl.Dataframe
        dataframe with firing rate for each channel or probe over the specified time period
    """
    if type(t1) == pd.Timestamp:
        bout_duration = (t2 - t1).total_seconds()
    elif type(t1) == np.datetime64:
        bout_duration = (t2 - t1) / np.timedelta64(1, "s")

    if by == "channel":
        spikes = df.ts(t1, t2).groupby(["probe", "channel"]).count()
        fr_bout = spikes.with_columns((pl.lit(bout_duration)).alias("bout_duration"))
    elif by == "probe":
        spikes = df.ts(t1, t2).groupby(["probe"]).count()
        fr_bout = spikes.with_columns((pl.lit(bout_duration)).alias("bout_duration"))
    else:
        raise ValueError("by parameter must be either 'channel' or 'probe'")
    return fr_bout

def __get_rel_fr_df(
    df,
    hyp,
    rel_state="NREM",
    window="120s",
    by="channel",
    t1=None,
    t2=None,
    over_bouts=False,
    return_early=False,
    arb=False,
):
    """gets the firing rate (either by probe or channel) relative to the baseline firing rate in a specified state

    ***NOTE*** the baseline is defined as 9am-9pm on the day of the first spike in df, unless t1 and t2 are specified!

    Args:
        df (_type_): polars spike dataframe
        hyp (_type_): hypnogram
        rel_state (str, optional): state to get the reference firing rates from. Defaults to 'NREM'.
        window (str, optional): window size for polars groupby_dynamic. Defaults to '120s'.
        by (str, optional): get relative firing rate by channel or by probe. Defaults to 'channel'.
        t1 (_type_, optional): If not provided, 9am on the day of the first spike in df is used. Defaults to None.
        t2 (_type_, optional): if not provided, 9pm on the day of the first spike in df is used. Defaults to None.
        over_bouts (bool, optional): if True, the firing rate is averaged over bouts, rather than all bouts being summed. Defaults to False.
        arb (bool, optional): if True, the baseline firing rate is calculated between t1 and t2, regardless of state.

    Returns:
        fr_rel: relative firing rate dataframe, with columns: probe, datetime, fr_rel, and channel if by='channel'
    """
    bl_date = str(df["datetime"].to_pandas().min()).split(" ")[0]
    bl_start = pd.Timestamp(f"{bl_date} 09:00:00")
    if t1 == None and t2 == None:
        t1 = bl_start
        t2 = bl_start + pd.Timedelta("12h")
    if arb == True:
        bl_frs = fr_arbitrary_bout(df, t1, t2, by="channel")
    else:
        bl_frs = get_state_fr(df, hyp, t1=t1, t2=t2, state=rel_state)
    window_time = int(window.strip("s"))
    if by == "probe":
        if over_bouts == True:
            bl_frs_by_probe = bl_frs.frates().groupby(["probe", "channel"]).mean()
        else:
            bl_frs_by_probe = bl_frs.groupby(["probe", "channel"]).sum()
            if return_early == True:
                return bl_frs_by_probe
            bl_frs_by_probe = bl_frs_by_probe.groupby(["probe", "bout_duration"]).sum()
            bl_frs_by_probe = bl_frs_by_probe.with_columns(
                (pl.col("count") / pl.col("bout_duration")).alias("fr")
            ).drop("count", "bout_duration")
        fr_window = df.groupby_dynamic(
            "datetime", every=window, start_by="datapoint", closed="left", by="probe"
        ).agg(pl.col("channel").count())
        fr_raw = fr_window.with_columns(
            (pl.col("channel") / window_time).alias("fr")
        ).drop("channel")
        fr_rel = fr_raw.with_columns(
            pl.when(pl.col("probe") == "NNXr")
            .then(pl.col("fr") / bl_frs_by_probe.prb("NNXr")["fr"][0])
            .otherwise(pl.col("fr") / bl_frs_by_probe.prb("NNXo")["fr"][0])
            .alias("fr_rel")
        )
        return fr_rel.drop("fr")
    elif by == "channel":
        bl_frs_by_channel = bl_frs.groupby(["probe", "channel"]).sum()
        bl_frs_by_channel = bl_frs_by_channel.with_columns(
            (pl.col("count") / pl.col("bout_duration")).alias("fr")
        ).drop("count", "bout_duration")
        if return_early == True:
            return bl_frs_by_channel
        fr_window = df.groupby_dynamic(
            "datetime",
            every=window,
            start_by="datapoint",
            closed="left",
            by=["probe", "channel"],
        ).agg(pl.col("channel").count().alias("count"))
        fr_rel = fr_window.with_columns(
            (pl.col("count") / window_time).alias("fr_rel")
        ).drop(
            "count"
        )  # not really relative yet
        for probe in fr_rel["probe"].unique():
            for channel in fr_rel.prb(probe).chunq():
                fr_rel = fr_rel.with_columns(
                    pl.when(
                        (pl.col("probe") == probe)
                        & (pl.col("channel") == channel)
                    )
                    .then(
                        pl.col("fr_rel")
                        / bl_frs_by_channel.pchan(probe, channel)["fr"][0]
                    )
                    .otherwise(pl.col("fr_rel"))
                )
        return fr_rel


def get_hypno_frs(df, h):
    """Loops through every bout in a hypnogram, counts the number of spikes and total duration for each probe-channel pair, then adds all of those to a new dataframe. 

    Parameters
    ----------
    df : _type_
        _description_
    h : _type_
        _description_
    """
    h_frs = []
    bout_num = 1
    for bout in h.itertuples():
        bout_start = bout.start_time
        bout_end = bout.end_time
        duration = (bout_end - bout_start).total_seconds()
        bout_df = df.ts(bout_start, bout_end)
        bout_df = bout_df.groupby(['probe', 'channel']).count()
        bout_df = bout_df.with_columns(bout_duration=pl.lit(duration))
        bout_df = bout_df.with_columns(bout_number=pl.lit(bout_num))
        h_frs.append(bout_df)
        bout_num += 1
    full_df = pl.concat(h_frs)
    return full_df

def get_reb_and_bl_frs(df, bh, rh):
    bl_frs = acr.mua.get_hypno_frs(df, bh)
    reb_frs = acr.mua.get_hypno_frs(df, rh)
    bl_frs = bl_frs.with_columns(cond=pl.lit('baseline'))
    reb_frs = reb_frs.with_columns(cond=pl.lit('rebound'))
    return pl.concat([bl_frs, reb_frs])

def add_states_to_df(df, h):
    start_times = h['start_time'].values
    end_times = h['end_time'].values
    states = h['state'].values
    times = df['datetime'].to_numpy()
    
    # Find the indices in the start_times where each time in `times` would be inserted
    indices = np.searchsorted(start_times, times, side='right') - 1

    # Ensure the found index is within bounds and the time is within the bout
    indices = np.clip(indices, 0, len(start_times) - 1)
    valid_mask = (times >= start_times[indices]) & (times <= end_times[indices])

    # Create an array of the same shape as `times` filled with the corresponding states
    state_array = np.empty(times.shape, dtype=states.dtype)
    state_array[valid_mask] = states[indices[valid_mask]]

    # If there are times outside of the hypnogram ranges (unlikely in your case)
    # you may want to handle them, e.g., by setting a default state like 'unknown'
    state_array[~valid_mask] = 'unlabelled'  # or whatever is appropriate
    df = df.with_columns(state=pl.Series(state_array.astype(str)))
    return df

def get_dynamic_fr_df(mua_df, every=60, closed='left', by=['probe', 'channel']):
    frt = mua_df.groupby_dynamic(
            "datetime",
            every=f'{every}s',
            start_by="datapoint",
            closed=closed,
            by=by,
        ).agg(pl.col("amplitude").count().alias("count"))
    return frt.with_columns(fr = pl.col('count')/every)

def make_fr_df_relative(frdf, t1=None, t2=None, state=['NREM']):
    bl_date = str(frdf["datetime"].to_pandas().min()).split(" ")[0]
    bl_start = pd.Timestamp(f"{bl_date} 09:00:00")
    if t1 == None and t2 == None:
        t1 = bl_start
        t2 = bl_start + pd.Timedelta("12h")
    frdf = frdf.with_columns(fr_rel = pl.lit(101))
    if state is not None:
        blfr = frdf.filter(pl.col('state').is_in(state)).ts(t1, t2)
    else:
        blfr = frdf.ts(t1, t2)
    blrates = blfr.groupby(['probe', 'channel']).mean().to_pandas().reset_index()
    for probe in blrates.prbs():
        for channel in blrates.prb(probe)['channel'].unique():
            blrate = blrates.prb(probe).chnl(channel)['fr'].values[0]
            frdf = frdf.with_columns(pl.when(((pl.col('probe') == probe) & (pl.col('channel') == channel))).then(pl.col('fr')/blrate).otherwise(pl.col('fr_rel')).alias('fr_rel'))
    return frdf

def _check_concat_df_exists(subject, exp, probe):
    save_folder = f'{raw_data_root}mua_data/{subject}/si_peak_dfs/concat_dfs'
    save_path = f'{save_folder}/PEAKS--{subject}--{exp}--{probe}.parquet'
    if os.path.exists(save_path) == False:
        return False
    elif os.path.exists(save_path) == True:
        return True
    else:
        return False
    
def detect_peaks_on_full_concat_recording(subject, exp, njobs=112, probes=['NNXr', 'NNXo'], overwrite=False):
    for probe in probes:
        if _check_concat_df_exists(subject, exp, probe) and overwrite==False:
            print(f'Peaks already detected for {subject}--{exp}--{probe}, set overwrite=True to overwrite')
            continue
        si_recs = []
        sort_id = f'{exp}-{probe}'
        recs, starts, durations = acr.units.get_time_info(subject, sort_id)
        for rec in recs:
            print(f'loading {subject}--{rec}--{probe}')
            assert check_for_preprocessed_mua_data(subject, rec, probe)
            si_rec = load_processed_mua_signal(subject, rec, probe, version='si')
            si_recs.append(si_rec)
        fullsi = si.concatenate_recordings(si_recs)
        print('calculating noise levels...')
        combo_noise = rt.get_noise_levels(fullsi, chunk_size=int(fullsi.sampling_frequency*2), num_chunks_per_segment=(len(recs)*100))
        fullsi.set_property(key="noise_level_mad_scaled", values=combo_noise)
        fullsi.set_property(key="noise_level_scaled", values=combo_noise)
        job_kwargs = dict(chunk_duration='1s', n_jobs=njobs, progress_bar=True)
        nl = fullsi.get_property('noise_level_mad_scaled')
        print('detecting peaks...')
        peaks = detect_peaks(fullsi, method='by_channel', noise_levels=nl, detect_threshold=4, peak_sign='neg', **job_kwargs)
        dtype = [('sample', 'int64'), ('channel', 'int64'), ('amplitude', 'float64'), ('segment', 'int64')]
        peaks_array = peaks.view(dtype).reshape(-1)
        peaks_df = pd.DataFrame(peaks_array)
        peaks_df.drop(columns='segment', inplace=True)
        peaks_df['time'] = peaks_df['sample'] / fullsi.get_sampling_frequency()
        peaks_df['subject'] = subject
        peaks_df['exp'] = exp
        peaks_df['probe'] = probe
        peaks_df = acr.units.assign_recordings_to_spike_df(peaks_df, recs, durations)
        peaks_df = acr.units.assign_datetimes_to_spike_df(peaks_df, recs, starts)
        base_df_dir = f'{raw_data_root}mua_data/{subject}/si_peak_dfs'
        if os.path.exists(base_df_dir) == False:
            os.mkdir(base_df_dir)
        save_folder = f'{raw_data_root}mua_data/{subject}/si_peak_dfs/concat_dfs'
        if os.path.exists(save_folder) == False:
            os.mkdir(save_folder)
        save_path = f'{save_folder}/PEAKS--{subject}--{exp}--{probe}.parquet'
        peaks_df.to_parquet(save_path, version='2.6')
    return

def load_concat_peaks_df(subject, exp, probes=['NNXo', 'NNXr'], segs=False):
    concat_dfs = []
    for probe in probes:
        save_folder = f'{raw_data_root}mua_data/{subject}/si_peak_dfs/concat_dfs'
        save_path = f'{save_folder}/PEAKS--{subject}--{exp}--{probe}.parquet'
        df = pl.read_parquet(save_path)
        df = df.with_columns(channel=pl.col('channel') + 1) #because initially the channel index was zero-based.
        df = df.with_columns(negchan=pl.col('channel')*-1)
        concat_dfs.append(df)
    mua = pl.concat(concat_dfs)
    if segs:
        mua = merge_segment_dfs_to_full_mua(subject, exp, mua)
    return mua

def full_mua_pipeline_for_subject(subject, list_of_exps=None, df_version='concat', interpol=False, overwrite=False, detect_jobs=112):
    if list_of_exps == None:
        list_of_exps = acr.info_pipeline.get_subject_experiments(subject)
        list_of_exps = list(list_of_exps.keys())
    for exp in list_of_exps:
        exp_recs = acr.info_pipeline.get_exp_recs(subject, exp)
        # 1. Preprocess the MUA data
        preprocess_data_for_mua(subject, exp, overwrite=overwrite, interpol=interpol, njobs=32)
        
        # 2. Run the detection on the preprocessed data, either by recording, or on the concatenated data, or both
        if df_version == 'concat':
            detect_peaks_on_full_concat_recording(subject, exp, njobs=detect_jobs, overwrite=overwrite)
        elif df_version == 'by-rec':
            for rec in exp_recs:
                for probe in ['NNXo', 'NNXr']:
                    detect_mua_spikes_si(subject, rec, probe, save=True, overwrite=overwrite, n_jobs=detect_jobs)
        elif df_version == 'both':
            detect_peaks_on_full_concat_recording(subject, exp, njobs=detect_jobs, overwrite=overwrite)
            for rec in exp_recs:
                for probe in ['NNXo', 'NNXr']:
                    detect_mua_spikes_si(subject, rec, probe, save=True, overwrite=overwrite, n_jobs=detect_jobs)
        else:
            raise ValueError('df_version must be one of "concat", "by-rec", "both"')
    return

def save_interpolated_mua_data(data, subject, recording, probe):
    save_folder = f'{raw_data_root}mua_data/{subject}'
    save_path = f'{save_folder}/MUA--{subject}--{recording}--{probe}.zarr'
    assert os.path.exists(save_path), 'original data must first exist to save interpolated data'
    
    #First we temporarily save the interpolated data
    temp_save_path = f'{save_folder}/TEMP--{subject}--{recording}--{probe}.zarr'
    data.save(folder=temp_save_path, overwrite=False, format="zarr", n_jobs=64, chunk_duration='100s', progress_bar=True)
    
    #Now we remove the original (uninterpolated) data
    os.system(f'rm -rf {save_path}')
    
    #Now we rename the temporary folder to the original data folder
    os.rename(temp_save_path, save_path)
    
    return

def interpol_and_resave_mua_data(subject, exp, probes=['NNXo', 'NNXr'], redo=False):
    for probe in probes:
        sort_id = f'{exp}-{probe}'
        recs, starts, durations = acr.units.get_time_info(subject, sort_id)
        for rec in recs:
            assert check_for_preprocessed_mua_data(subject, rec, probe), f'no preprocessed data found for {subject}--{rec}--{probe}'
            interped_chans = acr.info_pipeline.read_interpol_done(subject, rec, probe, version='ap')
            if interped_chans is not None:
                if redo == False:
                    print(f'interpolated data already exists for {subject}--{rec}--{probe}, set redo=True to interpolate again, but do this with caution!')
                    continue
            chans_to_interp = acr.info_pipeline.get_interpol_info(subject, probe)
            if len(chans_to_interp) == 0: # if there is nothing to interpolate for this probe, we just continue
                continue
            data = load_processed_mua_signal(subject, rec, probe, version='si')
            data = _interp_mua_data(data, chans_to_interp)
            save_interpolated_mua_data(data, subject, rec, probe)
            acr.info_pipeline.write_interpol_done(subject, rec, probe, chans=interped_chans, version='ap')
    return


def filter_on_amplitude(df, group_cols=['probe', 'channel'], q=0.2):
    # Group by the specified columns and calculate the 80th percentile
    if 'abs_amp' not in df.columns:
        df = df.with_columns(abs_amp=np.abs(df['amplitude']))
    
    
    
    threshold_df = df.group_by(group_cols).agg(
        pl.col('abs_amp').quantile(q).alias('threshold')
    )
    
    # Join the threshold back to the original DataFrame
    result = df.join(threshold_df, on=group_cols)
    
    # Filter to keep only the top 80% of values
    result = result.filter(pl.col('abs_amp') >= pl.col('threshold')).select(df.columns)
    
    return result

def merge_segment_dfs_to_full_mua(subject, exp, mua):
    seg_df_dir = f'{acr.utils.raw_data_root}mua_data/{subject}/seg_dfs'
    segs = []
    for f in os.listdir(seg_df_dir):
        if f.endswith('.parquet'):
            if f'{exp}--' in f:
                df = pl.read_parquet(f'{seg_df_dir}/{f}')
                df = df.with_columns(channel=pl.col('channel') + 1) #because initially the channel index was zero-based.
                df = df.with_columns(negchan=pl.col('channel')*-1)
                segs.append(df)
    seg_mua = pl.concat(segs)
    seg_start = seg_mua['datetime'].to_pandas().min()
    seg_end = seg_mua['datetime'].to_pandas().max()
    mua_cleaned = mua.filter(~(pl.col('datetime').is_between(seg_start, seg_end)))
    mua_new = pl.concat([mua_cleaned, seg_mua])
    mua_new = mua_new.sort('datetime')
    return mua_new