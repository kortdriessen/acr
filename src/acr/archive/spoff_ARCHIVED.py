#from offproj import core, tdt_core
import acr
import numpy as np
import pandas as pd
from acr.utils import swi_subs_exps, sub_probe_locations, sub_exp_types
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dask
import os
import dask_image
from functools import partial
import spikeinterface.preprocessing as sp
import kdephys as kde
import xarray as xr
import dask_image
import dask_image.ndmeasure
from dask_image import ndfilters
import dask.array as dska
from xhistogram.xarray import histogram
import pickle
import scipy
import polars as pl
import matplotlib as mpl
from acr.utils import raw_data_root

def _sp_off_df_path(subject, rec, probe):
    return f'{raw_data_root}mua_data/{subject}/spatial_off/spoff_dfs/{rec}--{probe}.parquet'

def check_spoff_full_exp(subject, exp, probes=['NNXo', 'NNXr']):
    recs = acr.info_pipeline.get_exp_recs(subject, exp)
    all_exist = True
    for rec in recs:
        for probe in probes:
            spdf_path = _sp_off_df_path(subject, rec, probe)
            if os.path.exists(spdf_path):
                continue
            else:
                all_exist = False
    return all_exist

def _check_for_off_mask(subject, rec, probe):
    save_folder = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/spatial_off'
    save_path = os.path.join(save_folder, f'mask__{rec}--{probe}.zarr')
    if os.path.exists(save_path):
        return True
    else:
        return False

def _check_for_prepro_data(subject, rec, probe):
    save_folder = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/spatial_off'
    save_path = os.path.join(save_folder, f'prepro__{rec}--{probe}.zarr')
    if os.path.exists(save_path):
        return True
    else:
        return False

def get_quantile_thresh(x, quantile):
    return np.quantile(x, quantile)

def _save_spatial_off_prepro_data(sig, subject, recording, probe, overwrite=False):
    save_folder = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/spatial_off'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, f'prepro__{recording}--{probe}.zarr')
    if os.path.exists(save_path):
        if overwrite==False:
            raise FileExistsError(f"File already exists at {save_path}")
        else:
            print(f"Overwriting file at {save_path}")
            os.system(f"rm -rf {save_path}")
            print(f"File removed at {save_path}, now re-saving")
    job_kwargs = dict(
        n_jobs=64, chunk_duration=f"{100}s", progress_bar=True
    )
    sig.save(folder=save_path, overwrite=True, format="zarr", **job_kwargs)
    return

def save_spatial_off_mask(om, subject, recording, probe, overwrite=False):
    save_folder = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/spatial_off'
    save_path = os.path.join(save_folder, f'mask__{recording}--{probe}.zarr')
    if os.path.exists(save_path):
        if overwrite==False:
            raise FileExistsError(f"File already exists at {save_path}")
        else:
            print(f"Overwriting file at {save_path}")
            os.system(f"rm -rf {save_path}")
            print(f"File removed at {save_path}, now re-saving")
    om.to_zarr(save_path)
    return


def prepro_and_save_spatial_off_data(subject, recording, probe, overwrite=False):
    sig = acr.mua.load_processed_mua_signal(subject, recording, probe, version='si')
    sp_sig = _prepro_spatial_off_data(sig)
    _save_spatial_off_prepro_data(sp_sig, subject, recording, probe, overwrite=overwrite)
    return


def run_full_spatial_off_detection(subject, recording, probe, thresh=None, med_filt_samples=10, med_filt_chans=1, save=True, overwrite=False):
    #data = load_spatial_off_prepro_data(subject, recording, probe)
    if thresh is None:
        thresh = get_threshold_for_spoff_detection(subject, probe)
    off_mask = run_detection(subject, recording, probe, thresh=thresh, med_filt_samples=med_filt_samples, med_filt_chans=med_filt_chans)
    if save:
        save_spatial_off_mask(off_mask, subject, recording, probe, overwrite=overwrite)
        return
    else:
        return off_mask

def load_spoff_full_exp(subject, exp, probes=['NNXo', 'NNXr']):
    all_dfs = []
    for probe in probes:
        sort_id = f'{exp}-{probe}'
        recs, starts, durations = acr.units.get_time_info(subject, sort_id)
        for rec, dur in zip(recs, durations):
            odf = acr.spoff.load_spoff_df(subject, rec, probe)
            all_dfs.append(odf)
    full_df = pd.concat(all_dfs)
    return pl.DataFrame(full_df)


def _prepro_spatial_off_data(sig, dec_factor=100, fmax=20):
    offsig = sp.zscore(sig, dtype='float32')
    offsig = sp.rectify(offsig)
    offsig = sp.gaussian_filter(offsig, freq_min=None, freq_max=fmax)
    offsig = sp.decimate(offsig, decimation_factor=dec_factor)

    total_t = sig.get_num_samples() / sig.get_sampling_frequency()
    time_vector = np.linspace(0, total_t, sig.get_num_samples())
    time_vector_dec = time_vector[::dec_factor]
    offsig.set_times(time_vector_dec)
    return offsig

def load_spatial_off_mask(subject, recording, probe):
    save_folder = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/spatial_off'
    save_path = os.path.join(save_folder, f'mask__{recording}--{probe}.zarr')
    assert os.path.exists(save_path), f"File does not exist at {save_path}"
    data = xr.open_zarr(save_path)
    data = data.assign_coords(y=('channel', data.channel.values*-1))
    return data['OFF mask']


def load_spatial_off_prepro_data(subject, recording, probe):
    save_folder = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/spatial_off'
    save_path = os.path.join(save_folder, f'prepro__{recording}--{probe}.zarr')
    assert os.path.exists(save_path), f"File does not exist at {save_path}"
    data = kde.xr.utils.load_processed_zarr_as_xarray(save_path)
    neg_chans = np.arange(-16, 0, 1)
    data = data.assign_coords(y = ("channel", data.channel.values*-1))
    return data


def _median_filter_prepro_data(data, med_filt_samples=10, med_filt_chans=1):
    smdata = data.copy()
    smdata.data = ndfilters.median_filter(
            smdata.data,
            footprint=np.ones((med_filt_samples, med_filt_chans)),)
    return smdata

def _thresh_da(arr):
    threshes = xr.DataArray(
        data=arr,
        dims=("channel"),
        coords={
            "channel": np.arange(1, len(arr)+1),
        },
        name="Detection threshold",
    )
    return threshes

def get_thresholds_entire_da(da, quantile=0.25):
    func = partial(get_quantile_thresh, quantile=quantile)
    _threshes = dask.array.apply_along_axis(func, 0, da)
    return _thresh_da(_threshes)


def _gen_mask(prepro_data, thresh=0.25):
    func = partial(get_quantile_thresh, quantile=thresh)
    _threshes = dask.array.apply_along_axis(func, 0, prepro_data)
    threshes = xr.DataArray(
        data=_threshes,
        dims=("channel"),
        coords={
            "channel": prepro_data.channel,
        },
        name="Detection threshold",
    )
    om_raw = prepro_data.copy() 
    om_raw.data = dask.array.where(prepro_data < threshes, True, False)
    om_raw.name = "OFF mask"
    neg_chans = np.arange(-16, 0, 1)
    om_raw = om_raw.assign_coords(y = ("channel", neg_chans))
    om_raw = om_raw.assign_coords(thresholds = ("channel", threshes.data))
    return om_raw

def run_detection(subject, recording, probe, thresh=0.25, med_filt_samples=10, med_filt_chans=1):
    if _check_for_prepro_data(subject, recording, probe) == False:
        raise FileExistsError(f"Preprocessed data does not exist for {subject} {recording} {probe}")
    data = load_spatial_off_prepro_data(subject, recording, probe)
    smdata = _median_filter_prepro_data(data, med_filt_samples=med_filt_samples, med_filt_chans=med_filt_chans)
    func = partial(get_quantile_thresh, quantile=thresh)
    _threshes = dask.array.apply_along_axis(func, 0, smdata)
    threshes = xr.DataArray(
        data=_threshes,
        dims=("channel"),
        coords={
            "channel": smdata.channel,
        },
        name="Detection threshold",
    )
    om_raw = smdata.copy() 
    om_raw.data = dask.array.where(smdata < threshes, True, False)
    om_raw.name = "OFF mask"
    neg_chans = np.arange(-16, 0, 1)
    om_raw = om_raw.assign_coords(y = ("channel", neg_chans))
    om_raw = om_raw.assign_coords(thresholds = ("channel", threshes.data))
    om_raw = om_raw.assign_attrs(thresh=thresh, med_filt_samples=med_filt_samples, med_filt_chans=med_filt_chans)
    return om_raw


def morpho_clean_mask(off_mask, vert_remove=4, horz_remove=5, vert_connect=3, vert_remove2=5, vert_connect2=5, morpho_ops=None):
    import dask_image.ndmorph
    from dask_image import ndmorph
    if morpho_ops is not None:
        horz_remove = morpho_ops['horz_remove']
        vert_remove = morpho_ops['vert_remove']
        vert_connect = morpho_ops['vert_connect']
        vert_remove2 = morpho_ops['vert_remove2']
        vert_connect2 = morpho_ops['vert_connect2']

    om = off_mask.copy()
    
    
    # # # vertical : Remove few-channel epochs
    struct = np.ones((1, vert_remove))
    print(f"vert remove: {struct.shape}")
    om.data = dask_image.ndmorph.binary_opening(om.data, structure=struct, iterations=1)
    
    # Horizontal  Remove shorter blobs
    struct = np.ones((horz_remove, 1))
    print(f"horizonal remove: {struct.shape}")
    om.data = dask_image.ndmorph.binary_opening(om.data, structure=struct, iterations=2)
    
    # vertical : Connect distant blobs vertically
    struct = np.ones((1, vert_connect))
    print(f"Vert connect: {struct.shape}")
    temp_pad = dska.pad(om.data, ((0, 0), (vert_connect, vert_connect)), mode='edge')
    temp_pad_cleaned = dask_image.ndmorph.binary_closing(temp_pad, structure=struct, iterations=1)
    # remove the padding
    om.data = temp_pad_cleaned[:, vert_connect:-vert_connect]
    
    # # # vertical : Remove few-channel epochs
    struct = np.ones((1, vert_remove2))
    print(f"vert remove: {struct.shape}")
    om.data = dask_image.ndmorph.binary_opening(om.data, structure=struct, iterations=1)
    
    # vertical : Connect distant blobs vertically
    struct = np.ones((1, vert_connect2))
    print(f"Vert connect: {struct.shape}")
    temp_pad = dska.pad(om.data, ((0, 0), (vert_connect2, vert_connect2)), mode='edge')
    temp_pad_cleaned = dask_image.ndmorph.binary_closing(temp_pad, structure=struct, iterations=1)
    # remove the padding
    om.data = temp_pad_cleaned[:, vert_connect2:-vert_connect2]
    
    # Horizontal  Remove shorter blobs
    struct = np.ones((3, 1))
    print(f"horizonal remove: {struct.shape}")
    om.data = dask_image.ndmorph.binary_opening(om.data, structure=struct, iterations=3)
    
    return om



def morpho_clean_mask_dev(off_mask, horz_remove=10, vert_remove=2, vert_connect=2, horz_connect=6, morpho_ops=None):
    import dask_image.ndmorph
    from dask_image import ndmorph
    if morpho_ops is not None:
        horz_remove = morpho_ops['horz_remove']
        vert_remove = morpho_ops['vert_remove']
        vert_connect = morpho_ops['vert_connect']
        horz_connect = morpho_ops['horz_connect']

    om = off_mask.copy()
    
    # # # vertical : Remove few-channel epochs
    struct = np.ones((1, vert_remove))
    print(f"Binary open: {struct.shape}")
    om.data = dask_image.ndmorph.binary_opening(om.data, structure=struct, iterations=1)
    
     # # Horizontal  Remove shorter blobs
    struct = np.ones((horz_remove, 1))
    print(f"Binary open: {struct.shape}")
    om.data = dask_image.ndmorph.binary_opening(om.data, structure=struct, iterations=1)
    
     # # vertical : Connect distant blobs vertically
     # vertical : Connect distant blobs vertically
    struct = np.ones((1, vert_connect))
    print(f"Binary close: {struct.shape}")
    temp_pad = dska.pad(om.data, ((0, 0), (vert_connect, vert_connect)), mode='edge')
    temp_pad_cleaned = dask_image.ndmorph.binary_closing(temp_pad, structure=struct, iterations=1)
    # remove the padding
    om.data = temp_pad_cleaned[:, vert_connect:-vert_connect]
    
    #horizontal closing: connect across small gaps
    struct = np.ones((horz_connect, 1))
    print(f"Binary close: {struct.shape}")
    om.data = dask_image.ndmorph.binary_closing(om.data, structure=struct, iterations=1)
    
    return om


def get_offs_df(da, lbl_ixs):
    """Generate offs dataframe.
    
    Args:
    da: xr.DataArray
        DataArray used for detection
    lbl_ixs: dict
        {<label>: (<col_indices>, <row_indices>)}
    """

    def _get_median_nframes(col_indices, row_indices):
        tmp = pd.DataFrame({'row': row_indices, 'col': col_indices})
        return tmp.groupby('row').count()['col'].median()

    _lbls = np.sort(list(lbl_ixs.keys()))
    start_frames = pd.DataFrame([lbl_ixs[lbl][0].min() for lbl in _lbls], columns=['start_frame'], index=_lbls)
    end_frames = pd.DataFrame([lbl_ixs[lbl][0].max() for lbl in _lbls], columns=['end_frame'], index=_lbls)
    median_nframes = pd.DataFrame([_get_median_nframes(*lbl_ixs[lbl]) for lbl in _lbls], columns=['median_nframes'], index=_lbls)
    lo_chan_ix = pd.DataFrame([lbl_ixs[lbl][1].min() for lbl in _lbls], columns=['lo_chan_idx'], index=_lbls)
    max_chan_idx = pd.DataFrame([lbl_ixs[lbl][1].max() for lbl in _lbls], columns=['max_chan_idx'], index=_lbls)

    # Can't compute area / convexity ratio this way if channels are not evenly spaced
    if len(set(np.diff(da.y.values))) > 1:
        raise NotImplementedError("Require evenly spaced channels")
    areas = pd.DataFrame([lbl_ixs[lbl][0].size for lbl in _lbls], columns=['area'], index=_lbls)

    df = pd.concat([areas, start_frames, end_frames, median_nframes, lo_chan_ix, max_chan_idx], axis=1).dropna()
    df['label'] = df.index

    y = da.y.values
    t = da.time.values

    df['start_time'] = t[df['start_frame'].values]
    df['end_time'] = t[df['end_frame'].values]
    df['duration'] = df['end_time'] - df['start_time']
    df['median_duration'] = df['median_nframes'] / da.attrs["fs"]
    df['lo'] = y[df['lo_chan_idx'].values]
    df['hi'] = y[df['max_chan_idx'].values]
    df['span'] = np.abs(df['hi'] - df['lo'])

    def _assign_convexity_metrics(df, lbl_ixs):
        def _get_row_area(row):
            from scipy.spatial import ConvexHull, QhullError
            try:
                return ConvexHull(np.transpose(np.array(lbl_ixs[row["label"]]))).volume
            except QhullError:
                convex_area = float("Inf")
        convex_area = df.apply(
            lambda row: _get_row_area(row),
            axis=1
        )
        df["convexity_ratio"] = df["area"] / convex_area
        return df

    df = _assign_convexity_metrics(df, lbl_ixs)

    return df.astype({
        'start_frame': np.dtype('int64'),
        'end_frame': np.dtype('int64'),
        'median_nframes': np.dtype('int64'), 
        'lo_chan_idx': np.dtype('int64'),
        'max_chan_idx': np.dtype('int64'),
        'label': np.dtype('int64')
    })


def get_threshold_for_spoff_detection(subject, probe):
    from acr.utils import NREM_QUANTILE_THRESHOLDS
    return NREM_QUANTILE_THRESHOLDS[(subject, probe, None)]

def assign_datetimes_to_off_df(offs, sorting_start):
    starts = offs['start_time'].values
    ends = offs['end_time'].values
    starts_td = pd.to_timedelta(starts, unit='s')
    ends_td = pd.to_timedelta(ends, unit='s')
    starts_dt = sorting_start + starts_td
    ends_dt = sorting_start + ends_td
    offs['start_datetime'] = starts_dt
    offs['end_datetime'] = ends_dt
    return offs

def assign_recordings_to_off_df(df, recs, starts, durations):
    for rec, start, duration in zip(recs, starts, durations):
        start = pd.Timestamp(start)
        end = start + pd.Timedelta(duration, unit='s')
        df.loc[df['start_datetime'].between(start, end), 'rec'] = rec
    return df

def _check_for_off_df(subject, rec, probe):
    save_folder = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/spatial_off/spoff_dfs'
    save_path = os.path.join(save_folder, f'{rec}--{probe}.parquet')
    return os.path.exists(save_path)

def save_off_dfs_whole_subject(subject, overwrite=False):
    mua_dir = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/'
    for f in os.listdir(mua_dir):
        if 'MUA--' not in f:
            continue
        rec = f.split('--')[2]
        probe = f.split('--')[3]
        probe = probe.split('.')[0]
        if _check_for_off_df(subject, rec, probe) == False:
            acr.spoff.save_off_indexes_and_df(subject, rec, probe)
        else:
            if overwrite == False:
                print(f"Off df for {subject} {rec} {probe} already exists, skipping")
                continue
            else:
                acr.spoff.save_off_indexes_and_df(subject, rec, probe)
    return

def full_sp_off_pipeline(subject, exp, probes=['NNXo', 'NNXr'], overwrite_prepro_data=False, overwrite_dfs=False, overwrite_off_mask=False):
    sort_id = f'{exp}-{probes[0]}'
    recs, starts, durations = acr.units.get_time_info(subject, sort_id)
    for rec in recs:
        for probe in probes:
            _full_sp_off_pipeline(subject, rec, probe, overwrite_prepro_data=overwrite_prepro_data, overwrite_off_mask=overwrite_off_mask)
    save_off_dfs_exp(subject, exp, probes=probes, overwrite=overwrite_dfs)
    return            
    
def _full_sp_off_pipeline(subject, rec, probe, overwrite_prepro_data=False, overwrite_off_mask=False):
    
    #First Preprocessing
    if overwrite_prepro_data:
        acr.spoff.prepro_and_save_spatial_off_data(subject, rec, probe, overwrite=overwrite_prepro_data)
    elif _check_for_prepro_data(subject, rec, probe) and overwrite_prepro_data==False:
        print(f"Preprocessing data for {subject} {rec} {probe} already exists, skipping")
    else:
        acr.spoff.prepro_and_save_spatial_off_data(subject, rec, probe, overwrite=False)
    
    #Then the actual Detection
    if overwrite_off_mask:
        acr.spoff.run_full_spatial_off_detection(subject, rec, probe, overwrite=overwrite_off_mask)
    elif _check_for_off_mask(subject, rec, probe) and overwrite_off_mask==False:
        print(f"Off mask for {subject} {rec} {probe} already exists, skipping")
    elif _check_for_off_mask(subject, rec, probe) == False:
        acr.spoff.run_full_spatial_off_detection(subject, rec, probe, save=True, overwrite=overwrite_off_mask)
    else:
        acr.spoff.run_full_spatial_off_detection(subject, rec, probe, save=True, overwrite=False)
    return


def _plot_histos_from_da(da):
    f, ax = plt.subplots(1, 1, figsize=(45, 15))
    max_bin = float(np.quantile(da.data, 0.95))
    min_bin = float(np.min(da.data)) - 0.05
    bins = np.linspace(
        min_bin,
        max_bin,
        100,
    )

    h_chan = histogram(
                da,
                dim=["time"],
                bins=bins,
                density=False,
            )
    h_chan = h_chan / h_chan.max(dim=h_chan.dims[1])
    h_chan.plot(ax=ax)
    
    return f, ax


def save_off_dfs_exp(subject, exp, probes=['NNXo', 'NNXr'], overwrite=False):
    for probe in probes:
        sort_id = f'{exp}-{probe}'
        recs, starts, durations = acr.units.get_time_info(subject, sort_id)
        
        for rec, dur in zip(recs, durations):
            if _check_for_off_df(subject, rec, probe) == False:
                save_off_indexes_and_df(subject, rec, probe)
            else:
                if overwrite == False:
                    print(f"Off df for {subject} {rec} {probe} already exists, skipping")
                    continue
                else:
                    save_off_indexes_and_df(subject, rec, probe)
    return

def _save_spoff_df(odf, subject, rec, probe):
    save_folder = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/spatial_off/spoff_dfs'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, f'{rec}--{probe}.parquet')
    odf.to_parquet(save_path, version='2.6')

def _save_off_indexes(lbl_ixs, subject, rec, probe):
    save_folder = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/spatial_off/off_indexes'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, f'{rec}--{probe}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(lbl_ixs, f)

def load_spoff_df(subject, rec, probe, span_min=3):
    save_folder = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/spatial_off/spoff_dfs'
    save_path = os.path.join(save_folder, f'{rec}--{probe}.parquet')
    odf = pd.read_parquet(save_path)
    odf['span'] = np.abs(odf['span'])
    odf = odf.loc[odf['span'] >= span_min]
    return odf

def load_off_indexes(subject, rec, probe):
    save_folder = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/spatial_off/off_indexes'
    save_path = os.path.join(save_folder, f'{rec}--{probe}.pkl')
    with open(save_path, 'rb') as f:
        lbl_ixs = pickle.load(f)
    return lbl_ixs

def assign_datetimes_to_spoff(df, subject, rec):
    rec_times = acr.info_pipeline.subject_info_section(subject, 'rec_times')
    start = pd.Timestamp(rec_times[rec]['start'])
    df['start_datetime'] = start + pd.to_timedelta(df['start_time'], unit='s')
    df['end_datetime'] = start + pd.to_timedelta(df['end_time'], unit='s')
    return df

def save_off_indexes_and_df(subject, rec, probe, morpho_ops=None):
    msk = acr.spoff.load_spatial_off_mask(subject, rec, probe)
    msk_clean = acr.spoff.morpho_clean_mask(msk, morpho_ops=morpho_ops)
    lbl_da = msk_clean.copy()
    lbl_img, _ = dask_image.ndmeasure.label(msk_clean)
    lbl_da.data = lbl_img
    lbl_da.name = "OFF label"
    lbl_ixs = scipy.ndimage.value_indices(np.array(lbl_da.data), ignore_value=0) # {lbl: (row/time_indices, col/chan_indices)}
    
    odf = acr.spoff.get_offs_df(msk, lbl_ixs)
    odf['lo_chan'] = np.abs(odf['lo'])
    odf['hi_chan'] = np.abs(odf['hi'])
    odf['subject'] = subject
    odf['recording'] = rec
    odf['probe'] = probe
    odf = assign_datetimes_to_spoff(odf, subject, rec)
    print('saving')
    _save_spoff_df(odf, subject, rec, probe)
    _save_off_indexes(lbl_ixs, subject, rec, probe)
    return

def add_states_to_spoff(odf, h):
    odf['state'] = 'NA'
    for bout in h.keep_states(['NREM']).itertuples():
        odf.loc[(odf.start_time>=bout.start_time) & (odf.end_time<=bout.end_time), 'state'] = 'NREM'
    for bout in h.keep_states(['Wake']).itertuples():
        odf.loc[(odf.start_time>=bout.start_time) & (odf.end_time<=bout.end_time), 'state'] = 'Wake'
    for bout in h.keep_states(['Wake-Good']).itertuples():
        odf.loc[(odf.start_time>=bout.start_time) & (odf.end_time<=bout.end_time), 'state'] = 'Wake-Good'
    for bout in h.keep_states(['REM']).itertuples():
        odf.loc[(odf.start_time>=bout.start_time) & (odf.end_time<=bout.end_time), 'state'] = 'REM'
    return odf

def label_spoff_with_hypno_conditions(oodf, hd):
    if type(oodf) == pd.DataFrame:
        oodf = pl.DataFrame(oodf)
    
    if 'condition' not in oodf.columns:
        oodf = oodf.with_columns(condition=pl.lit('None'))
    oodf_pd = oodf.to_pandas()
    for key in hd.keys():
        for bout in hd[key].itertuples():
            oodf_pd.loc[((oodf_pd['start_datetime'] >= bout.start_time) & (oodf_pd['end_datetime'] <= bout.end_time)), 'condition'] = key
    return pl.DataFrame(oodf_pd) #TODO: optimize for polars

def label_arb_df_with_hypno_conditions(oodf, hd, col='start_datetime'):
    if type(oodf) == pd.DataFrame:
        oodf = pl.DataFrame(oodf)
    
    if 'condition' not in oodf.columns:
        oodf = oodf.with_columns(condition=pl.lit('None'))
    oodf_pd = oodf.to_pandas()
    for key in hd.keys():
        for bout in hd[key].itertuples():
            oodf_pd.loc[((oodf_pd[col] >= bout.start_time) & (oodf_pd[col] <= bout.end_time)), 'condition'] = key
    return pl.DataFrame(oodf_pd) #TODO: optimize for polars


def plot_histos_with_threshes(detect_data, h, t1=0, t2=40000, threshes_to_plot=None, title_color='black', save=False, subject='na', rec='na', probe='na'):
    if threshes_to_plot == None:
        threshes_to_plot = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    ht = h.trim_select(t1, t2)
    plot_dat = detect_data.sel(time=slice(t1, t2))
    t_array = plot_dat.time.values
    nrem_mask = ht.keep_states(['NREM']).covers_time(t_array)
    wake_mask = ht.keep_states(['Wake-Good', 'Wake']).covers_time(t_array)
    
    nrem_dat = plot_dat.sel(time=nrem_mask)
    wake_dat = plot_dat.sel(time=wake_mask)
    
    nrem_thresh_vals = []
    wake_thresh_vals = []
    for t in threshes_to_plot:
        nrem_thresh_vals.append(get_thresholds_entire_da(nrem_dat, quantile=t))
        wake_thresh_vals.append(get_thresholds_entire_da(wake_dat, quantile=t))
    
    nf, nax = _plot_histos_from_da(nrem_dat)
    wf, wax = _plot_histos_from_da(wake_dat)
    
    colors = plt.cm.inferno(np.linspace(0, 0.6, len(threshes_to_plot)))
    sizes = np.linspace(8, 32, len(threshes_to_plot))
    
    #adds the thresholds to the plot
    for i in range(len(threshes_to_plot)):
        
        nrem_thresh_vals[i].plot(ax=nax, y='channel', color=colors[i], marker='+', linewidth=0, markersize=sizes[i], markeredgewidth=4)
        wake_thresh_vals[i].plot(ax=wax, y='channel', color=colors[i], marker='+', linewidth=0, markersize=sizes[i], markeredgewidth=4)

    nax.set_title(f"NREM threshold values: {threshes_to_plot}, t1: {ht.start_time.min()}, t2: {ht.end_time.max()}", color=title_color)
    wax.set_title(f"Wake threshold values: {threshes_to_plot}, t1: {ht.start_time.min()}, t2: {ht.end_time.max()}", color=title_color)
    plt.show()
    if save:
        save_folder = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/spatial_off/plots'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path_nrem = os.path.join(save_folder, f'histo_threshes__{rec}--{probe}_nrem.png')
        save_path_wake = os.path.join(save_folder, f'histo_threshes__{rec}--{probe}_wake.png')
        nf.savefig(save_path_nrem, bbox_inches='tight')
        wf.savefig(save_path_wake, bbox_inches='tight')
    return nax, wax

def _sel_random_bouts(h, n=5, min_dur='60s'):
    h_long = h.loc[h.duration > pd.to_timedelta(min_dur)]
    num_bouts = len(h_long)
    bouts_to_use = np.random.choice(num_bouts, n, replace=False)
    return h_long.iloc[bouts_to_use]

def save_snip_plot(subject, rec, probe, fold_name='random_snips', file_tag='notag'):
    save_folder = f'/Volumes/neuropixel_archive/Data/acr_archive/mua_data/{subject}/spatial_off/snip_plots/{fold_name}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, f'{rec}--{probe}__{file_tag}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return

def generate_off_detect_snippets(subject, exp, folder='random_snips', probes=['NNXo', 'NNXr']):

    mua = acr.mua.load_concat_peaks_df(subject, exp)
    hd = acr.hypnogram_utils.create_acr_hyp_dict(subject, exp)
    
    for probe in probes:
        sort_id = f'{exp}-{probe}'
        recs, starts, durations = acr.units.get_time_info(subject, sort_id)
        for rec, dur in zip(recs, durations):
            rt = acr.info_pipeline.subject_info_section(subject, 'rec_times')
            rec_start = pd.Timestamp(rt[rec]['start'])
            pp_data = acr.spoff.load_spatial_off_prepro_data(subject, rec, probe)
            pp_data = acr.spoff._median_filter_prepro_data(pp_data)
            det_data = acr.spoff.load_spatial_off_mask(subject, rec, probe)
            det_data = acr.spoff.morpho_clean_mask(det_data)
            odf = acr.spoff.load_spoff_df(subject, rec, probe)
            odf = acr.spoff.label_spoff_with_hypno_conditions(odf, hd)
            odf = odf.to_pandas()
            for cond in odf['condition'].unique():
                if cond == 'None':
                    continue
                if len(hd[cond].loc[hd[cond].duration > pd.to_timedelta('60s')]) < 7:
                    continue
                hc = _sel_random_bouts(hd[cond])
                for bout in hc.itertuples():
                    start = bout.start_time
                    end = bout.end_time
                    dur = (end-start).total_seconds()
                    rand_starts = np.random.choice(int(dur-8), 2, replace=False)
                    for rand_start in rand_starts:
                        t1 = start+pd.to_timedelta(rand_start, unit='s')
                        t2 = t1+pd.to_timedelta(8, unit='s')
                        t1f = acr.utils.dt_to_tdt(subject, rec, t1)
                        t2f = acr.utils.dt_to_tdt(subject, rec, t2)
                        f, ax = plt.subplots(figsize=(35, 10))
                        pp_data.ts(t1f, t2f).plot.imshow(ax=ax, x='time', y='y', vmin=0, vmax=1.5, cmap='inferno', interpolation=None)
                        det_data.ts(t1f, t2f).plot.imshow(ax=ax, x='time', y='y', vmin=0, vmax=1, cmap=mpl.colors.ListedColormap(['none', 'navy']), alpha=0.3)
                        snip_mua = mua.ts(t1, t2).prb(probe)
                        dts = snip_mua['datetime'].to_pandas()
                        float_times = (dts-rec_start).dt.total_seconds()
                        snip_mua = snip_mua.to_pandas()
                        snip_mua['tf'] = float_times
                        sns.scatterplot(data=snip_mua, x='tf', y='negchan', s=60, color='#10B100', ax=ax)
                        print(t1f, t2f, len(odf.oots(t1f, t2f)))
                        ax.set_title(f'{subject}, {probe}, {cond}')
                        for offper in odf.oots(t1f, t2f).itertuples():
                            high_y = 1 - (np.abs(offper.lo)*.0625)
                            low_y = 1 - (np.abs(offper.hi)*.0625)
                            low_y = low_y+.0625
                            ax.axvspan(offper.start_time, offper.end_time, ymin=low_y, ymax=high_y, facecolor='none', edgecolor='black', linewidth=4)
                        #save_snip_plot(subject, rec, probe, folder, file_tag=f'{cond}-{t1}-{t2}')


def rerun_entire_detection_pipeline(subject, exp, thresh=None, probes=['NNXo', 'NNXr'], morpho_ops=None):
    "Overwrite everything by default!!, does NOT redo any preprocessing, just the thresholded detection, and generation of the dfs and OFF indexes"
    for probe in probes:
        sort_id = f'{exp}-{probe}'
        recs, starts, durations = acr.units.get_time_info(subject, sort_id)
        if thresh == None:
            thresh = acr.spoff.get_threshold_for_spoff_detection(subject, probe)
        for rec in recs:
            run_full_spatial_off_detection(subject, rec, probe, thresh=thresh, med_filt_samples=10, med_filt_chans=1, overwrite=True, save=True)
            save_off_indexes_and_df(subject, rec, probe, morpho_ops=morpho_ops)
    return


# RELATIVIZING THE DATA

def make_spodf_relative(odf, rel_con='circ_bl', col='median_duration'):
    avgs = odf.group_by(['probe', 'condition']).mean()
    probe_dfs = []
    for probe in odf['probe'].unique():
        probe_df = odf.filter(pl.col('probe')==probe)
        avg = avgs.prb(probe).filter(pl.col('condition')==rel_con)[col].to_numpy()[0]
        rel_vals = probe_df[col].to_numpy()/avg
        probe_df = probe_df.with_columns(rel_val = pl.lit(rel_vals))
        probe_dfs.append(probe_df)
    return pl.concat(probe_dfs)

def compute_frequency_by_bout(sp_odf, hd):
    master_freq = pd.DataFrame()
    for condition in hd.keys():
        for probe in ['NNXo', 'NNXr']:
            for i, bout in enumerate(hd[condition].itertuples()):
                duration = (bout.end_time - bout.start_time).total_seconds()
                count = sp_odf.prb(probe).oots(bout.start_time, bout.end_time).count()['area'].to_numpy()[0]
                bout_df = pd.DataFrame({'probe':[probe], 'duration':[duration], 'count':[count], 'condition':[condition], 'bout_num':[i]})
                master_freq = pd.concat([master_freq, bout_df])
    master_freq['frequency'] = master_freq['count']/master_freq['duration']
    return master_freq