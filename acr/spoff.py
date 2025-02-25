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
import spikeinterface as si
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
from kdephys.hypno.ecephys_hypnogram import Hypnogram



# -----------------------------------------------------------------------------
# ===================== GENERAL HELPER FUNCTIONS =============================
# -----------------------------------------------------------------------------
def get_quantile_thresh(x, quantile):
    return np.quantile(x, quantile)

def _median_filter_prepro_data(data, med_filt_samples=10, med_filt_chans=1):
    smdata = data.copy()
    smdata.data = ndfilters.median_filter(
            smdata.data,
            footprint=np.ones((med_filt_samples, med_filt_chans)),)
    return smdata

def get_threshold_for_spoff_detection(subject, probe):
    from acr.utils import NREM_QUANTILE_THRESHOLDS
    return NREM_QUANTILE_THRESHOLDS[(subject, probe, None)]

# -----------------------------------------------------------------------------
# ===================== PREPROCESSING DATA ==================================
# -----------------------------------------------------------------------------

def get_full_si_time_array(start_times, num_sample_list, srate):
    gaps = np.diff(start_times).astype('timedelta64[s]')
    t_arrays = []
    total_t1 = num_sample_list[0]/srate
    t_array1 = np.linspace(0, total_t1, num_sample_list[0])
    t_arrays.append(t_array1)
    for i, st in enumerate(start_times):
        if i == 0:
            continue
        total_t = num_sample_list[i]/srate
        t_array = np.linspace(0, total_t, num_sample_list[i])
        t_array = t_array + gaps[i-1].astype('float32')
        t_arrays.append(t_array)
    return np.concatenate(t_arrays)

def _load_exp_mua_data_for_spoff(subject, exp, probe):
    sort_id = f'{exp}-{probe}'
    recs, starts, durations = acr.units.get_time_info(subject, sort_id)
    si_objs = []
    for rec in recs:
        si_obj = acr.mua.load_processed_mua_signal(subject, rec, probe, version='si')
        si_objs.append(si_obj)
    num_sample_list = [seg.get_num_samples() for seg in si_objs]
    fullt = get_full_si_time_array(starts, num_sample_list, si_objs[0].get_sampling_frequency())
    return si.concatenate_recordings(si_objs), fullt

def _prepro_spatial_off_data(sig, full_time_array, dec_factor=100, fmax=20):
    offsig = sp.zscore(sig, dtype='float32')
    offsig = sp.rectify(offsig)
    offsig = sp.gaussian_filter(offsig, freq_min=None, freq_max=fmax)
    offsig = sp.decimate(offsig, decimation_factor=dec_factor)
    
    time_vector_dec = full_time_array[::dec_factor]
    offsig.set_times(time_vector_dec) #MUST manually set this here!
    return offsig

def save_spoff_prepro_data(sig, subject, exp, probe, overwrite=True, hard_path=None):
    base_folder = f'{raw_data_root}mua_data/{subject}/spatial_off'
    if os.path.exists(base_folder) == False:
        os.mkdir(base_folder)
    exp_folder = f'{base_folder}/{exp}'
    if os.path.exists(exp_folder) == False:
        os.mkdir(exp_folder)
    save_path = f'{exp_folder}/prepro__{exp}--{probe}.zarr'
    if os.path.exists(save_path):
        if overwrite == False:
            print(f'File already exists at {save_path}')
            return
        else:
            os.system(f'rm -rf {save_path}')
    if hard_path is not None:
        save_path = hard_path
    job_kwargs = dict(
        n_jobs=128, chunk_duration=f"{100}s", progress_bar=True
    )
    sig.save(folder=save_path, overwrite=True, format="zarr", **job_kwargs)
    return

def _prepro_and_save_spoff_exp(subject, exp, probe, dec_factor=100, fmax=20, overwrite=True):
    fullsi, fullt = _load_exp_mua_data_for_spoff(subject, exp, probe)
    offsig = _prepro_spatial_off_data(fullsi, fullt, dec_factor, fmax)
    save_spoff_prepro_data(offsig, subject, exp, probe, overwrite=overwrite)
    return

def prepro_and_save_spoff_exp(subject, exp, probes=['NNXo', 'NNXr'], dec_factor=100, fmax=20, overwrite=True):
    for probe in probes:
        _prepro_and_save_spoff_exp(subject, exp, probe, dec_factor=dec_factor, fmax=fmax, overwrite=overwrite)
    return

def load_spoff_prepro(subject, exp, probe):
    data_path = f'{raw_data_root}mua_data/{subject}/spatial_off/{exp}/prepro__{exp}--{probe}.zarr'
    data = kde.xr.utils.load_processed_zarr_as_xarray(data_path, times=True) #channel and time dims will be assigned here, need times=True
    
    data = data.assign_coords(y = ("channel", data.channel.values*-1))
    return data

# -----------------------------------------------------------------------------
# ===================== RUNNING THE ACTUAL DETECTION ========================
# -----------------------------------------------------------------------------

def _save_cond_masks(subject, exp, data_masks):
    data_path = f'{raw_data_root}mua_data/{subject}/spatial_off/{exp}/cond_masks.pkl'
    if os.path.exists(data_path):
        os.remove(data_path)
    with open(data_path, 'wb') as f:
        pickle.dump(data_masks, f, protocol=pickle.HIGHEST_PROTOCOL)
    return

def gen_and_save_cond_masks(subject, exp, probe='NNXr'):
    data = load_spoff_prepro(subject, exp, probe)
    times = data.time.values
    data_masks = {}
    hd = acr.hypnogram_utils.create_acr_hyp_dict(subject, exp, float_hd=True, update=True)
    for key in hd.keys():
        data_masks[key] = hd[key].covers_time(times)
    _save_cond_masks(subject, exp, data_masks)
    return

def load_cond_masks(subject, exp):
    data_path = f'{raw_data_root}mua_data/{subject}/spatial_off/{exp}/cond_masks.pkl'
    if not os.path.exists(data_path):
        gen_and_save_cond_masks(subject, exp)
    with open(data_path, 'rb') as f:
        data_masks = pickle.load(f)
    return data_masks

def _save_cond_off_mask(subject, exp, probe, cond, off_mask):
    data_path = f'{raw_data_root}mua_data/{subject}/spatial_off/{exp}/mask__{exp}--{probe}--{cond}.zarr'
    if os.path.exists(data_path):
        os.system(f'rm -rf {data_path}')
    off_mask.to_zarr(data_path)
    return

def _save_spoff_df(subject, exp, probe, cond, odf,):
    save_path = f'{raw_data_root}mua_data/{subject}/spatial_off/{exp}/df__{exp}--{probe}--{cond}.parquet'
    if os.path.exists(save_path):
        os.system(f'rm -rf {save_path}')
    odf.to_parquet(save_path, version='2.6')
    return

def _save_off_lbl_ixs(subject, exp, probe, cond, lbl_ixs):
    save_path = f'{raw_data_root}mua_data/{subject}/spatial_off/{exp}/lbl_ixs__{exp}--{probe}--{cond}.pkl'
    if os.path.exists(save_path):
        os.system(f'rm -rf {save_path}')
    with open(save_path, 'wb') as f:
        pickle.dump(lbl_ixs, f)
    return

def load_cond_off_mask(subject, exp, probe, cond):
    data_path = f'{raw_data_root}mua_data/{subject}/spatial_off/{exp}/mask__{exp}--{probe}--{cond}.zarr'
    data = xr.open_zarr(data_path)
    return data['OFF mask']

def _gen_threshes(prepro_data, thresh=0.25):
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
    return threshes

def _gen_mask(prepro_data, thresh=0.25, threshes=None):
    if threshes is None:
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

def _gen_indices_and_df(off_mask, subject, exp, probe, cond):
    msk_clean = morpho_clean_mask(off_mask, morpho_ops=None)
    lbl_da = msk_clean.copy()
    lbl_img, _ = dask_image.ndmeasure.label(msk_clean)
    lbl_da.data = lbl_img
    lbl_da.name = "OFF label"
    lbl_ixs = scipy.ndimage.value_indices(np.array(lbl_da.data), ignore_value=0) # {lbl: (row/time_indices, col/chan_indices)}
    
    odf = get_offs_df(off_mask, lbl_ixs)
    odf['lo_chan'] = np.abs(odf['lo'])
    odf['hi_chan'] = np.abs(odf['hi'])
    odf['subject'] = subject
    odf['probe'] = probe
    odf['condition'] = cond
    odf['exp'] = exp
    return odf, lbl_ixs

def _run_full_detection_pipeline(subject, exp, probe, med_filt_samples=10, med_filt_chans=1, save_masks=True, save_dfs=True, regen_masks=False, thresh=None):
    """runs the full detection pipeline; works on the assumption that there is already preprocessed data for the entire experiment.

    Parameters
    ----------
    subject : _type_
        _description_
    exp : _type_
        _description_
    probe : _type_
        _description_
    med_filt_samples : int, optional
        _description_, by default 10
    med_filt_chans : int, optional
        _description_, by default 1
    save_masks : bool, optional
        _description_, by default True
    save_dfs : bool, optional
        _description_, by default True
    regen_masks : bool, optional
        _description_, by default False
    thresh : _type_, optional
        _description_, by default None
    """
    #load and median filter the data
    data = load_spoff_prepro(subject, exp, probe)
    data_sm = _median_filter_prepro_data(data, med_filt_samples=med_filt_samples, med_filt_chans=med_filt_chans)
    if thresh is None:
        thresh = get_threshold_for_spoff_detection(subject, probe)
    
    #get the condition masks used to select the data for detection within each condition.
    if regen_masks:
        gen_and_save_cond_masks(subject, exp, probe=probe)
    cond_masks = load_cond_masks(subject, exp)
    conds = cond_masks.keys()
    
    #loop through each condition
    for cond in conds:
        mask = cond_masks[cond]
        data_cond = data_sm.sel(time=mask) #select the data for this condition
        
        #Generate and save the mask
        off_mask_cond = _gen_mask(data_cond, thresh=thresh)
        off_mask_cond.data = off_mask_cond.data.rechunk()
        if save_masks:
            _save_cond_off_mask(subject, exp, probe, cond, off_mask_cond)
        
        #Generate and save the df and label indices
        odf, lbl_ixs = _gen_indices_and_df(off_mask_cond, subject, exp, probe, cond)
        if save_dfs:
            _save_spoff_df(subject, exp, probe, cond, odf)
            _save_off_lbl_ixs(subject, exp, probe, cond, lbl_ixs)
    return

def run_full_detection_pipeline(subject, exp, probes=['NNXo', 'NNXr'], **kwargs):
    for probe in probes:
        _run_full_detection_pipeline(subject, exp, probe, **kwargs)
    return


def _load_cond_df(subject, exp, probe, cond):
    data_path = f'{raw_data_root}mua_data/{subject}/spatial_off/{exp}/df__{exp}--{probe}--{cond}.parquet'
    if not os.path.exists(data_path):
        raise ValueError(f'{data_path} does not exist')
    df = pl.read_parquet(data_path)
    return df

def get_conds_from_folder(subject, exp):
    data_path = f'{raw_data_root}mua_data/{subject}/spatial_off/{exp}'
    conds = [f.split('--')[2].split('.parquet')[0] for f in os.listdir(data_path) if f.startswith('df__')]
    conds = np.unique(conds)
    return conds

def add_datetime_to_df(df, subject, exp):
    recs, starts, durations = acr.units.get_time_info(subject, f'{exp}-NNXo')
    ref_start = pd.Timestamp(starts[0])
    start_times = pd.to_timedelta(df['start_time'].to_numpy(), unit='s') + ref_start
    end_times = pd.to_timedelta(df['end_time'].to_numpy(), unit='s') + ref_start
    df = df.with_columns(start_datetime=pl.Series(start_times))
    df = df.with_columns(end_datetime=pl.Series(end_times))
    return df

def load_exp_spoff_df(subject, exp, probes=['NNXo', 'NNXr']):
    dfs = []
    for cond in get_conds_from_folder(subject, exp):
        for probe in probes:
            dfs.append(_load_cond_df(subject, exp, probe, cond))
    full_df = pl.concat(dfs)
    return add_datetime_to_df(full_df, subject, exp)

# -----------------------------------------------------------------------------
# === Concatenated Thresholding (no condition-specific detection) ============
# -----------------------------------------------------------------------------

def _convert_hypno_to_concat_time(h, exp_start):
    h['start_time'] = (h['start_time'] - exp_start).dt.total_seconds()
    h['end_time'] = (h['end_time'] - exp_start).dt.total_seconds()
    h['duration'] = h['end_time'] - h['start_time']
    return Hypnogram(h._df)

def _save_concat_off_mask(subject, exp, probe, off_mask):
    data_path = f'{raw_data_root}mua_data/{subject}/spatial_off/{exp}/CONCAT_mask__{exp}--{probe}.zarr'
    if os.path.exists(data_path):
        os.system(f'rm -rf {data_path}')
    off_mask.to_zarr(data_path)
    return

def _save_spoff_df_concat(subject, exp, probe, odf,):
    save_path = f'{raw_data_root}mua_data/{subject}/spatial_off/{exp}/CONCAT_df__{exp}--{probe}.parquet'
    if os.path.exists(save_path):
        os.system(f'rm -rf {save_path}')
    odf.to_parquet(save_path, version='2.6')
    return

def _save_off_lbl_ixs_concat(subject, exp, probe, lbl_ixs):
    save_path = f'{raw_data_root}mua_data/{subject}/spatial_off/{exp}/CONCAT_lbl_ixs__{exp}--{probe}.pkl'
    if os.path.exists(save_path):
        os.system(f'rm -rf {save_path}')
    with open(save_path, 'wb') as f:
        pickle.dump(lbl_ixs, f)
    return

def _run_full_detection_pipeline_concat(subject, exp, probe, med_filt_samples=10, med_filt_chans=1, save_masks=True, save_dfs=True, regen_masks=False, thresh=None, thresh_on='full'):
    """runs the full detection pipeline; works on the assumption that there is already preprocessed data for the entire experiment.

    Parameters
    ----------
    subject : _type_
        _description_
    exp : _type_
        _description_
    probe : _type_
        _description_
    med_filt_samples : int, optional
        _description_, by default 10
    med_filt_chans : int, optional
        _description_, by default 1
    save_masks : bool, optional
        _description_, by default True
    save_dfs : bool, optional
        _description_, by default True
    regen_masks : bool, optional
        _description_, by default False
    thresh : _type_, optional
        _description_, by default None
    """
    #load and median filter the data
    data = load_spoff_prepro(subject, exp, probe)
    data_sm = _median_filter_prepro_data(data, med_filt_samples=med_filt_samples, med_filt_chans=med_filt_chans)
    if thresh is None:
        thresh = get_threshold_for_spoff_detection(subject, probe)
    
    
    #Get the NREM Masks used to select the data for thresholding
    recs, starts, durations = acr.units.get_time_info(subject, f'{exp}-{probe}')
    exp_start = pd.Timestamp(starts[0])
    if thresh_on == 'full':
        h = acr.io.load_hypno_full_exp(subject, exp, update=False, float=False)
        h = _convert_hypno_to_concat_time(h, exp_start)
        times_to_mask = data_sm.time.values
        nrem_mask = h.covers_time(times_to_mask)
        data_nrem = data_sm.sel(time=nrem_mask) #NREM-only data with which we can compute the thresholding.
        threshes = _gen_threshes(data_nrem, thresh=thresh)
    elif thresh_on == 'early_bl':
        hd = acr.hypnogram_utils.create_acr_hyp_dict(subject, exp, float_hd=True)
        ebl_h = hd['early_bl']
        times_to_mask = data_sm.time.values
        nrem_mask = ebl_h.covers_time(times_to_mask)
        data_nrem = data_sm.sel(time=nrem_mask)
        threshes = _gen_threshes(data_nrem, thresh=thresh)
    else:
        print('Not implemented yet')
        return
    
    off_mask = _gen_mask(data_sm, thresh=thresh, threshes=threshes)
    off_mask.data = off_mask.data.rechunk()
    if save_masks:
        _save_concat_off_mask(subject, exp, probe, off_mask)
    
    #Generate and save the df and label indices
    odf, lbl_ixs = _gen_indices_and_df(off_mask, subject, exp, probe, cond='None')
    if save_dfs:
        _save_spoff_df_concat(subject, exp, probe, odf)
        _save_off_lbl_ixs_concat(subject, exp, probe, lbl_ixs)
    return

def run_full_detection_pipeline_concat(subject, exp, probes=['NNXo', 'NNXr'], thresh_on='full', **kwargs):
    for probe in probes:
        _run_full_detection_pipeline_concat(subject, exp, probe, thresh_on=thresh_on, **kwargs)
    return

def load_concat_spoff_df(subject, exp, probes=['NNXo', 'NNXr']):
    dfs = []
    for probe in probes:
        path = f'{raw_data_root}mua_data/{subject}/spatial_off/{exp}/CONCAT_df__{exp}--{probe}.parquet'
        df = pl.read_parquet(path)
        dfs.append(df)
    full_df = pl.concat(dfs)
    return add_datetime_to_df(full_df, subject, exp)