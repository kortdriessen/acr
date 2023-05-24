import polars as pl
import pandas as pd
import acr
from acr.units import *

def load_spikes_polars(subject, sort_id, cols2drop=None, info=True):
    if cols2drop == None:
        cols2drop = ['group', 'note', 'channel', 'exp', 'recording', 'state', 'amp', 'Amplitude']
    path = f"/Volumes/opto_loc/Data/{subject}/sorting_data/spike_dataframes/"
    sdfs = []
    for sid in sort_id:
        key = sid + ".parquet"
        polars_df = pl.read_parquet(path + key)
        sdfs.append(polars_df)
    spike_df = pl.concat(sdfs)
    spike_df = spike_df.drop(cols2drop)
    
    if info == True:
        idf = load_info_df(subject, sort_id)
        return spike_df, idf
    else:
        return spike_df
    
def load_spikes_polars_lazy(subject, sort_id, cols2drop=None):
    if cols2drop == None:
        cols2drop = ['group', 'note', 'channel', 'exp', 'recording', 'state', 'amp', 'Amplitude']
    path = f"/Volumes/opto_loc/Data/{subject}/sorting_data/spike_dataframes/"
    sdfs = []
    for sid in sort_id:
        key = sid + ".parquet"
        polars_df = pl.scan_parquet(path + key)
        sdfs.append(polars_df)
    spike_df = pl.concat(sdfs)
    spike_df = spike_df.drop(cols2drop)
    return spike_df

def get_state_fr(df, hyp, t1=None, t2=None, state='NREM'):
    """gets the firing rate for each cluster during a specified state

    Args:
        df (dataframe): spike dataframe
        hyp (dataframe): hypnogram
        t1 (datetime, optional): if not specified, the first 12 hours from the start of the first spike are used. Defaults to None.
        t2 (datetime, optional): if not specified, the first 12 hours from the start of the first spike are used. Defaults to None.
        state (str, optional): state to use. Defaults to 'NREM'.

    Returns:
        frs: dictionary of the firing rates for each cluster in all bouts of the state.
    """
    if t1 is None and t2 is None:
        t1 = df['datetime'].to_pandas().min()
        t2 = t1 + pd.Timedelta('12h')
    
    hyp_state = hyp.loc[hyp.state == state]
    hyp_state = hyp_state.loc[hyp_state.end_time < t2]

    fr_master = pl.DataFrame()

    for bout in hyp_state.itertuples(): # for each bout of the specified state get the firing rate
        start = bout.start_time
        end = bout.end_time
        bout_duration = bout.duration.total_seconds()
        spikes = df.ts(start, end).groupby(['probe', 'cluster_id']).count()
        fr_bout = spikes.with_columns((pl.col('count')/bout_duration).alias('fr')).drop('count')
        fr_master = pl.concat([fr_master, fr_bout])
    
    return fr_master
    