def get_bad_channels(subject, exp):
    ex_path = f'{materials_root}bad_channels.xlsx'
    ex= pd.read_excel(ex_path)
    ex.dropna(inplace=True)
    bad_chans = {}
    if subject in ex['subject'].unique():
        if exp in ex.sbj(subject)['exp'].unique():
            stores = ex.sbj(subject).expmt(exp)['store'].unique()
            for store in stores:
                dead_chans = ex.sbj(subject).expmt(exp).prb(store)['dead_channels'].values[0]
                if dead_chans == '-':
                    continue
                elif type(dead_chans) == str:
                    dead_chans = dead_chans.split(',')
                    dead_chans = [int(chan) for chan in dead_chans]
                    bad_chans[store] = dead_chans
                elif type(dead_chans) == int:
                    bad_chans[store] = [dead_chans]
                else:
                    continue
        return bad_chans
    return []

def nuke_bad_chans_from_xrds(ds, subject, exp, which='both', probe=None):
    bad_chans = get_channels_to_exlcude(subject, exp, which=which, probe=probe)
    if len(bad_chans) == 0:
        return ds
    stores_with_bad_chans = bad_chans.keys()
    if 'NNX' in probe:
        if type(ds) == xr.Dataset:
            for var_name in ds.data_vars:
                for channel in bad_chans['NNX']:
                    if 'store' in ds.dims:
                        ds[var_name].loc[{'channel': channel, 'store': probe}] = np.nan
                    elif 'store' not in ds.dims:
                        ds[var_name].loc[{'channel': channel}] = np.nan
        elif type(ds) == xr.DataArray:
            for channel in bad_chans['NNX']:
                if 'store' in ds.dims:
                    ds.loc[{'channel': channel, 'store': probe}] = np.nan
                elif 'store' not in ds.dims:
                    ds.loc[{'channel': channel}] = np.nan
    if probe == None:
        if type(ds) == xr.Dataset:
            for var_name in ds.data_vars:
                for channel in bad_chans['NNX']:
                        ds[var_name].loc[{'channel': channel}] = np.nan
        elif type(ds) == xr.DataArray:
            for channel in bad_chans['NNX']:
                ds.loc[{'channel': channel}] = np.nan
    else:
        raise ValueError('Probe not recognized')
    return ds
        
    
    
    if 'store' not in ds.dims:
        stores = [str(ds.store.values)]
        assert len(stores) == 1
    elif 'store' in ds.dims:
        stores = ds.store.values
    else:
        print('No store dimension or coordinate in dataset')
        return ds
    for store in stores:
        if store in stores_with_bad_chans:
            for chan in bad_chans[store]:
                if chan not in ds.channel.values:
                    print(f'Channel {chan} not in {store} for {subject} {exp}')
                    continue
                if type(ds) == xr.Dataset:
                    for var_name in ds.data_vars:
                        if 'store' in ds.dims:
                            ds[var_name].loc[{'channel': chan, 'store': store}] = np.nan
                        elif 'store' not in ds.dims:
                            ds[var_name].loc[{'channel': chan}] = np.nan
                elif type(ds) == xr.DataArray:
                    if 'store' in ds.dims:
                        ds.loc[{'channel': chan, 'store': store}] = np.nan
                    elif 'store' not in ds.dims:
                        ds.loc[{'channel': chan}] = np.nan
    return ds