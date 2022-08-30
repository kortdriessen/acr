import numpy as np
import scipy
import pandas as pd
import tdt
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import hypnogram as hp
import kd_analysis.main.kd_utils as kd
import kd_analysis.main.kd_plotting as kp
import kd_analysis.main.kd_hypno as kh
import kd_analysis.paxilline.pax_fin as kpx

bp_def = dict(sub_delta=(0.5, 2), delta=(0.5, 4), theta=(4, 8), alpha = (8, 13), sigma = (11, 16), beta = (13, 30), low_gamma = (30, 55), high_gamma = (65, 90), omega=(300, 700))

def simple_bp_lineplot(bp,
                     ax,
                     ss=12,
                     color='k',
                     linewidth = 2, 
                     hyp=None):
    """ 
    This is just a plotting function, does not do any calculation except, if ss 
    is specified, smooth a copy of the bandpower array for display purposes
    bp --> single channel bandpower data, xr.dataarray

    """
    if ss:
        bp = kd.get_smoothed_da(bp, smoothing_sigma=ss)
    ax = sns.lineplot(x=bp.datetime, y=bp, color=color, linewidth=linewidth, ax=ax)
    if hyp is not None:
        kp.shade_hypno_for_me(hypnogram=hyp, ax=ax)
    return ax

"""
MAIN PLOT #1 --> DELTA-BP AS % OF BASELINE OVER COURSE OF ENTIRE REBOUND
------------------------------------------------------------------------
"""
def simple_shaded_bp(bp,
                     hyp,
                     ax,
                     ss=12,
                     color='k',
                     linewidth = 2):
    """ 
    This is just a plotting function, does not do any calculation except, if ss 
    is specified, smooth a copy of the bandpower array for display purposes"""
    if ss:
        bp = kd.get_smoothed_da(bp, smoothing_sigma=ss)
    ax = sns.lineplot(x=bp.datetime, y=bp, color=color, linewidth=linewidth, ax=ax)
    kp.shade_hypno_for_me(hypnogram=hyp, ax=ax)
    return ax

def get_bp_rel(data, comp, comp_hyp, comp_state):
    if comp_state is not None:
        comp = kh.keep_states(comp, comp_hyp, comp_state)
    
    data_bp = kd.get_bp_set2(data, bp_def)
    comp_bp = kd.get_bp_set2(comp, bp_def)
    
    comp_mean = comp_bp.mean(dim='datetime')
    
    data_rel = (data_bp/comp_mean)*100
    
    return data_rel

def bp_pair_plot(bp1,
                 bp2,
                 h1,
                 h2,
                 names=['name1', 'name2'],
                 ylim = False):
    
    chans = bp1.channel.values
    fig_height = (10/3)*len(chans)
    fig, axes = plt.subplots(nrows=len(chans), ncols=2, figsize=(40, fig_height), sharex='col', sharey='row')
    
    lx1 = fig.add_subplot(111, frameon=False)
    lx1.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    lx1.set_ylabel("Delta Power as % of Baseline NREM Mean", fontsize=16, fontweight='bold')
    
    for chan in chans:
        simple_shaded_bp(bp1.sel(channel=chan), h1, axes[chan-1, 0])
        axes[chan-1, 0].set_title(names[0]+', Ch-'+str(chan), fontweight='bold')
        axes[chan-1, 0].set_ylabel(' ')
        axes[chan-1, 0].set_ylim(0, 500)
        axes[chan-1, 0].axhline(y=100, linestyle='--', linewidth=1.5, color='k')
        if ylim is not False:
            axes[chan-1, 0].set_ylim(0, ylim)
        simple_shaded_bp(bp2.sel(channel=chan), h2, axes[chan-1, 1], color='royalblue')
        axes[chan-1, 1].set_title(names[1]+', Ch-'+str(chan), fontweight='bold', color='darkblue')
        axes[chan-1, 1].axhline(y=100, linestyle='--', linewidth=1.5, color='royalblue')
        
        #plt.subplots_adjust(wspace=0, hspace=0.25)
    return fig, axes 

def bp_plot_set(x, spg, hyp, time='4-Hour', ylim=False):
    # define names
    bl1 = x[0]+'-bl'
    bl2 = x[1]+'-bl'
    
    #Get relative bandpower states
    bp_rel1 = get_bp_rel(spg[x[0]], spg[bl1], hyp[bl1], ['NREM']).delta
    bp_rel2 = get_bp_rel(spg[x[1]], spg[bl2], hyp[bl2], ['NREM']).delta

    # Plot
    fig, axes = bp_pair_plot(bp_rel1, bp_rel2, hyp[x[0]], hyp[x[1]], names=x, ylim=ylim)
    plt.tight_layout(pad=2, w_pad=0)
    fig.suptitle(spg['sub']+', '+x[2]+' | '+spg['dtype']+' | Sleep Rebound as % of Baseline | Delta Bandpower (0.5-4Hz) | '+time+' Rebound', x=0.52, y=1, fontsize=20, fontweight='bold')
    #plt.savefig('/Volumes/paxilline/Data/paxilline_project_materials/fin_plots_all/DELTA_BP-'+spg['sub']+'--'+x[0]+x[1]+'--'+spg['dtype']+'--'+spg['x-time']+'.png', dpi=200)

"------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

def get_state_spectrogram(spg, hyp, state):
    return spg.sel(datetime=hyp.keep_states(state).covers_time(spg.datetime))

def get_state_psd(spg, hyp, state):
    return get_state_spectrogram(spg, hyp, state).median(dim="datetime")

def get_psd_rel(x, spg, hyp):
    # define names
    bl1 = x[0]+'-bl'
    bl2 = x[1]+'-bl'
    
    #calc the PSD's
    x1_psd = get_state_psd(spg[x[0]], hyp[x[0]], ['NREM'])
    x2_psd = get_state_psd(spg[x[1]], hyp[x[1]], ['NREM'])
    
    bl1_psd = get_state_psd(spg[bl1], hyp[bl1], ['NREM'])
    bl2_psd = get_state_psd(spg[bl2], hyp[bl2], ['NREM'])
    
    rel_psd1 = (x1_psd/bl1_psd)*100
    rel_psd2 = (x2_psd/bl2_psd)*100
    
    return rel_psd1, rel_psd2

def n_freq_bins(da, f_range):
    return da.sel(frequency=slice(*f_range)).frequency.size


def auc_bandpowers(psd1, psd2, freqs):
    auc_df = pd.DataFrame()
    for f in kd.get_key_list(freqs):
        auc = kd.compare_auc(psd1, psd2, freqs[f], title=f)
        auc_df[f] = auc
    return auc_df.drop(labels='omega', axis=1)

def pax_scatter_quantal(df, chan, ax):
    ax.plot(df.index, df, color='k', linewidth=2, linestyle='--')
    ax.scatter(df.index, df, s=100, c='royalblue')
    ax.axhspan(ax.get_ylim()[0]-5, ymax=0, color='royalblue', alpha=0.2)
    ax.axhspan(ymin=0, ymax=ax.get_ylim()[1]+5, color='k', alpha=0.2)
    ax.axhline(y=0, color='k', linewidth=2)
    #ax.set_xlabel('Frequency Band')
    #ax.set_ylabel('Paxilline AUC - Saline AUC | Relative to Baselines')
    ax.set_title('Ch-'+str(chan), fontweight='bold')
    return ax

def psd_comp_quantal(psd1, psd2, keys, ax):
    df = pd.concat([psd1.to_dataframe("power"), psd2.to_dataframe("power")], keys=keys).rename_axis(index={None: 'Condition'})
    ax=sns.lineplot(data=df, x='frequency', y='power', hue='Condition', palette=['k', 'royalblue'], ax=ax)
    return ax

def auc_master_plot(x, spg, hyp, time='4-Hour'):
    r1, r2 = get_psd_rel(x, spg, hyp)
    auc_df = auc_bandpowers(r1, r2, bp_def)
    r1_comp = r1.sel(frequency=slice(0,40))
    r2_comp = r2.sel(frequency=slice(0,40))
    chans = r1.channel.values
    fig_height = (25/6)*len(chans)

    fig, axes = plt.subplots(figsize=(30, fig_height), nrows=len(chans), ncols=2, sharex='col')
    for chan in chans:
        psd1 = r1_comp.sel(channel=chan)
        psd2 = r2_comp.sel(channel=chan)
        ax = axes[chan-1, 0]
        ax = psd_comp_quantal(psd1, psd2, x, ax=ax)
        ax.set(ylabel=' ', xlabel=' ')
        ax.set_title('Ch-'+str(chan), fontweight='bold')
    for chan in auc_df.index:
        pax_scatter_quantal(auc_df.iloc[chan-1], chan, axes[chan-1, 1])
    
    
    # This block sets the ylabels for both columns
    lx1 = fig.add_subplot(121, frameon=False)
    lx1.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    lx1.set_ylabel("PSD as % of BL", fontsize=16, fontweight='bold')
    lx1.set_xlabel("Frequency", fontsize=16, fontweight='bold')
    lx2 = fig.add_subplot(122, frameon=False)
    lx2.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    lx2.set_ylabel("Paxilline AUC - Saline AUC | Relative to Baselines", fontsize=16, fontweight='bold')
    
    plt.tight_layout(pad=1.5, w_pad=2)
    fig.suptitle(spg['sub']+', '+x[2]+' | '+spg['dtype']+' | PSD as % of Baseline | '+time+' Rebound', x=0.52, y=1, fontsize=20, fontweight='bold')
    #plt.savefig('/Volumes/paxilline/Data/paxilline_project_materials/fin_plots_all/AUC-'+spg['sub']+'--'+x[0]+x[1]+'--'+spg['dtype']+'--'+spg['x-time']+'.png', dpi=200)
