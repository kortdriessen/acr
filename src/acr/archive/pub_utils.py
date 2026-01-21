import acr
from acr.utils import *
import matplotlib.pyplot as plt
import polars as pl
import pingouin as pg

def save_fig(figure_path, pad=0.2, dpi=600):
    plt.tight_layout(pad=pad)
    plt.savefig(figure_path, dpi=dpi)
    return
def write_plot_info(f, ax, figure_path, stats=None):
    txt_path = figure_path.replace('.svg', '_info.txt')
    with open(txt_path, 'w') as f:
        f.write(f'{figure_path}\n')
        if stats is not None:
            f.write(f'p-val: {stats["p-val"]}\n')
        elif stats is None:
            f.write('no stats\n')
        # write the y axis limits and tick values
        f.write(f'ylim: {ax.get_ylim()}\n')
        f.write(f'ytick: {ax.get_yticks()}\n')
        f.write(f'yticklabels: {ax.get_yticklabels()}\n')
        f.write(f'xlim: {ax.get_xlim()}\n')
        f.write(f'xtick: {ax.get_xticks()}\n')
        f.write(f'xticklabels: {ax.get_xticklabels()}\n')
    return 

def swa_boxplot(nnxr, nnxo, stats=False):
    f, ax = plt.subplots(1, 1, figsize=(3.5, 5))

    # boxplots
    box = ax.boxplot(nnxr, positions=[0.35], widths=0.06, patch_artist=True, capprops=dict(color='none', linewidth=0), whiskerprops=dict(color='k', linewidth=3), medianprops=dict(color='k', linewidth=3, zorder=101), showfliers=False)
    box['boxes'][0].set_facecolor(NNXR_GRAY)
    box['boxes'][0].set_alpha(0.8)
    box['boxes'][0].set_linewidth(0)
    box['boxes'][0].set_zorder(100)

    box_o = ax.boxplot(nnxo, positions=[0.65], widths=0.06, patch_artist=True, capprops=dict(color='none', linewidth=0), whiskerprops=dict(color='black', linewidth=3), medianprops=dict(color='k', linewidth=3, zorder=101), showfliers=False)
    box_o['boxes'][0].set_facecolor(SOM_BLUE)
    box_o['boxes'][0].set_alpha(0.8)
    box_o['boxes'][0].set_linewidth(0)
    box_o['boxes'][0].set_zorder(100)

    # line plots
    for i in range(len(nnxr)):
        ax.plot([0.40, 0.6], [nnxr[i], nnxo[i]], color=SOM_BLUE, alpha=0.85, linewidth=3.5, solid_capstyle='round', solid_joinstyle='round')

    # scatter plots for individual points
    for i in range(len(nnxr)):
        ax.scatter(0.4, nnxr[i], color=NNXR_GRAY, alpha=0.7, s=30, zorder=202)
        ax.scatter(0.6, nnxo[i], color=SOM_BLUE, alpha=0.7, s=30, zorder=203)
    if stats:
        stats = pg.ttest(nnxr, nnxo, paired=True)
        return f, ax, stats
    else:
        return f, ax

def drop_sub_channels(df, sub_channel_dict):
    # drop any rows containing BOTH the subject and channel
    # drop only rows where the subject matches subject AND channel matches channel
    for sub, channels in sub_channel_dict.items():
        df = df.filter(~((pl.col('subject') == sub) & (pl.col('channel').is_in(channels))))
    return df

def sub_regions_to_df(df):
    locs = acr.utils.sub_probe_locations
    df['region'] = 'none'
    for sub in df['subject'].unique():
        df.loc[df['subject'] == sub, 'region'] = locs[sub]
    return df