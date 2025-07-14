import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import gridspec
import seaborn as sns




def power_per_hour(base_path, condition):

    # Dataframe for label and Band
    stage_file = f'{base_path}EEGEMG/Compile/{condition}/{condition}_staging.xlsx'
    stage_dict = pd.read_excel(stage_file, usecols=[3], skiprows=2, sheet_name=None, header=None)
    stage_df = pd.concat({k: v.dropna().squeeze() for k, v in stage_dict.items()}, axis=1)

    band_file = f'{base_path}EEGEMG/Compile/{condition}/{condition}_EEG.xlsx' 
    band_dict = pd.read_excel(band_file, usecols=list(range(3, 11)), sheet_name=None, header=0)
    delta_dict = pd.read_excel(band_file, usecols=[45], sheet_name=None, header=0)
    
    W_all = {}
    NR_all = {}
    R_all = {}
    delta_NR_all = {}
    frequencies = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

    for mouse_id in band_dict.keys():
        if mouse_id not in stage_df.columns:
            continue

        band_df = band_dict[mouse_id]
        band_df.columns = frequencies
        delta_df = delta_dict[mouse_id]
        label_series = stage_df[mouse_id].dropna()
        label_numeric = label_series.map({'W': 1, 'NR': 2, 'R': 3}).fillna(0).astype(int)

        W_list = []
        NR_list = []
        R_list = []
        delta_NR_list = []

        for hour_start in range(0, len(band_df), epoch_per_1h):
            hour_end = hour_start + epoch_per_1h
            hour_band = band_df.iloc[hour_start:hour_end]
            hour_delta = delta_df.iloc[hour_start:hour_end]
            hour_label = label_numeric.iloc[hour_start:hour_end]

            for label_val, target_list in zip([1, 2, 3], [W_list, NR_list, R_list]):
                label_mask = hour_label == label_val

                if label_mask.sum() > 0:
                    label_band_avg = hour_band[label_mask.values].mean()
                else:
                    label_band_avg = pd.Series(np.nan, index=band_df.columns)
                target_list.append(label_band_avg)

            label_mask = hour_label == 2  # NR only
            if label_mask.sum() > 0:
                delta_values = pd.to_numeric(hour_delta['Delta'], errors='coerce')
                delta_avg = delta_values[label_mask.values].mean()
            else:
                delta_avg = np.nan

            delta_NR_list.append(delta_avg)
        
        W_all[mouse_id] = pd.DataFrame(W_list)
        NR_all[mouse_id] = pd.DataFrame(NR_list)
        R_all[mouse_id] = pd.DataFrame(R_list)
        delta_NR_all[mouse_id] = pd.Series(delta_NR_list)

    return W_all, NR_all, R_all, delta_NR_all



def calculate_bsl(stage_all):

    bsl_dict = {}
    bsl_days = 2

    for mouse_id, df in stage_all.items():

        daily_avg = []
        for hour in range(rows_per_day):

            rows_to_average = [hour + i * rows_per_day for i in range(bsl_days)]
            avg_row = df.iloc[rows_to_average].mean()
            daily_avg.append(avg_row)

        bsl_df = pd.DataFrame(daily_avg)
        if len(stage_all) % rows_per_day != 0:
            bsl_df = pd.concat([bsl_df.iloc[12:], bsl_df.iloc[:12]], ignore_index=True)
        bsl_df = pd.concat([bsl_df] * 2, ignore_index=True) 
        bsl_df.index = range(rows_per_day*bsl_days)
        bsl_dict[mouse_id] = bsl_df
    
    return bsl_dict



def calculate_post(stage_all):

    post_dict = {}
    post_days = 2

    for mouse_id, df in stage_all.items():

        post_df = pd.DataFrame(df.iloc[-rows_per_day*post_days:])
        post_df.index = range(rows_per_day*post_days)
        post_dict[mouse_id] = post_df

    return post_dict



def calculate_relative_swa(bsl_dict, post_dict):

    relative_swa_dict = {}

    for mouse_id in bsl_dict:

        bsl_df = bsl_dict[mouse_id]
        post_df = post_dict[mouse_id]

        relative_swa = (post_df / bsl_df) - 1
        relative_swa_dict[mouse_id] = relative_swa

    return relative_swa_dict



def fig_setting(ax, df, cmap, fontsize, cbar_ax):

    sns.heatmap(df.T, cmap=cmap, center=0, ax=ax, cbar=True, cbar_ax=cbar_ax)
    ax.set_xlabel('Time (h)', fontsize=fontsize)
    ax.set_ylabel('Frequency (Hz)', fontsize=fontsize)
    ax.set_xticks(np.arange(0, 49, 12))
    ax.set_xticklabels(np.arange(0, 49, 12), fontsize=fontsize)
    ax.set_yticks(np.arange(0, 9, 8))
    ax.set_yticklabels(np.arange(0, 5, 4), fontsize=fontsize, rotation=0)
    ax.invert_yaxis()



def add_ld_annotation(ax):

    ylim = ax.get_ylim()
    ymax = ylim[1]
    band_height = (ylim[1] - ylim[0]) * 0.05
    band_y = ylim[1] + band_height * 0.2

    for start in [0, 24]:
        rect = Rectangle((start, band_y), 12, band_height, facecolor='gray', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=3, transform=ax.transData, clip_on=False)
        ax.add_patch(rect)
    for start in [12, 36]:
        rect = Rectangle((start, band_y), 12, band_height, facecolor='white', edgecolor='black', linewidth=1.5, alpha=0.5, zorder=3, transform=ax.transData, clip_on=False)
        ax.add_patch(rect)



def relative_swa_spectrum_visualization(relative_swa_dict):

    stacked_list = []
    for mouse_id, df in relative_swa_dict.items():
        df['time'] = range(48)
        melted = df.melt(id_vars='time', var_name='freq', value_name='change')
        melted['mouse'] = mouse_id
        stacked_list.append(melted)
    stacked_df = pd.concat(stacked_list)

    mean_df = stacked_df.groupby(['time', 'freq'])['change'].mean().unstack()
    sem_df = stacked_df.groupby(['time', 'freq'])['change'].sem().unstack()

    fig_mean, ax1 = plt.subplots(figsize=(14, 6))
    fig_setting(ax1, mean_df, 'bwr', 20, cbar_ax=None)
    add_ld_annotation(ax1)

    fig_sem, ax2 = plt.subplots(figsize=(14, 6))
    fig_setting(ax2, sem_df, 'YlGnBu', 20, cbar_ax=None)
    add_ld_annotation(ax2)

    fig_mean.tight_layout()
    fig_sem.tight_layout()

    return fig_mean, fig_sem, mean_df, sem_df



def mean_swa_fig_setting(x, mean_values, sem_values, ax, fontsize):

    ax.plot(x, mean_values, color='k')
    ax.fill_between(x, mean_values - sem_values, mean_values + sem_values, alpha=0.3, color='k')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylabel('⊿SWA', fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)   

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)



def relative_mean_swa_visualization(delta_relative_swa_dict):

    delta_relative_swa_df = pd.concat(delta_relative_swa_dict.values(), axis=1)
    delta_relative_swa_df.columns = delta_relative_swa_dict.keys()

    mean_values = delta_relative_swa_df.mean(axis=1, skipna=True)
    sem_values = delta_relative_swa_df.sem(axis=1, skipna=True)

    fig_delta, ax = plt.subplots(figsize=(14, 2))
    x = delta_relative_swa_df.index
    mean_swa_fig_setting(x, mean_values, sem_values, ax, 20)

    plt.tight_layout()

    return fig_delta, mean_values, sem_values



def combined_relative_swa_visualization(mean_df, sem_df, mean_values, sem_values):

    fig_combined = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[3, 1], wspace=0.05, hspace=0.3)

    # swa spectrum
    ax_heatmap = fig_combined.add_subplot(gs[0, 0])
    cbar_ax = fig_combined.add_subplot(gs[0, 1])
    fig_setting(ax_heatmap, mean_df, 'bwr', 24, cbar_ax)
    add_ld_annotation(ax_heatmap)

    # mean swa
    ax_line = fig_combined.add_subplot(gs[1, 0])
    x = mean_df.index
    mean_swa_fig_setting(x, mean_values, sem_values, ax_line, 24)
    ax_line.set_xticks([])
    ax_line.set_xticklabels([])

    return fig_combined



base_path = '/mnt/d/Q_project/'
if not os.path.exists(base_path):
    base_path = '/mnt/e/Q_project/'
conditions = ['Ctrl_Stim(24h)', 'QIH(24h)', 'QIH(48h)', 'QIH(72h)', 'SD', 'SD+QIH(1h)', 'SD+QIH(2h)', 'SD+QIH(24h)']
# 'QIH(24h)': 3, 'QIH(48h)': 4, 'QIH(72h)': 5, 'Ctrl_Stim(24h)': 3, 'SD+QIH(48h)': 4

epoch_per_1h = 150
rows_per_day = 24

for condition in conditions:

    W_all, NR_all, R_all, delta_NR_all = power_per_hour(base_path, condition)
    NR_bsl = calculate_bsl(NR_all)
    NR_post = calculate_post(NR_all)
    NR_relative_swa = calculate_relative_swa(NR_bsl, NR_post)

    delta_NR_bsl = calculate_bsl(delta_NR_all)
    delta_NR_post = calculate_post(delta_NR_all)
    delta_relative_swa = calculate_relative_swa(delta_NR_bsl, delta_NR_post)

    fig_mean, fig_sem, mean_df, sem_df = relative_swa_spectrum_visualization(NR_relative_swa)
    fig_delta, mean_values, sem_values = relative_mean_swa_visualization(delta_relative_swa)

    save_path = f'{base_path}/BandAnalysis/{condition}/Power_spectrum'
    fig_mean.savefig(f'{save_path}/relative_swa.svg', format='svg')
    fig_sem.savefig(f'{save_path}/relative_swa_sem.svg', format='svg')
    fig_delta.savefig(f'{save_path}/relative_mean_swa.svg', format='svg')

    fig_combined = combined_relative_swa_visualization(mean_df, sem_df, mean_values, sem_values)
    fig_combined.savefig(f'{save_path}/combined.svg', format='svg')

    print(f'{condition}_SWA spectrum  <Done>')




