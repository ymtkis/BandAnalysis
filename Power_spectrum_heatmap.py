import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec



import sys
from pathlib import Path
btanalysis_path = Path(__file__).resolve().parents[1] / 'BtAnalysis'
sys.path.append(str(btanalysis_path))
from BtAnalysis import generate_all_figures



base_path = '/mnt/d/Q_project/'
if not os.path.exists(base_path):
    base_path = '/mnt/e/Q_project/'
conditions = {'QIH(24h)': 24, 'QIH(48h)': 48, 'QIH(72h)': 72, 'Ctrl_Stim(24h)': 24, 'SD': 6, 'SD+QIH(1h)': 7, 'SD+QIH(2h)': 8, 'SD+QIH(24h)': 30} #, 'SD+Sleep(1h)': 7, 'SD+Sleep(2h)': 8
# values represent the duration of intervention
# 'QIH(24h)','QIH(48h)', 'QIH(72h)', 'Ctrl_Stim(24h)', 'SD', 'SD+QIH(1h)', 'SD+QIH(2h)', 'SD+Sleep(1h)', 'SD+Sleep(2h)'
epoch_per_1h = 150
epoch_per_day = epoch_per_1h * 24



def plot_heatmap(average_df, bt_mean, bt_sem, duration, epoch_per_1h, epoch_per_day):
    plt.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[3, 1], hspace=0.05, wspace=0.05)

    # ----------- Heatmap -----------
    ax1 = fig.add_subplot(gs[0, 0])
    heatmap_data = np.flipud(average_df.T)
    heatmap = sns.heatmap(
        heatmap_data,
        cmap='Spectral_r',
        vmin=0,
        cbar_ax=fig.add_subplot(gs[0, 1]),
        ax=ax1
    )

    xticks = np.arange(0, len(average_df.index) + 1, epoch_per_day)
    yticks = np.arange(0, len(average_df.columns), 10)
    ax1.set_xticks(xticks + 0.5)
    ax1.set_yticks(yticks + 0.5)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([f'{int((len(average_df.columns) - 1 - y) / 2)}' for y in yticks], rotation=0)
    ax1.set_ylabel('Frequency (Hz)')

    # Intervention annotation
    xmin = len(average_df.index) - (epoch_per_1h * duration + epoch_per_day * 2)
    xmax = len(average_df.index) - epoch_per_day * 2
    ax1.hlines(-0.75, xmin, xmax, colors='gray', linestyles='solid', linewidth=16)
    ax1.set_ylim(len(average_df.columns), -1)

    # ----------- Tb plot -----------
    ax0 = fig.add_subplot(gs[1, 0])
    ax0.plot(bt_mean.index, bt_mean, color='k', linewidth=2)
    ax0.fill_between(bt_mean.index, bt_mean - bt_sem, bt_mean + bt_sem, color='k', alpha=0.3)

    xticks = np.arange(0, bt_mean.index.max() + 144, 144) 
    ax0.set_xticks(xticks)
    ax0.set_xticklabels([f'{int(label)}' for label in xticks / 6])
    ax0.set_xlim([0, bt_mean.index.max() + 1])
    ax0.set_ylim([25, 40])
    ax0.set_ylabel('Tb (℃)')
    ax0.set_xlabel('Time (h)')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)

    return fig




def power_per_hour(base_path, condition, duration, epoch_per_1h, epoch_per_day, bt_figs, bt_mean, bt_sem):

    # Dataframe for frequency
    freq_file = f'{base_path}EEGEMG/Compile/{condition}/Normalized_by_delta_{condition}_EEG.xlsx' 
    freq_dict = pd.read_excel(freq_file, usecols=range(3, 44), skiprows=1, sheet_name=None, header=None)
    
    freq_data_dict = {}
    
    for sheet_name, freq_data in freq_dict.items(): 

        freq_data_indiv = freq_data.rename(columns=lambda x: str(x-3))
        freq_data_dict[sheet_name] = freq_data_indiv 
    
    freq_data_df = pd.concat(freq_data_dict.values(), keys=freq_data_dict.keys())
    average_df = freq_data_df.groupby(level=1).mean()

    # outliers
    average_df = np.clip(average_df, -3, 3)

    # Bt object
    bt_ax = bt_figs[condition].axes[0]
    lines = bt_ax.get_lines()
    bt_index = lines[0].get_xdata()
    bt_mean = pd.Series(bt_mean, index=bt_index)
    bt_sem = pd.Series(bt_sem, index=bt_index)

    # Plot
    fig = plot_heatmap(average_df, bt_mean, bt_sem, duration, epoch_per_1h, epoch_per_day)

    # Save figure
    output_path = f'{base_path}BandAnalysis/{condition}/Power_spectrum/{condition}_heatmap.svg'
    fig.savefig(output_path, format='svg')    


bt_figs, bt_data = generate_all_figures()

for condition, duration in conditions.items():

    bt_mean, bt_sem = bt_data[condition]
    power_per_hour(base_path, condition, duration, epoch_per_1h, epoch_per_day, bt_figs, bt_mean, bt_sem)

    print(f'{condition}_Power spectrum heatmap  <Done>')






