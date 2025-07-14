import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

base_path = '/mnt/d/Q_project/'
if not os.path.exists(base_path):
    base_path = '/mnt/e/Q_project/'
conditions = {'SD': 3}
# 'QIH(24h)': 3, 'QIH(48h)': 4, 'QIH(72h)': 5, 'Ctrl_Stim(24h)': 3, 'SD+QIH(48h)': 4
bands = {'Delta': 45, 'Theta': 46, 'Alpha': 47, 'Low-beta': 48, 'High-beta': 49}
epoch_per_1h = 150
rows_per_day = 24


def calculate_bsl_post(data, coeff, rows_per_day, condition):

    bsl_df = pd.DataFrame()
    
    for col in data.columns:
    
        averages = []
    
        for i in range(rows_per_day):
            avg = (data[col].iloc[i] + data[col].iloc[i + rows_per_day]) / 2
            averages.append(avg)
    
        bsl_df[col] = averages

    for col in bsl_df.columns:
    
        if (condition == 'SD' and col in ['220105BD2', '1', '12', '14', '17', '18', '20', '22']) or (condition == 'SD+QIH(1h)' and col in ['66', '74', '77']) or (condition == 'SD+QIH(24h)' and col in ['220105BD2', '220105BD4', '1', '5', '14', '18', '20', '22']):
    
            BSL1_L_df = bsl_df[col][0:12]
            BSL1_D_df = bsl_df[col][12:24]
            BSL2_L_df = bsl_df[col][24:36]
            BSL2_D_df = bsl_df[col][36:48]
            BSL1_df = pd.concat([BSL1_D_df.reset_index(drop=True), BSL1_L_df.reset_index(drop=True)], ignore_index=True)
            BSL2_df = pd.concat([BSL2_D_df.reset_index(drop=True), BSL2_L_df.reset_index(drop=True)], ignore_index=True)
            bsl_df[col] = pd.concat([BSL1_df, BSL2_df]).groupby(level=0).mean().reset_index(drop=True)

        if (condition == 'SD' and col in ['220105BD2', '1', '12', '14', '17', '18', '20', '22']) or (condition == 'SD+QIH(1h)' and col in ['66', '74', '77']) or (condition == 'SD+QIH(24h)' and col in ['220105BD2', '220105BD4', '1', '5', '14', '18', '20', '22']):
            coeff = coeff - 0.5

    post1_df = data.iloc[int(coeff * rows_per_day) : int((coeff + 1) * rows_per_day)].reset_index(drop=True)
    post2_df = data.iloc[int((coeff + 1) * rows_per_day) : int((coeff + 2) * rows_per_day)].reset_index(drop=True)
    bsl_mean = bsl_df.mean(axis=1)
    post1_mean = post1_df.mean(axis=1)
    #print(post1_df)
    post2_mean = post2_df.mean(axis=1)
    mean_df = pd.DataFrame({'bsl': bsl_mean, 'post1': post1_mean, 'post2': post2_mean})
    
    bsl_sem = bsl_df.sem(axis=1)
    post1_sem = post1_df.sem(axis=1)
    post2_sem = post2_df.sem(axis=1)
    sem_df = pd.DataFrame({'bsl': bsl_sem, 'post1': post1_sem, 'post2': post2_sem})

    post1_significant_x = []
    post2_significant_x = []

    for index in bsl_df.index:
        bsl_valid_data = bsl_df.iloc[index, :].dropna()
        post1_valid_data = post1_df.iloc[index, :].dropna()
        post2_valid_data = post2_df.iloc[index, :].dropna()
        
        if len(bsl_valid_data) > 0 and len(post1_valid_data) > 0:
            u_stat, p_val = mannwhitneyu(bsl_valid_data, post1_valid_data, alternative='two-sided')    
            post1_significant_x.append(p_val)    
        if len(bsl_valid_data) > 0 and len(post2_valid_data) > 0:
            u_stat, p_val = mannwhitneyu(bsl_valid_data, post2_valid_data, alternative='two-sided')
            post2_significant_x.append(p_val)

    post1_significant_x = multipletests(post1_significant_x, alpha=0.05, method='holm')[1]
    post2_significant_x = multipletests(post2_significant_x, alpha=0.05, method='holm')[1]


    return mean_df, sem_df, post1_significant_x, post2_significant_x


def power_per_hour(base_path, condition, coeff, band, col):

    # Dataframe for label and Band
    stage_file = f'{base_path}EEGEMG/Compile/{condition}/{condition}_staging.xlsx'
    stage_dict = pd.read_excel(stage_file, usecols=[3], skiprows=2, sheet_name=None, header=None)
    stage_df = pd.DataFrame()
    stage_list = []

    for sheet_name, stage_data in stage_dict.items():
        stage_data = stage_data.dropna()
        stage_data = pd.DataFrame(stage_data.values, columns=[sheet_name])
        stage_list.append(stage_data)
    stage_df = pd.concat(stage_list, axis=1)

    band_file = f'{base_path}EEGEMG/Compile/{condition}/Normalized_{condition}_EEG.xlsx' 
    band_dict = pd.read_excel(band_file, usecols=[col], skiprows=1, sheet_name=None, header=None)
    band_df = pd.DataFrame()
    band_list = []
    
    for sheet_name, band_data in band_dict.items():
        band_data = band_data.dropna()
        band_data = pd.DataFrame(band_data.values, columns=[sheet_name])
        band_list.append(band_data)
    band_df = pd.concat(band_list, axis=1)
 
    label_indices = pd.DataFrame()
    
    for sheet_name in stage_df.columns:
        stage_data = stage_df[sheet_name]
        label_data = pd.Series(index=stage_data.index)
        label_data[stage_data == "W"] = 1
        label_data[stage_data == "NR"] = 2
        label_data[stage_data == "R"] = 3
        label_data.fillna(4, inplace=True) 
        label_indices[sheet_name] = label_data
    
    label_indices = label_indices.astype(int)

    # hourly split
    W_df = pd.DataFrame()
    NR_df = pd.DataFrame()
    R_df = pd.DataFrame()

    for hour_start in range(0, band_df.shape[0], epoch_per_1h):
    
        hour_end = hour_start + epoch_per_1h

        hour_band_data = band_df.iloc[hour_start:hour_end]
        hour_label_data = label_indices.iloc[hour_start:hour_end]

        # calculate hourly average
        for label in [1, 2, 3]:
        
            label_mask = hour_label_data == label
            label_band_data = hour_band_data[label_mask]
            label_average = label_band_data.mean()

            # add
            if label == 1:
                W_df = pd.concat([W_df, label_average.to_frame().T], ignore_index=True)
            elif label == 2:
                NR_df = pd.concat([NR_df, label_average.to_frame().T], ignore_index=True)
            elif label == 3:
                R_df = pd.concat([R_df, label_average.to_frame().T], ignore_index=True)

    W_df.columns = band_df.columns
    NR_df.columns = band_df.columns
    R_df.columns = band_df.columns

    W_mean_df, W_sem_df, W_post1_significant_x, W_post2_significant_x = calculate_bsl_post(W_df, coeff, rows_per_day, condition)
    NR_mean_df, NR_sem_df, NR_post1_significant_x, NR_post2_significant_x = calculate_bsl_post(NR_df, coeff, rows_per_day, condition)
    R_mean_df, R_sem_df, R_post1_significant_x, R_post2_significant_x = calculate_bsl_post(R_df, coeff, rows_per_day, condition)

    # Plot
    mean_dfs = {'W': W_mean_df, 'NR': NR_mean_df, 'R': R_mean_df}
    sem_dfs = {'W': W_sem_df, 'NR': NR_sem_df, 'R': R_sem_df}
    post1_sig_dfs = {'W': W_post1_significant_x, 'NR': NR_post1_significant_x, 'R': R_post1_significant_x}
    post2_sig_dfs = {'W': W_post2_significant_x, 'NR': NR_post2_significant_x, 'R': R_post2_significant_x}
    colors = {'W': (1, 0.5882, 0.5882), 'NR': (0.5333, 0.7412, 0.9882), 'R': (0.4392, 0.6784, 0.2784)}

    for stage, color in colors.items():

        plt.rcParams.update({'font.size': 20})
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        plt.suptitle(stage)

        bsl_sum_vals = mean_dfs[stage]['bsl'] + sem_dfs[stage]['bsl']
        post1_sum_vals = mean_dfs[stage]['post1'] + sem_dfs[stage]['post1']
        post2_sum_vals = mean_dfs[stage]['post2'] + sem_dfs[stage]['post2']
        if (bsl_sum_vals >= 5).any() or (post1_sum_vals >= 5).any() or (post2_sum_vals >= 5).any():
            y_factor = 2
        else:
            y_factor = 1

        axs[0].plot(mean_dfs[stage].index + 0.5, mean_dfs[stage]['bsl'], color='gray', linewidth=5)
        axs[0].plot(mean_dfs[stage].index + 0.5, mean_dfs[stage]['post1'], color=color, linewidth=5)
        axs[0].errorbar(mean_dfs[stage].index + 0.5, mean_dfs[stage]['bsl'], yerr=sem_dfs[stage]['bsl'], fmt='o', color='gray', capsize=7.5, capthick=3, label='BSL', linewidth=5)
        axs[0].errorbar(mean_dfs[stage].index + 0.5, mean_dfs[stage]['post1'], yerr=sem_dfs[stage]['post1'], fmt='o', color=color, capsize=7.5, capthick=3, label='Post(0-24h)', linewidth=5)
        axs[0].fill_between(x=np.arange(0, 13), y1=0, y2=6 * y_factor, color=(0.9, 0.9, 0.9))
        axs[0].set_xlabel('Time (h)')
        axs[0].set_xticks(np.arange(0, rows_per_day + 1, rows_per_day / 4))
        axs[0].set_xticklabels(np.arange(0, rows_per_day + 1, rows_per_day / 4).astype(int))
        axs[0].set_xlim([0, rows_per_day])
        axs[0].set_ylabel(f'{band} power (A.U.)')
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].set_yticks(np.arange(0, 5 * y_factor, 1 * y_factor))
        axs[0].set_yticklabels(np.arange(0, 5 * y_factor, 1 * y_factor))
        axs[0].set_ylim([0, 6 * y_factor])
        for idx in range(len(post1_sig_dfs[stage])):
            if post1_sig_dfs[stage][idx] < 0.05:
                axs[0].plot([idx, idx+1], [4*y_factor*0.98, 4*y_factor*0.98], '-', color=color, linewidth=4, alpha=0.5)
        axs[0].legend(loc='upper left', bbox_to_anchor=(0.55, 1), fontsize=20, frameon=False)

        axs[1].plot(mean_dfs[stage].index + 0.5, mean_dfs[stage]['bsl'], color='gray', linewidth=5)
        axs[1].plot(mean_dfs[stage].index + 0.5, mean_dfs[stage]['post2'], color=color, linewidth=5, alpha=0.5)
        axs[1].errorbar(mean_dfs[stage].index + 0.5, mean_dfs[stage]['bsl'], yerr=sem_dfs[stage]['bsl'], fmt='o', color='gray', capsize=7.5, capthick=3, label='BSL', linewidth=5)
        axs[1].errorbar(mean_dfs[stage].index + 0.5, mean_dfs[stage]['post2'], yerr=sem_dfs[stage]['post2'], fmt='o', color=color, alpha=0.5, capsize=7.5, capthick=3, label='Post(24-48h)', linewidth=5)
        axs[1].fill_between(x=np.arange(0, 13), y1=0, y2=6 * y_factor, color=(0.9, 0.9, 0.9))
        axs[1].set_xlabel('Time (h)')
        axs[1].set_xticks(np.arange(0, rows_per_day + 1, rows_per_day / 4))
        axs[1].set_xticklabels(np.arange(0, rows_per_day + 1, rows_per_day / 4).astype(int))
        axs[1].set_xlim([0, rows_per_day])
        axs[1].set_ylabel(f'{band} power (A.U.)')
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].set_yticks(np.arange(0, 5 * y_factor, 1 * y_factor))
        axs[1].set_yticklabels(np.arange(0, 5 * y_factor, 1 * y_factor))        
        axs[1].set_ylim([0, 6 * y_factor])
        for idx in range(len(post2_sig_dfs[stage])):
            if post2_sig_dfs[stage][idx] < 0.05:
                axs[1].plot([idx, idx+1], [4*y_factor*0.98, 4*y_factor*0.98], '-', color=color, linewidth=4, alpha=0.5)        
        axs[1].legend(loc='upper left', bbox_to_anchor=(0.55, 1), fontsize=20, frameon=False)

        # Save figure
        output_path = f'{base_path}BandAnalysis/{condition}/Power_per_hour/{stage}_{band}.tif'
        plt.savefig(output_path, format='tiff', dpi=350)    

for condition, coeff in conditions.items():
    for band, row in bands.items():
        power_per_hour(base_path, condition, coeff, band, row)
        print(f'{condition}_{band}_Power per hour  <Done>')




