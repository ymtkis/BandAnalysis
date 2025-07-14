import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

base_path = '/mnt/d/Q_project/'
if not os.path.exists(base_path):
    base_path = '/mnt/e/Q_project/'
conditions = {'SD': 3}
# 'QIH(24h)': 3, 'QIH(48h)': 4, 'QIH(72h)': 5, 'Ctrl_Stim(24h)': 3, 'SD+QIH(48h)': 4
stages = ['W', 'NR', 'R']
bands = {'Delta': 45} #, 'Theta': 46, 'Alpha': 47, 'Low-beta': 48, 'High-beta': 49
epoch_per_12h = 1800
row_num = 2



def calculate_bsl_post(data, coeff, row_num, condition):

    bsl_df = pd.DataFrame()
    for col in data.columns:
        averages = []
        for i in range(row_num):
            avg = (data[col].iloc[i] + data[col].iloc[i + row_num]) / 2
            averages.append(avg)
        bsl_df[col] = averages
    
    bsl_dict = bsl_df.to_dict(orient='list')
    bsl_list = []
    post1_list = []
    post2_list = []

    for col in bsl_dict.keys():
        if (condition == 'SD' and col in ['220105BD2', '1', '12', '14', '17', '18', '20', '22']) or (condition == 'SD+QIH(1h)' and col in ['66', '74', '77']) or (condition == 'SD+QIH(24h)' and col in ['220105BD2', '220105BD4', '1', '5', '14', '18', '20', '22']):
            bsl_list.append(bsl_dict[col][1])
        else:
            bsl_list.append(bsl_dict[col][0])

        if (condition == 'SD' and col in ['220105BD2', '1', '12', '14', '17', '18', '20', '22']) or (condition == 'SD+QIH(1h)' and col in ['66', '74', '77']) :
            coeff = 2.5

        if (condition == 'SD+QIH(24h)' and col in ['220105BD2', '220105BD4', '1', '5', '14', '18', '20', '22']):
            coeff = 3.5

        post1 = data.iloc[int(coeff * row_num) : int((coeff + 1) * row_num)].reset_index(drop=True)
        post2 = data.iloc[int((coeff + 1) * row_num) : int((coeff + 2) * row_num)].reset_index(drop=True)
    
        post1 = post1.to_dict(orient='list')
        post2 = post2.to_dict(orient='list')

        post1_list.append(post1[col][0])
        post2_list.append(post2[col][0])

    per_12h_results_per_stage = {'BSL': bsl_list, 'post1': post1_list, 'post2': post2_list}
++++++++++++++++++++++++++++++
    return per_12h_results_per_stage



def power_per_hour(base_path, condition, coeff, band, col):
    # Dataframe for label and SWA
    stage_file = f'{base_path}EEGEMG/Compile/{condition}/{condition}_staging.xlsx'
    stage_dict = pd.read_excel(stage_file, usecols=[3], skiprows=2, sheet_name=None, header=None)
    stage_df = pd.DataFrame()
    for sheet_name, stage_data in stage_dict.items():
        stage_df[sheet_name] = stage_data

    band_file = f'{base_path}EEGEMG/Compile/{condition}/Normalized_{condition}_EEG.xlsx' 
    band_dict = pd.read_excel(band_file, usecols=[col], skiprows=1, sheet_name=None, header=None)
    band_df = pd.DataFrame()
    for sheet_name, band_data in band_dict.items():
        band_df[sheet_name] = band_data

    
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

    W_df = pd.DataFrame()
    NR_df = pd.DataFrame()
    R_df = pd.DataFrame()

    # hourly split
    for hour_start in range(0, band_df.shape[0], epoch_per_12h):
        hour_end = hour_start + epoch_per_12h

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

    per_12h_W_df = calculate_bsl_post(W_df, coeff, row_num, condition)
    per_12h_NR_df = calculate_bsl_post(NR_df, coeff, row_num, condition)
    per_12h_R_df = calculate_bsl_post(R_df, coeff, row_num, condition)

    per_12h_results = {'W': per_12h_W_df, 'NR': per_12h_NR_df, 'R': per_12h_R_df}

    return per_12h_results



def per_12h_plot(per_12h_results, condition, stage):

    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(figsize=(4, 6))

    y_limit = (0, 6) if stage in ['NR'] else (0, 3)
    y_ticks = (0, 3, 6) if stage in ['NR'] else (0, 1.5, 3)
    colors = {'W': (1, 0.5882, 0.5882), 'NR': (0.5333, 0.7412, 0.9882), 'R': (0.4392, 0.6784, 0.2784)}
    
    time_labels = ['BSL', '0–12h', '24–36h']
    x = np.arange(len(time_labels))

    bsl_vals = per_12h_results[stage]['BSL']
    post1_vals = per_12h_results[stage]['post1']
    post2_vals = per_12h_results[stage]['post2']
    bsl_vals = np.array(bsl_vals).flatten()
    post1_vals = np.array(post1_vals).flatten()
    post2_vals = np.array(post2_vals).flatten()

    means = [np.mean(bsl_vals), np.mean(post1_vals), np.mean(post2_vals)]
    sems = [np.std(bsl_vals)/np.sqrt(len(bsl_vals)),
            np.std(post1_vals)/np.sqrt(len(post1_vals)),
            np.std(post2_vals)/np.sqrt(len(post2_vals))]
    
    bar_colors = ['gray', colors[stage], colors[stage]]
    alphas = [1, 1, 0.5]
    
    for i in range(3):
        axs.bar(x[i], means[i], yerr=sems[i], color=bar_colors[i], alpha=alphas[i], capsize=5, width=0.6)

    for i, values in enumerate([bsl_vals, post1_vals, post2_vals]):
        jitter = (np.random.rand(len(values)) - 0.5) * 0.2
        axs.scatter(np.full_like(values, x[i]) + jitter, values, color='black', s=15, alpha=0.7)

    comparisons = [
        (0, 1, bsl_vals, post1_vals),
        (1, 2, post1_vals, post2_vals),
        (0, 2, bsl_vals, post2_vals)
    ]

    p_values = []
    for i, j, data1, data2 in comparisons:
        stat, p = stats.ttest_rel(data1, data2, nan_policy='omit')
        p_adj = min(p * 3, 1.0)  # Bonferroni補正（3比較）
        p_values.append((i, j, p_adj))

    # p値注釈表示
    max_y = y_limit[1] * 0.8 
    step = y_limit[1] * 0.075
    for idx, (i, j, p) in enumerate(p_values):
        if p < 0.05:
            y = max_y + idx * step
            axs.plot([x[i], x[j]], [y, y], color='black')

            if p >= 0.001:
                axs.text((x[i] + x[j]) / 2, y + step / 10, f"{p:.3f}", ha='center', va='bottom', fontsize=14)
            else:
                axs.text((x[i] + x[j]) / 2, y + step / 10, "<0.001", ha='center', va='bottom', fontsize=14)


    # 軸設定
    axs.set_xticks(x)
    axs.set_xticklabels(time_labels, fontsize=16, rotation=45)
    axs.set_ylim(y_limit)
    axs.set_yticks(y_ticks)
    axs.set_ylabel('Average power (A.U.)')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    plt.tight_layout()
        
    return fig

for condition, coeff in conditions.items():
    for band, row in bands.items():
        
        per_12h_results = power_per_hour(base_path, condition, coeff, band, row)
        
        for stage in stages:
            fig = per_12h_plot(per_12h_results, condition, stage)
            output_path = f'{base_path}/BandAnalysis/{condition}/Power_per_12h'
            fig.savefig(f'{output_path}/{stage}_{band}.tif', format='tif', dpi=350)
        
        print(f'{condition}_{band}_Power per 12h  <Done>')




