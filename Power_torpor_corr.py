import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import stats
from itertools import combinations
from statsmodels.stats.multitest import multipletests

base_path = '/mnt/d/Q project/'
conditions = {'Ctrl_Stim(24h)': 3, 'QIH(24h)': 3, 'QIH(48h)': 4}
# 'QIH(24h)': 3, 'QIH(48h)': 4, 'QIH(72h)': 5, 'Ctrl_Stim(24h)': 3
bands = {'Delta': 45}
stages = ['W', 'NR', 'R']
# 'Delta': 45, 'Theta': 46, 'Alpha': 47, 'Low-beta': 48, 'High-beta': 49
epoch_per_1d = 3600


def calculate_bsl_post(data, coeff, condition):
    def reverse_half(data):
        n = len(data) // 2
        return pd.concat([data.iloc[n:], data.iloc[:n]]).reset_index(drop=True)

    if condition in ['SD', 'SD+QIH(1h)', 'SD+QIH(24h)', 'SD+QIH(48h)']
        bsl_df = ((data.iloc[0] + data.iloc[1]) / 2).to_frame()
        post1_df = data.iloc[coeff].to_frame()
        post2_df = data.iloc[coeff + 1].to_frame()
        bsl_df = reverse_half(bsl_df)
        post1_df = reverse_half(post1_df)
        post2_df = reverse_half(post2_df)
        indiv_df = pd.concat([bsl_df, post1_df, post2_df], axis=1)
        indiv_df.columns = ['bsl', 'post1', 'post2']

    else:
        bsl_df = ((data.iloc[0] + data.iloc[1]) / 2).to_frame()
        post1_df = data.iloc[coeff].to_frame()
        post2_df = data.iloc[coeff + 1].to_frame()
        indiv_df = pd.concat([bsl_df, post1_df, post2_df], axis=1)
        indiv_df.columns = ['bsl', 'post1', 'post2']

    bsl_mean = bsl_df.mean()
    post1_mean = post1_df.mean()
    post2_mean = post2_df.mean()
    mean_df = pd.DataFrame({'bsl': bsl_mean, 'post1': post1_mean, 'post2': post2_mean})

    bsl_sem = bsl_df.sem()
    post1_sem = post1_df.sem()
    post2_sem = post2_df.sem()
    sem_df = pd.DataFrame({'bsl': bsl_sem, 'post1': post1_sem, 'post2': post2_sem})

    return indiv_df, mean_df, sem_df



def power_torpor_corr(base_path, condition, coeff, band, col):
    # Dataframe for torpor_index
    torpor_ind_file = f'{base_path}Body_temperature/Compile/{condition}/{condition}_torpor_index.xlsx'
    torpor_ind_df = pd.read_excel(torpor_ind_file, usecols=[0, 2], skiprows=1, header=None).T
    torpor_ind_df.columns = torpor_ind_df.iloc[0].astype(int).astype(str)
    torpor_ind_df = torpor_ind_df.drop(torpor_ind_df.index[0])

    # Dataframe for label and bands
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

    common_columns = torpor_ind_df.columns.intersection(stage_df.columns)
    stage_df = stage_df[common_columns]

    common_columns = torpor_ind_df.columns.intersection(band_df.columns)
    band_df = band_df[common_columns]
    
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

    # daily split
    for day_start in range(0, band_df.shape[0], epoch_per_1d):
        day_end = day_start + epoch_per_1d

        hour_band_data = band_df.iloc[day_start:day_end]
        hour_label_data = label_indices.iloc[day_start:day_end]

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

    W_indiv_df, W_mean_df, W_sem_df = calculate_bsl_post(W_df, coeff, condition)
    NR_indiv_df, NR_mean_df, NR_sem_df = calculate_bsl_post(NR_df, coeff, condition)
    R_indiv_df, R_mean_df, R_sem_df = calculate_bsl_post(R_df, coeff, condition)

    # Plot
    indiv_dfs = {'W': W_indiv_df, 'NR': NR_indiv_df, 'R': R_indiv_df}
    colors = {'W': (1, 0.5882, 0.5882), 'NR': (0.5333, 0.7412, 0.9882), 'R': (0.4392, 0.6784, 0.2784)}
    bsl_color = (0.7, 0.7, 0.7)

    torpor_ind_df = torpor_ind_df.T


    for stage, color in colors.items():

        plt.rcParams.update({'font.size': 20})
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        plt.suptitle(stage)
    
        # Post1 - BSL
        post1_y[condition][band][stage] = []
        x1 = torpor_ind_df.values.flatten()
        y1 = (indiv_dfs[stage]['post1'] - indiv_dfs[stage]['bsl']).values
        post1_y[condition][band][stage] = y1
        axs[0].scatter(x1, y1, s=60, color=color)
        axs[0].set_ylim([-2, 2])
        axs[0].set_yticks(np.arange(-2, 2.1, 0.5))
        axs[0].set_xlabel('Torpor index')
        axs[0].set_ylabel(f'Δ {band} power (Post - BSL)')
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)

        # approximate line
        coeffs = np.polyfit(x1, y1, deg=1)
        poly_eq = np.poly1d(coeffs)
        axs[0].plot(x1, poly_eq(x1), color='black', linestyle='-')

        # correlation coefficient
        corr, _ = pearsonr(x1, y1)
        axs[0].text(1, 1, f'r = {corr:.2f}', transform=axs[0].transAxes, va='top', ha='right')

        # Post2 - BSL
        x2 = torpor_ind_df.loc[indiv_dfs[stage]['post2'].dropna().index].values.flatten()
        y2 = (indiv_dfs[stage]['post2'] - indiv_dfs[stage]['bsl']).values
        valid_indices = ~np.isnan(y2)
        y2 = y2[valid_indices]
        axs[1].scatter(x2, y2, s=60, color=color)
        axs[1].set_ylim([-2, 2])
        axs[1].set_yticks(np.arange(-2, 2.1, 0.5))
        axs[1].set_xlabel('Torpor index')
        axs[1].set_ylabel('Δ Power (Post - BSL)')
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)

        # approximate line
        coeffs = np.polyfit(x2, y2, deg=1)
        poly_eq = np.poly1d(coeffs)
        axs[1].plot(x2, poly_eq(x2), color='black', linestyle='-')

        # correlation coefficient
        corr, _ = pearsonr(x2, y2)
        axs[1].text(1, 1, f'r = {corr:.2f}', transform=axs[1].transAxes, va='top', ha='right')

        # Save figure
        output_path = f'{base_path}Band_analysis/{condition}/Power_torpor_corr/{condition}_{stage}_{band}.tif'
        plt.savefig(output_path, format='tiff', dpi=350)   

post1_y = {}

for condition, coeff in conditions.items():
    post1_y[condition] = {}

    for band, row in bands.items():
        post1_y[condition][band] = {}
        post1_y[condition][band] = {}
        power_torpor_corr(base_path, condition, coeff, band, row)
        print(f'{condition}_{band}_Power torpor correlation  <Done>')

test_results = []
for band in bands.keys():
    for stage in stages:
        fig, ax = plt.subplots(figsize=(10, 6))
        conditions_list = list(conditions.keys())
        x_positions = np.arange(len(conditions))
        means = []
        sems = []
        p_values = []
        cond_pairs = []

        for condition in conditions.keys():
            values = post1_y[condition][band][stage]
            means.append(np.mean(values))
            sems.append(stats.sem(values))

        # mean
        ax.bar(x_positions, means, yerr=sems, capsize=5, alpha=0.7, color='red')

        # indivisual value
        for i, condition in enumerate(conditions):
            x = np.ones(len(post1_y[condition][band][stage])) * x_positions[i]
            ax.scatter(x, post1_y[condition][band][stage], color='black', alpha=0.7)

        ax.set_ylabel(f'Δ {band} power (Post - BSL)')
        ax.set_ylim([-0.5, None])
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for cond1, cond2 in combinations(conditions_list, 2):
            t_stat, p_value = stats.ttest_ind(post1_y[cond1][band][stage], post1_y[cond2][band][stage])
            p_values.append(p_value)
            cond_pairs.append((cond1, cond2))

        # Holm法による補正
        corrected_pvals = multipletests(p_values, alpha=0.05, method='holm')[1]

        # 結果リストにテスト結果を追加（補正後のp値を含む）
        for ((cond1, cond2), p_value, adj_pval) in zip(cond_pairs, p_values, corrected_pvals):
            test_results.append({
                'Band': band,
                'Stage': stage,
                'Condition1': cond1,
                'Condition2': cond2,
                'T-statistic': t_stat,
                'P-value': p_value,
                'Adjusted P-value': adj_pval
            })

        for (cond1, cond2), adj_pval in zip(cond_pairs, corrected_pvals):
            if adj_pval < 0.05:
                # 有意差のマーカー位置を計算
                x1 = x_positions[conditions_list.index(cond1)]
                x2 = x_positions[conditions_list.index(cond2)]
                max_mean = max(means[conditions_list.index(cond1)], means[conditions_list.index(cond2)])
                ax.text((x1 + x2) / 2, 2.5, '*', ha='center', va='bottom', color='black', fontsize=14)

        ax.set_ylabel(f'Δ {band} power (Post - BSL)')
        ax.set_ylim([-0.5, None])
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)



        output_path = f'{base_path}Band_analysis/Power_diff/{stage}_{band}.tif'
        plt.savefig(output_path, format='tiff', dpi=350)


test_df = pd.DataFrame(test_results)
test_df.to_excel(f'{base_path}Band_analysis/Power_diff/test_results.xlsx', index=False)








