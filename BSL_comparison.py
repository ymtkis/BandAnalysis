import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

base_path = '/mnt/d/Q project/'
conditions = ['QIH(24h)', 'QIH(48h)', 'QIH(72h)', 'Ctrl_Stim(24h)']
experiments = ['QIH(24h)', 'QIH(48h)', 'QIH(72h)']
stages = ['W', 'NR', 'R']
epoch_per_2h = 300
row_num = 12


def calculate_freq(data):
    data_list = list(data.values())
    mean_df = pd.concat(data_list).groupby(level=0).mean()
    sem_df = pd.concat(data_list).groupby(level=0).sem()
    sem_df = pd.DataFrame(np.nan_to_num(sem_df))
    return mean_df, sem_df


def power_per_hour(base_path):

    stage_all_condition = {}
    mean_all_condition = {}
    sem_all_condition = {}
    for condition in conditions:
        # Dataframe for label and freq
        stage_file = f'{base_path}EEGEMG/Compile/{condition}/{condition}_staging.xlsx'
        stage_dict = pd.read_excel(stage_file, usecols=[3], skiprows=2, nrows=epoch_per_2h * row_num * 2, sheet_name=None, header=None)
        stage_df = pd.DataFrame()
        for sheet_name, stage_data in stage_dict.items():
            stage_df[sheet_name] = stage_data
        
        freq_file = f'{base_path}EEGEMG/Compile/{condition}/Normalized_by_delta_{condition}_EEG.xlsx' 
        freq_dict = pd.read_excel(freq_file, usecols=range(3, 44), skiprows=1, nrows=epoch_per_2h * row_num * 2, sheet_name=None, header=None)
        freq_data_dict = {}
        for sheet_name, freq_data in freq_dict.items():        
            freq_data_indiv = freq_data.rename(columns=lambda x: str(x-3))
            freq_data_dict[sheet_name] = freq_data_indiv 
        
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
        
        W_dict = {}
        NR_dict = {}
        R_dict = {}

        for sheet_name, freq_data in freq_data_dict.items():
            W_df = pd.DataFrame() 
            NR_df = pd.DataFrame()
            R_df = pd.DataFrame()
            
            # hourly split
            for hour_start in range(0, freq_data.shape[0], epoch_per_2h):
                hour_end = min(hour_start + epoch_per_2h, freq_data.shape[0])
                hour_freq_data = freq_data.iloc[hour_start:hour_end]
                hour_label_data = label_indices[sheet_name].iloc[hour_start:hour_end]

                for label in [1, 2, 3]:
                    label_mask = (hour_label_data == label).values
                    label_mask = label_mask[:len(hour_freq_data)]
                    label_freq_data = hour_freq_data[label_mask]
                    label_average = label_freq_data.mean(axis=0)

                    if label == 1:
                        W_df = pd.concat([W_df, label_average.to_frame().T], ignore_index=True)
                    elif label == 2:
                        NR_df = pd.concat([NR_df, label_average.to_frame().T], ignore_index=True)
                    elif label == 3:
                        R_df = pd.concat([R_df, label_average.to_frame().T], ignore_index=True)
        
            W_dict[sheet_name] = W_df
            NR_dict[sheet_name] = NR_df
            R_dict[sheet_name] = R_df    

        stage_dfs = {'W': W_dict, 'NR':NR_dict, 'R':R_dict}
        stage_all_condition[condition] = stage_dfs

        W_mean_df, W_sem_df = calculate_freq(W_dict)
        NR_mean_df, NR_sem_df = calculate_freq(NR_dict)
        R_mean_df, R_sem_df = calculate_freq(R_dict)

        mean_dfs = {'W': W_mean_df, 'NR': NR_mean_df, 'R': R_mean_df}
        sem_dfs = {'W': W_sem_df, 'NR': NR_sem_df, 'R': R_sem_df}
        
        mean_all_condition[condition] = mean_dfs
        sem_all_condition[condition] = sem_dfs


    # statistics
    with pd.ExcelWriter(f'{base_path}Band_analysis/BSL_comparison/anova_result.xlsx') as writer:
        for stage in stages:
            for i in range(row_num):
                anova = []
                for condition in conditions:
                    for sheet_name in stage_all_condition[condition][stage].keys():
                            data = stage_all_condition[condition][stage][sheet_name].iloc[[i, i+row_num]].mean()            
                            for freq in range(40):                              
                                anova.append({'Condition': condition, 'Frequency': freq, 'Indiv': sheet_name, 'Value': data.iloc[freq]})                
                anova = pd.DataFrame(anova)
                model = ols('Value ~ C(Condition) * C(Frequency)', data=anova).fit()
                anova_summary = pd.DataFrame(model.summary().tables[0])
                anova_results = pd.DataFrame(model.summary().tables[1])
                anova_results = pd.concat([anova_summary, anova_results], axis=0)                
                anova_results.to_excel(writer, sheet_name=f'BSL_{stage}_{i * 2}-{(i * 2) + 2}', index=False)

    for stage in stages:
        with pd.ExcelWriter(f'{base_path}Band_analysis/BSL_comparison/test_result_{stage}.xlsx') as writer:                       
            for i in range(row_num):
                test_results = pd.DataFrame(index=range(40))
                p_values = {experiment: [] for experiment in experiments}
                for freq in range(40):    
                    bsl_freq_data = {condition: [] for condition in conditions}  # 'Ctrl_Stim(24h)' を含むすべての条件
                    for condition in conditions:
                        for sheet_name in stage_all_condition[condition][stage].keys():
                            bsl_data = stage_all_condition[condition][stage][sheet_name].iloc[[i, i+row_num]].mean()                   
                            bsl_freq_data[condition].append(bsl_data.iloc[freq])

                    for experiment in experiments:  # 'Ctrl_Stim(24h)' と各実験条件を比較
                        # Normality test
                        normality_ctrl = stats.shapiro(bsl_freq_data['Ctrl_Stim(24h)']).pvalue
                        normality_exp = stats.shapiro(bsl_freq_data[experiment]).pvalue

                        # Homogeneity of variance test
                        equal_var = stats.levene(bsl_freq_data['Ctrl_Stim(24h)'], bsl_freq_data[experiment]).pvalue > 0.05

                        # Selection of t-test
                        if normality_ctrl > 0.05 and normality_exp > 0.05:
                            if equal_var:
                                t_test = stats.ttest_ind(bsl_freq_data['Ctrl_Stim(24h)'], bsl_freq_data[experiment], equal_var=True)
                            else:
                                t_test = stats.ttest_ind(bsl_freq_data['Ctrl_Stim(24h)'], bsl_freq_data[experiment], equal_var=False)
                        else:
                            t_test = stats.mannwhitneyu(bsl_freq_data['Ctrl_Stim(24h)'], bsl_freq_data[experiment])

                        p_values[experiment].append(t_test.pvalue)
                        test_results.loc[freq, f'p_{experiment}'] = t_test.pvalue
                
                for experiment in experiments:
                    p_values_adj = multipletests(p_values[experiment], alpha=0.05, method='fdr_bh')[1]
                    for freq in range(40):
                        test_results.loc[freq, f'q_{experiment}'] = p_values_adj[freq]
                test_results.to_excel(writer, sheet_name=f'{i * 2}-{(i * 2) + 2}', index=False)



    # Plot
    plt.rcParams.update({'font.size': 20})
    for stage in stages:
        fig, axs = plt.subplots(2, 6, figsize=(36, 20))
        plt.suptitle(stage, fontsize=28)

        for i in range(12):
            bsl_color = (0.8, 0.8, 0.8)
            for condition in conditions:
                x = mean_all_condition[condition][stage].columns
                y_bsl = mean_all_condition[condition][stage].iloc[[i, i+row_num]].mean()
                err_bsl = sem_all_condition[condition][stage].iloc[[i, i+row_num]].mean()            
                axs[i // 6, i % 6].errorbar(x, y_bsl, yerr=err_bsl, fmt='o', color=bsl_color, capsize=5, label=condition if i == 0 else '', linestyle='-')
                bsl_color = tuple(c - 0.2 for c in bsl_color)
                bsl_color = tuple(max(c, 0) for c in bsl_color)            
            axs[i // 6, i % 6].set_xlabel('Frequency (Hz)')
            axs[i // 6, i % 6].set_xticks(np.arange(0, len(x) + 1, len(x) / 4).astype(int))
            axs[i // 6, i % 6].set_xticklabels(np.round(np.arange(0, len(x) + 1, len(x) / 4) / 2).astype(str))
            axs[i // 6, i % 6].set_xlim([-1, len(x) + 1])
            axs[i // 6, i % 6].set_ylabel('Power')
            axs[i // 6, i % 6].spines['top'].set_visible(False)
            axs[i // 6, i % 6].spines['right'].set_visible(False)

            if stage == 'W':
                axs[i // 6, i % 6].set_yticks(np.arange(0, 2.1, 0.5))
                axs[i // 6, i % 6].set_yticklabels(np.arange(0, 2.1, 0.5).astype(str))
                axs[i // 6, i % 6].set_ylim([0, 2])
            elif stage == 'NR':
                axs[i // 6, i % 6].set_yticks(np.arange(0, 6.1, 2))
                axs[i // 6, i % 6].set_yticklabels(np.arange(0, 6.1, 2).astype(str))
                axs[i // 6, i % 6].set_ylim([0, 6])
            else:
                axs[i // 6, i % 6].set_yticks(np.arange(0, 3.1, 1))
                axs[i // 6, i % 6].set_yticklabels(np.arange(0, 3.1, 1).astype(str))
                axs[i // 6, i % 6].set_ylim([0, 3])
            axs[i // 6, i % 6].set_title(f't : {i * 2}-{(i + 1) * 2}')

            q_values = pd.read_excel(f'{base_path}Band_analysis/BSL_comparison/test_result_{stage}.xlsx', sheet_name=f'{i * 2}-{(i * 2) + 2}')
            q_values = q_values.reset_index(drop=True)
            
            
            # 各実験に対して線が追加されたかどうかを追跡するための辞書
            line_added = {experiment: False for experiment in experiments}
            for freq in range(40):
                bsl_color = [0.8, 0.8, 0.8]
                for iter, experiment in enumerate(experiments):
                    try:
                        if q_values.loc[freq, f'q_{experiment}'] <= 0.05:
                            y_min, y_max = axs[i // 6, i % 6].get_ylim()
                            current_color = [c - 0.2 * iter for c in bsl_color]
                            y_position = y_max * (0.9 - (iter / 40))
                            axs[i // 6, i % 6].hlines(y_position, xmin=freq - 0.5, xmax=freq + 0.5, linewidth=10, color=current_color)
                            line_added[experiment] = True  # 線が追加されたことを記録
                    except KeyError:
                        pass

            # 線が追加された実験に対してのみテキストを追加
            for iter, experiment in enumerate(experiments):
                if line_added[experiment]:
                    y_position = y_max * (0.9 - (iter / 40))
                    axs[i // 6, i % 6].text(x=30, y=y_position, s=experiment, va='center', color=[0.8 - 0.2 * iter] * 3, fontsize=14)


        fig.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=28)
        

        # Save figure
        output_path = f'{base_path}Band_analysis/BSL_comparison/BSL_comparison_{stage}.tif'
        plt.savefig(output_path, format='tiff', dpi=350)    

    
power_per_hour(base_path)
print(f'BSL_comparison  <Done>')




