import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

base_path = '/mnt/d/Q project/'
conditions = {'QIH(72h)': 5}
# 'QIH(24h)': 3,'QIH(48h)': 4, 'QIH(72h)': 5, 'Ctrl_Stim(24h)': 3, 'SD': 2.5, 'FIT(5d)': 7
stages = ['W', 'NR', 'R']
epoch_per_2h = 300
row_num = 12


def calculate_freq(data):

    data_list = list(data.values())
    mean_df = pd.concat(data_list).groupby(level=0).mean()
    sem_df = pd.concat(data_list).groupby(level=0).sem()
    sem_df = pd.DataFrame(np.nan_to_num(sem_df))
    
    return mean_df, sem_df


def power_per_hour(base_path, condition, coeff):
    
    # Dataframe for stage label
    stage_file = f'{base_path}EEGEMG/Compile/{condition}/{condition}_staging.xlsx'
    stage_dict = pd.read_excel(stage_file, usecols=[3], skiprows=2, sheet_name=None, header=None)
    
    stage_df = pd.DataFrame()
    label_indices = pd.DataFrame()

    for sheet_name, stage_data in stage_dict.items():
    
        if (condition == 'SD' and sheet_name in ['220105BD2', '1', '14', '18', '20', '22']):
            coeff = 2.5
    
        rows=int(epoch_per_2h * row_num * (coeff + 2))        
        stage_df[sheet_name] = stage_data[:rows, 0].reset_index(drop=True)

    for sheet_name in stage_df.columns:
    
        stage_data = stage_df[sheet_name]
        label_data = pd.Series(index=stage_data.index)
        label_data[stage_data == "W"] = 1
        label_data[stage_data == "NR"] = 2
        label_data[stage_data == "R"] = 3
        label_data.fillna(4, inplace=True) 
        label_indices[sheet_name] = label_data
    
    label_indices = label_indices.astype(int)
    
    # Dataframe for freq-power
    freq_file = f'{base_path}EEGEMG/Compile/{condition}/Normalized_by_delta_{condition}_EEG.xlsx' 
    freq_dict = pd.read_excel(freq_file, usecols=range(3, 44), skiprows=1, sheet_name=None, header=None)
    
    freq_data_dict = {}
    
    for sheet_name, freq_data in freq_dict.items():

        freq_data_indiv = freq_data.rename(columns=lambda x: str(x-3))
        freq_data_dict[sheet_name] = freq_data_indiv 

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
    
    W_mean_df, W_sem_df = calculate_freq(W_dict)
    NR_mean_df, NR_sem_df = calculate_freq(NR_dict)
    R_mean_df, R_sem_df = calculate_freq(R_dict)



    # statistics
    with pd.ExcelWriter(f'{base_path}Band_analysis/{condition}/Power_spectrum/anova_result.xlsx') as writer:
        for stage in stages:
            for i in range(row_num):
                
                bsl_anova = []
                post_anova = []
                
                for sheet_name in stage_dfs[stage].keys():
                
                    if (condition == 'SD' and sheet_name in ['220105BD2', '1', '14', '18', '20', '22']) or (condition in ['SD+QIH(24h)', 'QIH(1h)']):
                        bsl_data = stage_dfs[stage][sheet_name].iloc[i + int(row_num*0.5)]
                
                    else:
                        bsl_data = stage_dfs[stage][sheet_name].iloc[[i, i+row_num]].mean()
                    post_data = stage_dfs[stage][sheet_name].iloc[i+int(row_num * coeff)]
                
                    for freq in range(40):
                        bsl_anova.append({'Condition': 'BSL', 'Frequency': freq, 'Indiv': sheet_name, 'Value': bsl_data.iloc[freq]})                
                        post_anova.append({'Condition': 'Post', 'Frequency': freq, 'Indiv': sheet_name, 'Value': post_data.iloc[freq]})
                
                bsl_anova = pd.DataFrame(bsl_anova)
                post_anova = pd.DataFrame(post_anova)
                anova = pd.concat([bsl_anova, post_anova])

                model = ols('Value ~ C(Condition) * C(Frequency)', data=anova).fit()
                anova_summary = pd.DataFrame(model.summary().tables[0])
                anova_results = pd.DataFrame(model.summary().tables[1])
                anova_results = pd.concat([anova_summary, anova_results], axis=0)                
                anova_results.to_excel(writer, sheet_name=f'{stage}_{i * 2}-{(i * 2) + 2}', index=False)


    for stage in stages:
        with pd.ExcelWriter(f'{base_path}Band_analysis/{condition}/Power_spectrum/test_result_{stage}.xlsx') as writer:                    
            
            test_results = pd.DataFrame(index=range(40), columns=['p'] + ['q'])
            
            for i in range(row_num):
                p_values = []
            
                for freq in range(40):
            
                    bsl_freq_data = []
                    post_freq_data = []
            
                    for sheet_name in stage_dfs[stage].keys():
            
                        if (condition == 'SD' and sheet_name in ['220105BD2', '1', '14', '18', '20', '22']) or (condition in ['SD+QIH(24h)', 'QIH(1h)']):
                            bsl_data = stage_dfs[stage][sheet_name].iloc[i + int(row_num * 0.5)]    
            
                        else:
                            bsl_data = stage_dfs[stage][sheet_name].iloc[[i, i+row_num]].mean()
            
                        post_data = stage_dfs[stage][sheet_name].iloc[i+int(row_num * coeff)]                    
                        bsl_freq_data.append(bsl_data.iloc[freq])
                        post_freq_data.append(post_data.iloc[freq])

                    # Normality test
                    normality_bsl = stats.shapiro(bsl_freq_data).pvalue
                    normality_post = stats.shapiro(post_freq_data).pvalue

                    # Homogeneity of variance test
                    equal_var = stats.levene(bsl_freq_data, post_freq_data).pvalue > 0.05

                    # Selection of t-test
                    if normality_bsl > 0.05 and normality_post > 0.05:
                        if equal_var:
                            t_test = stats.ttest_rel(bsl_freq_data, post_freq_data)
                        else:
                            t_test = stats.ttest_ind(bsl_freq_data, post_freq_data, equal_var=False)
                    else:
                        t_test = stats.wilcoxon(bsl_freq_data, post_freq_data)

                    p_values.append(t_test.pvalue)
                    test_results.loc[freq, 'p'] = t_test.pvalue

                p_values_adj = multipletests(p_values, alpha=0.05, method='fdr_bh')[1]
                
                for freq in range(40):
                    test_results.loc[freq, 'q'] = p_values_adj[freq]
                test_results.to_excel(writer, sheet_name=f'{i * 2}-{(i * 2) + 2}', index=False)



    # Plot
    mean_dfs = {'W': W_mean_df, 'NR': NR_mean_df, 'R': R_mean_df}
    sem_dfs = {'W': W_sem_df, 'NR': NR_sem_df, 'R': R_sem_df}
    colors = {'W': (1, 0.5882, 0.5882), 'NR': (0.5333, 0.7412, 0.9882), 'R': (0.4392, 0.6784, 0.2784)}
    bsl_color = (0.7, 0.7, 0.7)

    plt.rcParams.update({'font.size': 20})
    for stage, color in colors.items():

        fig, axs = plt.subplots(2, 6, figsize=(36, 20))
        plt.suptitle(stage, fontsize=28)

        for i in range(row_num):
            x = mean_dfs[stage].columns
            if condition == 'SD' or condition == 'SD+QIH(24h)' or condition =='QIH(1h)':
                y_bsl = mean_dfs[stage].iloc[i + int(row_num * 0.5)]
                err_bsl = sem_dfs[stage].iloc[i + int(row_num * 0.5)] 
            else:    
                y_bsl = mean_dfs[stage].iloc[[i, i+row_num]].mean()
                err_bsl = sem_dfs[stage].iloc[[i, i+row_num]].mean()
            y_post = mean_dfs[stage].iloc[i+int(row_num * coeff)]
            err_post = sem_dfs[stage].iloc[i+int(row_num * coeff)]
            
            axs[i // 6, i % 6].errorbar(x, y_bsl, yerr=err_bsl, fmt='o', color=bsl_color, capsize=5, label='BSL' if i == 0 else "", linestyle='-')
            axs[i // 6, i % 6].errorbar(x, y_post, yerr=err_post, fmt='o', color=color, capsize=5, label='Post' if i == 0 else "", linestyle='-')
            axs[i // 6, i % 6].set_xlabel('Frequency (Hz)')
            axs[i // 6, i % 6].set_xticks(np.arange(0, len(x) + 1, len(x) / 4).astype(int))
            axs[i // 6, i % 6].set_xticklabels(np.round(np.arange(0, len(x) + 1, len(x) / 4) / 2).astype(str))
            axs[i // 6, i % 6].set_xlim([-1, len(x) + 1])
            axs[i // 6, i % 6].set_ylabel('Power')
            axs[i // 6, i % 6].spines['top'].set_visible(False)
            axs[i // 6, i % 6].spines['right'].set_visible(False)
            if condition == 'QIH(72h)':
                y_factor = 2
            else:
                y_factor = 1

            if stage == 'W':
                axs[i // 6, i % 6].set_yticks(np.arange(0, 2.1 * y_factor, 0.5 * y_factor))
                axs[i // 6, i % 6].set_yticklabels(np.arange(0, 2.1 * y_factor, 0.5 * y_factor).astype(str))
                axs[i // 6, i % 6].set_ylim([0, 2 * y_factor])
            elif stage == 'NR':
                axs[i // 6, i % 6].set_yticks(np.arange(0, 8.1, 2))
                axs[i // 6, i % 6].set_yticklabels(np.arange(0, 8.1, 2).astype(str))
                axs[i // 6, i % 6].set_ylim([0, 8])
            else:
                axs[i // 6, i % 6].set_yticks(np.arange(0, 3.1 * y_factor, 1 * y_factor))
                axs[i // 6, i % 6].set_yticklabels(np.arange(0, 3.1 * y_factor, 1 * y_factor).astype(str))
                axs[i // 6, i % 6].set_ylim([0, 3 * y_factor])
            axs[i // 6, i % 6].set_title(f't : {i * 2}-{(i + 1) * 2}')

            q_values = pd.read_excel(f'{base_path}Band_analysis/{condition}/Power_spectrum/test_result_{stage}.xlsx', sheet_name=f'{i * 2}-{(i * 2) + 2}')
            q_values = q_values.reset_index(drop=True)
            for freq in range(40):
                try:
                    if q_values.loc[freq, 'q'] <= 0.05:
                        y_min, y_max = axs[i // 6, i % 6].get_ylim()
                        axs[i //6, i % 6].hlines(y_max * 0.9, xmin=freq-0.5, xmax=freq+0.5, linewidth=10, color=color)
                except KeyError:
                    pass
        fig.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=28)
        

        # Save figure
        output_path = f'{base_path}Band_analysis/{condition}/Power_spectrum/{condition}_{stage}.tif'
        plt.savefig(output_path, format='tiff', dpi=350)    






for condition, coeff in conditions.items():

    power_per_hour(base_path, condition, coeff)
    
    print(f'{condition}_Power spectrum  <Done>')




