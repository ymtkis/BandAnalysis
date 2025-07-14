import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_path = '/mnt/d/Q project/'
conditions = {''}
# 'QIH(24h)': 3, 'QIH(48h)': 4, 'QIH(72h)': 5, 'Ctrl_Stim(24h)': 3
bands = {'Delta': 45, 'Theta': 46, 'Alpha': 47, 'Low-beta': 48, 'High-beta': 49}
epoch_per_1d = 3600


def power_per_epoch(base_path, condition, coeff, band, col):
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


    # Plot
    plt.rcParams.update({'font.size': 20})
    plt.figure()
    colors = [(1, 1, 1), (1, 0.5882, 0.5882), (0.5333, 0.7412, 0.9882), (0.4392, 0.6784, 0.2784), (0.7, 0.7, 0.7)]

    for sheet_name in band_df.columns:
        band_data = band_df[sheet_name].values.flatten()
        label_data = label_indices[sheet_name].values.flatten()
        for band_col in range(len(band_data)):
            if  2 * epoch_per_1d < band_col < coeff * epoch_per_1d + 1:
                plt.plot(band_col, band_data[band_col], '.', color=colors[4], markersize=1, alpha=0.5)
            else:
                plt.plot(band_col, band_data[band_col], '.', color=colors[label_data[band_col]], markersize=1, alpha=0.5)

    plt.xlabel('Time (h)')
    plt.xticks(np.arange(0, (coeff + 3) * epoch_per_1d, epoch_per_1d), [str(24 * i) for i in range(coeff + 3)])
    plt.xlim([0, (coeff + 2) * epoch_per_1d + 1])
    plt.ylabel('Power')
    plt.yticks(np.arange(0, 11, 2), ['0', '2', '4', '6', '8', '10'])
    plt.ylim([0, 10])
    plt.title(band)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tight_layout()

    # Save figure
    output_path = f'{base_path}Band_analysis/{condition}/Power_per_epoch/{condition}_{band}.tif'
    plt.savefig(output_path, format='tiff', dpi=350)

for condition, coeff in conditions.items():
    for band, row in bands.items():
        power_per_epoch(base_path, condition, coeff, band, row)
        print(f'{condition}_{band}_Power per epoch  <Done>')
