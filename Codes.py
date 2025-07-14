import subprocess

# Python file list
base_path = '/mnt/d/Q project/Band_analysis/Codes/'
python_files = [
    f'{base_path}Power_per_epoch.py',
    f'{base_path}Power_per_hour.py',
    f'{base_path}Power_spectrum.py',
]

graphs = ['Power_per_epoch', 'Power_per_hour', 'Power_spectrum']

# Execution
for file, graph in zip(python_files, graphs):
    print(f'{graph}  <Start>')
    subprocess.run(['python', file])
    print(f'{graph}  <Done>')