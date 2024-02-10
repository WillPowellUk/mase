import json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os

# Base path for TPE and Brute-Force
base_path_tpe = '/home/wfp23/ADL/mase/mase_output/jsc-tiny-TPE_'
base_path_bf = '/home/wfp23/ADL/mase/mase_output/jsc-tiny_BF_'

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    config_numbers = []
    accuracies = []
    for key, value in data.items():
        config_number = int(key)
        accuracy = value.get("user_attrs_software_metrics", {}).get("accuracy")
        config_numbers.append(config_number)
        accuracies.append(accuracy)
    return config_numbers, accuracies

plt.figure(figsize=(10, 6))

# Loop through the files for TPE and Brute-Force
for i in range(1, 6):
    # Construct file paths
    file_path_tpe = base_path_tpe + str(i) + '/software/search_ckpts/log.json'

    # Load and extract data
    config_numbers_tpe, accuracies_tpe = load_data(file_path_tpe)

    # Plot TPE data
    plt.plot(config_numbers_tpe[0:2000], accuracies_tpe[0:2000], label=f'TPE {i}', color='b')


# Loop through the files for TPE and Brute-Force
for i in range(1, 6):
    # Construct file paths
    file_path_bf = base_path_bf + str(i) + '/software/search_ckpts/log.json'
    
    # Load and extract data
    config_numbers_bf, accuracies_bf = load_data(file_path_bf)
    
    # Plot Brute-Force data
    plt.plot(config_numbers_bf[0:2000], accuracies_bf[0:2000], label=f'BF {i}', color='r')

plt.xlabel('Search Space Trial', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
legend_items = [Line2D([0], [0], color='blue', lw=2, label='TPE'),
                Line2D([0], [0], color='red', lw=2, label='Brute Force')]

# Add the custom legend to the plot
plt.xticks(np.arange(0, 18, 1))
plt.legend(handles=legend_items)
plt.grid(True)
plt.show()
plt.savefig('tpe_vs_bruteforce_comparison.png')