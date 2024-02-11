import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the new JSON data
file_path = '/home/wfp23/ADL/mase/docs/imgs/labs/Channel_Multiplier/channel_multiplier_search_Q3.json'
with open(file_path, 'r') as file:
    new_data = json.load(file)

# Process the new data to extract configurations and accuracies
configurations = []
accuracies = []

for entry in new_data:
    # Extracting configurations with channel multipliers for seq_blocks 2, 4, 6
    config_seq = [entry["search_space"]["seq_blocks_2"]["config"].get("channel_multiplier", 1),
                  entry["search_space"]["seq_blocks_4"]["config"].get("channel_multiplier_out", 
                  entry["search_space"]["seq_blocks_4"]["config"].get("channel_multiplier_in", 1)),
                  entry["search_space"]["seq_blocks_6"]["config"].get("channel_multiplier", 1)]
    configurations.append(config_seq)
    accuracies.append(entry["accuracy"])

# Convert configurations into a sortable format (tuples for natural sort)
config_tuples = [tuple(config) for config in configurations]

# Create a combined list of tuples for sorting (configuration tuple, accuracy)
combined_list = list(zip(config_tuples, accuracies))

# Sort based on the configuration tuple
sorted_combined_list = sorted(combined_list, key=lambda x: x[0])

# Unzip the sorted list into sorted configurations and accuracies
sorted_configurations, sorted_accuracies = zip(*sorted_combined_list)

# Convert sorted configurations back to string representation for plotting
sorted_config_strs = [f"{config[0]}-{config[1]}-{config[2]}" for config in sorted_configurations]

# Creating a DataFrame for sorted accuracies with sorted configuration as index
df_sorted_accuracies_config = pd.DataFrame({"Accuracy": sorted_accuracies}, index=sorted_config_strs)
df_sorted_accuracies_config.index.name = "Configuration (Seq Blocks 2-4-6 Channel Multipliers)"

# Plotting the sorted configurations
ax = df_sorted_accuracies_config.plot(kind='bar', figsize=(14, 8), color='darkslategray', width=0.8)

# plt.title('Sorted Accuracy per Configuration (Channel Multipliers for Seq Blocks 2, 4, 6)')
plt.ylabel('Accuracy', fontsize=18, labelpad=7)
plt.xlabel('Channel Multiplier Configurations for Seq Blocks 2, 4, 6', fontsize=18,labelpad=20)
plt.xticks(rotation=90)
plt.tight_layout()
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14) 
ax.legend([])
plt.show()
plt.savefig('/home/wfp23/ADL/mase/docs/imgs/labs/Channel_Multiplier/channel_multiplier_visual_Q3.png')
