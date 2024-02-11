import json
import pandas as pd
import matplotlib.pyplot as plt

file_path = '/home/wfp23/ADL/mase/mase_output/jsc-three-linear-layers/software/search_ckpts/log.json'

with open(file_path, 'r') as file:
    data = json.load(file)

# Extract necessary data
extracted_data = []

for key, values in data.items():
    for seq_block, config in values['user_attrs_sampled_config'].items():
        if seq_block in ['seq_blocks_2', 'seq_blocks_4', 'seq_blocks_6']:
            accuracy = values['user_attrs_scaled_metrics']['accuracy']
            channel_multiplier = config['config']['channel_multiplier']
            extracted_data.append({
                'Seq Block': seq_block,
                'Channel Multiplier': channel_multiplier,
                'Accuracy': accuracy
            })

# Extracting configurations and accuracies again to apply sorting
configurations = []
accuracies = []

for config_id, values in data.items():
    config_sequence = [values['user_attrs_sampled_config'][f'seq_blocks_{i}']['config']['channel_multiplier'] for i in [2, 4, 6]]
    configurations.append(config_sequence)
    accuracies.append(values['user_attrs_scaled_metrics']['accuracy'])

# Convert configurations into a sortable format (tuples for natural sort)
config_tuples = [tuple(config) for config in configurations]

# Create a combined list of tuples for sorting (configuration tuple, accuracy)
combined_list = list(zip(config_tuples, accuracies))

# Sort based on the configuration tuple
sorted_combined_list = sorted(combined_list, key=lambda x: x[0])

# Unzip the sorted list into sorted configurations and accuracies
sorted_configurations, sorted_accuracies = zip(*sorted_combined_list)

# Convert sorted configurations back to string representation for plotting
sorted_config_strs = [f"{config[0]} {config[1]} {config[2]}" for config in sorted_configurations]

# Creating a DataFrame for sorted accuracies with sorted configuration as index
df_sorted_accuracies_config = pd.DataFrame({"Accuracy": sorted_accuracies}, index=sorted_config_strs)
df_sorted_accuracies_config.index.name = "Configuration (Seq Blocks 2, 4, 6)"

# Exclude the last 4 bars
df_to_plot = df_sorted_accuracies_config.iloc[:-4]

# Plotting the sorted configurations without the last 4 bars
ax = df_to_plot.plot(kind='bar', figsize=(14, 8), legend=True, color='skyblue')
# plt.title('Sorted Accuracy per Configuration (Channel Multipliers for Seq Blocks 2, 4, 6) - Excluding Last 4')
plt.ylabel('Accuracy')
plt.xlabel('Configuration (Channel Multipliers)')
plt.xticks(rotation=45)
ax.legend([])

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

plt.xticks(ticks=range(len(sorted_config_strs)), labels=sorted_config_strs, rotation=45)
plt.tight_layout()

plt.show()
plt.savefig('/home/wfp23/ADL/mase/docs/imgs/labs/Q4/channel_multiplier_accuracy_modified.png')
