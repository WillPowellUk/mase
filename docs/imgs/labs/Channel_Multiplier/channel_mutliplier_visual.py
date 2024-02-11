import matplotlib.pyplot as plt
import json

with open('/home/wfp23/ADL/mase/docs/imgs/labs/Channel_Multiplier/channel_multiplier_search_Q2.json', 'r') as file:
    json_data = json.load(file)

# Extracting information for plotting
channel_multipliers = []
accuracies = []
for entry in json_data:
    # Assuming all channel_multipliers are the same within each search space
    cm = entry["search_space"]["seq_blocks_2"]["config"]["channel_multiplier"]
    accuracy = entry["accuracy"]
    channel_multipliers.append(cm)
    accuracies.append(accuracy)

# Plotting
fig, ax = plt.subplots()
ax.bar(channel_multipliers, accuracies, color='darkslategray')
ax.set_xlabel('Channel Multiplier for Each Linear Layer', fontsize=16)
ax.set_ylabel('Accuracy', fontsize=16)
# ax.set_ylim([0, 1])
# ax.set_title('Accuracy vs Channel Multiplier for each Search Space')
ax.set_xticks(channel_multipliers)
ax.set_xticklabels(channel_multipliers)

plt.show()
plt.savefig('/home/wfp23/ADL/mase/docs/imgs/labs/Channel_Multiplier/channel_multiplier_search_Q2.png')