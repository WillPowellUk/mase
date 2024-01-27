import json
import matplotlib.pyplot as plt

file_path = '/home/wfp23/ADL/mase/mase_output/JsonFiles/'

batch_sizes = [32, 128, 512]
train_files = [f"{file_path}train_acc_batch_{batch_size}.json" for batch_size in batch_sizes]
val_files = [f"{file_path}val_acc_batch_{batch_size}.json" for batch_size in batch_sizes]

# Initialize a dictionary to store accuracies
acc_data = {}

# Process training files
for file_name in train_files:
    with open(file_name, 'r') as file:
        # Extract batch size from file name
        batch_size = ''.join(filter(str.isdigit, file_name.split('_')[3]))
        # Read JSON data
        data = json.load(file)
        # Extract accuracies
        accuracies = [epoch[2] for epoch in data]
        # Store in dictionary
        acc_data[f"Training Batch {batch_size}"] = accuracies

# Process validation files
for file_name in val_files:
    with open(file_name, 'r') as file:
        # Extract batch size from file name
        batch_size = ''.join(filter(str.isdigit, file_name.split('_')[3]))
        # Read JSON data
        data = json.load(file)
        # Extract accuracies
        accuracies = [epoch[2] for epoch in data]
        # Store in dictionary
        acc_data[f"Validation Batch {batch_size}"] = accuracies

# Plotting
plt.figure(figsize=(10, 6))
for label, accuracies in acc_data.items():
    if 'Training' in label:
        plt.plot(accuracies, 'b--', label=label)  # Dotted line for training
    else:
        plt.plot(accuracies, 'g-', label=label)   # Solid line for validation

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.savefig("my_plot.png")
plt.show()

