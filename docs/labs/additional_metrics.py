import torch
from torchmetrics.classification import MulticlassAccuracy
import time
import subprocess
import psutil
import numpy as np
import matplotlib.pyplot as plt

from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)

metric = MulticlassAccuracy(num_classes=5)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def plot_metric_search_spaces(metric, title, x_label, y_label, search_space_config_x, search_space_config_y):
    """
    This function visualizes the performance metrics of different quantization configurations as a 3D bar chart. 
    It is designed to plot metrics (like accuracy, latency, etc.) across a two-dimensional grid of quantization 
    configurations, making it easier to compare the impact of different settings on model performance. 
    The metrics are plotted against two axes representing distinct dimensions of the search space (e.g., 
    data precision and weight precision configurations).

    Inputs:
    - metric: A list or array of metric values corresponding to each quantization configuration. 
            The length should match the product of the dimensions of the search space (e.g., 16 for a 4x4 grid).
    - title: The title of the plot, typically describing the metric being visualized.
    - x_label: Label for the x-axis, describing the dimension of the quantization configurations it represents.
    - y_label: Label for the y-axis, similarly describing its corresponding quantization dimension.
    - search_space_config_x: A list of labels for the x-axis ticks, representing distinct values in the first dimension of the search space.
    - search_space_config_y: A list of labels for the y-axis ticks, representing distinct values in the second dimension of the search space.

    Outputs:
    - None. The function directly creates and displays a 3D bar chart using matplotlib. The chart illustrates 
    the given metric across different configurations defined by `search_space_config_x` and `search_space_config_y`.

    Note: This function assumes that `metric` can be reshaped into a square grid (e.g., 4x4 for 16 configurations), 
    and it uses 3D plotting capabilities of matplotlib for visualization. It is particularly useful for analyzing 
    the effects of two varying quantization parameters on model performance in a visually intuitive manner.
    """
    # Reshape the metric values to fit a 4x4 grid (since we have 16 configurations)
    grid_shape = (4, 4)  # 4 data configurations x 4 weight configurations
    metric = np.array(metric).reshape(grid_shape)

    # Create the plot
    fig = plt.figure(figsize=(12,10))
    plt.tight_layout()

    ax = fig.add_subplot(111, projection='3d')

    # Adjust grid for plotting based on the new expectation
    y, x = np.meshgrid(np.arange(grid_shape[1]), np.arange(grid_shape[0]))
    x = x.flatten()
    y = y.flatten()
    z = np.zeros(grid_shape).flatten()

    # Bar width and depth
    dx = dy = 0.5

    # Plotting the metric as a 3D bar chart
    ax.bar3d(x, y, z, dx, dy, metric.T.flatten(), shade=True)

    # Set plot labels and title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticklabels([str(pair) for pair in search_space_config_x])
    ax.set_yticklabels([str(pair) for pair in search_space_config_y])

    ax.view_init(elev=30, azim=30)

    # Show the plot
    plt.show()
    plt.savefig(f'{title}.png')


def additional_metrics(mg, data_module, search_spaces, num_batchs, plot=True):
    """
    This function evaluates the performance of different quantization configurations on a neural network model 
    by measuring various metrics such as accuracy, loss, latency, and hardware (GPU/CPU) utilizations. 
    It iterates over a set of quantization configurations, applies each to the model, and performs inference 
    on batches of data to collect performance metrics. The function aims to provide insights into the impact 
    of quantization on model efficiency and hardware resource usage, facilitating the selection of optimal 
    quantization settings for deployment.

    Inputs:
    - mg: A model graph or model wrapper that supports quantization transformations and inference.
    - search_spaces: An iterable containing different quantization configurations to be evaluated.
    - plot (bool): Indicates whether to plot the collected metrics against the quantization configurations.

    Outputs:
    - None. The function prints average values for accuracy, loss, latency, and hardware utilizations directly. 
    If the 'plot' argument is True, it also generates plots for these metrics.
    """
    def get_gpu_power_usage():
        try:
            smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits']).decode().strip()
            power_usage = [float(x) for x in smi_output.split('\n')]  # power usage in watts
            return power_usage, True
        except Exception as e:
            print(f"{e}\nNo GPU found. Monitoring CPU usuage only.")
            return [], False

    def get_cpu_utilization(): 
        try:
            cpu_utilization = psutil.cpu_percent(interval=None)
            return cpu_utilization, True
        except Exception as e:
            print(f"{e}\nNo CPU found.")
            return [], False


    # Check for GPU / CPU
    _, gpu_found = get_gpu_power_usage()
    _, cpu_found = get_cpu_utilization()

    recorded_accs = []
    recorded_losses = []
    recorded_latencies = []
    recorded_gpu_utilizations = []
    recorded_cpu_utilizations = []
    latencies = []
    cpu_tdp = 25 # Assuming a Thermal Design Power for the CPU of 25 watts

    # Brute Force Search
    for i, config in enumerate(search_spaces):
        mg, _ = quantize_transform_pass(mg, config)
        j = 0
        acc_avg, loss_avg = 0, 0
        accs, losses, gpu_power_usages, cpu_power_usages  = [], [], [], []
        
        # Runner loop
        for inputs in data_module.train_dataloader():
            xs, ys = inputs

            # Reset CPU utilization measurement
            _, _ = get_cpu_utilization()  # Call once to reset the measurement

            # Measure GPU power usage before prediction and warm up the GPU
            if gpu_found:
                gpu_power_before = sum(get_gpu_power_usage()[0])
                steps = 10

                # Warmup GPU 
                for _ in range(steps):
                    mg.model(xs) # don't record time

                start_gpu = torch.cuda.Event(enable_timing=True)
                end_gpu = torch.cuda.Event(enable_timing=True)
                start.record()

            start_time = time.time()
            preds = mg.model(xs)  # Model prediction
            end_time = time.time()
            latency_gpu = 0
            # GPU latency is measured differently to CPU latency
            if gpu_found:
                end.record()
                torch.cuda.synchronize()  # Wait for GPU operations to finish/syncronize
                latency_gpu = start_gpu.elapsed_time(end_gpu) # measured in milliseconds
                latencies.append(latency_gpu * 1.0e6)  # Convert to nanoseconds
            else:
                latencies.append((end_time - start_time) * 1.0e6)  # Convert to nanoseconds

            # Measure GPU power usage after prediction
            if gpu_found:
                gpu_power_after = sum(get_gpu_power_usage()[0])
                gpu_power_used = (gpu_power_after - gpu_power_before)  # Measured in W
                gpu_power_usages.append(gpu_power_used)

            # Measure CPU utilization and estimate power usage
            if cpu_found:
                cpu_utilization, _ = get_cpu_utilization()  # Get CPU utilization over operation duration
                estimated_cpu_power = (cpu_utilization / 100) * cpu_tdp  # Measured in W
                cpu_power_usages.append(estimated_cpu_power)

            acc = metric(preds, ys)
            accs.append(acc)
            loss = torch.nn.functional.cross_entropy(preds, ys)
            losses.append(loss)

            if j > num_batchs:
                break
            j += 1

        acc_avg = sum(accs) / len(accs)
        loss_avg = sum(losses) / len(losses)
        if cpu_found:
            avg_cpu_utilizations = sum(cpu_power_usages) / len(cpu_power_usages)
            recorded_cpu_utilizations.append(avg_cpu_utilizations)
        if gpu_found:
            avg_gpu_utilizations = sum(gpu_power_usages) / len(gpu_power_usages)
            recorded_gpu_utilizations.append(avg_gpu_utilizations)
        recorded_accs.append(acc_avg)
        recorded_losses.append(loss_avg)
        recorded_latencies.append(sum(latencies) / len(latencies))

    # Convert each tensor to a float
    recorded_accs = [tensor.item() for tensor in recorded_accs]
    recorded_losses = [tensor.item() for tensor in recorded_losses]
    avg_acc = np.mean(recorded_accs)
    print(f"Average Accuracy per Batch: {avg_acc:.4g}")

    loss_avg = np.mean(recorded_losses)
    print(f"Average Loss per Batch: {loss_avg:.4g}")

    avg_latency = np.mean(recorded_latencies)
    print(f"Average Latency per Batch: {avg_latency:.2f} nanoseconds")

    if gpu_found:
        avg_gpu_power_usage = np.mean(recorded_cpu_utilizations)
        print(f"Average GPU Power Usage per Batch: {avg_gpu_power_usage:.4g}W")
    if cpu_found:
        avg_cpu_utilization = np.mean(recorded_cpu_utilizations)
        print(f"Average CPU Power Usage per Batch: {avg_cpu_utilization:.4g}W")

    if plot:
        plot_metric_search_spaces(recorded_accs, 'Avg Accuracy per Batch', 'Weights in Frac Widths Index', 'Data in Frac Widths Index', w_in_frac_widths, data_in_frac_widths)
        plot_metric_search_spaces(recorded_losses, 'Avg Loss per Batch', 'Weights in Frac Widths Index', 'Data in Frac Widths Index', w_in_frac_widths, data_in_frac_widths)
        plot_metric_search_spaces(recorded_latencies, 'Avg Latency per Batch (ns)', 'Weights in Frac Widths Index', 'Data in Frac Widths Index', w_in_frac_widths, data_in_frac_widths)

        if gpu_found:
            plot_metric_search_spaces(recorded_gpu_utilizations, 'Avg GPU Usage per Batch (W)', 'Weights in Frac Widths Index', 'Data in Frac Widths Index', w_in_frac_widths, data_in_frac_widths)
        if cpu_found:
            plot_metric_search_spaces(recorded_cpu_utilizations, 'Avg CPU Usage per Batch (W)', 'Weights in Frac Widths Index', 'Data in Frac Widths Index', w_in_frac_widths, data_in_frac_widths)
    return