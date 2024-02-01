# LAB 3

### 1. Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.
Three additional metrics which are very important to analyse are:

* **Latency** refers to the time it takes for an input to go through the model during the forward pass and produce an output - the inference time. This is a critical measurement of any model that has the intention of deployment into a real-life commercial setting. 

* **CPU/GPU Utilization** utilization can help identify how effectively a DL model uses the computational resources. High utilization rates might indicate good efficiency, whereas low utilization could suggest bottlenecks or inefficiencies in the model architecture, i.e. during synchronization and joining of threads. It also is directly proportional to power consumption which is a critical metric to understand if the model is feasible to train and if additional power supplies are required. By optimizing utilization, the model can achieve better performance without necessarily scaling up hardware resources, leading to cost savings.

* The number of **FLOPS** (Floating Point Operations Per Second) is calculated by the pre-defined function applied to each type of nn.module. This will correlate to both model-size and latency, although after optimization, like quantization, latency will decrease whilst FLOPS will not. FLOPS can help to understand the maximum computational capabilities of the hardware being used, and decide on hardware to be used both during training and inference.

### 2. Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric. It's important to note that in this particular case, accuracy and loss actually serve as the same quality metric (do you know why?).
The key metrics discussed in [Task 1](#1-explore-additional-metrics-that-can-serve-as-quality-metrics-for-the-search-process-for-example-you-can-consider-metrics-such-as-latency-model-size-or-the-number-of-flops-floating-point-operations-involved-in-the-model
) are now implemented in addition to accuracy and loss for the search-space defined. 

Accuracy and loss serve as the same quality metric since cross entropy Loss is defined as: 
$H(P^*, P) = -\sum_{i}{M} P^*(i) \log P(i)$
where $P^*$ is the true class probablity distribution and $P$ is the predicted class probablity distribution and $M$ is the number of classes. Since accuracy is measured using the `MulticlassAccuracy` metric which calculates the total accuracy for each of the classes collectively, cross entropy is inversley proportional, hence serves as the same metric and only one can be implemented. 

```python
import torch
from torchmetrics.classification import MulticlassAccuracy
import time
import subprocess
import psutil
import numpy as np
import matplotlib.pyplot as plt

mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

metric = MulticlassAccuracy(num_classes=5)
num_batchs = 5

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

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


def additional_metrics(mg, search_spaces, plot=True):
    """
    This function evaluates the performance of different quantization configurations on a neural network model by measuring various metrics such as accuracy, loss, latency, and hardware (GPU/CPU) utilizations. It iterates over a set of quantization configurations, applies each to the model, and performs inference on batches of data to collect performance metrics. The function aims to provide insights into the impact of quantization on model efficiency and hardware resource usage, facilitating the selection of optimal quantization settings for deployment.

    Inputs:
    - mg: A masegraph
    - search_spaces: An iterable containing different quantization configurations to be evaluated.
    - plot (bool): Indicates whether to plot the collected metrics against the quantization configurations.

    Outputs:
    - None. The function prints average values for accuracy, loss, latency, and hardware utilizations directly. 
    If the 'plot' argument is True, it also generates plots for these metrics.
    """
    # Initial hardware availability checks and setting baseline CPU power
    _, gpu_found = get_gpu_power_usage()
    _, cpu_found = get_cpu_utilization()

    # Lists to record various metrics over each search space configuration
    recorded_accs = []
    recorded_losses = []
    recorded_latencies = []
    recorded_gpu_utilizations = []
    recorded_cpu_utilizations = []
    latencies = []
    cpu_tdp = 25  # Assuming a Thermal Design Power (TDP) for the CPU of 25 watts

    # Iterating over each configuration in the search space
    for i, config in enumerate(search_spaces):
        # Apply quantization transformation based on current configuration
        mg, _ = quantize_transform_pass(mg, config)

        # Initialize accumulators for metrics
        j = 0
        acc_avg, loss_avg = 0, 0
        accs, losses, gpu_power_usages, cpu_power_usages = [], [], [], []

        # Iterate over batches in the training data loader
        for inputs in data_module.train_dataloader():
            xs, ys = inputs  # Unpack the input data tuple

            # Reset CPU utilization measurement to zero before the new batch starts
            _, _ = get_cpu_utilization()

            # Pre-batch GPU power measurement if GPU is available
            if gpu_found:
                gpu_power_before = sum(get_gpu_power_usage()[0])

            # Time and GPU power usage measurement before model prediction
            if gpu_found:
                start_gpu = torch.cuda.Event(enable_timing=True)
                end_gpu = torch.cuda.Event(enable_timing=True)
                start.record()

            # Record CPU-based start time
            start_time = time.time()
            preds = mg.model(xs)  # Model prediction
            end_time = time.time()  # Record CPU-based end time

            # Compute and record GPU latency if applicable
            latency_gpu = 0
            if gpu_found:
                end.record()
                torch.cuda.synchronize()
                latency_gpu = start_gpu.elapsed_time(end_gpu)
                latencies.append(latency_gpu * 1.0e6)  # Convert milliseconds to nanoseconds

            # Otherwise, compute and record CPU latency
            else:
                latencies.append((end_time - start_time) * 1.0e6)

            # Post-batch GPU power usage measurement if GPU is available
            if gpu_found:
                gpu_power_after = sum(get_gpu_power_usage()[0])
                gpu_power_used = gpu_power_after - gpu_power_before
                gpu_power_usages.append(gpu_power_used)

            # Compute and record CPU utilization and power usage
            if cpu_found:
                cpu_utilization, _ = get_cpu_utilization()
                estimated_cpu_power = (cpu_utilization / 100) * cpu_tdp
                cpu_power_usages.append(estimated_cpu_power)

            # Calculate and record the accuracy and loss for the batch
            acc = metric(preds, ys)
            accs.append(acc)
            loss = torch.nn.functional.cross_entropy(preds, ys)
            losses.append(loss)

            # Exit loop early if the specified number of batches is exceeded
            if j > num_batchs:
                break
            j += 1

        # Average the recorded metrics for the current configuration
        acc_avg = sum(accs) / len(accs)
        loss_avg = sum(losses) / len(losses)
        recorded_accs.append(acc_avg)
        recorded_losses.append(loss_avg)
        recorded_latencies.append(sum(latencies) / len(latencies))

        # Record average power utilizations if applicable
        if cpu_found:
            avg_cpu_utilizations = sum(cpu_power_usages) / len(cpu_power_usages)
            recorded_cpu_utilizations.append(avg_cpu_utilizations)
        if gpu_found:
            avg_gpu_utilizations = sum(gpu_power_usages) / len(gpu_power_usages)
            recorded_gpu_utilizations.append(avg_gpu_utilizations)

    # Convert tensors to floats for accuracy and loss
    recorded_accs = [tensor.item() for tensor in recorded_accs]
    recorded_losses = [tensor.item() for tensor in recorded_losses]

    # Print the average accuracy, loss, and latency per batch
    avg_acc = np.mean(recorded_accs)
    print(f"Average Accuracy per Batch: {avg_acc:.4g}")
    loss_avg = np.mean(recorded_losses)
    print(f"Average Loss per Batch: {loss_avg:.4g}")
    avg_latency = np.mean(recorded_latencies)
    print(f"Average Latency per Batch: {avg_latency:.2f} nanoseconds")

    if gpu_found:
        plot_metric_search_spaces(recorded_gpu_utilizations, 'Avg GPU Usage per Batch (W)', search_spaces)
    if cpu_found:
        plot_metric_search_spaces(recorded_cpu_utilizations, 'Avg CPU Usage per Batch (W)', search_spaces)
    return
```