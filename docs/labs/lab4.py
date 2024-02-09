import sys
import logging

import os
from pathlib import Path
from pprint import pprint as pp
import torch
from torchmetrics.classification import MulticlassAccuracy
from additional_metrics import additional_metrics
import json
from torch import nn
from chop.passes.graph.utils import get_parent_name
from copy import deepcopy
# figure out the correct path
# machop_path = "/home/wfp23/ADL/mase/machop"
# assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append("/home/wfp23/ADL/mase/machop/")

from chop.actions import train, test
from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity, get_logger
from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph
from chop.models import get_model_info, get_model

set_logging_verbosity("info")

logger = get_logger("chop")
logger.setLevel(logging.INFO)

batch_size = 256
model_name = "jsc-three-linear-layers"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = {"x": next(iter(data_module.train_dataloader()))[0]}

# define a new model
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear  2
            nn.Linear(16, 16),  # linear  3
            nn.Linear(16, 5),   # linear  4
            nn.ReLU(5),  # 5
        )

    def forward(self, x):
        return self.seq_blocks(x)


model = JSC_Three_Linear_Layers()

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)
mg, _ = init_metadata_analysis_pass(mg, None)

def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        if name is not None:
            ori_module = graph.modules[node.target]
            in_features = ori_module.in_features
            out_features = ori_module.out_features
            bias = ori_module.bias
            if name == "output_only":
                out_features = out_features * config["channel_multiplier"]
            elif name == "both":
                in_features = in_features * config["channel_multiplier"]
                out_features = out_features * config["channel_multiplier"]
            elif name == "input_only":
                in_features = in_features * config["channel_multiplier"]
            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
    return graph, {}



pass_config = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_2": {
    "config": {
        "name": "output_only",
        # weight
        "channel_multiplier": 2,
        }
    },
"seq_blocks_3": {
    "config": {
        "name": "both",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "input_only",
        "channel_multiplier": 2,
        }
    },
}

# this performs the architecture transformation based on the config
mg, _ = redefine_linear_transform_pass(
    graph=mg, pass_args={"config": pass_config})


# define a new model
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear seq_2
            nn.ReLU(16),  # 3
            nn.Linear(16, 16),  # linear seq_4
            nn.ReLU(16),  # 5
            nn.Linear(16, 5),  # linear seq_6
            nn.ReLU(5),  # 7
        )

    def forward(self, x):
        return self.seq_blocks(x)

'''

Question 1


'''

model = JSC_Three_Linear_Layers()
mg = MaseGraph(model)
 
print("Original Graph:")
for block in mg.model.seq_blocks._modules:
  print(f"Block number {block}: {mg.model.seq_blocks._modules[block]}")

pass_config = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_2": {
    "config": {
        "name": "output_only",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "both",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_6": {
    "config": {
        "name": "input_only",
        "channel_multiplier": 2,
        }
    },
}

mg, _ = redefine_linear_transform_pass(
    graph=mg, pass_args={"config": pass_config})


print("\nTransformed Graph:")
for block in mg.model.seq_blocks._modules:
  print(f"Block number {block}: {mg.model.seq_blocks._modules[block]}")


'''

Question 2

'''
def brute_force_search(search_spaces, json_file):
    best_accuracy = 0.0
    best_search_space = None
    for search_space in search_spaces:
        model = JSC_Three_Linear_Layers()
        mg = MaseGraph(model)
        mg, _ = redefine_linear_transform_pass(
        graph=mg, pass_args={"config": search_space})

        print("\nTransformed Graph:")
        for block in mg.model.seq_blocks._modules:
            print(f"Block number {block}: {mg.model.seq_blocks._modules[block]}")

        train(mg.model, model_info, data_module, data_module.dataset_info, task, optimizer, learning_rate, weight_decay, plt_trainer_args, auto_requeue, save_path, visualizer, load_name, load_type)
        
        metrics = test(mg.model, model_info, data_module, data_module.dataset_info, task, optimizer, learning_rate, weight_decay, plt_trainer_args, auto_requeue, save_path, visualizer, load_name, load_type, return_metrics=True)
        print(metrics)
        accuracy = metrics[0]['test_acc_epoch']
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_search_space = search_space
        data_to_store = {
            "search_space": search_space,
            "accuracy": metrics[0]['test_acc_epoch'],
            "loss": metrics[0]['test_loss_epoch']
        }
        # store to json
        with open(json_file, 'w') as f:
            json.dump(data_to_store, f)

    return best_accuracy, best_search_space

pass_config = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_2": {
    "config": {
        "name": "output_only",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "both",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_6": {
    "config": {
        "name": "input_only",
        "channel_multiplier": 2,
        }
    },
}

task = "channel_multiplier"
dataset_name = "jsc"
num_workers = os.cpu_count()
optimizer = "adam"
max_epochs: int = 2
# max_steps: int = -1
gradient_accumulation_steps: int = 1
learning_rate: float = 5e-3
weight_decay: float = 0.0
lr_scheduler_type: str = "linear"
num_warmup_steps: int = 0
save_path: str = "./ckpts/chMultiplier"
auto_requeue = False
load_name: str = None
load_type: str = ""
evaluate_before_training: bool = False
visualizer = None
profile: bool = True
plt_trainer_args = {
"max_epochs": max_epochs,
"accelerator": "cpu",
}

# build a search space
channel_multipliers = [1, 2, 3, 4, 5, 6]
search_spaces = []
for multiplier in channel_multipliers:
    pass_config['seq_blocks_2']["config"]["channel_multiplier"] = multiplier
    pass_config['seq_blocks_4']["config"]['channel_multiplier'] = multiplier
    pass_config['seq_blocks_6']["config"]['channel_multiplier'] = multiplier
    search_spaces.append(deepcopy(pass_config))

# find the best accuracy and the best multipliers, json results are also stored
best_accuracy, best_search_space = brute_force_search(search_spaces, json_file="/home/wfp23/ADL/mase/docs/labs/channel_multiplier_search.json")

print(f"Best accuracy: {best_accuracy}")
print(f"Best search space: {best_search_space}")

'''

Question 3


'''
# batch_size = 128
 
# model_name = "jsc-three-linear-layers"
# dataset_name = "jsc"
# task = "cls"
 
# model_info = get_model_info(model_name)
# dataset_info = get_dataset_info(dataset_name)
 
# data_module = MaseDataModule(
#     name=dataset_name,
#     batch_size=batch_size,
#     model_name=model_name,
#     num_workers=0,
# )
 
# data_module.prepare_data()
# data_module.setup()
 
# plt_trainer_args = {
#     "max_epochs": 5,
#     "max_steps": -1,
#     "devices": 1,
#     "num_nodes": 1,
#     "accelerator": 'gpu',
#     "strategy": 'auto',
#     "fast_dev_run": False,
#     "precision": "16-mixed",
#     "accumulate_grad_batches": 1,
#     "log_every_n_steps": 50,
# }

def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        if name is not None:
            ori_module = graph.modules[node.target]
            in_features = ori_module.in_features
            out_features = ori_module.out_features
            bias = ori_module.bias
            if name == "output_only":
                out_features = out_features * config["channel_multiplier"]
            elif name == "both":
                in_features = in_features * config["channel_multiplier_in"]
                out_features = out_features * config["channel_multiplier_out"]
            elif name == "input_only":
                in_features = in_features * config["channel_multiplier"]
            new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
    return graph, {}
 
 
# def brute_force(search_spaces):
 
#   best_acc = 0
 
#   recorded_accs = []
 
#   for i, config in enumerate(search_spaces):
#     model = JSC_Three_Linear_Layers()
#     config = copy.deepcopy(config)
 
#     mg = MaseGraph(model=model)
#     mg, _ = init_metadata_analysis_pass(mg, None)
 
#     print("Original Graph:")
#     for block in mg.model.seq_blocks._modules:
#       print(f"Block number {block}: {mg.model.seq_blocks._modules[block]}")
 
#     mg, _ = redefine_linear_transform_pass(mg, {"config": config})
 
#     print("Expanded Graph:")
#     for block in mg.model.seq_blocks._modules:
#       print(f"Block number {block}: {mg.model.seq_blocks._modules[block]}")
 
#     model = mg.model
 
#     input_generator = InputGenerator(
#         data_module=data_module,
#         model_info=model_info,
#         task="cls",
#         which_dataloader="train",
#     )
 
#     train(model, model_info, data_module, dataset_info, task,
#           optimizer="adam", learning_rate=1e-5, weight_decay=0,
#           plt_trainer_args=plt_trainer_args, auto_requeue=False,
#           save_path=None, visualizer=None, load_name=None, load_type=None)
 
#     test_results = test(model, model_info, data_module, dataset_info, task,
#                         optimizer="adam", learning_rate=1e-5, weight_decay=0,
#                         plt_trainer_args=plt_trainer_args, auto_requeue=False,
#                         save_path=None, visualizer=None, load_name=None, load_type=None,
#                       return_test_results=True)
 
#     acc_avg = test_results[0]['test_acc_epoch']
#     loss_avg = test_results[0]['test_loss_epoch']
#     recorded_accs.append(acc_avg)
 
#     if acc_avg > best_acc:
#       best_acc = acc_avg
#       best_multiplier_1 = config['seq_blocks_2']['config']['channel_multiplier']
#       best_multiplier_2 = config['seq_blocks_6']['config']['channel_multiplier']
 
#   return best_acc, best_multiplier_1, best_multiplier_2, recorded_accs

### Define Search Space
pass_config = {
"by": "name",
"default": {"config": {"name": None}},
"seq_blocks_2": {
    "config": {
        "name": "output_only",
        "channel_multiplier": 2,
        }
    },
"seq_blocks_4": {
    "config": {
        "name": "both",
        "channel_multiplier_in": 2,
        "channel_multiplier_out": 2,
        }
    },
"seq_blocks_6": {
    "config": {
        "name": "input_only",
        "channel_multiplier": 2,
        }
    },
}

channel_multipliers = [1, 2, 3, 4, 5, 6]
 
search_spaces = []
for channel_multiplier_1 in channel_multipliers:
  for channel_multiplier_2 in channel_multipliers:
    pass_config['seq_blocks_2']['config']['channel_multiplier'] = channel_multiplier_1
    pass_config['seq_blocks_4']['config']['channel_multiplier_in'] = channel_multiplier_1
    pass_config['seq_blocks_4']['config']['channel_multiplier_out'] = channel_multiplier_2
    pass_config['seq_blocks_6']['config']['channel_multiplier'] = channel_multiplier_2
    search_spaces.append(deepcopy(pass_config))
 
metric = MulticlassAccuracy(num_classes=5)

# find the best accuracy and the best multiplier, json results are also stored
best_accuracy, best_search_space = brute_force_search(search_spaces, json_file="/home/wfp23/ADL/mase/docs/labs/channel_multiplier_search.json")

print(f"Best accuracy: {best_accuracy}")
print(f"Best search space: {best_search_space}")