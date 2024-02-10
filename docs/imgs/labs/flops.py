import torch
import torch.nn as nn
import numpy as np
import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp

from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph
from chop.models import get_model_info, get_model
from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import get_logger
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)

from calc_modules_modified import calculate_modules
# from chop.passes.graph.analysis.flop_estimator.calculator import calculate_modules


# %cd /home/wfp23/ADL/mase/machop
# !./ch --help

logger = get_logger("chop")
logger.setLevel(logging.INFO)

batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
    # custom_dataset_cache_path="../../chop/dataset"
)
data_module.prepare_data()
data_module.setup()

model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False,
    checkpoint = None)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

mg = MaseGraph(model=model)
mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})


from chop.passes.graph.utils import get_node_actual_target

flop_calculations = {}
total_flops = 0
for node in mg.fx_graph.nodes:
    try:
        in_data = (node.meta['mase'].parameters['common']['args']['data_in_0']['value'],)
    except KeyError:
        in_data = (None,)
    out_data = (node.meta['mase'].parameters['common']['results']['data_out_0']['value'],)

    module = get_node_actual_target(node)
    if isinstance(module, torch.nn.Module):
        module_flops = calculate_modules(module, in_data, out_data)
        flop_calculations[module] = module_flops
        total_flops += module_flops['computations']

print("Flop Caluclation Breakdown: ", flop_calculations)
print("\nTotal Flops: ", total_flops)
