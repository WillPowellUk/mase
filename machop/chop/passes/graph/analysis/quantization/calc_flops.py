import numpy as np
import torch
from calc_modules_modified import calculate_modules
from chop.passes.graph.utils import get_node_actual_target

def calculate_flops_mg_analysis_pass(graph, pass_args: dict):
    """
    Calculate the floating-point operations (FLOPs) for the given graph. 
    This analysis helps in understanding the computational complexity and efficiency of the model, 
    especially useful for optimizing performance in resource-constrained environments.

    :param graph: The graph to analyze.
    :type graph: MaseGraph
    :param pass_args: Additional arguments for the analysis pass.
    :type pass_args: dict

    :return: A tuple containing the analyzed graph and a dictionary with FLOPs calculation details.
    :rtype: tuple
    :return graph: The analyzed graph.
    :rtype graph: MaseGraph
    :return dict: A dictionary with the following keys:
        - 'flop_module_breakdown' (dict): A breakdown of FLOPs by module.
        - 'total_flops' (int): The total number of floating-point operations for the graph.
    :rtype dict: dict
    """
    flop_calculations = {}
    total_flops = 0
    for node in graph.fx_graph.nodes:
        try:
            data_in = (node.meta['mase'].parameters['common']['args']['data_in_0']['value'],)
        except KeyError:
            data_in = (None,)
        data_out = (node.meta['mase'].parameters['common']['results']['data_out_0']['value'],)

        module = get_node_actual_target(node)
        if isinstance(module, torch.nn.Module):
            module_flops = calculate_modules(module, data_in, data_out)
            flop_calculations[module] = module_flops
            total_flops += module_flops['computations']

    print("Flop Caluclation Breakdown: ", flop_calculations)
    print("\nTotal Flops: ", total_flops)

    return graph, {"flop_module_breakdown": flop_calculations, "total_flops": total_flops}