from copy import copy, deepcopy
import logging
from torch import nn
from torch.nn import ReLU
from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass

from ...utils import (
    get_node_actual_target,
    get_parent_name,
)

logger = logging.getLogger(__name__)

CHANNEL_OP = (
    "linear",
    "relu",
    "batchnorm1d",
)
    

def redefine_transform_pass(graph, pass_args=None):
    """
    Redefines the transformation pass of a given graph based on specific configuration parameters.

    :param graph: The input graph to be transformed.
    :type graph: MaseGraph

    :param pass_args: Additional arguments for the transformation, including configuration details for each layer.
    :type pass_args: dict, optional

    :return: The transformed graph along with an empty dictionary (placeholder for future use).
    :rtype: tuple

    :raises ValueError: If the default configuration is not provided in `pass_args`.

    This function iterates through each node in the graph's function representation (fx_graph). It applies transformation
    passes according to the configuration provided in `pass_args`. These transformations include redefining linear, ReLU,
    and BatchNorm1d layers with new parameters. The function ensures that the transformations are applied correctly by
    adjusting parameters such as input/output features and layer-specific configurations based on the node's characteristics
    and the provided configuration. The transformation logic includes handling for different naming schemes and
    adjustments based on previous layer configurations to ensure consistency in the transformed graph. It is essential
    to provide a default configuration in `pass_args` to avoid a ValueError.
    """
    def create_linear_layer(in_features, out_features, bias):
        # Creates a linear layer for a neural network with specified input/output features and bias option.
        if bias is not None:  # Ensures bias is explicitly enabled if provided; this condition is ineffective due to logic error.
            bias = True
        return nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias)  # Returns an instance of the linear layer.

    def create_relu_inplace(inplace):
        # Returns a ReLU (Rectified Linear Unit) activation layer, with an option to do the operation in-place.
        return ReLU(inplace)  # Inplace determines if the input tensor is modified directly.

    def create_batchnorm_1d_layer(num_features, eps, momentum, affine, track_running_stats):
        # Initializes a 1D batch normalization layer with specific parameters.
        return nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)  # Returns an instance of BatchNorm1d.
    
    main_config = pass_args.pop('config')

    default = main_config.pop('default', None)

    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    pre_in = 1
    pre_out = 1
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)
        
        actual_target = get_node_actual_target(node)
        # Process Linear layers
        if isinstance(actual_target, nn.Linear):
            if name is not None:
                if node.target=='x' or node.target=='output':
                    continue
                ori_module = graph.modules[node.target]
                in_features = config.get('in_features', 16)
                out_features = config.get('out_features', 16)
                bias = ori_module.bias
                if name == "output_only":
                    in_features = ori_module.in_features
                    out_features = out_features * config["channel_multiplier"]
                    pre_out=config["channel_multiplier"]
                elif name == "both":
                    in_features = in_features * pre_out
                    out_features = out_features * config["channel_multiplier"]
                    pre_out = pre_in
                    pre_in = config["channel_multiplier"]
                elif name == "input_only":
                    in_features = in_features * pre_in
                    out_features = ori_module.out_features
                new_module = create_linear_layer(in_features, out_features, bias)
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)
            
        # Process ReLU layers
        elif isinstance(actual_target, ReLU):
            name = config.get("name")
            if name:
                ori_module = graph.modules[node.target]
                new_module = create_relu_inplace(ori_module.inplace)
                setattr(graph.modules[node.target], "inplace", new_module.inplace)
        
        # Process BatchNorm1d layers
        elif isinstance(actual_target, nn.BatchNorm1d):
            name = config.get("name")
            if name:
                ori_module = graph.modules[node.target]
                # Instantiate a new BatchNorm1d with the original module's parameters
                new_module = create_batchnorm_1d_layer(
                    ori_module.num_features, ori_module.eps, ori_module.momentum, 
                    ori_module.affine, ori_module.track_running_stats)
                parent_name, child_name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], child_name, new_module)           
    return graph, {}