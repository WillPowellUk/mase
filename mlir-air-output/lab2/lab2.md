## Turning you network to a graph
### 1. Explain the functionality of report_graph_analysis_pass and its printed jargons such as placeholder, get_attr … You might find the doc of torch.fx useful.

* As well as the other graph analysis passes, `add_common_metadata_analysis_pass` and `init_metadata_analysis_pass`, the `report_graph_analysis_pass` function generates a report for the Mase graph analysis and prints out an overview of the model in a table.
* It does not modify the graph like a tranform pass would, but outputs to the file if one is provided as an input otherwise prints to the console.
* It counts the different node operation types and module types found in the graph
* The graph contains sequence blocks which describes the Mase operators. For JSC_Tiny this would include BatchNorm1d, ReLU, Linear, ReLU with their respective parameters. 
* `Placeholders` correspond to the function parameters in the graph printout, in this lab's case the target 'x' and just represents a function input. 
* The `get_attr` operation is used to retrieve a specific attribute or parameter from a module.
* The `call_function` refers to the use of standalone, user-defined or built-in functions in the computation graph. If you define a function like myadd(x, y) which adds two values, and you use this function in your model, the FX symbolic tracer will represent this as a call_function node. The function is not bound to any particular object or module. It's a 'free' function that takes inputs and produces an output, independent of any object's state.
* The `call_module` encapsulates both data (parameters, states) and methods (like the forward method). When call_module is used, it's calling the forward method of that module with the given arguments.
* The `call_method` represents a method call on an object and is bound to the tensor object.

### 2. What are the functionalities of profile_statistics_analysis_pass and report_node_meta_param_analysis_pass respectively?

* `profile_statistics_analysis_pass` is a function used to gather and analyze statistical data from a graph related to weights and activations in the graph. The analysis methods such as `VariancePrecise` and `RangeQuantile` can be found in `stat.py`.
* `report_node_meta_param_analysis_pass` performs meta parameter analysis on the nodes in the graph, such as those passed to `profile_statistics_analysis_pass` and generates a report of this analayisis. In the `which` paramter selects either: `Common` which lists tensors with its shape, precision etc., `hardware` describing hardware based anaylsis and  `software` which shows paramters such as the `range_quantile` and `variance_precise`. 

### 3. Explain why only 1 OP is changed after the quantize_transform_pass.
Based on the configuration specified in `pass_args`, only modules of the `linear` type are impacted by the quantization transformation due to the `pass_args` whilst the other modules types are not modified. Other `QUANTIZEABLE_OP` could have been modified such as `relu`, `conv1d` etc. 

### 4. Write some code to traverse both mg and ori_mg, check and comment on the nodes in these two graphs. You might find the source code for the implementation of summarize_quantization_analysis_pass useful.
```
# Iterate through corresponding nodes in two graphs using zip
for mg_node, ori_mg_node in zip(mg.fx_graph.nodes, ori_mg.fx_graph.nodes):
    
    # Check if the types of the actual targets of the nodes are different
    if (type(get_node_actual_target(mg_node)) != type(get_node_actual_target(ori_mg_node))):
        
        # Get the types of the original and new modules
        original_module = type(get_node_actual_target(ori_mg_node))
        new_module = type(get_node_actual_target(ori_mg_node))
        
        # Print a message indicating a difference is found
        print(f'Difference found:')
        print(f'    Name: {mg_node.name}')  # Print the name of the node
        print(f'    Mase Type: {get_mase_type(mg_node)}')  # Print the Mase Type of the node
        print(f'    Mase Operation: {get_mase_op(mg_node)}')  # Print the Mase Operation of the node
        print(f'    Original Module: {original_module} --> New Module: {new_module}')  # Print module type differences
        
        # Get the weights of the nodes from their metadata
        mg_weight = mg_node.meta["mase"].parameters["common"]["args"]["weight"]
        ori_mg_weight = ori_mg_node.meta["mase"].parameters["common"]["args"]["weight"]
        
        # Print the weights of both nodes
        print(f"    mg Node: {mg_node.name}, Weight: {mg_weight} \n     ori_mg Node: {ori_mg_node.name}, Weight: {ori_mg_weight}")
```


### 5. Perform the same quantisation flow to the bigger JSC network that you have trained in lab1. You must be aware that now the `pass_args` for your custom network might be different if you have used more than the `Linear` layer in your network.

### 6. Write code to show and verify that the weights of these layers are indeed quantised. You might need to go through the source code of the implementation of the quantisation pass and also the implementation of the [Quantized Layers](../../machop/chop/passes/transforms/quantize/quantized_modules/linear.py) .