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