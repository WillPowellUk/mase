# Lab 4

### It is unusual to sequence three linear layers consecutively without interposing any non-linear activations (do you know why?)
When you stack multiple linear layers without any non-linear activations between them, the entire stack is still mathematically equivalent to a single linear layer, hence it would make sense to condense this to one layer with the same equivalent neurons.

### Q1. Can you edit your code, so that we can modify the above network to have layers expanded to double their sizes? Note: you will have to change the ReLU also.
To double the linear layers, but keeping the input and output the same, we can modify the `pass_config` 
```python
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
```

### Q2. In [Lab 3](../lab3/lab3.md), we have implemented a grid search, can we use the grid search to search for the best channel multiplier value?
