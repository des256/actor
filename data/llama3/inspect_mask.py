"""Trace the attention mask subgraph in the ONNX model to find TRT issues."""
import onnx

model = onnx.load("onnx/model.onnx", load_external_data=False)
graph = model.graph

# Build a map from output name to node
output_to_node: dict[str, object] = {}
for node in graph.node:
    for out in node.output:
        output_to_node[out] = node

# Find layer 0 attention mask subgraph
# Start from the Where node and trace backward
where_node = None
for node in graph.node:
    if node.op_type == "Where" and "layers.0" in node.name:
        where_node = node
        break

if where_node is None:
    print("Where node not found!")
    exit(1)

print(f"=== Where node: {where_node.name} ===")
print(f"  Inputs: {list(where_node.input)}")
print(f"  Outputs: {list(where_node.output)}")

# Recursively trace backward from Where inputs
def trace_back(output_name, depth=0, visited=None):
    if visited is None:
        visited = set()
    if output_name in visited:
        return
    visited.add(output_name)

    indent = "  " * depth
    node = output_to_node.get(output_name)
    if node is None:
        # Check if it's a graph input
        for inp in graph.input:
            if inp.name == output_name:
                shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
                dtype_names = {1: "FLOAT32", 10: "FLOAT16", 7: "INT64", 6: "INT32", 9: "BOOL"}
                dtype = inp.type.tensor_type.elem_type
                print(f"{indent}[INPUT] {output_name}: dtype={dtype_names.get(dtype, dtype)}, shape={shape}")
                return
        # Check if it's an initializer
        for init in graph.initializer:
            if init.name == output_name:
                dtype_names = {1: "FLOAT32", 10: "FLOAT16", 7: "INT64", 6: "INT32", 9: "BOOL"}
                print(f"{indent}[INIT] {output_name}: dtype={dtype_names.get(init.data_type, init.data_type)}, "
                      f"dims={list(init.dims)}")
                return
        print(f"{indent}[???] {output_name}")
        return

    attrs = {}
    for attr in node.attribute:
        if attr.type == 1:  # FLOAT
            attrs[attr.name] = attr.f
        elif attr.type == 2:  # INT
            attrs[attr.name] = attr.i
        elif attr.type == 3:  # STRING
            attrs[attr.name] = attr.s.decode()
        elif attr.type == 4:  # TENSOR
            import numpy as np
            t = onnx.numpy_helper.to_array(attr.t)
            attrs[attr.name] = f"tensor({t.shape}, {t.dtype}, val={t.flat[:5]}...)" if t.size > 5 else f"tensor({t})"

    print(f"{indent}{node.op_type}: {node.name}")
    if attrs:
        print(f"{indent}  attrs: {attrs}")
    print(f"{indent}  inputs: {list(node.input)}")

    for inp in node.input:
        if inp:
            trace_back(inp, depth + 1, visited)


print("\n=== Tracing Where condition (input 0) ===")
trace_back(where_node.input[0])

print("\n=== Tracing Where true_value (input 1 — should be -inf) ===")
trace_back(where_node.input[1])

print("\n=== Tracing Where false_value (input 2 — attn_weights) ===")
# Only trace 2 levels for the attention weights path (it's large)
node = output_to_node.get(where_node.input[2])
if node:
    print(f"  {node.op_type}: {node.name}")
    print(f"    inputs: {list(node.input)}")
