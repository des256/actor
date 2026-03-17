"""Inspect ONNX model graph structure to find ops that TRT might mishandle."""
import onnx

model = onnx.load("onnx/model.onnx", load_external_data=False)
graph = model.graph

# Count op types
op_counts: dict[str, int] = {}
for node in graph.node:
    op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

print("=== Op type counts ===")
for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
    print(f"  {op}: {count}")

print(f"\n=== Model opset ===")
for opset in model.opset_import:
    print(f"  domain='{opset.domain}' version={opset.version}")

# Look for potential TRT-problematic ops
problematic = ["Where", "Trilu", "ScatterND", "GatherND", "Einsum", "NonZero"]
print("\n=== Potentially problematic ops for TRT ===")
for op in problematic:
    if op in op_counts:
        print(f"  {op}: {op_counts[op]} occurrences")
        # Show first few nodes of this type
        for node in graph.node:
            if node.op_type == op:
                print(f"    name={node.name}, inputs={list(node.input)[:3]}, outputs={list(node.output)[:2]}")
                break

# Check for Cast ops (precision issues)
print("\n=== Cast ops ===")
cast_count = 0
for node in graph.node:
    if node.op_type == "Cast":
        cast_count += 1
        if cast_count <= 10:
            to_type = None
            for attr in node.attribute:
                if attr.name == "to":
                    to_type = attr.i
            type_names = {1: "FLOAT", 10: "FLOAT16", 7: "INT64", 6: "INT32", 9: "BOOL"}
            print(f"  {node.name}: to={type_names.get(to_type, to_type)}, "
                  f"inputs={list(node.input)[:2]}, outputs={list(node.output)[:1]}")
if cast_count > 10:
    print(f"  ... ({cast_count} total)")

# Check initializer dtypes
print("\n=== Initializer dtype distribution ===")
dtype_counts: dict[int, int] = {}
for init in graph.initializer:
    dtype_counts[init.data_type] = dtype_counts.get(init.data_type, 0) + 1
dtype_names = {1: "FLOAT32", 10: "FLOAT16", 7: "INT64", 6: "INT32", 9: "BOOL"}
for dt, count in sorted(dtype_counts.items()):
    print(f"  {dtype_names.get(dt, f'type_{dt}')}: {count}")

# Look at RoPE-related nodes (common source of TRT issues)
print("\n=== Nodes with 'cos' or 'sin' or 'rope' in name (RoPE) ===")
rope_nodes = [n for n in graph.node if any(kw in n.name.lower() for kw in ["cos", "sin", "rope", "rotary"])]
for node in rope_nodes[:10]:
    print(f"  {node.op_type}: {node.name}")
    print(f"    inputs: {list(node.input)}")
    print(f"    outputs: {list(node.output)}")
if len(rope_nodes) > 10:
    print(f"  ... ({len(rope_nodes)} total)")

# Check for any nodes that produce INT64 outputs that TRT might truncate to INT32
print("\n=== Graph inputs ===")
for inp in graph.input:
    shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
    dtype = inp.type.tensor_type.elem_type
    print(f"  {inp.name}: dtype={dtype_names.get(dtype, dtype)}, shape={shape}")

print("\n=== Graph outputs ===")
for out in graph.output:
    shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
    dtype = out.type.tensor_type.elem_type
    print(f"  {out.name}: dtype={dtype_names.get(dtype, dtype)}, shape={shape}")
