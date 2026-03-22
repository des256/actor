"""Replace ScatterElements(axis=2) with GatherElements-based equivalent.

The mimi decoder's transformer KV cache updates use ScatterElements to write
new KV values at computed positions in a [1, 8, 250, 64] cache. TensorRT 10.15
has a Myelin runtime bug where shape inference fails for this operation when
the dynamic temporal dimension exceeds 15.

The fix: instead of scattering updates INTO the cache, we GATHER from updates
to BUILD the result. For each cache position p, we compute which update entry
wrote to it using the actual scatter start position (end_offset % 250), then
gather that value. Positions not written by this scatter keep the original data.

scatter_elements(data, indices, updates, axis=2)
  where indices[b,h,t,d] = (start + t) % 250

becomes:
  start       = indices[0,0,0,0]
  reverse[p]  = (p - start + 250) % 250      for each p in [0, 250)
  valid[p]    = reverse[p] < N                where N = updates.shape[2]
  gathered    = gather_elements(updates, clamp(reverse, 0, N-1), axis=2)
  result      = where(valid, gathered, data)
"""

import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def replace_scatter_elements(model_path, output_path):
    model = onnx.load(model_path)
    graph = model.graph

    # Find all ScatterElements with axis=2
    scatter_nodes = []
    for node in graph.node:
        if node.op_type == "ScatterElements":
            for a in node.attribute:
                if a.name == "axis" and a.i == 2:
                    scatter_nodes.append(node)

    if not scatter_nodes:
        print("No ScatterElements with axis=2 found")
        return

    print(f"Replacing {len(scatter_nodes)} ScatterElements(axis=2) nodes")

    # Use the first scatter's index and updates tensors for shared computation.
    # All 4 scatters (2 layers × k,v) use the same index pattern.
    first_indices = scatter_nodes[0].input[1]
    first_updates = scatter_nodes[0].input[2]

    P = "rev_"  # prefix for new node/tensor names

    # --- Add constant initializers ---
    consts = {
        f"{P}c0": np.int64(0),
        f"{P}c1": np.int64(1),
        f"{P}c250": np.int64(250),
        f"{P}c2_idx": np.array([2], dtype=np.int64),
        f"{P}flat_shape": np.array([-1], dtype=np.int64),
        f"{P}shape_4d": np.array([1, 1, 250, 1], dtype=np.int64),
        f"{P}expand_shape": np.array([1, 8, 250, 64], dtype=np.int64),
        f"{P}gather_idx": np.array([0], dtype=np.int64),
    }
    for name, arr in consts.items():
        graph.initializer.append(numpy_helper.from_array(arr, name=name))

    new_nodes = []

    # --- Extract start position from actual scatter indices ---

    # start = indices[0, 0, 0, 0] (= end_offset % capacity)
    # Flatten indices to 1D, take first element
    new_nodes.append(
        helper.make_node(
            "Reshape", [first_indices, f"{P}flat_shape"], [f"{P}idx_flat"]
        )
    )
    new_nodes.append(
        helper.make_node(
            "Gather", [f"{P}idx_flat", f"{P}gather_idx"], [f"{P}start_arr"],
            axis=0,
        )
    )
    new_nodes.append(
        helper.make_node("Squeeze", [f"{P}start_arr"], [f"{P}start"])
    )

    # N = Shape(updates)[2]
    new_nodes.append(
        helper.make_node("Shape", [first_updates], [f"{P}u_shape"])
    )
    new_nodes.append(
        helper.make_node(
            "Gather", [f"{P}u_shape", f"{P}c2_idx"], [f"{P}N_arr"], axis=0
        )
    )
    new_nodes.append(
        helper.make_node("Squeeze", [f"{P}N_arr"], [f"{P}N"])
    )

    # pos = Range(0, 250, 1)  — cache positions [0..249]
    new_nodes.append(
        helper.make_node(
            "Range", [f"{P}c0", f"{P}c250", f"{P}c1"], [f"{P}pos"]
        )
    )

    # reverse_idx = (pos - start + 250) % 250
    # This maps each cache position to which update entry wrote to it.
    # Adding 250 ensures the value is positive before the mod.
    new_nodes.append(
        helper.make_node("Sub", [f"{P}pos", f"{P}start"], [f"{P}pos_sub"])
    )
    new_nodes.append(
        helper.make_node("Add", [f"{P}pos_sub", f"{P}c250"], [f"{P}pos_add"])
    )
    new_nodes.append(
        helper.make_node("Mod", [f"{P}pos_add", f"{P}c250"], [f"{P}rev_idx"])
    )

    # valid = reverse_idx < N
    new_nodes.append(
        helper.make_node("Less", [f"{P}rev_idx", f"{P}N"], [f"{P}valid"])
    )

    # safe_reverse = Where(valid, reverse_idx, 0)
    new_nodes.append(
        helper.make_node(
            "Where",
            [f"{P}valid", f"{P}rev_idx", f"{P}c0"],
            [f"{P}safe_rev"],
        )
    )

    # Reshape [250] → [1, 1, 250, 1], then Expand to [1, 8, 250, 64]
    new_nodes.append(
        helper.make_node(
            "Reshape", [f"{P}safe_rev", f"{P}shape_4d"], [f"{P}idx_4d"]
        )
    )
    new_nodes.append(
        helper.make_node(
            "Expand", [f"{P}idx_4d", f"{P}expand_shape"], [f"{P}idx_exp"]
        )
    )

    # Reshape valid mask [250] → [1, 1, 250, 1] (broadcasts in Where)
    new_nodes.append(
        helper.make_node(
            "Reshape", [f"{P}valid", f"{P}shape_4d"], [f"{P}valid_4d"]
        )
    )

    # --- Per-scatter replacement nodes ---
    per_scatter = {}
    for i, sn in enumerate(scatter_nodes):
        data_in = sn.input[0]    # existing cache [1, 8, 250, 64]
        updates_in = sn.input[2]  # new values [1, 8, T, 64]
        out_name = sn.output[0]

        gathered = f"{P}gathered_{i}"
        per_scatter[sn.name] = [
            # GatherElements(updates, idx_exp, axis=2) → gathered
            helper.make_node(
                "GatherElements",
                [updates_in, f"{P}idx_exp"],
                [gathered],
                axis=2,
            ),
            # Where(valid_4d, gathered, data) → output
            helper.make_node(
                "Where", [f"{P}valid_4d", gathered, data_in], [out_name]
            ),
        ]

    # --- Rebuild graph node list ---
    scatter_names = {sn.name for sn in scatter_nodes}

    first_idx = next(
        i for i, n in enumerate(graph.node) if n.name in scatter_names
    )

    rebuilt = []
    for i, n in enumerate(graph.node):
        if i == first_idx:
            rebuilt.extend(new_nodes)
        if n.name in per_scatter:
            rebuilt.extend(per_scatter[n.name])
        else:
            rebuilt.append(n)

    del graph.node[:]
    graph.node.extend(rebuilt)

    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    src = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/pocket/ckpt/mimi_decoder_model.onnx"
    )
    dst = (
        sys.argv[2]
        if len(sys.argv) > 2
        else src.replace(".onnx", "_no_scatter.onnx")
    )
    replace_scatter_elements(src, dst)
