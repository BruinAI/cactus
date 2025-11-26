#!/usr/bin/env python3
import argparse
import json
import os
import struct
from collections import deque

import numpy as np
import onnx
from onnx import numpy_helper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ONNX graph into topo-sorted ops + value table with custom .weights."
    )
    parser.add_argument("model", help="Path to .onnx model")
    parser.add_argument(
        "--out-json",
        default="graph.json",
        help="Path to output JSON file (default: graph.json)",
    )
    parser.add_argument(
        "--weights-dir",
        default="weights",
        help="Directory to store weight .weights files (default: weights)",
    )
    parser.add_argument(
        "--precision",
        default="FP32",
        choices=["FP32", "FP16", "INT8"],
        help="Precision for stored weights (default: FP32)",
    )
    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    """Make a filesystem-safe filename from a tensor name."""
    bad_chars = '<>:"/\\|?* '
    out = []
    for ch in name:
        if ch in bad_chars:
            out.append("_")
        else:
            out.append(ch)
    return "".join(out)


def save_tensor_with_header(
    tensor,
    output_path,
    precision="FP32",
):
    """
    Save numpy tensor to custom .weights binary with header:

    1. ndim (uint32)
    2. shape dims (uint64 each)
    3. precision (uint32): 0=INT8, 1=FP16, 2=FP32
    4. byte_size (uint64): total bytes of raw data
    5. scale (float32) if precision==INT8
    6. raw data bytes
    """
    if isinstance(tensor, np.ndarray):
        data = tensor
    else:
        data = np.array(tensor)

    original = data.astype(np.float32)
    precision = precision.upper()
    scale = 1.0

    if precision == "INT8":
        qmin, qmax = -128, 127
        abs_max = float(np.max(np.abs(original))) if original.size > 0 else 0.0
        scale = abs_max / 127.0 if abs_max != 0.0 else 1.0
        quantized = np.clip(
            np.round(original / scale), qmin, qmax
        ).astype(np.int8)
        data_to_write = quantized
        prec_val = 0
        element_size = 1
    elif precision == "FP16":
        data_to_write = original.astype(np.float16)
        prec_val = 1
        element_size = 2
    else:  # FP32
        data_to_write = original.astype(np.float32)
        prec_val = 2
        element_size = 4

    shape = list(data_to_write.shape)
    flat = data_to_write.flatten()
    byte_size = flat.size * element_size

    output_path = os.fspath(output_path)
    print(f"Saving {os.path.basename(output_path)}: {precision} {shape}")

    with open(output_path, "wb") as f:
        # 1. ndim
        ndim = len(shape)
        f.write(struct.pack("<I", ndim))
        # 2. dims
        for dim in shape:
            f.write(struct.pack("<Q", int(dim)))
        # 3. precision code
        f.write(struct.pack("<I", prec_val))
        # 4. byte_size
        f.write(struct.pack("<Q", byte_size))
        # 5. scale (INT8 only)
        if prec_val == 0:
            f.write(struct.pack("<f", float(scale)))
        # 6. raw data
        f.write(flat.tobytes())

    if precision == "INT8":
        scale_path = os.path.splitext(output_path)[0] + ".scale"
        with open(scale_path, "w") as f:
            f.write(f"{scale:.10f}\n")


def attr_to_python(attr):
    """Convert an onnx.AttributeProto to a plain Python value."""
    from onnx import AttributeProto

    if attr.type == AttributeProto.FLOAT:
        return float(attr.f)
    elif attr.type == AttributeProto.INT:
        return int(attr.i)
    elif attr.type == AttributeProto.STRING:
        return attr.s.decode("utf-8", errors="replace")
    elif attr.type == AttributeProto.FLOATS:
        return [float(x) for x in attr.floats]
    elif attr.type == AttributeProto.INTS:
        return [int(x) for x in attr.ints]
    elif attr.type == AttributeProto.TENSOR:
        arr = numpy_helper.to_array(attr.t)
        return {
            "tensor_dtype": str(arr.dtype),
            "tensor_shape": list(arr.shape),
        }
    else:
        return f"<unsupported_attr_type_{attr.type}>"


def main():
    args = parse_args()

    model = onnx.load(args.model)
    graph = model.graph

    os.makedirs(args.weights_dir, exist_ok=True)

    # ---- VALUE TABLE ----
    values = []  # list of dicts
    value_id_by_name = {}  # tensor_name -> value_id

    def get_value_id(name: str):
        """Get or create a value id for a given tensor name."""
        if name not in value_id_by_name:
            vid = len(values)
            value_id_by_name[name] = vid
            values.append(
                {
                    "id": vid,
                    "name": name,
                    "kind": "Intermediate",  # default; may be overwritten
                    # optional extras: producer, weight_file, etc.
                }
            )
        return value_id_by_name[name]

    # Initializer name map
    initializer_map = {init.name: init for init in graph.initializer}
    initializer_names = set(initializer_map.keys())

    # Mark graph inputs as values of kind "Input"
    for inp in graph.input:
        if inp.name in initializer_names:
            continue
        vid = get_value_id(inp.name)
        values[vid]["kind"] = "Input"

    # Initializers as weights: save to .weights and mark in value table
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        vid = get_value_id(init.name)
        values[vid]["kind"] = "Weight"

        safe_name = sanitize_filename(init.name)
        weight_path = os.path.join(args.weights_dir, f"{safe_name}.weights")
        save_tensor_with_header(arr, weight_path, precision=args.precision)
        values[vid]["weight_file"] = weight_path

    # ---- OP NODES ----
    ops = []  # each op: {id, type, inputs:[value_ids], outputs:[value_ids], attributes }
    value_producer = {}  # value_id -> {op_id, output_index}

    for node in graph.node:
        op_id = len(ops)

        input_value_ids = [get_value_id(n) for n in node.input]
        output_value_ids = [get_value_id(n) for n in node.output]

        attrs = {a.name: attr_to_python(a) for a in node.attribute}

        # --- Reshape: bake static shape & warn on dynamic ---
        if node.op_type == "Reshape" and len(node.input) >= 2:
            data_name = node.input[0]
            shape_name = node.input[1]

            if shape_name in initializer_map:
                shape_tensor = numpy_helper.to_array(
                    initializer_map[shape_name]
                ).astype(np.int64)
                target = shape_tensor.tolist()

                if any(d == 0 for d in target):
                    print(
                        f"WARNING: Reshape node {node.name} has a 0 in target shape "
                        f"{target} from tensor {shape_name}"
                    )

                # Store the raw shape list as an attribute; you handle -1/0 in C++
                attrs["shape"] = [int(x) for x in target]
            else:
                print(
                    f"WARNING: Reshape node {node.name} uses non-initializer shape tensor "
                    f"{shape_name}; dynamic reshape not baked into attributes."
                )

        # --- Unsqueeze: bake static axes & warn on dynamic ---
        if node.op_type == "Unsqueeze" and len(node.input) >= 2:
            axes_name = node.input[1]
            if axes_name in initializer_map:
                axes_tensor = numpy_helper.to_array(
                    initializer_map[axes_name]
                ).astype(np.int64)
                axes_list = axes_tensor.flatten().tolist()
                attrs["axes"] = [int(x) for x in axes_list]
            else:
                print(
                    f"WARNING: Unsqueeze node {node.name} uses non-initializer axes tensor "
                    f"{axes_name}; dynamic axes not baked into attributes."
                )

        # --- Slice: bake static starts/ends/axes/steps ---
        if node.op_type == "Slice":
            def _load_slice_input(index: int, attr_name: str, desc: str):
                if index >= len(node.input):
                    return None
                input_name = node.input[index]
                if not input_name:
                    return None
                if input_name in initializer_map:
                    tensor = numpy_helper.to_array(initializer_map[input_name])
                    flat = tensor.flatten().tolist()
                    values = [int(x) for x in flat]
                    attrs[attr_name] = values
                    return values
                print(
                    f"WARNING: Slice node {node.name} uses non-initializer {desc} tensor "
                    f"{input_name}; dynamic {desc} not baked into attributes."
                )
                return None

            starts = _load_slice_input(1, "starts", "starts")
            ends = _load_slice_input(2, "ends", "ends")
            axes = _load_slice_input(3, "axes", "axes")
            steps = _load_slice_input(4, "steps", "steps")

            if axes is not None and len(axes) > 1:
                print(
                    f"WARNING: Slice node {node.name} specifies multiple axes {axes}; "
                    "only single-axis slicing is baked."
                )
            if steps is not None and any(step != 1 for step in steps):
                print(
                    f"WARNING: Slice node {node.name} specifies steps {steps}; only step=1 "
                    "is supported in static slicing."
                )
            if starts is not None and any(s <= 0 for s in starts):
                print(
                    f"WARNING: Slice node {node.name} has non-positive start values {starts}; "
                    "only positive starts are supported."
                )
            if ends is not None and any(e <= 0 for e in ends):
                print(
                    f"WARNING: Slice node {node.name} has non-positive end values {ends}; "
                    "only positive ends are supported."
                )

            attrs["slice_starts"] = starts
            attrs["slice_ends"] = ends
            attrs["axes"] = axes
            attrs["slice_steps"] = steps

        # --- Resize: bake static roi/scales/sizes tensors ---
        if node.op_type == "Resize":
            roi_present = False
            if len(node.input) > 1:
                roi_name = node.input[1]
                if roi_name:
                    roi_present = True
                    print(
                        f"WARNING: Resize node {node.name} supplies ROI tensor "
                        f"{roi_name}; ROI must be empty and will be ignored."
                    )
            attrs["roi"] = roi_present

            if len(node.input) > 2:
                size_name = node.input[2]
                if size_name and size_name in initializer_map:
                    tensor = numpy_helper.to_array(initializer_map[size_name])
                    flat = tensor.flatten()
                    if flat.dtype.kind in {"i", "u"}:
                        attrs["sizes"] = [int(x) for x in flat.tolist()]
                    else:
                        attrs["scales"] = [float(x) for x in flat.tolist()]
                else:
                    print(
                        f"WARNING: Resize node {node.name} uses non-initializer "
                        f"input scales/sizes tensor {size_name}; dynamic values not baked."
                    )
        # --- Split: bake static splits when provided as second input ---
        if node.op_type == "Split" and len(node.input) > 1:
            split_sizes_name = node.input[1]
            if split_sizes_name and split_sizes_name in initializer_map:
                split_tensor = numpy_helper.to_array(initializer_map[split_sizes_name])
                split_values = split_tensor.flatten().tolist()
                attrs["splits"] = [int(x) for x in split_values]
            else:
                print(
                    f"WARNING: Split node {node.name} uses non-initializer split sizes tensor "
                    f"{split_sizes_name}; dynamic splits not baked into attributes."
                )

        # Mark outputs as produced by this op
        for idx, v_id in enumerate(output_value_ids):
            value_producer[v_id] = {"op_id": op_id, "output_index": idx}
            if values[v_id]["kind"] == "Intermediate":
                values[v_id]["producer"] = op_id
                values[v_id]["producer_index"] = idx

        ops.append(
            {
                "id": op_id,
                "type": node.op_type,
                "inputs": input_value_ids,
                "outputs": output_value_ids,
                "attributes": attrs,
            }
        )

    # ---- TOPO SORT OPS (using value producers) ----
    num_ops = len(ops)
    adjacency = [[] for _ in range(num_ops)]
    indegree = [0] * num_ops

    for b_id, op in enumerate(ops):
        for v_id in op["inputs"]:
            prod = value_producer.get(v_id)
            if prod is None:
                continue  # produced by Input or Weight or external
            a_id = prod["op_id"]
            adjacency[a_id].append(b_id)
            indegree[b_id] += 1

    q = deque([i for i in range(num_ops) if indegree[i] == 0])
    topo_ops = []
    while q:
        u = q.popleft()
        topo_ops.append(u)
        for v in adjacency[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)

    if len(topo_ops) != num_ops:
        print(
            f"Warning: op graph may contain cycles or disconnected pieces "
            f"(sorted {len(topo_ops)} of {num_ops} ops)"
        )

    # ---- BUILD FINAL JSON ----
    json_ops = []
    for op_id in topo_ops:
        op = ops[op_id]
        json_ops.append(
            {
                "id": op["id"],
                "type": op["type"],
                "inputs": op["inputs"],   # value IDs
                "outputs": op["outputs"], # value IDs
                "attributes": op["attributes"],
            }
        )

    out_obj = {
        "model_path": os.path.abspath(args.model),
        "values": values,  # each has id, name, kind, maybe weight_file, producer, ...
        "nodes": json_ops, # topo-sorted ops
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, sort_keys=True)

    print(f"Wrote graph JSON to {args.out_json}")
    print(f"Wrote weights to directory {args.weights_dir}")


if __name__ == "__main__":
    main()
