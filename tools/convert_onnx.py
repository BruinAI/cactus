#!/usr/bin/env python3
import argparse
import json
import itertools
import os
import struct
from collections import deque

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto


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
        "--out-bin",
        default="graph.bin",
        help="Path to output binary IR file (default: graph.bin)",
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


def value_info_to_shape(value_info):
    """Return the list of static dims for a ValueInfoProto, or None if unknown."""
    tensor_type = getattr(value_info.type, "tensor_type", None)
    if tensor_type is None:
        return None
    shape_proto = getattr(tensor_type, "shape", None)
    if shape_proto is None or len(shape_proto.dim) == 0:
        return None

    dims = []
    for dim in shape_proto.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            dims.append(-1)
    return dims


PRECISION_NAME_TO_CODE = {"INT8": 0, "FP16": 1, "FP32": 2}

NODE_TYPE_ORDER = [
    "Input",
    "Weight",
    "Add",
    "BatchNormalization",
    "Concat",
    "Conv",
    "ConvTranspose",
    "Cos",
    "Div",
    "Flatten",
    "Gather",
    "Gemm",
    "GlobalAveragePool",
    "MatMul",
    "Max",
    "MaxPool",
    "Min",
    "Mul",
    "Reshape",
    "Resize",
    "Sigmoid",
    "Sin",
    "Slice",
    "Softmax",
    "Split",
    "Sub",
    "Transpose",
    "Unsqueeze",
]

NODE_TYPE_TO_ID = {name: idx for idx, name in enumerate(NODE_TYPE_ORDER)}

ATTR_TYPE_FLOAT = 0
ATTR_TYPE_INT64 = 1
ATTR_TYPE_BOOL = 2
ATTR_TYPE_STRING = 3
ATTR_TYPE_FLOAT_ARRAY = 4
ATTR_TYPE_INT64_ARRAY = 5


def onnx_dtype_to_precision_name(elem_type):
    if elem_type == TensorProto.INT8:
        return "INT8"
    if elem_type == TensorProto.FLOAT16:
        return "FP16"
    return "FP32"


def _normalize_attr_value(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _normalize_attr_value(value.item())
        return [_normalize_attr_value(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        if isinstance(value, (np.floating, float)):
            return float(value)
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_normalize_attr_value(x) for x in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    raise TypeError(f"Unsupported attribute type: {type(value)}")


def normalize_attributes(attrs):
    return {k: _normalize_attr_value(v) for k, v in attrs.items()}


def _write_string(stream, text):
    encoded = text.encode("utf-8")
    stream.write(struct.pack("<I", len(encoded)))
    stream.write(encoded)


def _write_attr_value(stream, key, value):
    if key == "input_precision":
        if isinstance(value, str):
            code = PRECISION_NAME_TO_CODE.get(value.upper(), 2)
        else:
            code = int(value)
        stream.write(struct.pack("<B", ATTR_TYPE_INT64))
        stream.write(struct.pack("<q", code))
        return

    if isinstance(value, bool):
        stream.write(struct.pack("<B", ATTR_TYPE_BOOL))
        stream.write(struct.pack("<B", 1 if value else 0))
        return

    if isinstance(value, float):
        stream.write(struct.pack("<B", ATTR_TYPE_FLOAT))
        stream.write(struct.pack("<f", value))
        return

    if isinstance(value, int):
        stream.write(struct.pack("<B", ATTR_TYPE_INT64))
        stream.write(struct.pack("<q", value))
        return

    if isinstance(value, str):
        stream.write(struct.pack("<B", ATTR_TYPE_STRING))
        _write_string(stream, value)
        return

    if isinstance(value, (list, tuple)):
        if not value:
            stream.write(struct.pack("<B", ATTR_TYPE_INT64_ARRAY))
            stream.write(struct.pack("<I", 0))
            return

        contains_float = any(isinstance(x, float) for x in value)
        if contains_float:
            stream.write(struct.pack("<B", ATTR_TYPE_FLOAT_ARRAY))
            stream.write(struct.pack("<I", len(value)))
            for item in value:
                stream.write(struct.pack("<f", float(item)))
            return

        stream.write(struct.pack("<B", ATTR_TYPE_INT64_ARRAY))
        stream.write(struct.pack("<I", len(value)))
        for item in value:
            stream.write(struct.pack("<q", int(item)))
        return

    raise RuntimeError(f"Unsupported attribute value for '{key}': {value!r}")


def write_binary_ir(nodes, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    value_ids = [out for node in nodes for out in node["outputs"]]
    max_value_id = max(value_ids) if value_ids else -1
    value_count = max_value_id + 1 if max_value_id >= 0 else 0

    with open(output_path, "wb") as stream:
        stream.write(b"CAIR")
        stream.write(struct.pack("<B", 1))  # version
        stream.write(struct.pack("<I", len(nodes)))
        stream.write(struct.pack("<I", value_count))

        for node in nodes:
            node_type = node["type"]
            if node_type not in NODE_TYPE_TO_ID:
                raise RuntimeError(f"Unsupported node type '{node_type}' for binary IR")

            stream.write(struct.pack("<I", node["id"]))
            stream.write(struct.pack("<B", NODE_TYPE_TO_ID[node_type]))

            stream.write(struct.pack("<I", len(node["inputs"])))
            for input_value in node["inputs"]:
                stream.write(struct.pack("<I", input_value))

            stream.write(struct.pack("<I", len(node["outputs"])))
            for output_value in node["outputs"]:
                stream.write(struct.pack("<I", output_value))

            attrs = node["attributes"]
            stream.write(struct.pack("<I", len(attrs)))
            for key, value in attrs.items():
                _write_string(stream, key)
                _write_attr_value(stream, key, value)
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
        shape = value_info_to_shape(inp)
        if shape is not None:
            values[vid]["shape"] = shape

    # Initializers as weights: save to .weights and mark in value table
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        vid = get_value_id(init.name)
        values[vid]["kind"] = "Weight"

        safe_name = sanitize_filename(init.name)
        weight_path = os.path.join(args.weights_dir, f"{safe_name}.weights")
        save_tensor_with_header(arr, weight_path, precision=args.precision)
        values[vid]["weight_file"] = weight_path

    value_shape_map = {}
    for value_info in itertools.chain(graph.input, graph.value_info, graph.output):
        shape = value_info_to_shape(value_info)
        if shape is not None:
            value_shape_map[value_info.name] = shape

    graph_output_value_ids = set()
    for output in graph.output:
        if not output.name:
            continue
        graph_output_value_ids.add(get_value_id(output.name))

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

        def _normalize_axes(axes_list, tensor_name):
            if axes_list is None:
                return None
            shape = value_shape_map.get(tensor_name)
            rank = len(shape) if shape is not None else None
            normalized = []
            for axis in axes_list:
                adjusted = axis
                if rank is not None and adjusted < 0:
                    adjusted += rank
                normalized.append(adjusted)
            if rank is not None:
                for axis in normalized:
                    if axis < 0 or axis >= rank:
                        raise RuntimeError(
                            f"Slice node {node.name} has axis {axis} outside input rank {rank}"
                        )
            elif any(axis < 0 for axis in normalized):
                print(
                    f"WARNING: Slice node {node.name} uses negative axis values {axes_list}; "
                    "input rank unknown so they remain unadjusted."
                )
            return normalized

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
            axes = _normalize_axes(axes, node.input[0] if node.input else None)

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
            if steps is None:
                ref_length = len(axes) if axes else (len(starts) if starts else 1)
                steps = [1] * max(1, ref_length)
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

    # ---- BUILD NODE LIST ----
    input_nodes = []
    for inp in graph.input:
        if inp.name in initializer_names:
            continue
        vid = get_value_id(inp.name)
        attrs = {}
        shape = value_info_to_shape(inp)
        if shape is not None:
            attrs["input_shape"] = shape

        tensor_type = getattr(inp.type, "tensor_type", None)
        elem_type = getattr(tensor_type, "elem_type", None) if tensor_type else None
        attrs["input_precision"] = onnx_dtype_to_precision_name(elem_type)
        input_nodes.append(
            {
                "type": "Input",
                "inputs": [],
                "outputs": [vid],
                "attributes": attrs,
            }
        )

    weight_nodes = []
    for init in graph.initializer:
        vid = get_value_id(init.name)
        weight_path = values[vid].get("weight_file")
        if not weight_path:
            continue
        weight_nodes.append(
            {
                "type": "Weight",
                "inputs": [],
                "outputs": [vid],
                "attributes": {
                    "path_to_weights": os.path.abspath(weight_path)
                },
            }
        )

    final_nodes = []
    node_id = 0

    def append_node(node_type, inputs, outputs, attributes):
        nonlocal node_id
        attrs = dict(attributes)
        if any(value_id in graph_output_value_ids for value_id in outputs):
            attrs["is_graph_output"] = True
        final_nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "inputs": inputs,
                "outputs": outputs,
                "attributes": normalize_attributes(attrs),
            }
        )
        node_id += 1

    for node in input_nodes:
        append_node(node["type"], node["inputs"], node["outputs"], node["attributes"])

    for node in weight_nodes:
        append_node(node["type"], node["inputs"], node["outputs"], node["attributes"])

    for op_index in topo_ops:
        op = ops[op_index]
        append_node(op["type"], op["inputs"], op["outputs"], op["attributes"])

    out_obj = {
        "model_path": os.path.abspath(args.model),
        "value_count": len(values),
        "nodes": final_nodes,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, sort_keys=True)

    write_binary_ir(final_nodes, args.out_bin)

    print(f"Wrote graph JSON to {args.out_json}")
    print(f"Wrote binary IR to {args.out_bin}")
    print(f"Wrote weights to directory {args.weights_dir}")


if __name__ == "__main__":
    main()
