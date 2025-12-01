import argparse
import io
import math
from typing import List, Dict, Tuple

import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort
from PIL import Image


NUM_STAT_VALUES = 65536
NUM_PREVIEW_VALUES = 8


def add_all_node_outputs_as_graph_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Modify the model in-place so that every node's output is added as a graph output.
    """
    graph = model.graph
    existing_outputs = {o.name for o in graph.output}
    for node in graph.node:
        for out in node.output:
            if out and out not in existing_outputs:
                # We don't know the full type here, but it's fine to just
                # create a ValueInfoProto with a name only; ONNX Runtime
                # will still produce it.
                vi = helper.make_empty_tensor_value_info(out)
                graph.output.append(vi)
                existing_outputs.add(out)
    return model


def get_input_info(session: ort.InferenceSession):
    """Return (name, shape, dtype) for the first model input."""
    inp = session.get_inputs()[0]
    return inp.name, list(inp.shape), inp.type


def preprocess_image(image_path: str, input_shape: List[int]) -> np.ndarray:
    """
    Basic NCHW float32 preprocessing:
      - Load image
      - Convert to RGB
      - Resize to model's HxW if specified
      - Normalize to [0,1]
      - Transpose to NCHW
    """
    img = Image.open(image_path).convert("RGB")

    # input_shape is typically [N, C, H, W]
    if len(input_shape) != 4:
        raise ValueError(f"Expected 4D input, got shape: {input_shape}")

    _, C, H, W = input_shape

    if C != 3 and C != "None" and C is not None:
        raise ValueError(f"Script assumes 3-channel RGB input, but model C={C}")

    # If H/W are dynamic (None or 'None'), use image's size
    if (isinstance(H, str) or H is None) or (isinstance(W, str) or W is None):
        H, W = img.height, img.width
    else:
        # Resize to model's expected size
        img = img.resize((W, H), Image.BILINEAR)

    img_np = np.array(img).astype(np.float32) / 255.0  # [H, W, C]
    img_np = np.transpose(img_np, (2, 0, 1))           # [C, H, W]
    img_np = np.expand_dims(img_np, 0)                 # [1, C, H, W]

    return img_np


def format_precision(dtype: np.dtype) -> str:
    if dtype == np.float32:
        return "FP32"
    if dtype == np.float16:
        return "FP16"
    if dtype == np.int8:
        return "INT8"
    if dtype == np.int16:
        return "INT16"
    if dtype == np.int32:
        return "INT32"
    if dtype == np.int64:
        return "INT64"
    return str(dtype)


def compute_stats(arr: np.ndarray):
    flat = arr.ravel()
    if flat.size == 0:
        return None, None, None, None, []

    sample = flat[:min(flat.size, NUM_STAT_VALUES)]
    min_v = float(sample.min())
    max_v = float(sample.max())
    mean_v = float(sample.mean())
    std_v = float(sample.std())

    preview = sample[:NUM_PREVIEW_VALUES].tolist()
    return min_v, max_v, mean_v, std_v, preview


def main():
    model = 'yolo11n.onnx'
    image = 'test_monkey.png'

    # Load model and instrument it
    model = onnx.load(model)
    add_all_node_outputs_as_graph_outputs(model)

    # Create an in-memory ONNX Runtime session
    model_bytes = model.SerializeToString()
    sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])

    # Get input info and preprocess image
    input_name, input_shape, _ = get_input_info(sess)
    input_tensor = preprocess_image(image, input_shape)

    # Run inference, collecting all outputs
    output_names = [o.name for o in sess.get_outputs()]
    outputs = sess.run(output_names, {input_name: input_tensor})

    # Build mapping: tensor name -> (layer_index, node)
    # We'll treat each node in order as "Layer i"
    name_to_nodeinfo: Dict[str, Tuple[int, onnx.NodeProto]] = {}
    for idx, node in enumerate(model.graph.node):
        for out_name in node.output:
            if out_name:
                name_to_nodeinfo[out_name] = (idx, node)

    # Print stats for each output that corresponds to a node output
    name_to_output = {name: value for name, value in zip(output_names, outputs)}

    for tensor_name, value in name_to_output.items():
        if tensor_name not in name_to_nodeinfo:
            # This is probably a graph output that isn't directly a node output,
            # or comes from an initializer; skip for cleaner logs.
            continue

        layer_idx, node = name_to_nodeinfo[tensor_name]
        arr = np.asarray(value)

        min_v, max_v, mean_v, std_v, preview = compute_stats(arr)
        precision = format_precision(arr.dtype)
        shape_str = "[" + ",".join(str(d) for d in arr.shape) + "]"

        print("-" * 60)
        print(f"Layer {layer_idx} - {node.op_type.upper()}#{layer_idx} (node {node.name or layer_idx})")
        print(f"  Shape: {shape_str}  Precision: {precision}")

        if min_v is None:
            print("  Stats: tensor is empty")
        else:
            print(
                "  Stats: "
                f"min={min_v:.6f} max={max_v:.6f} "
                f"mean={mean_v:.6f} std={std_v:.6f}"
            )
            total_vals = arr.size
            if total_vals > NUM_STAT_VALUES:
                print(
                    f"  Note: stats computed on first {NUM_STAT_VALUES} "
                    f"of {total_vals} values"
                )
            else:
                print(f"  Note: stats computed on all {total_vals} values")

            preview_str = ", ".join(f"{v:.6f}" for v in preview)
            if total_vals > NUM_PREVIEW_VALUES:
                preview_str += ", ..."
            print(f"  Preview: [{preview_str}]")

    print("-" * 60)


if __name__ == "__main__":
    main()
