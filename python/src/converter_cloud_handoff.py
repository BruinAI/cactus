from .tensor_io import (
    create_quantization_stats,
    print_quantization_summary,
    save_tensor_with_header,
)
from .weight_patterns import CLOUD_HANDOFF_GLOBAL_WEIGHTS, CLOUD_HANDOFF_STATS_WEIGHTS


def convert_hf_cloud_handoff_weights(
    state_dict,
    output_dir,
    precision="FP16",
    args=None,
    meta=None,
):
    """Convert Cloud Handoff classifier weights to Cactus binary format."""
    quantization_stats = create_quantization_stats()
    saved = set()

    for name, save_name in CLOUD_HANDOFF_GLOBAL_WEIGHTS:
        if name in state_dict:
            tensor = state_dict[name]
            save_tensor_with_header(
                tensor,
                output_dir / save_name,
                precision,
                transpose=False,
                stats_tracker=quantization_stats,
                args=args,
                model_type="cloud_handoff",
            )
            saved.add(name)

    for name, save_name in CLOUD_HANDOFF_STATS_WEIGHTS:
        if name in state_dict:
            tensor = state_dict[name]
            save_tensor_with_header(
                tensor,
                output_dir / save_name,
                "FP32",
                transpose=False,
                stats_tracker=quantization_stats,
                args=args,
                model_type="cloud_handoff",
            )
            saved.add(name)

    if "classifier.fc1.weight" not in state_dict or "classifier.fc2.weight" not in state_dict:
        raise ValueError(
            "Missing required Cloud Handoff weights: classifier.fc1.weight and/or classifier.fc2.weight"
        )

    model_config = {
        "model_type": "cloud_handoff",
        "model_variant": "cloud_handoff",
        "num_layers": 0,
        "tie_word_embeddings": False,
        "cloud_handoff_enabled": True,
        "cloud_handoff_has_feature_stats": (
            "feature_mean" in state_dict and "feature_std" in state_dict
        ),
    }

    fc1 = state_dict["classifier.fc1.weight"]
    if hasattr(fc1, "shape") and len(fc1.shape) == 2:
        model_config["cloud_handoff_input_dim"] = int(fc1.shape[1])
        model_config["cloud_handoff_hidden_dim"] = int(fc1.shape[0])
        model_config["hidden_dim"] = int(fc1.shape[1])

    fc2 = state_dict["classifier.fc2.weight"]
    if hasattr(fc2, "shape") and len(fc2.shape) == 2:
        model_config["cloud_handoff_output_dim"] = int(fc2.shape[0])

    if isinstance(meta, dict):
        for key in ("threshold", "dropout", "activation", "input_dim", "hidden_dim"):
            if key in meta:
                model_config[f"cloud_handoff_{key}"] = meta[key]

    print_quantization_summary(quantization_stats, args)
    return model_config
