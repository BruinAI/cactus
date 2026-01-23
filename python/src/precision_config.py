"""Mixed precision configuration for Cactus weight conversion.

This module determines which tensors should use INT8 vs INT4 quantization
based on their sensitivity to quantization error.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

try:
    import torch
except ImportError:
    torch = None


GROUP_SIZE = 128
SMALL_MODEL_THRESHOLD = 300_000_000
LARGE_MODEL_THRESHOLD = 1_000_000_000 


def count_model_parameters(model) -> int:
    """Count trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    if torch is None:
        return 0

    if hasattr(model, 'parameters'):
        return sum(p.numel() for p in model.parameters())

    if hasattr(model, 'state_dict'):
        state_dict = model.state_dict()
    elif isinstance(model, dict):
        state_dict = model
    else:
        return 0

    total_params = 0
    for name, tensor in state_dict.items():
        if hasattr(tensor, 'numel'):
            total_params += tensor.numel()

    return total_params


SENSITIVE_PATTERNS = [
    'embed', 
    'lm_head', 
    'output_weight',  
    'attn_q', 
    'attn_k',   
    'layer_0_',  
    'layer_1_', 
]

TOLERANT_PATTERNS = [
    'attn_v',         
    'ffn_gate',      
    'ffn_up',   
    'ffn_down', 
    'attn_output', 
]

FP16_PATTERNS = [
    'norm',
    'bias',
    'vision',
]


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision quantization."""
    snr_threshold: float = 25.0
    sensitive_as_int8: bool = True
    tolerant_as_int4: bool = True
    last_n_layers_int8: int = 2
    force_int8: bool = False
    force_int4: bool = False 


def is_sensitive_tensor(output_name: str, layer_idx: Optional[int], num_layers: Optional[int],
                        config: MixedPrecisionConfig = None) -> bool:
    """Determine if a tensor should use INT8 (more sensitive) or INT4 (more tolerant).

    Args:
        output_name: The output filename for the tensor
        layer_idx: The layer index (None for non-layer weights like embeddings)
        num_layers: Total number of layers in the model
        config: Mixed precision configuration

    Returns:
        True if tensor should use INT8, False for INT4
    """
    if config is None:
        config = MixedPrecisionConfig()

    name_lower = output_name.lower()

    for pattern in FP16_PATTERNS:
        if pattern in name_lower:
            return True 

    if config.sensitive_as_int8:
        for pattern in SENSITIVE_PATTERNS:
            if pattern in name_lower:
                return True

    if layer_idx is not None and num_layers is not None:
        if layer_idx >= num_layers - config.last_n_layers_int8:
            return True

    if config.tolerant_as_int4:
        for pattern in TOLERANT_PATTERNS:
            if pattern in name_lower:
                return False

    return True


def determine_precision(output_name: str, base_precision: str,
                        layer_idx: Optional[int] = None,
                        num_layers: Optional[int] = None,
                        config: MixedPrecisionConfig = None) -> str:
    """Determine the actual precision to use for a tensor.

    Args:
        output_name: The output filename for the tensor
        base_precision: The base precision setting ('MIXED', 'INT4', 'INT8', 'FP16')
        layer_idx: The layer index (None for non-layer weights)
        num_layers: Total number of layers in the model
        config: Mixed precision configuration

    Returns:
        The precision to use: 'INT8', 'INT4', or 'FP16'
    """
    if base_precision != 'MIXED':
        return base_precision

    name_lower = output_name.lower()
    for pattern in FP16_PATTERNS:
        if pattern in name_lower:
            return 'FP16'

    if config is not None and config.force_int8:
        return 'INT8'

    if config is not None and config.force_int4:
        return 'INT4'

    if is_sensitive_tensor(output_name, layer_idx, num_layers, config):
        return 'INT8'
    else:
        return 'INT4'


def extract_layer_index(output_name: str) -> Optional[int]:
    """Extract the layer index from an output filename.

    Args:
        output_name: The output filename (e.g., 'layer_5_attn_q.weights')

    Returns:
        The layer index, or None if not a layer weight
    """
    import re
    match = re.search(r'layer_(\d+)_', output_name)
    if match:
        return int(match.group(1))
    return None


def trial_quantize_int4(data: np.ndarray) -> tuple:
    """Trial INT4 quantization to compute quality metrics.

    Args:
        data: The tensor data to quantize

    Returns:
        Tuple of (mse, snr_db, cos_sim)
    """
    if data.ndim != 2:
        return 0.0, float('inf'), 1.0

    N, K = data.shape

    if K % GROUP_SIZE != 0:
        pad_k = GROUP_SIZE - (K % GROUP_SIZE)
        data = np.pad(data, ((0, 0), (0, pad_k)), mode='constant', constant_values=0)
        K = data.shape[1]

    num_groups = K // GROUP_SIZE
    data_grouped = data.reshape(N, num_groups, GROUP_SIZE)

    group_abs_max = np.max(np.abs(data_grouped), axis=2)
    scales = (group_abs_max / 7.0).astype(np.float32)
    scales = np.maximum(scales, 1e-10)

    quantized = np.clip(
        np.round(data_grouped / scales[:, :, np.newaxis]),
        -8, 7
    ).astype(np.int8)

    dequantized = (quantized.astype(np.float32) * scales[:, :, np.newaxis]).reshape(N, K)

    original = data[:, :K]  # Match shape after padding
    mse = np.mean((original - dequantized) ** 2)
    variance = np.var(original)
    snr_db = 10 * np.log10(variance / mse) if mse > 0 else float('inf')

    original_flat = original.flatten()
    dequant_flat = dequantized.flatten()
    cos_sim = np.dot(original_flat, dequant_flat) / (
        np.linalg.norm(original_flat) * np.linalg.norm(dequant_flat) + 1e-10
    )

    return mse, snr_db, cos_sim


def adaptive_precision_from_snr(data: np.ndarray, output_name: str,
                                 snr_threshold: float = 25.0) -> str:
    """Determine precision based on trial INT4 quantization SNR.

    If INT4 quantization yields SNR below threshold, use INT8 instead.

    Args:
        data: The tensor data
        output_name: The output filename
        snr_threshold: Minimum acceptable SNR in dB for INT4

    Returns:
        'INT4' if quality is acceptable, 'INT8' otherwise
    """
    if data.ndim != 2:
        return 'INT8'  # Non-2D tensors use INT8 or FP16

    mse, snr_db, cos_sim = trial_quantize_int4(data)

    if snr_db >= snr_threshold:
        return 'INT4'
    else:
        return 'INT8'
