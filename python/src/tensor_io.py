import numpy as np
import struct
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import torch
except ImportError:
    torch = None

from .precision_config import determine_precision, MixedPrecisionConfig


GROUP_SIZE = 128

CACTUS_MAGIC = b'CACT'
CACTUS_VERSION = 1
CACTUS_ALIGNMENT = 32

FLAG_HAS_SCALES = 1 << 0
FLAG_PAGE_ALIGNED = 1 << 1
FLAG_TRANSPOSED = 1 << 2


def align_offset(offset: int, alignment: int) -> int:
    """Round up offset to next alignment boundary."""
    remainder = offset % alignment
    if remainder == 0:
        return offset
    return offset + (alignment - remainder)


def compute_padding(current_offset: int, alignment: int) -> bytes:
    """Compute padding bytes needed to reach alignment boundary."""
    aligned = align_offset(current_offset, alignment)
    padding_size = aligned - current_offset
    return b'\x00' * padding_size


def save_tensor_with_header(tensor, output_path, precision='FP16', transpose=False, stats_tracker=None, args=None, model_type=None, layer_idx=None, num_layers=None, mixed_config=None):
    """Save a tensor to binary format with header metadata and group-wise INT8/INT4 quantization.

    Args:
        tensor: The tensor to save (PyTorch or NumPy)
        output_path: Path to save the tensor
        precision: Quantization precision ('MIXED', 'INT4', 'INT8', 'FP16')
        transpose: Whether to transpose 2D tensors
        stats_tracker: Optional dict to track quantization statistics
        args: Optional args object with additional settings
        model_type: Model type string (e.g., 'gemma', 'llama')
        layer_idx: Layer index for mixed precision (None for non-layer weights)
        num_layers: Total number of layers for mixed precision calculation
        mixed_config: MixedPrecisionConfig for controlling quantization behavior
    """
    if torch is not None and isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy()
    else:
        data = np.array(tensor)

    original_data = data.copy()

    if model_type == 'gemma' and 'norm' in str(output_path):
        data = data + 1.0
        original_data = data.copy()

    # Determine actual precision for MIXED mode
    output_name = str(output_path.name) if hasattr(output_path, 'name') else str(output_path)
    if precision == 'MIXED':
        precision = determine_precision(output_name, 'MIXED', layer_idx, num_layers, mixed_config)

    if precision in ('INT8', 'INT4'):
        filename = output_path.name
        if any(x in filename for x in ['norm', 'bias', 'vision']):
            precision = 'FP16'

    shape = list(data.shape)
    if transpose and len(shape) == 2:
        data = data.T
        original_data = original_data.T
        shape = [shape[1], shape[0]]

    if precision == 'INT8':
        if len(shape) == 2:
            N, K = shape

            if K % GROUP_SIZE != 0:
                pad_k = GROUP_SIZE - (K % GROUP_SIZE)
                data = np.pad(data, ((0, 0), (0, pad_k)), mode='constant', constant_values=0)
                original_data = np.pad(original_data, ((0, 0), (0, pad_k)), mode='constant', constant_values=0)
                K = data.shape[1]
                shape = [N, K]

            num_groups = K // GROUP_SIZE

            data_grouped = data.reshape(N, num_groups, GROUP_SIZE)
            original_grouped = original_data.reshape(N, num_groups, GROUP_SIZE)

            group_abs_max = np.max(np.abs(data_grouped), axis=2) 
            scales = (group_abs_max / 127.0).astype(np.float32)
            scales = np.maximum(scales, 1e-10)  

            quantized = np.clip(
                np.round(data_grouped / scales[:, :, np.newaxis]),
                -128, 127
            ).astype(np.int8)
            quantized_flat = quantized.reshape(N, K)

            dequantized = (quantized.astype(np.float32) * scales[:, :, np.newaxis]).reshape(N, K)
            mse_error = np.mean((original_data - dequantized) ** 2)
            snr_db = 10 * np.log10(np.var(original_data) / mse_error) if mse_error > 0 else float('inf')

            original_flat = original_data.flatten()
            dequant_flat = dequantized.flatten()
            cos_sim = np.dot(original_flat, dequant_flat) / (np.linalg.norm(original_flat) * np.linalg.norm(dequant_flat) + 1e-10)

            scales_fp16 = scales.astype(np.float16)

        elif len(shape) == 1:
            K = shape[0]

            if K % GROUP_SIZE != 0:
                pad_k = GROUP_SIZE - (K % GROUP_SIZE)
                data = np.pad(data, (0, pad_k), mode='constant', constant_values=0)
                original_data = np.pad(original_data, (0, pad_k), mode='constant', constant_values=0)
                K = data.shape[0]
                shape = [K]

            num_groups = K // GROUP_SIZE
            N = 1

            data_grouped = data.reshape(1, num_groups, GROUP_SIZE)
            original_grouped = original_data.reshape(1, num_groups, GROUP_SIZE)

            group_abs_max = np.max(np.abs(data_grouped), axis=2)
            scales = (group_abs_max / 127.0).astype(np.float32)
            scales = np.maximum(scales, 1e-10)

            quantized = np.clip(
                np.round(data_grouped / scales[:, :, np.newaxis]),
                -128, 127
            ).astype(np.int8)
            quantized_flat = quantized.reshape(K)

            dequantized = (quantized.astype(np.float32) * scales[:, :, np.newaxis]).reshape(K)
            mse_error = np.mean((original_data - dequantized) ** 2)
            snr_db = 10 * np.log10(np.var(original_data) / mse_error) if mse_error > 0 else float('inf')
            cos_sim = np.dot(original_data, dequantized) / (np.linalg.norm(original_data) * np.linalg.norm(dequantized) + 1e-10)

            scales_fp16 = scales.astype(np.float16)
        else:
            precision = 'FP16'

    if precision == 'INT8':
        if stats_tracker:
            stats_tracker['int8_tensors'] += 1
            stats_tracker['quantized_parameters'] += original_data.size
            stats_tracker['mse_values'].append(mse_error)
            stats_tracker['snr_values'].append(snr_db)
            stats_tracker['cos_sim_values'].append(cos_sim)
            stats_tracker['total_tensors'] += 1
            stats_tracker['total_parameters'] += original_data.size

        with open(output_path, 'wb') as f:
            ndim = len(shape)
            data_bytes = quantized_flat.size
            scales_bytes = scales_fp16.size * 2
            flags = FLAG_HAS_SCALES
            if transpose:
                flags |= FLAG_TRANSPOSED

            # Fixed 64-byte header
            f.write(CACTUS_MAGIC)                          # 4 bytes
            f.write(struct.pack('<I', CACTUS_VERSION))     # 4 bytes
            f.write(struct.pack('<I', flags))              # 4 bytes
            f.write(struct.pack('<I', CACTUS_ALIGNMENT))   # 4 bytes
            f.write(struct.pack('<I', ndim))               # 4 bytes

            for i in range(4):
                if i < ndim:
                    f.write(struct.pack('<Q', shape[i]))
                else:
                    f.write(struct.pack('<Q', 0))          # 32 bytes total
            f.write(struct.pack('<I', 0))                  # precision: 0 = INT8 (4 bytes)
            f.write(struct.pack('<Q', data_bytes))         # 8 bytes
            f.write(struct.pack('<Q', scales_bytes))       # 8 bytes
            f.write(struct.pack('<I', GROUP_SIZE))         # 4 bytes
            f.write(struct.pack('<I', num_groups))         # 4 bytes (changed from Q to I)
            header_size = 80
            f.write(compute_padding(header_size, CACTUS_ALIGNMENT))

            f.write(scales_fp16.tobytes())
            scales_end = align_offset(header_size, CACTUS_ALIGNMENT) + scales_bytes
            f.write(compute_padding(scales_end, CACTUS_ALIGNMENT))

            f.write(quantized_flat.tobytes())

        return

    if precision == 'INT4':
        if len(shape) == 2:
            N, K = shape

            if K % GROUP_SIZE != 0:
                pad_k = GROUP_SIZE - (K % GROUP_SIZE)
                data = np.pad(data, ((0, 0), (0, pad_k)), mode='constant', constant_values=0)
                original_data = np.pad(original_data, ((0, 0), (0, pad_k)), mode='constant', constant_values=0)
                K = data.shape[1]
                shape = [N, K]

            num_groups = K // GROUP_SIZE
            data_grouped = data.reshape(N, num_groups, GROUP_SIZE)

            group_abs_max = np.max(np.abs(data_grouped), axis=2)
            scales = (group_abs_max / 7.0).astype(np.float32)
            scales = np.maximum(scales, 1e-10)

            quantized = np.clip(
                np.round(data_grouped / scales[:, :, np.newaxis]),
                -8, 7
            ).astype(np.int8)
            quantized_flat = quantized.reshape(N, K)

            # Pack two INT4 values per byte: low nibble = first, high nibble = second
            # K must be even after padding
            packed = np.zeros((N, K // 2), dtype=np.uint8)
            for i in range(0, K, 2):
                low = quantized_flat[:, i] & 0x0F     
                high = (quantized_flat[:, i+1] & 0x0F) << 4
                packed[:, i // 2] = (low | high).astype(np.uint8)

            dequantized = (quantized.astype(np.float32) * scales[:, :, np.newaxis]).reshape(N, K)
            mse_error = np.mean((original_data - dequantized) ** 2)
            snr_db = 10 * np.log10(np.var(original_data) / mse_error) if mse_error > 0 else float('inf')
            original_flat = original_data.flatten()
            dequant_flat = dequantized.flatten()
            cos_sim = np.dot(original_flat, dequant_flat) / (np.linalg.norm(original_flat) * np.linalg.norm(dequant_flat) + 1e-10)

            scales_fp16 = scales.astype(np.float16)

            if stats_tracker:
                stats_tracker['int4_tensors'] += 1
                stats_tracker['quantized_parameters'] += original_data.size
                stats_tracker['mse_values'].append(mse_error)
                stats_tracker['snr_values'].append(snr_db)
                stats_tracker['cos_sim_values'].append(cos_sim)
                stats_tracker['total_tensors'] += 1
                stats_tracker['total_parameters'] += original_data.size

            with open(output_path, 'wb') as f:
                ndim = len(shape)
                data_bytes = packed.size
                scales_bytes = scales_fp16.size * 2
                flags = FLAG_HAS_SCALES
                if transpose:
                    flags |= FLAG_TRANSPOSED

                # Fixed header
                f.write(CACTUS_MAGIC)                          # 4 bytes
                f.write(struct.pack('<I', CACTUS_VERSION))     # 4 bytes
                f.write(struct.pack('<I', flags))              # 4 bytes
                f.write(struct.pack('<I', CACTUS_ALIGNMENT))   # 4 bytes
                f.write(struct.pack('<I', ndim))               # 4 bytes
                for i in range(4):
                    if i < ndim:
                        f.write(struct.pack('<Q', shape[i]))
                    else:
                        f.write(struct.pack('<Q', 0))          # 32 bytes total
                f.write(struct.pack('<I', 3))                  # precision: 3 = INT4 (4 bytes)
                f.write(struct.pack('<Q', data_bytes))         # 8 bytes
                f.write(struct.pack('<Q', scales_bytes))       # 8 bytes
                f.write(struct.pack('<I', GROUP_SIZE))         # 4 bytes
                f.write(struct.pack('<I', num_groups))         # 4 bytes
                header_size = 80
                f.write(compute_padding(header_size, CACTUS_ALIGNMENT))

                f.write(scales_fp16.tobytes())
                scales_end = align_offset(header_size, CACTUS_ALIGNMENT) + scales_bytes
                f.write(compute_padding(scales_end, CACTUS_ALIGNMENT))

                f.write(packed.tobytes())

            return
        else:
            precision = 'FP16'

    data = data.astype(np.float16)

    if stats_tracker:
        stats_tracker['fp16_tensors'] += 1
        stats_tracker['total_tensors'] += 1
        stats_tracker['total_parameters'] += original_data.size

    data_flat = data.flatten()

    with open(output_path, 'wb') as f:
        ndim = len(shape)
        data_bytes = data_flat.size * 2  # FP16 = 2 bytes
        flags = 0
        if transpose:
            flags |= FLAG_TRANSPOSED

        # Fixed header (same structure, no scales)
        f.write(CACTUS_MAGIC)                          # 4 bytes
        f.write(struct.pack('<I', CACTUS_VERSION))     # 4 bytes
        f.write(struct.pack('<I', flags))              # 4 bytes
        f.write(struct.pack('<I', CACTUS_ALIGNMENT))   # 4 bytes
        f.write(struct.pack('<I', ndim))               # 4 bytes
        for i in range(4):
            if i < ndim:
                f.write(struct.pack('<Q', shape[i]))
            else:
                f.write(struct.pack('<Q', 0))          # 32 bytes total
        f.write(struct.pack('<I', 1))                  # precision: 1 = FP16 (4 bytes)
        f.write(struct.pack('<Q', data_bytes))         # 8 bytes
        f.write(struct.pack('<Q', 0))                  # scales_bytes: 0 (8 bytes)
        f.write(struct.pack('<I', 0))                  # group_size: 0 (4 bytes)
        f.write(struct.pack('<I', 0))                  # num_groups: 0 (4 bytes)
        header_size = 80
        f.write(compute_padding(header_size, CACTUS_ALIGNMENT))

        f.write(data_flat.tobytes())


def format_config_value(value):
    """Format a config value for writing to config.txt."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (list, tuple)):
        return ','.join(str(v) for v in value)
    return str(value)


def create_quantization_stats():
    """Create an empty stats tracker dictionary for quantization metrics."""
    return {
        'total_tensors': 0,
        'int8_tensors': 0,
        'int4_tensors': 0,
        'fp16_tensors': 0,
        'total_parameters': 0,
        'quantized_parameters': 0,
        'mse_values': [],
        'snr_values': [],
        'cos_sim_values': [],
        'saturation_warnings': 0
    }


def print_quantization_summary(quantization_stats, args=None):
    """Print a summary of quantization statistics."""
    int8_count = quantization_stats.get('int8_tensors', 0)
    int4_count = quantization_stats.get('int4_tensors', 0)
    fp16_count = quantization_stats.get('fp16_tensors', 0)
    quantized_count = int8_count + int4_count

    if quantized_count > 0:
        mse_values = np.array(quantization_stats['mse_values'])
        snr_values = np.array(quantization_stats['snr_values'])
        cos_sim_values = np.array(quantization_stats['cos_sim_values'])

        print("\nQuantization Summary:")
        print(f"MSE - Mean: {np.mean(mse_values):.2e}, Max: {np.max(mse_values):.2e}, Median: {np.median(mse_values):.2e}, Min: {np.min(mse_values):.2e}")
        print(f"SNR - Mean: {np.mean(snr_values):.1f}dB, Max: {np.max(snr_values):.1f}dB, Median: {np.median(snr_values):.1f}dB, Min: {np.min(snr_values):.1f}dB")
        print(f"CosSim - Mean: {np.mean(cos_sim_values):.6f}, Max: {np.max(cos_sim_values):.6f}, Median: {np.median(cos_sim_values):.6f}, Min: {np.min(cos_sim_values):.6f}")
        print(f"Processed {int8_count} INT8 tensors, {int4_count} INT4 tensors, {fp16_count} FP16 tensors")
