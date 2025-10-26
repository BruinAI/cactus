"""
SigLip2 Reference Implementation Test Script
============================================
This script uses the HuggingFace reference implementation to preprocess
an image and outputs results that can be compared with the C++ implementation.

Usage in Google Colab:
1. Upload your test image
2. Run this script
3. Compare output with C++ implementation
"""

import numpy as np
from PIL import Image
import sys

# The reference code from image_processing_siglip2.py
import math
from functools import lru_cache

@lru_cache(maxsize=256)
def get_image_size_for_max_num_patches(
    image_height: int, image_width: int, patch_size: int, max_num_patches: int, eps: float = 1e-5
) -> tuple[int, int]:
    """
    Determine image size based on max number of patches, ensure dimensions are divisible by patch size and image is at least 1 patch.
    """
    def get_scaled_image_size(scale: float, size: int, patch_size: int) -> int:
        scaled_size = size * scale
        scaled_size = math.ceil(scaled_size / patch_size) * patch_size  # make divisible by patch_size
        scaled_size = max(patch_size, scaled_size)  # ensure at least 1 patch
        return int(scaled_size)

    # Binary search for optimal scale
    scale_min, scale_max = eps / 10, 100.0
    while (scale_max - scale_min) >= eps:
        scale = (scale_min + scale_max) / 2
        target_height = get_scaled_image_size(scale, image_height, patch_size)
        target_width = get_scaled_image_size(scale, image_width, patch_size)
        num_patches = (target_height / patch_size) * (target_width / patch_size)

        if num_patches <= max_num_patches:
            scale_min = scale
        else:
            scale_max = scale

    scale = scale_min
    target_height = get_scaled_image_size(scale, image_height, patch_size)
    target_width = get_scaled_image_size(scale, image_width, patch_size)
    return target_height, target_width


def convert_image_to_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Convert 3D array image of shape (image_height, image_width, num_channels) into 2D array of patches of shape
    (num_patches_height * num_patches_width, patch_size * patch_size * num_channels).
    """
    image_height, image_width, num_channels = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(num_patches_height, patch_size, num_patches_width, patch_size, num_channels)
    patched_image = patched_image.transpose(0, 2, 1, 3, 4)
    patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
    return patched_image


def pad_along_first_dim(array: np.ndarray, target_length: int, pad_value: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad the array along the first dimension.
    """
    current_length = array.shape[0]
    padding_length = target_length - current_length
    mask = np.ones((target_length,), dtype=np.int32)
    if padding_length > 0:
        paddings = [(0, padding_length)] + [(0, 0)] * (array.ndim - 1)
        array = np.pad(array, paddings, mode="constant", constant_values=pad_value)
        mask[-padding_length:] = 0
    return array, mask


def preprocess_image(image_path, patch_size=16, max_num_patches=256, 
                     rescale_factor=1/255, image_mean=None, image_std=None):
    """
    Preprocess image following the exact SigLip2 pipeline
    """
    if image_mean is None:
        image_mean = [0.5, 0.5, 0.5]
    if image_std is None:
        image_std = [0.5, 0.5, 0.5]
    
    # Load image
    image = Image.open(image_path)
    
    # Convert to RGB
    image = image.convert('RGB')
    
    # Convert to numpy array
    image = np.array(image)  # Shape: (H, W, 3), dtype: uint8
    
    # Get target size
    height, width = get_image_size_for_max_num_patches(
        image_height=image.shape[0],
        image_width=image.shape[1],
        patch_size=patch_size,
        max_num_patches=max_num_patches,
    )
    
    # Resize using PIL (BILINEAR)
    if height != image.shape[0] or width != image.shape[1]:
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((width, height), Image.BILINEAR)
        image = np.array(pil_img)
    
    # Rescale: [0, 255] -> [0, 1]
    image = image.astype(np.float32) * rescale_factor
    
    # Normalize: (pixel - mean) / std
    image_mean_np = np.array(image_mean, dtype=np.float32)
    image_std_np = np.array(image_std, dtype=np.float32)
    image = (image - image_mean_np) / image_std_np
    
    # Convert to patches
    patches = convert_image_to_patches(image, patch_size)
    
    # Pad patches
    patches, mask = pad_along_first_dim(patches, max_num_patches)
    
    # Calculate spatial shapes
    num_patches_height = image.shape[0] // patch_size
    num_patches_width = image.shape[1] // patch_size
    
    return {
        'pixel_values': patches,
        'pixel_attention_mask': mask,
        'num_patches_height': num_patches_height,
        'num_patches_width': num_patches_width,
        'actual_num_patches': num_patches_height * num_patches_width
    }


def save_output(result, output_path):
    """Save preprocessing results to file"""
    with open(output_path, 'w') as f:
        f.write("=== METADATA ===\n")
        f.write(f"num_patches_height: {result['num_patches_height']}\n")
        f.write(f"num_patches_width: {result['num_patches_width']}\n")
        f.write(f"actual_num_patches: {result['actual_num_patches']}\n")
        f.write(f"pixel_values_shape: {result['pixel_values'].shape}\n")
        f.write("\n")
        
        # Attention mask
        f.write("=== ATTENTION MASK ===\n")
        mask = result['pixel_attention_mask']
        for i in range(len(mask)):
            f.write(str(mask[i]))
            if (i + 1) % 16 == 0:
                f.write("\n")
            else:
                f.write(" ")
        f.write("\n\n")
        
        # First 5 patches statistics
        f.write("=== FIRST 5 PATCHES STATISTICS ===\n")
        pixel_values = result['pixel_values']
        for patch_idx in range(min(5, len(mask))):
            if mask[patch_idx] == 0:
                continue
            
            f.write(f"Patch {patch_idx}:\n")
            patch = pixel_values[patch_idx]
            f.write(f"  Min: {patch.min():.6f}\n")
            f.write(f"  Max: {patch.max():.6f}\n")
            f.write(f"  Mean: {patch.mean():.6f}\n")
            f.write(f"  First 10 values: {' '.join(f'{v:.6f}' for v in patch[:10])}\n")
        f.write("\n")
        
        # First patch complete values
        f.write("=== FIRST PATCH COMPLETE VALUES ===\n")
        if mask[0] == 1:
            first_patch = pixel_values[0]
            for i in range(min(len(first_patch), 768)):  # 16*16*3 = 768
                f.write(f"{first_patch[i]:.6f}")
                if (i + 1) % 8 == 0:
                    f.write("\n")
                else:
                    f.write(" ")
        f.write("\n\n")
        
        # Global statistics
        f.write("=== GLOBAL STATISTICS ===\n")
        valid_patches = np.sum(mask)
        valid_pixel_values = pixel_values[mask == 1]
        f.write(f"Valid patches: {valid_patches}\n")
        f.write(f"Global min: {valid_pixel_values.min():.6f}\n")
        f.write(f"Global max: {valid_pixel_values.max():.6f}\n")
        f.write(f"Global mean: {valid_pixel_values.mean():.6f}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_siglip2_reference.py <image_path> [output_path]")
        print("Example: python test_siglip2_reference.py test_image.png output_python.txt")
        return
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "siglip2_output_python.txt"
    
    print("=== SigLip2 Reference Implementation Test ===")
    print(f"Image: {image_path}")
    print()
    
    try:
        # Preprocess image
        print("Loading and preprocessing image...")
        result = preprocess_image(
            image_path,
            patch_size=16,
            max_num_patches=256,
            rescale_factor=1/255,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5]
        )
        
        # Print results
        print("\n=== Preprocessing Results ===")
        print(f"Number of patches (height x width): {result['num_patches_height']} x {result['num_patches_width']}")
        print(f"Actual number of patches: {result['actual_num_patches']}")
        print(f"Padded to: {len(result['pixel_attention_mask'])} patches")
        print(f"Pixel values shape: {result['pixel_values'].shape}")
        print(f"Pixel values dtype: {result['pixel_values'].dtype}")
        
        # Statistics
        valid_mask = result['pixel_attention_mask'] == 1
        valid_pixels = result['pixel_values'][valid_mask]
        print(f"\nPixel value range: [{valid_pixels.min():.6f}, {valid_pixels.max():.6f}]")
        print(f"Valid patches (mask=1): {np.sum(valid_mask)}")
        
        # Save output
        print(f"\nSaving detailed output to: {output_path}")
        save_output(result, output_path)
        
        print("\n✓ Success!")
        print("\nNext steps:")
        print("1. Compare this output with the C++ output")
        print("2. Check that metadata, masks, and pixel values match")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

