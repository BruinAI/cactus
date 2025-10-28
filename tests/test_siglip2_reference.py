"""
SigLip2 Reference Implementation Test Script (Hugging Face Processor)
====================================================================
This script uses the official Hugging Face SigLIP-2 image processor
to preprocess an image and writes results you can compare with your C++
implementation.

Usage in Google Colab:
1) !pip install -q transformers pillow numpy
2) Upload your test image (or pass a path/URL you can read)
3) Run:  python test_siglip2_hf.py <image_path> [output_path] [ckpt]

Defaults:
- ckpt = 'google/siglip2-base-patch16-naflex'
"""

import sys
import io
import os
import numpy as np
from typing import Optional
from PIL import Image

def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        import requests
        resp = requests.get(path_or_url, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        return Image.open(path_or_url).convert("RGB")

def save_output_hf(result: dict, output_path: str, patch_size_hint: Optional[int] = None):
    """
    Save outputs from the HF processor call. For NaFlex checkpoints, the dict may contain:
      - pixel_values: (1, C, H, W)
      - pixel_attention_mask: (1, H, W) or (H, W)
      - spatial_shapes: (1, 2)  (height, width)
    For fixed-size SigLIP-2 checkpoints, only pixel_values is typically returned.
    """
    with open(output_path, "w") as f:
        f.write("=== METADATA ===\n")
        pv = result.get("pixel_values", None)
        if pv is None:
            f.write("pixel_values: None (unexpected)\n")
            return

        # Ensure numpy array
        if not isinstance(pv, np.ndarray):
            try:
                pv = pv.detach().cpu().numpy()  # torch tensor -> numpy
            except Exception:
                raise RuntimeError("Unsupported pixel_values type; expected NumPy or torch.Tensor.")

        f.write(f"pixel_values_shape: {pv.shape}\n")
        f.write(f"pixel_values_dtype: {pv.dtype}\n")

        # Pull CHW and batch dims if present
        if pv.ndim == 4:
            b, c, h, w = pv.shape
        elif pv.ndim == 3:
            # Some processors may return unbatched CHW
            c, h, w = pv.shape
            b = 1
            pv = pv[None, ...]
        else:
            raise ValueError(f"Unsupported pixel_values shape: {pv.shape}")

        # Optional fields (NaFlex-aware processors)
        pam = result.get("pixel_attention_mask", None)
        ss = result.get("spatial_shapes", None)

        if pam is not None:
            if isinstance(pam, np.ndarray):
                pam_np = pam
            else:
                try:
                    pam_np = pam.detach().cpu().numpy()
                except Exception:
                    raise RuntimeError("Unsupported pixel_attention_mask type; expected NumPy or torch.Tensor.")
            f.write(f"pixel_attention_mask_shape: {pam_np.shape}\n")
        else:
            f.write("pixel_attention_mask: None (likely a non-NaFlex checkpoint)\n")

        if ss is not None:
            if isinstance(ss, np.ndarray):
                ss_np = ss
            else:
                try:
                    ss_np = ss.detach().cpu().numpy()
                except Exception:
                    raise RuntimeError("Unsupported spatial_shapes type; expected NumPy or torch.Tensor.")
            f.write(f"spatial_shapes: {ss_np.tolist()}\n")
        else:
            f.write("spatial_shapes: None (likely a fixed-size checkpoint)\n")

        # Basic stats for the first (or only) image in batch
        f.write("\n=== STATISTICS (first image) ===\n")
        first = pv[0]  # (C,H,W)
        per_channel_means = [f"{float(first[i].mean()):.6f}" for i in range(first.shape[0])]
        f.write(f"Per-channel means: {per_channel_means}\n")
        f.write(f"Global min: {float(first.min()):.6f}\n")
        f.write(f"Global max: {float(first.max()):.6f}\n")
        f.write(f"Global mean: {float(first.mean()):.6f}\n")

        # Dump a small slice so you can spot-check numeric parity
        f.write("\n=== SAMPLE VALUES (first channel, top-left 8x8) ===\n")
        sample = first[0, :8, :8].reshape(-1)
        for i, v in enumerate(sample):
            f.write(f"{float(v):.6f} ")
            if (i + 1) % 8 == 0:
                f.write("\n")

        # If an attention mask exists, print it (it's 1D for NaFlex)
        if pam is not None:
            f.write("\n=== ATTENTION MASK (1D, num_patches) ===\n")
            pam1d = pam_np[0] if pam_np.ndim == 2 else pam_np  # Remove batch dim if present
            # Print 16 values per line for readability
            for i in range(0, len(pam1d), 16):
                row = pam1d[i:i+16]
                f.write(" ".join(str(int(x)) for x in row) + "\n")
            # Summary
            num_valid = int(pam1d.sum())
            num_padding = len(pam1d) - num_valid
            f.write(f"\nValid patches: {num_valid}, Padding: {num_padding}\n")

        # If you want to *estimate* patch grid (for comparison) and you know patch size:
        if patch_size_hint is not None:
            f.write("\n=== PATCH GRID ESTIMATE (using patch_size_hint) ===\n")
            est_h = h // patch_size_hint
            est_w = w // patch_size_hint
            f.write(f"estimated_num_patches_height: {est_h}\n")
            f.write(f"estimated_num_patches_width:  {est_w}\n")
            f.write(f"estimated_total_patches:      {est_h * est_w}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_siglip2_hf.py <image_path_or_url> [output_path] [ckpt]")
        print("Example: python test_siglip2_hf.py test_image.png output_python.txt google/siglip2-base-patch16-naflex")
        return

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "siglip2_output_hf.txt"
    ckpt = sys.argv[3] if len(sys.argv) > 3 else "google/siglip2-base-patch16-naflex"

    print("=== SigLip2 HF Processor Test ===")
    print(f"Image: {image_path}")
    print(f"Checkpoint: {ckpt}\n")

    # Lazy import transformers
    from transformers import AutoImageProcessor

    try:
        # Load image
        print("Loading image...")
        image = load_image(image_path)  # PIL RGB

        # Load the processor bound to the checkpoint.
        # For SigLIP-2, this instantiates Siglip2ImageProcessor under the hood.
        print("Loading HF image processor...")
        processor = AutoImageProcessor.from_pretrained(ckpt)

        # Run official preprocessing. By default this performs resizing (NaFlex-aware),
        # channel reordering to CHW, rescale/normalize etc., returning a dict.
        # Use PyTorch tensors (the fast processor only supports "pt").
        print("Preprocessing via HF processor...")
        processed = processor(images=image, return_tensors="pt")

        # Save results
        print(f"Saving detailed output to: {output_path}")
        # If you know the patch size for your checkpoint, pass it here to estimate patch grid.
        # Most SigLIP-2 models use patch_size=16.
        save_output_hf(processed, output_path, patch_size_hint=16)

        # Console summary
        pv = processed["pixel_values"]
        if not isinstance(pv, np.ndarray):
            pv = pv.detach().cpu().numpy()
        print("\n=== Summary ===")
        print(f"pixel_values shape: {pv.shape} (batch, C, H, W)")
        print(f"pixel_values dtype: {pv.dtype}")
        if "pixel_attention_mask" in processed:
            pam = processed["pixel_attention_mask"]
            if not isinstance(pam, np.ndarray):
                pam = pam.detach().cpu().numpy()
            print(f"pixel_attention_mask shape: {pam.shape}")
        if "spatial_shapes" in processed:
            ss = processed["spatial_shapes"]
            if not isinstance(ss, np.ndarray):
                ss = ss.detach().cpu().numpy()
            print(f"spatial_shapes: {ss.tolist()}")

        print("\n✓ Success! Compare these values/shapes with your C++ implementation.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
