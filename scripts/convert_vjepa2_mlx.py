#!/usr/bin/env python3
"""
V-JEPA2 ViT-Giant -> MLX Conversion & Feature Extraction

Architecture (from config.json):
  Encoder: 40 ViT-G layers, hidden=1408, heads=22, mlp_ratio=4.36
           3D patch embedding: [3, 2, 16, 16] -> 1408 (tubelet_size=2, patch_size=16)
           ~1.01B params (4.05 GB fp32, 2.07 GB fp16)
  Predictor: 12 layers, hidden=384, heads=12 (~22M params)
  Total: ~1.03B params

Strategy:
  1. Direct weight conversion: safetensors(PyTorch) -> safetensors(MLX)
     - No transpose needed (MLX and PyTorch both use [out, in] for Linear)
     - 3D conv (patch embed) kept as-is, handled via reshape+matmul in MLX model
     - Store as float16 to save memory (~2.0 GB)
  2. Build pure MLX model that loads converted weights
  3. Feature extraction mode: run PyTorch model, save features as .npz
  4. Benchmark inference speed on M1 Max

Usage:
  python convert_vjepa2_mlx.py --mode convert     # Convert weights to MLX
  python convert_vjepa2_mlx.py --mode benchmark    # Benchmark MLX inference
  python convert_vjepa2_mlx.py --mode extract      # Extract features via PyTorch -> .npz
  python convert_vjepa2_mlx.py --mode all          # Do everything
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Paths
MODEL_DIR = Path(__file__).parent / "vjepa2-vitg"
OUTPUT_DIR = Path(__file__).parent / "vjepa2-vitg-mlx"
CONFIG_PATH = MODEL_DIR / "config.json"
SAFETENSORS_PATH = MODEL_DIR / "model.safetensors"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────
# Step 1: Convert PyTorch safetensors weights to MLX safetensors
# ─────────────────────────────────────────────────────────────────────

def convert_weights(encoder_only=True, dtype_str="float16"):
    """
    Convert V-JEPA2 weights from PyTorch to MLX format.

    Key transformations:
    - Linear weights: NO transpose needed (MLX uses [out, in] same as PyTorch)
    - 3D patch embedding conv: [C_out, C_in, T, H, W] stays as-is (handled specially in MLX model)
    - Biases and LayerNorm: no change needed
    - Cast to float16 to halve memory usage

    Args:
        encoder_only: If True, only convert encoder (skip predictor). For feature
                      extraction we only need the encoder.
        dtype_str: Target dtype - "float16" or "float32"
    """
    import mlx.core as mx
    from safetensors import safe_open

    print("=" * 60)
    print("Step 1: Converting V-JEPA2 weights to MLX format")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()

    # Open source safetensors
    f = safe_open(str(SAFETENSORS_PATH), framework="numpy")
    keys = list(f.keys())

    target_dtype = mx.float16 if dtype_str == "float16" else mx.float32
    print(f"Source: {SAFETENSORS_PATH} ({len(keys)} tensors)")
    print(f"Target dtype: {dtype_str}")
    if encoder_only:
        keys = [k for k in keys if k.startswith("encoder.")]
        print(f"Encoder-only mode: {len(keys)} tensors")

    mlx_weights = {}
    total_params = 0
    skipped = 0

    for i, key in enumerate(sorted(keys)):
        np_tensor = f.get_tensor(key)
        shape = np_tensor.shape
        numel = np_tensor.size

        # MLX nn.Linear uses [out, in] same as PyTorch - no transpose needed
        # Conv weights and all others also kept as-is
        mlx_key = key

        # For 3D patch embedding: [C_out, C_in, T, H, W]
        # We keep it as-is and handle in the MLX model via manual conv
        if "patch_embeddings.proj.weight" in key:
            print(f"  3D patch embed: {key} shape={shape} (kept as-is)")

        mlx_tensor = mx.array(np_tensor).astype(target_dtype)
        mlx_weights[mlx_key] = mlx_tensor
        total_params += numel

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(keys)}] {key}: {shape}")

    print(f"\nTotal parameters converted: {total_params:,}")
    bytes_fp16 = total_params * 2
    print(f"Estimated size (fp16): {bytes_fp16 / 1e9:.2f} GB")

    # Save as MLX safetensors
    out_path = OUTPUT_DIR / "model.safetensors"
    print(f"Saving to {out_path} ...")
    mx.save_safetensors(str(out_path), mlx_weights)

    # Save config
    mlx_config = {
        **config,
        "mlx_dtype": dtype_str,
        "encoder_only": encoder_only,
        "num_params": total_params,
    }
    with open(OUTPUT_DIR / "config.json", "w") as cf:
        json.dump(mlx_config, cf, indent=2)

    file_size = out_path.stat().st_size
    print(f"Saved! File size: {file_size / 1e9:.2f} GB")
    print(f"Output directory: {OUTPUT_DIR}")
    return mlx_weights


# ─────────────────────────────────────────────────────────────────────
# Step 2: Pure MLX V-JEPA2 Encoder Model
# ─────────────────────────────────────────────────────────────────────

def build_mlx_model():
    """Build the V-JEPA2 encoder in pure MLX and load converted weights."""
    import mlx.core as mx
    import mlx.nn as nn

    config = load_config()

    class PatchEmbed3D(nn.Module):
        """3D patch embedding: video [B, T, C, H, W] -> [B, N, D]
        Uses the 5D conv kernel stored as a raw parameter, applied via
        strided reshape + matmul (since MLX doesn't have native 3D conv).
        """
        def __init__(self, in_chans=3, hidden_size=1408, patch_size=16, tubelet_size=2):
            super().__init__()
            self.patch_size = patch_size
            self.tubelet_size = tubelet_size
            self.hidden_size = hidden_size
            self.in_chans = in_chans
            # The proj weight is [C_out, C_in, T, H, W] from PyTorch
            # After our conversion it stays as [C_out, C_in, T, H, W]
            # We'll reshape input patches and do matmul
            kernel_size = in_chans * tubelet_size * patch_size * patch_size
            # We store proj as the weight itself (loaded from safetensors)
            self.proj_weight = mx.zeros((hidden_size, in_chans, tubelet_size, patch_size, patch_size))
            self.proj_bias = mx.zeros((hidden_size,))

        def __call__(self, x):
            # x: [B, C, T, H, W] (channels first from preprocessor)
            B, C, T, H, W = x.shape
            ps = self.patch_size
            ts = self.tubelet_size

            nT = T // ts
            nH = H // ps
            nW = W // ps

            # Reshape into patches: [B, nT, ts, nH, ps, nW, ps, C]
            # Then flatten each patch
            x = mx.reshape(x, (B, C, nT, ts, nH, ps, nW, ps))
            x = mx.transpose(x, (0, 2, 4, 6, 1, 3, 5, 7))  # [B, nT, nH, nW, C, ts, ps, ps]
            x = mx.reshape(x, (B, nT * nH * nW, C * ts * ps * ps))  # [B, N, patch_dim]

            # Weight: [C_out, C_in, ts, ps, ps] -> [C_out, patch_dim]
            w = mx.reshape(self.proj_weight, (self.hidden_size, -1))  # [hidden, patch_dim]
            # patches @ w.T = [B, N, patch_dim] @ [patch_dim, hidden] = [B, N, hidden]
            out = x @ mx.transpose(w, (1, 0)) + self.proj_bias  # [B, N, hidden]
            return out

    class Attention(nn.Module):
        def __init__(self, hidden_size=1408, num_heads=22):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            self.scale = self.head_dim ** -0.5
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)
            self.proj = nn.Linear(hidden_size, hidden_size)

        def __call__(self, x):
            B, N, C = x.shape
            h = self.num_heads
            d = self.head_dim

            q = self.query(x).reshape(B, N, h, d).transpose(0, 2, 1, 3)  # [B, h, N, d]
            k = self.key(x).reshape(B, N, h, d).transpose(0, 2, 1, 3)
            v = self.value(x).reshape(B, N, h, d).transpose(0, 2, 1, 3)

            # Scaled dot-product attention
            attn = (q @ mx.transpose(k, (0, 1, 3, 2))) * self.scale  # [B, h, N, N]
            attn = mx.softmax(attn, axis=-1)
            out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)  # [B, N, C]
            return self.proj(out)

    class MLP(nn.Module):
        def __init__(self, hidden_size=1408, mlp_ratio=4.363636363636363):
            super().__init__()
            mlp_hidden = int(hidden_size * mlp_ratio)
            self.fc1 = nn.Linear(hidden_size, mlp_hidden)
            self.fc2 = nn.Linear(mlp_hidden, hidden_size)

        def __call__(self, x):
            return self.fc2(nn.gelu(self.fc1(x)))

    class TransformerBlock(nn.Module):
        def __init__(self, hidden_size=1408, num_heads=22, mlp_ratio=4.363636363636363):
            super().__init__()
            self.attention = Attention(hidden_size, num_heads)
            self.mlp = MLP(hidden_size, mlp_ratio)
            self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)

        def __call__(self, x):
            x = x + self.attention(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x

    class VJEPA2Encoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            hs = config["hidden_size"]
            nh = config["num_attention_heads"]
            nl = config["num_hidden_layers"]
            mlp_r = config["mlp_ratio"]
            ps = config["patch_size"]
            ts = config["tubelet_size"]
            ic = config["in_chans"]

            self.embeddings = PatchEmbed3D(ic, hs, ps, ts)
            self.layer = [TransformerBlock(hs, nh, mlp_r) for _ in range(nl)]
            self.layernorm = nn.LayerNorm(hs, eps=1e-6)

        def __call__(self, pixel_values):
            # pixel_values: [B, C, T, H, W]
            x = self.embeddings(pixel_values)
            for block in self.layer:
                x = block(x)
            x = self.layernorm(x)
            return x

    class VJEPA2Model(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.encoder = VJEPA2Encoder(config)

        def __call__(self, pixel_values):
            return self.encoder(pixel_values)

    # Build model
    model = VJEPA2Model(config)

    # Load weights
    mlx_safetensors = OUTPUT_DIR / "model.safetensors"
    if not mlx_safetensors.exists():
        print("MLX weights not found. Run --mode convert first.")
        return None

    print("Loading MLX weights...")
    weights = mx.load(str(mlx_safetensors))

    # Map flat safetensors keys to nested model structure
    # The keys are like "encoder.layer.0.attention.query.weight"
    # MLX nn.Module.load_weights expects list of (key, tensor) tuples
    weight_list = list(weights.items())

    # Patch embed conv weight needs special handling:
    # It was NOT transposed during conversion (5D tensor), keep as-is
    # but we need to assign to PatchEmbed3D.proj_weight / proj_bias
    remapped = []
    for k, v in weight_list:
        if k == "encoder.embeddings.patch_embeddings.proj.weight":
            remapped.append(("encoder.embeddings.proj_weight", v))
        elif k == "encoder.embeddings.patch_embeddings.proj.bias":
            remapped.append(("encoder.embeddings.proj_bias", v))
        else:
            remapped.append((k, v))

    model.load_weights(remapped)
    mx.eval(model.parameters())
    print("MLX model loaded successfully!")
    return model, config


# ─────────────────────────────────────────────────────────────────────
# Step 3: Benchmark MLX inference
# ─────────────────────────────────────────────────────────────────────

def benchmark_mlx():
    """Benchmark MLX V-JEPA2 encoder inference on M1 Max."""
    import mlx.core as mx

    print("=" * 60)
    print("Step 3: Benchmarking MLX V-JEPA2 inference")
    print("=" * 60)

    result = build_mlx_model()
    if result is None:
        return
    model, config = result

    # Test configurations - memory-aware for M1 Max 64GB
    # 64 frames @ 256x256 = 8192 patches -> ~118 GB attention memory (impossible)
    # Key insight: sequence_length = (T/2) * (H/16) * (W/16)
    # N=2048 -> ~7.4 GB attention (feasible)
    # N=4096 -> ~29.6 GB attention (tight)
    test_configs = [
        # (B, T, H, W, label)
        (1, 2, 256, 256, "2 frames 256x256 (minimal, N=256)"),
        (1, 8, 256, 256, "8 frames 256x256 (N=1024)"),
        (1, 16, 256, 256, "16 frames 256x256 (N=2048)"),
        (1, 16, 128, 128, "16 frames 128x128 (N=512, low-res)"),
    ]

    for B, T, H, W, label in test_configs:
        print(f"\n--- {label} ---")
        C = config["in_chans"]

        # Create dummy input
        x = mx.random.normal((B, C, T, H, W)).astype(mx.float16)
        mx.eval(x)

        ps = config["patch_size"]
        ts = config["tubelet_size"]
        num_patches = (T // ts) * (H // ps) * (W // ps)
        print(f"  Input: [{B}, {C}, {T}, {H}, {W}]")
        print(f"  Patches: {num_patches} = {T//ts} x {H//ps} x {W//ps}")
        print(f"  Attention matrix: [{num_patches}, {num_patches}] per head")

        # Check memory feasibility
        # Attention needs O(N^2) memory per head per layer
        attn_mem_per_layer = num_patches * num_patches * config["num_attention_heads"] * 2  # fp16
        total_attn_mem = attn_mem_per_layer * config["num_hidden_layers"]
        print(f"  Est. attention memory: {total_attn_mem / 1e9:.2f} GB (all layers)")

        if total_attn_mem > 40e9:  # >40GB - too risky
            print(f"  SKIP: Would use too much memory")
            continue

        try:
            # Warmup
            print("  Running warmup...")
            t0 = time.time()
            out = model(x)
            mx.eval(out)
            warmup_time = time.time() - t0
            print(f"  Warmup: {warmup_time:.2f}s, output shape: {out.shape}")

            # Benchmark (3 runs)
            times = []
            for run in range(3):
                t0 = time.time()
                out = model(x)
                mx.eval(out)
                elapsed = time.time() - t0
                times.append(elapsed)
                print(f"  Run {run+1}: {elapsed:.2f}s")

            avg = np.mean(times)
            print(f"  Average: {avg:.2f}s ({1/avg:.2f} FPS)")
            print(f"  Output: {out.shape} {out.dtype}")

        except Exception as e:
            print(f"  ERROR: {e}")
            # Try to free memory
            del x
            if 'out' in dir():
                del out
            import gc
            gc.collect()

    # Chunked inference for full 64 frames
    print("\n--- Chunked inference: 64 frames via 8-frame chunks ---")
    print("  Strategy: split 64 frames into 8 chunks of 8 frames each,")
    print("  run encoder on each chunk, concatenate features.")
    T_total = 64
    chunk_T = 8  # 8 frames per chunk -> 1024 patches -> ~1.85 GB attn
    C = config["in_chans"]
    H = W = 256
    n_chunks = T_total // chunk_T

    x_full = mx.random.normal((1, C, T_total, H, W)).astype(mx.float16)
    mx.eval(x_full)

    t0 = time.time()
    chunk_outputs = []
    for ci in range(n_chunks):
        x_chunk = x_full[:, :, ci*chunk_T:(ci+1)*chunk_T, :, :]
        out_chunk = model(x_chunk)
        mx.eval(out_chunk)
        chunk_outputs.append(out_chunk)
    features = mx.concatenate(chunk_outputs, axis=1)
    mx.eval(features)
    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Output: {features.shape} (concatenated from {n_chunks} chunks)")
    print(f"  Note: full 64-frame single-pass would need ~118 GB attention memory")
    print(f"  Chunked approach: ~1.85 GB per chunk, easily fits in 64 GB")

    del x_full, chunk_outputs, features
    import gc
    gc.collect()

    print("\n" + "=" * 60)
    print("Benchmark complete")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────
# Step 4: PyTorch feature extraction (fallback)
# ─────────────────────────────────────────────────────────────────────

def extract_features_pytorch(video_path=None):
    """
    Extract V-JEPA2 features using PyTorch and save as .npz.
    This is the fallback if MLX inference is too slow.
    """
    import torch

    print("=" * 60)
    print("Step 4: PyTorch Feature Extraction")
    print("=" * 60)

    config = load_config()

    # Try loading via transformers AutoModel
    print("Loading V-JEPA2 via transformers...")
    try:
        from transformers import AutoModel, AutoVideoProcessor
        model = AutoModel.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float32)
        processor = AutoVideoProcessor.from_pretrained(str(MODEL_DIR))
        model.eval()
        print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"AutoModel failed: {e}")
        print("Trying manual safetensors load...")

        from safetensors.torch import load_file
        weights = load_file(str(SAFETENSORS_PATH))
        print(f"Loaded {len(weights)} tensors manually")
        print("Manual loading requires model class definition - skipping")
        return

    # Create dummy video input for benchmarking
    T = config["frames_per_clip"]  # 64
    H = W = config["image_size"]   # 256

    print(f"\nCreating dummy video: {T} frames x {H}x{W}")
    dummy_frames = [np.random.randint(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(T)]

    # Process
    inputs = processor(dummy_frames, return_tensors="pt")
    print(f"Processed input keys: {list(inputs.keys())}")
    for k, v in inputs.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape} {v.dtype}")

    # Inference
    print("\nRunning PyTorch inference...")
    with torch.no_grad():
        t0 = time.time()
        outputs = model.get_vision_features(**inputs)
        elapsed = time.time() - t0

    print(f"Output shape: {outputs.shape}")
    print(f"Inference time: {elapsed:.2f}s")

    # Save features
    features_dir = OUTPUT_DIR / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    out_path = features_dir / "dummy_features.npz"
    np.savez_compressed(
        str(out_path),
        features=outputs.numpy(),
        config=json.dumps(config),
    )
    print(f"Features saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Benchmark
    print("\nBenchmarking PyTorch (3 runs)...")
    times = []
    for i in range(3):
        with torch.no_grad():
            t0 = time.time()
            _ = model.get_vision_features(**inputs)
            elapsed = time.time() - t0
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.2f}s")
    print(f"  Average: {np.mean(times):.2f}s")

    return outputs


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V-JEPA2 ViT-G MLX Converter")
    parser.add_argument("--mode", choices=["convert", "benchmark", "extract", "all", "info"],
                        default="info", help="Operation mode")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16",
                        help="Target dtype for MLX weights")
    parser.add_argument("--encoder-only", action="store_true", default=True,
                        help="Only convert encoder (default, skip predictor)")
    parser.add_argument("--full-model", action="store_true",
                        help="Convert full model including predictor")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file for feature extraction")
    args = parser.parse_args()

    if args.full_model:
        args.encoder_only = False

    if args.mode == "info":
        config = load_config()
        print("V-JEPA2 ViT-Giant Model Info")
        print("=" * 50)
        print(f"Architecture: VJEPA2Model")
        print(f"Hidden size: {config['hidden_size']}")
        print(f"Attention heads: {config['num_attention_heads']}")
        print(f"Encoder layers: {config['num_hidden_layers']}")
        print(f"MLP ratio: {config['mlp_ratio']:.2f}")
        print(f"Patch size: {config['patch_size']}")
        print(f"Tubelet size: {config['tubelet_size']}")
        print(f"Frames per clip: {config['frames_per_clip']}")
        print(f"Image size: {config['image_size']}")
        print(f"Predictor layers: {config['pred_num_hidden_layers']}")
        print(f"Predictor hidden: {config['pred_hidden_size']}")
        print()
        print(f"Source weights: {SAFETENSORS_PATH}")
        print(f"  Size: {SAFETENSORS_PATH.stat().st_size / 1e9:.2f} GB")
        print(f"MLX output: {OUTPUT_DIR}")
        if (OUTPUT_DIR / "model.safetensors").exists():
            sz = (OUTPUT_DIR / "model.safetensors").stat().st_size
            print(f"  MLX weights: {sz / 1e9:.2f} GB (already converted)")
        else:
            print(f"  MLX weights: not yet converted")

        # Memory estimation for full inference
        T, H, W = 64, 256, 256
        ps, ts = config['patch_size'], config['tubelet_size']
        N = (T // ts) * (H // ps) * (W // ps)
        print(f"\nFull inference ({T} frames, {H}x{W}):")
        print(f"  Sequence length: {N} patches")
        print(f"  Model memory (fp16): ~2.1 GB")
        attn_gb = N * N * config['num_attention_heads'] * 2 * config['num_hidden_layers'] / 1e9
        print(f"  Attention memory (fp16): ~{attn_gb:.1f} GB")
        print(f"  Total estimated: ~{2.1 + attn_gb:.1f} GB")
        print(f"  M1 Max 64GB: {'feasible' if 2.1 + attn_gb < 50 else 'tight'}")

    elif args.mode == "convert":
        convert_weights(encoder_only=args.encoder_only, dtype_str=args.dtype)

    elif args.mode == "benchmark":
        benchmark_mlx()

    elif args.mode == "extract":
        extract_features_pytorch(video_path=args.video)

    elif args.mode == "all":
        convert_weights(encoder_only=args.encoder_only, dtype_str=args.dtype)
        print("\n")
        benchmark_mlx()
        print("\n")
        extract_features_pytorch(video_path=args.video)


if __name__ == "__main__":
    main()
