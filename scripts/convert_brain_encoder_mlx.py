#!/usr/bin/env python3
"""
Convert TRIBE v2 Brain Encoder from PyTorch checkpoint to MLX.

Architecture: 8-depth Transformer (16 total layers: 8 attention + 8 feedforward)
  - dim=1152, heads=8, head_dim=144
  - ScaleNorm (use_scalenorm=true)
  - Rotary positional embeddings
  - scale_residual=true
  - FeedForward with ff_mult=4 (inner_dim=4608), GELU activation
  - No cross-attention, non-causal
"""

import os
import sys
import time
import numpy as np

# ── Paths ──
CKPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(CKPT_DIR, "tribev2_ckpt", "best.ckpt")
OUTPUT_DIR = os.path.join(CKPT_DIR, "tribev2_mlx")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "brain_encoder.safetensors")

# ── Config ──
DIM = 1152
HEADS = 8
HEAD_DIM = DIM // HEADS  # 144
DEPTH = 8  # 8 attention + 8 ff = 16 total layers
FF_MULT = 4
FF_INNER = DIM * FF_MULT  # 4608
ROTARY_DIM = HEAD_DIM // 2  # 72 -> inv_freq has shape [36]
MAX_SEQ_LEN = 1024


# ==============================================================================
# Part 1: Load PyTorch checkpoint and extract encoder weights
# ==============================================================================

def load_pytorch_weights(verbose=True):
    """Load the PyTorch checkpoint and extract only the encoder weights."""
    import torch

    if verbose:
        print("Loading PyTorch checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # Extract only model.encoder.* keys
    encoder_weights = {}
    for k, v in state_dict.items():
        if k.startswith("model.encoder."):
            short_key = k[len("model.encoder."):]
            encoder_weights[short_key] = v.numpy().astype(np.float32)

    if verbose:
        print(f"Extracted {len(encoder_weights)} encoder weight tensors")
        for k, v in sorted(encoder_weights.items()):
            print(f"  {k}: {v.shape}")

    return encoder_weights


# ==============================================================================
# Part 2: MLX Brain Encoder Implementation
# ==============================================================================

import mlx.core as mx
import mlx.nn as nn


class ScaleNorm(nn.Module):
    """
    x-transformers ScaleNorm with unit_offset=True (default).
    gamma = g + 1.0, output = normalize(x) * dim**0.5 * gamma
    """

    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        # In x-transformers with unit_offset=True, g is initialized to 0.0
        # so gamma = g + 1.0 starts at 1.0.
        # The checkpoint stores the trained g value.
        self.g = mx.zeros((1,))  # will be loaded from checkpoint

    def __call__(self, x):
        gamma = self.g + 1.0  # unit_offset=True
        norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-12)
        x_normed = x / norm
        return x_normed * self.scale * gamma


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings matching x-transformers implementation."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        # dim is the rotary dim (72 for head_dim=144, using half).
        # inv_freq shape = [dim//2] = [36]
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        self.inv_freq = mx.array(inv_freq)  # shape [dim//2]

    def __call__(self, seq_len: int):
        """
        Return freqs of shape [1, seq_len, dim].

        x-transformers computes:
          freqs = einsum(t, inv_freq)  -> [1, seq, dim//2]
          freqs = stack((freqs, freqs), dim=-1) -> [1, seq, dim//2, 2]
          freqs = rearrange('... d r -> ... (d r)') -> [1, seq, dim]
        Result is [f0, f0, f1, f1, ...] -- each freq duplicated adjacently.
        """
        t = mx.arange(seq_len).astype(mx.float32).reshape(1, -1)  # [1, seq]
        freqs = mx.einsum("bi,j->bij", t, self.inv_freq)           # [1, seq, dim//2]
        freqs = mx.stack([freqs, freqs], axis=-1)                   # [1, seq, dim//2, 2]
        return mx.reshape(freqs, (1, seq_len, -1))                  # [1, seq, dim]


def rotate_half(x):
    """
    x-transformers rotate_half:
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

    Input shape: [..., dim] where dim is even
    Split into pairs: [..., dim//2, 2] -> x1=even indices, x2=odd indices
    Result: [-x2, x1] interleaved -> [-odd, even, -odd, even, ...]
    """
    # Reshape to [..., dim//2, 2]
    shape = x.shape
    d = shape[-1]
    x_pairs = mx.reshape(x, shape[:-1] + (d // 2, 2))
    x1 = x_pairs[..., 0]  # even indices
    x2 = x_pairs[..., 1]  # odd indices
    # Stack (-x2, x1) on last dim
    result = mx.stack([-x2, x1], axis=-1)
    return mx.reshape(result, shape)


def apply_rotary_pos_emb(t, freqs):
    """
    Apply rotary position embeddings.
    t: [batch, heads, seq, head_dim]
    freqs: [1, seq, rot_dim] where rot_dim <= head_dim

    Only applies to the first rot_dim dimensions of t.
    """
    rot_dim = freqs.shape[-1]
    seq_len = t.shape[-2]

    # Slice freqs to match sequence length
    freqs = freqs[:, -seq_len:, :]

    # Add head dimension: [1, 1, seq, rot_dim]
    freqs = mx.expand_dims(freqs, axis=1)

    # Split rotary and pass-through dimensions
    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]

    # Apply rotation
    t_rot = (t_rot * mx.cos(freqs)) + (rotate_half(t_rot) * mx.sin(freqs))

    return mx.concatenate([t_rot, t_pass], axis=-1)


class Attention(nn.Module):
    """Multi-head self-attention matching x-transformers Attention."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        # LinearNoBias in x-transformers
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def __call__(self, x, rotary_pos_emb=None):
        B, N, D = x.shape
        h = self.heads
        d = self.head_dim

        q = self.to_q(x).reshape(B, N, h, d).transpose(0, 2, 1, 3)  # [B, h, N, d]
        k = self.to_k(x).reshape(B, N, h, d).transpose(0, 2, 1, 3)
        v = self.to_v(x).reshape(B, N, h, d).transpose(0, 2, 1, 3)

        # Apply rotary position embeddings
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb(q, rotary_pos_emb)
            k = apply_rotary_pos_emb(k, rotary_pos_emb)

        # Scaled dot-product attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)

        out = attn @ v  # [B, h, N, d]
        out = out.transpose(0, 2, 1, 3).reshape(B, N, D)  # [B, N, D]
        return self.to_out(out)


class FeedForward(nn.Module):
    """FeedForward matching x-transformers (GELU, with bias)."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner = dim * mult
        # x-transformers uses Sequential(Linear+GELU, Dropout, Linear, Dropout)
        # In checkpoint: ff.0.0 = Linear(dim, inner), ff.2 = Linear(inner, dim)
        # ff.0.1 = GELU (no params)
        # ff.1 = Dropout (no params)
        self.w1 = nn.Linear(dim, inner, bias=True)
        self.w2 = nn.Linear(inner, dim, bias=True)

    def __call__(self, x):
        return self.w2(nn.gelu(self.w1(x)))


class TransformerBlock(nn.Module):
    """
    One attention or feedforward block with ScaleNorm and scaled residual.

    In x-transformers, each "layer" in the list is a tuple of:
      (norm_tuple, block, residual_fn)
    where norm_tuple = (pre_norm, post_branch_norm, post_main_norm)

    Flow: residual = x; x = pre_norm(x); x = block(x); x = residual_fn(x, residual)
    residual_fn with scale_residual: x + residual * residual_scale
    """

    def __init__(self, dim: int, heads: int, block_type: str):
        super().__init__()
        self.block_type = block_type
        self.norm = ScaleNorm(dim)

        if block_type == "attention":
            self.block = Attention(dim, heads)
        else:
            self.block = FeedForward(dim, FF_MULT)

        # scale_residual parameter
        self.residual_scale = mx.ones((dim,))

    def __call__(self, x, rotary_pos_emb=None):
        residual = x

        # Pre-norm
        x = self.norm(x)

        # Block
        if self.block_type == "attention":
            x = self.block(x, rotary_pos_emb=rotary_pos_emb)
        else:
            x = self.block(x)

        # Scaled residual connection
        x = x + residual * self.residual_scale

        return x


class BrainEncoder(nn.Module):
    """
    TRIBE v2 Brain Encoder in MLX.

    8-depth Transformer with alternating attention and feedforward layers.
    Total 16 layers: even indices = attention, odd indices = feedforward.
    """

    def __init__(self):
        super().__init__()

        # Rotary position embedding: dim=head_dim//2=72, inv_freq has shape [36]
        self.rotary_pos_emb = RotaryEmbedding(HEAD_DIM // 2)  # 72 -> inv_freq[36]

        # 16 layers alternating attention and feedforward
        self.layers = []
        for i in range(DEPTH * 2):  # 0..15
            if i % 2 == 0:
                self.layers.append(TransformerBlock(DIM, HEADS, "attention"))
            else:
                self.layers.append(TransformerBlock(DIM, HEADS, "feedforward"))

        # Final norm
        self.final_norm = ScaleNorm(DIM)

    def __call__(self, x):
        """
        x: [batch, seq_len, dim]
        Returns: [batch, seq_len, dim]
        """
        seq_len = x.shape[1]

        # Compute rotary embeddings
        rotary_emb = self.rotary_pos_emb(seq_len)

        # Pass through all layers
        for layer in self.layers:
            x = layer(x, rotary_pos_emb=rotary_emb)

        # Final normalization
        x = self.final_norm(x)
        return x


# ==============================================================================
# Part 3: Weight conversion from PyTorch to MLX
# ==============================================================================

def convert_weights(encoder_weights: dict, verbose=True) -> dict:
    """
    Convert PyTorch encoder weights to MLX format.

    Key mapping:
      PyTorch layers.{i}.0.0.g           -> layers.{i}.norm.g
      PyTorch layers.{i}.1.to_{q,k,v,out}.weight -> layers.{i}.block.to_{q,k,v,out}.weight
      PyTorch layers.{i}.1.ff.0.0.{weight,bias}  -> layers.{i}.block.w1.{weight,bias}
      PyTorch layers.{i}.1.ff.2.{weight,bias}    -> layers.{i}.block.w2.{weight,bias}
      PyTorch layers.{i}.2.residual_scale -> layers.{i}.residual_scale

    Both PyTorch and MLX nn.Linear store weight as [out, in]. No transpose needed.
    """
    mlx_weights = {}

    for layer_idx in range(DEPTH * 2):  # 0..15
        prefix = f"layers.{layer_idx}"

        # ScaleNorm g
        key_g = f"{prefix}.0.0.g"
        if key_g in encoder_weights:
            mlx_weights[f"layers.{layer_idx}.norm.g"] = mx.array(encoder_weights[key_g])

        # Residual scale
        key_rs = f"{prefix}.2.residual_scale"
        if key_rs in encoder_weights:
            mlx_weights[f"layers.{layer_idx}.residual_scale"] = mx.array(encoder_weights[key_rs])

        if layer_idx % 2 == 0:
            # Attention layer
            for proj in ["to_q", "to_k", "to_v", "to_out"]:
                key_w = f"{prefix}.1.{proj}.weight"
                if key_w in encoder_weights:
                    mlx_weights[f"layers.{layer_idx}.block.{proj}.weight"] = mx.array(
                        encoder_weights[key_w]
                    )
        else:
            # FeedForward layer
            # ff.0.0 = Linear(dim, inner) -> w1
            key_w1_w = f"{prefix}.1.ff.0.0.weight"
            key_w1_b = f"{prefix}.1.ff.0.0.bias"
            key_w2_w = f"{prefix}.1.ff.2.weight"
            key_w2_b = f"{prefix}.1.ff.2.bias"

            if key_w1_w in encoder_weights:
                mlx_weights[f"layers.{layer_idx}.block.w1.weight"] = mx.array(
                    encoder_weights[key_w1_w]
                )
            if key_w1_b in encoder_weights:
                mlx_weights[f"layers.{layer_idx}.block.w1.bias"] = mx.array(
                    encoder_weights[key_w1_b]
                )
            if key_w2_w in encoder_weights:
                mlx_weights[f"layers.{layer_idx}.block.w2.weight"] = mx.array(
                    encoder_weights[key_w2_w]
                )
            if key_w2_b in encoder_weights:
                mlx_weights[f"layers.{layer_idx}.block.w2.bias"] = mx.array(
                    encoder_weights[key_w2_b]
                )

    # Final norm
    if "final_norm.g" in encoder_weights:
        mlx_weights["final_norm.g"] = mx.array(encoder_weights["final_norm.g"])

    # Rotary embedding inv_freq
    if "rotary_pos_emb.inv_freq" in encoder_weights:
        mlx_weights["rotary_pos_emb.inv_freq"] = mx.array(
            encoder_weights["rotary_pos_emb.inv_freq"]
        )

    if verbose:
        print(f"\nConverted {len(mlx_weights)} weight tensors to MLX")
    return mlx_weights


def load_mlx_model(mlx_weights: dict) -> BrainEncoder:
    """Create MLX model and load converted weights."""
    model = BrainEncoder()
    model.load_weights(list(mlx_weights.items()))
    mx.eval(model.parameters())
    return model


# ==============================================================================
# Part 4: Validation - compare PyTorch vs MLX outputs
# ==============================================================================

def validate_outputs():
    """Run both PyTorch and MLX models on the same input, compare outputs."""
    import torch

    print("\n" + "=" * 60)
    print("VALIDATION: Comparing PyTorch vs MLX outputs")
    print("=" * 60)

    # Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # Build PyTorch model using x-transformers Encoder (non-causal by default)
    from x_transformers import Encoder as XTEncoder

    pt_encoder = XTEncoder(
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        cross_attend=False,
        attn_flash=False,
        attn_dropout=0.0,
        attn_dim_head=HEAD_DIM,  # grouped via prefix 'attn_' -> dim_head=144
        ff_mult=FF_MULT,
        ff_dropout=0.0,
        use_scalenorm=True,
        use_rmsnorm=False,
        rotary_pos_emb=True,
        rotary_xpos=False,
        residual_attn=False,
        scale_residual=True,
        layer_dropout=0.0,
    )
    pt_encoder.eval()

    # Load encoder weights into PyTorch model
    encoder_state = {}
    for k, v in state_dict.items():
        if k.startswith("model.encoder."):
            encoder_state[k[len("model.encoder."):]] = v

    missing, unexpected = pt_encoder.load_state_dict(encoder_state, strict=True)
    if missing:
        print(f"  WARNING: Missing keys: {missing}")
    if unexpected:
        print(f"  WARNING: Unexpected keys: {unexpected}")
    print("  PyTorch encoder weights loaded successfully")

    # Load MLX model
    encoder_weights_np = {k: v.numpy().astype(np.float32) for k, v in encoder_state.items()}
    mlx_weights = convert_weights(encoder_weights_np)
    mlx_model = load_mlx_model(mlx_weights)

    # Create test input
    np.random.seed(42)
    test_input_np = np.random.randn(1, 32, DIM).astype(np.float32)

    # PyTorch forward
    with torch.no_grad():
        pt_input = torch.from_numpy(test_input_np)
        pt_output = pt_encoder(pt_input)
        pt_output_np = pt_output.numpy()

    # MLX forward
    mlx_input = mx.array(test_input_np)
    mlx_output = mlx_model(mlx_input)
    mx.eval(mlx_output)
    mlx_output_np = np.array(mlx_output)

    # Compare
    abs_diff = np.abs(pt_output_np - mlx_output_np)
    rel_diff = abs_diff / (np.abs(pt_output_np) + 1e-8)

    print(f"\n  Input shape:      {test_input_np.shape}")
    print(f"  PT output shape:  {pt_output_np.shape}")
    print(f"  MLX output shape: {mlx_output_np.shape}")
    print(f"  PT output range:  [{pt_output_np.min():.6f}, {pt_output_np.max():.6f}]")
    print(f"  MLX output range: [{mlx_output_np.min():.6f}, {mlx_output_np.max():.6f}]")
    print(f"  Max abs diff:     {abs_diff.max():.8f}")
    print(f"  Mean abs diff:    {abs_diff.mean():.8f}")
    print(f"  Max rel diff:     {rel_diff.max():.8f}")
    print(f"  Mean rel diff:    {rel_diff.mean():.8f}")

    # Cosine similarity
    pt_flat = pt_output_np.flatten()
    mlx_flat = mlx_output_np.flatten()
    cos_sim = np.dot(pt_flat, mlx_flat) / (np.linalg.norm(pt_flat) * np.linalg.norm(mlx_flat))
    print(f"  Cosine similarity: {cos_sim:.8f}")

    if cos_sim > 0.9999:
        print("\n  PASS: Outputs match within tolerance (cosine sim > 0.9999)")
    elif cos_sim > 0.999:
        print("\n  ACCEPTABLE: Outputs close (cosine sim > 0.999)")
    else:
        print(f"\n  WARNING: Outputs differ significantly (cosine sim = {cos_sim:.6f})")

    return cos_sim


# ==============================================================================
# Part 5: Save as MLX safetensors
# ==============================================================================

def save_mlx_safetensors(mlx_weights: dict):
    """Save MLX weights in safetensors format."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mx.save_safetensors(OUTPUT_PATH, mlx_weights)

    file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\nSaved MLX weights to: {OUTPUT_PATH}")
    print(f"File size: {file_size:.1f} MB")


# ==============================================================================
# Part 6: Benchmark MLX inference speed
# ==============================================================================

def benchmark_mlx():
    """Benchmark MLX inference speed."""
    print("\n" + "=" * 60)
    print("BENCHMARK: MLX Inference Speed")
    print("=" * 60)

    # Load model
    encoder_weights = load_pytorch_weights(verbose=False)
    mlx_weights = convert_weights(encoder_weights, verbose=False)
    model = load_mlx_model(mlx_weights)

    # Test various sequence lengths
    batch_size = 1
    seq_lengths = [32, 64, 100, 128, 256]

    for seq_len in seq_lengths:
        x = mx.random.normal((batch_size, seq_len, DIM))
        mx.eval(x)

        # Warmup
        for _ in range(3):
            out = model(x)
            mx.eval(out)

        # Benchmark
        n_runs = 20
        start = time.perf_counter()
        for _ in range(n_runs):
            out = model(x)
            mx.eval(out)
        elapsed = time.perf_counter() - start

        ms_per_run = (elapsed / n_runs) * 1000
        print(f"  seq_len={seq_len:>4d}: {ms_per_run:.2f} ms/forward  "
              f"({n_runs / elapsed:.1f} forward/s)")


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 60)
    print("TRIBE v2 Brain Encoder: PyTorch -> MLX Conversion")
    print("=" * 60)

    # Step 1: Load and inspect PyTorch weights
    encoder_weights = load_pytorch_weights()

    # Step 2: Convert to MLX format
    mlx_weights = convert_weights(encoder_weights)

    # Step 3: Verify model can be loaded
    print("\nLoading MLX model...")
    model = load_mlx_model(mlx_weights)
    print("MLX model loaded successfully")

    # Quick sanity check
    test_x = mx.random.normal((1, 32, DIM))
    out = model(test_x)
    mx.eval(out)
    print(f"Sanity check - output shape: {out.shape}, "
          f"range: [{np.array(out).min():.4f}, {np.array(out).max():.4f}]")

    # Step 4: Save as safetensors
    save_mlx_safetensors(mlx_weights)

    # Step 5: Validate against PyTorch
    cos_sim = validate_outputs()

    # Step 6: Benchmark
    benchmark_mlx()

    # Step 7: Verify we can reload from safetensors
    print("\n" + "=" * 60)
    print("RELOAD TEST: Loading from safetensors")
    print("=" * 60)
    reloaded = dict(mx.load(OUTPUT_PATH))
    model2 = BrainEncoder()
    model2.load_weights(list(reloaded.items()))
    mx.eval(model2.parameters())

    test_x = mx.random.normal((1, 32, DIM))
    mx.eval(test_x)
    out1 = model(test_x)
    out2 = model2(test_x)
    mx.eval(out1, out2)
    reload_diff = np.abs(np.array(out1) - np.array(out2)).max()
    print(f"  Max diff after reload: {reload_diff:.10f}")
    if reload_diff < 1e-6:
        print("  PASS: Reload produces identical results")
    else:
        print("  WARNING: Reload results differ")

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Encoder: {DEPTH} attention + {DEPTH} FF layers, dim={DIM}, heads={HEADS}")
    print(f"  Validation cosine similarity: {cos_sim:.8f}")


if __name__ == "__main__":
    main()
