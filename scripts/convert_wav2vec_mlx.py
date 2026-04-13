#!/usr/bin/env python3
"""
Wav2Vec-BERT 2.0 PyTorch -> MLX conversion and inference.

Tasks:
1. Load PyTorch model, verify it works
2. Convert weights to MLX safetensors format
3. Implement forward pass in MLX
4. Validate PyTorch vs MLX output consistency
5. Benchmark MLX inference speed

Architecture: Conformer encoder (Conv + Attention)
- Input: mel-spectrogram features (80 mel bins, stride 2 -> 160-dim)
- 24 Conformer layers, hidden_size=1024, 16 heads
- position_embeddings_type = "relative_key"
- Each layer: FFN1 -> SelfAttn -> ConvModule -> FFN2
"""

import json
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# ============================================================
# Paths
# ============================================================
MODEL_DIR = "/Users/akaihuangm1/Desktop/2025/Aura/Aura/ml/tribev2/w2v-bert-2.0"
MLX_WEIGHTS_PATH = os.path.join(MODEL_DIR, "mlx_model.safetensors")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# ============================================================
# Config dataclass
# ============================================================
class W2VBertConfig:
    def __init__(self, d: dict):
        self.hidden_size = d.get("hidden_size", 1024)
        self.num_attention_heads = d.get("num_attention_heads", 16)
        self.intermediate_size = d.get("intermediate_size", 4096)
        self.num_hidden_layers = d.get("num_hidden_layers", 24)
        self.hidden_act = d.get("hidden_act", "swish")
        self.hidden_dropout = d.get("hidden_dropout", 0.0)
        self.attention_dropout = d.get("attention_dropout", 0.0)
        self.activation_dropout = d.get("activation_dropout", 0.0)
        self.feat_proj_dropout = d.get("feat_proj_dropout", 0.0)
        self.conformer_conv_dropout = d.get("conformer_conv_dropout", 0.0)
        self.layer_norm_eps = d.get("layer_norm_eps", 1e-5)
        self.feature_projection_input_dim = d.get("feature_projection_input_dim", 160)
        self.conv_depthwise_kernel_size = d.get("conv_depthwise_kernel_size", 31)
        self.position_embeddings_type = d.get("position_embeddings_type", "relative_key")
        self.left_max_position_embeddings = d.get("left_max_position_embeddings", 64)
        self.right_max_position_embeddings = d.get("right_max_position_embeddings", 8)
        self.head_size = self.hidden_size // self.num_attention_heads  # 64


# ============================================================
# MLX Modules
# ============================================================

def swish(x):
    return x * mx.sigmoid(x)


class MLXLayerNorm(nn.Module):
    def __init__(self, dims, eps=1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.bias = mx.zeros((dims,))
        self.eps = eps
        self.dims = dims

    def __call__(self, x):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        return self.weight * x + self.bias


class MLXFeatureProjection(nn.Module):
    def __init__(self, config: W2VBertConfig):
        super().__init__()
        self.layer_norm = MLXLayerNorm(config.feature_projection_input_dim, eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.feature_projection_input_dim, config.hidden_size)

    def __call__(self, hidden_states):
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        return hidden_states


class MLXFeedForward(nn.Module):
    def __init__(self, config: W2VBertConfig):
        super().__init__()
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = swish(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        return hidden_states


class MLXConvolutionModule(nn.Module):
    """Conformer convolution block.

    PyTorch uses channels-first Conv1d: input (B, C, T), weight (C_out, C_in/groups, K).
    MLX uses channels-last Conv1d: input (B, T, C), weight (C_out, K, C_in/groups).

    We implement the forward pass in MLX's channels-last convention.
    """
    def __init__(self, config: W2VBertConfig):
        super().__init__()
        H = config.hidden_size
        K = config.conv_depthwise_kernel_size

        self.layer_norm = MLXLayerNorm(H, eps=config.layer_norm_eps)
        # pointwise_conv1: Conv1d(H, 2H, kernel_size=1, bias=False)
        # MLX weight shape: (2H, 1, H)
        self.pointwise_conv1_weight = mx.zeros((2 * H, 1, H))
        # depthwise_conv: Conv1d(H, H, K, groups=H, bias=False)
        # MLX weight shape: (H, K, 1)  (groups=H means in_channels//groups=1)
        self.depthwise_conv_weight = mx.zeros((H, K, 1))
        self.depthwise_layer_norm = MLXLayerNorm(H, eps=config.layer_norm_eps)
        # pointwise_conv2: Conv1d(H, H, kernel_size=1, bias=False)
        # MLX weight shape: (H, 1, H)
        self.pointwise_conv2_weight = mx.zeros((H, 1, H))

        self.kernel_size = K
        self.hidden_size = H

    def __call__(self, hidden_states):
        # hidden_states: (B, T, H) - already channels-last in MLX
        hidden_states = self.layer_norm(hidden_states)

        # Pointwise conv1: (B, T, H) -> (B, T, 2H)
        hidden_states = mx.conv1d(hidden_states, self.pointwise_conv1_weight, stride=1, padding=0)

        # GLU: split along last dim, sigmoid gate
        x1, x2 = mx.split(hidden_states, 2, axis=-1)
        hidden_states = x1 * mx.sigmoid(x2)

        # Causal padding: pad only on left
        pad_len = self.kernel_size - 1
        hidden_states = mx.pad(hidden_states, [(0, 0), (pad_len, 0), (0, 0)])

        # Depthwise conv: groups=H
        hidden_states = mx.conv1d(hidden_states, self.depthwise_conv_weight,
                                   stride=1, padding=0, groups=self.hidden_size)

        hidden_states = self.depthwise_layer_norm(hidden_states)
        hidden_states = swish(hidden_states)

        # Pointwise conv2: (B, T, H) -> (B, T, H)
        hidden_states = mx.conv1d(hidden_states, self.pointwise_conv2_weight, stride=1, padding=0)

        return hidden_states


class MLXSelfAttention(nn.Module):
    """Self-attention with relative_key position embeddings."""
    def __init__(self, config: W2VBertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_size = config.head_size
        self.hidden_size = config.hidden_size

        self.linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_out = nn.Linear(config.hidden_size, config.hidden_size)

        # relative_key embeddings
        self.left_max = config.left_max_position_embeddings
        self.right_max = config.right_max_position_embeddings
        num_positions = self.left_max + self.right_max + 1  # 73
        self.distance_embedding_weight = mx.zeros((num_positions, self.head_size))

    def __call__(self, hidden_states):
        B, T, _ = hidden_states.shape

        query = self.linear_q(hidden_states).reshape(B, T, self.num_heads, self.head_size).transpose(0, 2, 1, 3)
        key = self.linear_k(hidden_states).reshape(B, T, self.num_heads, self.head_size).transpose(0, 2, 1, 3)
        value = self.linear_v(hidden_states).reshape(B, T, self.num_heads, self.head_size).transpose(0, 2, 1, 3)

        # (B, heads, T, T)
        scores = (query @ key.transpose(0, 1, 3, 2)) / math.sqrt(self.head_size)

        # Relative key position bias
        position_ids_l = mx.arange(T).reshape(-1, 1)  # (T, 1)
        position_ids_r = mx.arange(T).reshape(1, -1)  # (1, T)
        distance = position_ids_r - position_ids_l  # (T, T)
        distance = mx.clip(distance, -self.left_max, self.right_max)
        distance = distance + self.left_max  # shift to [0, left+right]

        positional_embedding = self.distance_embedding_weight[distance]  # (T, T, head_size)

        # einsum("bhld,lrd->bhlr", query, positional_embedding)
        # query: (B, heads, T, head_size), pos_emb: (T, T, head_size)
        # We do: for each batch, head: query[b,h] @ pos_emb^T
        # query: (B, H, T, D), pos_emb: (T, T, D)
        # result[b,h,l,r] = sum_d query[b,h,l,d] * pos_emb[l,r,d]
        # = (query @ pos_emb.transpose(0,2,1)) but pos_emb varies per l
        # Efficient: einsum via reshape
        # pos_emb: (T, T, D) -> expand to (1, 1, T, T, D) * query (B, H, T, 1, D)
        # -> sum over D -> (B, H, T, T)
        rel_attn = mx.sum(
            query[:, :, :, None, :] * positional_embedding[None, None, :, :, :],
            axis=-1
        )
        scores = scores + rel_attn / math.sqrt(self.head_size)

        probs = mx.softmax(scores, axis=-1)

        hidden_states = (probs @ value).transpose(0, 2, 1, 3).reshape(B, T, self.hidden_size)
        hidden_states = self.linear_out(hidden_states)
        return hidden_states


class MLXConformerLayer(nn.Module):
    def __init__(self, config: W2VBertConfig):
        super().__init__()
        H = config.hidden_size
        self.ffn1_layer_norm = MLXLayerNorm(H, eps=config.layer_norm_eps)
        self.ffn1 = MLXFeedForward(config)
        self.self_attn_layer_norm = MLXLayerNorm(H, eps=config.layer_norm_eps)
        self.self_attn = MLXSelfAttention(config)
        self.conv_module = MLXConvolutionModule(config)
        self.ffn2_layer_norm = MLXLayerNorm(H, eps=config.layer_norm_eps)
        self.ffn2 = MLXFeedForward(config)
        self.final_layer_norm = MLXLayerNorm(H, eps=config.layer_norm_eps)

    def __call__(self, hidden_states):
        # 1. FFN1
        residual = hidden_states
        hidden_states = self.ffn1_layer_norm(hidden_states)
        hidden_states = self.ffn1(hidden_states)
        hidden_states = hidden_states * 0.5 + residual

        # 2. Self-Attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = hidden_states + residual

        # 3. Convolution Module
        residual = hidden_states
        hidden_states = self.conv_module(hidden_states)
        hidden_states = residual + hidden_states

        # 4. FFN2
        residual = hidden_states
        hidden_states = self.ffn2_layer_norm(hidden_states)
        hidden_states = self.ffn2(hidden_states)
        hidden_states = hidden_states * 0.5 + residual
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class MLXWav2Vec2Bert(nn.Module):
    def __init__(self, config: W2VBertConfig):
        super().__init__()
        self.config = config
        self.feature_projection = MLXFeatureProjection(config)
        self.layers = [MLXConformerLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, input_features, output_hidden_states=False):
        """
        Args:
            input_features: (B, T, 160) mel-spectrogram features
            output_hidden_states: if True, return all layer hidden states
        Returns:
            last_hidden_state or (last_hidden_state, all_hidden_states)
        """
        hidden_states = self.feature_projection(input_features)

        all_hidden_states = []
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        if output_hidden_states:
            return hidden_states, all_hidden_states
        return hidden_states


# ============================================================
# Weight conversion: PyTorch safetensors -> MLX safetensors
# ============================================================

def convert_weights_to_mlx():
    """Convert PyTorch safetensors weights to MLX format.

    Key conversions:
    - Conv1d weights: PyTorch (C_out, C_in/groups, K) -> MLX (C_out, K, C_in/groups)
    - Linear weights: kept as-is (both use [out, in])
    - LayerNorm: kept as-is
    - distance_embedding: kept as-is
    """
    from safetensors import safe_open
    from safetensors.numpy import save_file as np_save_file

    print("Loading PyTorch weights from safetensors...")
    pt_weights = {}
    with safe_open(os.path.join(MODEL_DIR, "model.safetensors"), framework="numpy") as f:
        for key in f.keys():
            pt_weights[key] = f.get_tensor(key)

    print(f"  Loaded {len(pt_weights)} tensors")

    mlx_weights = {}

    # Feature projection
    mlx_weights["feature_projection.layer_norm.weight"] = pt_weights["feature_projection.layer_norm.weight"]
    mlx_weights["feature_projection.layer_norm.bias"] = pt_weights["feature_projection.layer_norm.bias"]
    mlx_weights["feature_projection.projection.weight"] = pt_weights["feature_projection.projection.weight"]
    mlx_weights["feature_projection.projection.bias"] = pt_weights["feature_projection.projection.bias"]

    # Encoder layers
    for i in range(24):
        prefix = f"encoder.layers.{i}"
        out_prefix = f"layers.{i}"

        # FFN1
        mlx_weights[f"{out_prefix}.ffn1.intermediate_dense.weight"] = pt_weights[f"{prefix}.ffn1.intermediate_dense.weight"]
        mlx_weights[f"{out_prefix}.ffn1.intermediate_dense.bias"] = pt_weights[f"{prefix}.ffn1.intermediate_dense.bias"]
        mlx_weights[f"{out_prefix}.ffn1.output_dense.weight"] = pt_weights[f"{prefix}.ffn1.output_dense.weight"]
        mlx_weights[f"{out_prefix}.ffn1.output_dense.bias"] = pt_weights[f"{prefix}.ffn1.output_dense.bias"]
        mlx_weights[f"{out_prefix}.ffn1_layer_norm.weight"] = pt_weights[f"{prefix}.ffn1_layer_norm.weight"]
        mlx_weights[f"{out_prefix}.ffn1_layer_norm.bias"] = pt_weights[f"{prefix}.ffn1_layer_norm.bias"]

        # Self-attention
        mlx_weights[f"{out_prefix}.self_attn.linear_q.weight"] = pt_weights[f"{prefix}.self_attn.linear_q.weight"]
        mlx_weights[f"{out_prefix}.self_attn.linear_q.bias"] = pt_weights[f"{prefix}.self_attn.linear_q.bias"]
        mlx_weights[f"{out_prefix}.self_attn.linear_k.weight"] = pt_weights[f"{prefix}.self_attn.linear_k.weight"]
        mlx_weights[f"{out_prefix}.self_attn.linear_k.bias"] = pt_weights[f"{prefix}.self_attn.linear_k.bias"]
        mlx_weights[f"{out_prefix}.self_attn.linear_v.weight"] = pt_weights[f"{prefix}.self_attn.linear_v.weight"]
        mlx_weights[f"{out_prefix}.self_attn.linear_v.bias"] = pt_weights[f"{prefix}.self_attn.linear_v.bias"]
        mlx_weights[f"{out_prefix}.self_attn.linear_out.weight"] = pt_weights[f"{prefix}.self_attn.linear_out.weight"]
        mlx_weights[f"{out_prefix}.self_attn.linear_out.bias"] = pt_weights[f"{prefix}.self_attn.linear_out.bias"]
        mlx_weights[f"{out_prefix}.self_attn.distance_embedding_weight"] = pt_weights[f"{prefix}.self_attn.distance_embedding.weight"]
        mlx_weights[f"{out_prefix}.self_attn_layer_norm.weight"] = pt_weights[f"{prefix}.self_attn_layer_norm.weight"]
        mlx_weights[f"{out_prefix}.self_attn_layer_norm.bias"] = pt_weights[f"{prefix}.self_attn_layer_norm.bias"]

        # Conv module - need to transpose conv weights!
        # PyTorch Conv1d weight: (C_out, C_in/groups, K) -> MLX: (C_out, K, C_in/groups)
        pw1 = pt_weights[f"{prefix}.conv_module.pointwise_conv1.weight"]  # (2H, H, 1)
        mlx_weights[f"{out_prefix}.conv_module.pointwise_conv1_weight"] = np.transpose(pw1, (0, 2, 1))  # (2H, 1, H)

        dw = pt_weights[f"{prefix}.conv_module.depthwise_conv.weight"]  # (H, 1, K)
        mlx_weights[f"{out_prefix}.conv_module.depthwise_conv_weight"] = np.transpose(dw, (0, 2, 1))  # (H, K, 1)

        pw2 = pt_weights[f"{prefix}.conv_module.pointwise_conv2.weight"]  # (H, H, 1)
        mlx_weights[f"{out_prefix}.conv_module.pointwise_conv2_weight"] = np.transpose(pw2, (0, 2, 1))  # (H, 1, H)

        mlx_weights[f"{out_prefix}.conv_module.layer_norm.weight"] = pt_weights[f"{prefix}.conv_module.layer_norm.weight"]
        mlx_weights[f"{out_prefix}.conv_module.layer_norm.bias"] = pt_weights[f"{prefix}.conv_module.layer_norm.bias"]
        mlx_weights[f"{out_prefix}.conv_module.depthwise_layer_norm.weight"] = pt_weights[f"{prefix}.conv_module.depthwise_layer_norm.weight"]
        mlx_weights[f"{out_prefix}.conv_module.depthwise_layer_norm.bias"] = pt_weights[f"{prefix}.conv_module.depthwise_layer_norm.bias"]

        # FFN2
        mlx_weights[f"{out_prefix}.ffn2.intermediate_dense.weight"] = pt_weights[f"{prefix}.ffn2.intermediate_dense.weight"]
        mlx_weights[f"{out_prefix}.ffn2.intermediate_dense.bias"] = pt_weights[f"{prefix}.ffn2.intermediate_dense.bias"]
        mlx_weights[f"{out_prefix}.ffn2.output_dense.weight"] = pt_weights[f"{prefix}.ffn2.output_dense.weight"]
        mlx_weights[f"{out_prefix}.ffn2.output_dense.bias"] = pt_weights[f"{prefix}.ffn2.output_dense.bias"]
        mlx_weights[f"{out_prefix}.ffn2_layer_norm.weight"] = pt_weights[f"{prefix}.ffn2_layer_norm.weight"]
        mlx_weights[f"{out_prefix}.ffn2_layer_norm.bias"] = pt_weights[f"{prefix}.ffn2_layer_norm.bias"]

        # Final layer norm
        mlx_weights[f"{out_prefix}.final_layer_norm.weight"] = pt_weights[f"{prefix}.final_layer_norm.weight"]
        mlx_weights[f"{out_prefix}.final_layer_norm.bias"] = pt_weights[f"{prefix}.final_layer_norm.bias"]

    print(f"  Converted {len(mlx_weights)} tensors for MLX")

    # Save
    print(f"  Saving to {MLX_WEIGHTS_PATH}...")
    np_save_file(mlx_weights, MLX_WEIGHTS_PATH)
    file_size_mb = os.path.getsize(MLX_WEIGHTS_PATH) / (1024 * 1024)
    print(f"  Saved! ({file_size_mb:.1f} MB)")

    return mlx_weights


def load_mlx_model(config: W2VBertConfig):
    """Build MLX model and load converted weights."""
    model = MLXWav2Vec2Bert(config)

    # Load weights
    from safetensors.numpy import load_file as np_load_file
    weights_np = np_load_file(MLX_WEIGHTS_PATH)

    # Convert numpy arrays to mx arrays and load into model
    weights_mx = {k: mx.array(v) for k, v in weights_np.items()}
    model.load_weights(list(weights_mx.items()))

    mx.eval(model.parameters())
    return model


# ============================================================
# Step 1: Verify PyTorch model
# ============================================================
def step1_verify_pytorch():
    print("=" * 60)
    print("STEP 1: Verify PyTorch Wav2Vec-BERT 2.0")
    print("=" * 60)

    import torch
    from transformers import Wav2Vec2BertModel, Wav2Vec2BertConfig

    print("Loading model from local directory...")
    t0 = time.time()
    model = Wav2Vec2BertModel.from_pretrained(MODEL_DIR)
    model.eval()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Create dummy input: batch=1, time=100 frames, 160-dim features
    # (SeamlessM4T feature extractor outputs 80 mel bins x stride 2 = 160 dim)
    np.random.seed(42)
    dummy_np = np.random.randn(1, 100, 160).astype(np.float32)
    dummy_input = torch.from_numpy(dummy_np)

    print("  Running forward pass (100 frames)...")
    with torch.no_grad():
        outputs = model(input_features=dummy_input, output_hidden_states=True)

    print(f"  last_hidden_state shape: {outputs.last_hidden_state.shape}")
    print(f"  Number of hidden states: {len(outputs.hidden_states)}")
    for i, hs in enumerate(outputs.hidden_states):
        print(f"    Layer {i}: {hs.shape}, mean={hs.mean().item():.6f}, std={hs.std().item():.6f}")
        if i > 3:
            print(f"    ... (skipping to last)")
            hs_last = outputs.hidden_states[-1]
            print(f"    Layer {len(outputs.hidden_states)-1}: {hs_last.shape}, mean={hs_last.mean().item():.6f}, std={hs_last.std().item():.6f}")
            break

    # Save reference outputs for validation
    ref_data = {
        "input": dummy_np,
        "last_hidden_state": outputs.last_hidden_state.numpy(),
    }
    # Save a few layer outputs for TRIBE v2 (layer 0.5=12, 0.75=18, 1.0=24)
    for layer_idx in [0, 12, 18, 24]:
        if layer_idx < len(outputs.hidden_states):
            ref_data[f"hidden_state_{layer_idx}"] = outputs.hidden_states[layer_idx].numpy()

    ref_path = os.path.join(MODEL_DIR, "pytorch_reference.npz")
    np.savez(ref_path, **ref_data)
    print(f"  Saved reference outputs to {ref_path}")

    return outputs


# ============================================================
# Step 2: Convert weights
# ============================================================
def step2_convert_weights():
    print()
    print("=" * 60)
    print("STEP 2: Convert weights to MLX safetensors")
    print("=" * 60)
    convert_weights_to_mlx()


# ============================================================
# Step 3: MLX forward pass
# ============================================================
def step3_mlx_inference(config: W2VBertConfig):
    print()
    print("=" * 60)
    print("STEP 3: MLX inference")
    print("=" * 60)

    print("Loading MLX model...")
    t0 = time.time()
    model = load_mlx_model(config)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Same dummy input
    np.random.seed(42)
    dummy_np = np.random.randn(1, 100, 160).astype(np.float32)
    dummy_input = mx.array(dummy_np)

    print("  Running forward pass (100 frames)...")
    t0 = time.time()
    last_hidden, all_hidden = model(dummy_input, output_hidden_states=True)
    mx.eval(last_hidden)
    print(f"  Forward pass took {time.time() - t0:.3f}s")

    print(f"  last_hidden_state shape: {last_hidden.shape}")
    print(f"  Number of hidden states: {len(all_hidden)} (includes initial projection)")

    return last_hidden, all_hidden


# ============================================================
# Step 4: Validate consistency
# ============================================================
def step4_validate(config: W2VBertConfig):
    print()
    print("=" * 60)
    print("STEP 4: Validate PyTorch vs MLX consistency")
    print("=" * 60)

    ref_path = os.path.join(MODEL_DIR, "pytorch_reference.npz")
    ref = np.load(ref_path)

    model = load_mlx_model(config)
    dummy_input = mx.array(ref["input"])

    last_hidden, all_hidden = model(dummy_input, output_hidden_states=True)
    mx.eval(last_hidden)
    for h in all_hidden:
        mx.eval(h)

    # Compare last hidden state
    pt_last = ref["last_hidden_state"]
    mlx_last = np.array(last_hidden)

    abs_diff = np.abs(pt_last - mlx_last)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    cos_sim = np.sum(pt_last * mlx_last) / (np.linalg.norm(pt_last) * np.linalg.norm(mlx_last) + 1e-8)

    print(f"  Last hidden state comparison:")
    print(f"    Max absolute diff:  {max_diff:.6e}")
    print(f"    Mean absolute diff: {mean_diff:.6e}")
    print(f"    Cosine similarity:  {cos_sim:.8f}")

    # Compare specific layers (TRIBE v2 layers)
    for layer_idx in [0, 12, 18]:
        key = f"hidden_state_{layer_idx}"
        if key in ref:
            pt_hs = ref[key]
            # all_hidden[0] = after feature projection (= layer 0 input)
            # all_hidden[i] = after encoder layer i-1
            # So hidden_state_0 from PT = all_hidden[0] from MLX
            # hidden_state_12 from PT = all_hidden[12]
            mlx_hs = np.array(all_hidden[layer_idx])
            diff = np.abs(pt_hs - mlx_hs)
            cs = np.sum(pt_hs * mlx_hs) / (np.linalg.norm(pt_hs) * np.linalg.norm(mlx_hs) + 1e-8)
            print(f"  Layer {layer_idx}: max_diff={diff.max():.6e}, mean_diff={diff.mean():.6e}, cos_sim={cs:.8f}")

    # Also compare final layer (24)
    key = "hidden_state_24"
    if key in ref:
        pt_hs = ref[key]
        mlx_hs = np.array(all_hidden[24])
        diff = np.abs(pt_hs - mlx_hs)
        cs = np.sum(pt_hs * mlx_hs) / (np.linalg.norm(pt_hs) * np.linalg.norm(mlx_hs) + 1e-8)
        print(f"  Layer 24: max_diff={diff.max():.6e}, mean_diff={diff.mean():.6e}, cos_sim={cs:.8f}")

    if cos_sim > 0.99:
        print("  [PASS] MLX output matches PyTorch (cosine sim > 0.99)")
    elif cos_sim > 0.95:
        print("  [WARN] MLX output close but not exact (cosine sim > 0.95)")
    else:
        print("  [FAIL] Significant divergence detected!")

    return cos_sim


# ============================================================
# Step 5: Benchmark MLX speed
# ============================================================
def step5_benchmark(config: W2VBertConfig):
    print()
    print("=" * 60)
    print("STEP 5: Benchmark MLX inference speed")
    print("=" * 60)

    model = load_mlx_model(config)

    # Test with different sequence lengths
    for num_frames in [50, 100, 200, 500]:
        dummy_input = mx.random.normal((1, num_frames, 160))
        mx.eval(dummy_input)

        # Warmup
        out = model(dummy_input)
        mx.eval(out)

        # Benchmark
        n_runs = 5
        times = []
        for _ in range(n_runs):
            t0 = time.time()
            out = model(dummy_input)
            mx.eval(out)
            times.append(time.time() - t0)

        avg_time = np.mean(times)
        std_time = np.std(times)
        # Each frame = 20ms of audio (stride=2, 10ms per mel frame, so 160-dim = 2 frames = 20ms)
        audio_duration_ms = num_frames * 20
        rtf = avg_time * 1000 / audio_duration_ms  # real-time factor

        print(f"  {num_frames} frames ({audio_duration_ms}ms audio): "
              f"{avg_time*1000:.1f} +/- {std_time*1000:.1f} ms  (RTF: {rtf:.2f}x)")

    # Also benchmark with output_hidden_states (needed for TRIBE v2)
    print()
    print("  With output_hidden_states=True (for TRIBE v2):")
    for num_frames in [100, 200, 500]:
        dummy_input = mx.random.normal((1, num_frames, 160))
        mx.eval(dummy_input)

        # Warmup
        out = model(dummy_input, output_hidden_states=True)
        mx.eval(out[0])

        n_runs = 5
        times = []
        for _ in range(n_runs):
            t0 = time.time()
            out = model(dummy_input, output_hidden_states=True)
            mx.eval(out[0])
            for h in out[1]:
                mx.eval(h)
            times.append(time.time() - t0)

        avg_time = np.mean(times)
        audio_duration_ms = num_frames * 20
        rtf = avg_time * 1000 / audio_duration_ms
        print(f"  {num_frames} frames ({audio_duration_ms}ms audio): "
              f"{avg_time*1000:.1f} ms  (RTF: {rtf:.2f}x)")


# ============================================================
# PyTorch benchmark for comparison
# ============================================================
def benchmark_pytorch():
    print()
    print("=" * 60)
    print("BONUS: PyTorch benchmark for comparison")
    print("=" * 60)

    import torch
    from transformers import Wav2Vec2BertModel

    model = Wav2Vec2BertModel.from_pretrained(MODEL_DIR)
    model.eval()

    for num_frames in [100, 200, 500]:
        dummy_input = torch.randn(1, num_frames, 160)
        # Warmup
        with torch.no_grad():
            _ = model(input_features=dummy_input, output_hidden_states=True)

        n_runs = 5
        times = []
        for _ in range(n_runs):
            t0 = time.time()
            with torch.no_grad():
                _ = model(input_features=dummy_input, output_hidden_states=True)
            times.append(time.time() - t0)

        avg_time = np.mean(times)
        audio_duration_ms = num_frames * 20
        rtf = avg_time * 1000 / audio_duration_ms
        print(f"  PyTorch {num_frames} frames ({audio_duration_ms}ms audio): "
              f"{avg_time*1000:.1f} ms  (RTF: {rtf:.2f}x)")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    with open(CONFIG_PATH) as f:
        config = W2VBertConfig(json.load(f))

    print(f"Model config: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, pos_type={config.position_embeddings_type}")
    print()

    # Step 1: Verify PyTorch
    step1_verify_pytorch()

    # Step 2: Convert weights
    step2_convert_weights()

    # Step 3: MLX inference
    step3_mlx_inference(config)

    # Step 4: Validate
    step4_validate(config)

    # Step 5: Benchmark
    step5_benchmark(config)

    # Bonus: PyTorch benchmark
    benchmark_pytorch()

    print()
    print("=" * 60)
    print("ALL DONE!")
    print("=" * 60)
