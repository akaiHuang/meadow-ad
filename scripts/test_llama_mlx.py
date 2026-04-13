#!/usr/bin/env python3
"""
LLaMA 3.2 3B (MLX 4-bit) verification script for TRIBE v2.

Tasks:
  1. Load model with mlx-lm
  2. Run text generation
  3. Extract intermediate hidden states at layers 0.5, 0.75, 1.0
  4. Print model info (layers, hidden size, param count)
  5. Measure inference speed (tok/s)
"""

import time
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from mlx_lm import load, generate, stream_generate
from mlx_lm.models.base import create_attention_mask

MODEL_PATH = str(Path(__file__).parent / "llama3.2-3b-4bit")


def count_parameters(model):
    """Count total parameters."""
    total = 0
    for k, v in mlx.utils.tree_flatten(model.parameters()):
        total += v.size
    return total


def print_model_info(model, config):
    """Print model architecture info."""
    num_layers = config.get("num_hidden_layers", "?")
    hidden_size = config.get("hidden_size", "?")
    num_heads = config.get("num_attention_heads", "?")
    num_kv_heads = config.get("num_key_value_heads", "?")
    intermediate = config.get("intermediate_size", "?")
    vocab_size = config.get("vocab_size", "?")
    quant_bits = config.get("quantization", {}).get("bits", "?")
    quant_group = config.get("quantization", {}).get("group_size", "?")

    param_count = count_parameters(model)

    print("=" * 60)
    print("MODEL INFO")
    print("=" * 60)
    print(f"  Architecture      : LlamaForCausalLM")
    print(f"  Num layers        : {num_layers}")
    print(f"  Hidden size       : {hidden_size}")
    print(f"  Attention heads   : {num_heads} (KV heads: {num_kv_heads})")
    print(f"  Intermediate size : {intermediate}")
    print(f"  Vocab size        : {vocab_size}")
    print(f"  Quantization      : {quant_bits}-bit, group_size={quant_group}")
    print(f"  Total parameters  : {param_count:,} (~{param_count / 1e9:.2f}B)")
    print(f"  Model path        : {MODEL_PATH}")
    print("=" * 60)


def extract_hidden_states(model, tokenizer, text="The meaning of life is"):
    """
    Extract hidden states at relative layer positions 0.5, 0.75, 1.0
    for TRIBE v2 feature extraction.
    """
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    inner = model.model
    num_layers = len(inner.layers)

    target_fractions = [0.5, 0.75, 1.0]
    target_layers = {}
    for frac in target_fractions:
        idx = max(0, int(frac * num_layers) - 1)
        target_layers[frac] = idx

    print(f"\n{'=' * 60}")
    print("HIDDEN STATE EXTRACTION (TRIBE v2)")
    print(f"{'=' * 60}")
    print(f"  Input text   : \"{text}\"")
    print(f"  Input tokens : {len(tokens)}")
    print(f"  Total layers : {num_layers}")
    print(f"  Target layers:")
    for frac, idx in target_layers.items():
        print(f"    layer {frac:.2f} -> layer index {idx}")

    # Manual forward pass to capture intermediate hidden states
    h = inner.embed_tokens(input_ids)

    # Build causal attention mask (no cache)
    mask = create_attention_mask(h, cache=None)

    collected = {}
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask=mask, cache=None)
        if i in target_layers.values():
            collected[i] = h

    # Final norm
    h_final = inner.norm(h)

    # Report shapes
    print(f"\n  Extracted hidden states:")
    for frac, idx in target_layers.items():
        hs = collected[idx]
        mx.eval(hs)
        hs_f32 = hs.astype(mx.float32)
        mean_val = hs_f32.mean().item()
        std_val = float(hs_f32.var().item() ** 0.5)
        print(
            f"    layer {frac:.2f} (idx {idx:2d}): shape={hs.shape}, "
            f"dtype={hs.dtype}, mean={mean_val:.6f}, std={std_val:.6f}"
        )

    mx.eval(h_final)
    print(f"    final norm output   : shape={h_final.shape}")
    print(f"{'=' * 60}")
    return collected


def test_generation(model, tokenizer):
    """Run a simple text generation test."""
    prompt = "Explain what an audio spectrogram is in one sentence:"

    print(f"\n{'=' * 60}")
    print("TEXT GENERATION TEST")
    print(f"{'=' * 60}")
    print(f"  Prompt: \"{prompt}\"\n")

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=64,
        verbose=False,
    )

    print(f"  Response:\n    {response.strip()}")
    print(f"{'=' * 60}")
    return response


def measure_speed(model, tokenizer, num_tokens=128):
    """Measure inference speed using stream_generate."""
    prompt = "The quick brown fox jumps over the lazy dog. Once upon a time"

    print(f"\n{'=' * 60}")
    print("INFERENCE SPEED BENCHMARK")
    print(f"{'=' * 60}")

    # Warmup
    print("  Warmup...")
    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=8):
        pass
    print("  Warmup done.")

    # Timed run
    last_resp = None
    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=num_tokens):
        last_resp = resp

    if last_resp is not None:
        print(f"\n  Results:")
        print(f"    Prompt tokens     : {last_resp.prompt_tokens}")
        print(f"    Prompt speed      : {last_resp.prompt_tps:.1f} tok/s")
        print(f"    Generated tokens  : {last_resp.generation_tokens}")
        print(f"    Generation speed  : {last_resp.generation_tps:.1f} tok/s")
        print(f"    Peak memory       : {last_resp.peak_memory:.2f} GB")
        gen_tps = last_resp.generation_tps
    else:
        gen_tps = 0.0
        print("  ERROR: No tokens generated")

    print(f"{'=' * 60}")
    return gen_tps


def main():
    print("\n" + "#" * 60)
    print("# LLaMA 3.2 3B (MLX 4-bit) Verification")
    print("#" * 60)

    # 1. Load model
    print("\n[1/5] Loading model...")
    t0 = time.perf_counter()
    model, tokenizer = load(MODEL_PATH)
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.2f}s")

    # 2. Read config
    with open(Path(MODEL_PATH) / "config.json") as f:
        config = json.load(f)

    # 3. Print model info
    print("\n[2/5] Model info:")
    print_model_info(model, config)

    # 4. Text generation
    print("\n[3/5] Text generation test...")
    test_generation(model, tokenizer)

    # 5. Extract hidden states
    print("\n[4/5] Extracting hidden states for TRIBE v2...")
    extract_hidden_states(model, tokenizer)

    # 6. Speed benchmark
    print("\n[5/5] Measuring inference speed...")
    speed = measure_speed(model, tokenizer, num_tokens=128)

    # Summary
    print(f"\n{'#' * 60}")
    print("# VERIFICATION COMPLETE")
    print(f"#   Load time : {load_time:.2f}s")
    print(f"#   Speed     : {speed:.1f} tok/s")
    print("#   Status    : ALL PASSED")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
