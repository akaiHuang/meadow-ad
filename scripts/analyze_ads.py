#!/usr/bin/env python3
"""
TRIBE v2 Ad Copy Brain Analysis Pipeline

Analyzes advertising copy by predicting brain activation patterns
using Meta's TRIBE v2 model converted to Apple MLX.

Pipeline: Text → LLaMA 3.2 → Text Projector → Brain Encoder → 20,484 cortical voxels

Usage:
    python analyze_ads.py --ads ads.json --output results/
    python analyze_ads.py --text "Your ad copy here"
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx


def load_tribe_v2_weights(ckpt_path: str):
    """Load TRIBE v2 projection and prediction weights."""
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt['state_dict']
    return {
        'text_proj_w': state['model.projectors.text.weight'].numpy(),
        'text_proj_b': state['model.projectors.text.bias'].numpy(),
        'low_rank_w': state['model.low_rank_head.weight'].numpy(),
        'predictor_w': state['model.predictor.weights'].numpy()[0],
        'predictor_b': state['model.predictor.bias'].numpy(),
    }


def extract_text_features(text: str, model, tokenizer, target_layers: list) -> np.ndarray:
    """Extract hidden states from LLaMA at specified layers."""
    tokens = mx.array(tokenizer.encode(text)).reshape(1, -1)
    h = model.model.embed_tokens(tokens)
    outs = []
    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask=None, cache=None)
        if isinstance(h, tuple):
            h = h[0]
        if i + 1 in target_layers:
            outs.append(np.array(h.mean(axis=1)))
    return np.concatenate(outs, axis=-1)


def predict_brain_response(text_feat: np.ndarray, weights: dict) -> np.ndarray:
    """Project text features through TRIBE v2 to predict 20,484 cortical voxels."""
    proj = text_feat @ weights['text_proj_w'].T + weights['text_proj_b']
    padded = np.zeros((1, 1152))
    padded[:, :384] = proj
    lr = padded @ weights['low_rank_w'].T
    brain = (lr @ weights['predictor_w']) + weights['predictor_b']
    return brain.flatten()


def score_ad(brain_map: np.ndarray, baseline: np.ndarray) -> dict:
    """Score an ad copy based on its brain activation vs baseline."""
    diff = brain_map - baseline
    pos_activation = float(np.sum(diff[diff > 0]))
    total_activation = float(np.sum(np.abs(diff)))
    max_activation = float(np.max(diff))
    uniqueness = float(np.std(diff))
    score = pos_activation * 0.4 + total_activation * 0.3 + max_activation * 100 * 0.2 + uniqueness * 500 * 0.1
    return {
        'score': round(score),
        'pos_activation': round(pos_activation),
        'total_activation': round(total_activation),
        'max_activation': round(max_activation, 3),
        'uniqueness': round(uniqueness, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="TRIBE v2 Ad Copy Brain Analysis")
    parser.add_argument("--ads", type=str, help="JSON file with ad copy list")
    parser.add_argument("--text", type=str, nargs="+", help="Ad copy text(s) to analyze")
    parser.add_argument("--llama-path", type=str, default="models/llama3.2-3b-4bit",
                        help="Path to MLX LLaMA model")
    parser.add_argument("--ckpt-path", type=str, default="models/tribev2_ckpt/best.ckpt",
                        help="Path to TRIBE v2 checkpoint")
    parser.add_argument("--output", type=str, default="data/", help="Output directory")
    args = parser.parse_args()

    # Collect ad texts
    ads = []
    if args.ads:
        with open(args.ads) as f:
            ads = json.load(f)
    if args.text:
        ads.extend(args.text)
    if not ads:
        print("Error: provide --ads or --text")
        sys.exit(1)

    print(f"Analyzing {len(ads)} ad copies...\n")

    # Load models
    print("Loading LLaMA 3.2 3B (MLX)...")
    from mlx_lm import load
    model, tokenizer = load(args.llama_path)

    print("Loading TRIBE v2 weights...")
    weights = load_tribe_v2_weights(args.ckpt_path)

    n_layers = len(model.model.layers)
    target_layers = [int(n_layers * 0.5), int(n_layers * 1.0)]

    # Process all ads
    print("\nExtracting brain responses...")
    brain_maps = {}
    for text in ads:
        t0 = time.perf_counter()
        feat = extract_text_features(text, model, tokenizer, target_layers)
        brain = predict_brain_response(feat, weights)
        elapsed = time.perf_counter() - t0
        brain_maps[text] = brain
        print(f"  [{elapsed:.2f}s] {text[:40]}")

    # Score relative to baseline
    baseline = np.mean(list(brain_maps.values()), axis=0)
    results = []
    for text in ads:
        s = score_ad(brain_maps[text], baseline)
        s['text'] = text
        results.append(s)

    results.sort(key=lambda x: -x['score'])
    max_score = results[0]['score'] if results else 1
    for r in results:
        r['score_pct'] = round(r['score'] / max_score * 100)

    # Output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save brain maps
    np.savez(output_dir / "brain_maps.npz",
             labels=np.array(ads),
             **{f"map_{i}": brain_maps[t] for i, t in enumerate(ads)})

    # Save scores
    with open(output_dir / "scores.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print results
    print(f"\n{'Rank':>4}  {'Score':>5}  Ad Copy")
    print("-" * 60)
    for i, r in enumerate(results):
        bar = "#" * (r['score_pct'] // 5)
        print(f"{i+1:4d}  {r['score_pct']:4d}%  {r['text']}")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
