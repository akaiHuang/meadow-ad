#!/bin/bash
set -e
echo "=== Meadow AD Setup ==="

# Install dependencies
echo "Installing Python dependencies..."
pip install mlx mlx-lm torch numpy nilearn safetensors

# Download models
echo ""
echo "Downloading models from Hugging Face..."
echo "[1/2] TRIBE v2 checkpoint (0.7 GB)..."
huggingface-cli download facebook/tribev2 --local-dir models/tribev2_ckpt

echo "[2/2] LLaMA 3.2 3B MLX 4-bit (1.7 GB)..."
huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit --local-dir models/llama3.2-3b-4bit

echo ""
echo "✅ Setup complete!"
echo ""
echo "Quick start:"
echo "  python scripts/analyze_ads.py --text 'Your ad copy here'"
echo "  cd demo && python -m http.server 8899"
