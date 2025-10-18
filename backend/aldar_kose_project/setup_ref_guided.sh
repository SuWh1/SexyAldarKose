#!/bin/bash
# Setup script for Reference-Guided Storyboard Generation
# Run this on your RunPod instance or local machine

set -e  # Exit on error

echo "=========================================="
echo "Reference-Guided Storyboard Setup"
echo "=========================================="
echo ""

# Check if running on GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found. GPU may not be available."
    echo "   Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Step 1: Installing ControlNet preprocessors..."
pip install controlnet-aux>=0.0.7

echo ""
echo "Step 2: Installing face detection dependencies..."
pip install insightface>=0.7.3
pip install onnxruntime-gpu>=1.16.0  # GPU version for faster processing

echo ""
echo "Step 3: Cloning IP-Adapter repository..."
if [ -d "IP-Adapter" ]; then
    echo "  IP-Adapter directory already exists. Pulling latest..."
    cd IP-Adapter
    git pull
    cd ..
else
    git clone https://github.com/tencent-ailab/IP-Adapter.git
fi

echo ""
echo "Step 4: Installing IP-Adapter..."
cd IP-Adapter
pip install -e .
cd ..

echo ""
echo "Step 5: Creating models directory..."
mkdir -p models/ip-adapter

echo ""
echo "Step 6: Downloading IP-Adapter SDXL checkpoint..."
if [ -f "models/ip-adapter/ip-adapter_sdxl.bin" ]; then
    echo "  Checkpoint already exists. Skipping download."
else
    echo "  Downloading from Hugging Face..."
    cd models/ip-adapter
    wget https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.bin
    cd ../..
    echo "  ✓ Downloaded successfully!"
fi

echo ""
echo "Step 7: Downloading image encoder (CLIP)..."
if [ -d "models/ip-adapter/image_encoder" ]; then
    echo "  Image encoder already exists. Skipping download."
else
    cd models/ip-adapter
    git clone https://huggingface.co/h94/IP-Adapter image_encoder
    cd ../..
    echo "  ✓ Downloaded successfully!"
fi

echo ""
echo "=========================================="
echo "✓ Installation Complete!"
echo "=========================================="
echo ""
echo "VRAM Requirements:"
echo "  - Simple mode: 8-10GB"
echo "  - Reference-guided mode: 16-20GB"
echo ""
echo "Test the installation:"
echo "  python scripts/ref_guided_storyboard.py --help"
echo ""
echo "Generate a storyboard with reference guidance:"
echo "  python scripts/prompt_storyboard.py \\"
echo "    --lora-path outputs/checkpoints/checkpoint-400 \\"
echo "    --story 'Your story here' \\"
echo "    --use-ref-guided"
echo ""
echo "=========================================="
