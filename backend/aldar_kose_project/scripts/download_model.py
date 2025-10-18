#!/usr/bin/env python3
"""
Quick Start Script - Download and Setup Base Model

This script helps you download the SDXL base model and verify your setup
before starting training.

Usage:
    python scripts/download_model.py
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    from diffusers import StableDiffusionXLPipeline
    import torch
except ImportError:
    print("Error: Required packages not installed.")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)


def check_disk_space(path: str, required_gb: float = 20):
    """Check if enough disk space is available"""
    import shutil
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024**3)
    
    print(f"\nDisk Space Check:")
    print(f"  Free space: {free_gb:.2f} GB")
    print(f"  Required: {required_gb} GB")
    
    if free_gb < required_gb:
        print(f"  ⚠️  Warning: Low disk space!")
        response = input("\n  Continue anyway? (y/n): ")
        return response.lower() == 'y'
    
    print(f"  ✅ Sufficient disk space")
    return True


def download_sdxl_model(model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"):
    """Download SDXL base model from Hugging Face"""
    
    print("=" * 70)
    print("  SDXL Base Model Download")
    print("=" * 70)
    
    cache_dir = Path.home() / ".cache" / "huggingface"
    
    # Check disk space
    if not check_disk_space(str(cache_dir.parent), required_gb=20):
        print("\nAborting download.")
        return False
    
    print(f"\nDownloading: {model_id}")
    print(f"Cache directory: {cache_dir}")
    print("\nThis may take a while (model is ~13GB)...")
    print("You can pause with Ctrl+C and resume later.\n")
    
    try:
        # Download model
        snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            resume_download=True,
        )
        
        print("\n" + "=" * 70)
        print("✅ Model downloaded successfully!")
        print("=" * 70)
        return True
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted. You can resume by running this script again.")
        return False
    except Exception as e:
        print(f"\n❌ Error downloading model: {str(e)}")
        return False


def verify_model(model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"):
    """Verify that the model can be loaded"""
    
    print("\n" + "=" * 70)
    print("  Verifying Model")
    print("=" * 70)
    
    try:
        print("\nLoading model components...")
        
        # Try to load the pipeline (this will use cached model)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            variant="fp16" if torch.cuda.is_available() else None,
            use_safetensors=True,
        )
        
        print("✅ Model loaded successfully!")
        print(f"   UNet: {pipe.unet.__class__.__name__}")
        print(f"   VAE: {pipe.vae.__class__.__name__}")
        print(f"   Text Encoder 1: {pipe.text_encoder.__class__.__name__}")
        print(f"   Text Encoder 2: {pipe.text_encoder_2.__class__.__name__}")
        
        del pipe
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


def check_huggingface_login():
    """Check if user is logged into Hugging Face"""
    
    print("\n" + "=" * 70)
    print("  Hugging Face Authentication")
    print("=" * 70)
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        try:
            user_info = api.whoami()
            print(f"\n✅ Logged in as: {user_info['name']}")
            return True
        except Exception:
            print("\n⚠️  Not logged into Hugging Face")
            print("\nSome models may require authentication.")
            print("Login with: huggingface-cli login")
            print("\nYou can continue without login for public models.")
            return False
            
    except ImportError:
        print("\n⚠️  huggingface_hub not installed")
        return False


def main():
    print("\n" + "🚀" * 35)
    print("  SDXL Setup - Base Model Download")
    print("🚀" * 35)
    
    # Check HF login
    check_huggingface_login()
    
    # Download model
    print("\n" + "-" * 70)
    response = input("\nDownload SDXL base model? (y/n): ")
    
    if response.lower() != 'y':
        print("Skipping download.")
        print("\nNote: The model will be downloaded automatically during training")
        print("if not already cached.")
        return 0
    
    success = download_sdxl_model()
    
    if not success:
        return 1
    
    # Verify model
    print("\n" + "-" * 70)
    response = input("\nVerify model can be loaded? (y/n): ")
    
    if response.lower() == 'y':
        verify_model()
    
    print("\n" + "=" * 70)
    print("  Setup Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Prepare your dataset (see DATA_FORMAT.md)")
    print("  2. Run: python scripts/prepare_dataset.py")
    print("  3. Configure: configs/training_config.yaml")
    print("  4. Start training: accelerate launch scripts/train_lora_sdxl.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
