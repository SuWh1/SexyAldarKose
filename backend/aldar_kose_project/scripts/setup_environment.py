#!/usr/bin/env python3
"""
Environment Setup Script for Aldar Kose SDXL Fine-tuning Project

This script checks your system for GPU availability, CUDA support,
and verifies that all required packages are installed.

Usage:
    python scripts/setup_environment.py
"""

import sys
import os
import subprocess
from pathlib import Path


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def check_python_version():
    """Check if Python version is 3.10 or higher"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ ERROR: Python 3.10 or higher is required!")
        print("   Please upgrade your Python installation.")
        return False
    
    print("✅ Python version is compatible")
    return True


def check_gpu_availability():
    """Check for CUDA-capable GPU"""
    print_header("Checking GPU & CUDA Availability")
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {props.name}")
                print(f"  - Compute Capability: {props.major}.{props.minor}")
                print(f"  - Total Memory: {props.total_memory / 1024**3:.2f} GB")
                
                # Check VRAM
                if props.total_memory / 1024**3 < 8:
                    print(f"  ⚠️  WARNING: Less than 8GB VRAM may cause issues")
                else:
                    print(f"  ✅ Sufficient VRAM for training")
            
            return True
        else:
            print("❌ ERROR: No CUDA-capable GPU detected!")
            print("   SDXL fine-tuning requires a CUDA GPU.")
            print("   Please check your GPU drivers and CUDA installation.")
            return False
            
    except ImportError:
        print("❌ ERROR: PyTorch is not installed!")
        print("   Please install PyTorch with CUDA support.")
        print("\n   Install command:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False


def check_required_packages():
    """Check if all required packages are installed"""
    print_header("Checking Required Packages")
    
    required_packages = {
        'torch': 'torch',
        'transformers': 'transformers',
        'diffusers': 'diffusers',
        'accelerate': 'accelerate',
        'peft': 'peft',
        'bitsandbytes': 'bitsandbytes',
        'wandb': 'wandb',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'safetensors': 'safetensors',
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - NOT INSTALLED")
            missing_packages.append(package_name)
    
    if missing_packages:
        print("\n⚠️  Missing packages detected!")
        print("   Install them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("\n   Or install all requirements:")
        print("   pip install -r requirements.txt")
        return False
    
    print("\n✅ All required packages are installed")
    return True


def check_xformers():
    """Check if xformers is available for memory optimization"""
    print_header("Checking Optional Optimizations")
    
    try:
        import xformers
        print(f"✅ xformers is installed (version: {xformers.__version__})")
        print("   Memory-efficient attention will be available during training")
        return True
    except ImportError:
        print("⚠️  xformers is not installed (optional but recommended)")
        print("   Install with: pip install xformers")
        print("   This can significantly reduce VRAM usage")
        return False


def check_directory_structure():
    """Verify that all required directories exist"""
    print_header("Checking Directory Structure")
    
    project_root = Path(__file__).parent.parent
    required_dirs = [
        'data/images',
        'data/captions',
        'data/processed_images',
        'configs',
        'scripts',
        'logs/wandb',
        'outputs/checkpoints',
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Some directories are missing!")
        print("   They should have been created during setup.")
    else:
        print("\n✅ All required directories exist")
    
    return all_exist


def print_accelerate_config():
    """Print recommended accelerate configuration"""
    print_header("Accelerate Configuration")
    
    print("For optimal training, configure accelerate with:")
    print("\n  accelerate config")
    print("\nRecommended settings for RTX 4060 (8GB VRAM):")
    print("  - Compute environment: This machine")
    print("  - Machine type: No distributed training")
    print("  - Use DeepSpeed: No")
    print("  - GPU(s): 1")
    print("  - Mixed precision: fp16")
    print("  - Gradient accumulation steps: 4")
    
    print("\n" + "-" * 70)
    print("Launch training with:")
    print("  accelerate launch scripts/train_lora_sdxl.py")
    print("-" * 70)


def check_wandb_login():
    """Check if user is logged into Weights & Biases"""
    print_header("Weights & Biases Setup")
    
    try:
        import wandb
        
        # Check if API key is configured
        api_key = wandb.api.api_key
        if api_key:
            print("✅ WandB is configured")
            print(f"   API key found (ending in: ...{api_key[-4:]})")
        else:
            print("⚠️  WandB API key not found")
            print("   Login with: wandb login")
            print("   Or set environment variable: WANDB_API_KEY=<your-key>")
            print("   Training will work without WandB, but logging will be disabled")
            
    except ImportError:
        print("⚠️  WandB not installed")
        print("   Install with: pip install wandb")


def main():
    """Main setup check function"""
    print("\n" + "🚀" * 35)
    print("  ALDAR KOSE SDXL FINE-TUNING - Environment Setup")
    print("🚀" * 35)
    
    checks = [
        ("Python Version", check_python_version),
        ("GPU & CUDA", check_gpu_availability),
        ("Required Packages", check_required_packages),
        ("Directory Structure", check_directory_structure),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error during {name} check: {str(e)}")
            results.append((name, False))
    
    # Optional checks
    check_xformers()
    check_wandb_login()
    
    # Summary
    print_header("Setup Summary")
    all_passed = all(result for _, result in results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    if all_passed:
        print("\n" + "=" * 70)
        print("🎉 Environment setup is complete!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Add your training images to: data/images/")
        print("  2. Add corresponding captions to: data/captions/")
        print("  3. Run: python scripts/prepare_dataset.py")
        print("  4. Configure accelerate: accelerate config")
        print("  5. Start training: accelerate launch scripts/train_lora_sdxl.py")
        print_accelerate_config()
        return 0
    else:
        print("\n" + "=" * 70)
        print("⚠️  Setup incomplete - please fix the issues above")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
