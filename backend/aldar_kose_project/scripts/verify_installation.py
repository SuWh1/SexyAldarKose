#!/usr/bin/env python3
"""
Installation Verification Script
Run this after installing dependencies to verify everything is set up correctly.

Usage:
    python scripts/verify_installation.py
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Verify Python version is compatible."""
    print("=" * 70)
    print("🐍 PYTHON VERSION")
    print("=" * 70)
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor in [10, 11]:
        print("✅ Python version compatible (3.10 or 3.11)")
        return True
    elif version.major == 3 and version.minor >= 8:
        print("⚠️  Python 3.10 or 3.11 recommended (you have 3.{})".format(version.minor))
        return True
    else:
        print("❌ Python 3.10 or 3.11 required (you have {}.{})".format(version.major, version.minor))
        return False

def check_cuda():
    """Check CUDA availability."""
    print("\n" + "=" * 70)
    print("🎮 GPU/CUDA CHECK")
    print("=" * 70)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            
            # Check VRAM
            gpu_props = torch.cuda.get_device_properties(0)
            total_memory_gb = gpu_props.total_memory / (1024**3)
            print(f"✅ GPU Memory: {total_memory_gb:.1f} GB")
            
            if total_memory_gb >= 18:
                print("✅ Sufficient VRAM for inference (18GB+ required)")
                return True
            else:
                print("⚠️  GPU has less than 18GB VRAM (may cause OOM errors)")
                print("   Consider using a cloud GPU or reducing batch size")
                return True
        else:
            print("⚠️  CUDA not available - will use CPU (VERY SLOW!)")
            print("   Install CUDA-enabled PyTorch:")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_dependencies():
    """Verify all required packages."""
    print("\n" + "=" * 70)
    print("📦 DEPENDENCIES CHECK")
    print("=" * 70)
    
    required_packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('diffusers', 'diffusers'),
        ('peft', 'peft'),
        ('PIL', 'Pillow'),
        ('openai', 'openai'),
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('tqdm', 'tqdm'),
        ('yaml', 'PyYAML'),
        ('dotenv', 'python-dotenv'),
    ]
    
    all_installed = True
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - MISSING")
            all_installed = False
    
    # Check optional packages
    print("\nOptional packages (recommended):")
    optional_packages = [
        ('controlnet_aux', 'controlnet-aux'),
        ('onnxruntime', 'onnxruntime'),
    ]
    
    for import_name, package_name in optional_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"⚠️  {package_name} - not installed (ControlNet may not work)")
    
    if not all_installed:
        print("\n❌ Some required packages are missing!")
        print("   Install with: pip install -r requirements_inference.txt")
        return False
    
    print("\n✅ All required dependencies installed!")
    return True

def check_openai_key():
    """Check if OpenAI API key is configured."""
    print("\n" + "=" * 70)
    print("🔑 OPENAI API KEY")
    print("=" * 70)
    
    # Check .env file
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        print(f"✅ .env file found at: {env_file}")
        with open(env_file) as f:
            content = f.read()
            if "OPENAI_API_KEY" in content:
                print("✅ OPENAI_API_KEY found in .env file")
                return True
    
    # Check environment variable
    if os.environ.get('OPENAI_API_KEY'):
        key = os.environ.get('OPENAI_API_KEY')
        masked = key[:7] + "..." + key[-4:] if len(key) > 11 else "***"
        print(f"✅ OPENAI_API_KEY found in environment: {masked}")
        return True
    
    print("⚠️  OPENAI_API_KEY not found")
    print("   Set it with:")
    print("   1. Create .env file with: OPENAI_API_KEY=sk-your-key")
    print("   2. Or set environment variable:")
    print("      export OPENAI_API_KEY=sk-your-key  (Linux/Mac)")
    print("      $env:OPENAI_API_KEY='sk-your-key'  (Windows PowerShell)")
    return False

def check_disk_space():
    """Check available disk space."""
    print("\n" + "=" * 70)
    print("💾 DISK SPACE")
    print("=" * 70)
    
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        print(f"Free disk space: {free_gb:.1f} GB")
        
        if free_gb >= 30:
            print("✅ Sufficient disk space (30GB+ required)")
            return True
        else:
            print("⚠️  Low disk space (less than 30GB)")
            print("   Models require ~6GB, outputs ~500MB per story")
            return False
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True

def main():
    """Run all verification checks."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "Aldar Kose Installation Verification" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    results = {
        "Python Version": check_python_version(),
        "CUDA/GPU": check_cuda(),
        "Dependencies": check_dependencies(),
        "OpenAI API Key": check_openai_key(),
        "Disk Space": check_disk_space(),
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check:20s} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n✅ ALL CHECKS PASSED!")
        print("   You're ready to run the demo:")
        print("   python scripts/submission_demo.py")
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED")
        print("   Fix the issues above before running the demo")
        return 1

if __name__ == "__main__":
    sys.exit(main())
