#!/usr/bin/env python3
"""
One-command installer for Aldar Kose Storyboard Generator
Run this to install all dependencies

Usage:
    python install_dependencies.py
    
    Or with options:
    python install_dependencies.py --cuda 12.1
    python install_dependencies.py --cpu
"""

import subprocess
import sys
import argparse

def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*70}")
    print(f"📦 {description}")
    print(f"{'='*70}")
    print(f"Running: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        return False
    
    print(f"✅ Complete: {description}")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Install Aldar Kose dependencies'
    )
    
    parser.add_argument(
        '--cuda',
        type=str,
        default='12.1',
        help='CUDA version (default: 12.1)'
    )
    
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Install CPU-only PyTorch'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎭 ALDAR KOSE - DEPENDENCY INSTALLER")
    print("="*70)
    
    # Step 1: Install PyTorch
    if args.cpu:
        pytorch_cmd = "pip install torch torchvision"
        pytorch_desc = "PyTorch (CPU only)"
    else:
        pytorch_cmd = f"pip install torch torchvision --index-url https://download.pytorch.org/whl/cu{args.cuda.replace('.', '')}"
        pytorch_desc = f"PyTorch (CUDA {args.cuda})"
    
    if not run_command(pytorch_cmd, pytorch_desc):
        sys.exit(1)
    
    # Step 2: Install inference dependencies
    if not run_command(
        "pip install -r requirements_inference.txt",
        "Inference dependencies (including ref-guided)"
    ):
        sys.exit(1)
    
    # Step 3: Verify installations
    print(f"\n{'='*70}")
    print("✅ VERIFICATION")
    print(f"{'='*70}\n")
    
    packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('diffusers', 'Diffusers'),
        ('peft', 'PEFT'),
        ('controlnet_aux', 'ControlNet-Aux'),
        ('insightface', 'InsightFace'),
        ('mediapipe', 'MediaPipe'),
        ('openai', 'OpenAI'),
        ('boto3', 'Boto3'),
    ]
    
    all_ok = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} (MISSING)")
            all_ok = False
    
    print(f"\n{'='*70}")
    
    if all_ok:
        print("🎉 ALL DEPENDENCIES INSTALLED!")
        print(f"{'='*70}\n")
        print("Next steps:")
        print("  1. Configure AWS: aws configure")
        print("  2. Set OpenAI key in .env file")
        print("  3. Run: python scripts/submission_demo.py\n")
    else:
        print("❌ SOME DEPENDENCIES MISSING")
        print(f"{'='*70}\n")
        sys.exit(1)

if __name__ == '__main__':
    main()
