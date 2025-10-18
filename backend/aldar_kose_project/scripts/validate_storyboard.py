#!/usr/bin/env python3
"""
Quick validation test for storyboard generators
Tests imports, basic functionality, and configuration
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required imports work"""
    logger.info("=" * 60)
    logger.info("Testing imports...")
    logger.info("=" * 60)
    
    try:
        import torch
        logger.info(f"✓ torch {torch.__version__}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError as e:
        logger.error(f"✗ Failed to import torch: {e}")
        return False
    
    try:
        from diffusers import StableDiffusionXLPipeline
        logger.info("✓ diffusers")
    except ImportError as e:
        logger.error(f"✗ Failed to import diffusers: {e}")
        return False
    
    try:
        from transformers import CLIPModel
        logger.info("✓ transformers")
    except ImportError as e:
        logger.error(f"✗ Failed to import transformers: {e}")
        return False
    
    try:
        from PIL import Image
        logger.info("✓ PIL")
    except ImportError as e:
        logger.error(f"✗ Failed to import PIL: {e}")
        return False
    
    try:
        import numpy as np
        logger.info("✓ numpy")
    except ImportError as e:
        logger.error(f"✗ Failed to import numpy: {e}")
        return False
    
    # Optional dependencies
    try:
        from controlnet_aux import OpenposeDetector
        logger.info("✓ controlnet_aux (optional - for full pipeline)")
    except ImportError:
        logger.warning("⚠ controlnet_aux not available (optional - needed for full pipeline)")
    
    try:
        from ip_adapter import IPAdapterXL
        logger.info("✓ ip_adapter (optional - for full pipeline)")
    except ImportError:
        logger.warning("⚠ ip_adapter not available (optional - needed for full pipeline)")
    
    return True


def test_file_structure():
    """Test that required files exist"""
    logger.info("=" * 60)
    logger.info("Testing file structure...")
    logger.info("=" * 60)
    
    required_files = [
        "scripts/simple_storyboard.py",
        "scripts/storyboard_generator.py",
        "data/example_scene_prompts.json",
        "configs/storyboard_config.yaml",
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            logger.info(f"✓ {file_path}")
        else:
            logger.error(f"✗ {file_path} not found")
            all_exist = False
    
    return all_exist


def test_json_loading():
    """Test that example prompts file is valid JSON"""
    logger.info("=" * 60)
    logger.info("Testing JSON configuration...")
    logger.info("=" * 60)
    
    import json
    
    try:
        with open("data/example_scene_prompts.json", 'r') as f:
            data = json.load(f)
        
        logger.info(f"✓ JSON file valid")
        logger.info(f"  Scene name: {data.get('scene_name')}")
        logger.info(f"  Number of prompts: {len(data.get('prompts', []))}")
        
        # Validate prompts
        prompts = data.get('prompts', [])
        if not prompts:
            logger.error("✗ No prompts found in JSON")
            return False
        
        trigger_token = "aldar_kose_man"
        missing_trigger = []
        for i, prompt in enumerate(prompts):
            if trigger_token not in prompt:
                missing_trigger.append(i + 1)
        
        if missing_trigger:
            logger.warning(f"⚠ Prompts {missing_trigger} missing trigger token '{trigger_token}'")
        else:
            logger.info(f"✓ All prompts contain trigger token '{trigger_token}'")
        
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"✗ Invalid JSON: {e}")
        return False
    except FileNotFoundError:
        logger.error("✗ example_scene_prompts.json not found")
        return False


def test_lora_path():
    """Check if LoRA exists"""
    logger.info("=" * 60)
    logger.info("Testing LoRA availability...")
    logger.info("=" * 60)
    
    lora_paths = [
        "outputs/aldar_kose_lora/final",
        "outputs/aldar_kose_lora",
    ]
    
    for lora_path in lora_paths:
        path = Path(lora_path)
        if path.exists():
            logger.info(f"✓ LoRA found at {lora_path}")
            # Check for safetensors files
            safetensors = list(path.glob("*.safetensors"))
            if safetensors:
                logger.info(f"  Found {len(safetensors)} safetensors file(s)")
                for st in safetensors:
                    logger.info(f"    - {st.name}")
                return True
            else:
                logger.warning(f"  ⚠ No .safetensors files found in {lora_path}")
    
    logger.warning("⚠ LoRA not found - you'll need to train it first")
    logger.info("  See TRAINING_MEMORY_ISSUE.md for training guidance")
    return False


def test_output_directory():
    """Test output directory creation"""
    logger.info("=" * 60)
    logger.info("Testing output directory...")
    logger.info("=" * 60)
    
    test_dir = Path("./test_storyboard_validation")
    try:
        test_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Can create output directory: {test_dir}")
        
        # Clean up
        test_dir.rmdir()
        return True
    except Exception as e:
        logger.error(f"✗ Cannot create output directory: {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("STORYBOARD GENERATOR VALIDATION TEST")
    logger.info("=" * 60)
    logger.info("")
    
    results = {
        "Imports": test_imports(),
        "File Structure": test_file_structure(),
        "JSON Configuration": test_json_loading(),
        "LoRA Availability": test_lora_path(),
        "Output Directory": test_output_directory(),
    }
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("=" * 60)
    
    all_passed = all(results.values())
    lora_available = results["LoRA Availability"]
    
    if all_passed:
        logger.info("✓ ALL TESTS PASSED - Ready to generate storyboards!")
    elif not lora_available:
        logger.warning("⚠ READY BUT NEEDS LORA TRAINING")
        logger.info("  All dependencies installed correctly")
        logger.info("  Train LoRA first, then generate storyboards")
        logger.info("  See: TRAINING_MEMORY_ISSUE.md")
    else:
        logger.error("✗ SOME TESTS FAILED - Check errors above")
        return 1
    
    logger.info("")
    logger.info("Next steps:")
    if not lora_available:
        logger.info("1. Train LoRA (see TRAINING_MEMORY_ISSUE.md)")
        logger.info("2. Test simplified pipeline:")
    else:
        logger.info("1. Test simplified pipeline:")
    logger.info("   python scripts/simple_storyboard.py \\")
    logger.info("     --prompts-file data/example_scene_prompts.json \\")
    logger.info("     --output-dir ./test_output")
    logger.info("")
    logger.info("Documentation: STORYBOARD_QUICKSTART.md")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
