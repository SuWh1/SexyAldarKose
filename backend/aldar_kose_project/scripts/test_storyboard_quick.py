#!/usr/bin/env python3
"""
Quick Storyboard Test Script
Tests the simplified storyboard generator with checkpoint-400
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from simple_storyboard import SimplifiedStoryboardGenerator
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_storyboard():
    """Run a quick storyboard test"""
    
    # Configuration
    LORA_PATH = "outputs/checkpoints/checkpoint-400"  # Your trained model
    PROMPTS_FILE = "test_story_prompts.json"
    OUTPUT_DIR = "outputs/test_storyboard"
    BASE_SEED = 42
    
    logger.info("=" * 80)
    logger.info("🎬 ALDAR KOSE STORYBOARD TEST")
    logger.info("=" * 80)
    logger.info(f"LoRA: {LORA_PATH}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("")
    
    # Load prompts
    with open(PROMPTS_FILE, 'r') as f:
        data = json.load(f)
        prompts = data["prompts"]
        scene_name = data.get("scene_name", "Test Scene")
    
    logger.info(f"Scene: {scene_name}")
    logger.info(f"Frames: {len(prompts)}")
    logger.info("")
    
    # Initialize generator
    logger.info("Initializing pipeline...")
    generator = SimplifiedStoryboardGenerator(
        lora_path=LORA_PATH,
    )
    
    # Generate sequence
    logger.info("")
    logger.info("Generating frames...")
    frames = generator.generate_sequence(
        prompts=prompts,
        base_seed=BASE_SEED,
        num_inference_steps=40,  # Slightly lower for faster testing
        guidance_scale=7.5,
        consistency_threshold=0.65,  # Accept if character similarity > 0.65
        max_retries=3,  # Try up to 3 times per frame
        output_dir=OUTPUT_DIR,
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"✅ SUCCESS! Generated {len(frames)} frames")
    logger.info(f"📁 Check: {OUTPUT_DIR}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Review the frames visually")
    logger.info("2. Check report.json for consistency scores")
    logger.info("3. If consistent, try story_scene_prompts.json")
    logger.info("")


if __name__ == "__main__":
    test_storyboard()
