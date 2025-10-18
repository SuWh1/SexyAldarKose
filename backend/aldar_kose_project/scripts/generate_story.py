#!/usr/bin/env python3
"""
Simple Terminal Story Generator
Quick CLI for generating Aldar Kose stories with a single prompt.

Usage:
    python scripts/generate_story.py "Your story prompt here"
    
    # With seed for deterministic output
    python scripts/generate_story.py "Aldar tricks a merchant" --seed 42
    
    # With all options
    python scripts/generate_story.py "Aldar on horseback" --seed 42 --temp 0.0 --ref-guided --frames 8
"""

# Suppress library warnings and verbose logging
import warnings
import os
import sys  # FIXED: Import sys before using it

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress TensorFlow/MediaPipe verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error only
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prompt_storyboard import PromptStoryboardGenerator
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Generate Aldar Kose story from a text prompt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple generation with defaults
  python scripts/generate_story.py "Aldar Kose tricks a wealthy merchant"
  
  # Fully deterministic output (same result every time)
  python scripts/generate_story.py "Aldar riding across steppe" --seed 42 --temp 0.0
  
  # Reference-guided mode for better face consistency
  python scripts/generate_story.py "Aldar at the bazaar" --ref-guided --seed 123
  
  # Custom output location
  python scripts/generate_story.py "Aldar adventures" --output my_story --seed 42
        """
    )
    
    parser.add_argument(
        'prompt',
        type=str,
        help='Story prompt (e.g., "Aldar Kose tricks a merchant and steals his horse")'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42). Same seed + same prompt = same output'
    )
    
    parser.add_argument(
        '--temp',
        '--temperature',
        type=float,
        default=0.7,
        dest='temperature',
        help='GPT temperature 0.0-1.0 (default: 0.7). Use 0.0 for fully deterministic scene breakdown'
    )
    
    parser.add_argument(
        '--frames',
        type=int,
        default=None,
        help='Number of frames to generate (6-10, default: auto - GPT decides)'
    )
    
    parser.add_argument(
        '--ref-guided',
        action='store_true',
        dest='ref_guided',
        help='Use reference-guided mode (better face consistency, slower)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory name (default: outputs/terminal_generation_<timestamp>)'
    )
    
    parser.add_argument(
        '--lora-path',
        type=str,
        default='outputs/checkpoints/checkpoint-1000',
        help='Path to LoRA checkpoint (default: outputs/checkpoints/checkpoint-1000)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.prompt.strip():
        logger.error("❌ Error: Prompt cannot be empty")
        sys.exit(1)
    
    if args.temperature < 0.0 or args.temperature > 1.0:
        logger.error("❌ Error: Temperature must be between 0.0 and 1.0")
        sys.exit(1)
    
    if args.frames is not None and (args.frames < 6 or args.frames > 10):
        logger.error("❌ Error: Frames must be between 6 and 10")
        sys.exit(1)
    
    # Print configuration
    logger.info("=" * 60)
    logger.info("🎬 Aldar Kose Story Generator")
    logger.info("=" * 60)
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Mode: {'Reference-guided' if args.ref_guided else 'Simple'}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Temperature: {args.temperature} ({'deterministic' if args.temperature == 0.0 else 'creative' if args.temperature >= 0.9 else 'balanced'})")
    logger.info(f"Frames: {args.frames if args.frames else 'auto (GPT decides)'}")
    logger.info(f"LoRA: {args.lora_path}")
    if args.output:
        logger.info(f"Output: outputs/{args.output}")
    logger.info("=" * 60)
    
    # Check for deterministic mode
    if args.seed is not None and args.temperature == 0.0:
        logger.info("✅ FULLY DETERMINISTIC MODE")
        logger.info("   Same prompt + same seed = identical output")
        logger.info("-" * 60)
    
    try:
        # Initialize generator
        logger.info("🔧 Initializing generator...")
        generator = PromptStoryboardGenerator(
            openai_api_key=os.environ.get('OPENAI_API_KEY'),
            lora_path=args.lora_path,
            use_ref_guided=args.ref_guided
        )
        
        logger.info("✅ Generator ready!")
        logger.info("")
        logger.info("🎨 Generating story... (this will take 4-5 minutes)")
        logger.info("")
        
        # Generate output path
        if args.output:
            output_base = f"outputs/{args.output}"
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_base = f"outputs/terminal_generation_{timestamp}"
        
        # Generate storyboard
        generator.generate_storyboard(
            story=args.prompt,
            num_frames=args.frames,
            output_dir=output_base,
            base_seed=args.seed,
            temperature=args.temperature
        )
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ Story generation complete!")
        logger.info(f"📁 Frames saved to: {output_base}/")
        logger.info("=" * 60)
        
        # Show deterministic info again
        if args.seed is not None and args.temperature == 0.0:
            logger.info("")
            logger.info("🔒 To regenerate the exact same story, use:")
            logger.info(f'   python scripts/generate_story.py "{args.prompt}" --seed {args.seed} --temp 0.0{" --ref-guided" if args.ref_guided else ""}')
        
    except FileNotFoundError as e:
        logger.error(f"❌ Error: LoRA checkpoint not found at {args.lora_path}")
        logger.error(f"   Make sure the model is downloaded and the path is correct")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
