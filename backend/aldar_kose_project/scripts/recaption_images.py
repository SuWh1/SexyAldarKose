#!/usr/bin/env python3
"""
Re-caption existing images with updated prompt (no clothing descriptions)

This script re-generates captions for images that were already captioned,
using the new improved prompt that focuses on action/setting rather than clothing.

Usage:
    # Re-caption all existing images
    python scripts/recaption_images.py

    # Or with specific directories
    python scripts/recaption_images.py \
        --images data/images/ \
        --captions data/captions/
"""

import argparse
import sys
from pathlib import Path
from label_images import ImageLabeler
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Re-caption existing images")
    parser.add_argument(
        "--images",
        type=str,
        default="data/images",
        help="Directory containing images"
    )
    parser.add_argument(
        "--captions",
        type=str,
        default="data/captions",
        help="Directory containing caption files (will be overwritten)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or use OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually re-captioning"
    )
    
    args = parser.parse_args()
    
    images_dir = Path(args.images)
    captions_dir = Path(args.captions)
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        sys.exit(1)
    
    # Find all images with existing captions
    image_files = []
    supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff'}
    
    for ext in supported_formats:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))
    
    # Filter to only images that have existing captions
    images_to_recaption = []
    for img_path in image_files:
        caption_path = captions_dir / f"{img_path.stem}.txt"
        if caption_path.exists():
            images_to_recaption.append((img_path, caption_path))
    
    if not images_to_recaption:
        logger.warning("No images with existing captions found!")
        return
    
    logger.info(f"Found {len(images_to_recaption)} images to re-caption")
    
    if args.dry_run:
        logger.info("DRY RUN - Would re-caption:")
        for img_path, cap_path in images_to_recaption[:5]:
            logger.info(f"  {img_path.name}")
        if len(images_to_recaption) > 5:
            logger.info(f"  ... and {len(images_to_recaption) - 5} more")
        return
    
    # Initialize labeler
    try:
        labeler = ImageLabeler(
            api_key=args.api_key,
            model=args.model,
        )
    except Exception as e:
        logger.error(f"Failed to initialize labeler: {e}")
        sys.exit(1)
    
    # Re-caption images
    logger.info("Re-captioning images with improved prompts...")
    success_count = 0
    error_count = 0
    
    for img_path, caption_path in tqdm(images_to_recaption, desc="Re-captioning"):
        try:
            # Read old caption for comparison
            with open(caption_path, 'r', encoding='utf-8') as f:
                old_caption = f.read().strip()
            
            # Generate new caption
            new_caption = labeler.generate_caption(img_path)
            
            # Save new caption
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(new_caption)
            
            logger.info(f"\n{img_path.name}:")
            logger.info(f"  OLD: {old_caption}")
            logger.info(f"  NEW: {new_caption}")
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
            error_count += 1
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✓ Re-captioning complete!")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Errors:  {error_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
