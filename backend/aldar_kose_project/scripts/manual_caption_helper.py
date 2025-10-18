#!/usr/bin/env python3
"""
Manual Caption Helper

Opens images one by one and lets you write captions manually.
Much faster than editing files individually.

Usage:
    python scripts/manual_caption_helper.py --input_dir raw_images/
"""

import argparse
import os
from pathlib import Path
from PIL import Image

TRIGGER_TOKEN = "aldar_kose_man"

def main():
    parser = argparse.ArgumentParser(description="Manual caption helper")
    parser.add_argument("--input_dir", type=str, default="raw_images", help="Input directory with images")
    parser.add_argument("--output_images", type=str, default="data/images", help="Output directory for images")
    parser.add_argument("--output_captions", type=str, default="data/captions", help="Output directory for captions")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_images = Path(args.output_images)
    output_captions = Path(args.output_captions)
    
    # Create output directories
    output_images.mkdir(parents=True, exist_ok=True)
    output_captions.mkdir(parents=True, exist_ok=True)
    
    # Supported formats
    supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff'}
    
    # Get all images (avoiding duplicates on case-insensitive filesystems)
    image_files = []
    seen = set()
    for f in input_dir.iterdir():
        if f.is_file() and f.suffix.lower() in supported_formats:
            canonical = f.resolve()
            if canonical not in seen:
                seen.add(canonical)
                image_files.append(f)
    
    print(f"\n{'='*70}")
    print(f"  MANUAL CAPTION HELPER - Aldar Kose Dataset")
    print(f"{'='*70}\n")
    print(f"Found {len(image_files)} images to caption\n")
    print(f"Instructions:")
    print(f"  1. Image will be displayed (or shown in path)")
    print(f"  2. Enter a caption (trigger token '{TRIGGER_TOKEN}' will be added automatically)")
    print(f"  3. Press Enter to skip if already captioned")
    print(f"  4. Type 'quit' to exit\n")
    print(f"{'='*70}\n")
    
    captioned = 0
    skipped = 0
    
    for i, image_path in enumerate(sorted(image_files), 1):
        # Check if already captioned
        caption_file = output_captions / f"{image_path.stem}.txt"
        existing_caption = ""
        
        if caption_file.exists():
            existing_caption = caption_file.read_text().strip()
            # Skip if it's a good caption (not a refusal)
            if existing_caption and "can't" not in existing_caption.lower() and "sorry" not in existing_caption.lower():
                print(f"[{i}/{len(image_files)}] ✓ {image_path.name} - Already captioned")
                skipped += 1
                continue
        
        print(f"\n{'='*70}")
        print(f"[{i}/{len(image_files)}] 📸 {image_path.name}")
        print(f"{'='*70}")
        print(f"Path: {image_path}")
        
        if existing_caption:
            print(f"Current caption: {existing_caption}")
        
        # Try to open and show image info
        try:
            with Image.open(image_path) as img:
                print(f"Size: {img.size[0]}x{img.size[1]} | Mode: {img.mode}")
                # Try to display (works in some terminals)
                try:
                    img.show()
                except:
                    print("(Could not display image - please open manually)")
        except Exception as e:
            print(f"Error loading image: {e}")
        
        # Get caption
        print(f"\nEnter caption description (without '{TRIGGER_TOKEN}'):")
        print(f"Examples:")
        print(f"  - smiling, traditional blue coat, outdoor scene")
        print(f"  - riding a horse, detailed 3D render")
        print(f"  - friendly expression, ornate clothing, studio lighting")
        print(f"\nYour caption: ", end="")
        
        user_input = input().strip()
        
        if user_input.lower() == 'quit':
            print("\nExiting...")
            break
        
        if not user_input:
            print("Skipped")
            skipped += 1
            continue
        
        # Build full caption
        if TRIGGER_TOKEN.lower() not in user_input.lower():
            full_caption = f"{TRIGGER_TOKEN}, {user_input}"
        else:
            full_caption = user_input
        
        # Save caption
        caption_file.write_text(full_caption)
        
        # Copy image to output
        import shutil
        output_image = output_images / image_path.name
        shutil.copy2(image_path, output_image)
        
        print(f"✓ Saved: {full_caption}")
        captioned += 1
    
    print(f"\n{'='*70}")
    print(f"  CAPTIONING COMPLETE")
    print(f"{'='*70}")
    print(f"\nSummary:")
    print(f"  ✓ Captioned: {captioned}")
    print(f"  ⏭  Skipped: {skipped}")
    print(f"  📁 Total images: {len(image_files)}")
    print(f"\nNext steps:")
    print(f"  1. Review captions in: {output_captions}")
    print(f"  2. Run: python scripts/prepare_dataset.py")
    print(f"  3. Start training!")
    print()

if __name__ == "__main__":
    main()
