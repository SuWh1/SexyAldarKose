#!/usr/bin/env python3
"""
Dataset Preparation Script for Aldar Kose SDXL Fine-tuning

This script validates and preprocesses the training dataset:
- Verifies image-caption pairs
- Resizes images to target resolution
- Creates a dataset manifest
- Performs basic quality checks

Usage:
    python scripts/prepare_dataset.py [--resize] [--resolution 1024]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import yaml

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    """Load training configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_image_files(data_dir: Path) -> List[Path]:
    """Get all image files from the data directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = []
    seen = set()
    
    for ext in image_extensions:
        for file in data_dir.glob(f"*{ext}"):
            canonical = file.resolve()
            if canonical not in seen:
                seen.add(canonical)
                image_files.append(file)
        for file in data_dir.glob(f"*{ext.upper()}"):
            canonical = file.resolve()
            if canonical not in seen:
                seen.add(canonical)
                image_files.append(file)
    
    return sorted(image_files)


def validate_dataset(data_dir: Path, captions_dir: Path) -> Tuple[List[Dict], List[str]]:
    """
    Validate that each image has a corresponding caption file
    
    Returns:
        Tuple of (valid pairs, error messages)
    """
    print("\n📋 Validating dataset...")
    
    image_files = get_image_files(data_dir)
    
    if not image_files:
        return [], [f"❌ No images found in {data_dir}"]
    
    print(f"Found {len(image_files)} images")
    
    valid_pairs = []
    errors = []
    
    for image_path in tqdm(image_files, desc="Checking image-caption pairs"):
        # Expected caption file path
        caption_path = captions_dir / f"{image_path.stem}.txt"
        
        if not caption_path.exists():
            errors.append(f"Missing caption for: {image_path.name}")
            continue
        
        # Read caption
        try:
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            
            if not caption:
                errors.append(f"Empty caption for: {image_path.name}")
                continue
            
            # Get image dimensions
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    mode = img.mode
                    
                    # Check if image can be loaded properly
                    if mode not in ['RGB', 'RGBA', 'L']:
                        errors.append(f"Unsupported image mode '{mode}' for: {image_path.name}")
                        continue
                    
                    valid_pairs.append({
                        'image_path': str(image_path),
                        'caption_path': str(caption_path),
                        'caption': caption,
                        'width': width,
                        'height': height,
                        'mode': mode,
                        'filename': image_path.name
                    })
                    
            except Exception as e:
                errors.append(f"Error reading image {image_path.name}: {str(e)}")
                
        except Exception as e:
            errors.append(f"Error reading caption {caption_path.name}: {str(e)}")
    
    return valid_pairs, errors


def analyze_dataset_statistics(valid_pairs: List[Dict]) -> Dict:
    """Analyze and print dataset statistics"""
    if not valid_pairs:
        return {}
    
    widths = [p['width'] for p in valid_pairs]
    heights = [p['height'] for p in valid_pairs]
    caption_lengths = [len(p['caption']) for p in valid_pairs]
    
    stats = {
        'total_samples': len(valid_pairs),
        'resolution_stats': {
            'min_width': min(widths),
            'max_width': max(widths),
            'avg_width': np.mean(widths),
            'min_height': min(heights),
            'max_height': max(heights),
            'avg_height': np.mean(heights),
        },
        'caption_stats': {
            'min_length': min(caption_lengths),
            'max_length': max(caption_lengths),
            'avg_length': np.mean(caption_lengths),
        }
    }
    
    print("\n📊 Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"\n  Resolution (width):")
    print(f"    Min: {stats['resolution_stats']['min_width']}px")
    print(f"    Max: {stats['resolution_stats']['max_width']}px")
    print(f"    Avg: {stats['resolution_stats']['avg_width']:.0f}px")
    print(f"\n  Resolution (height):")
    print(f"    Min: {stats['resolution_stats']['min_height']}px")
    print(f"    Max: {stats['resolution_stats']['max_height']}px")
    print(f"    Avg: {stats['resolution_stats']['avg_height']:.0f}px")
    print(f"\n  Caption length (characters):")
    print(f"    Min: {stats['caption_stats']['min_length']}")
    print(f"    Max: {stats['caption_stats']['max_length']}")
    print(f"    Avg: {stats['caption_stats']['avg_length']:.0f}")
    
    return stats


def resize_image(image: Image.Image, target_size: int, center_crop: bool = True) -> Image.Image:
    """
    Resize image to target resolution while maintaining aspect ratio
    
    Args:
        image: PIL Image
        target_size: Target resolution (will be used for both width and height)
        center_crop: Whether to center crop to square
    
    Returns:
        Resized PIL Image
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    if center_crop:
        # Center crop to square
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        image = image.crop((left, top, right, bottom))
        
        # Resize to target size
        image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    else:
        # Resize maintaining aspect ratio (longer side = target_size)
        width, height = image.size
        if width > height:
            new_width = target_size
            new_height = int(height * (target_size / width))
        else:
            new_height = target_size
            new_width = int(width * (target_size / height))
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image


def process_images(
    valid_pairs: List[Dict],
    output_dir: Path,
    target_resolution: int,
    center_crop: bool
) -> List[Dict]:
    """
    Resize and save processed images
    
    Returns:
        Updated list of valid pairs with processed image paths
    """
    print(f"\n🔧 Processing images (target resolution: {target_resolution}px)...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_pairs = []
    
    for pair in tqdm(valid_pairs, desc="Resizing images"):
        try:
            # Load image
            with Image.open(pair['image_path']) as img:
                # Resize
                processed_img = resize_image(img, target_resolution, center_crop)
                
                # Save processed image
                output_path = output_dir / Path(pair['filename']).with_suffix('.png').name
                processed_img.save(output_path, 'PNG', optimize=True)
                
                # Update pair info
                processed_pair = pair.copy()
                processed_pair['processed_path'] = str(output_path)
                processed_pair['processed_width'] = target_resolution
                processed_pair['processed_height'] = target_resolution
                processed_pairs.append(processed_pair)
                
        except Exception as e:
            print(f"\n⚠️  Error processing {pair['filename']}: {str(e)}")
    
    print(f"✅ Processed {len(processed_pairs)} images")
    return processed_pairs


def save_manifest(pairs: List[Dict], output_path: Path, stats: Dict):
    """Save dataset manifest as JSON"""
    manifest = {
        'metadata': {
            'total_samples': len(pairs),
            'statistics': stats,
        },
        'samples': pairs
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved manifest to: {output_path}")


def check_trigger_token(pairs: List[Dict], trigger_token: str):
    """Check if captions contain the trigger token"""
    print(f"\n🔍 Checking for trigger token: '{trigger_token}'")
    
    missing_token = []
    for pair in pairs:
        if trigger_token.lower() not in pair['caption'].lower():
            missing_token.append(pair['filename'])
    
    if missing_token:
        print(f"\n⚠️  Warning: {len(missing_token)} captions don't contain trigger token:")
        for filename in missing_token[:5]:  # Show first 5
            print(f"    - {filename}")
        if len(missing_token) > 5:
            print(f"    ... and {len(missing_token) - 5} more")
        print(f"\n   Consider adding '{trigger_token}' to your captions for better identity learning")
    else:
        print(f"✅ All captions contain the trigger token")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for SDXL fine-tuning")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to training config file'
    )
    parser.add_argument(
        '--resize',
        action='store_true',
        help='Resize images to target resolution'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=None,
        help='Target resolution (overrides config)'
    )
    parser.add_argument(
        '--no-crop',
        action='store_true',
        help='Disable center cropping (maintain aspect ratio)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get paths from config
    data_dir = Path(config['data_dir'])
    captions_dir = Path(config['captions_dir'])
    processed_dir = Path(config['processed_dir'])
    
    resolution = args.resolution or config['resolution']
    center_crop = config.get('center_crop', True) and not args.no_crop
    trigger_token = config.get('trigger_token', 'aldar_kose_man')
    
    print("=" * 70)
    print("  DATASET PREPARATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Captions directory: {captions_dir}")
    print(f"  Target resolution: {resolution}px")
    print(f"  Center crop: {center_crop}")
    print(f"  Resize images: {args.resize}")
    
    # Create directories if they don't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate dataset
    valid_pairs, errors = validate_dataset(data_dir, captions_dir)
    
    # Print errors
    if errors:
        print(f"\n⚠️  Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    if not valid_pairs:
        print("\n❌ No valid image-caption pairs found!")
        print("\nPlease ensure:")
        print("  1. Images are in: data/images/")
        print("  2. Captions are in: data/captions/")
        print("  3. Each image has a matching .txt caption file")
        print("  4. Caption files have the same name as images (e.g., img001.jpg -> img001.txt)")
        return 1
    
    print(f"\n✅ Found {len(valid_pairs)} valid image-caption pairs")
    
    # Analyze statistics
    stats = analyze_dataset_statistics(valid_pairs)
    
    # Check trigger token usage
    check_trigger_token(valid_pairs, trigger_token)
    
    # Process images if requested
    if args.resize:
        processed_pairs = process_images(valid_pairs, processed_dir, resolution, center_crop)
        # Save manifest with processed images
        save_manifest(processed_pairs, Path('data/dataset_manifest_processed.json'), stats)
    else:
        # Save manifest with original images
        save_manifest(valid_pairs, Path('data/dataset_manifest.json'), stats)
        print("\n💡 Tip: Use --resize flag to preprocess images to target resolution")
    
    print("\n" + "=" * 70)
    print("✅ Dataset preparation complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review the manifest file: data/dataset_manifest.json")
    print("  2. Configure accelerate: accelerate config")
    print("  3. Start training: accelerate launch scripts/train_lora_sdxl.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
