#!/usr/bin/env python3
"""
Automated Image Labeling Pipeline using OpenAI Vision API

This script automatically generates descriptive captions for Aldar Kose images
using GPT-4 Vision, ensuring consistent trigger token usage and proper formatting.

Features:
- Supports multiple image formats (JPG, PNG, WebP, BMP, etc.)
- Handles various image sizes automatically
- Generates captions with trigger token "aldar_kose_man"
- Batch processing with progress tracking
- Saves captions in correct format for training
- Configurable caption style and length
- Error handling and retry logic

Usage:
    # Basic usage
    python scripts/label_images.py --input_dir raw_images/

    # With custom settings
    python scripts/label_images.py \
        --input_dir raw_images/ \
        --output_images data/images/ \
        --output_captions data/captions/ \
        --api_key YOUR_OPENAI_KEY

    # Use environment variable for API key
    export OPENAI_API_KEY=your_key_here
    python scripts/label_images.py --input_dir raw_images/

Requirements:
    pip install openai pillow python-dotenv
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import BytesIO

from PIL import Image
from tqdm import tqdm

try:
    import openai
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed.")
    print("Install with: pip install openai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv not installed. API key must be provided via --api_key")
    load_dotenv = None


# Configuration
TRIGGER_TOKEN = "aldar_kose_man"
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff'}

DEFAULT_SYSTEM_PROMPT = """You are an expert at describing visual scenes and artistic elements for AI training data.

You are captioning screenshots from a 3D animated video featuring a FICTIONAL CARTOON CHARACTER from Kazakh folklore called "Aldar Kose". 

IMPORTANT RULES:
1. DO NOT describe the character's clothing, outfit, hat, or traditional costume (this is constant across all images)
2. DO NOT describe his physical appearance or facial features (the model already knows this)
3. FOCUS ONLY ON: pose, expression, action, setting, background, lighting, camera angle, artistic style

This is training data where the character's IDENTITY and CLOTHING are already learned. Only describe WHAT HE'S DOING and WHERE.

Guidelines:
1. Start your caption with "aldar_kose_man" as an identifier token
2. Describe ONLY: pose, facial expression, action/activity, setting/location
3. Mention artistic style (3D render, CGI, cartoon style) and shot type (close-up, wide shot, etc.)
4. Include background/environment details and lighting
5. DO NOT mention clothing, outfit, costume, hat, traditional dress, etc.
6. Keep captions 40-100 characters focused on action and setting
7. Use action-focused descriptive language

Example captions (GOOD):
- "aldar_kose_man standing in village marketplace, smiling, 3D animation, wide shot"
- "aldar_kose_man riding horse across steppe, dynamic pose, cinematic lighting"
- "aldar_kose_man laughing heartily, close-up portrait, studio lighting"
- "aldar_kose_man sitting by campfire, storytelling gesture, warm atmosphere"
- "aldar_kose_man walking confidently, outdoor scene, dramatic sunset lighting"

Example captions (BAD - DO NOT USE):
- "aldar_kose_man in blue traditional coat and hat" ❌ (mentions clothing)
- "aldar_kose_man wearing ornate costume" ❌ (mentions outfit)
- "aldar_kose_man with colorful traditional dress" ❌ (mentions clothing)

Focus on ACTION, SETTING, and MOOD - NOT clothing or appearance."""


class ImageLabeler:
    """Automated image labeling using OpenAI Vision API"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 150,
        temperature: float = 0.7,
        retry_attempts: int = 3,
        retry_delay: int = 2,
    ):
        """
        Initialize the labeler
        
        Args:
            api_key: OpenAI API key (will try env var if not provided)
            model: OpenAI model to use (gpt-4o, gpt-4-turbo, etc.)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            retry_attempts: Number of retry attempts on failure
            retry_delay: Delay between retries (seconds)
        """
        # Load environment variables
        if load_dotenv:
            load_dotenv()
        
        # Get API key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Provide via --api_key or set OPENAI_API_KEY environment variable"
            )
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
    
    def encode_image(self, image_path: Path) -> str:
        """
        Encode image to base64 string
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Open and convert image to RGB
            with Image.open(image_path) as img:
                # Convert RGBA to RGB (remove transparency)
                if img.mode == 'RGBA':
                    # Create a white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                    img = background
                elif img.mode not in ('RGB',):
                    # Convert any other mode to RGB
                    img = img.convert('RGB')
                
                # Resize if too large (to save API costs)
                max_size = 2048
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to JPEG bytes
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                img_bytes = buffered.getvalue()
                
                # Encode to base64
                return base64.b64encode(img_bytes).decode('utf-8')
                
        except Exception as e:
            raise ValueError(f"Error encoding image {image_path}: {str(e)}")
    
    def generate_caption(
        self,
        image_path: Path,
        custom_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate caption for an image using OpenAI Vision API
        
        Args:
            image_path: Path to image file
            custom_prompt: Custom prompt (overrides default)
            
        Returns:
            Generated caption string
        """
        # Encode image
        base64_image = self.encode_image(image_path)
        
        # Prepare prompt
        user_prompt = custom_prompt or "Describe this 3D animated scene focusing ONLY on: character's pose, facial expression, action/activity, setting/location, background, lighting, and camera angle. DO NOT describe clothing, outfit, or physical appearance. This is animation training data where the character identity is already known."
        
        # Make API call with retry logic
        for attempt in range(self.retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": DEFAULT_SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": user_prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                
                # Extract caption
                caption = response.choices[0].message.content.strip()
                
                # Ensure trigger token is present
                if TRIGGER_TOKEN not in caption.lower():
                    # Insert trigger token at the beginning
                    caption = f"{TRIGGER_TOKEN}, {caption}"
                
                return caption
                
            except openai.RateLimitError as e:
                if attempt < self.retry_attempts - 1:
                    wait_time = self.retry_delay * (attempt + 1)
                    print(f"\n⚠️  Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Rate limit exceeded after {self.retry_attempts} attempts")
                    
            except openai.APIError as e:
                if attempt < self.retry_attempts - 1:
                    print(f"\n⚠️  API error, retrying... ({attempt + 1}/{self.retry_attempts})")
                    time.sleep(self.retry_delay)
                else:
                    raise Exception(f"API error after {self.retry_attempts} attempts: {str(e)}")
            
            except Exception as e:
                raise Exception(f"Error generating caption: {str(e)}")
        
        raise Exception("Failed to generate caption after all retry attempts")
    
    def process_batch(
        self,
        image_paths: List[Path],
        output_captions_dir: Path,
        output_images_dir: Optional[Path] = None,
        skip_existing: bool = True,
    ) -> Dict[str, any]:
        """
        Process a batch of images
        
        Args:
            image_paths: List of image file paths
            output_captions_dir: Directory to save caption files
            output_images_dir: Directory to copy images (optional, for organization)
            skip_existing: Skip images that already have captions
            
        Returns:
            Dictionary with processing statistics
        """
        output_captions_dir.mkdir(parents=True, exist_ok=True)
        if output_images_dir:
            output_images_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'total': len(image_paths),
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'errors': []
        }
        
        for image_path in tqdm(image_paths, desc="Labeling images"):
            try:
                # Check if caption already exists
                caption_filename = f"{image_path.stem}.txt"
                caption_path = output_captions_dir / caption_filename
                
                if skip_existing and caption_path.exists():
                    stats['skipped'] += 1
                    continue
                
                # Generate caption
                caption = self.generate_caption(image_path)
                
                # Save caption
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption)
                
                # Copy image if output directory specified
                if output_images_dir:
                    # Use PNG format for consistency
                    output_image_path = output_images_dir / f"{image_path.stem}.png"
                    
                    # Copy and convert image
                    with Image.open(image_path) as img:
                        if img.mode == 'RGBA':
                            # Create white background for RGBA images
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                            background.save(output_image_path, 'PNG')
                        else:
                            img.convert('RGB').save(output_image_path, 'PNG')
                
                stats['processed'] += 1
                
                # Small delay to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                stats['failed'] += 1
                stats['errors'].append({
                    'file': str(image_path),
                    'error': str(e)
                })
                print(f"\n❌ Error processing {image_path.name}: {str(e)}")
        
        return stats


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory"""
    image_files = []
    seen = set()
    
    for ext in SUPPORTED_FORMATS:
        for file in directory.glob(f"*{ext}"):
            # Use resolve() to get canonical path and avoid duplicates on case-insensitive filesystems
            canonical = file.resolve()
            if canonical not in seen:
                seen.add(canonical)
                image_files.append(file)
        for file in directory.glob(f"*{ext.upper()}"):
            canonical = file.resolve()
            if canonical not in seen:
                seen.add(canonical)
                image_files.append(file)
    
    return sorted(image_files)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Automated image labeling for Aldar Kose dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        '--input_dir',
        type=str,
        default='raw_images',
        help='Directory containing images to label (default: raw_images)',
    )
    
    # Optional arguments
    parser.add_argument(
        '--output_images',
        type=str,
        default='data/images',
        help='Output directory for processed images (default: data/images)',
    )
    parser.add_argument(
        '--output_captions',
        type=str,
        default='data/captions',
        help='Output directory for captions (default: data/captions)',
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='OpenAI API key (or set OPENAI_API_KEY env var)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        choices=['gpt-4o', 'gpt-4-turbo', 'gpt-4o-mini'],
        help='OpenAI model to use (default: gpt-4o)',
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        default=True,
        help='Skip images that already have captions (default: True)',
    )
    parser.add_argument(
        '--no_skip_existing',
        action='store_false',
        dest='skip_existing',
        help='Re-process all images, overwriting existing captions',
    )
    parser.add_argument(
        '--no_copy_images',
        action='store_true',
        help='Do not copy images to output directory (only generate captions)',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (0.0-2.0, default: 0.7)',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be processed without making API calls',
    )
    
    return parser.parse_args()


def print_summary(stats: Dict, processing_time: float):
    """Print processing summary"""
    print("\n" + "=" * 70)
    print("  LABELING COMPLETE")
    print("=" * 70)
    print(f"\nProcessing Summary:")
    print(f"  Total images: {stats['total']}")
    print(f"  ✅ Successfully labeled: {stats['processed']}")
    print(f"  ⏭️  Skipped (already labeled): {stats['skipped']}")
    print(f"  ❌ Failed: {stats['failed']}")
    print(f"\n  ⏱️  Processing time: {processing_time:.1f}s")
    
    if stats['processed'] > 0:
        print(f"  Average time per image: {processing_time / stats['processed']:.1f}s")
    
    if stats['errors']:
        print(f"\n⚠️  Errors encountered:")
        for error in stats['errors'][:5]:  # Show first 5 errors
            print(f"  - {error['file']}: {error['error']}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more errors")
    
    print("\n" + "=" * 70)


def main():
    args = parse_args()
    
    print("=" * 70)
    print("  AUTOMATED IMAGE LABELING - Aldar Kose Dataset")
    print("=" * 70)
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"\n❌ Error: Input directory not found: {input_dir}")
        return 1
    
    # Get image files
    image_files = get_image_files(input_dir)
    
    if not image_files:
        print(f"\n❌ No images found in {input_dir}")
        print(f"   Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return 1
    
    print(f"\n📁 Input directory: {input_dir}")
    print(f"   Found {len(image_files)} images")
    
    # Setup output directories
    output_images_dir = None if args.no_copy_images else Path(args.output_images)
    output_captions_dir = Path(args.output_captions)
    
    print(f"\n📁 Output directories:")
    if output_images_dir:
        print(f"   Images: {output_images_dir}")
    print(f"   Captions: {output_captions_dir}")
    
    # Check for existing captions
    if args.skip_existing:
        existing_captions = list(output_captions_dir.glob("*.txt")) if output_captions_dir.exists() else []
        if existing_captions:
            print(f"\n   {len(existing_captions)} existing captions will be skipped")
    
    print(f"\n🤖 Model: {args.model}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Trigger token: {TRIGGER_TOKEN}")
    
    # Dry run mode
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No API calls will be made")
        print("\nImages that would be processed:")
        for img_path in image_files[:10]:
            caption_path = output_captions_dir / f"{img_path.stem}.txt"
            status = "skip" if args.skip_existing and caption_path.exists() else "process"
            print(f"  [{status}] {img_path.name}")
        if len(image_files) > 10:
            print(f"  ... and {len(image_files) - 10} more")
        return 0
    
    # Confirm before processing
    print("\n" + "-" * 70)
    response = input("\nProceed with labeling? This will use OpenAI API credits. (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return 0
    
    # Initialize labeler
    try:
        labeler = ImageLabeler(
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
        )
    except ValueError as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    # Process images
    start_time = time.time()
    
    try:
        stats = labeler.process_batch(
            image_paths=image_files,
            output_captions_dir=output_captions_dir,
            output_images_dir=output_images_dir,
            skip_existing=args.skip_existing,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        return 1
    
    processing_time = time.time() - start_time
    
    # Print summary
    print_summary(stats, processing_time)
    
    # Next steps
    if stats['processed'] > 0:
        print("\n📝 Next steps:")
        print("  1. Review the generated captions in:", output_captions_dir)
        print("  2. Edit any captions that need improvement")
        print("  3. Run: python scripts/prepare_dataset.py")
        print("  4. Start training!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
