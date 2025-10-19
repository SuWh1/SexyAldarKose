#!/usr/bin/env python3
"""
Submission Demo Script
Downloads LoRA model from AWS S3 and runs deterministic inference with ref-guided generation.

Usage:
    # Interactive mode (prompts for story)
    python scripts/submission_demo.py
    
    # With story prompt
    python scripts/submission_demo.py "Aldar Kose tricks a greedy merchant"
    
    # With custom AWS bucket
    python scripts/submission_demo.py "Aldar wins a horse race" --bucket my-custom-bucket
    
    # Skip download (model already local)
    python scripts/submission_demo.py "Aldar at bazaar" --skip-download
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import json

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Constants
AWS_BUCKET = "aldarkose"
S3_MODEL_PATH = "checkpoint-1000"  # s3://aldarkose/checkpoint-1000/
LOCAL_MODEL_PATH = "outputs/checkpoints/checkpoint-1000"
TEMP = 0.0  # Fully deterministic
SEED = 42
REF_GUIDED = True  # Reference-guided mode for best consistency
DEFAULT_PROMPT = "Aldar Kose, the clever trickster, outwits a greedy merchant in an exciting bazaar confrontation"


def download_model_from_aws(bucket: str, s3_path: str, local_path: str) -> bool:
    """Download LoRA model from AWS S3 using boto3."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("📦 DOWNLOADING MODEL FROM AWS")
    logger.info("=" * 70)
    logger.info(f"Bucket: s3://{bucket}/")
    logger.info(f"Source: {s3_path}/")
    logger.info(f"Local: {local_path}/")
    
    # Create local directory
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        logger.info("")
        logger.info("Connecting to S3...")
        
        # Create S3 client
        s3_client = boto3.client('s3')
        
        # List objects in the S3 path
        logger.info(f"Listing objects from s3://{bucket}/{s3_path}/")
        
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=s3_path)
        
        downloaded_count = 0
        for page in pages:
            if 'Contents' not in page:
                logger.warning(f"No objects found at s3://{bucket}/{s3_path}/")
                return False
            
            for obj in page['Contents']:
                key = obj['Key']
                # Skip directory markers
                if key.endswith('/'):
                    continue
                
                # Get local file path
                relative_path = key[len(s3_path):].lstrip('/')
                local_file = Path(local_path) / relative_path
                local_file.parent.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"  ⬇️  Downloading: {key}")
                
                try:
                    s3_client.download_file(bucket, key, str(local_file))
                    downloaded_count += 1
                except ClientError as e:
                    logger.error(f"Failed to download {key}: {e}")
                    return False
        
        if downloaded_count == 0:
            logger.error(f"No files downloaded from s3://{bucket}/{s3_path}/")
            return False
        
        logger.info("")
        logger.info(f"✅ Downloaded {downloaded_count} files successfully!")
        logger.info(f"   Location: {local_path}")
        
        # Verify download
        if not Path(local_path).exists():
            logger.error(f"❌ Model path does not exist: {local_path}")
            return False
        
        return True
        
    except ImportError:
        logger.error("❌ boto3 not found. Please install: pip install boto3")
        return False
    except Exception as e:
        logger.error(f"❌ Error downloading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def verify_dependencies() -> bool:
    """Verify all required packages are installed."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ CHECKING DEPENDENCIES")
    logger.info("=" * 70)
    
    required_packages = [
        'torch',
        'transformers',
        'diffusers',
        'peft',
        'PIL',
        'openai',
        'boto3'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"  ✅ {package}")
        except ImportError:
            logger.warning(f"  ❌ {package} (missing)")
            missing.append(package)
    
    if missing:
        logger.error(f"\n❌ Missing packages: {', '.join(missing)}")
        logger.error(f"   Install with: pip install {' '.join(missing)}")
        return False
    
    logger.info("")
    logger.info("✅ All dependencies installed!")
    return True


def verify_openai_key() -> bool:
    """Verify OpenAI API key is set."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ CHECKING OPENAI API KEY")
    logger.info("=" * 70)
    
    if not os.environ.get('OPENAI_API_KEY'):
        logger.error("❌ OPENAI_API_KEY environment variable not set")
        logger.error("   Set with: export OPENAI_API_KEY=sk-...")
        logger.error("   Or in .env file in project root")
        return False
    
    api_key = os.environ.get('OPENAI_API_KEY')
    masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 11 else "***"
    logger.info(f"  ✅ OpenAI API Key: {masked_key}")
    
    return True


def run_inference(prompt: str, seed: int = SEED, temp: float = TEMP, ref_guided: bool = True):
    """Run story generation with deterministic settings."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("🎬 GENERATING STORY")
    logger.info("=" * 70)
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Mode: {'Reference-Guided' if ref_guided else 'Simple'}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Temperature: {temp} (DETERMINISTIC)")
    logger.info(f"LoRA Model: {LOCAL_MODEL_PATH}")
    
    # Build command
    cmd = [
        "python",
        "scripts/generate_story.py",
        prompt,
        f"--seed={seed}",
        f"--temp={temp}",
        f"--lora-path={LOCAL_MODEL_PATH}"
    ]
    
    if ref_guided:
        cmd.append("--ref-guided")
    
    logger.info("")
    logger.info("Running: " + " ".join(cmd))
    logger.info("")
    logger.info("⏳ This will take 4-5 minutes (ref-guided mode)...")
    logger.info("   First 1-2 min: Model loading and initialization")
    logger.info("   Next 3-4 min: Story generation and consistency checks")
    logger.info("")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode != 0:
            logger.error(f"❌ Generation failed with exit code {result.returncode}")
            return False
        
        logger.info("")
        logger.info("✅ Story generation complete!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error running generation: {str(e)}")
        return False


def get_user_prompt() -> str:
    """Get story prompt from user."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("📝 ENTER STORY PROMPT")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Examples:")
    logger.info("  - 'Aldar Kose outwits a greedy merchant'")
    logger.info("  - 'Aldar wins a horse race against a champion'")
    logger.info("  - 'Aldar teaches a valuable lesson through clever tricks'")
    logger.info("")
    
    prompt = input("Enter your story prompt (or press Enter for default): ").strip()
    
    if not prompt:
        logger.info(f"Using default: '{DEFAULT_PROMPT}'")
        return DEFAULT_PROMPT
    
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description='Submission Demo: Download model and generate story',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/submission_demo.py
  
  # With story prompt
  python scripts/submission_demo.py "Aldar tricks a merchant"
  
  # Skip AWS download (model already local)
  python scripts/submission_demo.py "Aldar adventure" --skip-download
  
  # Custom AWS bucket
  python scripts/submission_demo.py "Aldar story" --bucket my-bucket
  
  # Simple mode (faster, less VRAM)
  python scripts/submission_demo.py "Aldar adventure" --no-ref-guided
        """
    )
    
    parser.add_argument(
        'prompt',
        nargs='?',
        default=None,
        help='Story prompt (if not provided, will ask interactively)'
    )
    
    parser.add_argument(
        '--bucket',
        type=str,
        default=AWS_BUCKET,
        help=f'AWS S3 bucket (default: {AWS_BUCKET})'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip AWS download (use existing local model)'
    )
    
    parser.add_argument(
        '--no-ref-guided',
        action='store_true',
        help='Use simple mode instead of reference-guided (faster, less VRAM, fewer dependencies)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=SEED,
        help=f'Random seed (default: {SEED})'
    )
    
    args = parser.parse_args()
    
    # Main execution
    logger.info("")
    logger.info("╔" + "=" * 68 + "╗")
    logger.info("║" + " " * 15 + "🎭 ALDAR KOSE STORY GENERATOR 🎭" + " " * 19 + "║")
    logger.info("║" + " " * 10 + "Submission Demo - Download Model & Generate Story" + " " * 8 + "║")
    logger.info("╚" + "=" * 68 + "╝")
    
    # Step 1: Check dependencies
    if not verify_dependencies():
        sys.exit(1)
    
    # Step 2: Check OpenAI key
    if not verify_openai_key():
        logger.error("\n💡 Quick setup:")
        logger.error("   1. Get key from: https://platform.openai.com/api-keys")
        logger.error("   2. Set in terminal: export OPENAI_API_KEY=sk-...")
        logger.error("   3. Run script again")
        sys.exit(1)
    
    # Step 3: Download model from AWS (unless skipped)
    if not args.skip_download:
        if not download_model_from_aws(args.bucket, S3_MODEL_PATH, LOCAL_MODEL_PATH):
            logger.error("\n💡 If model is already local, use: --skip-download")
            sys.exit(1)
    else:
        logger.info("")
        logger.info("=" * 70)
        logger.info("⏭️  SKIPPING AWS DOWNLOAD (--skip-download)")
        logger.info("=" * 70)
        logger.info(f"Using local model: {LOCAL_MODEL_PATH}")
        
        if not Path(LOCAL_MODEL_PATH).exists():
            logger.error(f"❌ Local model not found at: {LOCAL_MODEL_PATH}")
            logger.error("   Run without --skip-download to download from AWS")
            sys.exit(1)
    
    # Step 4: Get story prompt
    prompt = args.prompt if args.prompt else get_user_prompt()
    
    # Step 5: Run inference
    ref_guided = not args.no_ref_guided
    if not run_inference(prompt, seed=args.seed, temp=TEMP, ref_guided=ref_guided):
        sys.exit(1)
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("🎉 SUBMISSION READY!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("✅ Generated files:")
    
    timestamp = datetime.now().strftime('%Y%m%d')
    output_dir = f"outputs/terminal_generation_{timestamp}"
    
    logger.info(f"   📁 Location: {output_dir}")
    logger.info(f"   🖼️  Images: {output_dir}/frame_*.png")
    logger.info(f"   📝 Metadata: {output_dir}/scene_breakdown.json")
    logger.info("")
    logger.info("✨ To regenerate identical output, use:")
    logger.info(f'   python scripts/submission_demo.py "{prompt}" --seed {args.seed} --skip-download')
    logger.info("")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
