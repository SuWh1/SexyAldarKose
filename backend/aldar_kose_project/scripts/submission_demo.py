#!/usr/bin/env python3
"""
Submission Demo Script
Downloads LoRA model from AWS S3 (via public HTTPS URLs) and runs deterministic inference.

🎯 NO AWS CONFIGURATION NEEDED - Downloads via public HTTPS URLs!

Usage:
    # Interactive mode (prompts for story)
    python scripts/submission_demo.py
    
    # With story prompt
    python scripts/submission_demo.py "Aldar Kose tricks a greedy merchant"
    
    # Skip download (model already local)
    python scripts/submission_demo.py "Aldar at bazaar" --skip-download
    
    # Simple mode (faster, less VRAM)
    python scripts/submission_demo.py "Aldar adventure" --no-ref-guided
"""

import os
import sys
import argparse
import logging
import subprocess
import urllib.request
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
S3_BUCKET_URL = "https://aldarkose.s3.amazonaws.com"  # Public S3 URL (no auth needed!)
S3_MODEL_PATH = "checkpoint-1000"
LOCAL_MODEL_PATH = "outputs/checkpoints/checkpoint-1000"
TEMP = 0.0  # Fully deterministic
SEED = 42
REF_GUIDED = True  # Reference-guided mode for best consistency
DEFAULT_PROMPT = "Aldar Kose, the clever trickster, outwits a greedy merchant in an exciting bazaar confrontation"

# Files to download (from S3)
MODEL_FILES = {
    "unet_lora/adapter_config.json": "adapter_config.json",
    "unet_lora/adapter_model.safetensors": "adapter_model.safetensors",
    "text_encoder_one_lora/adapter_config.json": "adapter_config.json",
    "text_encoder_one_lora/adapter_model.safetensors": "adapter_model.safetensors",
    "text_encoder_two_lora/adapter_config.json": "adapter_config.json",
    "text_encoder_two_lora/adapter_model.safetensors": "adapter_model.safetensors",
}


def download_model_from_s3(base_url: str, model_path: str, local_path: str) -> bool:
    """Download LoRA model from public S3 URL (NO AWS credentials needed!)."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("📦 DOWNLOADING MODEL FROM S3")
    logger.info("=" * 70)
    logger.info(f"Source: {base_url}/{model_path}/")
    logger.info(f"Local: {local_path}/")
    logger.info("")
    logger.info("🔓 No AWS credentials needed - using public HTTPS URLs!")
    
    # Create local directory
    Path(local_path).mkdir(parents=True, exist_ok=True)
    
    try:
        downloaded_count = 0
        total_size = 0
        
        # Download files
        for s3_file, local_file in MODEL_FILES.items():
            s3_full_path = f"{model_path}/{s3_file}"
            url = f"{base_url}/{s3_full_path}"
            
            # Create local file path
            local_file_path = Path(local_path) / s3_file
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"  ⬇️  {s3_file}")
            
            try:
                # Download with progress
                urllib.request.urlretrieve(
                    url,
                    str(local_file_path),
                    reporthook=lambda block_num, block_size, total_size: None
                )
                
                file_size = local_file_path.stat().st_size / (1024 * 1024)  # MB
                total_size += file_size
                downloaded_count += 1
                logger.info(f"       ✓ {file_size:.1f} MB")
                
            except urllib.error.URLError as e:
                logger.error(f"  ❌ Failed to download: {str(e)}")
                return False
            except Exception as e:
                logger.error(f"  ❌ Error: {str(e)}")
                return False
        
        if downloaded_count == 0:
            logger.error(f"No files downloaded from {base_url}/{model_path}/")
            return False
        
        logger.info("")
        logger.info(f"✅ Downloaded {downloaded_count} files successfully!")
        logger.info(f"   Total size: {total_size:.1f} MB")
        logger.info(f"   Location: {local_path}")
        
        # Verify download
        adapter_files = list(Path(local_path).glob("**/adapter_model.safetensors"))
        if len(adapter_files) < 3:
            logger.error(f"❌ Expected 3 adapter files, found {len(adapter_files)}")
            return False
        
        return True
        
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
    
    missing = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            logger.info(f"  ✅ {package_name}")
        except ImportError:
            logger.warning(f"  ❌ {package_name} (missing)")
            missing.append(package_name)
    
    # Check for optional but recommended packages
    optional_packages = [
        ('controlnet_aux', 'controlnet-aux'),
        ('onnxruntime', 'onnxruntime'),
    ]
    
    for import_name, package_name in optional_packages:
        try:
            __import__(import_name)
            logger.info(f"  ✅ {package_name} (optional)")
        except ImportError:
            logger.info(f"  ⚠️  {package_name} (optional - recommended for ControlNet)")
    
    if missing:
        logger.error(f"\n❌ Missing required packages: {', '.join(missing)}")
        logger.error(f"   Install with:")
        logger.error(f"   pip install -r requirements_inference.txt")
        logger.error(f"\n   Or individually:")
        logger.error(f"   pip install {' '.join(missing)}")
        return False
    
    logger.info("")
    logger.info("✅ All required dependencies installed!")
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
        '--skip-download',
        action='store_true',
        help='Skip S3 download (use existing local model)'
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
    
    # Step 3: Check if model exists locally
    model_exists = Path(LOCAL_MODEL_PATH).exists() and list(Path(LOCAL_MODEL_PATH).glob("**/adapter_model.safetensors"))
    
    if model_exists and args.skip_download:
        logger.info("")
        logger.info("=" * 70)
        logger.info("⏭️  SKIPPING AWS DOWNLOAD (--skip-download)")
        logger.info("=" * 70)
        logger.info(f"✅ Using cached local model: {LOCAL_MODEL_PATH}")
    elif model_exists and not args.skip_download:
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ MODEL ALREADY DOWNLOADED")
        logger.info("=" * 70)
        logger.info(f"Using cached model: {LOCAL_MODEL_PATH}")
    elif not model_exists:
        logger.info("")
        logger.info("=" * 70)
        logger.info("⬇️  MODEL NOT FOUND - DOWNLOADING FROM S3")
        logger.info("=" * 70)
        if not download_model_from_s3(S3_BUCKET_URL, S3_MODEL_PATH, LOCAL_MODEL_PATH):
            logger.error("\n❌ Download failed")
            logger.error("   Check your internet connection and try again")
            sys.exit(1)
    
    # Final verification
    if not Path(LOCAL_MODEL_PATH).exists():
        logger.error(f"❌ Model not available at: {LOCAL_MODEL_PATH}")
        sys.exit(1)
    
    # Step 4: Get story prompt
    prompt = args.prompt if args.prompt else get_user_prompt()
    
    # Step 5: Run inference
    ref_guided = not args.no_ref_guided
    if not run_inference(prompt, seed=args.seed, temp=TEMP, ref_guided=ref_guided):
        sys.exit(1)
    
    # Find the most recent generation folder
    output_root = Path("outputs")
    generation_dirs = sorted(
        [d for d in output_root.glob("terminal_generation_*") if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if generation_dirs:
        actual_output_dir = str(generation_dirs[0])
    else:
        # Fallback (shouldn't happen)
        actual_output_dir = f"outputs/terminal_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("🎉 SUBMISSION READY!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("✅ Generated files:")
    
    logger.info(f"   📁 Location: {actual_output_dir}")
    logger.info(f"   🖼️  Images: {actual_output_dir}/frame_*.png")
    logger.info(f"   📝 Metadata: {actual_output_dir}/scene_breakdown.json")
    logger.info("")
    logger.info("✨ To regenerate identical output, use:")
    logger.info(f'   python scripts/submission_demo.py "{prompt}" --seed {args.seed} --skip-download')
    logger.info("")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
