#!/usr/bin/env python3
"""
Quick Test Inference Script

Generate a set of test images to verify the trained LoRA model.

Usage:
    python scripts/test_inference.py --checkpoint outputs/checkpoints/checkpoint-500
    python scripts/test_inference.py --checkpoint outputs/aldar_kose_lora/final
"""

import argparse
import torch
from pathlib import Path
from datetime import datetime

from diffusers import StableDiffusionXLPipeline
from peft import PeftModel


# Test prompts for Aldar Kose character
TEST_PROMPTS = [
    "3D render of aldar_kose_man smiling happily, traditional Kazakh clothing, high quality",
    "aldar_kose_man riding a donkey, outdoor village scene, 3D animation style",
    "portrait of aldar_kose_man with friendly expression, detailed, cinematic lighting",
    "aldar_kose_man character full body, white background, 3D CGI render",
    "aldar_kose_man in traditional blue coat, standing confidently, bright day",
    "aldar_kose_man character waving hello, cheerful mood, 3D animation",
]


def load_lora_pipeline(checkpoint_path: str, base_model: str):
    """Load SDXL pipeline with LoRA weights"""
    print(f"\n🔧 Loading base model: {base_model}")
    
    checkpoint_path = Path(checkpoint_path)
    
    # Load base pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    
    print(f"🔧 Loading LoRA weights from: {checkpoint_path}")
    
    # Load LoRA weights for UNet
    unet_lora_path = checkpoint_path / "unet_lora"
    if unet_lora_path.exists():
        pipeline.unet = PeftModel.from_pretrained(
            pipeline.unet,
            str(unet_lora_path),
            torch_dtype=torch.float16,
        )
        print("✅ Loaded UNet LoRA weights")
    else:
        print(f"⚠️  Warning: UNet LoRA not found at {unet_lora_path}")
    
    # Load LoRA weights for text encoders (if trained)
    text_encoder_one_path = checkpoint_path / "text_encoder_one_lora"
    text_encoder_two_path = checkpoint_path / "text_encoder_two_lora"
    
    if text_encoder_one_path.exists():
        pipeline.text_encoder = PeftModel.from_pretrained(
            pipeline.text_encoder,
            str(text_encoder_one_path),
            torch_dtype=torch.float16,
        )
        print("✅ Loaded Text Encoder 1 LoRA weights")
    
    if text_encoder_two_path.exists():
        pipeline.text_encoder_2 = PeftModel.from_pretrained(
            pipeline.text_encoder_2,
            str(text_encoder_two_path),
            torch_dtype=torch.float16,
        )
        print("✅ Loaded Text Encoder 2 LoRA weights")
    
    # Move to GPU
    pipeline.to("cuda")
    pipeline.enable_attention_slicing()
    
    # Try to enable xformers for memory efficiency
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        print("✅ Enabled xformers memory efficient attention")
    except:
        print("⚠️  xformers not available, using standard attention")
    
    return pipeline


def generate_test_images(
    pipeline,
    prompts,
    output_dir,
    negative_prompt="low quality, blurry, distorted, deformed",
    num_inference_steps=30,
    guidance_scale=7.5,
    seeds=[42, 123, 456],
):
    """Generate test images for multiple prompts"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = output_dir / f"test_{timestamp}"
    test_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Saving images to: {test_dir}")
    print(f"🎨 Generating {len(prompts)} test images...")
    print(f"⚙️  Settings: steps={num_inference_steps}, guidance={guidance_scale}\n")
    
    all_images = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Generating: {prompt[:60]}...")
        
        # Use first seed for consistency
        generator = torch.Generator(device="cuda").manual_seed(seeds[0])
        
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=1024,
            width=1024,
        ).images[0]
        
        # Save image
        image_path = test_dir / f"test_{i:02d}.png"
        image.save(image_path)
        
        # Save prompt
        prompt_path = test_dir / f"test_{i:02d}_prompt.txt"
        with open(prompt_path, 'w') as f:
            f.write(prompt)
        
        all_images.append((image, prompt))
        print(f"✅ Saved: {image_path.name}")
    
    # Create a grid/contact sheet
    try:
        from PIL import Image as PILImage
        
        # Create 2x3 grid
        grid_width = 2048
        grid_height = 3072
        grid = PILImage.new('RGB', (grid_width, grid_height), 'white')
        
        for idx, (img, _) in enumerate(all_images):
            row = idx // 2
            col = idx % 2
            img_resized = img.resize((1024, 1024))
            grid.paste(img_resized, (col * 1024, row * 1024))
        
        grid_path = test_dir / "grid.png"
        grid.save(grid_path)
        print(f"\n✅ Created image grid: {grid_path.name}")
    except Exception as e:
        print(f"⚠️  Could not create grid: {e}")
    
    return test_dir


def main():
    parser = argparse.ArgumentParser(description="Test inference with trained LoRA")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/test_generations",
        help="Output directory for test images",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Guidance scale",
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("  ALDAR KOSE LORA - TEST INFERENCE")
    print("="*70)
    
    # Load pipeline
    pipeline = load_lora_pipeline(args.checkpoint, args.base_model)
    
    # Generate test images
    test_dir = generate_test_images(
        pipeline,
        TEST_PROMPTS,
        args.output_dir,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
    )
    
    print("\n" + "="*70)
    print("  ✅ TEST GENERATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Results saved to: {test_dir}")
    print(f"📊 Generated {len(TEST_PROMPTS)} test images")
    print("\n💡 Review the images to check:")
    print("   - Character identity consistency")
    print("   - Quality and details")
    print("   - Response to different prompts")
    print("   - Lighting and composition")
    print("\n")


if __name__ == "__main__":
    main()
