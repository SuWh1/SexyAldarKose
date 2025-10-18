#!/usr/bin/env python3
"""
Inference Script for SDXL LoRA - Aldar Kose Character Generation

Load trained LoRA weights and generate images with the fine-tuned model.

Usage:
    python scripts/inference.py --checkpoint outputs/checkpoints/checkpoint-1000 --prompt "3D render of aldar_kose_man smiling"
    python scripts/inference.py --checkpoint outputs/final --prompt "aldar_kose_man in traditional clothing" --num_images 4
"""

import argparse
import torch
from pathlib import Path
from PIL import Image

from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with SDXL LoRA")
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
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="low quality, blurry, distorted, deformed",
        help="Negative prompt",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/generated_images",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height",
    )
    
    return parser.parse_args()


def load_lora_pipeline(checkpoint_path: str, base_model: str):
    """Load SDXL pipeline with LoRA weights"""
    print(f"Loading base model: {base_model}")
    
    # Load base pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    
    checkpoint_path = Path(checkpoint_path)
    
    # Load LoRA weights for UNet
    unet_lora_path = checkpoint_path / "unet_lora"
    if unet_lora_path.exists():
        print(f"Loading UNet LoRA from: {unet_lora_path}")
        pipeline.unet = PeftModel.from_pretrained(pipeline.unet, str(unet_lora_path))
    else:
        print(f"Warning: UNet LoRA not found at {unet_lora_path}")
    
    # Load LoRA weights for text encoders (if available)
    text_encoder_one_lora_path = checkpoint_path / "text_encoder_one_lora"
    text_encoder_two_lora_path = checkpoint_path / "text_encoder_two_lora"
    
    if text_encoder_one_lora_path.exists():
        print(f"Loading Text Encoder 1 LoRA from: {text_encoder_one_lora_path}")
        pipeline.text_encoder = PeftModel.from_pretrained(
            pipeline.text_encoder, 
            str(text_encoder_one_lora_path)
        )
    
    if text_encoder_two_lora_path.exists():
        print(f"Loading Text Encoder 2 LoRA from: {text_encoder_two_lora_path}")
        pipeline.text_encoder_2 = PeftModel.from_pretrained(
            pipeline.text_encoder_2,
            str(text_encoder_two_lora_path)
        )
    
    # Enable memory optimizations
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        print("Enabled xformers memory efficient attention")
    except Exception as e:
        print(f"Could not enable xformers: {e}")
    
    pipeline.enable_model_cpu_offload()
    
    return pipeline


def generate_images(
    pipeline,
    prompt: str,
    negative_prompt: str,
    num_images: int,
    num_inference_steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int = None,
):
    """Generate images using the pipeline"""
    
    images = []
    
    for i in range(num_images):
        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed + i)
        else:
            generator = None
        
        print(f"\nGenerating image {i+1}/{num_images}...")
        print(f"Prompt: {prompt}")
        
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        ).images[0]
        
        images.append(image)
    
    return images


def save_images(images: list, output_dir: str, prompt: str, seed: int = None):
    """Save generated images to disk"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a safe filename from prompt
    safe_prompt = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in prompt)
    safe_prompt = safe_prompt[:50]  # Limit length
    
    saved_paths = []
    
    for i, image in enumerate(images):
        if seed is not None:
            filename = f"{safe_prompt}_seed{seed+i}_{i:03d}.png"
        else:
            filename = f"{safe_prompt}_{i:03d}.png"
        
        filepath = output_path / filename
        image.save(filepath)
        saved_paths.append(filepath)
        print(f"Saved: {filepath}")
    
    return saved_paths


def main():
    args = parse_args()
    
    print("=" * 70)
    print("  ALDAR KOSE SDXL LORA - Image Generation")
    print("=" * 70)
    
    # Load pipeline
    pipeline = load_lora_pipeline(args.checkpoint, args.base_model)
    
    # Generate images
    images = generate_images(
        pipeline=pipeline,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_images=args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    
    # Save images
    saved_paths = save_images(images, args.output_dir, args.prompt, args.seed)
    
    print("\n" + "=" * 70)
    print(f"✅ Generated {len(images)} images successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
