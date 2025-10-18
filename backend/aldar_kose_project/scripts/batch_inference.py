#!/usr/bin/env python3
"""
Batch Inference Script - Generate Multiple Images with Different Prompts

Generate a batch of images for evaluation or demonstration.

Usage:
    python scripts/batch_inference.py --checkpoint outputs/final --prompts_file prompts.txt
"""

import argparse
import json
from pathlib import Path
from typing import List
import torch
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionXLPipeline
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Batch image generation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to LoRA checkpoint",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="Path to text file with prompts (one per line)",
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
        default="outputs/batch_generated",
        help="Output directory",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images per prompt",
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
        help="CFG scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    
    return parser.parse_args()


def load_prompts(prompts_file: str) -> List[str]:
    """Load prompts from text file"""
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def load_pipeline(checkpoint_path: str, base_model: str):
    """Load SDXL pipeline with LoRA"""
    print("Loading pipeline...")
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    
    checkpoint_path = Path(checkpoint_path)
    
    # Load LoRA weights
    unet_lora_path = checkpoint_path / "unet_lora"
    if unet_lora_path.exists():
        pipeline.unet = PeftModel.from_pretrained(pipeline.unet, str(unet_lora_path))
    
    text_encoder_one_lora_path = checkpoint_path / "text_encoder_one_lora"
    text_encoder_two_lora_path = checkpoint_path / "text_encoder_two_lora"
    
    if text_encoder_one_lora_path.exists():
        pipeline.text_encoder = PeftModel.from_pretrained(
            pipeline.text_encoder,
            str(text_encoder_one_lora_path)
        )
    
    if text_encoder_two_lora_path.exists():
        pipeline.text_encoder_2 = PeftModel.from_pretrained(
            pipeline.text_encoder_2,
            str(text_encoder_two_lora_path)
        )
    
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except:
        pass
    
    pipeline.enable_model_cpu_offload()
    
    return pipeline


def main():
    args = parse_args()
    
    # Load prompts
    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts")
    
    # Load pipeline
    pipeline = load_pipeline(args.checkpoint, args.base_model)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate images
    results = []
    
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        for img_idx in range(args.num_images_per_prompt):
            seed = args.seed + prompt_idx * args.num_images_per_prompt + img_idx
            generator = torch.Generator(device="cuda").manual_seed(seed)
            
            image = pipeline(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]
            
            # Save image
            filename = f"prompt{prompt_idx:03d}_img{img_idx:02d}_seed{seed}.png"
            filepath = output_dir / filename
            image.save(filepath)
            
            results.append({
                'prompt': prompt,
                'seed': seed,
                'filename': filename,
            })
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Generated {len(results)} images in {output_dir}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
