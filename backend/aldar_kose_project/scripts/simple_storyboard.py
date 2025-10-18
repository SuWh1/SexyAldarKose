#!/usr/bin/env python3
"""
Simplified Multi-Frame Inference with Basic Consistency
Uses only SDXL + LoRA (no additional models required)

This is a lightweight alternative that maintains consistency through:
1. Fixed seed per scene (with small variations)
2. Consistent prompts with scene anchors
3. CLIP similarity validation
4. Simple img2img guidance from previous frame

Use this if you don't have enough VRAM for ControlNet + IP-Adapter.
For full consistency pipeline, use storyboard_generator.py instead.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional
import torch
from PIL import Image
from tqdm import tqdm

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, CLIPTextModelWithProjection
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimplifiedStoryboardGenerator:
    """
    Lightweight storyboard generator using SDXL + LoRA only
    """
    
    def __init__(
        self,
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lora_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize simplified pipeline"""
        self.device = device
        self.dtype = dtype
        
        logger.info("Initializing Simplified Storyboard Generator...")
        
        # Use cache directory from training
        cache_dir = "/root/.cache/huggingface/hub"
        
        # Load base pipeline from cache
        logger.info("Loading SDXL base model from cache...")
        self.txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )
        self.txt2img_pipe.to(device)
        
        # Load PEFT LoRA weights if provided
        if lora_path:
            lora_path = Path(lora_path)
            logger.info(f"Loading PEFT LoRA from {lora_path}...")
            
            # Load UNet LoRA
            unet_lora_path = lora_path / "unet_lora"
            if unet_lora_path.exists():
                logger.info(f"  Loading UNet LoRA...")
                self.txt2img_pipe.unet = PeftModel.from_pretrained(
                    self.txt2img_pipe.unet,
                    str(unet_lora_path),
                )
            
            # Load Text Encoder LoRAs
            text_encoder_one_path = lora_path / "text_encoder_one_lora"
            text_encoder_two_path = lora_path / "text_encoder_two_lora"
            
            if text_encoder_one_path.exists():
                logger.info(f"  Loading Text Encoder 1 LoRA...")
                self.txt2img_pipe.text_encoder = PeftModel.from_pretrained(
                    self.txt2img_pipe.text_encoder,
                    str(text_encoder_one_path),
                )
            
            if text_encoder_two_path.exists():
                logger.info(f"  Loading Text Encoder 2 LoRA...")
                self.txt2img_pipe.text_encoder_2 = PeftModel.from_pretrained(
                    self.txt2img_pipe.text_encoder_2,
                    str(text_encoder_two_path),
                )
            
            logger.info("  ✓ LoRA weights loaded!")
        
        # Memory optimizations
        self.txt2img_pipe.enable_attention_slicing()
        self.txt2img_pipe.enable_vae_slicing()
        
        # Img2img pipeline for subsequent frames
        logger.info("Creating img2img pipeline...")
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.img2img_pipe.to(device)
        self.img2img_pipe.enable_attention_slicing()
        self.img2img_pipe.enable_vae_slicing()
        
        # Load CLIP for consistency
        logger.info("Loading CLIP for consistency validation...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        logger.info("✓ Pipeline ready!")
    
    def compute_clip_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute CLIP similarity between images"""
        inputs = self.clip_processor(
            images=[img1, img2],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        
        similarity = torch.nn.functional.cosine_similarity(
            features[0:1], features[1:2]
        )
        return similarity.item()
    
    def generate_sequence(
        self,
        prompts: List[str],
        base_seed: int = 42,
        negative_prompt: str = "blurry, low quality, distorted",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        consistency_threshold: float = 0.65,
        max_retries: int = 3,
        output_dir: str = "./simple_storyboard",
    ) -> List[Image.Image]:
        """
        Generate sequence of story frames with character consistency
        
        Strategy:
        - ALL frames: Full txt2img generation (different scenes/compositions)
        - Character consistency via LoRA (trained identity)
        - CLIP validation against first frame (face similarity)
        - Different seeds for scene variety
        - Retry if character doesn't match
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        frames = []
        scores = []
        
        # Generate all frames with txt2img for proper story progression
        for idx in range(len(prompts)):
            logger.info("=" * 60)
            logger.info(f"FRAME {idx + 1}/{len(prompts)}")
            logger.info("=" * 60)
            logger.info(f"Prompt: {prompts[idx][:80]}...")
            
            best_frame = None
            best_score = 0.0
            
            for attempt in range(max_retries):
                logger.info(f"Attempt {attempt + 1}/{max_retries}...")
                
                # Unique seed per frame (for variety) + attempt offset (for retries)
                seed = base_seed + (idx * 1000) + (attempt * 10)
                generator = torch.Generator(device=self.device).manual_seed(seed)
                
                # Generate fresh frame with txt2img (NOT img2img!)
                frame = self.txt2img_pipe(
                    prompt=prompts[idx],
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
                
                # For first frame, always accept
                if idx == 0:
                    best_frame = frame
                    best_score = 1.0
                    logger.info(f"  First frame - accepting")
                    break
                
                # Validate character consistency with first frame
                score = self.compute_clip_similarity(frames[0], frame)
                logger.info(f"  Character similarity: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_frame = frame
                
                if score >= consistency_threshold:
                    logger.info(f"  ✓ Meets threshold ({consistency_threshold})")
                    break
                else:
                    logger.warning(f"  ✗ Below threshold, regenerating...")
            
            frames.append(best_frame)
            scores.append(best_score)
            best_frame.save(output_path / f"frame_{idx + 1:03d}.png")
            logger.info(f"✓ Frame {idx + 1} saved (score: {best_score:.3f})")
        
        # Save report
        import numpy as np
        report = {
            "base_seed": base_seed,
            "num_frames": len(prompts),
            "generation_method": "txt2img (full scene generation)",
            "average_consistency": float(np.mean(scores)) if scores else 1.0,
            "min_consistency": float(np.min(scores)) if scores else 1.0,
            "frames": [
                {
                    "frame_id": i + 1,
                    "prompt": p,
                    "consistency_score": scores[i] if i < len(scores) else 1.0
                }
                for i, p in enumerate(prompts)
            ]
        }
        
        with open(output_path / "report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("=" * 60)
        logger.info(f"✓ Complete! {len(frames)} frames in {output_path}")
        logger.info(f"Average consistency: {report['average_consistency']:.3f}")
        logger.info("=" * 60)
        
        return frames


def main():
    parser = argparse.ArgumentParser(
        description="Simplified storyboard generator (SDXL + LoRA only)"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="outputs/aldar_kose_lora/final",
        help="Path to LoRA weights"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        required=True,
        help="JSON file with scene prompts"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./simple_storyboard",
        help="Output directory"
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed"
    )
    parser.add_argument(
        "--consistency-threshold",
        type=float,
        default=0.65,
        help="Minimum CLIP character similarity (0.60-0.70 recommended for variety)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Inference steps"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale (higher = follow prompt more)"
    )
    
    args = parser.parse_args()
    
    # Load prompts
    with open(args.prompts_file, 'r') as f:
        data = json.load(f)
        prompts = data.get("prompts", [])
    
    if not prompts:
        logger.error("No prompts found!")
        return
    
    # Initialize generator
    generator = SimplifiedStoryboardGenerator(
        lora_path=args.lora_path,
    )
    
    # Generate sequence
    frames = generator.generate_sequence(
        prompts=prompts,
        base_seed=args.base_seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        consistency_threshold=args.consistency_threshold,
        output_dir=args.output_dir,
    )
    
    logger.info(f"✓ Generated {len(frames)} frames!")


if __name__ == "__main__":
    main()
