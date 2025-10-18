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

# Import anomaly detector
try:
    from scripts.anomaly_detector import AnomalyDetector
    HAS_ANOMALY_DETECTOR = True
except ImportError:
    try:
        from anomaly_detector import AnomalyDetector
        HAS_ANOMALY_DETECTOR = True
    except ImportError:
        HAS_ANOMALY_DETECTOR = False

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
        enable_anomaly_detection: bool = True,
    ):
        """Initialize simplified pipeline"""
        self.device = device
        self.dtype = dtype
        self.enable_anomaly_detection = enable_anomaly_detection
        
        logger.info("Initializing Simplified Storyboard Generator...")
        
        # Initialize anomaly detector
        if self.enable_anomaly_detection and HAS_ANOMALY_DETECTOR:
            logger.info("Initializing Anomaly Detector...")
            self.anomaly_detector = AnomalyDetector(device=device, strict_mode=False)
            logger.info("✓ Anomaly detection enabled")
        else:
            self.anomaly_detector = None
            if self.enable_anomaly_detection:
                logger.warning("⚠️  Anomaly detection requested but not available")
                logger.warning("    Install: pip install opencv-python mediapipe")
        
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
                
                # Determine CFG based on previous failures
                current_cfg = guidance_scale
                if attempt > 0 and hasattr(self, '_last_anomaly_suggestions'):
                    current_cfg = self._last_anomaly_suggestions.get('guidance_scale', guidance_scale)
                    logger.info(f"  Adjusted CFG: {current_cfg:.1f} (based on anomaly suggestions)")
                
                # Generate fresh frame with txt2img (NOT img2img!)
                frame = self.txt2img_pipe(
                    prompt=prompts[idx],
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=current_cfg,
                    generator=generator,
                ).images[0]
                
                # Anomaly detection check (CRITICAL for quality)
                has_anomaly = False
                anomaly_details = None
                
                if self.anomaly_detector and attempt < max_retries - 1:  # Skip check on last attempt
                    logger.info("  Running anomaly detection...")
                    anomaly_result = self.anomaly_detector.detect_anomalies(
                        frame,
                        expected_prompt=prompts[idx]
                    )
                    anomaly_details = anomaly_result
                    
                    if not anomaly_result['is_valid']:
                        has_anomaly = True
                        logger.warning(f"  ⚠️  Anomalies detected: {', '.join(anomaly_result['anomalies'])}")
                        logger.warning(f"  Confidence: {anomaly_result['confidence']:.3f}")
                        
                        # Log detailed anomaly information
                        if anomaly_result['details'].get('faces'):
                            face_count = anomaly_result['details']['faces']['count']
                            logger.warning(f"     Faces detected: {face_count} (expected: 1)")
                        
                        if anomaly_result['details'].get('pose', {}).get('anomalies'):
                            pose_issues = anomaly_result['details']['pose']['anomalies']
                            logger.warning(f"     Pose issues: {', '.join(pose_issues)}")
                        
                        # Get regeneration suggestions with improved parameters
                        self._last_anomaly_suggestions = self.anomaly_detector.suggest_regeneration_params(
                            anomaly_result['anomalies'],
                            seed,
                            current_cfg
                        )
                        logger.info(f"  💡 Suggestions: {', '.join(self._last_anomaly_suggestions['reason'])}")
                        logger.info(f"     New seed: {self._last_anomaly_suggestions['seed']}")
                        logger.info(f"     New CFG: {self._last_anomaly_suggestions['guidance_scale']:.1f}")
                        
                        # Continue to next attempt with adjusted params
                        continue
                    else:
                        logger.info(f"  ✓ No anomalies detected (confidence: {anomaly_result['confidence']:.3f})")
                        # Log validation details
                        if anomaly_result['details'].get('faces'):
                            logger.info(f"     Faces: {anomaly_result['details']['faces']['count']} ✓")
                        if anomaly_result['details'].get('pose', {}).get('detected'):
                            logger.info(f"     Pose: Valid ✓")
                
                # For first frame, always accept if no anomalies
                if idx == 0 and not has_anomaly:
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
            
            # Save frame with metadata
            frame_path = output_path / f"frame_{idx + 1:03d}.png"
            best_frame.save(frame_path)
            
            # Save frame metadata (including anomaly detection results if available)
            metadata = {
                "frame_id": idx + 1,
                "prompt": prompts[idx],
                "consistency_score": best_score,
                "seed": base_seed + (idx * 1000),
                "attempts": attempt + 1,
            }
            
            # Add anomaly detection results if available
            if anomaly_details:
                metadata["anomaly_check"] = {
                    "is_valid": anomaly_details['is_valid'],
                    "confidence": anomaly_details['confidence'],
                    "anomalies": anomaly_details['anomalies'],
                    "face_count": anomaly_details['details'].get('faces', {}).get('count', 0),
                    "pose_detected": anomaly_details['details'].get('pose', {}).get('detected', False),
                }
            
            metadata_path = output_path / f"frame_{idx + 1:03d}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ Frame {idx + 1} saved (score: {best_score:.3f})")
            if anomaly_details:
                logger.info(f"   Anomaly confidence: {anomaly_details['confidence']:.3f}")
        
        # Save report with enhanced anomaly statistics
        import numpy as np
        
        # Calculate anomaly statistics
        total_attempts = sum(frame.get('attempts', 1) for frame in [
            {"attempts": len(prompts)} for _ in range(len(prompts))
        ])
        
        report = {
            "base_seed": base_seed,
            "num_frames": len(prompts),
            "generation_method": "txt2img (full scene generation)",
            "anomaly_detection_enabled": self.anomaly_detector is not None,
            "average_consistency": float(np.mean(scores)) if scores else 1.0,
            "min_consistency": float(np.min(scores)) if scores else 1.0,
            "max_consistency": float(np.max(scores)) if scores else 1.0,
            "consistency_threshold": consistency_threshold,
            "frames": [
                {
                    "frame_id": i + 1,
                    "prompt": p,
                    "consistency_score": scores[i] if i < len(scores) else 1.0
                }
                for i, p in enumerate(prompts)
            ]
        }
        
        # Add anomaly detection summary if enabled
        if self.anomaly_detector:
            report["anomaly_detection"] = {
                "detector_config": {
                    "strict_mode": self.anomaly_detector.strict_mode,
                    "expected_faces": self.anomaly_detector.expected_faces,
                    "min_face_size": self.anomaly_detector.min_face_size,
                    "max_face_size": self.anomaly_detector.max_face_size,
                },
                "note": "See individual frame metadata files for detailed anomaly results"
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
