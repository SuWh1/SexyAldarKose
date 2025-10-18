#!/usr/bin/env python3
"""
Enhanced Storyboard Generator with Frame-to-Frame Consistency
Using SDXL + LoRA + ControlNet + IP-Adapter + CLIP

This pipeline ensures visual consistency across 6-10 sequential frames by:
1. Character identity preservation (Aldar Köse LoRA)
2. Pose/depth consistency (ControlNet)
3. Visual reference matching (IP-Adapter)
4. Semantic consistency validation (CLIP)
5. Optional refinement pass (SDXL Refiner)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Diffusion & LoRA
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
)
from diffusers.utils import load_image

# IP-Adapter
try:
    from ip_adapter import IPAdapterXL
    IP_ADAPTER_AVAILABLE = True
except ImportError:
    IP_ADAPTER_AVAILABLE = False
    logging.warning("IP-Adapter not available. Install: pip install ip-adapter")

# ControlNet Processors
try:
    from controlnet_aux import OpenposeDetector, MidasDetector
    CONTROLNET_AUX_AVAILABLE = True
except ImportError:
    CONTROLNET_AUX_AVAILABLE = False
    logging.warning("controlnet_aux not available. Install: pip install controlnet-aux")

# CLIP for consistency scoring
from transformers import CLIPProcessor, CLIPModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StoryboardConsistencyPipeline:
    """
    Main pipeline for generating consistent storyboard frames
    """
    
    def __init__(
        self,
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lora_path: Optional[str] = None,
        controlnet_type: str = "pose",  # "pose" or "depth"
        use_ip_adapter: bool = True,
        use_refiner: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the consistency pipeline
        
        Args:
            base_model: Base SDXL model path
            lora_path: Path to Aldar Köse LoRA weights
            controlnet_type: Type of ControlNet ("pose" or "depth")
            use_ip_adapter: Whether to use IP-Adapter for visual reference
            use_refiner: Whether to use SDXL refiner for final pass
            device: Device to run on
            dtype: Data type for inference
        """
        self.device = device
        self.dtype = dtype
        self.lora_path = lora_path
        self.use_ip_adapter = use_ip_adapter and IP_ADAPTER_AVAILABLE
        self.use_refiner = use_refiner
        
        logger.info("Initializing Storyboard Consistency Pipeline...")
        
        # 1. Load base SDXL pipeline for first frame
        logger.info("Loading base SDXL model...")
        self.base_pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
        )
        self.base_pipe.to(device)
        
        # Load LoRA if provided
        if lora_path:
            logger.info(f"Loading Aldar Köse LoRA from {lora_path}...")
            self.base_pipe.load_lora_weights(lora_path)
            self.base_pipe.fuse_lora(lora_scale=0.8)  # Adjust scale as needed
        
        # Enable memory optimizations
        self.base_pipe.enable_attention_slicing()
        self.base_pipe.enable_vae_slicing()
        
        # 2. Load ControlNet
        logger.info(f"Loading ControlNet ({controlnet_type})...")
        self.controlnet_type = controlnet_type
        
        if controlnet_type == "pose":
            controlnet_model = "thibaud/controlnet-openpose-sdxl-1.0"
            if CONTROLNET_AUX_AVAILABLE:
                self.preprocessor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        elif controlnet_type == "depth":
            controlnet_model = "diffusers/controlnet-depth-sdxl-1.0"
            if CONTROLNET_AUX_AVAILABLE:
                self.preprocessor = MidasDetector.from_pretrained("lllyasviel/Annotators")
        else:
            raise ValueError(f"Unknown controlnet_type: {controlnet_type}")
        
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=dtype,
        )
        
        self.controlnet_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
        )
        self.controlnet_pipe.to(device)
        
        # Load LoRA into ControlNet pipeline as well
        if lora_path:
            self.controlnet_pipe.load_lora_weights(lora_path)
            self.controlnet_pipe.fuse_lora(lora_scale=0.8)
        
        self.controlnet_pipe.enable_attention_slicing()
        self.controlnet_pipe.enable_vae_slicing()
        
        # 3. Load IP-Adapter (optional)
        if self.use_ip_adapter:
            logger.info("Loading IP-Adapter for visual reference...")
            try:
                self.ip_adapter = IPAdapterXL(
                    self.controlnet_pipe,
                    image_encoder_path="h94/IP-Adapter",
                    ip_ckpt="ip-adapter_sdxl.bin",
                    device=device,
                )
                logger.info("IP-Adapter loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load IP-Adapter: {e}. Disabling.")
                self.use_ip_adapter = False
        
        # 4. Load CLIP for consistency scoring
        logger.info("Loading CLIP for consistency validation...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # 5. Load refiner (optional)
        if use_refiner:
            logger.info("Loading SDXL Refiner...")
            self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                torch_dtype=dtype,
                variant="fp16",
                use_safetensors=True,
            )
            self.refiner_pipe.to(device)
            self.refiner_pipe.enable_attention_slicing()
            self.refiner_pipe.enable_vae_slicing()
        
        logger.info("Pipeline initialization complete!")
    
    def extract_control_map(self, image: Image.Image) -> Image.Image:
        """
        Extract pose or depth map from image
        
        Args:
            image: Input PIL Image
            
        Returns:
            Control map (pose or depth)
        """
        if not CONTROLNET_AUX_AVAILABLE:
            logger.warning("controlnet_aux not available, returning original image")
            return image
        
        if self.controlnet_type == "pose":
            logger.debug("Extracting pose map...")
            control_map = self.preprocessor(image)
        elif self.controlnet_type == "depth":
            logger.debug("Extracting depth map...")
            control_map = self.preprocessor(image)
        else:
            control_map = image
        
        return control_map
    
    def compute_clip_similarity(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        Compute CLIP similarity between two images
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Process images
        inputs = self.clip_processor(
            images=[image1, image2],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get image features
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            image_features[0:1], 
            image_features[1:2]
        )
        
        return similarity.item()
    
    def generate_first_frame(
        self,
        prompt: str,
        negative_prompt: str = "",
        seed: int = 42,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 1024,
        height: int = 1024,
    ) -> Image.Image:
        """
        Generate the first frame of the scene
        
        Args:
            prompt: Text prompt for the scene
            negative_prompt: Negative prompt
            seed: Random seed
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            width: Image width
            height: Image height
            
        Returns:
            Generated first frame
        """
        logger.info("Generating first frame...")
        
        # Set seed for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate
        output = self.base_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        
        return output.images[0]
    
    def generate_subsequent_frame(
        self,
        prompt: str,
        previous_frame: Image.Image,
        control_map: Image.Image,
        reference_frame: Image.Image,
        seed: int,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.8,
        ip_adapter_scale: float = 0.5,
    ) -> Image.Image:
        """
        Generate subsequent frame with consistency mechanisms
        
        Args:
            prompt: Text prompt for the frame
            previous_frame: Previous frame for control extraction
            control_map: Pre-extracted control map (pose/depth)
            reference_frame: First frame for visual reference
            seed: Random seed
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            controlnet_conditioning_scale: ControlNet influence strength
            ip_adapter_scale: IP-Adapter influence strength
            
        Returns:
            Generated frame
        """
        logger.debug(f"Generating frame with seed {seed}...")
        
        # Set seed with slight variation
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate with ControlNet
        if self.use_ip_adapter:
            # Use IP-Adapter + ControlNet for maximum consistency
            logger.debug("Using IP-Adapter + ControlNet...")
            output = self.ip_adapter.generate(
                pil_image=reference_frame,
                prompt=prompt,
                negative_prompt=negative_prompt,
                control_image=control_map,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                scale=ip_adapter_scale,
                generator=generator,
            )
            image = output[0]
        else:
            # Use ControlNet only
            logger.debug("Using ControlNet only...")
            output = self.controlnet_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_map,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
            )
            image = output.images[0]
        
        return image
    
    def generate_scene_sequence(
        self,
        scene_prompts: List[str],
        base_seed: int = 42,
        negative_prompt: str = "blurry, low quality, distorted, deformed",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        consistency_threshold: float = 0.75,
        max_retries: int = 3,
        seed_interpolation: float = 0.1,
        output_dir: str = "./storyboard_output",
    ) -> List[Image.Image]:
        """
        Generate a sequence of consistent frames for a scene
        
        Args:
            scene_prompts: List of prompts for each frame
            base_seed: Base random seed for the scene
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            consistency_threshold: Minimum CLIP similarity with first frame
            max_retries: Maximum regeneration attempts
            seed_interpolation: Seed variation between frames (0.0-1.0)
            output_dir: Directory to save outputs
            
        Returns:
            List of generated frames
        """
        num_frames = len(scene_prompts)
        logger.info(f"Generating scene sequence with {num_frames} frames...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        frames = []
        control_maps = []
        consistency_scores = []
        
        # Generate first frame
        logger.info("=" * 60)
        logger.info("FRAME 1: First frame generation")
        logger.info("=" * 60)
        
        first_frame = self.generate_first_frame(
            prompt=scene_prompts[0],
            negative_prompt=negative_prompt,
            seed=base_seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        frames.append(first_frame)
        first_frame.save(output_path / "frame_001.png")
        logger.info(f"✓ Frame 1 saved to {output_path / 'frame_001.png'}")
        
        # Extract control map from first frame
        control_map = self.extract_control_map(first_frame)
        control_maps.append(control_map)
        control_map.save(output_path / "control_map_001.png")
        
        # Generate subsequent frames
        for frame_idx in range(1, num_frames):
            logger.info("=" * 60)
            logger.info(f"FRAME {frame_idx + 1}: Subsequent frame generation")
            logger.info("=" * 60)
            
            # Calculate seed with interpolation
            seed_offset = int(seed_interpolation * 1000 * frame_idx)
            current_seed = base_seed + seed_offset
            logger.info(f"Using seed: {current_seed} (base: {base_seed}, offset: {seed_offset})")
            
            # Get previous frame and its control map
            previous_frame = frames[frame_idx - 1]
            previous_control_map = control_maps[frame_idx - 1]
            
            # Generate frame with retries for consistency
            best_frame = None
            best_similarity = 0.0
            
            for attempt in range(max_retries):
                logger.info(f"Attempt {attempt + 1}/{max_retries}...")
                
                # Generate frame
                generated_frame = self.generate_subsequent_frame(
                    prompt=scene_prompts[frame_idx],
                    previous_frame=previous_frame,
                    control_map=previous_control_map,
                    reference_frame=first_frame,
                    seed=current_seed + attempt,  # Vary seed slightly on retries
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
                
                # Compute consistency score with first frame
                similarity = self.compute_clip_similarity(first_frame, generated_frame)
                logger.info(f"  CLIP similarity with frame 1: {similarity:.3f}")
                
                # Keep best frame
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_frame = generated_frame
                
                # Check if meets threshold
                if similarity >= consistency_threshold:
                    logger.info(f"  ✓ Meets consistency threshold ({consistency_threshold})")
                    break
                else:
                    logger.warning(f"  ✗ Below threshold, regenerating...")
            
            # Use best frame even if didn't meet threshold
            final_frame = best_frame
            frames.append(final_frame)
            consistency_scores.append(best_similarity)
            
            # Save frame
            frame_filename = f"frame_{frame_idx + 1:03d}.png"
            final_frame.save(output_path / frame_filename)
            logger.info(f"✓ Frame {frame_idx + 1} saved (similarity: {best_similarity:.3f})")
            
            # Extract control map for next frame
            control_map = self.extract_control_map(final_frame)
            control_maps.append(control_map)
            control_map.save(output_path / f"control_map_{frame_idx + 1:03d}.png")
        
        # Save consistency report
        self._save_consistency_report(
            output_path, 
            scene_prompts, 
            consistency_scores,
            base_seed,
        )
        
        # Optional: Refiner pass
        if self.use_refiner:
            logger.info("=" * 60)
            logger.info("REFINER PASS: Harmonizing frames")
            logger.info("=" * 60)
            frames = self._refiner_pass(frames, output_path)
        
        logger.info("=" * 60)
        logger.info(f"✓ Scene generation complete! {num_frames} frames saved to {output_path}")
        logger.info("=" * 60)
        
        return frames
    
    def _refiner_pass(
        self,
        frames: List[Image.Image],
        output_path: Path,
    ) -> List[Image.Image]:
        """
        Apply SDXL refiner to harmonize lighting and color
        
        Args:
            frames: List of generated frames
            output_path: Path to save refined frames
            
        Returns:
            Refined frames
        """
        refined_frames = []
        refine_dir = output_path / "refined"
        refine_dir.mkdir(exist_ok=True)
        
        for idx, frame in enumerate(tqdm(frames, desc="Refining frames")):
            logger.info(f"Refining frame {idx + 1}/{len(frames)}...")
            
            # Refine with high strength for color/lighting harmony
            refined = self.refiner_pipe(
                image=frame,
                strength=0.3,  # Light touch to preserve content
                num_inference_steps=20,
            ).images[0]
            
            refined_frames.append(refined)
            refined.save(refine_dir / f"frame_{idx + 1:03d}_refined.png")
        
        logger.info(f"✓ Refined frames saved to {refine_dir}")
        return refined_frames
    
    def _save_consistency_report(
        self,
        output_path: Path,
        prompts: List[str],
        scores: List[float],
        base_seed: int,
    ):
        """Save consistency analysis report"""
        report = {
            "base_seed": base_seed,
            "num_frames": len(prompts),
            "average_consistency": float(np.mean(scores)) if scores else 1.0,
            "min_consistency": float(np.min(scores)) if scores else 1.0,
            "max_consistency": float(np.max(scores)) if scores else 1.0,
            "frames": [
                {
                    "frame_id": idx + 1,
                    "prompt": prompt,
                    "consistency_score": score if idx < len(scores) else 1.0,
                }
                for idx, (prompt, score) in enumerate(zip(prompts, [1.0] + scores))
            ]
        }
        
        with open(output_path / "consistency_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✓ Consistency report saved")
        logger.info(f"  Average consistency: {report['average_consistency']:.3f}")
        logger.info(f"  Min: {report['min_consistency']:.3f}, Max: {report['max_consistency']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate consistent storyboard frames with SDXL + LoRA + ControlNet + IP-Adapter"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="outputs/aldar_kose_lora/final",
        help="Path to Aldar Köse LoRA weights",
    )
    parser.add_argument(
        "--controlnet-type",
        type=str,
        choices=["pose", "depth"],
        default="pose",
        help="Type of ControlNet to use",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        required=True,
        help="JSON file containing scene prompts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./storyboard_output",
        help="Output directory for frames",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--consistency-threshold",
        type=float,
        default=0.75,
        help="Minimum CLIP similarity threshold",
    )
    parser.add_argument(
        "--use-ip-adapter",
        action="store_true",
        help="Enable IP-Adapter for visual reference",
    )
    parser.add_argument(
        "--use-refiner",
        action="store_true",
        help="Enable SDXL refiner pass",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    
    args = parser.parse_args()
    
    # Load prompts
    with open(args.prompts_file, 'r') as f:
        data = json.load(f)
        scene_prompts = data.get("prompts", [])
    
    if not scene_prompts:
        logger.error("No prompts found in file!")
        return
    
    # Initialize pipeline
    pipeline = StoryboardConsistencyPipeline(
        lora_path=args.lora_path,
        controlnet_type=args.controlnet_type,
        use_ip_adapter=args.use_ip_adapter,
        use_refiner=args.use_refiner,
    )
    
    # Generate sequence
    frames = pipeline.generate_scene_sequence(
        scene_prompts=scene_prompts,
        base_seed=args.base_seed,
        num_inference_steps=args.steps,
        consistency_threshold=args.consistency_threshold,
        output_dir=args.output_dir,
    )
    
    logger.info(f"✓ Generated {len(frames)} frames successfully!")


if __name__ == "__main__":
    main()
