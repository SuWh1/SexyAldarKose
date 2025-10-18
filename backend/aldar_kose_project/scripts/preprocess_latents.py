#!/usr/bin/env python3
"""
Pre-encode images to latents for faster training

This script pre-encodes all training images to latent space using the VAE,
which saves significant GPU memory during training. Instead of encoding
images on-the-fly during training, we use the cached latents.

Usage:
    python scripts/preprocess_latents.py --config configs/training_config.yaml
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict
import yaml
import torch
import logging

from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_manifest(manifest_path: str) -> Dict:
    """Load dataset manifest"""
    with open(manifest_path, 'r') as f:
        return json.load(f)


def preprocess_latents(config, manifest_path: str = None):
    """Pre-encode all images to latents"""
    
    # Load manifest
    if manifest_path is None:
        if Path('data/dataset_manifest_processed.json').exists():
            manifest_path = 'data/dataset_manifest_processed.json'
        else:
            manifest_path = 'data/dataset_manifest.json'
    
    logger.info(f"Loading manifest from {manifest_path}")
    manifest = load_manifest(manifest_path)
    
    # Create latents directory
    latents_dir = Path(config['output_dir']) / 'latents'
    latents_dir.mkdir(exist_ok=True, parents=True)
    
    # Load VAE
    logger.info("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        config.get('vae_model') or config['base_model'],
        subfolder="vae" if not config.get('vae_model') else None,
    )
    vae = vae.to('cuda', dtype=torch.float16)
    vae.eval()
    
    # Transforms
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize(config['resolution'], interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(config['resolution']) if config.get('center_crop', True) else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # Process each image
    logger.info(f"Processing {len(manifest['samples'])} images to latents...")
    
    for idx, sample in enumerate(tqdm(manifest['samples'], desc="Encoding to latents")):
        # Get image path
        image_path = sample.get('processed_path') or sample['image_path']
        
        # Skip if latent already exists
        latent_path = latents_dir / f"{Path(image_path).stem}_latent.pt"
        if latent_path.exists():
            sample['latent_path'] = str(latent_path)
            continue
        
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to('cuda', dtype=torch.float16)
            
            # Encode to latent
            with torch.no_grad():
                latent = vae.encode(image).latent_dist.sample()
                latent = latent * vae.config.scaling_factor
            
            # Remove batch dimension for single image saving
            latent = latent.squeeze(0)
            
            # Save latent
            torch.save(latent.cpu(), str(latent_path))
            sample['latent_path'] = str(latent_path)
            
        except Exception as e:
            logger.warning(f"Failed to process {image_path}: {e}")
            continue
    
    # Save updated manifest with latent paths
    updated_manifest_path = Path(config['output_dir']) / 'dataset_manifest_with_latents.json'
    with open(updated_manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Latents saved to {latents_dir}")
    logger.info(f"Updated manifest saved to {updated_manifest_path}")
    logger.info("Pre-processing complete! Now you can train with latents.")


def main():
    parser = argparse.ArgumentParser(description="Pre-encode images to latents")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to dataset manifest (auto-detected if not provided)",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Pre-process
    preprocess_latents(config, args.manifest)


if __name__ == "__main__":
    main()
