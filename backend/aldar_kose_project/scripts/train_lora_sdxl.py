#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for SDXL - Aldar Kose Character

This script trains a LoRA adapter on SDXL for character consistency.
Optimized for RTX 4060 (8GB VRAM) with gradient checkpointing and mixed precision.

Usage:
    accelerate launch scripts/train_lora_sdxl.py
    accelerate launch scripts/train_lora_sdxl.py --config configs/training_config.yaml
"""

import argparse
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
import yaml

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image

import wandb


logger = get_logger(__name__, log_level="INFO")


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SDXL LoRA fine-tuning for Aldar Kose")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to dataset manifest JSON (auto-detected if not provided)",
    )
    
    args = parser.parse_args()
    return args


class AldarKoseDataset(Dataset):
    """Custom dataset for loading image-caption pairs"""
    
    def __init__(
        self,
        manifest_path: str,
        tokenizer_one: CLIPTokenizer,
        tokenizer_two: CLIPTokenizer,
        resolution: int = 1024,
        center_crop: bool = True,
        random_flip: bool = False,
        use_processed: bool = False,
    ):
        super().__init__()
        
        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        self.samples = manifest['samples']
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.resolution = resolution
        self.center_crop = center_crop
        self.use_processed = use_processed
        
        # Image transforms
        self.transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def tokenize_captions(self, caption: str):
        """Tokenize caption with both CLIP tokenizers"""
        # Tokenize with first tokenizer (OpenCLIP)
        inputs_one = self.tokenizer_one(
            caption,
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize with second tokenizer (OpenCLIP-L)
        inputs_two = self.tokenizer_two(
            caption,
            padding="max_length",
            max_length=self.tokenizer_two.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        return inputs_one.input_ids[0], inputs_two.input_ids[0]
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Check if we have pre-encoded latents
        if 'latent_path' in sample:
            # Load pre-encoded latent
            latent = torch.load(sample['latent_path'])
        else:
            # Fallback: load image and encode on-the-fly (slow, high VRAM)
            image_path = sample.get('processed_path') if self.use_processed else sample['image_path']
            image = Image.open(image_path).convert('RGB')
            image = self.transforms(image)
            latent = image  # Return as latent (will be encoded during training)
        
        # Tokenize caption
        caption = sample['caption']
        input_ids_one, input_ids_two = self.tokenize_captions(caption)
        
        return {
            'latent': latent if 'latent_path' in sample else None,
            'pixel_values': latent if 'latent_path' not in sample else None,
            'input_ids_one': input_ids_one,
            'input_ids_two': input_ids_two,
            'caption': caption,
        }


def collate_fn(examples):
    """Collate function for DataLoader"""
    # Check if using pre-encoded latents or raw images
    if examples[0].get('latent') is not None:
        # Pre-encoded latents - stack them into batch
        latents = torch.stack([example['latent'].squeeze(0) if example['latent'].dim() > 3 else example['latent'] for example in examples])
        return {
            'latents': latents.float(),
            'input_ids_one': torch.stack([example['input_ids_one'] for example in examples]),
            'input_ids_two': torch.stack([example['input_ids_two'] for example in examples]),
        }
    else:
        # Raw images (fallback)
        pixel_values = torch.stack([example['pixel_values'] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        
        return {
            'pixel_values': pixel_values,
            'input_ids_one': torch.stack([example['input_ids_one'] for example in examples]),
            'input_ids_two': torch.stack([example['input_ids_two'] for example in examples]),
        }


def encode_prompt(text_encoders, tokenizers, prompt, device):
    """Encode prompt with both text encoders"""
    prompt_embeds_list = []
    
    for text_encoder, tokenizer in zip(text_encoders, tokenizers):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids.to(device)
        
        prompt_embeds = text_encoder(
            text_input_ids,
            output_hidden_states=True,
        )
        
        # Use pooled output for second encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        
        prompt_embeds_list.append(prompt_embeds)
    
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    return prompt_embeds, pooled_prompt_embeds


def compute_time_ids(original_size, crops_coords_top_left, target_size, dtype, device):
    """Compute time embeddings for SDXL"""
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    return add_time_ids


def generate_validation_images(
    pipeline,
    prompts: List[str],
    seeds: List[int],
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
):
    """Generate validation images"""
    images = []
    
    for prompt, seed in zip(prompts, seeds):
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        
        image = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        images.append(image)
    
    return images


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=config['output_dir'],
        logging_dir=os.path.join(config['output_dir'], "logs"),
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        mixed_precision=config['mixed_precision'],
        log_with="wandb" if config.get('use_wandb', False) else None,
        project_config=accelerator_project_config,
    )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Set seed for reproducibility
    if config.get('seed') is not None:
        set_seed(config['seed'])
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Initialize CSV logging for metrics
    metrics_file = None
    if accelerator.is_main_process:
        metrics_file = Path(config['output_dir']) / "training_metrics.csv"
        with open(metrics_file, 'w') as f:
            f.write("step,loss,learning_rate,epoch\n")
        logger.info(f"Logging metrics to: {metrics_file}")
    
    # Initialize WandB
    if accelerator.is_main_process and config.get('use_wandb', False):
        wandb.init(
            project=config.get('wandb_project', 'aldar_kose_finetune'),
            name=config.get('wandb_run_name'),
            config=config,
        )
    
    # Load models
    logger.info("Loading SDXL models...")
    
    # Load tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        config['base_model'],
        subfolder="tokenizer",
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        config['base_model'],
        subfolder="tokenizer_2",
    )
    
    # Load text encoders
    text_encoder_one = CLIPTextModel.from_pretrained(
        config['base_model'],
        subfolder="text_encoder",
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        config['base_model'],
        subfolder="text_encoder_2",
    )
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        config.get('vae_model') or config['base_model'],
        subfolder="vae" if not config.get('vae_model') else None,
    )
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        config['base_model'],
        subfolder="unet",
    )
    
    # Freeze VAE
    vae.requires_grad_(False)
    
    # Enable memory optimizations
    if config.get('enable_xformers', False):
        try:
            unet.enable_xformers_memory_efficient_attention()
            vae.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
    
    if config.get('gradient_checkpointing', False):
        unet.enable_gradient_checkpointing()
        if config.get('train_text_encoder', False):
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    
    # Enable CPU offloading to save VRAM
    if config.get('enable_cpu_offload', False):
        try:
            unet.enable_attention_slicing()  # First try attention slicing
            logger.info("Enabled attention slicing for memory efficiency")
        except:
            pass
        try:
            # Use enable_sequential_cpu_offload for UNet
            unet.enable_sequential_cpu_offload()
            logger.info("Enabled sequential CPU offloading for UNet")
        except Exception as e:
            logger.warning(f"Could not enable CPU offload: {e}")
    
    # Setup LoRA
    logger.info("Setting up LoRA adapters...")
    
    lora_config = LoraConfig(
        r=config['lora_rank'],
        lora_alpha=config['lora_alpha'],
        target_modules=config['lora_target_modules'],
        lora_dropout=config['lora_dropout'],
        bias="none",
    )
    
    # Add LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Add LoRA to text encoders if training them
    if config.get('train_text_encoder', False):
        # Text encoders use different module names than UNet
        # Use the generic "c_proj" and "q_proj" modules found in transformer text encoders
        text_encoder_lora_config = LoraConfig(
            r=config['lora_rank'],
            lora_alpha=config['lora_alpha'],
            target_modules=["q_proj", "v_proj"],  # Common in CLIP models
            lora_dropout=config['lora_dropout'],
            bias="none",
        )
        text_encoder_one = get_peft_model(text_encoder_one, text_encoder_lora_config)
        text_encoder_two = get_peft_model(text_encoder_two, text_encoder_lora_config)
        logger.info("Added LoRA to text encoders")
    else:
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
    
    # Setup optimizer
    logger.info("Setting up optimizer...")
    
    # Collect parameters to optimize
    params_to_optimize = list(unet.parameters())
    
    if config.get('train_text_encoder', False):
        params_to_optimize += list(text_encoder_one.parameters())
        params_to_optimize += list(text_encoder_two.parameters())
    
    # Use 8-bit Adam if available
    if config.get('use_8bit_adam', False):
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            logger.info("Using 8-bit AdamW optimizer")
        except ImportError:
            optimizer_class = torch.optim.AdamW
            logger.warning("bitsandbytes not available, using standard AdamW")
    else:
        optimizer_class = torch.optim.AdamW
    
    optimizer = optimizer_class(
        params_to_optimize,
        lr=config['learning_rate'],
        betas=(config.get('adam_beta1', 0.9), config.get('adam_beta2', 0.999)),
        weight_decay=config.get('adam_weight_decay', 0.01),
        eps=config.get('adam_epsilon', 1e-8),
    )
    
    # Setup noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        config['base_model'],
        subfolder="scheduler",
    )
    
    # Prepare dataset
    logger.info("Loading dataset...")
    
    # Auto-detect manifest if not provided
    manifest_path = args.manifest
    if manifest_path is None:
        # Prefer manifest with pre-encoded latents
        if Path('outputs/aldar_kose_lora/dataset_manifest_with_latents.json').exists():
            manifest_path = 'outputs/aldar_kose_lora/dataset_manifest_with_latents.json'
            logger.info("Using pre-encoded latents for faster training!")
            use_processed = True
        elif Path('data/dataset_manifest_processed.json').exists():
            manifest_path = 'data/dataset_manifest_processed.json'
            use_processed = True
        elif Path('data/dataset_manifest.json').exists():
            manifest_path = 'data/dataset_manifest.json'
            use_processed = False
        else:
            raise FileNotFoundError(
                "No dataset manifest found. Please run prepare_dataset.py first."
            )
    else:
        use_processed = 'processed' in manifest_path or 'latents' in manifest_path
    
    dataset = AldarKoseDataset(
        manifest_path=manifest_path,
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        resolution=config['resolution'],
        center_crop=config.get('center_crop', True),
        random_flip=config.get('random_flip', False),
        use_processed=use_processed,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.get('num_workers', 0),
    )
    
    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / config['gradient_accumulation_steps']
    )
    
    if config.get('max_train_epochs') is not None:
        max_train_steps = config['max_train_epochs'] * num_update_steps_per_epoch
    else:
        max_train_steps = config['max_steps']
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        config.get('lr_scheduler', 'constant'),
        optimizer=optimizer,
        num_warmup_steps=config.get('lr_warmup_steps', 0) * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )
    
    # Prepare everything with accelerator
    unet, text_encoder_one, text_encoder_two, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder_one, text_encoder_two, optimizer, dataloader, lr_scheduler
    )
    
    # Move VAE to device
    vae.to(accelerator.device, dtype=torch.float16 if config['precision'] == 'fp16' else torch.float32)
    
    # Calculate total batch size
    total_batch_size = (
        config['batch_size'] * accelerator.num_processes * config['gradient_accumulation_steps']
    )
    
    logger.info("***** Training Configuration *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num batches per epoch = {len(dataloader)}")
    logger.info(f"  Instantaneous batch size = {config['batch_size']}")
    logger.info(f"  Gradient accumulation steps = {config['gradient_accumulation_steps']}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    # Training loop
    global_step = 0
    first_epoch = 0
    
    # Resume from checkpoint if specified
    if config.get('resume_from_checkpoint'):
        # Implementation for checkpoint resumption would go here
        pass
    
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(first_epoch, 999999):  # Effectively infinite
        unet.train()
        if config.get('train_text_encoder', False):
            text_encoder_one.train()
            text_encoder_two.train()
        
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                # Use pre-encoded latents or encode images on-the-fly
                if 'latents' in batch:
                    # Pre-encoded latents (fast, low memory)
                    latents = batch['latents'].to(accelerator.device, dtype=torch.float16)
                else:
                    # Fallback: encode images on-the-fly (slow, high memory)
                    latents = vae.encode(batch['pixel_values'].to(dtype=vae.dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                
                # Sample timesteps
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                ).long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Encode prompts
                with torch.no_grad():
                    encoder_hidden_states_one = text_encoder_one(
                        batch['input_ids_one'],
                        output_hidden_states=True,
                    )
                    encoder_hidden_states_two = text_encoder_two(
                        batch['input_ids_two'],
                        output_hidden_states=True,
                    )
                    
                    # Concatenate embeddings
                    encoder_hidden_states = torch.cat(
                        [encoder_hidden_states_one.hidden_states[-2], 
                         encoder_hidden_states_two.hidden_states[-2]],
                        dim=-1
                    )
                    
                    # Pooled embeddings
                    pooled_embeds = encoder_hidden_states_two[0]
                
                # Prepare added embeddings
                add_time_ids = compute_time_ids(
                    (config['resolution'], config['resolution']),
                    (0, 0),
                    (config['resolution'], config['resolution']),
                    dtype=encoder_hidden_states.dtype,
                    device=latents.device,
                ).repeat(batch_size, 1)
                
                # Prepare added kwargs
                added_cond_kwargs = {
                    'text_embeds': pooled_embeds,
                    'time_ids': add_time_ids,
                }
                
                # Predict noise
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                
                # Calculate loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                # Apply SNR weighting if configured
                if config.get('snr_gamma'):
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, config['snr_gamma'] * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    loss = (loss * mse_loss_weights).mean()
                else:
                    loss = loss.mean()
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, config.get('max_grad_norm', 1.0))
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Logging
                if global_step % config.get('log_every', 10) == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                        "epoch": epoch,
                    }
                    progress_bar.set_postfix(**logs)
                    
                    # Log to terminal with color
                    if accelerator.is_main_process:
                        logger.info(
                            f"Step {global_step:05d} | "
                            f"Loss: {logs['loss']:.4f} | "
                            f"LR: {logs['lr']:.2e} | "
                            f"Epoch: {epoch}"
                        )
                        
                        # Write to CSV
                        if metrics_file:
                            with open(metrics_file, 'a') as f:
                                f.write(f"{global_step},{logs['loss']:.6f},{logs['lr']:.8f},{epoch}\n")
                    
                    if config.get('use_wandb', False) and accelerator.is_main_process:
                        wandb.log(logs, step=global_step)
                
                # Save checkpoint
                if global_step % config['save_every'] == 0:
                    if accelerator.is_main_process:
                        save_path = Path(config['checkpoint_dir']) / f"checkpoint-{global_step}"
                        save_path.mkdir(parents=True, exist_ok=True)
                        
                        # Save LoRA weights
                        unet_lora = accelerator.unwrap_model(unet)
                        unet_lora.save_pretrained(save_path / "unet_lora")
                        
                        if config.get('train_text_encoder', False):
                            text_encoder_one_lora = accelerator.unwrap_model(text_encoder_one)
                            text_encoder_two_lora = accelerator.unwrap_model(text_encoder_two)
                            text_encoder_one_lora.save_pretrained(save_path / "text_encoder_one_lora")
                            text_encoder_two_lora.save_pretrained(save_path / "text_encoder_two_lora")
                        
                        logger.info(f"Saved checkpoint to {save_path}")
                
                # Validation
                if config.get('validate_every') and global_step % config['validate_every'] == 0:
                    if accelerator.is_main_process:
                        logger.info("Generating validation images...")
                        
                        # Create pipeline for inference
                        pipeline = StableDiffusionXLPipeline.from_pretrained(
                            config['base_model'],
                            vae=vae,
                            text_encoder=accelerator.unwrap_model(text_encoder_one),
                            text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                            unet=accelerator.unwrap_model(unet),
                            torch_dtype=torch.float16 if config['precision'] == 'fp16' else torch.float32,
                        )
                        pipeline.to(accelerator.device)
                        
                        # Generate images
                        validation_images = generate_validation_images(
                            pipeline,
                            config.get('validation_prompts', []),
                            config.get('validation_seeds', []),
                        )
                        
                        # Save validation images locally
                        val_dir = Path(config['output_dir']) / "validation_images" / f"step-{global_step}"
                        val_dir.mkdir(parents=True, exist_ok=True)
                        
                        for i, (img, prompt) in enumerate(zip(validation_images, config.get('validation_prompts', []))):
                            img_path = val_dir / f"image_{i:02d}.png"
                            img.save(img_path)
                            
                            # Save prompt as text file
                            prompt_path = val_dir / f"image_{i:02d}_prompt.txt"
                            with open(prompt_path, 'w') as f:
                                f.write(prompt)
                        
                        logger.info(f"Saved {len(validation_images)} validation images to {val_dir}")
                        
                        # Log to wandb
                        if config.get('use_wandb', False):
                            wandb.log({
                                "validation": [
                                    wandb.Image(img, caption=prompt)
                                    for img, prompt in zip(validation_images, config.get('validation_prompts', []))
                                ]
                            }, step=global_step)
                        
                        del pipeline
                        torch.cuda.empty_cache()
            
            if global_step >= max_train_steps:
                break
        
        if global_step >= max_train_steps:
            break
    
    # Save final checkpoint
    if accelerator.is_main_process:
        save_path = Path(config['output_dir']) / "final"
        save_path.mkdir(parents=True, exist_ok=True)
        
        unet_lora = accelerator.unwrap_model(unet)
        unet_lora.save_pretrained(save_path / "unet_lora")
        
        if config.get('train_text_encoder', False):
            text_encoder_one_lora = accelerator.unwrap_model(text_encoder_one)
            text_encoder_two_lora = accelerator.unwrap_model(text_encoder_two)
            text_encoder_one_lora.save_pretrained(save_path / "text_encoder_one_lora")
            text_encoder_two_lora.save_pretrained(save_path / "text_encoder_two_lora")
        
        # Save training summary
        summary_file = Path(config['output_dir']) / "training_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("  TRAINING COMPLETE - Aldar Kose LoRA\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total steps: {global_step}\n")
            f.write(f"Total epochs: {epoch}\n")
            f.write(f"Final learning rate: {lr_scheduler.get_last_lr()[0]:.2e}\n")
            f.write(f"\nOutput locations:\n")
            f.write(f"  - Final model: {save_path}\n")
            f.write(f"  - Checkpoints: {config['checkpoint_dir']}\n")
            f.write(f"  - Validation images: {config['output_dir']}/validation_images\n")
            f.write(f"  - Training metrics: {config['output_dir']}/training_metrics.csv\n")
            f.write(f"\nConfiguration:\n")
            f.write(f"  - Base model: {config['base_model']}\n")
            f.write(f"  - LoRA rank: {config['lora_rank']}\n")
            f.write(f"  - Learning rate: {config['learning_rate']}\n")
            f.write(f"  - Batch size: {config['batch_size']}\n")
            f.write(f"  - Gradient accumulation: {config['gradient_accumulation_steps']}\n")
            f.write("\n" + "=" * 70 + "\n")
        
        logger.info(f"Training complete! Final model saved to {save_path}")
        logger.info(f"Training summary saved to {summary_file}")
        
        # Print summary to terminal
        print("\n" + "=" * 70)
        print("  🎉 TRAINING COMPLETE! 🎉")
        print("=" * 70)
        print(f"\n📊 Training Stats:")
        print(f"   Steps completed: {global_step}")
        print(f"   Epochs completed: {epoch}")
        print(f"\n💾 Outputs saved to:")
        print(f"   Final model: {save_path}")
        print(f"   Checkpoints: {config['checkpoint_dir']}")
        print(f"   Validation images: {config['output_dir']}/validation_images")
        print(f"   Metrics CSV: {config['output_dir']}/training_metrics.csv")
        print(f"   Summary: {summary_file}")
        print("\n" + "=" * 70 + "\n")
    
    accelerator.end_training()
    
    if config.get('use_wandb', False) and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
