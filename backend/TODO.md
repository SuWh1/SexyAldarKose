You are an ML engineer setting up the initial project structure for a custom SDXL fine-tuning pipeline to reproduce the 3D animated Kazakh character Aldar Kose with high identity consistency.

OBJECTIVE
Generate a complete initial setup (no training yet) for a diffusion-based project that will later be used for LoRA fine-tuning, ControlNet conditioning, and evaluation.

ENVIRONMENT
- Local machine: RTX 4060 (8 GB VRAM), 16 GB RAM
- Python 3.10+
- Libraries: PyTorch >=2.0, diffusers, transformers, accelerate, bitsandbytes, wandb, opencv-python, Pillow
- Base model: stabilityai/stable-diffusion-xl-base-1.0

REQUIREMENTS FOR OUTPUT
Produce a ready-to-run Python project scaffold containing the files and contents described below. All scripts should be runnable from the repository root and include helpful CLI usage comments.

1) DIRECTORY STRUCTURE (create exactly)
aldar_kose_project/
├── data/
│   ├── images/          # training images (user will put files here)
│   └── captions/        # paired plain-text captions named like image files
├── configs/
│   └── training_config.yaml
├── scripts/
│   ├── setup_environment.py
│   ├── prepare_dataset.py
│   └── train_lora_sdxl.py
├── logs/
│   └── wandb/           # default wandb log dir
├── outputs/
│   └── checkpoints/
└── README.md
Also return a requirements.txt.

2) setup_environment.py
- Check GPU availability and print CUDA + torch CUDA device info.
- Install missing Python packages if not present (safe: print pip install commands rather than running them).
- Print recommended accelerate launch command.
- Exit with helpful error messages if GPU not found.

3) prepare_dataset.py
- Create data/images and data/captions if missing.
- Verify every image file in data/images has a corresponding caption file in data/captions with same basename and .txt extension. Print errors for mismatches.
- Optionally offer to resize images to the resolution specified in configs/training_config.yaml (default 768 or 1024), and save processed copies to data/processed_images/.
- Produce a small sample csv or json summary manifest listing: filename, caption, width, height.

4) configs/training_config.yaml
Include default keys and sensible defaults:
base_model: "stabilityai/stable-diffusion-xl-base-1.0"
resolution: 768
batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1e-4
max_steps: 2000
save_every: 250
precision: "fp16"
output_dir: "./outputs/aldar_kose_lora"
wandb_project: "aldar_kose_finetune"
seed: 42

5) train_lora_sdxl.py (skeleton, ready to run with minor edits)
- Load configs from configs/training_config.yaml.
- Verify dataset via prepare_dataset.py step or manifest.
- Load SDXL base model components (UNet, text encoder, VAE) with diffusers.
- Show how to wrap UNet/text encoder with LoRA adapters (placeholders) and where to insert the training loop. Use memory-saver options (enable fp16, enable_attention_slicing, enable gradient checkpointing suggestions).
- Setup AdamW optimizer (show bitsandbytes 8-bit usage if available).
- Add training loop pseudocode: batch loading, tokenization of captions, forward pass, loss logging, backward, optimizer step, checkpoint saving every save_every steps.
- Integrate WandB init and basic logging (loss, step, sample generation at intervals). If WandB API key missing, print instructions.
- Provide an inference snippet at the end to load saved LoRA and generate one sample image for verification.

6) Logging & checkpoints
- Save checkpoints to outputs/checkpoints with step number.
- Save a prompts.txt and seeds.txt alongside each checkpoint.
- Log CLIP-based identity similarity in WandB as an optional metric (sketch code showing how to compute it).

7) README.md
- Short project description.
- Setup steps (run setup_environment.py, prepare_dataset.py).
- Example commands:
  python scripts/setup_environment.py
  python scripts/prepare_dataset.py
  accelerate launch scripts/train_lora_sdxl.py
- Explain where to place images/captions and how captions should be formatted (one-line descriptive caption using trigger token e.g., "3D render of aldar_kose_man smiling").

8) requirements.txt
List minimal pinned versions or commonly compatible versions:
torch
transformers
diffusers
accelerate
bitsandbytes
safetensors
wandb
opencv-python
Pillow
PyYAML
tqdm
ftfy

DELIVERABLE
Return the full text contents for all files listed above (scripts, config, README, requirements.txt) in this single response. Keep code runnable and well-commented; do not perform training in this step. Make the scripts modular and easy to extend for DreamBooth, ControlNet, or training on a cloud GPU later.
