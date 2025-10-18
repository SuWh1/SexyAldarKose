# Aldar Kose SDXL Fine-tuning Project

A complete pipeline for fine-tuning Stable Diffusion XL (SDXL) to generate the 3D animated Kazakh character **Aldar Kose** with high identity consistency using LoRA (Low-Rank Adaptation).

## 🎯 Project Overview

This project implements a custom training pipeline for:
- **LoRA fine-tuning** on SDXL for character-specific image generation
- **Identity consistency** through proper dataset preparation and evaluation
- **Memory-efficient training** optimized for RTX 4060 (8GB VRAM)
- **Comprehensive monitoring** with WandB integration
- **Automated validation** and quality metrics

## 📁 Project Structure

```
aldar_kose_project/
├── data/
│   ├── images/              # Training images (place your images here)
│   ├── captions/            # Text captions for each image
│   ├── processed_images/    # Preprocessed images (auto-generated)
│   └── dataset_manifest.json  # Dataset metadata (auto-generated)
├── configs/
│   └── training_config.yaml # Training configuration
├── scripts/
│   ├── setup_environment.py  # Environment validation
│   ├── prepare_dataset.py    # Dataset preprocessing
│   ├── train_lora_sdxl.py    # Main training script
│   ├── inference.py          # Image generation
│   └── evaluate_identity.py  # Identity consistency evaluation
├── logs/
│   └── wandb/               # Training logs
├── outputs/
│   ├── checkpoints/         # Training checkpoints
│   └── generated_images/    # Generated samples
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Environment Setup

⚠️ **Important:** Due to PyTorch dependency conflicts, use the automated installer or follow the step-by-step guide in [INSTALLATION.md](INSTALLATION.md).

**Option 1: Automated Install (Recommended)**

```powershell
.\install.ps1
```

**Option 2: Manual Install**

See [INSTALLATION.md](INSTALLATION.md) for detailed instructions.

**Option 3: Simple Install (if you know your CUDA version)**

```powershell
# Step 1: Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install other packages
pip install transformers diffusers accelerate peft bitsandbytes safetensors opencv-python Pillow PyYAML datasets wandb tensorboard tqdm ftfy numpy scipy openai python-dotenv huggingface_hub
```

Verify your environment:

```bash
python scripts/setup_environment.py
```

This will check:
- ✅ Python version (3.10+)
- ✅ GPU and CUDA availability
- ✅ Required packages
- ✅ Directory structure

### 2. Prepare Your Dataset

See **[DATA_FORMAT.md](DATA_FORMAT.md)** for detailed instructions on preparing your training data.

#### Option A: Automated Labeling (Recommended)

Use the automated labeling pipeline with OpenAI Vision API:

```bash
# Set your OpenAI API key
$env:OPENAI_API_KEY="your-api-key-here"

# Label all images automatically
python scripts/label_images.py --input_dir raw_images/
```

See **[LABELING_GUIDE.md](LABELING_GUIDE.md)** for complete automated labeling instructions.

#### Option B: Manual Labeling

**Quick summary:**
1. Place training images in `data/images/`
2. Create corresponding caption files in `data/captions/`
3. Each image needs a matching `.txt` file with the same name

Example:
```
data/images/aldar001.jpg  →  data/captions/aldar001.txt
data/images/aldar002.png  →  data/captions/aldar002.txt
```

Run the dataset preparation script:

```bash
# Validate dataset
python scripts/prepare_dataset.py

# Validate and resize images
python scripts/prepare_dataset.py --resize --resolution 1024
```

### 3. Configure Training

Edit `configs/training_config.yaml` to adjust:
- Learning rate and batch size
- LoRA rank and alpha
- Number of training steps
- Validation prompts

**Note:** WandB is disabled by default. Training logs are saved locally:
- Terminal output with real-time metrics
- CSV file: `outputs/aldar_kose_lora/training_metrics.csv`
- Validation images: `outputs/aldar_kose_lora/validation_images/`
- Training summary: `outputs/aldar_kose_lora/training_summary.txt`

See [LOCAL_LOGGING_GUIDE.md](LOCAL_LOGGING_GUIDE.md) for details.

### 4. Configure Accelerate

Set up Hugging Face Accelerate for optimized training:

```bash
accelerate config
```

**Recommended settings for RTX 4060 (8GB VRAM):**
- Compute environment: This machine
- Machine type: No distributed training
- Use DeepSpeed: No
- GPU(s): 1
- Mixed precision: fp16
- Gradient accumulation steps: 4

### 5. Start Training

Launch training with accelerate:

```bash
accelerate launch scripts/train_lora_sdxl.py
```

Monitor training progress:
- Console output shows loss and learning rate
- WandB dashboard (if enabled) shows detailed metrics
- Validation images generated every N steps
- Checkpoints saved to `outputs/checkpoints/`

## 🎨 Generate Images

After training, generate images with your LoRA:

```bash
# Generate a single image
python scripts/inference.py \
  --checkpoint outputs/checkpoints/checkpoint-1000 \
  --prompt "3D render of aldar_kose_man smiling, high quality"

# Generate multiple images
python scripts/inference.py \
  --checkpoint outputs/final \
  --prompt "aldar_kose_man in traditional Kazakh clothing" \
  --num_images 4 \
  --seed 42
```

**Generation parameters:**
- `--checkpoint`: Path to saved LoRA checkpoint
- `--prompt`: Text description of desired image
- `--negative_prompt`: What to avoid (default: "low quality, blurry")
- `--num_images`: Number of images to generate
- `--num_inference_steps`: Denoising steps (default: 30)
- `--guidance_scale`: CFG scale (default: 7.5)
- `--seed`: Random seed for reproducibility

## 📊 Evaluate Identity Consistency

Measure how well your model maintains character identity:

```bash
python scripts/evaluate_identity.py \
  --reference_image data/images/reference.jpg \
  --generated_dir outputs/generated_images
```

This computes CLIP-based similarity scores between a reference image and all generated images.

**Interpretation:**
- **> 0.85**: Excellent identity consistency ✅
- **0.75-0.85**: Good identity consistency ✓
- **0.65-0.75**: Moderate identity consistency ~
- **< 0.65**: Low consistency - consider more training ⚠

## ⚙️ Configuration Options

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resolution` | 1024 | Image resolution (SDXL native) |
| `batch_size` | 1 | Batch size per GPU |
| `gradient_accumulation_steps` | 4 | Effective batch size = 4 |
| `learning_rate` | 1e-4 | UNet learning rate |
| `lora_rank` | 32 | LoRA rank (higher = more capacity) |
| `max_steps` | 2000 | Total training steps |
| `save_every` | 250 | Checkpoint interval |
| `mixed_precision` | fp16 | Use FP16 for memory efficiency |

### Memory Optimization

For 8GB VRAM, these settings are crucial:
- ✅ `gradient_checkpointing: true` - Reduces memory usage
- ✅ `enable_xformers: true` - Memory-efficient attention
- ✅ `use_8bit_adam: true` - 8-bit optimizer
- ✅ `mixed_precision: "fp16"` - Half precision training

## 📈 Monitoring & Logging

### WandB Integration

Configure in `training_config.yaml`:
```yaml
use_wandb: true
wandb_project: "aldar_kose_finetune"
wandb_entity: "your-username"
```

Login to WandB:
```bash
wandb login
```

### Metrics Tracked

- Training loss
- Learning rate
- Validation images
- Identity similarity scores (optional)
- GPU memory usage
- Training throughput (steps/sec)

## 🔧 Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:
1. Reduce `batch_size` to 1
2. Increase `gradient_accumulation_steps`
3. Lower `lora_rank` (e.g., 16 or 8)
4. Disable text encoder training: `train_text_encoder: false`
5. Reduce `resolution` to 768

### Low Quality Generations

To improve quality:
1. Train for more steps (e.g., 3000-5000)
2. Increase LoRA rank (e.g., 64)
3. Use more diverse training data
4. Adjust learning rate (try 5e-5 or 2e-4)
5. Enable `train_text_encoder: true`

### Inconsistent Identity

To improve identity consistency:
1. Use consistent trigger token in all captions
2. Add more training images (15-30 recommended)
3. Disable `random_flip` if character is asymmetric
4. Train text encoder with lower LR
5. Use higher LoRA rank

## 📚 Advanced Usage

### Resume from Checkpoint

```yaml
# In training_config.yaml
resume_from_checkpoint: "outputs/checkpoints/checkpoint-500"
```

### Custom Validation Prompts

```yaml
validation_prompts:
  - "3D render of aldar_kose_man smiling"
  - "aldar_kose_man in winter clothes"
  - "portrait of aldar_kose_man, dramatic lighting"
  - "aldar_kose_man riding a horse"
```

### Multi-GPU Training

Configure with accelerate:
```bash
accelerate config  # Select multi-GPU training
accelerate launch scripts/train_lora_sdxl.py
```

## 🔗 Resources

- [SDXL Paper](https://arxiv.org/abs/2307.01952)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [WandB Documentation](https://docs.wandb.ai/)

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{aldar_kose_sdxl,
  title={Aldar Kose SDXL Fine-tuning Pipeline},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/yourusername/aldar-kose-sdxl}
}
```

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ❓ Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Consult the troubleshooting section above

---

**Happy Training! 🎉**

For more details on data preparation, see [DATA_FORMAT.md](DATA_FORMAT.md)
