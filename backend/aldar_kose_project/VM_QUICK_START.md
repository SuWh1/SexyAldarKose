# 🚀 VM Training Quick Start Card

**One-page reference for experienced users**

---

## ⚡ 5-Minute Setup

```bash
# 1. Clone repo
git clone https://github.com/SuWh1/SexyAldarKose.git
cd SexyAldarKose/backend/aldar_kose_project

# 2. Install PyTorch + CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure accelerate
accelerate config default

# 5. Upload data to data/images/ and data/captions/

# 6. Prepare dataset
python scripts/prepare_dataset.py --resize --resolution 512

# 7. Pre-encode latents (RECOMMENDED)
python scripts/preprocess_latents.py

# 8. Start training in tmux
tmux new -s training
accelerate launch scripts/train_lora_sdxl.py
# Ctrl+B, D to detach
```

---

## 📊 Key Configuration (configs/training_config.yaml)

```yaml
# GPU Memory Settings
resolution: 512           # 8GB GPU | 1024 for 16GB+
lora_rank: 16            # 8GB GPU | 32 for 16GB+
train_text_encoder: false # Keep false for 8GB

# Training Duration
max_steps: 2000          # ~45 min on T4 | 3000 for production

# Memory Optimizations (DO NOT CHANGE)
gradient_checkpointing: true
use_8bit_adam: true
enable_cpu_offload: true
mixed_precision: "fp16"
```

---

## 🎯 Essential Commands

```bash
# Check GPU
nvidia-smi

# Monitor GPU
watch -n 1 nvidia-smi

# View training progress
tail -f outputs/aldar_kose_lora/training_metrics.csv

# Reattach to tmux
tmux attach -t training

# Download model (from local machine)
scp -r user@vm-ip:~/path/to/outputs/aldar_kose_lora/final .
```

---

## 🔧 Common Issues

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `resolution: 512`, `lora_rank: 8` |
| Training slow | Run `python scripts/preprocess_latents.py` |
| No images found | Upload to `data/images/` and `data/captions/` |
| Training crashes | Check disk space: `df -h` |

---

## ✅ Success Checklist

- [ ] `nvidia-smi` shows GPU
- [ ] PyTorch sees CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] 32 images in `data/images/`
- [ ] 32 captions in `data/captions/`
- [ ] Dataset validated (no errors)
- [ ] Latents pre-encoded
- [ ] Training started in tmux
- [ ] GPU at 90%+ utilization
- [ ] Loss decreasing

---

## 📥 Expected Outputs

```
outputs/aldar_kose_lora/
├── final/                        # Your trained model (600MB)
│   └── unet_lora/
│       └── adapter_model.safetensors
├── training_metrics.csv          # Loss history
├── training_summary.txt          # Final summary
└── validation_images/            # Test generations
    ├── step-50/
    └── ...
```

---

## 💰 Cost Estimates

| Provider | GPU | Price/hr | 2000 steps | Total |
|----------|-----|----------|------------|-------|
| Vast.ai | RTX 4090 | $0.30 | 28 min | **$0.14** |
| RunPod | RTX 4090 | $0.69 | 28 min | $0.32 |
| Lambda | A100 | $1.10 | 19 min | $0.35 |
| Colab Pro | T4/A100 | $10/mo | 75 min | **$0.00** |

**Tip:** Use Colab Pro for multiple experiments ($10/month unlimited)

---

## 🎯 Performance Targets

- **Steps/second:** 0.4-2.0 (depending on GPU)
- **GPU Utilization:** 90-100%
- **VRAM Usage:** 6-8GB / 15GB
- **Temperature:** 60-80°C
- **Training Time:** 20-75 minutes (2000 steps)

---

## 🆘 Emergency Commands

```bash
# Kill stuck training
pkill -f train_lora_sdxl

# Free disk space
rm -rf outputs/checkpoints/checkpoint-[0-9]*

# Reset GPU
nvidia-smi --gpu-reset

# Resume from checkpoint
# Edit configs/training_config.yaml:
# resume_from_checkpoint: "outputs/checkpoints/checkpoint-1000"
```

---

**Full Guide:** See `VM_TRAINING_GUIDE.md` for detailed instructions

**Version:** 1.0 | **Date:** Oct 18, 2025
