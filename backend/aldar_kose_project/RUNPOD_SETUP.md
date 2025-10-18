# 🚀 RunPod Training Setup - Complete Guide

**One-page guide to get training running on RunPod in 10 minutes**

---

## 📋 Step 1: Launch RunPod Instance

1. Go to https://www.runpod.io/
2. Click **"Deploy"** → **"GPU Instances"**
3. Select GPU:
   - **Recommended:** RTX 4090 (24GB VRAM, $0.69/hr)
   - **Budget:** RTX 3090 (24GB VRAM, $0.49/hr)
   - **Minimum:** RTX 3060 (12GB VRAM, $0.34/hr)
4. Template: **"RunPod Pytorch 2.1"** (includes CUDA 12.1)
5. Storage: **50GB Container Disk** + **50GB Volume Disk**
6. Click **"Deploy On-Demand"** or **"Deploy Spot"** (cheaper)
7. Wait ~2 minutes for pod to start
8. Click **"Connect"** → **"Start Web Terminal"** or **"SSH via Web Terminal"**

---

## 📦 Step 2: Verify GPU and Install Git

**Run these commands in the RunPod terminal:**

```bash
# Check GPU
nvidia-smi
# Should show: RTX 4090 with 24GB or your selected GPU

# Verify CUDA
nvcc --version
# Should show: CUDA 12.x (12.1, 12.8, or similar - all compatible)

# Install git (if not present)
apt-get update && apt-get install -y git wget curl tmux
```

---

## 📥 Step 3: Clone Repository

```bash
# Navigate to workspace
cd /workspace

# Clone your repo
git clone https://github.com/SuWh1/SexyAldarKose.git

# Navigate to project
cd SexyAldarKose/backend/aldar_kose_project

# Verify structure
ls -la
# Should see: scripts/, configs/, data/, requirements.txt
```

---

## 🐍 Step 4: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA (usually pre-installed on RunPod, but verify)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch sees GPU
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
# Should print: CUDA Available: True, GPU: NVIDIA GeForce RTX 4090

# Install all dependencies
pip install -r requirements.txt

# This installs:
# - transformers, diffusers, accelerate, peft
# - bitsandbytes (8-bit optimizer)
# - PIL, OpenCV, PyYAML, tqdm
# Takes ~3-5 minutes
```

---

## ⚙️ Step 5: Configure Accelerate

```bash
# Quick config (accept all defaults)
accelerate config default

# Or manual config (recommended):
accelerate config

# Answer these questions:
# - In which compute environment? [0] This machine
# - Which type of machine? [0] No distributed training  
# - Do you want to run on CPU only? NO
# - Do you wish to optimize with torch dynamo? NO
# - Do you want to use DeepSpeed? NO
# - What GPU(s) should be used? 0
# - Do you wish to use FP16 or BF16? fp16

# Verify config
accelerate env
```

---

## 📁 Step 6: Upload Training Data

**Option A: Upload from Local Machine (SCP)**

```bash
# On YOUR LOCAL MACHINE (not RunPod):
# Compress your data
cd path/to/SexyAldarKose/backend/aldar_kose_project
tar -czf training_data.tar.gz data/images data/captions

# Upload to RunPod (get SSH details from RunPod dashboard)
scp -P YOUR_SSH_PORT training_data.tar.gz root@YOUR_POD_IP:/workspace/SexyAldarKose/backend/aldar_kose_project/

# Back on RUNPOD terminal:
cd /workspace/SexyAldarKose/backend/aldar_kose_project
tar -xzf training_data.tar.gz
rm training_data.tar.gz
```

**Option B: Upload via RunPod Web Interface**

1. In RunPod dashboard, click **"Files"** tab
2. Navigate to `/workspace/SexyAldarKose/backend/aldar_kose_project/data/`
3. Create folders: `images/` and `captions/`
4. Upload your files via web interface (drag & drop)

**Option C: Download from Cloud Storage**

```bash
cd /workspace/SexyAldarKose/backend/aldar_kose_project

# Google Drive (install gdown first)
pip install gdown
gdown YOUR_GOOGLE_DRIVE_SHARE_LINK -O training_data.zip
unzip training_data.zip -d data/

# Or wget if hosted elsewhere
wget https://your-server.com/training_data.zip
unzip training_data.zip -d data/

# Or from GitHub (if you stored data there)
# Make sure data is in your repo
git pull origin main
```

**Verify Data Upload:**

```bash
# Check image count
ls data/images/ | wc -l
# Should output: 32 (or your image count)

# Check caption count  
ls data/captions/ | wc -l
# Should output: 32 (matching images)

# View first caption
head -1 data/captions/*.txt | head -5
# Should contain "aldar_kose_man" trigger token
```

---

## 🔧 Step 7: Configure Training for Your GPU

```bash
# Edit config for RunPod GPU
nano configs/training_config.yaml
# Or use vim: vim configs/training_config.yaml
```

**For RTX 4090 (24GB) - RECOMMENDED:**

```yaml
# Find and change these lines:

resolution: 1024          # Full quality (was 512)
lora_rank: 32            # Full capacity (was 16)
train_text_encoder: true  # Better quality (was false)
max_steps: 3000          # Production run (was 2000)
enable_cpu_offload: false # Not needed with 24GB (was true)
save_every: 200          # Less frequent (was 50)
validate_every: 200      # Less frequent (was 50)
```

**For RTX 3090 (24GB):**
```yaml
resolution: 1024
lora_rank: 32
train_text_encoder: true
max_steps: 3000
enable_cpu_offload: false
```

**For RTX 3060 (12GB):**
```yaml
resolution: 768           # Medium quality
lora_rank: 24
train_text_encoder: false
max_steps: 2500
enable_cpu_offload: false
```

**Save and exit:**
- Nano: `Ctrl+O`, `Enter`, `Ctrl+X`
- Vim: `Esc`, `:wq`, `Enter`

---

## 📊 Step 8: Prepare Dataset

```bash
# Validate dataset
python scripts/prepare_dataset.py

# Expected output:
# ✅ Found 32 valid image-caption pairs
# ✅ All captions contain the trigger token

# Preprocess images to target resolution
python scripts/prepare_dataset.py --resize --resolution 1024

# Creates:
# - data/processed_images/ (resized to 1024x1024)
# - data/dataset_manifest_processed.json

# Takes ~2 minutes for 32 images
```

---

## ⚡ Step 9: Pre-encode Latents (CRITICAL - 40% Faster!)

```bash
# Pre-encode all images to latent space
python scripts/preprocess_latents.py

# Expected output:
# INFO:__main__:Loading VAE...
# Loading checkpoint shards: 100%|██████████| 2/2
# INFO:__main__:Processing 32 images to latents...
# Encoding to latents: 100%|█████████| 32/32 [00:45<00:00]
# INFO:__main__:Latents saved to outputs/aldar_kose_lora/latents
# INFO:__main__:Pre-processing complete!

# Takes ~5 minutes (downloads SDXL model on first run)
# Saves 40% training time + 2GB VRAM

# Verify latents created
ls outputs/aldar_kose_lora/latents/ | wc -l
# Should output: 32
```

---

## 🚀 Step 10: Start Training

**Option A: Direct Training (simple)**

```bash
# Start training directly
accelerate launch scripts/train_lora_sdxl.py

# Training will start immediately
# Press Ctrl+C to stop
```

**Option B: Training in Tmux (RECOMMENDED - prevents disconnection)**

```bash
# Create tmux session
tmux new -s training

# Start training
accelerate launch scripts/train_lora_sdxl.py

# Detach from tmux (training continues in background)
# Press: Ctrl+B, then D

# Your training is now running!
# You can close the browser, disconnect, etc.

# To reattach later:
tmux attach -t training

# To kill session:
tmux kill-session -t training
```

**Expected Training Output:**

```bash
INFO:__main__:Loading SDXL models...
Loading checkpoint shards: 100%|████████████| 2/2
# (First run downloads ~15GB SDXL model, takes 5-10 min)
# (Subsequent runs use cached model, instant)

INFO:__main__:Enabled gradient checkpointing
INFO:__main__:Setting up LoRA adapters...
trainable params: 34,079,744 || all params: 2,619,094,784 || trainable%: 1.3011
INFO:__main__:Using 8-bit AdamW optimizer
INFO:__main__:Using pre-encoded latents for faster training!

INFO:__main__:***** Training Configuration *****
INFO:__main__:  Num examples = 32
INFO:__main__:  Total optimization steps = 3000

Steps:   0%|                                    | 0/3000 [00:00<?, ?it/s]
INFO:__main__:Step 00010 | Loss: 0.1234 | LR: 1.00e-04 | Epoch: 0
Steps:   0%|▍                           | 10/3000 [00:18<1:32:15,  1.85s/it]
INFO:__main__:Step 00020 | Loss: 0.1123 | LR: 1.00e-04 | Epoch: 0
...
INFO:__main__:Step 00200 | Loss: 0.0856 | LR: 1.00e-04 | Epoch: 6
INFO:__main__:Saved checkpoint to outputs/checkpoints/checkpoint-200
INFO:__main__:Generating validation images...
INFO:__main__:Saved 4 validation images to outputs/aldar_kose_lora/validation_images/step-200
...
```

**Training Progress:**
- **Step 0-100:** Loss ~0.12 → 0.10 (learning character)
- **Step 500:** Loss ~0.08 (getting better)
- **Step 1000:** Loss ~0.05 (good quality)
- **Step 2000:** Loss ~0.03 (very good)
- **Step 3000:** Loss ~0.02 (excellent, converged)

---

## 📊 Step 11: Monitor Training

**Terminal 1: Training Progress**

```bash
# Reattach to tmux
tmux attach -t training

# Watch training progress
# You'll see step updates every 10 steps
```

**Terminal 2: GPU Monitoring (open new terminal)**

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Expected during training:
# GPU Utilization: 95-100%
# Memory Used: 14-16GB / 24GB (with 1024px resolution)
# Temperature: 65-80°C
# Power: 300-350W / 450W
```

**Terminal 3: View Metrics**

```bash
# Watch loss in real-time
tail -f outputs/aldar_kose_lora/training_metrics.csv

# Output:
# step,loss,learning_rate,epoch
# 10,0.123456,0.00010000,0
# 20,0.112345,0.00010000,0
# 30,0.105678,0.00010000,1
# ... (loss should decrease over time)
```

**View Validation Images:**

```bash
# List validation checkpoints
ls outputs/aldar_kose_lora/validation_images/

# View specific step
ls outputs/aldar_kose_lora/validation_images/step-200/

# Download validation images to local machine (from your computer):
# scp -P SSH_PORT -r root@POD_IP:/workspace/SexyAldarKose/backend/aldar_kose_project/outputs/aldar_kose_lora/validation_images ./
```

---

## ⏱️ Training Time Estimates

| GPU | Resolution | Steps | Time | Cost @ $0.69/hr |
|-----|-----------|-------|------|----------------|
| RTX 4090 | 1024px | 3000 | ~42 min | **$0.48** |
| RTX 4090 | 512px | 3000 | ~25 min | $0.29 |
| RTX 3090 | 1024px | 3000 | ~55 min | $0.45 @ $0.49/hr |
| RTX 3060 | 768px | 2500 | ~90 min | $0.51 @ $0.34/hr |

**With pre-encoded latents (recommended):**
- 40% faster than without
- Above times assume latents are pre-encoded

---

## 🎉 Step 12: Training Complete!

**When training finishes, you'll see:**

```bash
======================================================================
  🎉 TRAINING COMPLETE! 🎉
======================================================================

📊 Training Stats:
   Steps completed: 3000
   Epochs completed: 93

💾 Outputs saved to:
   Final model: outputs/aldar_kose_lora/final
   Checkpoints: outputs/checkpoints
   Validation images: outputs/aldar_kose_lora/validation_images
   Metrics CSV: outputs/aldar_kose_lora/training_metrics.csv
   Summary: outputs/aldar_kose_lora/training_summary.txt

======================================================================
```

**Verify Output:**

```bash
# Check final model
ls -lh outputs/aldar_kose_lora/final/unet_lora/
# Should show:
# adapter_config.json (1KB)
# adapter_model.safetensors (~600-1200MB)

# View training summary
cat outputs/aldar_kose_lora/training_summary.txt

# Check final loss
tail -1 outputs/aldar_kose_lora/training_metrics.csv
# Should show loss < 0.03 (good) or < 0.02 (excellent)
```

---

## 📥 Step 13: Download Your Trained Model

**Option A: Download via RunPod Web Interface**

1. Click **"Files"** tab in RunPod dashboard
2. Navigate to `/workspace/SexyAldarKose/backend/aldar_kose_project/outputs/aldar_kose_lora/`
3. Right-click `final` folder → **"Download"**
4. Save to your local machine

**Option B: Download via SCP (faster for large files)**

```bash
# On YOUR LOCAL MACHINE:
# Create download directory
mkdir -p ~/aldar_kose_trained_model

# Download final model
scp -P YOUR_SSH_PORT -r root@YOUR_POD_IP:/workspace/SexyAldarKose/backend/aldar_kose_project/outputs/aldar_kose_lora/final ~/aldar_kose_trained_model/

# Download validation images (optional)
scp -P YOUR_SSH_PORT -r root@YOUR_POD_IP:/workspace/SexyAldarKose/backend/aldar_kose_project/outputs/aldar_kose_lora/validation_images ~/aldar_kose_trained_model/

# Download metrics (optional)
scp -P YOUR_SSH_PORT root@YOUR_POD_IP:/workspace/SexyAldarKose/backend/aldar_kose_project/outputs/aldar_kose_lora/training_metrics.csv ~/aldar_kose_trained_model/
```

**Option C: Upload to Cloud Storage**

```bash
# Install cloud CLI tools (on RunPod)
pip install awscli  # For AWS S3
# or
pip install gcsutil  # For Google Cloud Storage

# Upload to S3
aws s3 sync outputs/aldar_kose_lora/final s3://your-bucket/models/aldar_kose_final/

# Upload to Google Drive (using gdown)
pip install gdown
# Use Google Drive API or rclone for upload
```

**Option D: Commit to GitHub (if using Git LFS)**

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "outputs/aldar_kose_lora/final/**"
git lfs track "*.safetensors"

# Add and commit
git add .gitattributes
git add outputs/aldar_kose_lora/final/
git commit -m "Add trained Aldar Kose LoRA model"
git push origin main
```

---

## 💾 Step 14: Stop RunPod Instance (IMPORTANT!)

**After downloading your model:**

```bash
# Exit tmux session
tmux kill-session -t training

# Or if still in tmux, press: Ctrl+B, then :kill-session
```

**In RunPod Dashboard:**

1. Go to **"My Pods"**
2. Find your running pod
3. Click **"Stop"** (if you might use it again soon)
4. Or click **"Terminate"** (to permanently delete and stop billing)

**⚠️ IMPORTANT:** RunPod bills per minute while pod is running!
- Always terminate pods when done
- Even stopped pods may incur storage charges
- Terminated pods are deleted (backup your model first!)

---

## 🔧 Troubleshooting

### Issue 1: CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**

```bash
# Edit config to use less memory
nano configs/training_config.yaml

# Reduce resolution:
resolution: 768  # Down from 1024

# Or reduce LoRA rank:
lora_rank: 24  # Down from 32

# Or disable text encoder training:
train_text_encoder: false

# Save and restart training
```

### Issue 2: Training Very Slow

**Symptoms:** < 1 step/second

**Check:**

```bash
# 1. Verify GPU is being used
nvidia-smi
# GPU Utilization should be 95-100%

# 2. Check if latents were pre-encoded
ls outputs/aldar_kose_lora/latents/ | wc -l
# Should show 32 files

# 3. If no latents, pre-encode them:
python scripts/preprocess_latents.py

# 4. Restart training
```

### Issue 3: Model Download Taking Forever

**Solution:**

```bash
# Pre-download SDXL model
python scripts/download_model.py

# Or set HF cache to volume storage (persists between pods)
export HF_HOME=/workspace/huggingface_cache
mkdir -p $HF_HOME

# Add to start of training:
echo 'export HF_HOME=/workspace/huggingface_cache' >> ~/.bashrc
source ~/.bashrc
```

### Issue 4: "No module named 'accelerate'"

**Solution:**

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install accelerate directly
pip install accelerate
```

### Issue 5: Data Upload Failed

**Solution:**

```bash
# Check if data directories exist
ls -la data/
# Should show: images/ and captions/

# If missing, create them:
mkdir -p data/images data/captions

# Re-upload your data
```

### Issue 6: Training Loss Not Decreasing

**Check:**

```bash
# View loss trend
tail -20 outputs/aldar_kose_lora/training_metrics.csv

# If loss stuck or increasing:
# 1. Check data quality
head -5 data/captions/*.txt
# Each caption should be descriptive (50-150 chars)

# 2. Try lower learning rate
nano configs/training_config.yaml
# Change: learning_rate: 5.0e-5  # Down from 1.0e-4

# 3. Train longer
# Change: max_steps: 4000  # Up from 3000
```

### Issue 7: Connection Lost During Training

**Solution:**

```bash
# If you used tmux, training continues!
# Just reconnect to RunPod and:
tmux attach -t training

# Check training is still running:
ps aux | grep train_lora_sdxl
# Should show python process running

# Check progress:
tail outputs/aldar_kose_lora/training_metrics.csv
```

---

## 📋 Complete Command Checklist

**Copy-paste these commands in order:**

```bash
# 1. Setup
cd /workspace
git clone https://github.com/SuWh1/SexyAldarKose.git
cd SexyAldarKose/backend/aldar_kose_project

# 2. Install dependencies
pip install --upgrade pip
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. Configure
accelerate config default

# 4. Upload data (your method here)
# ... upload data to data/images/ and data/captions/ ...

# 5. Verify data
ls data/images/ | wc -l
ls data/captions/ | wc -l

# 6. Configure for RTX 4090 (edit if different GPU)
nano configs/training_config.yaml
# Set: resolution=1024, lora_rank=32, train_text_encoder=true

# 7. Prepare dataset
python scripts/prepare_dataset.py --resize --resolution 1024

# 8. Pre-encode latents (CRITICAL)
python scripts/preprocess_latents.py

# 9. Start training in tmux
tmux new -s training
accelerate launch scripts/train_lora_sdxl.py
# Ctrl+B, D to detach

# 10. Monitor (in new terminal)
watch -n 1 nvidia-smi
tail -f outputs/aldar_kose_lora/training_metrics.csv

# 11. After training, download model
# (Use SCP or RunPod web interface)

# 12. Terminate pod!
```

---

## 💰 Cost Calculator

**RunPod RTX 4090 @ $0.69/hr:**

```
Setup time: 15 minutes = $0.17
Pre-encoding: 5 minutes = $0.06
Training (3000 steps): 42 minutes = $0.48
Buffer: 8 minutes = $0.09
─────────────────────────────────
Total: ~70 minutes = $0.80
```

**Cost optimization tips:**
1. ✅ Use spot instances (70% cheaper, but can be interrupted)
2. ✅ Pre-encode latents before starting GPU time
3. ✅ Set up data on volume storage (persists between pods)
4. ✅ Download model immediately after training
5. ✅ **ALWAYS terminate pod when done!**

---

## 🎯 Quick Reference

**Essential Commands:**

```bash
# Check GPU
nvidia-smi

# Monitor training
tmux attach -t training

# View loss
tail -f outputs/aldar_kose_lora/training_metrics.csv

# Check training status
ps aux | grep train_lora_sdxl

# Kill training (if needed)
pkill -f train_lora_sdxl

# Download model (from local machine)
scp -P PORT -r root@IP:/workspace/SexyAldarKose/backend/aldar_kose_project/outputs/aldar_kose_lora/final ./
```

**Expected Files After Training:**

```
outputs/aldar_kose_lora/
├── final/                           ⭐ YOUR TRAINED MODEL
│   └── unet_lora/
│       ├── adapter_config.json      (1KB)
│       └── adapter_model.safetensors (~600-1200MB)
├── training_metrics.csv             (Training history)
├── training_summary.txt             (Final statistics)
└── validation_images/               (Generated samples)
    ├── step-200/
    ├── step-400/
    └── ...

outputs/checkpoints/                 (Intermediate checkpoints)
├── checkpoint-200/
├── checkpoint-400/
└── ... (last 5 kept)
```

---

## ✅ Success Checklist

- [ ] RunPod instance launched (RTX 4090 recommended)
- [ ] GPU verified with `nvidia-smi`
- [ ] Repository cloned
- [ ] Dependencies installed
- [ ] Accelerate configured
- [ ] Training data uploaded (32 images + captions)
- [ ] Dataset validated (no errors)
- [ ] Config adjusted for GPU
- [ ] Images preprocessed
- [ ] Latents pre-encoded (CRITICAL)
- [ ] Training started in tmux
- [ ] GPU at 95-100% utilization
- [ ] Loss decreasing over time
- [ ] Model trained successfully
- [ ] Model downloaded to local machine
- [ ] **Pod terminated** (to stop billing)

---

## 🆘 Emergency Commands

```bash
# Training stuck? Kill it:
pkill -f train_lora_sdxl

# GPU frozen? Reset:
nvidia-smi --gpu-reset

# Out of disk space? Clean up:
rm -rf outputs/checkpoints/checkpoint-*
rm -rf ~/.cache/huggingface/hub/*

# Lost tmux session? Find it:
tmux ls
tmux attach -t training

# Start over:
cd /workspace
rm -rf SexyAldarKose
# ... then start from Step 3
```

---

## 📞 Support

**Documentation:**
- Full guide: `VM_TRAINING_GUIDE.md`
- Technical review: `TRAINING_PIPELINE_REVIEW.md`
- Quick reference: `VM_QUICK_START.md`

**RunPod Support:**
- Discord: https://discord.gg/runpod
- Docs: https://docs.runpod.io/

**Common Issues:**
- GPU not detected → Restart pod
- Training slow → Pre-encode latents
- Out of memory → Reduce resolution
- Connection lost → Training continues in tmux

---

**Time to Complete:** ~10-15 minutes setup + 40-60 minutes training
**Total Cost:** ~$0.80 per training run
**Result:** Professional LoRA model for your character

**Happy Training! 🚀**

---

**Version:** 1.0 | **Date:** Oct 18, 2025 | **Platform:** RunPod Optimized
