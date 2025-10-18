# Scripts Overview

This directory contains all the scripts for the Aldar Kose LoRA training and storyboard generation pipeline.

## 🎬 Storyboard Generation

### **prompt_storyboard.py** ⭐ NEW! User-Friendly
**Prompt-based storyboard generation** - Takes a simple story description and generates 6-10 frames automatically.

```bash
# Interactive mode
python scripts/prompt_storyboard.py --lora-path outputs/checkpoints/checkpoint-400

# Direct story
python scripts/prompt_storyboard.py \
    --lora-path outputs/checkpoints/checkpoint-400 \
    --story "Aldar Kose riding his horse to his yurt at sunset" \
    --num-frames 8
```

**Features:**
- Uses OpenAI GPT-4 to break down story into scenes
- Automatically creates diverse camera angles
- Generates 6-10 images with character consistency
- See: `PROMPT_STORYBOARD_GUIDE.md`

---

### **test_prompt_storyboard.py** ⭐ Test Without GPU
Test story breakdown **without** running image generation.

```bash
python scripts/test_prompt_storyboard.py
```

**Use this to:**
- Preview scene breakdown before GPU usage
- Verify camera angles and variety
- Iterate on story phrasing
- No cost, no GPU needed

---

### **simple_storyboard.py** - Advanced
Generate storyboard from **pre-written scene prompts**.

```bash
python scripts/simple_storyboard.py \
    --lora-path outputs/checkpoints/checkpoint-400 \
    --prompts-file story_scene_prompts.json \
    --output-dir outputs/trick_story
```

**Use this when:**
- You want full control over each scene
- Testing specific prompt variations
- You already have a JSON file with prompts

---

### **test_storyboard_quick.py** - Quick Test
Quick test with 6 predefined diverse scenes.

```bash
python scripts/test_storyboard_quick.py
```

**Use for:**
- Testing model quality
- Verifying diversity
- Quick checkpoint validation

---

## 🖼️ Image Captioning

### **label_images.py** - Automated Captioning
Generate captions for training images using OpenAI Vision API.

```bash
python scripts/label_images.py \
    --input_dir raw_images/ \
    --output_images data/images/ \
    --output_captions data/captions/
```

**Features:**
- Uses GPT-4 Vision to describe images
- Automatically adds trigger token "aldar_kose_man"
- Follows caption guidelines (no clothing descriptions)
- Batch processing with progress tracking

---

### **clean_captions.py** - Remove Clothing Descriptions
Automated cleanup to remove clothing/outfit mentions from captions.

```bash
python scripts/clean_captions.py
```

**What it does:**
- Removes patterns like "wearing X", "in Y coat", "green hat"
- Creates backup before modifying
- Shows before/after for each change
- Preserves action/setting descriptions

---

### **polish_captions.py** - Clean Up Artifacts
Post-processing to fix artifacts after cleaning.

```bash
python scripts/polish_captions.py
```

**Fixes:**
- Orphaned color words
- Double commas
- Extra spaces
- Incomplete phrases

---

### **recaption_images.py** - Regenerate All Captions
Re-generate captions with **updated** OpenAI prompts (improved quality).

```bash
python scripts/recaption_images.py
```

**Use when:**
- You've updated caption guidelines
- Initial captions have quality issues
- You want to regenerate from scratch

---

### **manual_caption_helper.py** - Manual Editing Helper
Interactive tool for manually reviewing and editing captions.

```bash
python scripts/manual_caption_helper.py
```

**Features:**
- Shows image + current caption
- Allows manual edits
- Tracks progress
- Useful for final quality pass

---

## 🏋️ Training

### **train_lora_sdxl.py** - Main Training Script
Train SDXL LoRA on your Aldar Kose dataset.

```bash
python scripts/train_lora_sdxl.py
```

**Configured via:** `configs/training_config.yaml`

**Key settings:**
- Batch size: 4
- LoRA rank: 64
- Steps: 400
- Learning rate: 1e-4
- Mixed precision: bf16

**See:** `TRAINING_PIPELINE_REVIEW.md`, `VM_TRAINING_GUIDE.md`

---

## 🔍 Testing & Validation

### **inference.py** - Single Image Generation
Test inference with a single prompt.

```bash
python scripts/inference.py \
    --lora-path outputs/checkpoints/checkpoint-400 \
    --prompt "aldar_kose_man riding horse, sunset, 3D animation" \
    --output inference_test.png
```

---

### **batch_inference.py** - Multiple Images
Generate multiple images from a list of prompts.

```bash
python scripts/batch_inference.py \
    --lora-path outputs/checkpoints/checkpoint-400 \
    --prompts-file test_prompts.json \
    --output-dir outputs/batch_test/
```

---

### **test_inference.py** - Quick Inference Test
Quick test to verify model loads and generates correctly.

```bash
python scripts/test_inference.py
```

---

### **evaluate_identity.py** - Character Consistency Check
Measure how well the LoRA preserves character identity across generations.

```bash
python scripts/evaluate_identity.py \
    --lora-path outputs/checkpoints/checkpoint-400 \
    --reference-images data/images/
```

**Metrics:**
- CLIP similarity scores
- Identity preservation
- Consistency across prompts

---

### **validate_storyboard.py** - Storyboard Quality Check
Validate a generated storyboard for quality and consistency.

```bash
python scripts/validate_storyboard.py \
    --input-dir outputs/test_storyboard/
```

**Checks:**
- Frame consistency
- Character identity preservation
- Scene diversity

---

## 📊 Data Preparation

### **prepare_dataset.py** - Dataset Preparation
Prepare images and captions for training.

```bash
python scripts/prepare_dataset.py
```

**Does:**
- Validates image dimensions (1024x1024)
- Checks caption file existence
- Creates train/val splits
- Reports dataset statistics

---

### **preprocess_latents.py** - Pre-compute Latents
Pre-compute VAE latents to speed up training.

```bash
python scripts/preprocess_latents.py
```

**Benefits:**
- Faster training (skips VAE encoding)
- Lower VRAM usage during training
- One-time preprocessing cost

---

## 🛠️ Utilities

### **setup_environment.py** - Environment Setup
Set up Python environment with all dependencies.

```bash
python scripts/setup_environment.py
```

---

### **download_model.py** - Download Base Model
Download SDXL base model before training.

```bash
python scripts/download_model.py
```

---

## 📋 Workflow Quick Reference

### 1️⃣ Initial Setup
```bash
python scripts/setup_environment.py
python scripts/download_model.py
```

### 2️⃣ Prepare Training Data
```bash
# Option A: Automated captioning
python scripts/label_images.py --input_dir raw_images/

# Option B: Manual captions + cleanup
python scripts/clean_captions.py
python scripts/polish_captions.py

# Prepare dataset
python scripts/prepare_dataset.py
```

### 3️⃣ Train LoRA
```bash
python scripts/train_lora_sdxl.py
```

### 4️⃣ Test Model
```bash
# Quick test
python scripts/test_storyboard_quick.py

# Prompt-based generation
python scripts/prompt_storyboard.py \
    --lora-path outputs/checkpoints/checkpoint-400 \
    --story "Your story here"
```

### 5️⃣ Generate Final Storyboard
```bash
python scripts/prompt_storyboard.py \
    --lora-path outputs/checkpoints/checkpoint-400 \
    --story "Aldar Kose riding his horse to his yurt at sunset" \
    --num-frames 8 \
    --output-dir outputs/final_storyboard
```

---

## 🆘 Troubleshooting

### Common Issues

**Import errors:**
```bash
pip install -r requirements.txt
```

**CUDA out of memory:**
- Reduce batch size in `training_config.yaml`
- Use `--num-inference-steps 30` for inference
- Clear GPU cache: `torch.cuda.empty_cache()`

**OpenAI API errors:**
```bash
export OPENAI_API_KEY=sk-...
```

**Caption quality issues:**
1. Run `clean_captions.py`
2. Run `polish_captions.py`
3. Review with `manual_caption_helper.py`

---

## 📚 Documentation

- **`PROMPT_STORYBOARD_GUIDE.md`** - Comprehensive guide for prompt-based generation
- **`TRAINING_PIPELINE_REVIEW.md`** - Full training process overview
- **`CAPTION_GUIDELINES.md`** - Caption quality guidelines
- **`VM_TRAINING_GUIDE.md`** - RunPod setup and training
- **`VM_QUICK_START.md`** - Quick start for RunPod
- **`RUNPOD_SETUP.md`** - Detailed RunPod configuration

---

## 💡 Pro Tips

1. **Test story breakdown first** (free): `python scripts/test_prompt_storyboard.py`
2. **Use checkpoint-400** - identified as good model
3. **Start with 8 frames** - good balance of story and generation time
4. **Be specific in stories** - "riding horse to yurt at sunset" better than "riding horse"
5. **Review captions regularly** - clean data = better model

---

## 🚀 Quick Commands

```bash
# Test story breakdown (no GPU)
python scripts/test_prompt_storyboard.py

# Generate storyboard from prompt
python scripts/prompt_storyboard.py --lora-path outputs/checkpoints/checkpoint-400

# Quick storyboard test
python scripts/test_storyboard_quick.py

# Generate captions
python scripts/label_images.py --input_dir raw_images/

# Clean captions
python scripts/clean_captions.py && python scripts/polish_captions.py

# Train model
python scripts/train_lora_sdxl.py
```
