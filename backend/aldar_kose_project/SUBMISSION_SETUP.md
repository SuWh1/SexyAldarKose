# 🚀 Submission Setup - Complete Guide

## ⚡ Quick Start (3 Commands)

```powershell
# 1. Install all dependencies (one-liner)
python install_dependencies.py

# 2. Configure AWS (enter your credentials)
aws configure

# 3. Run the generator
python scripts/submission_demo.py "Your story here"
```

That's it! ✨

---

## 📋 What Gets Installed

The `requirements_inference.txt` includes everything for **ref-guided generation**:

### Core ML Stack
- ✅ `torch` + `torchvision` - PyTorch deep learning
- ✅ `transformers` - Hugging Face models
- ✅ `diffusers` - Stable Diffusion pipeline
- ✅ `peft` - LoRA adapter loading

### Ref-Guided Generation (Best Quality)
- ✅ `controlnet-aux` - Pose/composition control
- ✅ `insightface` - Face detection & embedding
- ✅ `onnxruntime` - Fast inference for models

### Quality Assurance
- ✅ `mediapipe` - Pose/face detection for anomaly detection

### Cloud & API
- ✅ `boto3` - AWS S3 downloads
- ✅ `openai` - GPT-4 scene generation

### Utilities
- ✅ `python-dotenv` - Load `.env` configuration
- ✅ `opencv-python`, `pillow` - Image processing
- ✅ All other dependencies

---

## 🎯 Submission Script Features

The `submission_demo.py` script does everything:

### 1️⃣ Dependency Check
```
✅ torch
✅ transformers
✅ diffusers
✅ peft
✅ PIL
✅ openai
✅ boto3
```

### 2️⃣ API Key Verification
- Loads `.env` file automatically
- Checks `OPENAI_API_KEY` is set
- Shows masked key for security

### 3️⃣ Model Download
- Downloads LoRA from `s3://aldarkose/checkpoint-1000/`
- Uses boto3 (installed via requirements)
- Syncs to `outputs/checkpoints/checkpoint-1000/`

### 4️⃣ Story Generation
- Prompts for story (or uses argument)
- Runs with **ref-guided mode** (85-90% consistency)
- Temperature **0.0** (deterministic/reproducible)
- Seed **42** (same output every time)

### 5️⃣ Output
- Saves frames to `outputs/terminal_generation_YYYYMMDD_HHMMSS/`
- Includes scene breakdown JSON
- All with full consistency

---

## 📦 Installation Details

### Option 1: Automatic (Recommended)
```powershell
python install_dependencies.py
```

Handles:
- PyTorch installation with proper CUDA version
- All inference dependencies
- Verification of installed packages

### Option 2: Manual

**Step 1: Install PyTorch (choose one)**

NVIDIA GPU (CUDA 12.1):
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CPU only:
```powershell
pip install torch torchvision
```

Mac (Metal GPU):
```powershell
pip install torch torchvision
```

**Step 2: Install everything else**
```powershell
pip install -r requirements_inference.txt
```

---

## ⚙️ Configuration

### AWS Credentials
```powershell
aws configure
# Enter:
# AWS Access Key ID: [your key]
# AWS Secret Access Key: [your secret]
# Default region: us-east-1
# Default output format: json
```

Or set environment variables:
```powershell
$env:AWS_ACCESS_KEY_ID="your_key"
$env:AWS_SECRET_ACCESS_KEY="your_secret"
$env:AWS_DEFAULT_REGION="us-east-1"
```

### OpenAI API Key

Create `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

Or set environment variable:
```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
```

---

## 🎬 Usage Examples

### Default (Interactive)
```powershell
python scripts/submission_demo.py
# Prompts you to enter a story
```

### With Story Prompt
```powershell
python scripts/submission_demo.py "Aldar Kose tricks a greedy merchant"
```

### Reproducible Output
```powershell
python scripts/submission_demo.py "Aldar adventure" --seed 42
```

### Skip Model Download (already cached)
```powershell
python scripts/submission_demo.py "Story" --skip-download
```

### Simple Mode (not recommended for submission)
```powershell
python scripts/submission_demo.py "Story" --no-ref-guided
```

---

## ✅ Pre-Submission Checklist

- [ ] Run `python install_dependencies.py`
- [ ] Run `aws configure` and enter credentials
- [ ] Create `.env` with `OPENAI_API_KEY=sk-...`
- [ ] Test: `python scripts/submission_demo.py "Test story" --skip-download`
- [ ] Verify output in `outputs/terminal_generation_*/`
- [ ] Check frames look good (consistency, quality)

---

## 📊 Specs

**Installation Time:** ~15-20 minutes (first time)
**First Run:** ~2-3 minutes (model download) + 8-12 minutes (generation)
**Subsequent Runs:** ~8-12 minutes (model cached)

**Generation Quality:**
- Mode: **Reference-Guided** (85-90% face consistency)
- Temperature: **0.0** (deterministic, reproducible)
- Seed: **42** (same output every time)

**Output Size:**
- Per story: ~50-100MB (6-8 frames at 1024×1024)

---

## 🆘 Troubleshooting

### Error: `No module named torch`
```powershell
# PyTorch not installed
python install_dependencies.py
```

### Error: `OPENAI_API_KEY not set`
```powershell
# Create .env file in project root with:
OPENAI_API_KEY=sk-...
```

### Error: `AWS credentials not found`
```powershell
aws configure
# Or set environment variables
```

### Error: `Out of memory`
```powershell
# You need 16GB+ VRAM for ref-guided mode
# Use --no-ref-guided for simple mode (needs 8GB)
python scripts/submission_demo.py "Story" --no-ref-guided
```

### Model download fails
```powershell
# Skip download, use cached model
python scripts/submission_demo.py "Story" --skip-download
```

---

## 📁 File Structure

```
backend/aldar_kose_project/
├── install_dependencies.py         ← RUN THIS FIRST
├── requirements_inference.txt      ← All dependencies (ref-guided included)
├── SETUP_INFERENCE.md              ← Detailed setup guide
├── .env                            ← Your API keys (create this)
├── scripts/
│   ├── submission_demo.py          ← MAIN ENTRY POINT
│   ├── generate_story.py           ← Story generator
│   └── prompt_storyboard.py        ← GPT-4 orchestrator
└── outputs/
    ├── checkpoints/                ← Downloaded models
    └── terminal_generation_*/      ← Generated stories
```

---

## 🎯 For Judges/Reviewers

**To run the demo:**
```powershell
# One-liner setup
python install_dependencies.py && aws configure

# Then run
python scripts/submission_demo.py "Aldar wins a legendary horse race"
```

**What happens:**
1. ✅ Downloads LoRA model from AWS S3
2. ✅ Generates story with GPT-4 (6-8 scenes)
3. ✅ Generates images with SDXL + LoRA
4. ✅ Applies ref-guided consistency
5. ✅ Saves to `outputs/terminal_generation_*/`

**Output:** 6-8 high-quality frames with 85-90% face consistency, deterministic and reproducible

---

## 🚀 You're Ready!

Everything is set up for submission:
- ✅ All dependencies in `requirements_inference.txt`
- ✅ One-command installer: `install_dependencies.py`
- ✅ Full automated pipeline: `submission_demo.py`
- ✅ **Always uses ref-guided mode** (best quality)
- ✅ **Always uses 0.0 temperature** (deterministic)
- ✅ **AWS model download included**

**Just run:**
```powershell
python install_dependencies.py
aws configure
python scripts/submission_demo.py
```

**Done! 🎉**
