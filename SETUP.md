# Aldar Köse Storyboard Generator - Setup Guide

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- **Python 3.10 or 3.11**
- **GPU with 18GB+ VRAM** (NVIDIA recommended)
- **30GB free disk space**
- **OpenAI API key**

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/SuWh1/SexyAldarKose.git
cd SexyAldarKose/backend/aldar_kose_project
```

**2. Install PyTorch with CUDA (do this FIRST!):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**3. Install remaining dependencies:**
```bash
pip install -r requirements_inference.txt
```

**4. Set your OpenAI API key:**

**Windows:**
```powershell
$env:OPENAI_API_KEY="sk-your-api-key-here"
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

**5. Run the demo:**
```bash
python scripts/submission_demo.py
```

---

## 📊 What Happens

1. ✅ Verifies all dependencies installed
2. ✅ Downloads LoRA model from S3 (~10MB)
3. ✅ Generates 6-frame storyboard
4. ✅ Saves to `outputs/terminal_generation_YYYYMMDD_HHMMSS/`

**Time:** 8-9 minutes (first run), 4-5 minutes (subsequent runs)

---

## 🔍 Verify Installation (Optional)

```bash
python scripts/verify_installation.py
```

This checks:
- Python version
- CUDA/GPU availability
- All dependencies
- API key configuration
- Disk space

---

## 🎯 Custom Stories

```bash
python scripts/submission_demo.py "Your custom story here"
```

**Examples:**
```bash
python scripts/submission_demo.py "Aldar tricks a greedy merchant"
python scripts/submission_demo.py "Aldar finds treasure in the mountains"
```

---

## 📦 What Gets Installed

**Essential packages (14 total):**
- `torch` - PyTorch with CUDA
- `transformers` - Hugging Face models
- `diffusers` - Stable Diffusion
- `peft` - LoRA adapters
- `openai` - GPT-4 API
- `opencv-python` - Image processing
- `Pillow` - Image manipulation
- `numpy`, `tqdm`, `PyYAML`, `python-dotenv`
- `controlnet-aux`, `onnxruntime` - ControlNet support
- `accelerate`, `safetensors` - Model loading

**Total download size:** ~2GB (PyTorch) + ~100MB (other packages)

---

## ❓ Troubleshooting

### "OPENAI_API_KEY not found"
Set the environment variable:
```bash
export OPENAI_API_KEY="sk-your-key"  # Linux/Mac
$env:OPENAI_API_KEY="sk-your-key"    # Windows
```

### "torch.cuda.OutOfMemoryError"
Your GPU needs 18GB+ VRAM. Check with:
```bash
nvidia-smi
```

### "Failed to download model"
Check internet connection and verify access to:
```
https://aldarkose.s3.amazonaws.com
```

### Models downloading slowly
First run downloads ~6GB of models from Hugging Face.
Subsequent runs are much faster (models cached).

---

## 📁 Output Structure

```
outputs/terminal_generation_20251019_143052/
├── frame_001.png          # 1024×1024 storyboard frames
├── frame_002.png
├── frame_003.png
├── frame_004.png
├── frame_005.png
├── frame_006.png
├── scene_breakdown.json   # GPT-4's scene analysis
├── sdxl_prompts.json      # Refined prompts
└── report.json            # Consistency metrics
```

---

## 🎓 Key Features

- **85-90% face consistency** (LoRA + CLIP validation)
- **Fully deterministic** (seed=42, temp=0.0)
- **No AWS configuration** (public S3 downloads)
- **Automatic quality control** (CLIP validates each frame)
- **High resolution** (1024×1024 native)

---

## 📖 More Documentation

- **JUDGES_QUICKSTART.md** - Comprehensive guide for judges
- **ARCHITECTURE_FLOW.md** - Complete system architecture
- **TEXT_TO_IMAGE_TRANSFORMATION.md** - Training/inference details

---

## 🎉 Success Criteria

You'll know it worked when:
- ✅ 6 frames generated
- ✅ Aldar Köse's face recognizable across frames
- ✅ Story flows logically
- ✅ Average CLIP consistency ≥ 0.70

**Questions?** Open an issue on GitHub.
