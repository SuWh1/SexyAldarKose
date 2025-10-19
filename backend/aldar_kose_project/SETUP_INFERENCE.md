# 🚀 Quick Setup Guide for Inference

## Prerequisites
- Python 3.8+
- pip package manager
- 8GB+ VRAM (simple mode) or 16GB+ (ref-guided mode)

## Installation Steps

### 1. **Install PyTorch First** (Choose one based on your GPU)

#### For NVIDIA GPU (CUDA 12.1):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### For CPU only:
```bash
pip install torch torchvision
```

#### For Mac (Metal GPU):
```bash
pip install torch torchvision
```

### 2. **Install Inference Dependencies**
```bash
pip install -r requirements_inference.txt
```

### 3. **Configure AWS Credentials** (for S3 model download)
```bash
aws configure
```
Or set environment variables:
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

### 4. **Set OpenAI API Key**
Create `.env` file in the project root:
```
OPENAI_API_KEY=sk-your-key-here
```

Or set environment variable:
```bash
export OPENAI_API_KEY=sk-your-key-here
```

---

## Running Story Generation

### **Basic Usage** (downloads model automatically)
```bash
python scripts/submission_demo.py
```

### **With Story Prompt**
```bash
python scripts/submission_demo.py "Aldar Kose tricks a greedy merchant"
```

### **Deterministic Output** (always same result)
```bash
python scripts/submission_demo.py "Your story" --seed 42
```

### **Simple Mode** (faster, fewer dependencies)
```bash
python scripts/submission_demo.py "Your story" --no-ref-guided
```

### **Skip AWS Download** (model already local)
```bash
python scripts/submission_demo.py "Your story" --skip-download
```

---

## Generation Modes

### **Reference-Guided Mode** (DEFAULT)
- ✅ Best consistency (85-90%)
- ❌ Requires more dependencies
- ❌ Slower (10-15 sec/frame)
- ❌ Needs 16GB+ VRAM
- ⚙️ Install: `pip install controlnet-aux insightface onnxruntime`

### **Simple Mode** (`--no-ref-guided`)
- ✅ Works with minimal dependencies
- ✅ Faster (5-8 sec/frame)
- ✅ Needs 8-12GB VRAM
- ❌ Lower consistency (70-75%)
- ✅ Automatic fallback if dependencies missing

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'torch'`
**Solution:** Install PyTorch (see step 1 above)

### Error: `ModuleNotFoundError: No module named 'controlnet_aux'`
**Options:**
1. Install: `pip install controlnet-aux insightface onnxruntime`
2. Or use simple mode: `--no-ref-guided`

### Error: `OPENAI_API_KEY not set`
**Solution:** 
- Create `.env` file with `OPENAI_API_KEY=sk-...`
- Or run: `export OPENAI_API_KEY=sk-...`

### Error: `AWS credentials not found`
**Solution:**
```bash
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region
```

### Error: `Out of memory`
**Options:**
1. Use simple mode: `--no-ref-guided`
2. Use different GPU
3. Reduce batch size in config

---

## Typical Installation Time

- PyTorch: 5-10 minutes
- Dependencies: 2-5 minutes
- Model download (first run): 1-2 minutes
- **Total: ~15-20 minutes for first run**

## Typical Generation Time

- Simple mode: 5-8 minutes (6-8 frames)
- Ref-guided mode: 8-12 minutes (6-8 frames)

---

## File Structure

```
backend/aldar_kose_project/
├── scripts/
│   ├── submission_demo.py          # Main entry point
│   ├── generate_story.py           # Story generator
│   └── prompt_storyboard.py        # GPT-4 orchestrator
├── outputs/
│   ├── checkpoints/                # Downloaded model cache
│   │   └── checkpoint-1000/        # LoRA adapter
│   └── terminal_generation_*/      # Generated stories
├── .env                            # Your API keys (don't commit!)
└── requirements_inference.txt      # Minimal dependencies
```

---

## Support

For issues or questions, check:
1. `.env` file has `OPENAI_API_KEY`
2. AWS credentials configured
3. Required packages installed: `pip list | grep torch`
4. Python version: `python --version` (3.8+)

---

**You're all set! 🚀 Run: `python scripts/submission_demo.py`**
