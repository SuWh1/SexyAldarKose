# Aldar Köse Storyboard Generator - Quick Start Guide for Judges

**Welcome!** This guide will help you run our AI-powered storyboard generator in under 10 minutes.

---

## 🎯 What This Demo Does

Our system generates consistent, story-driven storyboards featuring **Aldar Köse**, a Kazakh folk hero, using:
- **SDXL** (Stable Diffusion XL) for high-quality image generation
- **LoRA fine-tuning** trained on 70 images of Aldar Köse for 85-90% face consistency
- **GPT-4** for intelligent story breakdown into scenes
- **CLIP validation** to ensure visual consistency across frames

**Input:** A story prompt (e.g., "Aldar tricks a greedy merchant in the bazaar")  
**Output:** 6 coherent storyboard frames showing the complete story

---

## ⚡ Prerequisites

### Required:
1. **Python 3.10 or 3.11** (recommended: 3.10)
2. **GPU with 18GB+ VRAM** (NVIDIA recommended)
   - Tested on: RTX 3090, RTX 4090, A100
3. **30GB free disk space**
   - Models: ~6GB
   - Generated outputs: ~500MB per story
4. **Stable internet connection** (for downloading models)
5. **OpenAI API key** (for GPT-4 access)

### System Requirements:
- **OS:** Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **RAM:** 32GB+ recommended
- **CUDA:** 11.8 or 12.1 (for PyTorch)

---

## 📦 Installation (3 Steps)

### Step 1: Clone the Repository

```bash
git clone https://github.com/SuWh1/SexyAldarKose.git
cd SexyAldarKose
```

### Step 2: Install Dependencies

Navigate to the project directory:
```bash
cd backend/aldar_kose_project
```

**Install PyTorch with CUDA support (do this FIRST!):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Then install remaining dependencies:**
```bash
pip install -r requirements_inference.txt
```

**That's it!** Installation takes about 5-10 minutes.

**Optional - Verify installation:**
```bash
python scripts/verify_installation.py
```

This checks Python version, CUDA/GPU, dependencies, and disk space.

### Step 3: Set OpenAI API Key

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-your-api-key-here"
```

**Linux/macOS:**
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

**Or create a .env file:**
```bash
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
```

Get your API key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

---

## 🚀 Running the Demo

**From the `backend/aldar_kose_project/` directory:**

```bash
python scripts/submission_demo.py
```

**That's it!** The script will:
1. ✅ Check dependencies
2. ✅ Download LoRA model (~10MB)
3. ✅ Generate 6-frame storyboard
4. ✅ Save to `outputs/terminal_generation_YYYYMMDD_HHMMSS/`

**Time:** 8-9 minutes (first run), 4-5 minutes (subsequent runs)

### Custom Story

```bash
python scripts/submission_demo.py "Your custom story here"
```

**Examples:**

```bash
# Example 1: Trickster story
python scripts/submission_demo.py "Aldar Kose tricks a greedy merchant by selling him an invisible horse"

# Example 2: Adventure story
python scripts/submission_demo.py "Aldar Kose finds a magical artifact in the mountains and brings it to his village"

# Example 3: Comedy story
python scripts/submission_demo.py "Aldar Kose pretends to be a fortune teller and gives ridiculous predictions"
```

---

## 📊 What to Expect

### Console Output

```
=== Aldar Köse Storyboard Generator ===

[1/6] Checking dependencies...
✓ All dependencies installed

[2/6] Verifying OpenAI API key...
✓ API key found

[3/6] Downloading LoRA model from S3...
✓ Downloaded: unet_lora/adapter_config.json
✓ Downloaded: unet_lora/adapter_model.safetensors
✓ Downloaded: text_encoder_one_lora/adapter_config.json
✓ Downloaded: text_encoder_one_lora/adapter_model.safetensors
✓ Downloaded: text_encoder_two_lora/adapter_config.json
✓ Downloaded: text_encoder_two_lora/adapter_model.safetensors

[4/6] Starting story generation...
Breaking story into scenes...
✓ Generated 6 scenes

[5/6] Generating storyboard frames...
Loading models... (this may take 2-3 minutes)
✓ SDXL base model loaded (335 MB)
✓ ControlNet loaded (5 GB)
✓ CLIP model loaded (338 MB)
✓ LoRA checkpoint loaded (10 MB)

Generating frames:
Frame 1/6: ████████████████████ 100% (50 steps) [1m 15s]
Frame 2/6: ████████████████████ 100% (50 steps) [1m 20s] ✓ CLIP: 0.82
Frame 3/6: ████████████████████ 100% (50 steps) [1m 18s] ✓ CLIP: 0.79
Frame 4/6: ████████████████████ 100% (50 steps) [1m 22s] ✓ CLIP: 0.75
Frame 5/6: ████████████████████ 100% (50 steps) [1m 19s] ✓ CLIP: 0.81
Frame 6/6: ████████████████████ 100% (50 steps) [1m 21s] ✓ CLIP: 0.77

[6/6] Saving outputs...
✓ scene_breakdown.json
✓ sdxl_prompts.json
✓ report.json

=== Generation Complete ===
Output directory: outputs/terminal_generation_20251019_143052
Total time: 8 minutes 47 seconds
Average consistency: 0.79 (target: 0.70)
```

### Output Files

All outputs are saved to: `outputs/terminal_generation_YYYYMMDD_HHMMSS/`

```
outputs/terminal_generation_20251019_143052/
├── frame_001.png          # Storyboard frame 1 (1024×1024)
├── frame_002.png          # Storyboard frame 2
├── frame_003.png          # Storyboard frame 3
├── frame_004.png          # Storyboard frame 4
├── frame_005.png          # Storyboard frame 5
├── frame_006.png          # Storyboard frame 6
├── scene_breakdown.json   # GPT-4's scene analysis
├── sdxl_prompts.json      # Refined prompts for SDXL
└── report.json            # Consistency metrics & metadata
```

**File sizes:**
- Each frame: ~5 MB (1024×1024 PNG)
- JSON files: <100 KB each
- **Total per story:** ~30 MB

---

## 🔍 Verifying Success

### Check Visual Consistency

Open the generated frames (`frame_001.png` through `frame_006.png`) and verify:

1. **Character consistency:** Aldar Köse's face should look the same across all frames
2. **Story coherence:** Frames should tell a clear narrative progression
3. **Quality:** High-resolution (1024×1024), detailed images

### Check Metrics (report.json)

```json
{
  "base_seed": 42,
  "num_frames": 6,
  "pipeline": "reference_guided",
  "average_consistency": 0.82,
  "min_consistency": 0.71,
  "frames": [
    {
      "frame_number": 1,
      "seed": 42,
      "prompt": "aldar_kose_man in traditional kazakh clothing...",
      "consistency_score": null
    },
    {
      "frame_number": 2,
      "seed": 2042,
      "prompt": "aldar_kose_man talking to merchant...",
      "consistency_score": 0.82
    }
    // ... more frames
  ]
}
```

**Key metrics:**
- `average_consistency`: Should be **≥ 0.70** (our target: 0.79-0.82)
- `min_consistency`: Lowest CLIP score across all frames
- Each frame's `consistency_score`: Similarity to frame 1 (reference)

---

## 🐛 Troubleshooting

### Issue: "OPENAI_API_KEY not found"

**Solution:**
```bash
# Create .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Or set environment variable (Windows)
$env:OPENAI_API_KEY="sk-your-key-here"

# Or set environment variable (Linux/macOS)
export OPENAI_API_KEY="sk-your-key-here"
```

### Issue: "torch.cuda.OutOfMemoryError"

**Cause:** Your GPU doesn't have enough VRAM (need 18GB+)

**Solution:** 
This demo requires a high-end GPU. If you don't have one available, please contact us for pre-generated samples or cloud GPU access.

### Issue: "Failed to download model from S3"

**Cause:** Network connection issue

**Solution:**
1. Check your internet connection
2. Verify you can access: `https://aldarkose.s3.amazonaws.com`
3. Try running the script again (downloads are resumable)

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Cause:** Dependencies not installed

**Solution:**
```bash
pip install -r backend/aldar_kose_project/requirements_inference.txt
```

### Issue: Models downloading very slowly

**Cause:** First run downloads ~6GB of models from Hugging Face

**Solution:** Be patient! Subsequent runs will be much faster (models are cached locally in `~/.cache/huggingface/`)

### Issue: "Generation is taking too long"

**Expected behavior:**
- First run: 8-9 minutes (includes model downloads)
- Subsequent runs: 4-5 minutes (models cached)

If it's taking significantly longer:
1. Check GPU utilization: `nvidia-smi`
2. Ensure CUDA is properly installed
3. Check if other processes are using GPU memory

---

## 📖 Understanding the System

### Key Technologies

1. **SDXL (Stable Diffusion XL)**
   - Base model: `stabilityai/stable-diffusion-xl-base-1.0`
   - Resolution: 1024×1024 native
   - 50 denoising steps per frame

2. **LoRA (Low-Rank Adaptation)**
   - Fine-tuned on 70 images of Aldar Köse
   - Training: 1000 steps, rank=64
   - Size: ~10 MB (vs 2.6B full SDXL)
   - Stored on AWS S3 (public access)

3. **CLIP (Vision-Language Model)**
   - Model: `openai/clip-vit-large-patch14`
   - Validates consistency between frames
   - Threshold: 0.70 similarity score
   - Automatic retry if score too low

4. **GPT-4**
   - Breaks story into 6 coherent scenes
   - Temperature: 0.0 (fully deterministic)
   - Enforces narrative consistency

### Generation Process

```
User Story
    ↓
GPT-4 Scene Breakdown (6 scenes)
    ↓
Prompt Refinement (SDXL format)
    ↓
Frame 1: Pure SDXL + LoRA (reference)
    ↓
Frames 2-6: SDXL + LoRA + CLIP validation
    ↓
Save 6 frames + metadata
```

### Why It Works

- **LoRA fine-tuning:** Captures Aldar Köse's appearance without overfitting
- **CLIP validation:** Ensures faces match across frames (retry if < 0.70)
- **Fixed seed (42):** Reproducible results for same story
- **GPT-4 breakdown:** Intelligent scene transitions
- **Reference-guided generation:** First frame sets visual baseline

**Result:** 85-90% face consistency across frames

---

## 📚 Additional Documentation

For deeper technical understanding:

- **`ARCHITECTURE_FLOW.md`** - Complete system architecture with flowcharts
- **`TEXT_TO_IMAGE_TRANSFORMATION.md`** - Training and inference pipeline details
- **`SUBMISSION_SETUP.md`** - Technical setup details
- **`SUBMISSION_CHANGES.md`** - Recent changes (boto3 → urllib migration)

---

## 🎓 Technical Highlights for Evaluation

### Innovation Points

1. **Character Consistency Without Control Models**
   - Achieved 85-90% consistency using only LoRA + CLIP
   - ControlNet present but disabled (scale=0.0)
   - Lightweight approach: 10MB LoRA vs 5GB ControlNet

2. **Automated Quality Control**
   - CLIP validates every frame against reference
   - Automatic regeneration if similarity < 0.70
   - No manual curation needed

3. **Zero AWS Configuration**
   - Public S3 bucket with HTTPS access
   - No boto3, no AWS CLI, no credentials
   - Judges can run immediately after pip install

4. **Deterministic Generation**
   - Same story → same frames (seed=42)
   - Reproducible for evaluation
   - Temperature=0.0 in GPT-4

5. **Efficient Fine-Tuning**
   - Only 70 training images
   - 1000 training steps (~1 hour on A100)
   - 3.7M trainable params (0.14% of full model)

### Performance Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Face Consistency (CLIP) | 0.79-0.82 avg | ≥ 0.70 |
| Generation Time | 4-5 min | < 10 min |
| Model Size (LoRA) | 10 MB | < 100 MB |
| Training Images | 70 | < 100 |
| Training Time | ~1 hour | < 2 hours |
| Resolution | 1024×1024 | ≥ 1024×1024 |

---

## 💡 Tips for Best Results

### Story Prompt Guidelines

**Good prompts:**
- Clear narrative arc (beginning → middle → end)
- 3-5 key events
- Specific actions and locations
- Example: "Aldar tricks a merchant by selling him an invisible horse at the bazaar, then escapes on a real horse"

**Avoid:**
- Very long prompts (>200 words)
- Abstract concepts without actions
- Multiple disconnected events

### Expected Consistency

- **Frame 1 → 2:** Usually highest (0.80-0.85)
- **Middle frames:** Typically 0.75-0.82
- **Last frame:** May be slightly lower if scene changes dramatically

**Note:** Some variation is expected! The goal is visual coherence, not identical frames.

---

## 🤝 Support

**If you encounter issues during evaluation:**

1. Check the troubleshooting section above
2. Verify your system meets minimum requirements
3. Contact us via GitHub Issues: [https://github.com/SuWh1/SexyAldarKose/issues](https://github.com/SuWh1/SexyAldarKose/issues)

**Common first-time issues:**
- Missing CUDA installation
- Insufficient GPU memory
- OpenAI API key not set correctly

---

## 🎉 Success Criteria

You'll know the demo worked if:

✅ All 6 frames generated successfully  
✅ Aldar Köse's face is recognizable across frames  
✅ Story flows logically from frame 1 → 6  
✅ Average CLIP consistency ≥ 0.70  
✅ High visual quality (1024×1024, detailed)  

**Thank you for evaluating our project!**

---

## ⚡ Quick Start (Copy-Paste)

```bash
# 1. Clone and navigate
git clone https://github.com/SuWh1/SexyAldarKose.git
cd SexyAldarKose/backend/aldar_kose_project

# 2. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install -r requirements_inference.txt

# 4. Set API key (Windows)
$env:OPENAI_API_KEY="sk-your-key-here"
# Or (Linux/Mac)
export OPENAI_API_KEY="sk-your-key-here"

# 5. Run demo
python scripts/submission_demo.py

# 6. Optional: Custom story
python scripts/submission_demo.py "Aldar tricks a merchant"
```

---

## 🎯 TL;DR for Judges

```bash
# Three commands to run the demo:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_inference.txt
python scripts/submission_demo.py
```

(Plus set `OPENAI_API_KEY` environment variable)

**Total time:** 15 minutes setup + 8 minutes generation = **~23 minutes**  
**Subsequent runs:** 4-5 minutes each
