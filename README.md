# Aldar Köse AI Storyboard Generator

> Generate consistent, story-driven storyboards featuring Aldar Köse, the Kazakh folk hero, using SDXL + LoRA fine-tuning.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Quick Start

```bash
# 1. Clone and navigate
git clone https://github.com/SuWh1/SexyAldarKose.git
cd SexyAldarKose/backend/aldar_kose_project

# 2. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_inference.txt

# 3. Set API key
export OPENAI_API_KEY="sk-your-key"  # Linux/Mac
# OR
$env:OPENAI_API_KEY="sk-your-key"    # Windows

# 4. Run demo
python scripts/submission_demo.py
```

Outputs 6-frame storyboard

## 🌟 Features

- **85-90% Face Consistency** - LoRA fine-tuned on 70 images of Aldar Köse
- **Intelligent Scene Breakdown** - GPT-4 splits stories into coherent scenes
- **Automatic Quality Control** - CLIP validates visual consistency
- **100% Reproducible** - Fixed seed ensures same results
- **No AWS Configuration** - Public S3 downloads, zero setup

## 📊 Example Output

**Input:** *"Aldar Kose tricks a greedy merchant in the bazaar"*

**Output:** 6 frames (1024×1024) showing:
1. Aldar approaches merchant
2. Aldar talks to merchant
3. Aldar shows goods
4. Aldar negotiates price
5. Aldar receives payment
6. Aldar leaves triumphant

## 🎓 How It Works

```
Story Prompt → GPT-4 Scene Breakdown → SDXL + LoRA Generation → CLIP Validation → 6 Frames
```

1. **GPT-4** breaks story into 6 scenes
2. **SDXL + LoRA** generates each frame (50 denoising steps)
3. **CLIP** validates face consistency (≥0.70 threshold)
4. **Auto-retry** if consistency too low

## 📚 Documentation

- **[SETUP.md](SETUP.md)** - Quick setup guide
- **[JUDGES_QUICKSTART.md](JUDGES_QUICKSTART.md)** - Comprehensive guide for judges
- **[ARCHITECTURE_FLOW.md](ARCHITECTURE_FLOW.md)** - System architecture
- **[TEXT_TO_IMAGE_TRANSFORMATION.md](backend/aldar_kose_project/TEXT_TO_IMAGE_TRANSFORMATION.md)** - Training/inference details

## 🔧 Requirements

- Python 3.10 or 3.11
- NVIDIA GPU with 18GB+ VRAM
- 30GB free disk space
- OpenAI API key

## 🎨 Technical Highlights

- **Model:** SDXL base (stabilityai/stable-diffusion-xl-base-1.0)
- **Fine-tuning:** LoRA (rank=64, 3.7M params, 0.14% of full model)
- **Training:** 70 images, 1000 steps, ~1 hour on A100
- **Consistency:** CLIP (openai/clip-vit-large-patch14)
- **Resolution:** 1024×1024 native

## 📈 Performance

| Metric | Value |
|--------|-------|
| Face Consistency | 85-90% (CLIP ≥0.70) |
| Generation Time | 4-5 min (cached), 8-9 min (first run) |
| Model Size | 10 MB (LoRA only) |
| Training Time | ~1 hour (A100) |

## 🤝 Contributing

This is a hackathon submission project. Issues and PRs welcome!

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- Stable Diffusion XL by Stability AI
- LoRA by Microsoft
- CLIP by OpenAI
- Aldar Köse - Kazakh folk hero

---

**Made with ❤️ for preserving cultural heritage through AI**
