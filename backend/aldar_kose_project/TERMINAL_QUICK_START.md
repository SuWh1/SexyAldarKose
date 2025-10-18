# Quick Terminal Generation Cheat Sheet

## ⚡ Fastest Way (One-Liner with Defaults)

```bash
cd backend/aldar_kose_project
python scripts/generate_story.py "Aldar Kose tricks a wealthy merchant and steals his horse"
```

That's it! Uses default settings:
- ✅ Seed: 42
- ✅ Temperature: 0.7
- ✅ Frames: Auto (GPT decides)
- ✅ Mode: Simple (faster)
- ✅ LoRA: checkpoint-1000

---

## 🔒 Fully Deterministic (Same Output Every Time)

```bash
python scripts/generate_story.py "Aldar riding across the steppe" --seed 42 --temp 0.0
```

**Key:** `--seed 42 --temp 0.0` = IDENTICAL output on every run

---

## 🎯 Most Common Commands

```bash
# Default everything
python scripts/generate_story.py "Your story prompt"

# Deterministic
python scripts/generate_story.py "Your prompt" --seed 42 --temp 0.0

# Reference-guided (better faces)
python scripts/generate_story.py "Your prompt" --ref-guided --seed 42

# Custom frames
python scripts/generate_story.py "Your prompt" --frames 8 --seed 42

# Named output
python scripts/generate_story.py "Your prompt" --output my_story --seed 42
```

---

## 📊 Quick Options Reference

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--seed` | any integer | 42 | Random seed for reproducibility |
| `--temp` | 0.0 - 1.0 | 0.7 | GPT creativity (0=deterministic, 1=creative) |
| `--frames` | 6 - 10 | auto | Number of frames to generate |
| `--ref-guided` | flag | off | Better face consistency (slower) |
| `--output` | any name | timestamp | Output folder name |

---

## 💡 Examples

```bash
# Quick test
python scripts/generate_story.py "Aldar on horseback"

# Production (deterministic)
python scripts/generate_story.py "Aldar tricks merchant" --seed 42 --temp 0.0 --output production_v1

# Creative exploration
python scripts/generate_story.py "Aldar magical adventure" --temp 1.0 --seed 999

# Best quality
python scripts/generate_story.py "Aldar at bazaar" --ref-guided --seed 42 --temp 0.0 --frames 8
```

---

## ⏱️ Time Estimates

- Simple mode: ~4 minutes (8 frames)
- Ref-guided: ~5 minutes (8 frames)

---

## 📁 Output Location

Default: `outputs/terminal_generation_YYYYMMDD_HHMMSS/`
Custom: `outputs/YOUR_NAME/` (use `--output YOUR_NAME`)

---

## 🆘 Help

```bash
python scripts/generate_story.py --help
```

---

## 🚀 Copy-Paste Ready Commands

```bash
# Windows PowerShell
cd C:\Users\aidyn\Desktop\PR\SexyAldarKose\backend\aldar_kose_project
python scripts/generate_story.py "Aldar Kose tricks a wealthy merchant" --seed 42 --temp 0.0

# Linux/Mac
cd ~/SexyAldarKose/backend/aldar_kose_project
python scripts/generate_story.py "Aldar Kose tricks a wealthy merchant" --seed 42 --temp 0.0
```
