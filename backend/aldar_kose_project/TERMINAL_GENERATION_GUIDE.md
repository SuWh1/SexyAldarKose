# Terminal Story Generation Guide

## Quick Start

### Simple Generation (with defaults)
Just provide a prompt - everything else uses defaults:

```bash
cd backend/aldar_kose_project
python scripts/generate_story.py "Aldar Kose tricks a wealthy merchant and steals his horse"
```

**Default Settings:**
- Mode: Simple (faster)
- Seed: 42
- Temperature: 0.7 (balanced creativity)
- Frames: Auto (GPT decides)
- LoRA: `outputs/checkpoints/final`

---

## 🔒 Fully Deterministic Output

To get **EXACTLY THE SAME OUTPUT** every time, use:

```bash
python scripts/generate_story.py "Aldar riding across the steppe at sunset" --seed 42 --temp 0.0
```

**Key Points:**
- `--seed 42` - Sets random seed for SDXL image generation
- `--temp 0.0` - Makes GPT-4 scene breakdown deterministic
- Same prompt + same seed + temp 0.0 = **IDENTICAL OUTPUT**

### Example: Deterministic Generation

```bash
# Run 1
python scripts/generate_story.py "Aldar tricks a merchant" --seed 123 --temp 0.0 --output run1

# Run 2 (will produce IDENTICAL frames)
python scripts/generate_story.py "Aldar tricks a merchant" --seed 123 --temp 0.0 --output run2

# Compare the two - they should be pixel-perfect identical!
```

---

## Common Use Cases

### 1. Quick Test (Default Everything)
```bash
python scripts/generate_story.py "Aldar on horseback"
```

### 2. Deterministic for Production
```bash
python scripts/generate_story.py "Aldar at the bazaar" --seed 42 --temp 0.0
```

### 3. Creative Exploration
```bash
python scripts/generate_story.py "Aldar adventures" --seed 999 --temp 1.0
```

### 4. Reference-Guided (Better Face Consistency)
```bash
python scripts/generate_story.py "Aldar tricks merchant" --ref-guided --seed 42 --temp 0.0
```

### 5. Custom Frame Count
```bash
python scripts/generate_story.py "Aldar story" --frames 8 --seed 42
```

### 6. Custom Output Location
```bash
python scripts/generate_story.py "Aldar adventures" --output my_story_name --seed 42
```

### 7. Different LoRA Checkpoint
```bash
python scripts/generate_story.py "Aldar tricks merchant" --lora-path outputs/checkpoints/checkpoint-400
```

---

## All Command-Line Options

```bash
python scripts/generate_story.py PROMPT [OPTIONS]

Required:
  PROMPT                 Story description (in quotes if it has spaces)

Optional:
  --seed INT            Random seed (default: 42)
                        Same seed = reproducible output
  
  --temp FLOAT          GPT temperature 0.0-1.0 (default: 0.7)
  --temperature FLOAT   0.0 = deterministic, 1.0 = very creative
  
  --frames INT          Number of frames 6-10 (default: auto)
                        If not set, GPT decides based on story
  
  --ref-guided          Use reference-guided mode
                        Better face consistency, slightly slower
  
  --output NAME         Output directory name (default: auto timestamp)
                        Saves to outputs/NAME/
  
  --lora-path PATH      LoRA checkpoint path
                        (default: outputs/checkpoints/final)
```

---

## Understanding Temperature

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| `0.0` | Fully deterministic | Production, reproducibility |
| `0.3` | Slightly creative | Consistent but varied |
| `0.7` | **Balanced (default)** | Good mix of consistency + variety |
| `1.0` | Very creative | Exploration, unique stories |

---

## Understanding Seeds

The `--seed` parameter controls the random number generator for SDXL image generation:

- **Same seed + same scene prompt = same image**
- **Different seed = different variations**
- Default is `42` (you can use any integer)

**Example:**
```bash
# These will produce different images (different seeds)
python scripts/generate_story.py "Aldar on horse" --seed 1 --temp 0.0
python scripts/generate_story.py "Aldar on horse" --seed 2 --temp 0.0

# These will produce IDENTICAL images (same seed + same temp)
python scripts/generate_story.py "Aldar on horse" --seed 42 --temp 0.0
python scripts/generate_story.py "Aldar on horse" --seed 42 --temp 0.0
```

---

## Output Location

By default, frames are saved to:
```
outputs/terminal_generation_YYYYMMDD_HHMMSS/
  frame_001.png
  frame_002.png
  ...
  metadata.json
```

With `--output my_story`:
```
outputs/my_story/
  frame_001.png
  frame_002.png
  ...
  metadata.json
```

---

## Performance

| Mode | VRAM | Time/Frame | Total (8 frames) |
|------|------|------------|------------------|
| Simple | 10GB | ~30s | ~4 minutes |
| Ref-guided | 18GB | ~35s | ~5 minutes |

**Plus GPT-4 scene breakdown: ~5-10 seconds**

---

## Troubleshooting

### Error: LoRA checkpoint not found
```bash
# Check if the checkpoint exists
ls outputs/checkpoints/final

# Or use a different checkpoint
python scripts/generate_story.py "prompt" --lora-path outputs/checkpoints/checkpoint-400
```

### Error: CUDA out of memory
```bash
# Use simple mode (not ref-guided)
python scripts/generate_story.py "prompt" --seed 42

# Or reduce number of frames
python scripts/generate_story.py "prompt" --frames 6
```

### Error: OpenAI API key not set
```bash
# Set environment variable (Windows PowerShell)
$env:OPENAI_API_KEY="sk-your-key-here"

# Or (Linux/Mac)
export OPENAI_API_KEY="sk-your-key-here"
```

---

## Examples with Full Context

### Example 1: Quick Test
```bash
python scripts/generate_story.py "Aldar Kose tricks a greedy merchant"
```
Output: `outputs/terminal_generation_20250119_143052/`

### Example 2: Deterministic Production Run
```bash
python scripts/generate_story.py "Aldar Kose riding through the mountains at dawn" --seed 42 --temp 0.0 --output mountain_scene
```
Output: `outputs/mountain_scene/` (reproducible with same command)

### Example 3: High-Quality Ref-Guided
```bash
python scripts/generate_story.py "Aldar Kose at the village festival" --ref-guided --seed 100 --temp 0.5 --frames 8 --output festival_story
```
Output: `outputs/festival_story/` with better face consistency

### Example 4: Creative Exploration
```bash
python scripts/generate_story.py "Aldar Kose magical adventure in the desert" --seed 777 --temp 1.0 --frames 10
```
Output: Highly creative, unique interpretation

---

## Quick Reference Card

```bash
# Simplest (defaults)
python scripts/generate_story.py "Your prompt here"

# Deterministic (exact same output every time)
python scripts/generate_story.py "Your prompt" --seed 42 --temp 0.0

# Best quality (ref-guided + deterministic)
python scripts/generate_story.py "Your prompt" --ref-guided --seed 42 --temp 0.0

# Creative exploration
python scripts/generate_story.py "Your prompt" --temp 1.0 --seed 999
```

---

## Help

To see all options:
```bash
python scripts/generate_story.py --help
```
