# API Quick Start Guide

## Prerequisites

1. **Install dependencies:**
```bash
pip install fastapi uvicorn pydantic requests
```

2. **Set environment variables:**
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-key-here"
$env:LORA_PATH="outputs/checkpoints/final"

# Linux/Mac
export OPENAI_API_KEY="sk-your-key-here"
export LORA_PATH="outputs/checkpoints/final"
```

---

## Start the Server

### Option 1: Direct Python (Simple)
```bash
cd backend/aldar_kose_project
python api/server.py
```

### Option 2: With Custom Settings
```bash
python api/server.py --host 0.0.0.0 --port 8000 --lora-path outputs/checkpoints/checkpoint-1000
```

### Option 3: Using Uvicorn (Production)
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

**Expected output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Loading LoRA model from outputs/checkpoints/final...
INFO:     Generator initialized successfully!
INFO:     Server ready!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## Test the API

### 1. Health Check (Browser)
Open: **http://localhost:8000/docs**

You'll see the Swagger UI with all endpoints.

### 2. Health Check (Command Line)
```bash
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "lora_path": "outputs/checkpoints/final",
  "ref_guided_available": true
}
```

### 3. Generate Story (Test Client)

**Simple mode (faster, good quality):**
```bash
python api/test_client.py \
  --prompt "Aldar Kose tricks a wealthy merchant and steals his horse"
```

**Reference-guided mode (better consistency):**
```bash
python api/test_client.py \
  --prompt "Aldar Kose riding across the steppe at sunset" \
  --use-ref-guided
```

**With custom settings:**
```bash
python api/test_client.py \
  --prompt "Aldar Kose tricks a merchant" \
  --seed 42 \
  --gpt-temperature 0.0 \
  --num-frames 8 \
  --output my_story
```

**Expected output:**
```
Testing health endpoint: http://localhost:8000/health
✓ Server is healthy
  Model loaded: True
  LoRA path: outputs/checkpoints/final
  Ref-guided available: True

============================================================
Generating story...
============================================================
Prompt: Aldar Kose tricks a merchant and steals his horse
Mode: simple
Seed: 42
GPT Temperature: 0.7
Max frames: auto

Sending request to API...

============================================================
Generation Complete!
============================================================
Success: True
Frames: 8
Time: 245.3s
Mode: simple

Saving frames to: outputs/api_test_simple

  Frame 1: Aldar standing in marketplace, front-facing (CLIP: 0.850)
  Frame 2: Aldar approaching merchant on horseback (CLIP: 0.820)
  Frame 3: Aldar negotiating with merchant, hand gestures (CLIP: 0.815)
  ... 

✓ All frames saved to: outputs/api_test_simple
✓ Metadata saved to: outputs/api_test_simple/metadata.json
```

### 4. Generate Story (cURL)

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Aldar Kose tricks a merchant and steals his horse",
    "use_ref_guided": false,
    "seed": 42,
    "gpt_temperature": 0.7
  }'
```

### 5. Generate Story (Python)

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# Send request
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Aldar Kose tricks a merchant and steals his horse",
        "use_ref_guided": False,
        "seed": 42,
        "gpt_temperature": 0.7
    }
)

data = response.json()

# Save frames
for frame in data["frames"]:
    img_data = base64.b64decode(frame["image"])
    img = Image.open(BytesIO(img_data))
    img.save(f"frame_{frame['frame_number']:03d}.png")
    print(f"Frame {frame['frame_number']}: {frame['prompt']}")
```

---

## API Parameters Explained

### Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | **required** | Story description (10-500 chars) |
| `use_ref_guided` | boolean | `false` | Use reference-guided mode for better consistency |
| `num_frames` | int | `null` | Max frames (6-10), null = GPT decides |
| `seed` | int | `42` | Random seed (same seed = same output) |
| `gpt_temperature` | float | `0.7` | GPT creativity (0.0=deterministic, 1.0=creative) |

### Response Fields

| Field | Type | Description |
|---|---|---|
| `success` | boolean | Generation succeeded |
| `story_prompt` | string | Original prompt |
| `num_frames` | int | Number of frames generated |
| `frames` | array | List of frame objects |
| `generation_time_seconds` | float | Total time taken |
| `mode` | string | "simple" or "ref-guided" |
| `seed` | int | Seed used |
| `gpt_temperature` | float | Temperature used |

### Frame Object

| Field | Type | Description |
|---|---|---|
| `frame_number` | int | Frame index (1-based) |
| `image` | string | Base64-encoded PNG |
| `prompt` | string | Scene description |
| `clip_score` | float | Quality score (0-1) |

---

## Common Use Cases

### 1. Deterministic Generation (Testing)
```bash
# Same output every time
python api/test_client.py \
  --prompt "Aldar Kose riding his horse" \
  --seed 123 \
  --gpt-temperature 0.0
```

### 2. Creative Exploration
```bash
# Different outputs each time
python api/test_client.py \
  --prompt "Aldar Kose adventures in the steppe" \
  --seed 456 \
  --gpt-temperature 1.0
```

### 3. Maximum Quality (Production)
```bash
# Best consistency
python api/test_client.py \
  --prompt "Aldar Kose tricks a merchant" \
  --use-ref-guided \
  --seed 42 \
  --gpt-temperature 0.7
```

### 4. Quick Iteration
```bash
# Fast generation for testing
python api/test_client.py \
  --prompt "Aldar on horseback" \
  --num-frames 6
```

---

## Troubleshooting

### Server won't start

**Issue**: `Generator not initialized`
```bash
# Check LORA_PATH exists
ls outputs/checkpoints/final

# Set environment variable
$env:LORA_PATH="outputs/checkpoints/final"  # Windows
export LORA_PATH="outputs/checkpoints/final"  # Linux/Mac
```

**Issue**: `OPENAI_API_KEY not set`
```bash
# Set API key
$env:OPENAI_API_KEY="sk-..."  # Windows
export OPENAI_API_KEY="sk-..."  # Linux/Mac
```

### Generation fails

**Issue**: `CUDA out of memory`
- Use simple mode (not ref-guided)
- Close other GPU applications
- Reduce num_frames

**Issue**: Slow generation
- Simple mode: ~30s per frame
- Ref-guided: ~35s per frame
- 8 frames ≈ 4-5 minutes total

### Test client errors

**Issue**: `Connection refused`
```bash
# Check server is running
curl http://localhost:8000/health

# Start server if not running
python api/server.py
```

**Issue**: Images not saving
```bash
# Check output directory permissions
# Default: outputs/api_test_simple/
```

---

## Performance Benchmarks

| Configuration | VRAM | Time/Frame | Total (8 frames) |
|---|---|---|---|
| Simple mode | 10GB | ~30s | ~4 min |
| Ref-guided mode | 18GB | ~35s | ~5 min |

**GPT-4 scene breakdown**: ~5-10 seconds

---

## Next Steps

1. **Test basic generation**: Run test client with default settings
2. **Try different seeds**: Compare outputs with seed=42 vs seed=123
3. **Adjust temperature**: Test temperature 0.0 vs 1.0
4. **Compare modes**: Simple vs ref-guided quality
5. **Integrate with frontend**: Use the API in your React app

---

## Quick Reference

```bash
# Start server
python api/server.py

# Test health
curl http://localhost:8000/health

# Generate story (simple)
python api/test_client.py --prompt "Your story here"

# Generate story (best quality)
python api/test_client.py \
  --prompt "Your story" \
  --use-ref-guided \
  --seed 42 \
  --gpt-temperature 0.7

# View API docs
Open: http://localhost:8000/docs
```

---

## Environment Setup (Complete)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Set environment variables
$env:OPENAI_API_KEY="sk-proj-..."
$env:LORA_PATH="outputs/checkpoints/final"

# 3. Verify LoRA model exists
ls outputs/checkpoints/final

# 4. Start server
python api/server.py

# 5. Test in new terminal
python api/test_client.py --prompt "Aldar Kose on horseback"
```

That's it! 🚀
