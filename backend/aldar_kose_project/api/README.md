# Aldar Kose Storyboard Generation API

REST API server for generating multi-frame story sequences using fine-tuned SDXL + LoRA.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pydantic
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Start Server

```bash
# Simple command
python api/server.py --lora-path outputs/checkpoints/final

# Or with uvicorn directly
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 2
```

**Environment Variables:**
- `LORA_PATH`: Path to LoRA checkpoint (default: `outputs/checkpoints/final`)

### 3. Test API

Open browser: http://localhost:8000/docs

---

## API Endpoints

### GET `/` or `/health`
Health check

**Response:**
```json
{
  "status": "running",
  "model_loaded": true,
  "lora_path": "outputs/checkpoints/final",
  "ref_guided_available": true
}
```

### POST `/generate`
Generate story from prompt

**Request:**
```json
{
  "prompt": "Aldar Kose tricks a wealthy merchant and steals his horse",
  "use_ref_guided": false,
  "num_frames": null,
  "seed": 42,
  "gpt_temperature": 0.7
}
```

**Parameters:**
- `prompt` (string, required): Story description (10-500 chars)
- `use_ref_guided` (boolean, optional): Use reference-guided mode for better consistency (default: false)
- `num_frames` (int, optional): Max frames to generate (6-10), null = GPT decides (default: null)
- `seed` (int, optional): Random seed for reproducibility (default: 42)
  - **Same seed + same prompt = same output** (deterministic)
- `gpt_temperature` (float, optional): GPT-4 creativity level (0.0-1.0, default: 0.7)
  - `0.0` = Deterministic, consistent scene breakdowns
  - `0.7` = Balanced creativity (default)
  - `1.0` = Maximum creativity, varied outputs

**Response:**
```json
{
  "success": true,
  "story_prompt": "Aldar Kose tricks a merchant and steals his horse",
  "num_frames": 8,
  "frames": [
    {
      "frame_number": 1,
      "image": "iVBORw0KGgoAAAANSUhEUgAA...",  // Base64 PNG
      "prompt": "Aldar standing in marketplace, front-facing",
      "clip_score": 0.85
    },
    {
      "frame_number": 2,
      "image": "iVBORw0KGgoAAAANSUhEUgAA...",
      "prompt": "Aldar approaching merchant on horseback",
      "clip_score": 0.82
    }
    // ... more frames
  ],
  "generation_time_seconds": 245.3,
  "mode": "simple"
}
```

---

## Usage Examples

### cURL

```bash
# Simple generation
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Aldar Kose tricks a merchant and steals his horse",
    "use_ref_guided": false,
    "seed": 42
  }'

# Reference-guided mode
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Aldar Kose riding across the steppe",
    "use_ref_guided": true,
    "num_frames": 8
  }'
```

### Python Client

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# Generate story
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Aldar Kose tricks a merchant and steals his horse",
        "use_ref_guided": False,
        "seed": 42
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

### JavaScript/Fetch

```javascript
const response = await fetch('http://localhost:8000/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: 'Aldar Kose tricks a merchant and steals his horse',
    use_ref_guided: false,
    seed: 42
  })
});

const data = await response.json();

// Display frames
data.frames.forEach(frame => {
  const img = document.createElement('img');
  img.src = `data:image/png;base64,${frame.image}`;
  document.body.appendChild(img);
});
```

---

## Performance

| Mode | VRAM | Time per Frame | Total (8 frames) |
|---|---|---|---|
| **Simple** | 10GB | ~30s | ~4 minutes |
| **Ref-Guided** | 18GB | ~35s | ~5 minutes |

**GPT-4 Scene Breakdown**: ~5-10 seconds

---

## Modes Comparison

### Simple Mode (Default)
- **Use when**: Fast generation, good diversity
- **VRAM**: 10GB
- **Consistency**: 85% (LoRA + CLIP validation)
- **Speed**: Faster
- **Best for**: Quick tests, production with good enough quality

### Reference-Guided Mode
- **Use when**: Maximum face consistency needed
- **VRAM**: 18GB
- **Consistency**: 90% (LoRA + IP-Adapter 20% + CLIP)
- **Speed**: Slower
- **Best for**: Final outputs, competition judging

**Recommendation**: Start with simple mode. Use ref-guided only if face consistency is critical.

---

## Error Handling

**Common Errors:**

1. **Generator not initialized**
   ```json
   {"detail": "Generator not initialized. Set LORA_PATH environment variable."}
   ```
   **Fix**: Set `LORA_PATH` or pass `--lora-path` when starting server

2. **Out of memory**
   ```json
   {"detail": "Generation failed: CUDA out of memory"}
   ```
   **Fix**: Use simple mode or reduce batch size

3. **Invalid prompt**
   ```json
   {"detail": "prompt must be at least 10 characters"}
   ```
   **Fix**: Provide longer, descriptive story prompt

---

## Production Deployment

### With Docker

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV LORA_PATH=outputs/checkpoints/final
EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t aldar-kose-api .
docker run -p 8000:8000 --gpus all aldar-kose-api
```

### With Gunicorn (multiple workers)

```bash
gunicorn api.server:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 600
```

### Environment Variables

```bash
export LORA_PATH=outputs/checkpoints/final
export OPENAI_API_KEY=sk-...  # For GPT-4 scene breakdown
```

---

## Monitoring

### Logs

Server logs include:
- Request details (prompt, mode, seed)
- GPT-4 scene breakdown (frame count)
- Generation progress
- CLIP scores
- Total generation time

### Metrics

Track:
- Generation time per request
- Frame count distribution
- CLIP score averages
- Error rates

---

## Development

### Run with Auto-Reload

```bash
python api/server.py --reload
```

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Testing

```bash
# Health check
curl http://localhost:8000/health

# Test generation
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Aldar Kose riding his horse", "seed": 123}'
```

---

## Architecture

```
Client Request
    ↓
FastAPI Server (api/server.py)
    ↓
Break down story with GPT-4 (scripts/prompt_storyboard.py)
    ↓
Generate frames (scripts/simple_storyboard.py or ref_guided_storyboard.py)
    ↓
SDXL + LoRA + CLIP validation
    ↓
Convert images to Base64
    ↓
Return JSON response
```

---

## Troubleshooting

**Server won't start:**
- Check Python version (3.10+)
- Install dependencies: `pip install -r requirements.txt`
- Verify LoRA path exists

**Slow generation:**
- Use simple mode instead of ref-guided
- Check GPU utilization: `nvidia-smi`
- Reduce num_frames

**Poor quality:**
- Try ref-guided mode
- Adjust seed value
- Improve prompt clarity

---

## License

Part of the Aldar Kose project.
