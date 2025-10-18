# Storyboard Generation Architecture

## Overview
The system uses a modular architecture with three layers for story generation.

## Architecture Layers

### Layer 1: Orchestration (User Interface)
**Files:**
- `scripts/prompt_storyboard.py` - Main orchestrator that handles GPT-4 story breakdown
- `scripts/generate_story.py` - Terminal CLI wrapper (simple interface)
- `api/server.py` - REST API wrapper (web interface)

**Responsibilities:**
- Accept user story prompts
- Break down story into 6-10 scenes using GPT-4
- Enforce consistency rules (visual descriptors, backgrounds, elements)
- Call appropriate generator based on mode
- Save results and metadata

### Layer 2: Image Generators (Rendering Engines)
**Files:**
- `scripts/simple_storyboard.py` - **DEFAULT** - Lightweight SDXL + LoRA generator
- `scripts/ref_guided_storyboard.py` - **OPTIONAL** - Advanced IP-Adapter + ControlNet generator

**Responsibilities:**
- Load SDXL + LoRA models
- Generate images from prompts
- Validate consistency with CLIP
- Retry on low quality/anomalies
- Return PIL Image objects

### Layer 3: Quality Assurance
**Files:**
- `scripts/anomaly_detector.py` - Detects double heads, size issues, pose problems

**Responsibilities:**
- Face detection (MediaPipe)
- Body keypoint validation
- Automatic regeneration suggestions
- Confidence scoring

---

## Generation Flow

```
User Prompt
    ↓
prompt_storyboard.py
    ├─→ GPT-4 Scene Breakdown (6-10 scenes)
    ├─→ Apply Consistency Rules
    │   ├─ Visual descriptors (brown horse, bearded merchant)
    │   ├─ Background consistency (steppe, bazaar)
    │   └─ Story element persistence (horse in all frames)
    ↓
Choose Generator Mode
    ├─→ simple_storyboard.py (default, lightweight)
    │   ├─ SDXL + LoRA
    │   ├─ CLIP validation
    │   └─ Anomaly detection
    │
    └─→ ref_guided_storyboard.py (--ref-guided flag)
        ├─ Frame 1: SDXL + LoRA (reference)
        ├─ Frame 2+: IP-Adapter (face) + ControlNet (pose)
        └─ Higher consistency, more VRAM
    ↓
Anomaly Detection
    ├─ Check faces (1 expected)
    ├─ Check pose validity
    └─ Suggest regeneration params if issues
    ↓
Save Results
    ├─ Images (frame_*.png)
    ├─ Metadata (scene_breakdown.json)
    └─ Prompts (sdxl_prompts.json)
```

---

## File Responsibilities

### `prompt_storyboard.py` (Main Orchestrator)
**Purpose:** Convert user story → consistent scene sequence
**Key Features:**
- GPT-4 scene breakdown with 10 critical rules
- Visual descriptor enforcement ("brown horse", "bearded merchant")
- Background consistency ("steppe background" in all frames)
- Frame 1 front-facing portrait requirement
- Temperature control (0.0-1.0 for determinism/creativity)

**Critical GPT-4 Rules:**
1. Frame 1 MUST be front-facing reference with ALL elements
2. Assign consistent visual descriptors (brown horse, bearded merchant)
3. Track story elements through ALL frames
4. Keep same background unless location changes
5. Use EXACT same descriptors in every frame

**Usage:**
```bash
# Direct usage (advanced)
python scripts/prompt_storyboard.py \
    --lora-path outputs/checkpoints/checkpoint-1000 \
    --story "Aldar Kose winning a race with his horse"

# Via terminal CLI (recommended)
python scripts/generate_story.py "Aldar racing his horse" --seed 42
```

---

### `simple_storyboard.py` (Default Generator)
**Purpose:** Lightweight image generation with anomaly detection
**Key Features:**
- SDXL + LoRA only (lower VRAM: ~10-12GB)
- CLIP similarity validation (threshold: 0.65)
- Integrated anomaly detection
- Automatic retry with adjusted params
- txt2img for all frames (not img2img)

**Why txt2img for all frames?**
- Each scene has different composition/action
- img2img would lock pose/layout
- txt2img allows: standing → riding → celebrating
- LoRA maintains character identity
- CLIP validates consistency

**Anomaly Handling:**
```python
# Detected: Multiple faces
# Action: Increase CFG to 8.5, change seed offset
# Reason: Low CFG allows composition drift

# Detected: No faces
# Action: Decrease CFG to 6.5, change seed offset  
# Reason: High CFG may overshoot prompt
```

**Usage:**
```python
generator = SimplifiedStoryboardGenerator(
    lora_path="outputs/checkpoints/checkpoint-1000",
    enable_anomaly_detection=True
)
frames = generator.generate_sequence(
    prompts=["aldar_kose_man with brown horse, steppe background"],
    base_seed=42,
    consistency_threshold=0.70,
    max_retries=3
)
```

---

### `ref_guided_storyboard.py` (Advanced Generator)
**Purpose:** Maximum consistency with IP-Adapter + ControlNet
**Key Features:**
- Frame 1: SDXL + LoRA (establishes identity)
- Frame 2+: IP-Adapter (face reference) + ControlNet (pose)
- Higher consistency (~85-90% vs 70-75%)
- Requires more VRAM (~16-20GB)
- Optional mode via --ref-guided flag

**Architecture:**
```
Frame 1 (Reference)
    ↓ (extract face encoding)
Frames 2-N
    ├─ Text Prompt → ControlNet (pose/depth)
    ├─ Reference Face → IP-Adapter (facial features)
    └─ Character Identity → LoRA
    ↓
Combined → High consistency with story variation
```

**Trade-offs:**
- ✅ Better face consistency (85-90%)
- ✅ More control over poses
- ❌ Higher VRAM (16-20GB)
- ❌ Slower generation
- ❌ Requires controlnet-aux, ip-adapter packages

**When to use:**
- Professional production quality needed
- Face recognition critical
- Have sufficient VRAM (16GB+)
- Willing to install extra dependencies

**Usage:**
```bash
python scripts/generate_story.py "Aldar at bazaar" --ref-guided
```

---

### `anomaly_detector.py` (Quality Assurance)
**Purpose:** Detect and fix generation anomalies
**Detects:**
- Multiple heads/faces (expects 1)
- Missing faces (expects at least 1)
- Invalid poses (MediaPipe validation)
- Size/composition issues

**Suggestions:**
```python
# Anomaly: Multiple faces detected
suggestions = {
    'seed': 52,  # original + 10
    'guidance_scale': 8.5,  # increased
    'reason': ['Low CFG caused composition drift']
}

# Anomaly: No faces detected
suggestions = {
    'seed': 52,
    'guidance_scale': 6.5,  # decreased
    'reason': ['High CFG overshot prompt']
}
```

**Integration:**
```python
# In simple_storyboard.py
if self.anomaly_detector:
    result = self.anomaly_detector.detect_anomalies(
        frame, 
        expected_prompt=prompt
    )
    if not result['is_valid']:
        suggestions = self.anomaly_detector.suggest_regeneration_params(
            result['anomalies'], seed, cfg
        )
        # Retry with adjusted params
```

---

## Removed Files

### ❌ `storyboard_generator.py` (Deprecated)
**Reason for removal:**
- NOT used by any active script
- Redundant with simple_storyboard.py + ref_guided_storyboard.py
- Complex, hard to maintain
- Old architecture before anomaly detection

**What it did:**
- Combined ControlNet + IP-Adapter (now in ref_guided_storyboard.py)
- Manual consistency pipeline (now automated)
- No anomaly detection

**Migration:**
- Features → moved to ref_guided_storyboard.py
- Use `--ref-guided` flag for advanced mode

---

## Usage Recommendations

### For Most Users (Default)
```bash
python scripts/generate_story.py "Your story" --seed 42 --temp 0.7
```
**Uses:** prompt_storyboard.py → simple_storyboard.py
**VRAM:** 10-12GB
**Speed:** ~5-8 sec/frame
**Consistency:** 70-75% CLIP similarity

### For High Quality (Advanced)
```bash
python scripts/generate_story.py "Your story" --ref-guided --seed 42
```
**Uses:** prompt_storyboard.py → ref_guided_storyboard.py
**VRAM:** 16-20GB
**Speed:** ~10-15 sec/frame
**Consistency:** 85-90% CLIP similarity

### For Deterministic Output
```bash
python scripts/generate_story.py "Your story" --seed 42 --temp 0.0
```
**Effect:** 
- `--seed 42`: Same image seed
- `--temp 0.0`: Deterministic GPT-4 (no randomness)
- Result: Identical output every time

---

## Key Parameters

### GPT-4 (Scene Breakdown)
- **temperature** (0.0-1.0): Creativity level
  - 0.0: Deterministic, same scenes every time
  - 0.7: Balanced (default)
  - 1.0: Maximum creativity
- **num_frames** (6-10): Max scenes (GPT decides optimal)

### SDXL (Image Generation)
- **seed**: Reproducibility (42 default)
- **guidance_scale** (5.0-10.0): Prompt adherence
  - 6.5: More freedom
  - 7.5: Balanced (default)
  - 8.5: Strict prompt following
- **num_inference_steps** (30-50): Quality vs speed
  - 30: Fast, lower quality
  - 40: Balanced (default)
  - 50: Slow, highest quality

### Consistency
- **consistency_threshold** (0.6-0.8): CLIP similarity
  - 0.65: Lenient (default)
  - 0.70: Balanced
  - 0.75: Strict (may retry often)
- **max_retries** (2-5): Attempts per frame
  - 2: Fast, may accept lower quality
  - 3: Balanced (default)
  - 5: Slow, ensures quality

---

## Dependencies

### Core (Required)
```bash
pip install torch diffusers transformers peft pillow openai python-dotenv
```

### Anomaly Detection (Recommended)
```bash
pip install opencv-python mediapipe
```

### Advanced Mode (Optional)
```bash
pip install controlnet-aux ip-adapter insightface onnxruntime
```

---

## Environment Setup

### Required Environment Variables
```bash
# .env file
OPENAI_API_KEY=sk-...your-key...
```

### Model Paths
```python
# Default LoRA checkpoint
LORA_PATH = "outputs/checkpoints/checkpoint-1000"

# Base SDXL model (cached)
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CACHE_DIR = "/root/.cache/huggingface/hub"
```

---

## Performance Metrics

### Simple Mode (Default)
- **VRAM Usage:** ~10-12GB
- **Generation Speed:** ~5-8 sec/frame
- **CLIP Consistency:** 70-75%
- **Anomaly Detection:** ✅ Enabled
- **Face Consistency:** 70-80%

### Ref-Guided Mode (Advanced)
- **VRAM Usage:** ~16-20GB
- **Generation Speed:** ~10-15 sec/frame
- **CLIP Consistency:** 85-90%
- **Anomaly Detection:** ❌ Not integrated yet
- **Face Consistency:** 85-90%

### Complete Story (6 frames)
- **Simple Mode:** ~30-50 seconds total
- **Ref-Guided Mode:** ~60-90 seconds total

---

## Troubleshooting

### Issue: Horse changing colors between frames
**Solution:** Visual descriptors now enforced
```python
# Old (generic)
"aldar_kose_man with horse"

# New (specific descriptor)
"aldar_kose_man with brown horse"  # Same in ALL frames
```

### Issue: Background jumping between scenes
**Solution:** Background consistency enforced
```python
# Each frame explicitly mentions setting
"aldar_kose_man riding, steppe background"
"aldar_kose_man celebrating, steppe background"
```

### Issue: Story elements disappearing
**Solution:** Element persistence enforced
```python
# Frame 1: Introduce ALL elements
"aldar_kose_man portrait with brown horse, steppe background"

# Frames 2-6: Keep same elements
"aldar_kose_man mounting brown horse, steppe background"
"aldar_kose_man riding brown horse, steppe background"
```

### Issue: Double heads/anomalies
**Solution:** Anomaly detection auto-retries
```python
# Detected: Multiple faces
# Action: Auto-retry with CFG=8.5, seed+10
# Usually fixes in 1-2 retries
```

---

## Future Enhancements

### Planned
- [ ] Integrate anomaly detection into ref_guided mode
- [ ] Add depth ControlNet option (alongside pose)
- [ ] Implement frame interpolation for smoother sequences
- [ ] Add video export (MP4 from frames)
- [ ] Fine-tune descriptor vocabulary (more horse types, merchant types)

### Under Consideration
- [ ] Multi-character support (Aldar + merchant in same frame)
- [ ] Location transitions (steppe → bazaar in story)
- [ ] Expression control (happy Aldar, angry merchant)
- [ ] Camera angle control (close-up, wide shot)

---

## Summary

**Keep Using:**
1. `prompt_storyboard.py` - Main orchestrator (GPT-4 + consistency rules)
2. `simple_storyboard.py` - Default generator (SDXL + LoRA + anomaly detection)
3. `ref_guided_storyboard.py` - Advanced generator (IP-Adapter + ControlNet)
4. `anomaly_detector.py` - Quality assurance
5. `generate_story.py` - Terminal CLI
6. `api/server.py` - REST API

**Removed:**
- `storyboard_generator.py` - Deprecated, unused, redundant

**Recommended Command:**
```bash
python scripts/generate_story.py "Aldar Kose winning a race with his horse" --seed 42 --temp 0.7
```

This uses the optimal architecture: GPT-4 scene breakdown → simple generator → anomaly detection → consistent output.
