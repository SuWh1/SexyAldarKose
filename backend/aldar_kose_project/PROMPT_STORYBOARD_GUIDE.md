# Prompt-Based Storyboard Generation

Generate complete storyboards (6-10 frames) from a simple text description using OpenAI API + SDXL LoRA.

## Overview

This system takes a user's story description (e.g., "Aldar Kose riding his horse to his yurt at sunset") and automatically:

1. **Breaks down the story** into 6-10 distinct scenes using GPT-4
2. **Creates optimized prompts** for SDXL image generation
3. **Generates images** using your trained LoRA model
4. **Validates consistency** using CLIP similarity

## Quick Start

### 1. Test Story Breakdown (No GPU needed)

Test the story breakdown logic before generating images:

```bash
cd backend/aldar_kose_project
python scripts/test_prompt_storyboard.py
```

This will:
- Show you the scene breakdown
- Verify camera angles and variety
- Save the breakdown to `test_scene_breakdown.json`
- **No image generation** - just validates the story structure

### 2. Generate Full Storyboard

Once you're happy with the scene breakdown, generate images:

```bash
# Interactive mode - will prompt you for the story
python scripts/prompt_storyboard.py \
    --lora-path outputs/checkpoints/checkpoint-400

# Direct story mode
python scripts/prompt_storyboard.py \
    --lora-path outputs/checkpoints/checkpoint-400 \
    --story "Aldar Kose riding his horse across the steppe to his yurt at sunset" \
    --num-frames 8

# Custom output directory
python scripts/prompt_storyboard.py \
    --lora-path outputs/checkpoints/checkpoint-400 \
    --story "Aldar Kose tricks a wealthy merchant" \
    --num-frames 10 \
    --output-dir outputs/merchant_trick_story
```

## How It Works

### Architecture

```
User Story
    ↓
[OpenAI GPT-4] ← Scene breakdown rules
    ↓
Scene Descriptions (6-10)
    ↓
[Prompt Refinement] ← Add quality tags, trigger token
    ↓
SDXL-Optimized Prompts
    ↓
[SimplifiedStoryboardGenerator] ← Your trained LoRA
    ↓
Generated Images + CLIP Validation
    ↓
Final Storyboard
```

### Scene Breakdown Rules

The system uses GPT-4 with specific rules to ensure good storyboards:

**✅ DO:**
- Show story PROGRESSION (not repetitive scenes)
- Vary camera angles (wide, close-up, medium, POV)
- Include lighting and mood
- Focus on ACTION and SETTING
- Use "aldar_kose_man" as character identifier

**❌ DON'T:**
- Mention clothing, outfits, costumes
- Describe physical appearance
- Repeat similar scenes
- Use static descriptions

### Example Transformation

**Input Story:**
```
Aldar Kose riding his horse across the steppe to his yurt at sunset
```

**Generated Scenes:**
1. Wide shot: Vast steppe landscape, small figure on horseback, golden hour
2. Medium shot: aldar_kose_man riding horse, dynamic movement, wind effect
3. Close-up: aldar_kose_man's expression, determined, sunset lighting
4. POV shot: View of yurt appearing on horizon from horseback
5. Medium shot: Approaching yurt, horse slowing, warm orange sky
6. Wide shot: aldar_kose_man dismounting at yurt entrance, dramatic silhouette
7. Close-up: Hand opening yurt door, welcoming interior light
8. Medium shot: aldar_kose_man entering yurt, satisfied expression, home

## Output Structure

After generation, you'll find:

```
outputs/prompt_storyboard_TIMESTAMP/
├── scene_breakdown.json       # Original story + scene descriptions
├── sdxl_prompts.json          # Refined prompts for SDXL
├── frame_001.png              # Generated images
├── frame_002.png
├── ...
└── storyboard_grid.png        # All frames in a grid (if generated)
```

### scene_breakdown.json
```json
{
  "story": "Aldar Kose riding his horse...",
  "timestamp": "2025-10-18T...",
  "num_frames": 8,
  "scenes": [
    {
      "frame": 1,
      "description": "aldar_kose_man riding horse across vast steppe...",
      "camera": "wide shot",
      "mood": "adventurous"
    },
    ...
  ]
}
```

### sdxl_prompts.json
```json
[
  "aldar_kose_man riding horse across vast steppe, wide establishing shot, golden hour lighting, high quality, detailed, 3D animation, professional render",
  ...
]
```

## Configuration Options

### Story Parameters

```bash
--story "Your story here"           # Story description
--num-frames 8                       # 6-10 frames (default: 8)
```

### Model Parameters

```bash
--lora-path outputs/checkpoints/checkpoint-400   # LoRA checkpoint
--device cuda                                     # cuda or cpu
```

### Generation Parameters

```bash
--base-seed 42                      # Base seed for reproducibility
--num-inference-steps 40            # SDXL steps (default: 40)
--guidance-scale 7.5                # Guidance scale (default: 7.5)
```

### Output Parameters

```bash
--output-dir outputs/my_story       # Custom output directory
```

## API Requirements

### OpenAI API Key

Set your OpenAI API key:

```bash
# Option 1: Environment variable (recommended)
export OPENAI_API_KEY=sk-...

# Option 2: Command line argument
python scripts/prompt_storyboard.py \
    --api-key sk-... \
    --lora-path outputs/checkpoints/checkpoint-400
```

### Model Requirements

- **GPT-4 Turbo**: For scene breakdown (uses `gpt-4-turbo-preview`)
- **API Cost**: ~$0.01-0.03 per storyboard (scene breakdown only)

## Example Stories

### Simple Journey
```bash
python scripts/prompt_storyboard.py \
    --lora-path outputs/checkpoints/checkpoint-400 \
    --story "Aldar Kose walking through a village marketplace at midday"
```

### Trickster Tale
```bash
python scripts/prompt_storyboard.py \
    --lora-path outputs/checkpoints/checkpoint-400 \
    --story "Aldar Kose tricks a wealthy merchant by selling him a magical pot" \
    --num-frames 10
```

### Musical Performance
```bash
python scripts/prompt_storyboard.py \
    --lora-path outputs/checkpoints/checkpoint-400 \
    --story "Aldar Kose playing music for villagers around a campfire under the stars"
```

## Workflow Recommendations

### 1. Test First (Free)

Run `test_prompt_storyboard.py` to verify scene breakdown **before** using GPU:

```bash
python scripts/test_prompt_storyboard.py
# Review the scene breakdown
# Adjust story if needed
# No GPU or image generation costs
```

### 2. Generate Storyboard (GPU)

Once happy with scenes, generate images:

```bash
python scripts/prompt_storyboard.py \
    --lora-path outputs/checkpoints/checkpoint-400 \
    --story "Your story here"
```

### 3. Iterate

If you need different scenes:
- Adjust the story description (be more specific)
- Change `--num-frames` (6-10)
- Try different `--base-seed` values

## Troubleshooting

### OpenAI API Errors

**Error: "OpenAI API key not found"**
```bash
# Set the environment variable
export OPENAI_API_KEY=sk-...

# Or use --api-key flag
python scripts/prompt_storyboard.py --api-key sk-...
```

**Error: "Rate limit exceeded"**
- Wait a few seconds and retry
- Check your OpenAI account has available credits

### Scene Breakdown Issues

**Scenes are too similar:**
- Make your story more detailed
- Specify different locations/actions
- Example: Instead of "Aldar Kose riding horse", use "Aldar Kose riding horse across steppe, stopping at river, then arriving at yurt"

**Scenes mention clothing:**
- This should be filtered by the system prompt
- If it happens, the OpenAI model didn't follow instructions
- Try regenerating with a different story phrasing

### Image Generation Issues

**Out of memory:**
```bash
# Reduce inference steps
--num-inference-steps 30

# Use smaller batch size (edit simple_storyboard.py)
```

**Images don't match character:**
- Check that LoRA is loaded correctly
- Verify trigger token "aldar_kose_man" is in prompts
- Check CLIP similarity threshold (default: 0.70)

**Images are too similar:**
- The system uses unique seeds per frame
- Check that txt2img is being used (not img2img)
- Verify in `simple_storyboard.py`

## Integration with Frontend

### API Endpoint (Future)

You can wrap this in a FastAPI endpoint:

```python
from fastapi import FastAPI
from prompt_storyboard import PromptStoryboardGenerator

app = FastAPI()

@app.post("/generate-storyboard")
async def generate_storyboard(story: str, num_frames: int = 8):
    generator = PromptStoryboardGenerator(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        lora_path="outputs/checkpoints/checkpoint-400"
    )
    
    frames = generator.generate_storyboard(
        story=story,
        num_frames=num_frames
    )
    
    # Return image URLs or base64
    return {"frames": frames, "status": "success"}
```

## Performance

### Timing Estimates

On **H100 80GB GPU**:

- **Scene breakdown**: ~3-5 seconds (OpenAI API)
- **Image generation**: ~2-3 seconds per frame (SDXL + LoRA)
- **Total for 8 frames**: ~20-30 seconds

On **A100 40GB GPU**:

- **Image generation**: ~4-5 seconds per frame
- **Total for 8 frames**: ~35-45 seconds

### Optimization Tips

1. **Parallel generation**: Generate multiple frames in parallel (requires more VRAM)
2. **Reduce steps**: Use `--num-inference-steps 30` for faster generation
3. **Cache model**: Keep generator loaded for multiple requests
4. **Batch processing**: Process multiple stories sequentially

## File Reference

- **`scripts/prompt_storyboard.py`**: Main script for prompt-based generation
- **`scripts/test_prompt_storyboard.py`**: Test scene breakdown without GPU
- **`scripts/simple_storyboard.py`**: Core image generation (txt2img approach)
- **`scripts/label_images.py`**: Caption generation (similar prompt philosophy)

## Related Documentation

- [`TRAINING_PIPELINE_REVIEW.md`](TRAINING_PIPELINE_REVIEW.md) - Full training process
- [`CAPTION_GUIDELINES.md`](CAPTION_GUIDELINES.md) - Caption quality guidelines
- [`VM_TRAINING_GUIDE.md`](VM_TRAINING_GUIDE.md) - RunPod training setup

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the test script output (`test_prompt_storyboard.py`)
3. Verify OpenAI API key and LoRA checkpoint path
4. Check GPU memory availability (`nvidia-smi`)
