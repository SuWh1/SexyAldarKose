# Aldar Köse Storyboard Generator - Architecture Flow

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    USER ENTRY POINT                                      │
│                  submission_demo.py                                      │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  1. DEPENDENCY CHECK                 │
        │  ✓ torch, transformers, diffusers   │
        │  ✓ peft, openai, boto3              │
        └──────────────────────┬───────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────┐
        │  2. OPENAI API KEY CHECK             │
        │  Load from .env file                │
        └──────────────────────┬───────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────┐
        │  3. AWS S3 MODEL DOWNLOAD            │
        │  boto3.client('s3').download()      │
        │  Downloads to:                      │
        │  outputs/checkpoints/checkpoint-1000│
        └──────────────────────┬───────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────┐
        │  4. CALL generate_story.py           │
        │  subprocess.run([                    │
        │    "python",                        │
        │    "scripts/generate_story.py",     │
        │    prompt,                          │
        │    "--seed=42",                     │
        │    "--temp=0.0",                    │
        │    "--lora-path=...",               │
        │    "--ref-guided"                   │
        │  ])                                 │
        └──────────────────────┬───────────────┘
                               │
                               ▼
                 ┌─────────────────────────────┐
                 │  GENERATION SUBPROCESS      │
                 │  generate_story.py          │
                 └──────────┬──────────────────┘
                            │
                ┌───────────┬───────────┐
                │           │           │
                ▼           ▼           ▼
    ┌──────────────────┐ ┌──────────┐ ┌──────────────┐
    │ STEP 1:          │ │ STEP 2:  │ │ STEP 3:      │
    │ Story Breakdown  │ │ Prompt   │ │ Load Model   │
    │ (GPT-4)          │ │ Refinement│ │ (SDXL+LoRA) │
    └────────┬─────────┘ └────┬─────┘ └──────┬───────┘
             │                │              │
             ▼                ▼              ▼
    ┌─────────────────────────────────────────────┐
    │ STEP 4: GENERATE STORYBOARD                 │
    │ prompt_storyboard.py                        │
    │ generate_storyboard()                       │
    └──────────────┬────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
    USE_REF_GUIDED        USE_SIMPLE
    = True                = False
        │                     │
        ▼                     ▼
┌─────────────────────┐ ┌──────────────────┐
│ ReferenceGuided     │ │ Simplified       │
│ StoryboardGen       │ │ StoryboardGen    │
│                     │ │                  │
│ • Load SDXL         │ │ • Load SDXL      │
│ • Load LoRA         │ │ • Load LoRA      │
│ • Load ControlNet   │ │ • No ControlNet  │
│ • Load CLIP         │ │ • Load CLIP      │
│                     │ │ • Load Anomaly   │
│                     │ │   Detector       │
└────────┬────────────┘ └────────┬─────────┘
         │                       │
         │ generate_sequence()   │ generate_sequence()
         │                       │
         ▼                       ▼
    ┌────────────────────────────────┐
    │ FOR EACH PROMPT (6 frames):    │
    │                                │
    │ Frame 1:                       │
    │  • generate_reference_frame()  │
    │  • Pure SDXL + LoRA            │
    │  • Save frame_001.png          │
    │                                │
    │ Frames 2-6:                    │
    │  • generate_guided_frame()     │
    │  • SDXL + LoRA                 │
    │  • CLIP validation loop:       │
    │    - Compute similarity        │
    │    - If < 0.70: retry         │
    │    - If >= 0.70: accept       │
    │  • Save frame_00X.png          │
    │                                │
    └────────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │ SAVE OUTPUTS:                  │
    │                                │
    │ outputs/terminal_generation_   │
    │  YYYYMMDD_HHMMSS/             │
    │  ├─ frame_001.png             │
    │  ├─ frame_002.png             │
    │  ├─ frame_003.png             │
    │  ├─ frame_004.png             │
    │  ├─ frame_005.png             │
    │  ├─ frame_006.png             │
    │  ├─ scene_breakdown.json      │
    │  ├─ sdxl_prompts.json         │
    │  └─ report.json               │
    └────────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │ RETURN TO submission_demo.py   │
    │ Find most recent output dir    │
    │ Print summary with path        │
    └────────────────────────────────┘
```

---

## Training Configuration

```
LoRA Training Config (from accelerate_config.yaml & training_config.yaml):

├─ Model:
│  ├─ base_model_name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"
│  └─ pretrained_model_name_or_path: same
│
├─ LoRA Hyperparameters:
│  ├─ rank: 64                    (low-rank matrices: 64×64)
│  ├─ alpha: 32                   (scaling factor)
│  ├─ dropout: 0.1                (regularization: 10% weights dropped)
│  ├─ target_modules: ["to_q", "to_v"]  (attention layers)
│  └─ lora_init_type: "gaussian"  (weight initialization)
│
├─ Training Parameters:
│  ├─ num_train_epochs: 1         (1 pass through 70 images)
│  ├─ max_train_steps: 1000       (1000 gradient updates)
│  ├─ train_batch_size: 4         (4 images per batch)
│  ├─ gradient_accumulation_steps: 1
│  ├─ learning_rate: 1e-4         (0.0001 - typical for LoRA)
│  ├─ lr_scheduler: "cosine"      (warmup then decay)
│  ├─ warmup_steps: 100           (first 100 steps: gradual increase)
│  └─ seed: 42                    (reproducibility)
│
├─ Data:
│  ├─ train_data_dir: "data/images"
│  ├─ caption_column: "text"      (from caption files)
│  ├─ resolution: [1024, 1024]    (SDXL native size)
│  ├─ center_crop: True           (crop to square)
│  └─ random_flip: False          (preserve Aldar's orientation)
│
├─ Optimization:
│  ├─ optimizer: "adamw"          (Adam with weight decay)
│  ├─ weight_decay: 0.01
│  ├─ max_grad_norm: 1.0          (gradient clipping)
│  └─ enable_xformers_memory_efficient_attention: True
│
├─ Checkpointing:
│  ├─ checkpointing_steps: 100    (save every 100 steps)
│  ├─ output_dir: "outputs/checkpoints"
│  └─ resume_from_checkpoint: None (start fresh)
│
└─ Hardware:
   ├─ accelerator: "gpu"
   ├─ device_ids: [0]             (first GPU)
   ├─ mixed_precision: "fp16"     (half-precision for speed)
   └─ compute_environment: "LOCAL_MACHINE"

Total Trainable Parameters (LoRA only): ~3.7M (0.14% of SDXL)
Full SDXL Parameters: ~2.6B
Non-trainable (frozen): ~2.596B
```

---

## Detailed Component Flow

### **TRAINING PIPELINE** (Not in submission, but context)

```
┌──────────────────────────────────────────────────────────────┐
│ train_lora_sdxl.py                                           │
└────────────────────────┬─────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    Load Models      Load Dataset    Setup LoRA
        │                │                │
        │                │                │
    ├─ SDXL         ├─ 70 images    ├─ LoRA config
    │   Base         │  + captions      │  rank=64
    │                │                 │  alpha=32
    ├─ Text          ├─ AldarKoseDataset │  dropout=0.1
    │  Encoders      │                 │
    │                │                 └─ get_peft_model()
    ├─ VAE           │                    Applied to:
    │                │                    • UNet
    ├─ UNet          │                    • Text Encoder 1
    │                │                    • Text Encoder 2
    └─ Scheduler     │
                     │
                     ▼
         ┌────────────────────────────┐
         │ TRAINING LOOP (1000 steps) │
         └───────────┬────────────────┘
                     │
         ┌───────────┴────────────────┐
         │ FOR step in 0..999:        │
         │                            │
         │ 1. Load batch (4 images)   │
         │ 2. Encode images → latents │
         │ 3. Add noise to latents    │
         │ 4. Tokenize + Encode text  │
         │ 5. UNet predicts noise     │
         │    (with LoRA)             │
         │ 6. Calculate MSE loss      │
         │ 7. Backprop → Update LoRA  │
         │ 8. Save checkpoint every   │
         │    100 steps               │
         │                            │
         └───────────┬────────────────┘
                     │
                     ▼
         ┌────────────────────────────┐
         │ INFERENCE LOOP (50 steps)  │
         │                            │
         │ FOR each of the prompts:   │
         │                            │
         │ 1. Tokenize text prompt    │
         │ 2. Encode text → embedding │
         │    [1, 77, 2048]           │
         │ 3. Create random noise     │
         │    [1, 4, 256, 256]        │
         │    (with fixed seed=42)    │
         │ 4. FOR denoising step in   │
         │    0..49:                  │
         │    • Compute timestep t    │
         │    • UNet predicts noise   │
         │      (with LoRA)           │
         │    • Remove predicted      │
         │      noise from latents    │
         │ 5. VAE decode clean        │
         │    latents → image         │
         │ 6. CLIP validate with      │
         │    Frame 1 (if not F1):    │
         │    • If similarity < 0.70: │
         │      retry with new seed   │
         │    • If similarity >= 0.70:│
         │      accept frame          │
         │ 7. Save frame PNG          │
         │                            │
         └────────────┬───────────────┘
                     │
                     ▼
         ┌────────────────────────────┐
         │ CHECKPOINT SAVED:          │
         │                            │
         │ outputs/checkpoints/       │
         │ checkpoint-1000/           │
         │  ├─ unet_lora/             │
         │  │  └─ adapter_model.bin   │
         │  ├─ text_encoder_one_lora/ │
         │  │  └─ adapter_model.bin   │
         │  └─ text_encoder_two_lora/ │
         │     └─ adapter_model.bin   │
         │                            │
         │ Total: 8-10 MB             │
         └────────────────────────────┘
```

---

### **INFERENCE PIPELINE** (What happens when you run submission_demo.py)

```
┌────────────────────────────────────┐
│ STAGE 1: INITIALIZATION            │
├────────────────────────────────────┤
│                                    │
│ 1. Check dependencies              │
│ 2. Verify OpenAI API key           │
│ 3. Download LoRA from AWS S3       │
│                                    │
└──────────────────┬─────────────────┘
                   │
                   ▼
┌────────────────────────────────────┐
│ STAGE 2: TEXT PROCESSING (GPT-4)   │
├────────────────────────────────────┤
│                                    │
│ User input: "Aldar tricks merchant"│
│          ↓                         │
│ GPT-4 breaks into 6 scenes:       │
│  1. "Aldar approaches merchant"   │
│  2. "Aldar talks to merchant"     │
│  3. "Aldar shows goods"           │
│  4. "Aldar negotiates price"      │
│  5. "Aldar takes payment"         │
│  6. "Aldar leaves triumphant"     │
│          ↓                         │
│ Save: scene_breakdown.json        │
│                                    │
└──────────────────┬─────────────────┘
                   │
                   ▼
┌────────────────────────────────────┐
│ STAGE 3: PROMPT REFINEMENT         │
├────────────────────────────────────┤
│                                    │
│ Each scene → SDXL prompt:         │
│  1. "aldar_kose_man approaching   │
│      merchant, bazaar, front-face"│
│  2. "aldar_kose_man talking to    │
│      merchant, negotiating"       │
│  ... (refined with detail)        │
│          ↓                         │
│ Save: sdxl_prompts.json           │
│                                    │
└──────────────────┬─────────────────┘
                   │
                   ▼
┌────────────────────────────────────┐
│ STAGE 4: MODEL LOADING             │
├────────────────────────────────────┤
│                                    │
│ Load from HuggingFace:            │
│  ✓ SDXL (335 MB)                 │
│  ✓ ControlNet (5 GB)             │
│  ✓ CLIP (338 MB)                 │
│                                    │
│ Load from Local:                   │
│  ✓ LoRA checkpoint (10 MB)        │
│                                    │
│ Total VRAM: ~18 GB                │
│                                    │
└──────────────────┬─────────────────┘
                   │
                   ▼
┌────────────────────────────────────┐
│ STAGE 5: FRAME GENERATION          │
├────────────────────────────────────┤
│                                    │
│ FOR each of 6 prompts:            │
│                                    │
│ FRAME 1:                          │
│  Input: seed=42, temp=0.0         │
│  Text: "aldar_kose_man portrait"  │
│          ↓                         │
│  Tokenize text (77 tokens)        │
│          ↓                         │
│  Encode text → [1,77,2048]        │
│          ↓                         │
│  Create noise tensor              │
│    randn(1,4,256,256, seed=42)   │
│          ↓                         │
│  DENOISING LOOP (50 steps):       │
│   for t in 999..0:                │
│     • UNet predicts noise         │
│     • Remove from latents         │
│     • Add scheduled noise         │
│          ↓                         │
│  Clean latents → VAE decode       │
│          ↓                         │
│  RGB Image (1024×1024)            │
│          ↓                         │
│  Save: frame_001.png              │
│                                    │
│ FRAMES 2-6:                       │
│  (Same as Frame 1, but...)        │
│  + Compute CLIP similarity        │
│    to Frame 1                     │
│  + If similarity < 0.70:          │
│    Regenerate with new seed       │
│  + If similarity >= 0.70:         │
│    Accept and move to Frame 3     │
│                                    │
└──────────────────┬─────────────────┘
                   │
                   ▼
┌────────────────────────────────────┐
│ STAGE 6: OUTPUT & VALIDATION       │
├────────────────────────────────────┤
│                                    │
│ Generate report.json:             │
│  {                                │
│    "base_seed": 42,              │
│    "num_frames": 6,              │
│    "pipeline": "reference_guided",│
│    "average_consistency": 0.82,  │
│    "min_consistency": 0.71,      │
│    "frames": [...]               │
│  }                                │
│                                    │
└──────────────────┬─────────────────┘
                   │
                   ▼
┌────────────────────────────────────┐
│ OUTPUT DIRECTORY STRUCTURE:        │
├────────────────────────────────────┤
│                                    │
│ outputs/                          │
│  └─ terminal_generation_          │
│     20251019_052851/             │
│      ├─ frame_001.png            │
│      ├─ frame_002.png            │
│      ├─ frame_003.png            │
│      ├─ frame_004.png            │
│      ├─ frame_005.png            │
│      ├─ frame_006.png            │
│      ├─ scene_breakdown.json     │
│      ├─ sdxl_prompts.json        │
│      └─ report.json              │
│                                    │
│ Total size: ~30 MB                │
│ (6 PNG frames)                    │
│                                    │
└────────────────────────────────────┘
```

---

## Per-Frame Generation Detail

```
SINGLE FRAME GENERATION (Frame 2):

Input:
  prompt = "aldar_kose_man talking to merchant, bazaar"
  seed = 42 + (2 * 1000) + (0 * 10) = 2042
  temperature = 0.0 (deterministic)

Step 1: TEXT ENCODING
┌─────────────────────────────┐
│ "aldar_kose_man talking..." │
│         ↓                   │
│ TOKENIZER_1:                │
│ [49496, 50221, ...]         │
│ (77 tokens max)             │
│         ↓                   │
│ TEXT_ENCODER_1:             │
│ [1, 77, 768]                │
│         ↓                   │
│ TOKENIZER_2:                │
│ [49496, 50221, ...]         │
│         ↓                   │
│ TEXT_ENCODER_2:             │
│ [1, 77, 1280]               │
│         ↓                   │
│ CONCATENATE:                │
│ [1, 77, 2048]               │
│ ↑                           │
│ TEXT EMBEDDING              │
└─────────────────────────────┘

Step 2: NOISE INITIALIZATION
┌─────────────────────────────┐
│ seed = 2042                 │
│ torch.Generator().          │
│   manual_seed(2042)         │
│         ↓                   │
│ torch.randn(                │
│   1, 4, 256, 256,           │
│   seed=2042                 │
│ )                           │
│         ↓                   │
│ latents = [1, 4, 256, 256]  │
│ (100% noise, t=999)         │
│ ↑                           │
│ INITIAL STATE               │
└─────────────────────────────┘

Step 3: DENOISING (50 steps)
┌─────────────────────────────┐
│ for t in range(999, -1, 20):│
│  # (20-step sampling)       │
│                             │
│  t=999: noise=100%, img=0%  │
│   UNet predicts noise       │
│   latents -= noise_pred     │
│                             │
│  t=979: noise=98%, img=2%   │
│   UNet predicts noise       │
│   latents -= noise_pred     │
│                             │
│  ...                        │
│                             │
│  t=19: noise=2%, img=98%    │
│   UNet predicts noise       │
│   latents -= noise_pred     │
│                             │
│  t=0: noise=0%, img=100%    │
│   (clean latents)           │
│                             │
│ ↑                           │
│ CLEAN LATENTS               │
│ [1, 4, 256, 256]            │
└─────────────────────────────┘

Step 4: VAE DECODE
┌─────────────────────────────┐
│ latents [1, 4, 256, 256]   │
│         ↓                   │
│ VAE.decode()                │
│         ↓                   │
│ Upscale 4×                  │
│ 256→1024                    │
│         ↓                   │
│ Reconstruct from features   │
│         ↓                   │
│ image [3, 1024, 1024]       │
│ ↑                           │
│ RGB IMAGE (PIL)             │
└─────────────────────────────┘

Step 5: CLIP VALIDATION
┌─────────────────────────────┐
│ image = frame_002           │
│ reference = frame_001       │
│         ↓                   │
│ CLIP.get_image_features()   │
│ on both images              │
│         ↓                   │
│ cosine_similarity()         │
│         ↓                   │
│ similarity = 0.72           │
│         ↓                   │
│ Is 0.72 >= 0.70?            │
│ YES → Accept frame          │
│ NO  → Regenerate with new   │
│       seed                  │
│                             │
│ ↑                           │
│ DECISION                    │
└─────────────────────────────┘

Output:
  ✓ frame_002.png saved
  ✓ consistency_score = 0.72 in report.json
  ✓ Ready for Frame 3
```

---

## Data Types & Shapes

```
THROUGHOUT THE PIPELINE:

Text:
  prompt (str): "aldar_kose_man talking to merchant"
  
After tokenization:
  input_ids_1 (torch): [1, 77]  (token IDs)
  input_ids_2 (torch): [1, 77]  (token IDs)

After text encoding:
  encoder_hidden_states (torch): [1, 77, 2048]
  pooled_embeds (torch): [1, 1280]

Images:
  pixel_values (PIL→torch): [1, 3, 1024, 1024] (RGB, float32)
  
After VAE encoding:
  latents (torch): [1, 4, 256, 256] (float16)
  
After adding noise:
  noisy_latents (torch): [1, 4, 256, 256]
  noise (torch): [1, 4, 256, 256]
  
UNet prediction:
  predicted_noise (torch): [1, 4, 256, 256]

After 50 denoising steps:
  clean_latents (torch): [1, 4, 256, 256]
  
After VAE decoding:
  decoded_image (torch): [1, 3, 1024, 1024]
  
Saved:
  frame.png (PIL Image): 1024×1024 RGB

CLIP similarity:
  similarity (float): 0.0-1.0
  (typically 0.70-0.85)
```

---

## Error Handling Flow

```
submission_demo.py

├─ Dependency check fails?
│  └─ Exit with error message
│     "pip install -r requirements.txt"
│
├─ OpenAI key missing?
│  └─ Exit with error message
│     "Set OPENAI_API_KEY in .env"
│
├─ S3 download fails?
│  └─ Check AWS credentials
│     "Run: aws configure"
│
├─ Model loading fails?
│  └─ Check disk space
│     "Need ~6 GB free"
│
├─ Generation fails mid-frame?
│  └─ CLIP validation error
│     Automatic retry with different seed
│
└─ Output directory creation fails?
   └─ Check write permissions
      "outputs/ directory writable?"

If any step fails:
  sys.exit(1)
  (clean exit with error code)
```

---

## Execution Timeline

```
submission_demo.py runs:

T+0s:   Start script
T+1s:   Check dependencies
T+2s:   Verify OpenAI API key
T+5s:   Download LoRA from S3 (varies by size)
T+10s:  Call generate_story.py subprocess
        ├─ T+15s: Break story into 6 scenes (GPT-4)
        ├─ T+20s: Refine prompts for SDXL
        ├─ T+30s: Load SDXL (335 MB)
        ├─ T+40s: Load ControlNet (5 GB)
        ├─ T+50s: Load CLIP (338 MB)
        ├─ T+60s: Load LoRA checkpoint (10 MB)
        └─ T+70s: Start generation
        
T+70s:  Frame 1 generation (50 steps)
        └─ T+4m30s: frame_001.png saved
        
T+4m30s-8m30s: Frames 2-6
        (with CLIP validation retries)
        
T+8m30s: Save scene_breakdown.json
T+8m35s: Save sdxl_prompts.json
T+8m40s: Save report.json

T+8m45s: Find output directory
T+8m46s: Print summary
T+8m47s: Exit

TOTAL: ~8-9 minutes (first run)
       ~4-5 minutes (subsequent runs, models cached)
```
