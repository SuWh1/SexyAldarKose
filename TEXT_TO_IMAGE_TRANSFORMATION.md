# Text-to-Image Transformation: How LoRA Training & Inference Work

## High-Level Overview

```
INPUT TEXT + IMAGE PAIR (Training)
    ↓
TOKENIZATION (Convert text to numbers)
    ↓
TEXT ENCODING (Text → Semantic Embeddings)
    ↓
IMAGE ENCODING (Image → Latent Space)
    ↓
DIFFUSION PROCESS (Add noise, learn to remove it)
    ↓
LoRA ADAPTER (Small parameter update for Aldar)
    ↓
OPTIMIZED CHECKPOINT (3 files: unet, text_encoder_1, text_encoder_2)
```

---

## PART 1: TRAINING FLOW (How LoRA Learns "Aldar")

### Step 1: Input Preparation

```python
# From your training dataset:
TEXT:  "aldar_kose_man portrait, steppe background, front-facing"
IMAGE: <actual photo of Aldar>
```

### Step 2: Text Tokenization

```
INPUT TEXT: "aldar_kose_man portrait, steppe background"
    ↓
TOKENIZER_1 (CLIP-ViT-B):
    - "aldar_kose_man" → [49496, 50221, ...]  (each word is a number)
    - "portrait" → [1546, ...]
    - Total: ~77 tokens (padded)
    ↓
TOKENIZER_2 (CLIP-ViT-L):
    - Same text → different encoding
    - More detailed semantic representation
    ↓
RESULT: Two parallel token sequences
```

### Step 3: Text Encoding (Text → Embeddings)

```
TOKEN_SEQUENCE_1
    ↓
TEXT_ENCODER_1 (CLIP-ViT-B)
    ↓
EMBEDDING_1: [1, 77, 768]  (77 tokens, 768-dim vectors each)
    ↓
hidden_states[-2] → Use second-to-last layer
    ↓ (repeat for TEXT_ENCODER_2)
    ↓
EMBEDDING_2: [1, 77, 1280]  (77 tokens, 1280-dim vectors each)
    ↓
CONCATENATED: [1, 77, 2048]  (FULL TEXT EMBEDDING)

Also extract POOLED embedding (used for conditioning)
    ↓
POOLED: [1, 1280]  (summary of entire prompt)
```

**What this means:**
- The text is converted to a 2048-dimensional mathematical representation
- Each word/concept has a position in this space
- "aldar_kose_man" now has a unique semantic vector

### Step 4: Image Encoding (Image → Latent Space)

```
ORIGINAL IMAGE (RGB 1024×1024)
    ↓
VAE ENCODER
    ↓
Compress image into latent space:
    - 4× smaller: 256×256
    - But 4× more channels
    - Result: [1, 4, 256, 256]
    
WHY? 
- Much smaller to work with
- Works in compressed space (faster)
- VAE is trained to preserve important info
```

### Step 5: Add Noise (Diffusion Process)

```
CLEAN LATENTS: [1, 4, 256, 256]
    ↓
PICK RANDOM TIMESTEP: t=500 (out of 1000)
    ↓
ADD GAUSSIAN NOISE:
    NOISY = sqrt(1 - β_t) × CLEAN + sqrt(β_t) × GAUSSIAN_NOISE
    
    At t=500: ~50% clean, ~50% noise
    At t=0:   100% clean (no noise)
    At t=1000: ~100% noise (random)
    ↓
NOISY LATENTS: [1, 4, 256, 256] (with noise added)
```

### Step 6: UNet Predicts Noise (Core Training)

```
INPUT TO UNET:
├─ NOISY_LATENTS: [1, 4, 256, 256] (image with noise)
├─ TIMESTEP: t=500 (which step are we at?)
├─ TEXT_EMBEDDING: [1, 77, 2048] (semantic prompt)
└─ ADDED_CONDITIONS:
    ├─ pooled_embeds: [1, 1280]
    └─ time_ids: resolution info

    ↓
    UNet forward pass:
    - Self-attention: Process image features
    - Cross-attention: Incorporate text guidance
    - Convolutions: Refine predictions
    ↓

OUTPUT: PREDICTED_NOISE [1, 4, 256, 256]

GOAL: predicted_noise ≈ actual_noise
```

### Step 7: Where LoRA Fits In (The Magic)

```
ORIGINAL UNET LAYERS (880MB):
├─ Attention layers (key, query, value, out projection)
├─ FFN layers (feed-forward networks)
├─ Conv layers
└─ Total: 2.6 BILLION parameters

LORA ADAPTER (5-10MB):
├─ For each target layer:
│   ├─ LOW_RANK_A: [hidden_dim, rank=32]
│   ├─ LOW_RANK_B: [rank=32, hidden_dim]
│   └─ Computation: Output += α/r × B @ A @ Input
│
├─ Target modules (attention layers):
│   - q_proj (query projection)
│   - v_proj (value projection)
│   - out_proj (output projection)
│   - k_proj (key projection)
│
└─ Total trainable: 3.7 MILLION parameters (0.14%)

DURING INFERENCE:
    Standard UNet output + LoRA update
    = Output * (1 - LoRA_scale) + Output * LoRA_scale
    = Model adapted for Aldar!
```

**Why LoRA is genius:**
- Full model: 880MB (frozen)
- LoRA adapter: 5MB (trained)
- Combined: 880MB + 5MB = Works exactly like fine-tuned 880MB
- But only 0.5% extra parameters to update!

### Step 8: Calculate Loss (Learning Signal)

```
PREDICTED_NOISE: [1, 4, 256, 256]  (what UNet predicted)
ACTUAL_NOISE: [1, 4, 256, 256]     (noise we added in Step 5)

LOSS = Mean Squared Error (MSE):
    loss = mean((predicted - actual) ^ 2)

Optional SNR Weighting:
    - Some timesteps are easier/harder to predict
    - Weight loss by signal-to-noise ratio
    - Focus training on important timesteps

RESULT: Single scalar loss value
Example: loss = 0.0234
```

### Step 9: Backpropagation (Gradient Update)

```
loss = 0.0234
    ↓
Backprop through UNet
    ↓
ONLY LoRA parameters get gradients:
    ∂loss/∂LoRA_A, ∂loss/∂LoRA_B
    
All UNet parameters: FROZEN (no gradients)
    ↓
Update LoRA weights:
    LoRA_A ← LoRA_A - learning_rate × ∂loss/∂LoRA_A
    LoRA_B ← LoRA_B - learning_rate × ∂loss/∂LoRA_B

After 1000 training examples:
    LoRA learns patterns about Aldar
    (face features, clothing, pose)
```

### Step 10: Save Checkpoint

```
After N epochs of training:

SAVED FILES:
├─ unet_lora/
│   ├─ adapter_config.json (LoRA settings)
│   └─ adapter_model.safetensors (LoRA weights, 5MB)
├─ text_encoder_one_lora/
│   ├─ adapter_config.json
│   └─ adapter_model.safetensors (1-2MB)
├─ text_encoder_two_lora/
│   ├─ adapter_config.json
│   └─ adapter_model.safetensors (1-2MB)
└─ config.yaml (training config)

Total: ~8-10MB per checkpoint
```

---

## PART 2: INFERENCE FLOW (How Generation Happens)

### During Inference (Your `submission_demo.py`)

```
USER PROMPT: "Aldar Kose tricks a greedy merchant"
SEED: 42
TEMPERATURE: 0.0 (deterministic)
    ↓
STEP 1: Text Tokenization & Encoding
    (Same as training Step 2-3)
    
    Result: [1, 77, 2048] text embedding
    ↓
STEP 2: Prepare Latent Starting Point
    
    NOISE = torch.randn(1, 4, 256, 256, seed=42)
    
    With seed=42:
    └─> Same random tensor every time
        → DETERMINISTIC OUTPUT!
    ↓
STEP 3: Reverse Diffusion Loop (Denoising)
    
    for timestep in [999, 998, ..., 1, 0]:
        
        # Predict noise at this timestep
        predicted_noise = UNet(
            noisy_latents,           # Current noisy image
            timestep,                # Current step
            text_embedding,          # Your prompt guidance
            LoRA_weights             # Aldar-specific knowledge
        )
        
        # Remove predicted noise
        noisy_latents -= step_size × predicted_noise
        
        # Continue to next (less noisy) timestep
    
    After 50 steps:
    Result: Clean latents [1, 4, 256, 256]
    ↓
STEP 4: Decode Latents Back to Image
    
    CLEAN_LATENTS: [1, 4, 256, 256]
        ↓
    VAE DECODER:
        - Upscale 4× (256×256 → 1024×1024)
        - Reconstruct image from compressed form
        ↓
    OUTPUT IMAGE: [3, 1024, 1024] (RGB)
    ↓
    PNG File Saved!
```

---

## Key Parameters & Their Effects

### During Training

| Parameter | Value | Effect |
|-----------|-------|--------|
| **lora_rank** | 32 | Size of LoRA matrices (bigger = more capacity) |
| **lora_alpha** | 32 | Scale factor for LoRA output |
| **learning_rate** | 1e-4 | How aggressively to update weights |
| **batch_size** | 2 | Process 2 image-caption pairs at once |
| **gradient_accumulation** | 4 | Simulate larger batches (2 × 4 = 8) |
| **num_epochs** | 100 | Repeat dataset 100 times |
| **train_text_encoder** | False | Keep text encoders frozen (faster) |

### During Inference

| Parameter | Value | Effect |
|-----------|-------|--------|
| **seed** | 42 | Initial noise (deterministic) |
| **guidance_scale** | 7.5 | Strength of text prompt (1.0=weak, 15=strict) |
| **num_inference_steps** | 50 | Diffusion iterations (more = higher quality) |
| **ip_adapter_scale** | 0.0 | Disabled (not used) |
| **controlnet_scale** | 0.0 | Disabled (not used) |

---

## Complete Data Flow Diagram

```
TRAINING:
┌─────────────────────────────────────────────────────────┐
│ IMAGE-CAPTION PAIR (aldar_000.jpg, "aldar_kose_man...") │
└──────────────────┬──────────────────────────────────────┘
                   ↓
       ┌───────────────────────────┐
       │ TOKENIZE TEXT (→ tokens)  │
       └───────────┬───────────────┘
                   ↓
    ┌──────────────────────────────┐
    │ TEXT ENCODER (→ embedding)   │
    │ 768 + 1280 dimensional       │
    └───────────┬──────────────────┘
                ↓
     ┌──────────────────────────┐
     │ IMAGE → VAE → LATENTS    │
     │ [1,4,256,256]            │
     └───────────┬──────────────┘
                 ↓
         ┌───────────────────┐
         │ ADD NOISE (t=500) │
         └────────┬──────────┘
                  ↓
    ┌─────────────────────────────┐
    │ UNET (with LoRA)            │
    │ Predict noise               │
    │ LoRA adapts for Aldar       │
    └────────┬────────────────────┘
             ↓
      ┌──────────────────┐
      │ LOSS CALCULATION │
      │ predicted vs     │
      │ actual noise     │
      └────────┬─────────┘
               ↓
    ┌──────────────────────────┐
    │ BACKPROP → UPDATE LoRA   │
    │ Only 3.7M params change  │
    └────────┬─────────────────┘
             ↓
    ┌──────────────────────────┐
    │ REPEAT 1000s of times    │
    │ Aldar pattern emerges... │
    └────────┬─────────────────┘
             ↓
    ┌──────────────────────────┐
    │ SAVE CHECKPOINT          │
    │ unet_lora/ (5MB)         │
    │ text_encoder_*_lora/ ... │
    └──────────────────────────┘


INFERENCE:
┌──────────────────────────────────────┐
│ USER PROMPT: "Aldar tricks merchant" │
└────────┬─────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│ TEXT ENCODER (frozen)              │
│ → [1, 77, 2048] embedding          │
└────────┬─────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│ RANDOM NOISE (seed=42)             │
│ → [1, 4, 256, 256] (deterministic) │
└────────┬─────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│ LOAD LoRA CHECKPOINT               │
│ UNet + LoRA adapters activated     │
└────────┬─────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│ FOR t in 999 down to 0:            │
│   - UNet predicts noise            │
│   - Remove noise                   │
│   - Continue denoising             │
│   - LoRA guides: "This is Aldar"   │
└────────┬─────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│ CLEAN LATENTS [1, 4, 256, 256]    │
└────────┬─────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│ VAE DECODER                        │
│ → [3, 1024, 1024] RGB image        │
└────────┬─────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│ SAVE PNG: frame_001.png            │
│ Character: Aldar                   │
│ Consistency: 85-90%                │
└────────────────────────────────────┘
```

---

## The Role of Each Component

### Text Encoders (CLIP Models)
- **Purpose:** Convert text prompt to semantic vectors
- **Why two?** SDXL uses both CLIP-B and CLIP-L for richer understanding
- **During training:** Frozen (not updated)
- **Output:** 2048-dimensional embedding capturing meaning

### UNet (The Main Model)
- **Purpose:** Learn to predict noise at each diffusion step
- **Size:** 2.6 billion parameters
- **With LoRA:** Can generate Aldar-specific images
- **Process:** 
  1. Self-attention: Process current image
  2. Cross-attention: Incorporate text guidance
  3. Convolutions: Refine

### VAE (Variational Autoencoder)
- **Purpose:** Compress/decompress images
- **Why:** Diffusion in pixel space is slow; latent space is 16× faster
- **During training:** Frozen (pre-trained)
- **Encoding:** Image → compressed latents
- **Decoding:** Latents → final image

### LoRA (Low-Rank Adaptation)
- **Purpose:** Learn character-specific patterns
- **Strategy:** Add small matrices (A, B) to attention layers
- **Effect:** Model output × (1 - scale) + Model output × scale
- **Result:** Same 880MB model, but with Aldar knowledge injected

---

## Why This Design Works for Your Project

```
TRADITIONAL FINE-TUNING (Bad):
- Train full 880MB model
- Requires 20GB+ VRAM
- Takes weeks to train
- 880MB checkpoint per version

YOUR APPROACH (Good):
- Train only 3.7M LoRA parameters (0.4%)
- Requires 8GB VRAM
- Trains in 3-4 hours
- 5-10MB checkpoint per version

Result:
- Same quality output
- 100× smaller model
- 10× faster training
- Fits on consumer GPUs
```

---

## Summary

**Text → Image Process:**

1. **Text** is tokenized and embedded into 2048-dimensional semantic space
2. **Random noise** is created deterministically from seed
3. **UNet + LoRA** learns to remove noise while respecting text guidance
4. **50 denoising steps** gradually reveal the image
5. **Character knowledge** comes from LoRA (Aldar patterns)
6. **Final latents** are decoded by VAE back to RGB pixels
7. **PNG saved** with 85-90% face consistency via LoRA

The entire process is deterministic when seed=0.0 temperature, so same prompt = same image every time.
