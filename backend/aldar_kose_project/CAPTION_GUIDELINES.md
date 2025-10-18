# Caption Generation Guidelines

## Overview

This document explains the caption generation strategy for Aldar Kose LoRA training data.

## Philosophy

**Key Principle:** Since LoRA learns the character's identity and appearance from the images themselves, captions should focus on **what the character is DOING** and **where they are**, NOT what they look like or what they're wearing.

## What to Include

✅ **DO Describe:**
- **Action/Activity:** walking, riding, laughing, sitting, storytelling
- **Pose:** standing, sitting cross-legged, gesturing
- **Expression:** smiling, laughing, surprised, thoughtful
- **Setting/Location:** market, yurt interior, steppe, village, campfire
- **Background:** outdoor scene, indoor setting, landscape
- **Lighting:** studio lighting, sunset, firelight, dramatic lighting
- **Camera/Composition:** close-up, wide shot, portrait, action shot
- **Artistic Style:** 3D render, CGI, animation, cinematic

## What to Exclude

❌ **DO NOT Describe:**
- **Clothing:** traditional outfit, blue coat, ornate costume, dress
- **Physical Appearance:** facial features, hair, skin tone
- **Accessories:** hat, jewelry, decorations
- **Colors of clothing:** (background colors are OK)

## Why This Matters

1. **Avoids Overfitting:** If every caption mentions "blue traditional coat", the model might think this clothing is required
2. **Enables Flexibility:** User can specify clothing in their prompts if they want variations
3. **Focuses on Identity:** LoRA learns facial features and body type from images, not captions
4. **Better Composition Control:** Focuses training on scene understanding rather than outfit details

## Example Captions

### ✅ Good Captions (Action/Setting Focused)

```
aldar_kose_man standing in village marketplace, confident smile, wide shot
aldar_kose_man riding horse across steppe, dynamic pose, cinematic lighting
aldar_kose_man sitting by campfire, storytelling gesture, warm atmosphere
aldar_kose_man laughing heartily, close-up portrait, studio lighting
aldar_kose_man walking down street, animated expression, outdoor scene
aldar_kose_man in traditional yurt interior, welcoming gesture, soft lighting
```

### ❌ Bad Captions (Clothing/Appearance Focused)

```
aldar_kose_man in blue traditional coat and hat standing in market ❌
aldar_kose_man wearing ornate Kazakh costume, smiling ❌
aldar_kose_man with colorful traditional dress riding horse ❌
aldar_kose_man in ceremonial outfit, detailed embroidery ❌
```

## Caption Structure Template

```
aldar_kose_man [ACTION/POSE], [EXPRESSION], [SETTING], [SHOT TYPE/LIGHTING]
```

Examples:
- `aldar_kose_man [walking], [confident smile], [village street], [wide shot]`
- `aldar_kose_man [sitting cross-legged], [storytelling gesture], [yurt interior], [warm firelight]`
- `aldar_kose_man [riding horse], [determined expression], [steppe landscape], [cinematic lighting]`

## Using the Scripts

### Generate New Captions

```bash
# Caption new images
python scripts/label_images.py --input_dir raw_images/
```

### Re-caption Existing Images

```bash
# Update old captions with new approach
python scripts/recaption_images.py --images data/images/ --captions data/captions/

# Dry run to see what would change
python scripts/recaption_images.py --dry-run
```

## Training Benefits

With action-focused captions:
- ✅ Character identity learned from images
- ✅ Composition/scene understanding from captions
- ✅ Flexible prompt control (users can add clothing if desired)
- ✅ Better generalization to new poses/settings
- ✅ Less overfitting to specific outfits

## Prompt Usage During Inference

**For character generation:**
```python
# Simple - lets LoRA use learned appearance
prompt = "aldar_kose_man riding horse, sunset"

# Explicit - override clothing if desired
prompt = "aldar_kose_man in modern suit, office setting"
```

The LoRA will maintain facial identity either way!
