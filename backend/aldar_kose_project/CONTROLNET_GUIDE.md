# ControlNet Integration Guide

## Overview

ControlNet adds structural guidance to SDXL generation, drastically reducing pose and anatomy anomalies by enforcing skeleton/pose consistency.

## Benefits

1. **Pose Consistency**: Enforce same pose across frames
2. **Anatomy Prevention**: Prevent double heads, extra limbs
3. **Spatial Control**: Control character position and composition
4. **Style Transfer**: Maintain character appearance while changing pose

## Installation

```bash
# Install ControlNet dependencies
pip install controlnet-aux

# Models will auto-download from Hugging Face
```

## Usage

### 1. OpenPose ControlNet (Recommended for Characters)

```python
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from controlnet_aux import OpenposeDetector
from PIL import Image

# Initialize ControlNet
controlnet = ControlNetModel.from_pretrained(
    "thibaud/controlnet-openpose-sdxl-1.0",
    torch_dtype=torch.float16
)

# Initialize pipeline with ControlNet
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Load your LoRA
pipe.load_lora_weights("outputs/checkpoints/checkpoint-1000")

# Extract pose from reference image
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
reference_image = Image.open("reference_pose.png")
pose_image = openpose(reference_image)

# Generate with pose guidance
output = pipe(
    prompt="aldar_kose_man portrait",
    image=pose_image,
    controlnet_conditioning_scale=0.8,  # 0.5-1.0, higher = stricter pose match
    num_inference_steps=40,
).images[0]
```

### 2. Depth ControlNet (For Spatial Consistency)

```python
from transformers import DPTImageProcessor, DPTForDepthEstimation

# Initialize depth estimator
processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

# Extract depth from reference
inputs = processor(images=reference_image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    depth = outputs.predicted_depth

# Load depth ControlNet
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    torch_dtype=torch.float16
)

# Generate with depth guidance
output = pipe(
    prompt="aldar_kose_man riding horse",
    image=depth_image,
    controlnet_conditioning_scale=0.7,
).images[0]
```

### 3. Canny Edge ControlNet (For Outline Consistency)

```python
import cv2

# Extract canny edges
image_np = np.array(reference_image)
edges = cv2.Canny(image_np, 100, 200)
edges_pil = Image.fromarray(edges)

# Load canny ControlNet
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

# Generate with edge guidance
output = pipe(
    prompt="aldar_kose_man portrait",
    image=edges_pil,
    controlnet_conditioning_scale=0.6,  # Lower for more creative freedom
).images[0]
```

## Integration with Storyboard Generator

### Option 1: Reference-Guided Mode (Already Implemented)

The `ref_guided_storyboard.py` already uses ControlNet:

```python
from scripts.ref_guided_storyboard import ReferenceGuidedStoryboardGenerator

generator = ReferenceGuidedStoryboardGenerator(
    lora_path="outputs/checkpoints/checkpoint-1000",
    use_controlnet=True,  # Enable ControlNet
    use_ip_adapter=True,  # Enable IP-Adapter for face consistency
)

frames = generator.generate_sequence(
    prompts=prompts,
    controlnet_conditioning_scale=0.8,  # Adjust strength
)
```

### Option 2: Add ControlNet to Simple Generator

Update `simple_storyboard.py` to optionally use ControlNet:

```python
class SimplifiedStoryboardGenerator:
    def __init__(
        self,
        lora_path: str,
        use_controlnet: bool = False,
        controlnet_type: str = "openpose",  # "openpose", "depth", "canny"
    ):
        if use_controlnet:
            # Load ControlNet
            if controlnet_type == "openpose":
                self.controlnet = ControlNetModel.from_pretrained(
                    "thibaud/controlnet-openpose-sdxl-1.0",
                    torch_dtype=torch.float16
                )
            
            # Create ControlNet pipeline
            self.txt2img_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                controlnet=self.controlnet,
                torch_dtype=torch.float16
            ).to(self.device)
```

## Best Practices

### 1. Choose Right ControlNet Type

| Use Case | ControlNet Type | Conditioning Scale |
|----------|----------------|-------------------|
| Character consistency | OpenPose | 0.7-0.9 |
| Spatial layout | Depth | 0.6-0.8 |
| Outline preservation | Canny | 0.5-0.7 |
| Combination | Multi-ControlNet | 0.6-0.8 each |

### 2. Conditioning Scale Guidelines

- **0.3-0.5**: Loose guidance (creative freedom)
- **0.6-0.8**: Balanced (recommended)
- **0.9-1.0**: Strict guidance (exact match)

### 3. Extract Control Images from Frame 1

```python
# Generate Frame 1 normally
frame_1 = generate_frame_1()

# Extract pose/depth/edges from Frame 1
pose_map = extract_openpose(frame_1)

# Use as guidance for Frame 2+
frame_2 = generate_with_controlnet(
    prompt=prompt_2,
    control_image=pose_map,
    controlnet_conditioning_scale=0.7
)
```

### 4. Multi-ControlNet for Maximum Consistency

```python
# Combine OpenPose + Depth
controlnets = [
    ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0"),
    ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0"),
]

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnets,
).to("cuda")

# Generate with both controls
output = pipe(
    prompt="aldar_kose_man riding horse",
    image=[pose_map, depth_map],
    controlnet_conditioning_scale=[0.8, 0.6],  # Different scales per control
).images[0]
```

## Performance Considerations

### Memory Usage
- **Single ControlNet**: +2GB VRAM
- **Multi-ControlNet**: +3-4GB VRAM
- **Minimum VRAM**: 12GB recommended (16GB+ for multi-ControlNet)

### Speed Impact
- **Control image extraction**: ~200ms per frame
- **ControlNet inference**: +10-20% generation time
- **Total overhead**: Minimal (quality improvement worth it)

### Optimization Tips

```python
# Enable memory optimizations
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()  # If low on VRAM

# Use lower resolution for control images
control_image = control_image.resize((512, 512))  # Instead of 1024x1024
```

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce resolution
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    ...,
    torch_dtype=torch.float16,  # Use FP16
)

# Enable CPU offloading
pipe.enable_model_cpu_offload()
```

### ControlNet Too Strict
```python
# Lower conditioning scale
controlnet_conditioning_scale=0.5  # More creative freedom
```

### Pose Detection Fails
```python
# Use manual pose maps or skip problematic frames
if pose_keypoints.shape[0] < 10:  # Too few keypoints
    logger.warning("Pose detection failed, using txt2img fallback")
    output = txt2img_pipe(prompt=prompt).images[0]
```

## Example: Full Integration

```python
from scripts.simple_storyboard import SimplifiedStoryboardGenerator
from controlnet_aux import OpenposeDetector

class ControlNetStoryboardGenerator(SimplifiedStoryboardGenerator):
    def __init__(self, lora_path: str):
        super().__init__(lora_path, enable_anomaly_detection=True)
        
        # Load OpenPose
        self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        
        # Load ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-openpose-sdxl-1.0",
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Recreate pipeline with ControlNet
        self.txt2img_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Load LoRA
        self.txt2img_pipe.load_lora_weights(lora_path)
    
    def generate_with_pose_guidance(
        self,
        prompt: str,
        reference_frame: Image.Image,
        conditioning_scale: float = 0.7,
    ):
        # Extract pose from reference
        pose_map = self.openpose(reference_frame)
        
        # Generate with pose guidance
        output = self.txt2img_pipe(
            prompt=prompt,
            image=pose_map,
            controlnet_conditioning_scale=conditioning_scale,
            num_inference_steps=40,
        ).images[0]
        
        # Run anomaly detection
        anomaly_result = self.anomaly_detector.detect_anomalies(
            output,
            expected_prompt=prompt
        )
        
        return output, anomaly_result

# Usage
generator = ControlNetStoryboardGenerator(
    lora_path="outputs/checkpoints/checkpoint-1000"
)

# Generate Frame 1
frame_1 = generator.txt2img_pipe(
    prompt="aldar_kose_man portrait",
    num_inference_steps=40,
).images[0]

# Generate Frame 2 with pose guidance from Frame 1
frame_2, anomaly_result = generator.generate_with_pose_guidance(
    prompt="aldar_kose_man riding horse",
    reference_frame=frame_1,
    conditioning_scale=0.7
)

if anomaly_result['is_valid']:
    print("✓ Frame 2 valid!")
else:
    print(f"✗ Anomalies: {anomaly_result['anomalies']}")
```

## References

- **ControlNet Paper**: https://arxiv.org/abs/2302.05543
- **SDXL ControlNet**: https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0
- **controlnet-aux**: https://github.com/patrickvonplaten/controlnet_aux
- **OpenPose**: https://github.com/CMU-Perceptual-Computing-Lab/openpose

## Next Steps

1. **Test ControlNet**: Try `ref_guided_storyboard.py` with `--use-controlnet`
2. **Extract Poses**: Use `openpose_detector` on your best frames
3. **Fine-tune Scales**: Experiment with conditioning scales (0.5-0.9)
4. **Combine with Anomaly Detection**: Best results with both systems enabled

---

**Status**: ControlNet integration available in `ref_guided_storyboard.py`  
**Future**: Add ControlNet option to `simple_storyboard.py` and `generate_story.py`
