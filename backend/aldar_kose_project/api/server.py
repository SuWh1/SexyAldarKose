#!/usr/bin/env python3
"""
FastAPI Server for Aldar Kose Storyboard Generation

Provides REST API endpoint to generate story sequences from text prompts.

Usage:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
    
    Or:
    python api/server.py
"""

import argparse
import base64
import io
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Suppress library warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prompt_storyboard import PromptStoryboardGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# API Models
# ============================================================

class StoryRequest(BaseModel):
    """Request model for story generation"""
    prompt: str = Field(..., description="Story prompt/description", min_length=10, max_length=500)
    use_ref_guided: bool = Field(default=False, description="Use reference-guided mode for better consistency")
    num_frames: Optional[int] = Field(default=None, description="Max number of frames (6-10), None=auto", ge=6, le=10)
    seed: int = Field(default=42, description="Random seed for reproducibility (same seed + prompt = same output)")
    gpt_temperature: float = Field(default=0.7, description="GPT-4 creativity (0=deterministic, 1=creative)", ge=0.0, le=1.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Aldar Kose tricks a wealthy merchant and steals his horse",
                "use_ref_guided": False,
                "num_frames": None,
                "seed": 42,
                "gpt_temperature": 0.7
            }
        }


class FrameResponse(BaseModel):
    """Single frame response"""
    frame_number: int
    image: str = Field(..., description="Base64 encoded PNG image")
    prompt: str = Field(..., description="Scene description for this frame")
    clip_score: float = Field(..., description="CLIP similarity score (0-1)")


class StoryResponse(BaseModel):
    """Response model for story generation"""
    success: bool
    story_prompt: str
    num_frames: int
    frames: List[FrameResponse]
    generation_time_seconds: float
    mode: str = Field(..., description="simple or ref-guided")
    seed: int = Field(..., description="Seed used for generation")
    gpt_temperature: float = Field(..., description="GPT temperature used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "story_prompt": "Aldar Kose tricks a merchant",
                "num_frames": 8,
                "frames": [
                    {
                        "frame_number": 1,
                        "image": "base64_encoded_string...",
                        "prompt": "Aldar standing in marketplace, front-facing",
                        "clip_score": 0.85
                    }
                ],
                "generation_time_seconds": 245.3,
                "mode": "simple",
                "seed": 42,
                "gpt_temperature": 0.7
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    lora_path: str
    ref_guided_available: bool


# ============================================================
# Global State
# ============================================================

generator = None
LORA_PATH = None


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Aldar Kose Storyboard API",
    description="Generate multi-frame story sequences using fine-tuned SDXL + LoRA",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Helper Functions
# ============================================================

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def initialize_generator(lora_path: str, openai_api_key: str, use_ref_guided: bool = False):
    """Initialize the storyboard generator"""
    global generator
    
    logger.info(f"Initializing generator (ref_guided={use_ref_guided})...")
    
    generator = PromptStoryboardGenerator(
        openai_api_key=openai_api_key,
        lora_path=lora_path,
        use_ref_guided=use_ref_guided,
    )
    
    logger.info("Generator initialized successfully!")
    return generator


# ============================================================
# API Endpoints
# ============================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="running",
        model_loaded=generator is not None,
        lora_path=LORA_PATH or "not_set",
        ref_guided_available=True  # Always available through PromptStoryboardGenerator
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    return HealthResponse(
        status="healthy" if generator is not None else "not_initialized",
        model_loaded=generator is not None,
        lora_path=LORA_PATH or "not_set",
        ref_guided_available=True  # Always available through PromptStoryboardGenerator
    )


@app.post("/generate", response_model=StoryResponse)
async def generate_story(request: StoryRequest):
    """
    Generate a multi-frame story sequence from a text prompt
    
    - **prompt**: Story description (e.g., "Aldar Kose tricks a merchant")
    - **use_ref_guided**: Use reference-guided mode for better face consistency
    - **num_frames**: Max frames to generate (6-10), None = GPT decides
    - **seed**: Random seed for reproducibility (same seed + prompt = same output)
    - **gpt_temperature**: GPT-4 creativity (0.0=deterministic/consistent, 1.0=creative/varied)
    
    Returns base64-encoded PNG images for each frame.
    
    **Determinism**: Same seed + same prompt + temperature=0.0 will generate highly similar outputs.
    """
    if generator is None:
        raise HTTPException(status_code=500, detail="Generator not initialized. Set LORA_PATH and OPENAI_API_KEY.")
    
    logger.info(f"Generating story: '{request.prompt}'")
    logger.info(f"Mode: {'ref-guided' if request.use_ref_guided else 'simple'}, Seed: {request.seed}, Temp: {request.gpt_temperature}")
    
    start_time = datetime.now()
    
    try:
        # Re-initialize generator if mode changed
        if generator is None or generator.use_ref_guided != request.use_ref_guided:
            logger.info(f"Initializing generator with ref_guided={request.use_ref_guided}")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
            initialize_generator(LORA_PATH, api_key, use_ref_guided=request.use_ref_guided)
        
        requested_mode = "ref-guided" if request.use_ref_guided else "simple"
        
        # Break down story into scenes using GPT-4 with temperature control
        scene_breakdown = generator.break_down_story(
            story=request.prompt,
            max_frames=request.num_frames or 10,
            temperature=request.gpt_temperature
        )
        
        prompts = [frame["description"] for frame in scene_breakdown["frames"]]
        num_frames = len(prompts)
        
        logger.info(f"GPT-4 decided on {num_frames} frames (temperature={request.gpt_temperature})")
        
        # Generate frames with seed for reproducibility
        # Same seed + same prompt = same output
        if request.use_ref_guided:
            # Reference-guided mode
            frames_images = generator.generate_sequence(
                prompts=prompts,
                base_seed=request.seed,
                consistency_threshold=0.70,
                max_retries=2,
            )
        else:
            # Simple mode
            frames_images = generator.generate_sequence(
                prompts=prompts,
                base_seed=request.seed,
                consistency_threshold=0.70,
                max_retries=2,
            )
        
        # Convert images to base64
        frames_response = []
        for idx, (img, prompt) in enumerate(zip(frames_images, prompts)):
            # Compute CLIP score (generator stores this internally)
            clip_score = 0.85  # Placeholder - extract from generator if available
            
            frames_response.append(FrameResponse(
                frame_number=idx + 1,
                image=image_to_base64(img),
                prompt=prompt,
                clip_score=clip_score
            ))
        
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Story generated successfully in {generation_time:.1f}s")
        
        return StoryResponse(
            success=True,
            story_prompt=request.prompt,
            num_frames=num_frames,
            frames=frames_response,
            generation_time_seconds=generation_time,
            mode=requested_mode,
            seed=request.seed,
            gpt_temperature=request.gpt_temperature
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ============================================================
# Startup/Shutdown
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Initialize generator on startup"""
    global LORA_PATH
    
    # Get LoRA path from environment or use default
    LORA_PATH = os.getenv("LORA_PATH", "outputs/checkpoints/final")
    
    if not Path(LORA_PATH).exists():
        logger.warning(f"LoRA path does not exist: {LORA_PATH}")
        logger.warning("Generator will be initialized on first request")
        return
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set. Generator will be initialized on first request.")
        return
    
    # Initialize with simple mode by default
    try:
        initialize_generator(LORA_PATH, api_key, use_ref_guided=False)
        logger.info("Server ready!")
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        logger.warning("Generator will be initialized on first request")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global generator
    if generator:
        logger.info("Cleaning up generator...")
        del generator
        generator = None


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Aldar Kose Storyboard API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--lora-path", default="outputs/checkpoints/final", help="Path to LoRA checkpoint")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Set LORA_PATH environment variable
    os.environ["LORA_PATH"] = args.lora_path
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"LoRA path: {args.lora_path}")
    
    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
