#!/usr/bin/env python3
"""
Prompt-Based Storyboard Generator

Takes a user's story description and generates a complete storyboard (6-10 frames)
by breaking down the story into scenes using OpenAI API, then generating images.

Usage:
    # Interactive mode
    python scripts/prompt_storyboard.py --lora-path outputs/checkpoints/checkpoint-400
    
    # Direct prompt
    python scripts/prompt_storyboard.py \
        --lora-path outputs/checkpoints/checkpoint-400 \
        --story "Aldar Kose riding his horse across the steppe towards his yurt at sunset" \
        --num-frames 8
    
    # Custom output directory
    python scripts/prompt_storyboard.py \
        --lora-path outputs/checkpoints/checkpoint-400 \
        --story "Aldar Kose tricks a wealthy merchant" \
        --output-dir outputs/my_story

Requirements:
    pip install openai diffusers transformers torch pillow
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

import torch
from PIL import Image

try:
    import openai
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed.")
    print("Install with: pip install openai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import the storyboard generator
from simple_storyboard import SimplifiedStoryboardGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
TRIGGER_TOKEN = "aldar_kose_man"

SCENE_BREAKDOWN_SYSTEM_PROMPT = """You are an expert storyboard artist and cinematographer specializing in visual storytelling.

Your task is to break down a story description into 6-10 distinct, visually compelling scenes for a storyboard.

IMPORTANT RULES:
1. Each scene should be a DIFFERENT moment in the story - show PROGRESSION
2. Focus on ACTION, SETTING, EMOTION, and CINEMATOGRAPHY
3. DO NOT mention clothing, outfits, costumes, or physical appearance
4. Include camera angles and shot types for variety (close-up, wide shot, medium shot, POV, etc.)
5. Include artistic direction (lighting, mood, atmosphere)
6. Make scenes visually distinct from each other

The character is "aldar_kose_man" - a clever trickster from Kazakh folklore.

Story Structure Guidelines:
- Opening: Establish setting and character
- Rising Action: Show the journey/conflict/trick developing
- Climax: The key moment of the story
- Resolution: The outcome
- Closing: Final emotional beat

Camera Variety:
- Wide shots for establishing scenes
- Close-ups for emotional moments
- Medium shots for action
- Dynamic angles for dramatic moments

Return your response as a JSON array with this structure:
[
    {
        "frame": 1,
        "description": "aldar_kose_man riding horse across vast steppe, wide establishing shot, golden hour lighting, sense of adventure",
        "camera": "wide shot",
        "mood": "adventurous"
    },
    {
        "frame": 2,
        "description": "aldar_kose_man approaching traditional yurt in distance, medium shot from behind, sunset casting long shadows",
        "camera": "medium shot",
        "mood": "anticipation"
    },
    ...
]

Remember: NEVER mention clothing, costumes, or physical appearance. Focus on what's HAPPENING, WHERE, and HOW it's SHOT."""


class PromptStoryboardGenerator:
    """
    Converts a user's story description into a complete storyboard
    """
    
    def __init__(
        self,
        openai_api_key: str,
        lora_path: str,
        device: str = "cuda",
    ):
        """Initialize the prompt-based storyboard generator"""
        self.client = OpenAI(api_key=openai_api_key)
        self.lora_path = lora_path
        self.device = device
        
        # Initialize the image generator (lazy loading)
        self.generator = None
        
        logger.info("Prompt Storyboard Generator initialized")
    
    def break_down_story(
        self,
        story: str,
        num_frames: int = 8,
    ) -> List[Dict]:
        """
        Use OpenAI to break down a story into individual scenes
        
        Args:
            story: The user's story description
            num_frames: Number of frames to generate (6-10)
            
        Returns:
            List of scene descriptions with metadata
        """
        logger.info(f"Breaking down story into {num_frames} scenes...")
        
        user_prompt = f"""Break down the following story into exactly {num_frames} distinct, visually compelling scenes for a storyboard.

Story: "{story}"

Remember:
- {num_frames} scenes showing PROGRESSION through the story
- Each scene should be visually DIFFERENT
- Include camera angles and lighting
- Focus on ACTION and SETTING, NOT clothing or appearance
- Use "aldar_kose_man" as the character identifier

Return ONLY a JSON array, no other text."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": SCENE_BREAKDOWN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(content)
                # Handle different possible response structures
                if isinstance(result, dict) and "scenes" in result:
                    scenes = result["scenes"]
                elif isinstance(result, dict) and "frames" in result:
                    scenes = result["frames"]
                elif isinstance(result, list):
                    scenes = result
                else:
                    # If it's a dict with numbered keys
                    scenes = list(result.values())
                
                logger.info(f"✓ Successfully generated {len(scenes)} scenes")
                return scenes
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Response content: {content}")
                raise
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def refine_prompts_for_sdxl(
        self,
        scenes: List[Dict]
    ) -> List[str]:
        """
        Convert scene descriptions into optimized SDXL prompts
        
        Args:
            scenes: List of scene dictionaries from story breakdown
            
        Returns:
            List of optimized prompts for SDXL generation
        """
        prompts = []
        
        for scene in scenes:
            # Extract description
            if isinstance(scene, dict):
                description = scene.get("description", scene.get("prompt", str(scene)))
            else:
                description = str(scene)
            
            # Ensure trigger token is present
            if TRIGGER_TOKEN not in description.lower():
                description = f"{TRIGGER_TOKEN} {description}"
            
            # Add quality tags
            prompt = f"{description}, high quality, detailed, 3D animation, professional render"
            
            prompts.append(prompt)
        
        return prompts
    
    def generate_storyboard(
        self,
        story: str,
        num_frames: int = 8,
        output_dir: str = "outputs/prompt_storyboard",
        base_seed: int = 42,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.5,
        max_attempts: int = 3,
    ) -> List[Image.Image]:
        """
        Complete pipeline: story -> scenes -> images
        
        Args:
            story: User's story description
            num_frames: Number of frames to generate (6-10)
            output_dir: Where to save results
            base_seed: Base seed for generation
            num_inference_steps: SDXL steps
            guidance_scale: SDXL guidance
            max_attempts: Retry attempts per frame
            
        Returns:
            List of generated PIL Images
        """
        # Step 1: Break down story into scenes
        logger.info("=" * 60)
        logger.info("STEP 1: Breaking down story into scenes")
        logger.info("=" * 60)
        scenes = self.break_down_story(story, num_frames)
        
        # Save scene breakdown
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        scene_file = output_path / "scene_breakdown.json"
        with open(scene_file, 'w', encoding='utf-8') as f:
            json.dump({
                "story": story,
                "timestamp": datetime.now().isoformat(),
                "num_frames": num_frames,
                "scenes": scenes
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Scene breakdown saved to {scene_file}")
        
        # Print scenes
        print("\n" + "=" * 60)
        print("SCENE BREAKDOWN:")
        print("=" * 60)
        for i, scene in enumerate(scenes, 1):
            if isinstance(scene, dict):
                print(f"\nFrame {i}:")
                print(f"  Description: {scene.get('description', scene.get('prompt', ''))}")
                print(f"  Camera: {scene.get('camera', 'N/A')}")
                print(f"  Mood: {scene.get('mood', 'N/A')}")
            else:
                print(f"\nFrame {i}: {scene}")
        print("=" * 60 + "\n")
        
        # Step 2: Refine prompts for SDXL
        logger.info("=" * 60)
        logger.info("STEP 2: Refining prompts for SDXL")
        logger.info("=" * 60)
        prompts = self.refine_prompts_for_sdxl(scenes)
        
        # Save prompts
        prompts_file = output_path / "sdxl_prompts.json"
        with open(prompts_file, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ SDXL prompts saved to {prompts_file}")
        
        # Step 3: Initialize image generator (lazy loading)
        if self.generator is None:
            logger.info("=" * 60)
            logger.info("STEP 3: Loading SDXL + LoRA model")
            logger.info("=" * 60)
            self.generator = SimplifiedStoryboardGenerator(
                lora_path=self.lora_path,
                device=self.device,
            )
        
        # Step 4: Generate images
        logger.info("=" * 60)
        logger.info("STEP 4: Generating storyboard images")
        logger.info("=" * 60)
        
        frames = self.generator.generate_sequence(
            prompts=prompts,
            output_dir=str(output_path),
            base_seed=base_seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            consistency_threshold=0.70,  # CLIP similarity threshold
            max_attempts=max_attempts,
        )
        
        logger.info("=" * 60)
        logger.info("✓ STORYBOARD GENERATION COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Generated {len(frames)} frames")
        logger.info(f"Saved to: {output_path}")
        logger.info(f"Scene breakdown: {scene_file}")
        logger.info(f"SDXL prompts: {prompts_file}")
        logger.info("=" * 60)
        
        return frames


def main():
    parser = argparse.ArgumentParser(
        description="Generate storyboard from text prompt using OpenAI + SDXL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/prompt_storyboard.py --lora-path outputs/checkpoints/checkpoint-400
  
  # Direct story
  python scripts/prompt_storyboard.py \\
      --lora-path outputs/checkpoints/checkpoint-400 \\
      --story "Aldar Kose riding his horse to his yurt at sunset"
  
  # Custom settings
  python scripts/prompt_storyboard.py \\
      --lora-path outputs/checkpoints/checkpoint-400 \\
      --story "Aldar Kose tricks a wealthy merchant" \\
      --num-frames 10 \\
      --output-dir outputs/merchant_trick
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="Path to trained LoRA checkpoint"
    )
    
    # Story input
    parser.add_argument(
        "--story",
        type=str,
        default=None,
        help="Story description (if not provided, will prompt interactively)"
    )
    
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        choices=range(6, 11),
        metavar="6-10",
        help="Number of frames to generate (6-10, default: 8)"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/prompt_storyboard_TIMESTAMP)"
    )
    
    # OpenAI API
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env variable)"
    )
    
    # Generation settings
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed for generation"
    )
    
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=40,
        help="Number of inference steps"
    )
    
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Guidance scale"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found!")
        logger.error("Provide via --api-key or set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Get story (interactive if not provided)
    story = args.story
    if not story:
        print("\n" + "=" * 60)
        print("PROMPT-BASED STORYBOARD GENERATOR")
        print("=" * 60)
        print("\nDescribe your story for Aldar Kose:")
        print("(Example: 'Aldar Kose riding his horse across the steppe to his yurt')")
        print()
        story = input("Your story: ").strip()
        
        if not story:
            logger.error("No story provided!")
            sys.exit(1)
    
    # Set output directory
    output_dir = args.output_dir
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/prompt_storyboard_{timestamp}"
    
    # Initialize generator
    logger.info("Initializing Prompt Storyboard Generator...")
    generator = PromptStoryboardGenerator(
        openai_api_key=api_key,
        lora_path=args.lora_path,
        device=args.device,
    )
    
    # Generate storyboard
    try:
        frames = generator.generate_storyboard(
            story=story,
            num_frames=args.num_frames,
            output_dir=output_dir,
            base_seed=args.base_seed,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"✓ Generated {len(frames)} frames")
        print(f"✓ Saved to: {output_dir}")
        print("=" * 60 + "\n")
        
    except Exception as e:
        logger.error(f"Storyboard generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
