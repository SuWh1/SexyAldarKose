#!/usr/bin/env python3
"""
Prompt-Based Storyboard Generator

Takes a user's story description and generates a complete storyboard (6-10 frames)
by breaking down the story into scenes using OpenAI API, then generating images.

Usage:
    # Interactive mode (GPT decides number of frames)
    python scripts/prompt_storyboard.py --lora-path outputs/checkpoints/checkpoint-400
    
    # Direct prompt (GPT decides optimal frame count)
    python scripts/prompt_storyboard.py \
        --lora-path outputs/checkpoints/checkpoint-400 \
        --story "Aldar Kose riding his horse across the steppe towards his yurt at sunset"
    
    # Specify max frames (GPT decides up to this limit)
    python scripts/prompt_storyboard.py \
        --lora-path outputs/checkpoints/checkpoint-400 \
        --story "Aldar Kose tricks a wealthy merchant" \
        --num-frames 10 \
        --output-dir outputs/my_story

Requirements:
    pip install openai diffusers transformers torch pillow
"""

# Suppress library warnings and verbose logging
import warnings
import os
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Suppress TensorFlow/MediaPipe verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error only
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

import argparse
import json
import logging
import sys
import warnings
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

# Import the storyboard generators
# Support both direct script execution and module import
try:
    from scripts.simple_storyboard import SimplifiedStoryboardGenerator
except ImportError:
    from simple_storyboard import SimplifiedStoryboardGenerator

try:
    try:
        from scripts.ref_guided_storyboard import ReferenceGuidedStoryboardGenerator
    except ImportError:
        from ref_guided_storyboard import ReferenceGuidedStoryboardGenerator
    HAS_REF_GUIDED = True
except ImportError:
    HAS_REF_GUIDED = False
    print("⚠️  Reference-guided generator not available (missing dependencies)")
    print("    Install: pip install controlnet-aux")
    print("    Fallback: Using simple storyboard generator")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
TRIGGER_TOKEN = "aldar_kose_man"


SCENE_BREAKDOWN_SYSTEM_PROMPT = """You are an expert storyboard artist specializing in clear, simple visual storytelling.

Your task is to break down a story into 6-10 very SIMPLE scenes optimized for image generation.

CRITICAL RULES:
1. Return only the minimal JSON described below (do NOT include camera or mood fields)
2. Keep each "description" SHORT and CLEAR (action + location; max ~10 words)
3. **FRAME 1 IS MANDATORY**: Must ALWAYS show aldar_kose_man's FACE clearly from the FRONT
   - Frame 1 establishes the character's identity for all subsequent frames
   - MUST show face clearly from the front (never from back, side, or obscured)
   - Character should be PROPERLY VISIBLE (not zoomed too close, not too far)
   - Include ALL key story elements in Frame 1 (e.g., if story has horse, show horse)
   - Examples: "aldar_kose_man portrait with brown horse, looking at camera, steppe background, front-facing"
4. FRAMES 2+: Ensure aldar_kose_man is PROPERLY VISIBLE and recognizable in each frame
5. **STORY ELEMENT CONSISTENCY**: Maintain ALL key story elements throughout frames
   - If story mentions a HORSE → include horse in ALL relevant frames (riding, racing, with horse, etc.)
   - **IMPORTANT**: Add CONSISTENT VISUAL DESCRIPTORS for recurring elements to maintain appearance
     - For horses: specify color/type (e.g., "brown horse", "white horse", "dark horse")
     - For merchants: specify appearance (e.g., "bearded merchant", "old merchant")
     - For objects: specify details (e.g., "golden bag", "wooden cart")
   - Use the SAME descriptor in ALL frames (e.g., if Frame 1 has "brown horse", ALL frames must say "brown horse")
   - If story mentions a MERCHANT → keep merchant present in relevant frames
   - If story has OBJECTS (bag, gold, etc.) → track them through the story
   - Do NOT drop important story elements halfway through
   - Key elements should appear from Frame 1 and persist throughout with SAME descriptors
6. **BACKGROUND CONSISTENCY**: Keep the background/setting CONSISTENT across frames
   - If story starts in steppe, keep steppe background unless story explicitly changes location
   - If story is at a bazaar, keep bazaar setting throughout
   - Only change background when the story CLEARLY indicates a location change
   - Always EXPLICITLY mention the background/setting in each frame description
   - Examples: "steppe background", "yurt setting", "bazaar background", "mountain backdrop"
7. Minimum 6 scenes, maximum 10 scenes
8. Do NOT mention clothing, outfits, costumes, or physical appearance
9. Make scenes visually distinct through ACTION, not by dropping story elements
10. NEVER use back views, silhouettes, or obscured faces for Frame 1

The character is "aldar_kose_man" - a clever trickster from Kazakh folklore.

FRAME 1 EXAMPLES (MANDATORY front-facing reference with ALL key elements AND descriptors):
- "aldar_kose_man portrait with brown horse, looking at camera, steppe background, front-facing"
- "aldar_kose_man facing camera next to bearded merchant, bazaar background, front-facing"
- "aldar_kose_man looking forward with white horse, yurt setting, portrait, front-facing"

GOOD STORY ELEMENT CONSISTENCY EXAMPLES (with consistent descriptors):

Story with HORSE (race) - Notice "brown horse" used consistently:
  Frame 1: "aldar_kose_man portrait with brown horse, steppe background, front-facing"
  Frame 2: "aldar_kose_man mounting brown horse, steppe background"
  Frame 3: "aldar_kose_man riding brown horse in race, steppe background"
  Frame 4: "aldar_kose_man racing on brown horse, steppe background"
  Frame 5: "aldar_kose_man on brown horse crossing finish, steppe background"
  Frame 6: "aldar_kose_man celebrating on brown horse, steppe background"
  ✅ Horse present in ALL frames
  ✅ SAME descriptor "brown horse" in ALL frames (ensures color consistency)

Story with MERCHANT (trick) - Notice "bearded merchant" used consistently:
  Frame 1: "aldar_kose_man portrait with bearded merchant, bazaar background, front-facing"
  Frame 2: "aldar_kose_man talking to bearded merchant, bazaar background"
  Frame 3: "aldar_kose_man showing goods to bearded merchant, bazaar background"
  Frame 4: "aldar_kose_man receiving gold from bearded merchant, bazaar background"
  ✅ Merchant present in ALL frames
  ✅ SAME descriptor "bearded merchant" in ALL frames

BAD EXAMPLES (dropping story elements OR changing descriptors - NEVER DO THIS):
❌ Story: "Aldar winning race with horse"
  Frame 1: aldar_kose_man with brown horse
  Frame 2: aldar_kose_man riding horse (MISSING COLOR!)
  Frame 3: aldar_kose_man racing on white horse (WRONG COLOR!)
  Frame 4: aldar_kose_man running (NO HORSE - WRONG!)
  
❌ Story: "Aldar tricks merchant"
  Frame 1: aldar_kose_man with old merchant
  Frame 2: aldar_kose_man talking to merchant (MISSING DESCRIPTOR!)
  Frame 3: aldar_kose_man with young merchant (WRONG DESCRIPTOR!)
  Frame 4: aldar_kose_man alone (NO MERCHANT - WRONG!)

GOOD BACKGROUND CONSISTENCY EXAMPLES:
Story in steppe:
  Frame 1: "aldar_kose_man portrait, steppe background, front-facing"
  Frame 2: "aldar_kose_man riding horse, steppe background"
  Frame 3: "aldar_kose_man dismounting horse, steppe background"
  Frame 4: "aldar_kose_man celebrating victory, steppe background"

Story at bazaar:
  Frame 1: "aldar_kose_man portrait, bazaar background, front-facing"
  Frame 2: "aldar_kose_man talking to merchant, bazaar background"
  Frame 3: "aldar_kose_man pointing at goods, bazaar background"

BAD EXAMPLES (inconsistent backgrounds - AVOID):
- Frame 1: steppe, Frame 2: mountains, Frame 3: desert, Frame 4: village
- Changing location every frame when story doesn't indicate movement
- Omitting background/setting from frame descriptions

BAD EXAMPLES for Frame 1 (NEVER do this):
- "aldar_kose_man from behind, riding away"
- "aldar_kose_man silhouette against sunset"
- "aldar_kose_man in the distance, wide shot"
- "back view of aldar_kose_man"

Return a JSON object EXACTLY in this format (no extra text):

EXAMPLE for race story with horse:
{{
    "num_scenes": 6,
    "reasoning": "Six scenes to show the complete race sequence with horse",
    "scenes": [
        {{"frame": 1, "description": "aldar_kose_man portrait with horse, steppe background, front-facing"}},
        {{"frame": 2, "description": "aldar_kose_man mounting horse, steppe background"}},
        {{"frame": 3, "description": "aldar_kose_man riding horse in race, steppe background"}},
        {{"frame": 4, "description": "aldar_kose_man racing on horse, steppe background"}},
        {{"frame": 5, "description": "aldar_kose_man on horse crossing finish, steppe background"}},
        {{"frame": 6, "description": "aldar_kose_man celebrating on horse, steppe background"}}
    ]
}}

EXAMPLE for merchant trick story:
{{
    "num_scenes": 6,
    "reasoning": "Six scenes to show complete trick interaction with merchant",
    "scenes": [
        {{"frame": 1, "description": "aldar_kose_man portrait with merchant, bazaar background, front-facing"}},
        {{"frame": 2, "description": "aldar_kose_man talking to merchant, bazaar background"}},
        {{"frame": 3, "description": "aldar_kose_man showing goods to merchant, bazaar background"}},
        {{"frame": 4, "description": "aldar_kose_man receiving payment from merchant, bazaar background"}},
        {{"frame": 5, "description": "aldar_kose_man leaving with merchant watching, bazaar background"}},
        {{"frame": 6, "description": "aldar_kose_man celebrating, merchant disappointed, bazaar background"}}
    ]
}}

Remember: 
- Frame 1 MUST show the face clearly from the front AND include key story elements
- ALL key elements (horse, merchant, objects) must persist throughout relevant frames
- Character should be PROPERLY VISIBLE in all frames (not too zoomed, not too far)
- Keep same background/setting unless story explicitly changes location
- DO NOT include separate "camera" or "mood" fields
- Put framing instructions (front-facing/portrait) inside the `description` string
"""
class PromptStoryboardGenerator:
    """
    Converts a user's story description into a complete storyboard
    """
    
    def __init__(
        self,
        openai_api_key: str,
        lora_path: str,
        device: str = "cuda",
        use_ref_guided: bool = False,
    ):
        """Initialize the prompt-based storyboard generator"""
        self.client = OpenAI(api_key=openai_api_key)
        self.lora_path = lora_path
        self.device = device
        self.use_ref_guided = use_ref_guided and HAS_REF_GUIDED
        
        # Initialize the image generator (lazy loading)
        self.generator = None
        
        if self.use_ref_guided:
            logger.info("Prompt Storyboard Generator initialized (REFERENCE-GUIDED MODE)")
        else:
            logger.info("Prompt Storyboard Generator initialized (SIMPLE MODE)")
    
    def break_down_story(
        self,
        story: str,
        max_frames: int = 10,
        temperature: float = 0.7,
    ) -> List[Dict]:
        """
        Use OpenAI to break down a story into individual scenes
        
        Args:
            story: The user's story description
            max_frames: Maximum number of frames (GPT decides optimal count up to this limit)
            temperature: GPT-4 creativity (0.0=deterministic, 1.0=creative)
            
        Returns:
            List of scene descriptions with metadata
        """
        logger.info(f"Analyzing story and determining optimal number of scenes (max: {max_frames})...")
        user_prompt = f"""Break down the following story into a storyboard with the OPTIMAL number of scenes.

Story: "{story}"

CRITICAL INSTRUCTIONS:
1. **IDENTIFY KEY STORY ELEMENTS FIRST**:
   - List ALL important elements: characters (horse, merchant, etc.), objects (gold, bag, etc.), setting
   - For each element, assign a CONSISTENT VISUAL DESCRIPTOR that will be used in ALL frames
   - Examples:
     - Horse → choose descriptor: "brown horse", "white horse", "dark horse", "grey horse"
     - Merchant → choose descriptor: "bearded merchant", "old merchant", "wealthy merchant"
     - Object → choose descriptor: "golden bag", "wooden cart", "silver coins"
   - Once chosen, use EXACT SAME descriptor in every single frame
   - These elements MUST appear consistently throughout ALL relevant frames

2. **FRAME 1 MUST BE A REFERENCE SHOT**: 
   - Frame 1 establishes character identity AND key story elements WITH their descriptors
   - MUST be a clear FRONT-FACING portrait showing aldar_kose_man's FACE
   - MUST include ALL key story elements with their DESCRIPTORS visible in Frame 1
   - NEVER from back, side, silhouette, or distance
   - Character should be PROPERLY VISIBLE (not too zoomed in)
   - MUST include clear background/setting
   - Example for race story: "aldar_kose_man portrait with brown horse, steppe background, front-facing"
   - Example for merchant story: "aldar_kose_man portrait with bearded merchant, bazaar background, front-facing"

3. **STORY ELEMENT PERSISTENCE WITH CONSISTENT DESCRIPTORS**:
   - Once you introduce an element with descriptor, keep BOTH in ALL subsequent relevant frames
   - DO NOT drop elements halfway through the story
   - DO NOT change descriptors (e.g., don't switch from "brown horse" to "white horse")
   - Track each element with its EXACT descriptor through the entire narrative
   - Example: Race story → "brown horse" in frames 1, 2, 3, 4, 5, 6 (NOT "horse", "brown horse", "white horse")
   - Example: Trick story → "bearded merchant" in all frames where interaction happens
   - Consistent descriptors = consistent appearance in generated images

4. **BACKGROUND CONSISTENCY**:
   - Identify the PRIMARY SETTING from the story (steppe, bazaar, yurt, mountain, village, etc.)
   - Keep this SAME background/setting in ALL frames unless story explicitly changes location
   - ALWAYS mention the background in EVERY frame description
   - Example: If story is about a race → all frames should have "steppe background"
   - Example: If story is about a bazaar trick → all frames should have "bazaar background"
   - Only change background if story clearly indicates character moves to a different place

5. FRAMES 2-{max_frames}:
   - Show the story progression through CHARACTER ACTIONS with consistent elements
   - Ensure aldar_kose_man is PROPERLY VISIBLE in each frame
   - Keep descriptions SHORT and SIMPLE (action + key elements + background, ~10-12 words)
   - Always include the background/setting AND key story elements explicitly
   - Example: "aldar_kose_man racing on horse, steppe background" (NOT "aldar_kose_man racing")

6. General Rules:
   - Decide how many scenes needed (minimum 6, maximum {max_frames})
   - Use "aldar_kose_man" as the character identifier
   - DO NOT include separate camera or mood fields in the JSON output
   - DO NOT mention clothing or appearance

Return JSON object following these EXACT examples:

FOR RACE STORY (with horse):
{{
    "num_scenes": 6,
    "reasoning": "Six scenes to show complete race with horse",
    "scenes": [
        {{"frame": 1, "description": "aldar_kose_man portrait with horse, steppe background, front-facing"}},
        {{"frame": 2, "description": "aldar_kose_man mounting horse, steppe background"}},
        {{"frame": 3, "description": "aldar_kose_man riding horse in race, steppe background"}},
        {{"frame": 4, "description": "aldar_kose_man racing on horse, steppe background"}},
        {{"frame": 5, "description": "aldar_kose_man on horse crossing finish, steppe background"}},
        {{"frame": 6, "description": "aldar_kose_man celebrating on horse, steppe background"}}
    ]
}}

FOR MERCHANT TRICK STORY:
{{
    "num_scenes": 6,
    "reasoning": "Six scenes to show trick interaction with merchant",
    "scenes": [
        {{"frame": 1, "description": "aldar_kose_man portrait with merchant, bazaar background, front-facing"}},
        {{"frame": 2, "description": "aldar_kose_man talking to merchant, bazaar background"}},
        {{"frame": 3, "description": "aldar_kose_man showing goods to merchant, bazaar background"}},
        {{"frame": 4, "description": "aldar_kose_man receiving gold from merchant, bazaar background"}},
        {{"frame": 5, "description": "aldar_kose_man leaving with gold, merchant watching, bazaar background"}},
        {{"frame": 6, "description": "aldar_kose_man celebrating, merchant disappointed, bazaar background"}}
    ]
}}

CRITICAL: 
- Frame 1 MUST include ALL key story elements (horse, merchant, etc.)
- ALL elements must persist through ALL relevant frames
- Keep same background unless story explicitly changes location
- DO NOT drop horse/merchant/objects halfway through!"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": SCENE_BREAKDOWN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,  # Use provided temperature for determinism/creativity control
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content

            # Parse JSON response
            try:
                result = json.loads(content)

                # Extract metadata if present
                num_scenes_decided = result.get("num_scenes", None)
                reasoning = result.get("reasoning", None)

                if reasoning:
                    logger.info(f"GPT-4 reasoning: {reasoning}")
                    logger.info(f"GPT-4 decided on {num_scenes_decided} scenes")

                # Expect the canonical structure with scenes array; fall back safely
                if isinstance(result, dict) and "scenes" in result and isinstance(result["scenes"], list):
                    scenes = result["scenes"]
                elif isinstance(result, list):
                    scenes = result
                else:
                    # If unexpected, try to extract list-like values
                    scenes = []
                    for v in result.values() if isinstance(result, dict) else []:
                        if isinstance(v, list):
                            scenes = v
                            break

                # Validate we got scenes
                if not scenes or len(scenes) == 0:
                    logger.error("No scenes generated!")
                    logger.error(f"Response: {content}")
                    raise ValueError("OpenAI returned no scenes")

                # Validate within limits
                if len(scenes) > max_frames:
                    logger.warning(f"Got {len(scenes)} scenes, trimming to max {max_frames}")
                    scenes = scenes[:max_frames]

                if len(scenes) < 6:
                    logger.warning(f"Got only {len(scenes)} scenes, minimum is 6. Padding...")
                    for i in range(len(scenes), 6):
                        scenes.append({
                            "frame": i + 1,
                            "description": f"aldar_kose_man continuation scene {i+1}"
                        })
                
                # Validate Frame 1 is a proper reference shot
                if scenes and len(scenes) > 0:
                    frame1_desc = scenes[0].get("description", "").lower()
                    bad_keywords = ["back", "behind", "silhouette", "distance", "far", "away from camera"]
                    has_bad_keyword = any(bad in frame1_desc for bad in bad_keywords)
                    has_front_keyword = any(kw in frame1_desc for kw in ["front", "portrait", "face", "looking"])
                    
                    if has_bad_keyword or not has_front_keyword:
                        logger.warning("Frame 1 is not a proper front-facing reference! Fixing...")
                        # Force Frame 1 to be a clear reference shot
                        scenes[0] = {
                            "frame": 1,
                            "description": "aldar_kose_man portrait, looking at camera, steppe background, front-facing"
                        }
                        logger.info("✓ Frame 1 fixed to be a proper front-facing reference shot")
                
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
            # Extract description and mood
            if isinstance(scene, dict):
                description = scene.get("description", scene.get("prompt", str(scene)))
                mood = scene.get("mood", "")
            else:
                description = str(scene)
                mood = ""
            
            # Ensure trigger token is present
            if TRIGGER_TOKEN not in description.lower():
                description = f"{TRIGGER_TOKEN} {description}"
            
            # Build prompt with mood if available
            if mood:
                prompt = f"{description}, {mood} mood, high quality, detailed, 3D animation, professional render"
            else:
                prompt = f"{description}, high quality, detailed, 3D animation, professional render"
            
            prompts.append(prompt)
        
        return prompts
    
    def generate_storyboard(
        self,
        story: str,
        num_frames: int = None,  # None = let GPT decide
        output_dir: str = "outputs/prompt_storyboard",
        base_seed: int = 42,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.5,
        max_attempts: int = 2,
        temperature: float = 0.7,  # GPT temperature for scene breakdown
    ) -> List[Image.Image]:
        """
        Complete pipeline: story -> scenes -> images
        
        Args:
            story: User's story description
            num_frames: Max number of frames (None = GPT decides, max 10)
            output_dir: Where to save results
            base_seed: Base seed for generation
            num_inference_steps: SDXL steps
            guidance_scale: SDXL guidance
            max_attempts: Retry attempts per frame
            temperature: GPT-4 creativity (0.0=deterministic, 1.0=creative)
            
        Returns:
            List of generated PIL Images
        """
        # Step 1: Break down story into scenes
        logger.info("=" * 60)
        logger.info("STEP 1: Breaking down story into scenes")
        logger.info("=" * 60)
        
        max_frames = num_frames if num_frames is not None else 10
        scenes = self.break_down_story(story, max_frames=max_frames, temperature=temperature)
        
        # Save scene breakdown
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        scene_file = output_path / "scene_breakdown.json"
        with open(scene_file, 'w', encoding='utf-8') as f:
            json.dump({
                "story": story,
                "timestamp": datetime.now().isoformat(),
                "num_frames": len(scenes),
                "max_frames_requested": max_frames,
                "gpt_decided": num_frames is None,
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
            
            if self.use_ref_guided:
                logger.info("Loading Reference-Guided Generator (IP-Adapter + ControlNet)...")
                self.generator = ReferenceGuidedStoryboardGenerator(
                    lora_path=self.lora_path,
                    device=self.device,
                    use_controlnet=True,
                    use_ip_adapter=True,
                    ip_adapter_model="h94/IP-Adapter",  # Enable IP-Adapter for better face consistency
                )
            else:
                logger.info("Loading Simple Generator (LoRA + CLIP)...")
                self.generator = SimplifiedStoryboardGenerator(
                    lora_path=self.lora_path,
                    device=self.device,
                )
        
        # Step 4: Generate images
        logger.info("=" * 60)
        logger.info("STEP 4: Generating storyboard images")
        logger.info("=" * 60)
        
        # Enable random seeds for creative mode (temperature > 0)
        use_random_seed = temperature > 0.0
        if use_random_seed:
            logger.info(f"🎲 Creative mode enabled (temperature={temperature})")
            logger.info("   Images will have random variations even with same seed")
        else:
            logger.info(f"🔒 Deterministic mode (temperature={temperature})")
            logger.info("   Same seed + same prompt = identical output")
        
        frames = self.generator.generate_sequence(
            prompts=prompts,
            output_dir=str(output_path),
            base_seed=base_seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            consistency_threshold=0.70,  # CLIP similarity threshold
            max_retries=max_attempts,  # Correct parameter name
            use_random_seed=use_random_seed,  # NEW: Pass randomness flag
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
  # Let GPT decide optimal number of frames
  python scripts/prompt_storyboard.py --lora-path outputs/checkpoints/checkpoint-400
  
  # Direct story (GPT decides frame count)
  python scripts/prompt_storyboard.py \\
      --lora-path outputs/checkpoints/checkpoint-400 \\
      --story "Aldar Kose riding his horse to his yurt at sunset"
  
  # Set max frames (GPT decides up to 10)
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
        default=None,
        choices=range(6, 11),
        metavar="6-10",
        help="Max number of frames (6-10). If not specified, GPT-4 decides optimal count (max 10)"
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
    
    parser.add_argument(
        "--use-ref-guided",
        action="store_true",
        help="Use reference-guided mode (IP-Adapter + ControlNet for max consistency)"
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
        use_ref_guided=args.use_ref_guided,
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
