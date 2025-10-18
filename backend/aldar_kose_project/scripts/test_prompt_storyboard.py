#!/usr/bin/env python3
"""
Quick test for prompt-based storyboard generation

This script tests the story breakdown WITHOUT running the full image generation,
so you can verify the scene breakdown is good before spending GPU time.

Usage:
    python scripts/test_prompt_storyboard.py
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SCENE_BREAKDOWN_SYSTEM_PROMPT = """You are an expert storyboard artist specializing in clear, simple visual storytelling.

Your task is to break down a story into 6-10 distinct, SIMPLE scenes for image generation.

CRITICAL RULES:
1. **FRAME 1 IS MANDATORY**: Must ALWAYS show aldar_kose_man's FACE clearly from the FRONT
   - Frame 1 establishes the character's identity for all subsequent frames
   - MUST show face clearly from the front (never from back, side, or obscured)
   - Character should be PROPERLY VISIBLE (not too zoomed in, not too far away)
2. Keep descriptions SIMPLE and CLEAR - focus on ONE main action per scene
3. Each scene should be DIFFERENT - show story PROGRESSION
4. DO NOT mention clothing, outfits, costumes, or physical appearance
5. Ensure aldar_kose_man is PROPERLY VISIBLE in each frame
6. Make scenes visually distinct from each other

The character is "aldar_kose_man" - a clever trickster from Kazakh folklore.

FRAME 1 EXAMPLES (MANDATORY front-facing reference):
✅ "aldar_kose_man portrait, looking at camera, steppe background, front-facing"
✅ "aldar_kose_man facing camera, slight smile, outdoors, front-facing"
✅ "aldar_kose_man looking forward, yurt background, portrait, front-facing"

FRAMES 2+ EXAMPLES (character properly visible):
✅ "aldar_kose_man riding horse, steppe, front-facing"
✅ "aldar_kose_man entering yurt, warm light"
✅ "aldar_kose_man laughing, warm light"

❌ BAD for Frame 1 (NEVER do this):
- "aldar_kose_man from behind, riding away"
- "aldar_kose_man silhouette against sunset"
- "back view of aldar_kose_man"

Return JSON object exactly:
{
    "num_scenes": <number>,
    "reasoning": "<one short sentence>",
    "scenes": [
        {"frame": 1, "description": "aldar_kose_man portrait, looking at camera, steppe background, front-facing"},
        {"frame": 2, "description": "aldar_kose_man riding horse, steppe"}
    ]
}

Remember: Frame 1 MUST show the face clearly from the front with character properly visible (this is the reference for all other frames)"""


def test_story_breakdown(story: str, max_frames: int = 10):
    """Test the story breakdown without generating images"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("Set it with: export OPENAI_API_KEY=your_key_here")
        return
    
    client = OpenAI(api_key=api_key)
    
    print("=" * 70)
    print("TESTING PROMPT-BASED STORYBOARD BREAKDOWN")
    print("=" * 70)
    print(f"\nStory: {story}")
    print(f"Max frames: {max_frames} (GPT-4 will decide optimal count)\n")
    
    user_prompt = f"""Break down the following story into a storyboard with the OPTIMAL number of scenes.

Story: "{story}"

INSTRUCTIONS:
- Decide how many scenes needed (minimum 6, maximum {max_frames})
- Simple stories: 6-7 scenes, Medium: 8-9 scenes, Complex: 10 scenes
- Keep descriptions SHORT and SIMPLE (max 10-12 words each)
- Focus on: ACTION + LOCATION + CAMERA ANGLE only
- Each scene must be DIFFERENT and show PROGRESSION
- Use "aldar_kose_man" as the character identifier
- NO clothing, costumes, or complex metaphors

EXAMPLES OF SIMPLE DESCRIPTIONS:
✅ "aldar_kose_man riding horse, steppe landscape, wide shot"
✅ "aldar_kose_man entering yurt, medium shot, sunset lighting"
✅ "aldar_kose_man laughing, close-up, warm light"
❌ "aldar_kose_man atop noble steed silhouetted against infinite horizon" (too complex!)

Return JSON object:
{{
  "num_scenes": <number>,
  "reasoning": "<why this number>",
  "scenes": [
    {{"frame": 1, "description": "simple action + location + camera", "camera": "wide shot", "mood": "one word"}},
    {{"frame": 2, "description": "...", "camera": "...", "mood": "..."}},
    ... ({max_frames} max)
  ]
}}"""

    try:
        print("🔄 Calling OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": SCENE_BREAKDOWN_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON
        result = json.loads(content)
        
        # Extract metadata
        num_scenes_decided = result.get("num_scenes", None)
        reasoning = result.get("reasoning", "Not provided")
        
        print(f"✓ GPT-4 Decision: {num_scenes_decided} scenes")
        print(f"✓ Reasoning: {reasoning}\n")
        
        # Handle different response structures
        if isinstance(result, dict) and "scenes" in result:
            scenes = result["scenes"]
        elif isinstance(result, dict) and "frames" in result:
            scenes = result["frames"]
        elif isinstance(result, list):
            scenes = result
        else:
            scenes = list(result.values())
        
        print(f"✓ Generated {len(scenes)} scenes\n")
        
        # Display scenes
        print("=" * 70)
        print("SCENE BREAKDOWN:")
        print("=" * 70)
        
        for i, scene in enumerate(scenes, 1):
            print(f"\n🎬 Frame {i}")
            print("-" * 70)
            if isinstance(scene, dict):
                print(f"Description: {scene.get('description', scene.get('prompt', 'N/A'))}")
                print(f"Camera: {scene.get('camera', 'N/A')}")
                print(f"Mood: {scene.get('mood', 'N/A')}")
            else:
                print(f"Description: {scene}")
        
        print("\n" + "=" * 70)
        print("✓ TEST COMPLETE - Scene breakdown looks good!")
        print("=" * 70)
        print("\nTo generate the full storyboard with images, run:")
        print(f'python scripts/prompt_storyboard.py \\')
        print(f'    --lora-path outputs/checkpoints/checkpoint-400 \\')
        print(f'    --story "{story}"')
        print(f'# GPT-4 will decide optimal frame count (decided: {num_scenes_decided})')
        print("=" * 70 + "\n")
        
        # Save to file
        with open("test_scene_breakdown.json", "w", encoding="utf-8") as f:
            json.dump({
                "story": story,
                "max_frames": max_frames,
                "num_scenes_decided": num_scenes_decided,
                "reasoning": reasoning,
                "scenes": scenes
            }, f, indent=2, ensure_ascii=False)
        
        print("💾 Saved scene breakdown to: test_scene_breakdown.json\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test stories
    test_stories = [
        "Aldar Kose riding his horse across the steppe towards his yurt at sunset",
        "Aldar Kose tricks a wealthy merchant by selling him a magical pot",
        "Aldar Kose playing music for villagers around a campfire under the stars",
    ]
    
    print("\nSelect a test story or enter your own:")
    print("\n1. " + test_stories[0])
    print("2. " + test_stories[1])
    print("3. " + test_stories[2])
    print("4. Enter custom story\n")
    
    choice = input("Choice (1-4): ").strip()
    
    if choice == "1":
        story = test_stories[0]
    elif choice == "2":
        story = test_stories[1]
    elif choice == "3":
        story = test_stories[2]
    elif choice == "4":
        story = input("\nEnter your story: ").strip()
        if not story:
            print("❌ No story provided!")
            exit(1)
    else:
        print("❌ Invalid choice!")
        exit(1)
    
    max_frames = input("\nMax frames (6-10, default 10 - GPT decides): ").strip()
    max_frames = int(max_frames) if max_frames else 10
    
    if max_frames < 6 or max_frames > 10:
        print("❌ Max frames must be between 6 and 10!")
        exit(1)
    
    test_story_breakdown(story, max_frames)
