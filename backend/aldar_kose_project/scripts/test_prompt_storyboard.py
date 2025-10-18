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
    }
]

Remember: NEVER mention clothing, costumes, or physical appearance. Focus on what's HAPPENING, WHERE, and HOW it's SHOT."""


def test_story_breakdown(story: str, num_frames: int = 8):
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
    print(f"Frames: {num_frames}\n")
    
    user_prompt = f"""Break down the following story into exactly {num_frames} distinct, visually compelling scenes for a storyboard.

Story: "{story}"

Remember:
- {num_frames} scenes showing PROGRESSION through the story
- Each scene should be visually DIFFERENT
- Include camera angles and lighting
- Focus on ACTION and SETTING, NOT clothing or appearance
- Use "aldar_kose_man" as the character identifier

Return ONLY a valid JSON array with the structure specified in the system prompt."""

    try:
        print("🔄 Calling OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": SCENE_BREAKDOWN_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON
        result = json.loads(content)
        
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
        print(f'    --story "{story}" \\')
        print(f'    --num-frames {num_frames}')
        print("=" * 70 + "\n")
        
        # Save to file
        with open("test_scene_breakdown.json", "w", encoding="utf-8") as f:
            json.dump({
                "story": story,
                "num_frames": num_frames,
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
    
    num_frames = input("\nNumber of frames (6-10, default 8): ").strip()
    num_frames = int(num_frames) if num_frames else 8
    
    if num_frames < 6 or num_frames > 10:
        print("❌ Number of frames must be between 6 and 10!")
        exit(1)
    
    test_story_breakdown(story, num_frames)
