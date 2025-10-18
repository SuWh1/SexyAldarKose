#!/usr/bin/env python3
"""
Test client for Aldar Kose Storyboard API

Usage:
    python api/test_client.py --prompt "Aldar Kose tricks a merchant"
    python api/test_client.py --prompt "Story..." --use-ref-guided --output output_dir
"""

import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


def test_health(base_url: str):
    """Test health endpoint"""
    print(f"Testing health endpoint: {base_url}/health")
    response = requests.get(f"{base_url}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Server is {data['status']}")
        print(f"  Model loaded: {data['model_loaded']}")
        print(f"  LoRA path: {data['lora_path']}")
        print(f"  Ref-guided available: {data['ref_guided_available']}")
        return True
    else:
        print(f"✗ Health check failed: {response.status_code}")
        return False


def generate_story(
    base_url: str,
    prompt: str,
    use_ref_guided: bool = False,
    num_frames: int = None,
    seed: int = 42,
    gpt_temperature: float = 0.7,
    output_dir: str = None
):
    """Generate story and save frames"""
    print(f"\n{'='*60}")
    print(f"Generating story...")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Mode: {'ref-guided' if use_ref_guided else 'simple'}")
    print(f"Seed: {seed}")
    print(f"GPT Temperature: {gpt_temperature}")
    print(f"Max frames: {num_frames or 'auto'}")
    
    # Prepare request
    payload = {
        "prompt": prompt,
        "use_ref_guided": use_ref_guided,
        "seed": seed,
        "gpt_temperature": gpt_temperature
    }
    
    if num_frames:
        payload["num_frames"] = num_frames
    
    # Send request
    print("\nSending request to API...")
    response = requests.post(
        f"{base_url}/generate",
        json=payload,
        timeout=600  # 10 minutes timeout
    )
    
    if response.status_code != 200:
        print(f"✗ Request failed: {response.status_code}")
        print(response.text)
        return False
    
    # Parse response
    data = response.json()
    
    print(f"\n{'='*60}")
    print(f"Generation Complete!")
    print(f"{'='*60}")
    print(f"Success: {data['success']}")
    print(f"Frames: {data['num_frames']}")
    print(f"Time: {data['generation_time_seconds']:.1f}s")
    print(f"Mode: {data['mode']}")
    
    # Create output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path(f"outputs/api_test_{data['mode']}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save frames
    print(f"\nSaving frames to: {output_path}")
    
    for frame_data in data["frames"]:
        frame_num = frame_data["frame_number"]
        frame_prompt = frame_data["prompt"]
        clip_score = frame_data["clip_score"]
        
        # Decode base64 image
        img_bytes = base64.b64decode(frame_data["image"])
        img = Image.open(BytesIO(img_bytes))
        
        # Save image
        img_path = output_path / f"frame_{frame_num:03d}.png"
        img.save(img_path)
        
        print(f"  Frame {frame_num}: {frame_prompt[:60]}... (CLIP: {clip_score:.3f})")
    
    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        # Remove base64 images from metadata (too large)
        metadata = data.copy()
        for frame in metadata["frames"]:
            frame["image"] = f"frame_{frame['frame_number']:03d}.png"
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ All frames saved to: {output_path}")
    print(f"✓ Metadata saved to: {metadata_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Aldar Kose Storyboard API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--prompt", required=True, help="Story prompt")
    parser.add_argument("--use-ref-guided", action="store_true", help="Use reference-guided mode")
    parser.add_argument("--num-frames", type=int, help="Max frames (6-10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--gpt-temperature", type=float, default=0.7, help="GPT-4 temperature (0.0-1.0)")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--skip-health", action="store_true", help="Skip health check")
    
    args = parser.parse_args()
    
    # Test health
    if not args.skip_health:
        if not test_health(args.url):
            print("\n✗ Health check failed. Is the server running?")
            print(f"   Start with: python api/server.py")
            sys.exit(1)
    
    # Generate story
    success = generate_story(
        base_url=args.url,
        prompt=args.prompt,
        use_ref_guided=args.use_ref_guided,
        num_frames=args.num_frames,
        seed=args.seed,
        gpt_temperature=args.gpt_temperature,
        output_dir=args.output
    )
    
    if success:
        print("\n✓ Test completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
