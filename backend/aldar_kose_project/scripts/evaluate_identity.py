#!/usr/bin/env python3
"""
Evaluation Script - Compute Identity Consistency Metrics

This script evaluates the identity consistency of generated images
using CLIP embeddings and cosine similarity.

Usage:
    python scripts/evaluate_identity.py --reference_image data/images/reference.jpg --generated_dir outputs/generated_images
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple

try:
    import clip
except ImportError:
    print("Error: CLIP not installed. Install with: pip install clip-by-openai")
    exit(1)

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate identity consistency with CLIP")
    parser.add_argument(
        "--reference_image",
        type=str,
        required=True,
        help="Path to reference image of Aldar Kose",
    )
    parser.add_argument(
        "--generated_dir",
        type=str,
        required=True,
        help="Directory containing generated images to evaluate",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/identity_scores.txt",
        help="File to save evaluation results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
        help="CLIP model variant to use",
    )
    
    return parser.parse_args()


def load_clip_model(model_name: str = "ViT-B/32"):
    """Load CLIP model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device


def get_image_embedding(image_path: str, model, preprocess, device):
    """Get CLIP embedding for an image"""
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu().numpy()


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings"""
    similarity = np.dot(emb1.flatten(), emb2.flatten())
    return float(similarity)


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def evaluate_identity_consistency(
    reference_image: str,
    generated_dir: str,
    model_name: str = "ViT-B/32",
) -> Tuple[List[Tuple[str, float]], dict]:
    """
    Evaluate identity consistency between reference and generated images
    
    Returns:
        List of (filename, similarity_score) tuples and statistics dict
    """
    print("Loading CLIP model...")
    model, preprocess, device = load_clip_model(model_name)
    
    print(f"Computing reference embedding from: {reference_image}")
    reference_emb = get_image_embedding(reference_image, model, preprocess, device)
    
    # Get generated images
    generated_path = Path(generated_dir)
    image_files = get_image_files(generated_path)
    
    if not image_files:
        print(f"No images found in {generated_dir}")
        return [], {}
    
    print(f"\nEvaluating {len(image_files)} generated images...")
    
    scores = []
    
    for image_path in tqdm(image_files, desc="Computing similarities"):
        try:
            gen_emb = get_image_embedding(str(image_path), model, preprocess, device)
            similarity = compute_cosine_similarity(reference_emb, gen_emb)
            scores.append((image_path.name, similarity))
        except Exception as e:
            print(f"\nError processing {image_path.name}: {e}")
    
    # Compute statistics
    if scores:
        similarity_values = [score for _, score in scores]
        stats = {
            'mean': np.mean(similarity_values),
            'std': np.std(similarity_values),
            'min': np.min(similarity_values),
            'max': np.max(similarity_values),
            'median': np.median(similarity_values),
        }
    else:
        stats = {}
    
    return scores, stats


def save_results(scores: List[Tuple[str, float]], stats: dict, output_file: str):
    """Save evaluation results to file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  IDENTITY CONSISTENCY EVALUATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        if stats:
            f.write("Overall Statistics:\n")
            f.write(f"  Mean Similarity:   {stats['mean']:.4f}\n")
            f.write(f"  Std Deviation:     {stats['std']:.4f}\n")
            f.write(f"  Min Similarity:    {stats['min']:.4f}\n")
            f.write(f"  Max Similarity:    {stats['max']:.4f}\n")
            f.write(f"  Median Similarity: {stats['median']:.4f}\n")
            f.write("\n" + "-" * 70 + "\n\n")
        
        f.write("Individual Image Scores:\n\n")
        
        # Sort by similarity (highest first)
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        for filename, score in sorted_scores:
            f.write(f"  {score:.4f}  {filename}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"\nResults saved to: {output_path}")


def print_results(scores: List[Tuple[str, float]], stats: dict):
    """Print evaluation results to console"""
    print("\n" + "=" * 70)
    print("  IDENTITY CONSISTENCY EVALUATION RESULTS")
    print("=" * 70)
    
    if stats:
        print("\nOverall Statistics:")
        print(f"  Mean Similarity:   {stats['mean']:.4f}")
        print(f"  Std Deviation:     {stats['std']:.4f}")
        print(f"  Min Similarity:    {stats['min']:.4f}")
        print(f"  Max Similarity:    {stats['max']:.4f}")
        print(f"  Median Similarity: {stats['median']:.4f}")
        
        # Interpretation
        print("\nInterpretation:")
        if stats['mean'] > 0.85:
            print("  ✅ Excellent identity consistency")
        elif stats['mean'] > 0.75:
            print("  ✓  Good identity consistency")
        elif stats['mean'] > 0.65:
            print("  ~  Moderate identity consistency")
        else:
            print("  ⚠  Low identity consistency - consider more training")
        
        print("\n" + "-" * 70)
        print("\nTop 5 Most Similar Images:")
        
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        for i, (filename, score) in enumerate(sorted_scores[:5], 1):
            print(f"  {i}. {score:.4f}  {filename}")
    
    print("\n" + "=" * 70)


def main():
    args = parse_args()
    
    # Validate inputs
    if not Path(args.reference_image).exists():
        print(f"Error: Reference image not found: {args.reference_image}")
        return 1
    
    if not Path(args.generated_dir).exists():
        print(f"Error: Generated images directory not found: {args.generated_dir}")
        return 1
    
    # Run evaluation
    scores, stats = evaluate_identity_consistency(
        args.reference_image,
        args.generated_dir,
        args.model,
    )
    
    if not scores:
        print("No images were evaluated successfully.")
        return 1
    
    # Display results
    print_results(scores, stats)
    
    # Save results
    save_results(scores, stats, args.output_file)
    
    return 0


if __name__ == "__main__":
    exit(main())
