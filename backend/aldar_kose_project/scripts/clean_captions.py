#!/usr/bin/env python3
"""
Clean existing captions by removing clothing/outfit descriptions

This script reads all caption files and removes clothing-related phrases,
keeping only action, pose, setting, and artistic style descriptions.
"""

import re
from pathlib import Path
import shutil
from datetime import datetime

# Clothing-related keywords to remove
CLOTHING_PATTERNS = [
    # Specific clothing items
    r'\bwearing\s+[^,\.]+',
    r'\bin\s+(a\s+)?(brown|orange|green|blue|red|yellow|patterned|traditional|ornate|textured|patched|sleeveless)\s+(coat|robe|jacket|shirt|tunic|vest|pants|attire|outfit|clothing|dress|uniform)',
    r'\b(brown|orange|green|blue|red|yellow|patterned|traditional|ornate|textured|patched|sleeveless)\s+(coat|robe|jacket|shirt|tunic|vest|pants|attire|outfit|clothing|dress|uniform)',
    r'\b(coat|robe|jacket|shirt|tunic|vest|pants|attire|outfit|clothing|dress|uniform)\b',
    
    # Headwear
    r'\bwearing\s+(a\s+)?(green|brown|orange|patterned|traditional)\s+(hat|cap|headband|headscarf|headdress)',
    r'\b(green|brown|orange|patterned|traditional)\s+(hat|cap|headband|headscarf|headdress)',
    r'\b(hat|cap|headband|headscarf|headdress)\b',
    
    # Generic clothing phrases
    r'\btraditional\s+attire\b',
    r'\btraditional\s+clothing\b',
    r'\bin\s+traditional\s+\w+',
    r'\bornate\s+costume\b',
    r'\bcasual\s+pose\b',
    r'\band\s+(dark|light|brown|orange|green)\s+pants',
    
    # Compound patterns
    r',\s*wearing\s+[^,\.]+',
    r',\s*in\s+(a\s+)?(brown|orange|green|blue)\s+\w+',
]

# Clean up extra commas, spaces, and conjunction issues
def clean_text(text):
    """Clean up text after removing phrases"""
    # Remove double commas
    text = re.sub(r',\s*,', ',', text)
    # Remove leading/trailing commas
    text = re.sub(r'^\s*,\s*', '', text)
    text = re.sub(r',\s*$', '', text)
    # Remove double spaces
    text = re.sub(r'\s+', ' ', text)
    # Fix orphaned 'and'
    text = re.sub(r',\s+and\s+,', ',', text)
    text = re.sub(r'\sand\s+,', ',', text)
    text = re.sub(r',\s+and\s+\.', '.', text)
    # Clean up spacing
    text = text.strip()
    return text


def remove_clothing_descriptions(caption):
    """Remove clothing-related descriptions from caption"""
    original = caption
    
    # Apply all patterns
    for pattern in CLOTHING_PATTERNS:
        caption = re.sub(pattern, '', caption, flags=re.IGNORECASE)
    
    # Clean up the result
    caption = clean_text(caption)
    
    # Ensure it starts with trigger token
    if not caption.startswith('aldar_kose_man'):
        caption = f"aldar_kose_man {caption}"
    
    return caption


def main():
    captions_dir = Path("data/captions")
    
    if not captions_dir.exists():
        print(f"Error: {captions_dir} not found")
        return
    
    # Create backup
    backup_dir = Path("data/captions_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating backup in: {backup_dir}")
    
    # Get all caption files
    caption_files = list(captions_dir.glob("*.txt"))
    caption_files = [f for f in caption_files if f.name != ".gitkeep"]
    
    print(f"Found {len(caption_files)} caption files")
    print()
    
    changes_made = 0
    
    for caption_file in sorted(caption_files):
        # Read original
        with open(caption_file, 'r', encoding='utf-8') as f:
            original = f.read().strip()
        
        # Backup original
        backup_file = backup_dir / caption_file.name
        shutil.copy2(caption_file, backup_file)
        
        # Clean caption
        cleaned = remove_clothing_descriptions(original)
        
        # Show changes
        if cleaned != original:
            print(f"{'='*70}")
            print(f"FILE: {caption_file.name}")
            print(f"BEFORE: {original}")
            print(f"AFTER:  {cleaned}")
            print()
            
            # Write cleaned version
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            
            changes_made += 1
        else:
            print(f"UNCHANGED: {caption_file.name}")
    
    print()
    print("="*70)
    print(f"✓ Complete!")
    print(f"  Files processed: {len(caption_files)}")
    print(f"  Files changed:   {changes_made}")
    print(f"  Backup location: {backup_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
