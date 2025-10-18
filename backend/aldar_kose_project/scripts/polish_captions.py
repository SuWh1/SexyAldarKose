#!/usr/bin/env python3
"""
Polish cleaned captions - remove artifacts and improve readability
"""

import re
from pathlib import Path

def polish_caption(caption):
    """Polish caption by removing artifacts and cleaning up"""
    
    # Remove orphaned words after cleaning
    caption = re.sub(r'\b(and|or|with)\s+(dark|light|brown|orange|green|blue|tan|gray|grey)\s*,', ',', caption)
    caption = re.sub(r',\s+(dark|light|brown|orange|green|blue|tan|gray|grey)\s+,', ',', caption)
    caption = re.sub(r',\s+(dark|light|brown|orange|green|blue|tan|gray|grey)\s*\.', '.', caption)
    
    # Remove standalone color words
    caption = re.sub(r'\s+(dark|light|brown|orange|green|blue|tan|gray|grey)\s+and\s+', ' ', caption)
    caption = re.sub(r'\bin\s+(tan|brown|gray|grey)\s+(and|,)', '', caption)
    
    # Remove "traditional ,"
    caption = re.sub(r'traditional\s*,', '', caption)
    
    # Remove orphaned "in ," or "in ."
    caption = re.sub(r'\bin\s*,', ',', caption)
    caption = re.sub(r'\bin\s*\.', '.', caption)
    
    # Remove "Wears an and" artifacts  
    caption = re.sub(r'Wears?\s+an?\s+(and|,|\.)', '', caption)
    
    # Fix "character in ,"
    caption = re.sub(r'character\s+in\s*,', 'character,', caption)
    
    # Remove double spaces
    caption = re.sub(r'\s+', ' ', caption)
    
    # Remove double commas
    caption = re.sub(r',\s*,+', ',', caption)
    
    # Remove comma before period
    caption = re.sub(r',\s*\.', '.', caption)
    
    # Remove leading/trailing punctuation
    caption = re.sub(r'^\s*[,\s]+', '', caption)
    caption = re.sub(r'[,\s]+$', '', caption)
    
    # Fix spacing around punctuation
    caption = re.sub(r'\s+,', ',', caption)
    caption = re.sub(r'\s+\.', '.', caption)
    caption = re.sub(r',(\S)', r', \1', caption)
    caption = re.sub(r'\.(\S)', r'. \1', caption)
    
    # Ensure trigger token at start
    if not caption.startswith('aldar_kose_man'):
        caption = f"aldar_kose_man {caption.lstrip()}"
    
    # Final cleanup
    caption = caption.strip()
    
    return caption


def main():
    captions_dir = Path("data/captions")
    
    if not captions_dir.exists():
        print(f"Error: {captions_dir} not found")
        return
    
    caption_files = [f for f in captions_dir.glob("*.txt") if f.name != ".gitkeep"]
    
    print(f"Polishing {len(caption_files)} caption files...")
    print()
    
    changes = 0
    
    for caption_file in sorted(caption_files):
        with open(caption_file, 'r', encoding='utf-8') as f:
            original = f.read().strip()
        
        polished = polish_caption(original)
        
        if polished != original:
            print(f"{'='*70}")
            print(f"FILE: {caption_file.name}")
            print(f"BEFORE: {original}")
            print(f"AFTER:  {polished}")
            print()
            
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(polished)
            
            changes += 1
    
    print("="*70)
    print(f"✓ Polishing complete!")
    print(f"  Files changed: {changes}")
    print("="*70)


if __name__ == "__main__":
    main()
