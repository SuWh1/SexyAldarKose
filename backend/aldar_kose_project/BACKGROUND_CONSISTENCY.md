# Background Consistency Improvements

## Problem
Backgrounds were changing drastically between frames (e.g., steppe → mountains → desert → village), making the storyboard feel disjointed and inconsistent.

## Solution
Updated GPT-4 prompt engineering to enforce background consistency across all frames.

---

## Changes Made

### 1. **Added Background Consistency Rule**
```
**BACKGROUND CONSISTENCY**: Keep the background/setting CONSISTENT across frames
- If story starts in steppe, keep steppe background unless story explicitly changes location
- If story is at a bazaar, keep bazaar setting throughout
- Only change background when the story CLEARLY indicates a location change
- Always EXPLICITLY mention the background/setting in each frame description
```

### 2. **Enhanced Examples with Consistent Backgrounds**

**GOOD Examples:**
```
Story in steppe:
  Frame 1: "aldar_kose_man portrait, steppe background, front-facing"
  Frame 2: "aldar_kose_man riding horse, steppe background"
  Frame 3: "aldar_kose_man dismounting horse, steppe background"
  Frame 4: "aldar_kose_man celebrating victory, steppe background"

Story at bazaar:
  Frame 1: "aldar_kose_man portrait, bazaar background, front-facing"
  Frame 2: "aldar_kose_man talking to merchant, bazaar background"
  Frame 3: "aldar_kose_man pointing at goods, bazaar background"
```

**BAD Examples (now explicitly shown to avoid):**
```
- Frame 1: steppe, Frame 2: mountains, Frame 3: desert, Frame 4: village
- Changing location every frame when story doesn't indicate movement
- Omitting background/setting from frame descriptions
```

### 3. **Updated User Prompt Template**

Added explicit instructions:
```
**BACKGROUND CONSISTENCY**:
- Identify the PRIMARY SETTING from the story (steppe, bazaar, yurt, mountain, village, etc.)
- Keep this SAME background/setting in ALL frames unless story explicitly changes location
- ALWAYS mention the background in EVERY frame description
- Example: If story is about a race → all frames should have "steppe background"
- Example: If story is about a bazaar trick → all frames should have "bazaar background"
- Only change background if story clearly indicates character moves to a different place
```

---

## Expected Behavior

### Before (Inconsistent)
```
Story: "Aldar Kose winning a race with his horse"

Frame 1: aldar_kose_man portrait, steppe
Frame 2: aldar_kose_man riding horse, mountains
Frame 3: aldar_kose_man racing, desert
Frame 4: aldar_kose_man celebrating, village
❌ Background changes every frame!
```

### After (Consistent)
```
Story: "Aldar Kose winning a race with his horse"

Frame 1: aldar_kose_man portrait, steppe background, front-facing
Frame 2: aldar_kose_man riding horse, steppe background
Frame 3: aldar_kose_man racing ahead, steppe background
Frame 4: aldar_kose_man celebrating victory, steppe background
✅ Same background throughout - race happens in ONE location!
```

---

## Common Settings for Aldar Kose Stories

| Story Type | Primary Setting |
|------------|----------------|
| Race/Horse riding | `steppe background` |
| Market tricks | `bazaar background` |
| Home scenes | `yurt setting` |
| Journey | `steppe background` (unless explicitly traveling) |
| Village events | `village background` |
| Mountain adventure | `mountain backdrop` |

---

## When Background SHOULD Change

Background should only change when:
1. Story explicitly mentions traveling ("travels to the mountains")
2. Scene clearly indicates location shift ("enters the yurt", "arrives at bazaar")
3. Multiple distinct locations are core to the plot

Example of valid background change:
```
Story: "Aldar Kose travels from his yurt to the bazaar to trick a merchant"

Frame 1: aldar_kose_man portrait, yurt setting, front-facing
Frame 2: aldar_kose_man leaving yurt, yurt setting
Frame 3: aldar_kose_man riding horse, steppe background (traveling)
Frame 4: aldar_kose_man arriving at bazaar, bazaar background
Frame 5: aldar_kose_man talking to merchant, bazaar background
Frame 6: aldar_kose_man triumphant, bazaar background
```

---

## Impact

✅ **Visual Consistency**: Frames look like they belong to the same story
✅ **Better Flow**: Viewer focus on action, not jarring background changes
✅ **More Professional**: Storyboards feel cohesive and intentional
✅ **Fewer Artifacts**: Consistent backgrounds = more stable generation

---

## Testing

To test the improvements:
```bash
# Test 1: Race story (should have consistent steppe)
python scripts/generate_story.py "Aldar Kose winning a race with his horse" --seed 42 --temp 0.0

# Test 2: Bazaar story (should have consistent bazaar)
python scripts/generate_story.py "Aldar Kose tricks a merchant at the bazaar" --seed 43 --temp 0.0

# Test 3: Journey story (should show clear location transitions)
python scripts/generate_story.py "Aldar Kose travels from his yurt to the mountains" --seed 44 --temp 0.0
```

Check that:
- Single-location stories keep the same background
- Multi-location stories only change background when story indicates it
- Every frame explicitly mentions the background/setting
