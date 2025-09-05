#!/usr/bin/env python3

"""
Direct keyframe generation test - bypasses complex imports
Generates actual subtitle images for visual analysis
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
import sys
import os

def get_system_font(font_name: str = "Arial", font_size: int = 20) -> ImageFont.FreeTypeFont:
    """Get system font or fallback to default"""
    try:
        # Try common font paths
        font_paths = [
            f"/System/Library/Fonts/{font_name}.ttf",  # macOS
            f"/usr/share/fonts/truetype/dejavu/DejaVu{font_name}.ttf",  # Linux
            f"C:/Windows/Fonts/{font_name.lower()}.ttf",  # Windows
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                return ImageFont.truetype(path, font_size)
        
        # Fallback to PIL default
        return ImageFont.load_default()
    except:
        return ImageFont.load_default()

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_line_groups(words: List[Dict], max_lines: int = 2) -> List[List[List[Dict]]]:
    """Create line groups where each word group becomes a line"""
    if not words:
        return []
    
    # Simple implementation: group words by line capacity
    words_per_line = max(1, len(words) // max_lines)
    
    line_groups = []
    current_group = []
    current_line = []
    
    for i, word in enumerate(words):
        current_line.append(word)
        
        # If line is full or end of words
        if len(current_line) >= words_per_line or i == len(words) - 1:
            current_group.append(current_line)
            current_line = []
            
            # If group has enough lines
            if len(current_group) >= max_lines or i == len(words) - 1:
                line_groups.append(current_group)
                current_group = []
    
    return line_groups

def get_highlighted_word_at_time(words: List[Dict], timestamp: float) -> str:
    """Find which word should be highlighted at given timestamp"""
    for word_data in words:
        if word_data["start"] <= timestamp <= word_data["end"]:
            return word_data["word"]
    return ""

def render_subtitle_keyframe(
    words: List[Dict], 
    timestamp: float,
    width: int = 312, 
    height: int = 304,
    base_settings: Dict = None,
    highlight_settings: Dict = None
) -> Image.Image:
    """Render a single subtitle keyframe"""
    
    # Default settings
    if base_settings is None:
        base_settings = {
            "font_name": "Arial",
            "font_size": 20,
            "color": "#000000",
            "bg_color": "#FFFFFF"
        }
    
    if highlight_settings is None:
        highlight_settings = {
            "color": "#0000FF",
            "font_size": 20,
            "font_name": "Arial"
        }
    
    # Create image
    bg_color = hex_to_rgb(base_settings["bg_color"])
    image = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(image)
    
    # Get fonts
    base_font = get_system_font(base_settings["font_name"], base_settings["font_size"])
    highlight_font = get_system_font(highlight_settings["font_name"], highlight_settings["font_size"])
    
    # Find highlighted word
    highlighted_word = get_highlighted_word_at_time(words, timestamp)
    
    # Create text lines
    text_lines = []
    current_line = []
    
    for word_data in words:
        current_line.append(word_data["word"])
        # Simple line breaking - 6 words per line
        if len(current_line) >= 6:
            text_lines.append(" ".join(current_line))
            current_line = []
    
    if current_line:
        text_lines.append(" ".join(current_line))
    
    # Render each line
    y_offset = height // 4
    line_height = base_settings["font_size"] + 5
    
    base_color = hex_to_rgb(base_settings["color"])
    highlight_color = hex_to_rgb(highlight_settings["color"])
    
    for line_text in text_lines:
        words_in_line = line_text.split()
        x_offset = 10
        
        for word in words_in_line:
            # Choose color and font
            if word == highlighted_word:
                color = highlight_color
                font = highlight_font
            else:
                color = base_color
                font = base_font
            
            # Draw word
            draw.text((x_offset, y_offset), word, font=font, fill=color)
            
            # Update x position
            word_width = draw.textbbox((0, 0), word + " ", font=font)[2]
            x_offset += word_width
        
        y_offset += line_height
    
    return image

def generate_test_keyframes():
    """Generate test keyframes for visual analysis"""
    print("Generating direct keyframe test images...")
    
    # Create output directory
    output_dir = Path("visual_test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Test settings (user's exact specifications)
    base_settings = {
        "font_name": "Arial",
        "font_size": 20,
        "color": "#000000",
        "bg_color": "#FFFFFF"
    }
    
    highlight_settings = {
        "color": "#0000FF",
        "font_size": 20,
        "font_name": "Arial"
    }
    
    word_timings = [
        {"word": "Hello", "start": 0.0, "end": 0.46},
        {"word": "world", "start": 0.46, "end": 0.92},
        {"word": "this", "start": 0.92, "end": 1.38},
        {"word": "is", "start": 1.38, "end": 1.84},
        {"word": "a", "start": 1.84, "end": 2.3},
        {"word": "test", "start": 2.3, "end": 2.76},
        {"word": "of", "start": 2.76, "end": 3.22},
        {"word": "the", "start": 3.22, "end": 3.68},
        {"word": "subtitle", "start": 3.68, "end": 4.14},
        {"word": "generation", "start": 4.14, "end": 4.6},
        {"word": "system", "start": 4.6, "end": 5.06},
        {"word": "with", "start": 5.06, "end": 5.52},
        {"word": "word", "start": 5.52, "end": 5.98},
        {"word": "highlighting", "start": 5.98, "end": 6.44},
        {"word": "and", "start": 6.44, "end": 6.9},
        {"word": "multiple", "start": 6.9, "end": 7.36},
        {"word": "line", "start": 7.36, "end": 7.82},
        {"word": "support", "start": 7.82, "end": 8.26}
    ]
    
    print(f"Settings: {base_settings}")
    print(f"Highlight: {highlight_settings}")
    print(f"Word count: {len(word_timings)}")
    
    # Generate keyframes at word transition points
    keyframe_times = []
    for word_data in word_timings:
        keyframe_times.append(word_data["start"])
        keyframe_times.append((word_data["start"] + word_data["end"]) / 2)  # Mid-word
    
    # Remove duplicates and sort
    keyframe_times = sorted(set(keyframe_times))
    
    generated_images = []
    
    for i, timestamp in enumerate(keyframe_times):
        print(f"Generating keyframe {i+1}/{len(keyframe_times)} at {timestamp:.2f}s")
        
        # Render keyframe
        image = render_subtitle_keyframe(
            word_timings, 
            timestamp,
            312, 304,
            base_settings,
            highlight_settings
        )
        
        # Find highlighted word
        highlighted = get_highlighted_word_at_time(word_timings, timestamp)
        
        # Save image
        filename = f"keyframe_{i:03d}_{timestamp:.2f}s.png"
        filepath = output_dir / filename
        image.save(filepath)
        
        generated_images.append({
            "timestamp": timestamp,
            "filename": filename,
            "highlighted_word": highlighted,
            "image_size": image.size
        })
        
        print(f"  Saved: {filename} (highlighted: '{highlighted}')")
    
    # Save metadata
    metadata = {
        "test_info": {
            "resolution": "312x304",
            "total_keyframes": len(generated_images),
            "settings": {"base": base_settings, "highlight": highlight_settings}
        },
        "word_timings": word_timings,
        "generated_keyframes": generated_images
    }
    
    metadata_file = output_dir / "keyframe_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Generated {len(generated_images)} keyframe images")
    print(f"✓ Saved to: {output_dir.absolute()}")
    print(f"✓ Metadata: {metadata_file}")
    
    # Basic analysis
    analyze_keyframes(generated_images)
    
    return generated_images

def analyze_keyframes(generated_images: List[Dict]):
    """Analyze generated keyframe quality"""
    print("\n" + "="*50)
    print("KEYFRAME ANALYSIS REPORT")
    print("="*50)
    
    if not generated_images:
        print("No keyframes to analyze")
        return
    
    # Check consistency
    sizes = set(img["image_size"] for img in generated_images)
    highlighted_words = [img["highlighted_word"] for img in generated_images]
    unique_highlights = set(highlighted_words)
    
    print(f"Total keyframes: {len(generated_images)}")
    print(f"Image sizes: {list(sizes)}")
    print(f"Unique highlighted words: {len(unique_highlights)}")
    print(f"Highlighting sequence: {highlighted_words}")
    
    # Check for expected resolution
    if (312, 304) in sizes:
        print("✓ Correct resolution (312x304) maintained")
    else:
        print(f"✗ Resolution issue! Expected (312,304), got {list(sizes)}")
    
    # Check highlighting progression
    empty_highlights = sum(1 for h in highlighted_words if not h)
    if empty_highlights == 0:
        print("✓ All keyframes have word highlighting")
    else:
        print(f"⚠ {empty_highlights} keyframes without highlighting")
    
    # Timing analysis
    timestamps = [img["timestamp"] for img in generated_images]
    duration = max(timestamps) - min(timestamps)
    print(f"Duration covered: {duration:.2f} seconds")
    print(f"Average keyframe interval: {duration/len(timestamps):.2f}s")
    
    print("\n✓ Analysis complete - check generated PNG files for visual quality")

if __name__ == "__main__":
    generate_test_keyframes()