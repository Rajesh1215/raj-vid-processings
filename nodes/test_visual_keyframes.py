#!/usr/bin/env python3

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image, ImageDraw, ImageFont

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Mock logger and utils to avoid import issues
class MockLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")

# Mock utils module 
sys.modules['utils'] = type('MockModule', (), {'logger': MockLogger()})()

# Now import the subtitle engine - use the class name from the file
try:
    from subtitle_engine import RajSubtitleEngine as SubtitleEngine
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import approach...")
    
    # Alternative: directly instantiate the class if import fails
    SubtitleEngine = None

def create_test_output_dir():
    """Create output directory for test images"""
    output_dir = Path("visual_test_output")
    output_dir.mkdir(exist_ok=True)
    return output_dir

def test_keyframe_generation():
    """Generate actual keyframe images for visual analysis"""
    print("Starting visual keyframe generation test...")
    
    # Create output directory
    output_dir = create_test_output_dir()
    
    # User's exact settings
    base_settings = {
        "font_name": "Arial",
        "font_size": 20,
        "color": "#000000",
        "bg_color": "#FFFFFF",
        "alignment": "center",
        "x_position": 0.5,
        "y_position": 0.5,
        "line_spacing": 1.2,
        "margin_x": 10,
        "margin_y": 10
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
    
    # Initialize subtitle engine
    engine = SubtitleEngine()
    
    print("Generating keyframes with settings:")
    print(f"Resolution: 312x304")
    print(f"Base: {base_settings}")
    print(f"Highlight: {highlight_settings}")
    print(f"Word count: {len(word_timings)}")
    
    try:
        # Generate keyframes
        result = engine.generate_subtitle_keyframes(
            word_timings=word_timings,
            base_settings=base_settings,
            highlight_settings=highlight_settings,
            max_lines=2,
            use_line_groups=True,
            use_area_based_grouping=False,
            box_width=312,
            box_height=304
        )
        
        if "keyframes" not in result:
            print("ERROR: No keyframes generated!")
            return
            
        keyframes = result["keyframes"]
        print(f"Generated {len(keyframes)} keyframes")
        
        # Save each keyframe as PNG
        for i, keyframe_data in enumerate(keyframes):
            if "image" in keyframe_data:
                image = keyframe_data["image"]
                timestamp = keyframe_data.get("timestamp", f"frame_{i}")
                
                # Save image
                filename = f"keyframe_{i:03d}_{timestamp:.2f}s.png"
                filepath = output_dir / filename
                image.save(filepath)
                print(f"Saved: {filename}")
                
                # Print keyframe info
                active_words = keyframe_data.get("active_words", [])
                highlighted_word = keyframe_data.get("highlighted_word", "none")
                print(f"  - Active words: {active_words}")
                print(f"  - Highlighted: {highlighted_word}")
                print(f"  - Size: {image.size}")
        
        # Save metadata
        metadata = {
            "settings": {
                "base": base_settings,
                "highlight": highlight_settings,
                "resolution": "312x304",
                "max_lines": 2,
                "use_line_groups": True
            },
            "word_timings": word_timings,
            "keyframe_count": len(keyframes),
            "total_duration": word_timings[-1]["end"] if word_timings else 0
        }
        
        metadata_file = output_dir / "test_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nTest completed successfully!")
        print(f"Images saved to: {output_dir}")
        print(f"Metadata saved to: {metadata_file}")
        
        # Basic quality analysis
        analyze_generated_images(output_dir, keyframes)
        
    except Exception as e:
        print(f"ERROR during keyframe generation: {e}")
        import traceback
        traceback.print_exc()

def analyze_generated_images(output_dir: Path, keyframes: List[Dict]):
    """Perform basic analysis of generated images"""
    print("\n" + "="*50)
    print("VISUAL ANALYSIS REPORT")
    print("="*50)
    
    if not keyframes:
        print("No keyframes to analyze")
        return
    
    # Check image consistency
    sizes = set()
    formats = set()
    
    for i, keyframe_data in enumerate(keyframes):
        if "image" in keyframe_data:
            image = keyframe_data["image"]
            sizes.add(image.size)
            formats.add(image.mode)
    
    print(f"Total keyframes: {len(keyframes)}")
    print(f"Image sizes: {list(sizes)}")
    print(f"Image formats: {list(formats)}")
    
    # Check for expected resolution
    expected_size = (312, 304)
    if expected_size in sizes:
        print("✓ Correct resolution maintained")
    else:
        print(f"✗ Resolution mismatch! Expected {expected_size}, got {list(sizes)}")
    
    # Check highlighting progression
    highlighted_words = []
    for keyframe_data in keyframes:
        highlighted = keyframe_data.get("highlighted_word", "none")
        highlighted_words.append(highlighted)
    
    unique_highlights = set(highlighted_words)
    print(f"Unique highlighted words: {len(unique_highlights)}")
    print(f"Highlighting progression: {highlighted_words}")
    
    # Check for duplicates
    duplicate_count = len(highlighted_words) - len(unique_highlights)
    if duplicate_count == 0:
        print("✓ No duplicate highlighting detected")
    else:
        print(f"⚠ {duplicate_count} potential duplicates found")
    
    print(f"\nAll test images saved to: {output_dir.absolute()}")
    print("Review the generated PNG files for visual quality assessment.")

if __name__ == "__main__":
    test_keyframe_generation()