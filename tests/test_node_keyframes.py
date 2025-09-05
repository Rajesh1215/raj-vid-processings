#!/usr/bin/env python3

"""
Test the node's keyframe generation process to ensure highlighting works
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Mock utils module to avoid import issues
class MockLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")

def test_node_keyframe_generation():
    """Test that the node's keyframe generation includes highlighting"""
    print("Testing node keyframe generation with highlighting...")
    
    # Create mock utils before importing subtitle engine
    sys.modules['utils'] = type('MockModule', (), {'logger': MockLogger()})()
    
    try:
        # Import the keyframe generation method from the subtitle engine
        from subtitle_engine import RajSubtitleEngine
        
        # Create instance
        engine = RajSubtitleEngine()
        
        # Test settings (same as user's specifications)
        base_settings = {
            "font_config": {
                "font_name": "Arial",
                "font_size": 20,
                "color": "#000000"
            },
            "layout_config": {
                "alignment": "center",
                "x_position": 0.5,
                "y_position": 0.5,
                "line_spacing": 1.2,
                "margin_x": 10,
                "margin_y": 10
            },
            "output_config": {
                "output_width": 312,
                "output_height": 304,
                "bg_color": "#FFFFFF"
            }
        }
        
        highlight_settings = {
            "font_config": {
                "font_name": "Arial",
                "font_size": 20,
                "color": "#0000FF"  # Blue
            }
        }
        
        word_timings = [
            {"word": "Hello", "start": 0.0, "end": 0.46},
            {"word": "world", "start": 0.46, "end": 0.92},
            {"word": "this", "start": 0.92, "end": 1.38}
        ]
        
        print(f"Testing with {len(word_timings)} words")
        print("Calling generate_subtitle_keyframes...")
        
        # Call the keyframe generation method
        result = engine.generate_subtitle_keyframes(
            word_timings=word_timings,
            base_settings=base_settings,
            highlight_settings=highlight_settings,
            max_lines=2,
            use_line_groups=False,
            use_area_based_grouping=False,
            box_width=312,
            box_height=304
        )
        
        if "keyframes" in result:
            keyframes = result["keyframes"]
            print(f"✓ Generated {len(keyframes)} keyframes")
            
            # Check first few keyframes for highlighting
            for i, keyframe in enumerate(keyframes[:5]):
                timestamp = keyframe.get("timestamp", 0)
                highlighted = keyframe.get("highlighted_word", "None")
                active_words = keyframe.get("active_words", [])
                print(f"  Keyframe {i}: {timestamp:.2f}s -> highlighted='{highlighted}', active={len(active_words)} words")
                
                # Save first keyframe as test
                if i == 0 and "image" in keyframe:
                    output_dir = Path("node_test_output")
                    output_dir.mkdir(exist_ok=True)
                    keyframe["image"].save(output_dir / f"node_keyframe_{i}.png")
                    print(f"    Saved test image: {output_dir}/node_keyframe_{i}.png")
            
            print("✓ Node keyframe generation test successful")
        else:
            print("✗ No keyframes generated")
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_highlighting_function_directly():
    """Test the highlighting function directly with node's data"""
    print("\nTesting highlighting function with node data structure...")
    
    sys.modules['utils'] = type('MockModule', (), {'logger': MockLogger()})()
    
    from subtitle_utils import get_current_highlighted_word, parse_whisper_word_data
    
    # Raw word data (as would come from whisper)
    raw_word_data = [
        {"word": "Hello", "start": 0.0, "end": 0.46},
        {"word": "world", "start": 0.46, "end": 0.92},
        {"word": "this", "start": 0.92, "end": 1.38}
    ]
    
    # Parse it like the node would
    parsed_data = parse_whisper_word_data(raw_word_data)
    print(f"Parsed {len(parsed_data)} words")
    
    # Test highlighting at specific times
    test_times = [0.1, 0.5, 1.0]
    for test_time in test_times:
        highlighted = get_current_highlighted_word(parsed_data, test_time)
        word = highlighted.get('word', 'None') if highlighted else 'None'
        print(f"  Time {test_time}s: highlighted word = '{word}'")
    
    print("✓ Direct highlighting test successful")

if __name__ == "__main__":
    test_highlighting_function_directly()
    test_node_keyframe_generation()
    print("\n🎯 Node highlighting tests completed!")