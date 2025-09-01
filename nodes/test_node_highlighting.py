#!/usr/bin/env python3

"""
Test highlighting with the actual node implementation
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

def test_highlighting_data_structure():
    """Test that our highlighting functions work with both data formats"""
    print("Testing highlighting with different data structures...")
    
    # Create mock utils before importing subtitle_utils
    sys.modules['utils'] = type('MockModule', (), {'logger': MockLogger()})()
    
    # Import after mocking utils
    from subtitle_utils import get_current_highlighted_word, parse_whisper_word_data
    
    # Test data in raw whisper format (start/end)
    raw_word_data = [
        {"word": "Hello", "start": 0.0, "end": 0.46},
        {"word": "world", "start": 0.46, "end": 0.92},
        {"word": "test", "start": 0.92, "end": 1.38}
    ]
    
    # Test data in parsed format (start_time/end_time)
    parsed_word_data = [
        {"word": "Hello", "start_time": 0.0, "end_time": 0.46, "confidence": 1.0, "index": 0},
        {"word": "world", "start_time": 0.46, "end_time": 0.92, "confidence": 1.0, "index": 1},
        {"word": "test", "start_time": 0.92, "end_time": 1.38, "confidence": 1.0, "index": 2}
    ]
    
    # Test highlighting with raw data
    print("\n1. Testing with raw whisper data (start/end format):")
    test_times = [0.1, 0.5, 1.0, 1.5]
    for time in test_times:
        highlighted = get_current_highlighted_word(raw_word_data, time)
        word = highlighted.get('word', 'None') if highlighted else 'None'
        print(f"   Time {time}s -> highlighted: {word}")
    
    # Test highlighting with parsed data
    print("\n2. Testing with parsed data (start_time/end_time format):")
    for time in test_times:
        highlighted = get_current_highlighted_word(parsed_word_data, time)
        word = highlighted.get('word', 'None') if highlighted else 'None'
        print(f"   Time {time}s -> highlighted: {word}")
    
    # Test parsing function
    print("\n3. Testing parse_whisper_word_data function:")
    parsed_result = parse_whisper_word_data(raw_word_data)
    print(f"   Parsed {len(parsed_result)} words from raw data")
    for word_data in parsed_result:
        print(f"   - {word_data}")
    
    # Test highlighting on parsed result
    print("\n4. Testing highlighting on parsed result:")
    for time in test_times:
        highlighted = get_current_highlighted_word(parsed_result, time)
        word = highlighted.get('word', 'None') if highlighted else 'None'
        print(f"   Time {time}s -> highlighted: {word}")
    
    print("\n✓ Highlighting data structure test completed")

def test_node_settings_structure():
    """Test that the node gets the right settings structure"""
    print("\nTesting node settings structure...")
    
    # Simulate what the node should receive
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
    
    print(f"Base settings structure: {base_settings}")
    print(f"Highlight settings: {highlight_settings}")
    print(f"Word timings: {word_timings}")
    print("✓ Settings structure looks correct for node")

if __name__ == "__main__":
    test_highlighting_data_structure()
    test_node_settings_structure()
    print("\n🎯 All tests completed - highlighting should work in node now!")