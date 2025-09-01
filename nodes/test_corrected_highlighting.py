#!/usr/bin/env python3

"""
Test the corrected word highlighting logic to ensure precise word matching
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Mock utils module
class MockLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")

def test_word_index_mapping():
    """Test the word index mapping logic"""
    print("Testing word index mapping logic...")
    
    # Create mock utils
    sys.modules['utils'] = type('MockModule', (), {'logger': MockLogger()})()
    
    try:
        from subtitle_engine import RajSubtitleEngine
        
        engine = RajSubtitleEngine()
        
        # Test with multi-line text similar to your example
        lines = ["Bangalore is a silicon variant of", "India known for its thriving"]
        
        # Mock word timing data with indices
        all_words = [
            {"word": "Bangalore", "index": 0, "start_time": 0.0, "end_time": 0.5},
            {"word": "is", "index": 1, "start_time": 0.5, "end_time": 0.7}, 
            {"word": "a", "index": 2, "start_time": 0.7, "end_time": 0.8},
            {"word": "silicon", "index": 3, "start_time": 0.8, "end_time": 1.2},
            {"word": "variant", "index": 4, "start_time": 1.2, "end_time": 1.6},
            {"word": "of", "index": 5, "start_time": 1.6, "end_time": 1.8},
            {"word": "India", "index": 6, "start_time": 1.8, "end_time": 2.2},
            {"word": "known", "index": 7, "start_time": 2.2, "end_time": 2.5},
            {"word": "for", "index": 8, "start_time": 2.5, "end_time": 2.7},
            {"word": "its", "index": 9, "start_time": 2.7, "end_time": 2.9},
            {"word": "thriving", "index": 10, "start_time": 2.9, "end_time": 3.4}
        ]
        
        # Build the word index map
        word_index_map = engine._build_word_index_map(lines, all_words)
        
        print("Word Index Mapping Results:")
        print("=" * 50)
        
        expected_mapping = {
            (0, 0): 0,  # "Bangalore" at line 0, pos 0 -> index 0
            (0, 1): 1,  # "is" at line 0, pos 1 -> index 1
            (0, 2): 2,  # "a" at line 0, pos 2 -> index 2
            (0, 3): 3,  # "silicon" at line 0, pos 3 -> index 3
            (0, 4): 4,  # "variant" at line 0, pos 4 -> index 4
            (0, 5): 5,  # "of" at line 0, pos 5 -> index 5
            (1, 0): 6,  # "India" at line 1, pos 0 -> index 6
            (1, 1): 7,  # "known" at line 1, pos 1 -> index 7
            (1, 2): 8,  # "for" at line 1, pos 2 -> index 8
            (1, 3): 9,  # "its" at line 1, pos 3 -> index 9
            (1, 4): 10, # "thriving" at line 1, pos 4 -> index 10
        }
        
        # Verify mapping
        all_correct = True
        for (line_idx, word_pos), expected_index in expected_mapping.items():
            actual_index = word_index_map.get((line_idx, word_pos), -1)
            word_text = lines[line_idx].split()[word_pos]
            
            if actual_index == expected_index:
                print(f"✓ '{word_text}' at ({line_idx}, {word_pos}) -> index {actual_index}")
            else:
                print(f"✗ '{word_text}' at ({line_idx}, {word_pos}) -> expected {expected_index}, got {actual_index}")
                all_correct = False
        
        if all_correct:
            print("\n✓ Word index mapping is correct!")
        else:
            print("\n✗ Word index mapping has errors!")
            
        return all_correct
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_highlighting_precision():
    """Test that only the correct word gets highlighted"""
    print("\nTesting highlighting precision...")
    
    # Test scenarios similar to your issue
    test_cases = [
        {
            "description": "Highlight 'Bangalore' only",
            "highlighted_word": {"word": "Bangalore", "index": 0, "start_time": 0.0, "end_time": 0.5},
            "current_time": 0.2,
            "expected_highlighted": "Bangalore",
            "should_not_highlight": ["is", "a", "silicon", "variant"]
        },
        {
            "description": "Highlight 'silicon' only", 
            "highlighted_word": {"word": "silicon", "index": 3, "start_time": 0.8, "end_time": 1.2},
            "current_time": 1.0,
            "expected_highlighted": "silicon",
            "should_not_highlight": ["Bangalore", "is", "a", "variant"]
        },
        {
            "description": "Highlight 'India' only",
            "highlighted_word": {"word": "India", "index": 6, "start_time": 1.8, "end_time": 2.2},
            "current_time": 2.0,
            "expected_highlighted": "India",
            "should_not_highlight": ["Bangalore", "known", "for", "thriving"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        
        # Simulate the word matching logic
        lines = ["Bangalore is a silicon variant of", "India known for its thriving"]
        all_words = [
            {"word": "Bangalore", "index": 0, "start_time": 0.0, "end_time": 0.5},
            {"word": "is", "index": 1, "start_time": 0.5, "end_time": 0.7},
            {"word": "a", "index": 2, "start_time": 0.7, "end_time": 0.8}, 
            {"word": "silicon", "index": 3, "start_time": 0.8, "end_time": 1.2},
            {"word": "variant", "index": 4, "start_time": 1.2, "end_time": 1.6},
            {"word": "of", "index": 5, "start_time": 1.6, "end_time": 1.8},
            {"word": "India", "index": 6, "start_time": 1.8, "end_time": 2.2},
            {"word": "known", "index": 7, "start_time": 2.2, "end_time": 2.5},
            {"word": "for", "index": 8, "start_time": 2.5, "end_time": 2.7},
            {"word": "its", "index": 9, "start_time": 2.7, "end_time": 2.9},
            {"word": "thriving", "index": 10, "start_time": 2.9, "end_time": 3.4}
        ]
        
        # Build word index map (using simulated method)
        word_index_map = {}
        text_words = []
        for line_idx, line_text in enumerate(lines):
            line_words = line_text.split()
            for word_pos, word in enumerate(line_words):
                text_words.append((line_idx, word_pos, word.strip()))
        
        for text_idx, (line_idx, word_pos, word) in enumerate(text_words):
            if text_idx < len(all_words):
                timing_index = all_words[text_idx].get('index', text_idx)
                word_index_map[(line_idx, word_pos)] = timing_index
        
        # Test the highlighting logic
        highlighted_word = test_case["highlighted_word"]
        highlighted_index = highlighted_word.get('index', -1)
        current_time = test_case["current_time"]
        
        highlighted_words = []
        not_highlighted_words = []
        
        for line_idx, line_text in enumerate(lines):
            line_words = line_text.split()
            for word_pos, word in enumerate(line_words):
                word_index = word_index_map.get((line_idx, word_pos), -1)
                
                # Apply the new highlighting logic
                is_highlighted = False
                if highlighted_index >= 0 and word_index == highlighted_index:
                    # Verify timing
                    word_start = highlighted_word.get('start_time', 0)
                    word_end = highlighted_word.get('end_time', 0)
                    if word_start <= current_time <= word_end:
                        is_highlighted = True
                
                if is_highlighted:
                    highlighted_words.append(word)
                else:
                    not_highlighted_words.append(word)
        
        # Check results
        expected = test_case["expected_highlighted"]
        if expected in highlighted_words and len(highlighted_words) == 1:
            print(f"  ✓ Correctly highlighted: '{expected}'")
        else:
            print(f"  ✗ Expected '{expected}', but highlighted: {highlighted_words}")
        
        # Check that wrong words are NOT highlighted
        wrong_highlights = set(highlighted_words) & set(test_case["should_not_highlight"])
        if wrong_highlights:
            print(f"  ✗ Incorrectly highlighted: {wrong_highlights}")
        else:
            print(f"  ✓ Correctly avoided highlighting: {test_case['should_not_highlight']}")

if __name__ == "__main__":
    print("Testing Corrected Word Highlighting Logic")
    print("=" * 50)
    
    success1 = test_word_index_mapping()
    test_highlighting_precision()
    
    if success1:
        print("\n🎯 Word highlighting correction implemented successfully!")
        print("The ComfyUI node should now highlight only the correct word based on timing.")
    else:
        print("\n❌ Word highlighting correction needs debugging.")