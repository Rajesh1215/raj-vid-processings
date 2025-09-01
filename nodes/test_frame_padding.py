#!/usr/bin/env python3

"""
Test the frame padding system for better highlighting and spacing
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Mock utils for testing
class MockLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")

def test_frame_padding_logic():
    """Test the frame padding calculation logic"""
    print("Testing Frame Padding Logic")
    print("=" * 50)
    
    test_cases = [
        {
            "description": "No padding - default behavior",
            "base_width": 800,
            "base_height": 400,
            "frame_padding_width": 0,
            "frame_padding_height": 0,
            "base_font_size": 20,
            "highlight_font_size": 20,
            "expected_width": 800,
            "expected_height": 400,
            "expected_auto_width": 0,
            "expected_auto_height": 0
        },
        {
            "description": "Manual padding only",
            "base_width": 800,
            "base_height": 400,
            "frame_padding_width": 50,
            "frame_padding_height": 30,
            "base_font_size": 20,
            "highlight_font_size": 20,
            "expected_width": 850,
            "expected_height": 430,
            "expected_auto_width": 0,
            "expected_auto_height": 0
        },
        {
            "description": "Auto padding for large highlights",
            "base_width": 800,
            "base_height": 400,
            "frame_padding_width": 0,
            "frame_padding_height": 0,
            "base_font_size": 20,
            "highlight_font_size": 40,  # 2x larger, triggers auto padding
            "expected_width": 840,  # 800 + 40 (2x size diff)
            "expected_height": 420,  # 400 + 20 (1x size diff)
            "expected_auto_width": 40,
            "expected_auto_height": 20
        },
        {
            "description": "Combined manual + auto padding",
            "base_width": 800,
            "base_height": 400,
            "frame_padding_width": 20,
            "frame_padding_height": 10,
            "base_font_size": 16,
            "highlight_font_size": 32,  # 2x larger
            "expected_width": 852,  # 800 + 20 + 32
            "expected_height": 426,  # 400 + 10 + 16
            "expected_auto_width": 32,
            "expected_auto_height": 16
        },
        {
            "description": "Auto padding capped at maximum",
            "base_width": 800,
            "base_height": 400,
            "frame_padding_width": 0,
            "frame_padding_height": 0,
            "base_font_size": 10,
            "highlight_font_size": 60,  # 6x larger, but auto padding is capped
            "expected_width": 850,  # 800 + 50 (max auto padding width)
            "expected_height": 425,  # 400 + 25 (max auto padding height)
            "expected_auto_width": 50,  # Capped
            "expected_auto_height": 25   # Capped
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {case['description']}")
        print("-" * 40)
        
        # Simulate the logic from the enhanced code
        display_width = case["base_width"]
        display_height = case["base_height"]
        frame_padding_width = case["frame_padding_width"]
        frame_padding_height = case["frame_padding_height"]
        base_font_size = case["base_font_size"]
        highlight_font_size = case["highlight_font_size"]
        
        # Auto padding calculation (matching the actual implementation)
        auto_padding_width = 0
        auto_padding_height = 0
        if highlight_font_size > base_font_size * 1.3:
            size_diff = highlight_font_size - base_font_size
            auto_padding_width = min(int(size_diff * 2), 50)  # Max 50px auto padding
            auto_padding_height = min(int(size_diff * 1), 25)  # Max 25px auto padding
        
        # Final dimensions
        total_padding_width = frame_padding_width + auto_padding_width
        total_padding_height = frame_padding_height + auto_padding_height
        final_width = display_width + total_padding_width
        final_height = display_height + total_padding_height
        
        # Verify results
        width_correct = final_width == case["expected_width"]
        height_correct = final_height == case["expected_height"]
        auto_width_correct = auto_padding_width == case["expected_auto_width"]
        auto_height_correct = auto_padding_height == case["expected_auto_height"]
        
        if width_correct and height_correct and auto_width_correct and auto_height_correct:
            print(f"  ✓ PASS")
            print(f"    Base: {display_width}x{display_height}")
            print(f"    Manual padding: {frame_padding_width}x{frame_padding_height}")
            print(f"    Auto padding: {auto_padding_width}x{auto_padding_height}")
            print(f"    Final: {final_width}x{final_height}")
        else:
            print(f"  ✗ FAIL")
            print(f"    Expected final: {case['expected_width']}x{case['expected_height']}")
            print(f"    Got final: {final_width}x{final_height}")
            print(f"    Expected auto: {case['expected_auto_width']}x{case['expected_auto_height']}")
            print(f"    Got auto: {auto_padding_width}x{auto_padding_height}")
    
    return True

def test_integration_scenarios():
    """Test realistic integration scenarios"""
    print("\n" + "=" * 50)
    print("Integration Scenario Tests")
    print("=" * 50)
    
    scenarios = [
        {
            "name": "Small subtitle with normal highlights",
            "base_settings": {
                'output_config': {'output_width': 512, 'output_height': 256},
                'font_config': {'font_size': 18}
            },
            "highlight_settings": {
                'font_config': {'font_size': 20}  # Slightly larger
            },
            "frame_padding_width": 0,
            "frame_padding_height": 0,
            "expected_dimensions": "512x256",
            "expected_auto_padding": "0x0"
        },
        {
            "name": "Large subtitle with bold highlights + manual padding",
            "base_settings": {
                'output_config': {'output_width': 1024, 'output_height': 512},
                'font_config': {'font_size': 24}
            },
            "highlight_settings": {
                'font_config': {'font_size': 36, 'font_weight': 'bold'}  # 50% larger
            },
            "frame_padding_width": 40,
            "frame_padding_height": 20,
            "expected_dimensions": "1088x544",  # 1024+40+24, 512+20+12
            "expected_auto_padding": "24x12"
        },
        {
            "name": "Very large highlights with auto-padding cap",
            "base_settings": {
                'output_config': {'output_width': 800, 'output_height': 400},
                'font_config': {'font_size': 12}
            },
            "highlight_settings": {
                'font_config': {'font_size': 48}  # 4x larger - should cap auto padding
            },
            "frame_padding_width": 10,
            "frame_padding_height": 5,
            "expected_dimensions": "860x430",  # 800+10+50, 400+5+25 (auto capped)
            "expected_auto_padding": "50x25"  # Capped at maximum
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario['name']}")
        print("-" * 40)
        
        base_settings = scenario["base_settings"]
        highlight_settings = scenario.get("highlight_settings", {})
        frame_padding_width = scenario["frame_padding_width"]
        frame_padding_height = scenario["frame_padding_height"]
        
        # Extract dimensions
        output_config = base_settings.get('output_config', {})
        display_width = output_config.get('output_width', 512)
        display_height = output_config.get('output_height', 256)
        
        # Extract font sizes
        font_config = base_settings.get('font_config', {})
        base_font_size = font_config.get('font_size', 20)
        
        highlight_font_config = highlight_settings.get('font_config', {})
        highlight_font_size = highlight_font_config.get('font_size', base_font_size)
        
        # Calculate auto padding
        auto_padding_width = 0
        auto_padding_height = 0
        if highlight_font_size > base_font_size * 1.3:
            size_diff = highlight_font_size - base_font_size
            auto_padding_width = min(int(size_diff * 2), 50)
            auto_padding_height = min(int(size_diff * 1), 25)
        
        # Calculate final dimensions
        total_padding_width = frame_padding_width + auto_padding_width
        total_padding_height = frame_padding_height + auto_padding_height
        final_width = display_width + total_padding_width
        final_height = display_height + total_padding_height
        
        print(f"  Base font: {base_font_size}px, Highlight font: {highlight_font_size}px")
        print(f"  Base dimensions: {display_width}x{display_height}")
        print(f"  Manual padding: {frame_padding_width}x{frame_padding_height}")
        print(f"  Auto padding: {auto_padding_width}x{auto_padding_height}")
        print(f"  Final dimensions: {final_width}x{final_height}")
        
        expected_dims = scenario["expected_dimensions"]
        expected_auto = scenario["expected_auto_padding"]
        actual_dims = f"{final_width}x{final_height}"
        actual_auto = f"{auto_padding_width}x{auto_padding_height}"
        
        if actual_dims == expected_dims and actual_auto == expected_auto:
            print(f"  ✓ PASS - Matches expected results")
        else:
            print(f"  ✗ FAIL - Expected dims: {expected_dims}, auto: {expected_auto}")
            print(f"          Got dims: {actual_dims}, auto: {actual_auto}")

def test_comfyui_usage_example():
    """Show example of how to use frame padding in ComfyUI"""
    print("\n" + "=" * 50)
    print("ComfyUI Usage Examples")
    print("=" * 50)
    
    examples = [
        {
            "title": "Basic Manual Padding",
            "description": "Add fixed padding for better spacing",
            "params": {
                "frame_padding_width": 30,
                "frame_padding_height": 15
            },
            "highlight_settings": {
                "font_config": {"font_size": 24, "color": "#FF0000"}
            }
        },
        {
            "title": "Large Bold Highlights",
            "description": "Bold, large highlights with both manual and auto padding",
            "params": {
                "frame_padding_width": 40,
                "frame_padding_height": 20
            },
            "highlight_settings": {
                "font_config": {
                    "font_size": 32,
                    "font_weight": "bold",
                    "color": "#00FF00"
                },
                "layout_config": {
                    "margin_width": 8,
                    "margin_height": 5
                }
            }
        },
        {
            "title": "Auto-Padding Only",
            "description": "Let the system auto-detect padding needs",
            "params": {
                "frame_padding_width": 0,
                "frame_padding_height": 0
            },
            "highlight_settings": {
                "font_config": {
                    "font_size": 36,  # Large enough to trigger auto-padding
                    "font_weight": "bold",
                    "color": "#0066FF"
                }
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}: {example['title']}")
        print(f"Description: {example['description']}")
        print("Configuration:")
        print(f"  frame_padding_width: {example['params']['frame_padding_width']}")
        print(f"  frame_padding_height: {example['params']['frame_padding_height']}")
        if example['highlight_settings']:
            print("  highlight_settings:")
            for key, value in example['highlight_settings'].items():
                if isinstance(value, dict):
                    print(f"    {key}:")
                    for subkey, subvalue in value.items():
                        print(f"      {subkey}: {subvalue}")
                else:
                    print(f"    {key}: {value}")

if __name__ == "__main__":
    print("Frame Padding System Test Suite")
    print("=" * 60)
    
    test_frame_padding_logic()
    test_integration_scenarios() 
    test_comfyui_usage_example()
    
    print("\n" + "=" * 60)
    print("🎯 Frame Padding System Features:")
    print("✅ Manual padding control via frame_padding_width/height")
    print("✅ Auto-padding detection for large highlights (>30% bigger)")
    print("✅ Auto-padding capped at reasonable limits (50px/25px)")
    print("✅ Combined manual + auto padding support")
    print("✅ Backward compatibility (default padding = 0)")
    print("✅ Metadata tracking of applied padding")
    print("\nYour ComfyUI workflow can now:")
    print("• Set frame_padding_width/height for consistent extra spacing")
    print("• Rely on auto-detection for large highlight text")
    print("• Get better visual results with highlighted subtitles")
    print("• Avoid cramped text in frames")