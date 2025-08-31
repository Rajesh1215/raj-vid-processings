#!/usr/bin/env python3
"""
Comprehensive test for all new text styling features in RajTextGenerator.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

def test_all_styling_features():
    """Test all new text styling features."""
    print("🎨 TESTING ALL TEXT STYLING FEATURES")
    print("=" * 50)
    
    from nodes.text_generator import RajTextGenerator
    
    generator = RajTextGenerator()
    
    # Test cases for different styling features
    test_cases = [
        {
            "name": "Font Weight - Normal",
            "params": {
                "text": "Normal Weight Text",
                "font_weight": "normal",
                "font_size": 36,
                "font_name": "Arial"
            }
        },
        {
            "name": "Font Weight - Bold",
            "params": {
                "text": "Bold Weight Text",
                "font_weight": "bold",
                "font_size": 36,
                "font_name": "Arial"
            }
        },
        {
            "name": "Font Weight - Italic",
            "params": {
                "text": "Italic Weight Text",
                "font_weight": "italic",
                "font_size": 36,
                "font_name": "Arial"
            }
        },
        {
            "name": "Font Weight - Bold Italic",
            "params": {
                "text": "Bold Italic Text",
                "font_weight": "bold_italic",
                "font_size": 36,
                "font_name": "Arial"
            }
        },
        {
            "name": "Text Border - Red",
            "params": {
                "text": "Text with Border",
                "text_border_width": 3,
                "text_border_color": "#FF0000",
                "font_size": 42,
                "font_color": "#FFFFFF"
            }
        },
        {
            "name": "Shadow Effects",
            "params": {
                "text": "Text with Shadow",
                "shadow_enabled": True,
                "shadow_offset_x": 5,
                "shadow_offset_y": 5,
                "shadow_color": "#000000",
                "shadow_blur": 3,
                "font_size": 40,
                "font_color": "#FFFF00"
            }
        },
        {
            "name": "Background Highlight",
            "params": {
                "text": "Highlighted Text",
                "text_bg_enabled": True,
                "text_bg_color": "#FF00FF",
                "text_bg_padding": 10,
                "font_size": 36,
                "font_color": "#FFFFFF"
            }
        },
        {
            "name": "Gradient Text - Vertical",
            "params": {
                "text": "Gradient Vertical",
                "gradient_enabled": True,
                "font_color": "#FF0000",
                "gradient_color2": "#00FF00",
                "gradient_direction": "vertical",
                "font_size": 48
            }
        },
        {
            "name": "Gradient Text - Horizontal",
            "params": {
                "text": "Gradient Horizontal",
                "gradient_enabled": True,
                "font_color": "#0000FF",
                "gradient_color2": "#FFFF00",
                "gradient_direction": "horizontal",
                "font_size": 48
            }
        },
        {
            "name": "Combined Effects",
            "params": {
                "text": "ALL EFFECTS!",
                "font_weight": "bold",
                "text_border_width": 2,
                "text_border_color": "#000000",
                "shadow_enabled": True,
                "shadow_offset_x": 3,
                "shadow_offset_y": 3,
                "shadow_color": "#333333",
                "shadow_blur": 2,
                "text_bg_enabled": True,
                "text_bg_color": "#444444",
                "text_bg_padding": 8,
                "gradient_enabled": True,
                "font_color": "#FF0000",
                "gradient_color2": "#FFFF00",
                "gradient_direction": "diagonal",
                "font_size": 44
            }
        }
    ]
    
    # Default parameters
    default_params = {
        "output_width": 512,
        "output_height": 256,
        "font_name": "Arial",
        "font_size": 36,
        "font_color": "#FFFFFF",
        "background_color": "#000000",
        "text_align": "center",
        "vertical_align": "middle",
        "words_per_line": 0,
        "max_lines": 0,
        "line_spacing": 1.2,
        "letter_spacing": 0,
        "margin_x": 20,
        "margin_y": 20,
        "auto_size": False,
        "base_opacity": 1.0,
        "font_weight": "normal",
        "text_border_width": 0,
        "text_border_color": "#000000",
        "shadow_enabled": False,
        "shadow_offset_x": 2,
        "shadow_offset_y": 2,
        "shadow_color": "#000000",
        "shadow_blur": 2,
        "text_bg_enabled": False,
        "text_bg_color": "#FFFF00",
        "text_bg_padding": 5,
        "gradient_enabled": False,
        "gradient_color2": "#FF0000",
        "gradient_direction": "vertical"
    }
    
    results = {}
    
    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i:2d}. Testing: {test_case['name']}")
        print("-" * 40)
        
        # Merge parameters
        params = default_params.copy()
        params.update(test_case['params'])
        
        try:
            # Generate text
            result = generator.generate_text(**params)
            
            if result and result[0] is not None:
                image_tensor = result[0]
                print(f"    ✅ Generated successfully")
                print(f"    📏 Shape: {image_tensor.shape}")
                
                # Save image for visual inspection
                img_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                filename = f"test_styling_{i:02d}_{test_case['name'].lower().replace(' ', '_').replace('-', '_')}.png"
                img.save(filename)
                print(f"    💾 Saved: {filename}")
                
                results[test_case['name']] = "✅ PASS"
            else:
                print(f"    ❌ Failed to generate text")
                results[test_case['name']] = "❌ FAIL"
                
        except Exception as e:
            print(f"    💥 Error: {e}")
            results[test_case['name']] = f"❌ ERROR: {str(e)[:50]}..."
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*50}")
    
    pass_count = 0
    for test_name, result in results.items():
        print(f"{result} {test_name}")
        if result.startswith("✅"):
            pass_count += 1
    
    total_tests = len(test_cases)
    print(f"\n📈 Overall: {pass_count}/{total_tests} tests passed ({pass_count/total_tests*100:.1f}%)")
    
    if pass_count == total_tests:
        print("🎉 ALL TEXT STYLING FEATURES WORKING PERFECTLY!")
    elif pass_count >= total_tests * 0.8:
        print("✅ Most features working well - minor issues may exist")
    else:
        print("⚠️ Several features need attention")
    
    return results

def test_font_discovery():
    """Test font discovery and weight loading."""
    print("\n🔍 TESTING FONT DISCOVERY")
    print("=" * 30)
    
    from nodes.text_generator import RajTextGenerator
    
    generator = RajTextGenerator()
    
    # Test different fonts and weights
    font_tests = [
        ("Arial", "normal"),
        ("Arial", "bold"),
        ("Arial", "italic"),
        ("Arial", "bold_italic"),
        ("Helvetica", "normal"),
        ("Helvetica", "bold"),
        ("Times", "normal"),
        ("Times", "bold"),
        ("NonExistentFont", "normal")  # Should fallback
    ]
    
    for font_name, weight in font_tests:
        print(f"  Testing: {font_name} - {weight}")
        font = generator.get_font_with_style(font_name, 24, weight)
        if font:
            print(f"    ✅ Loaded successfully")
        else:
            print(f"    ❌ Failed to load")

if __name__ == "__main__":
    # Run font discovery test
    test_font_discovery()
    
    # Run main styling tests
    results = test_all_styling_features()
    
    print(f"\n✨ Text styling test complete!")
    print(f"Check the generated PNG files to visually verify each effect.")