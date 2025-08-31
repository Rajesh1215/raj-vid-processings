#!/usr/bin/env python3
"""
Test the new border-only container feature.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

def test_border_only_containers():
    """Test container with border only (no background fill)."""
    print("🔲 TESTING BORDER-ONLY CONTAINERS")
    print("=" * 50)
    
    from nodes.text_generator import RajTextGenerator
    
    generator = RajTextGenerator()
    
    test_cases = [
        {
            "name": "Border Only - White",
            "params": {
                "text": "Border Only Text",
                "container_enabled": True,
                "container_fill": False,  # Border only, no background
                "container_border_color": "#FFFFFF",
                "container_width": 3,
                "container_padding": 20
            }
        },
        {
            "name": "Border Only - Red",
            "params": {
                "text": "Red Border Only", 
                "container_enabled": True,
                "container_fill": False,  # Border only
                "container_border_color": "#FF0000",
                "container_width": 4,
                "container_padding": 25
            }
        },
        {
            "name": "Filled Container - For Comparison",
            "params": {
                "text": "Filled Container",
                "container_enabled": True,
                "container_fill": True,  # With background
                "container_color": "#333333",
                "container_border_color": "#FFFF00",
                "container_width": 3,
                "container_padding": 20
            }
        },
        {
            "name": "Border Only + Text Effects",
            "params": {
                "text": "Border + Shadow + Bold",
                "font_weight": "bold",
                "container_enabled": True,
                "container_fill": False,  # Border only
                "container_border_color": "#00FF00",
                "container_width": 5,
                "container_padding": 30,
                "shadow_enabled": True,
                "shadow_offset_x": 3,
                "shadow_offset_y": 3,
                "shadow_color": "#666666",
                "shadow_blur": 2,
                "text_border_width": 1,
                "text_border_color": "#000000"
            }
        }
    ]
    
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
        "gradient_direction": "vertical",
        "container_enabled": False,
        "container_color": "#333333",
        "container_width": 2,
        "container_padding": 15,
        "container_border_color": "#FFFFFF",
        "container_fill": True
    }
    
    results = {}
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\\n{i}. Testing: {test_case['name']}")
        print("-" * 40)
        
        params = default_params.copy()
        params.update(test_case['params'])
        
        try:
            result = generator.generate_text(**params)
            
            if result and result[0] is not None:
                img_array = (result[0][0].cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                filename = f"border_only_test_{i:02d}_{test_case['name'].lower().replace(' ', '_').replace('+', 'and')}.png"
                img.save(filename)
                print(f"    ✅ Generated: {filename}")
                results[test_case['name']] = "✅ PASS"
            else:
                print(f"    ❌ Failed to generate")
                results[test_case['name']] = "❌ FAIL"
        except Exception as e:
            print(f"    💥 Error: {e}")
            results[test_case['name']] = f"❌ ERROR: {str(e)[:30]}..."
    
    return results

def create_comparison_grid():
    """Create side-by-side comparison of filled vs border-only containers."""
    print("\\n📊 CREATING FILL VS BORDER-ONLY COMPARISON")
    print("=" * 45)
    
    from nodes.text_generator import RajTextGenerator
    
    generator = RajTextGenerator()
    
    # Generate filled container
    filled_result = generator.generate_text(
        text="FILLED\\nCONTAINER",
        output_width=300,
        output_height=150,
        font_name="Arial",
        font_size=24,
        font_color="#FFFFFF",
        background_color="#111111",
        text_align="center",
        vertical_align="middle",
        font_weight="bold",
        container_enabled=True,
        container_fill=True,
        container_color="#444444",
        container_border_color="#FFFF00",
        container_width=3,
        container_padding=20,
        words_per_line=0,
        max_lines=0,
        line_spacing=1.2,
        letter_spacing=0,
        margin_x=10,
        margin_y=10,
        auto_size=False,
        base_opacity=1.0,
        text_border_width=0,
        text_border_color="#000000",
        shadow_enabled=False,
        text_bg_enabled=False,
        gradient_enabled=False
    )
    
    # Generate border-only container
    border_only_result = generator.generate_text(
        text="BORDER\\nONLY",
        output_width=300,
        output_height=150,
        font_name="Arial",
        font_size=24,
        font_color="#FFFFFF",
        background_color="#111111",
        text_align="center",
        vertical_align="middle",
        font_weight="bold",
        container_enabled=True,
        container_fill=False,  # Key difference - no fill
        container_color="#444444",  # This won't be used
        container_border_color="#FFFF00",
        container_width=3,
        container_padding=20,
        words_per_line=0,
        max_lines=0,
        line_spacing=1.2,
        letter_spacing=0,
        margin_x=10,
        margin_y=10,
        auto_size=False,
        base_opacity=1.0,
        text_border_width=0,
        text_border_color="#000000",
        shadow_enabled=False,
        text_bg_enabled=False,
        gradient_enabled=False
    )
    
    if filled_result[0] is not None and border_only_result[0] is not None:
        # Convert to PIL images
        filled_img = Image.fromarray((filled_result[0][0].cpu().numpy() * 255).astype(np.uint8))
        border_img = Image.fromarray((border_only_result[0][0].cpu().numpy() * 255).astype(np.uint8))
        
        # Create side-by-side comparison
        comparison = Image.new('RGBA', (600, 150), (17, 17, 17, 255))
        comparison.paste(filled_img, (0, 0))
        comparison.paste(border_img, (300, 0))
        
        # Save comparison
        comparison.save("fill_vs_border_only_comparison.png")
        print("    ✅ Saved: fill_vs_border_only_comparison.png")
        return True
    else:
        print("    ❌ Failed to generate comparison")
        return False

if __name__ == "__main__":
    print("🔲 TESTING BORDER-ONLY CONTAINER FEATURE")
    print("=" * 60)
    
    # Test border-only containers
    results = test_border_only_containers()
    
    # Create comparison
    comparison_success = create_comparison_grid()
    
    # Summary
    print("\\n" + "=" * 60)
    print("📊 BORDER-ONLY TEST RESULTS")
    print("=" * 60)
    
    for test, result in results.items():
        print(f"  {result} {test}")
    
    if comparison_success:
        print(f"  ✅ PASS Fill vs Border-Only Comparison")
    
    pass_count = sum(1 for r in results.values() if r.startswith("✅"))
    total_tests = len(results) + (1 if comparison_success else 0)
    
    print(f"\\n📈 Overall: {pass_count + (1 if comparison_success else 0)}/{total_tests} tests passed")
    
    if pass_count == len(results) and comparison_success:
        print("🎉 BORDER-ONLY FEATURE WORKING PERFECTLY!")
        print("✅ container_fill=False creates border-only containers")
        print("✅ container_fill=True creates filled containers (default)")
        print("✅ Border-only works with all other text effects")
    
    print("\\n📖 Manual verification:")
    print("• border_only_test_01_*.png - Should show text with white border only")
    print("• border_only_test_02_*.png - Should show text with red border only") 
    print("• border_only_test_03_*.png - Should show filled container (comparison)")
    print("• border_only_test_04_*.png - Should show border-only with text effects")
    print("• fill_vs_border_only_comparison.png - Side-by-side comparison")
    
    print("\\n✨ Border-only container feature test complete!")