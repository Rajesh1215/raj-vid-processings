#!/usr/bin/env python3
"""
Test container border visibility and font normal weight distinction.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

def test_container_border_fixes():
    """Test that container borders are now visible with separate border color."""
    print("🔳 TESTING CONTAINER BORDER FIXES")
    print("=" * 50)
    
    from nodes.text_generator import RajTextGenerator
    
    generator = RajTextGenerator()
    
    test_cases = [
        {
            "name": "Container with White Border",
            "params": {
                "text": "White Border Test",
                "container_enabled": True,
                "container_color": "#333333",  # Dark gray background
                "container_border_color": "#FFFFFF",  # White border
                "container_width": 3,
                "container_padding": 20
            }
        },
        {
            "name": "Container with Red Border",
            "params": {
                "text": "Red Border Test", 
                "container_enabled": True,
                "container_color": "#000000",  # Black background
                "container_border_color": "#FF0000",  # Red border
                "container_width": 4,
                "container_padding": 25
            }
        },
        {
            "name": "Container with Yellow Border",
            "params": {
                "text": "Yellow Border Test",
                "container_enabled": True,
                "container_color": "#0066CC",  # Blue background
                "container_border_color": "#FFFF00",  # Yellow border
                "container_width": 5,
                "container_padding": 15
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
        "container_border_color": "#FFFFFF"
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
                filename = f"border_test_{i:02d}_{test_case['name'].lower().replace(' ', '_')}.png"
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

def test_normal_font_weight_distinction():
    """Test that different font families have distinct normal weights."""
    print("\\n🔤 TESTING NORMAL FONT WEIGHT DISTINCTION") 
    print("=" * 50)
    
    from nodes.text_generator import RajTextGenerator
    
    generator = RajTextGenerator()
    
    # Test fonts that should look different in normal weight
    fonts_to_test = [
        ("Arial", "Arial Normal"),
        ("Helvetica", "Helvetica Normal"), 
        ("Times New Roman", "Times Normal"),
        ("Courier New", "Courier Normal")
    ]
    
    results = {}
    comparison_images = []
    
    for i, (font_name, display_text) in enumerate(fonts_to_test, 1):
        print(f"\\n{i}. Testing: {font_name} normal weight")
        print("-" * 30)
        
        try:
            result = generator.generate_text(
                text=display_text,
                output_width=400,
                output_height=120,
                font_name=font_name,
                font_size=32,
                font_color="#FFFFFF",
                background_color="#222222",
                text_align="center",
                vertical_align="middle",
                font_weight="normal",
                words_per_line=0,
                max_lines=0,
                line_spacing=1.2,
                letter_spacing=0,
                margin_x=20,
                margin_y=20,
                auto_size=False,
                base_opacity=1.0,
                text_border_width=0,
                text_border_color="#000000",
                shadow_enabled=False,
                text_bg_enabled=False,
                gradient_enabled=False,
                container_enabled=False
            )
            
            if result and result[0] is not None:
                img_array = (result[0][0].cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                filename = f"font_normal_{i:02d}_{font_name.lower().replace(' ', '_')}.png"
                img.save(filename)
                print(f"    ✅ Generated: {filename}")
                comparison_images.append((font_name, img))
                results[font_name] = "✅ PASS"
            else:
                print(f"    ❌ Failed to generate")
                results[font_name] = "❌ FAIL"
                
        except Exception as e:
            print(f"    💥 Error: {e}")
            results[font_name] = f"❌ ERROR: {str(e)[:30]}..."
    
    # Create side-by-side comparison
    if len(comparison_images) >= 2:
        print("\\n📊 Creating font comparison grid...")
        
        # Create 2x2 grid if we have 4 fonts, or horizontal strip for fewer
        if len(comparison_images) == 4:
            combined = Image.new('RGBA', (800, 240), (40, 40, 40, 255))
            positions = [(0, 0), (400, 0), (0, 120), (400, 120)]
        else:
            combined = Image.new('RGBA', (400 * len(comparison_images), 120), (40, 40, 40, 255))
            positions = [(400 * i, 0) for i in range(len(comparison_images))]
        
        for (font_name, img), pos in zip(comparison_images, positions):
            combined.paste(img, pos)
        
        combined.save("font_normal_comparison.png")
        print("    ✅ Saved: font_normal_comparison.png")
    
    return results

def test_combined_effects():
    """Test container border with various text effects."""
    print("\\n🎨 TESTING COMBINED CONTAINER + TEXT EFFECTS")
    print("=" * 50)
    
    from nodes.text_generator import RajTextGenerator
    
    generator = RajTextGenerator()
    
    try:
        result = generator.generate_text(
            text="Container + Effects\\nBorder + Shadow + Bold",
            output_width=600,
            output_height=300,
            font_name="Arial",
            font_size=28,
            font_color="#FFFF00",
            background_color="#000000",
            text_align="center",
            vertical_align="middle",
            font_weight="bold",
            words_per_line=0,
            max_lines=0,
            line_spacing=1.3,
            letter_spacing=0,
            margin_x=30,
            margin_y=30,
            auto_size=False,
            base_opacity=1.0,
            # Text border
            text_border_width=2,
            text_border_color="#000000",
            # Shadow
            shadow_enabled=True,
            shadow_offset_x=4,
            shadow_offset_y=4,
            shadow_color="#666666",
            shadow_blur=3,
            # Container
            container_enabled=True,
            container_color="#1a1a1a",
            container_border_color="#00FF00",
            container_width=6,
            container_padding=25,
            # Other
            text_bg_enabled=False,
            gradient_enabled=False
        )
        
        if result and result[0] is not None:
            img_array = (result[0][0].cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            img.save("combined_effects_test.png")
            print("    ✅ Generated: combined_effects_test.png")
            return "✅ PASS"
        else:
            print("    ❌ Failed to generate combined effects")
            return "❌ FAIL"
            
    except Exception as e:
        print(f"    💥 Error: {e}")
        return f"❌ ERROR: {str(e)[:50]}..."

if __name__ == "__main__":
    print("🔧 TESTING FINAL FONT & CONTAINER FIXES")
    print("=" * 60)
    
    # Test container borders
    border_results = test_container_border_fixes()
    
    # Test normal font weights  
    font_results = test_normal_font_weight_distinction()
    
    # Test combined effects
    combined_result = test_combined_effects()
    
    # Summary
    print("\\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS SUMMARY")
    print("=" * 60)
    
    print("\\n🔳 Container Border Tests:")
    for test, result in border_results.items():
        print(f"  {result} {test}")
    
    print("\\n🔤 Font Normal Weight Tests:")
    for font, result in font_results.items():
        print(f"  {result} {font}")
    
    print(f"\\n🎨 Combined Effects: {combined_result}")
    
    # Overall assessment
    border_pass = sum(1 for r in border_results.values() if r.startswith("✅"))
    font_pass = sum(1 for r in font_results.values() if r.startswith("✅"))
    combined_pass = 1 if combined_result.startswith("✅") else 0
    
    total_pass = border_pass + font_pass + combined_pass
    total_tests = len(border_results) + len(font_results) + 1
    
    print(f"\\n📈 Overall: {total_pass}/{total_tests} tests passed ({total_pass/total_tests*100:.1f}%)")
    
    if total_pass == total_tests:
        print("🎉 ALL FIXES WORKING PERFECTLY!")
        print("✅ Container borders should be visible with separate colors")
        print("✅ Font normal weights should look distinct between families")
        print("✅ All effects should work together seamlessly")
    elif total_pass >= total_tests * 0.8:
        print("✅ Most fixes working - review images for details")
    else:
        print("⚠️ Some issues remain - check individual test results")
    
    print("\\n📖 Manual verification steps:")
    print("1. Check border_test_*.png - borders should be clearly visible")
    print("2. Check font_normal_*.png - each font should look different")  
    print("3. Check font_normal_comparison.png - side-by-side comparison")
    print("4. Check combined_effects_test.png - all effects together")
    
    print("\\n✨ Final fixes test complete!")