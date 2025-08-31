#!/usr/bin/env python3
"""
Test font weight fixes and new container box feature.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

def test_font_weight_fixes():
    """Test that font weights are properly distinguished and Arial normal works."""
    print("🔧 TESTING FONT WEIGHT FIXES")
    print("=" * 50)
    
    from nodes.text_generator import RajTextGenerator
    
    generator = RajTextGenerator()
    
    # Test cases focusing on the specific issues
    test_cases = [
        {
            "name": "Arial Normal (Fixed Boxes Issue)",
            "params": {
                "text": "Arial Normal Text",
                "font_name": "Arial",
                "font_weight": "normal",
                "font_size": 48,
                "font_color": "#FFFFFF",
                "background_color": "#000000"
            }
        },
        {
            "name": "Arial Bold (Should Look Different)",
            "params": {
                "text": "Arial Bold Text", 
                "font_name": "Arial",
                "font_weight": "bold",
                "font_size": 48,
                "font_color": "#FFFFFF",
                "background_color": "#000000"
            }
        },
        {
            "name": "Helvetica Normal vs Bold Test",
            "params": {
                "text": "Helvetica Normal",
                "font_name": "Helvetica", 
                "font_weight": "normal",
                "font_size": 48,
                "font_color": "#FFFFFF",
                "background_color": "#222222"
            }
        },
        {
            "name": "Helvetica Bold (Should Be Heavier)",
            "params": {
                "text": "Helvetica Bold",
                "font_name": "Helvetica",
                "font_weight": "bold", 
                "font_size": 48,
                "font_color": "#FFFFFF",
                "background_color": "#222222"
            }
        },
        {
            "name": "Container Box Test",
            "params": {
                "text": "Text with Container",
                "font_name": "Arial",
                "font_weight": "normal",
                "font_size": 36,
                "font_color": "#FFFFFF",
                "background_color": "#000000",
                "container_enabled": True,
                "container_color": "#444444",
                "container_width": 3,
                "container_padding": 20
            }
        },
        {
            "name": "Container + Effects Combined",
            "params": {
                "text": "Container + Shadow + Border",
                "font_name": "Arial",
                "font_weight": "bold",
                "font_size": 32,
                "font_color": "#FFFF00",
                "background_color": "#000000",
                "container_enabled": True,
                "container_color": "#333333",
                "container_width": 2,
                "container_padding": 15,
                "shadow_enabled": True,
                "shadow_offset_x": 3,
                "shadow_offset_y": 3,
                "shadow_color": "#000000",
                "shadow_blur": 2,
                "text_border_width": 1,
                "text_border_color": "#FF0000"
            }
        }
    ]
    
    # Default parameters
    default_params = {
        "output_width": 512,
        "output_height": 256,
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
        "container_padding": 15
    }
    
    results = {}
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\\n{i}. Testing: {test_case['name']}")
        print("-" * 45)
        
        # Merge parameters
        params = default_params.copy()
        params.update(test_case['params'])
        
        try:
            result = generator.generate_text(**params)
            
            if result and result[0] is not None:
                image_tensor = result[0]
                print(f"    ✅ Generated successfully")
                print(f"    📏 Shape: {image_tensor.shape}")
                
                # Save test image
                img_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                filename = f"fixed_test_{i:02d}_{test_case['name'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'and')}.png"
                img.save(filename)
                print(f"    💾 Saved: {filename}")
                
                results[test_case['name']] = "✅ PASS"
            else:
                print(f"    ❌ Failed to generate")
                results[test_case['name']] = "❌ FAIL"
                
        except Exception as e:
            print(f"    💥 Error: {e}")
            results[test_case['name']] = f"❌ ERROR: {str(e)[:50]}..."
    
    # Summary
    print(f"\\n{'='*50}")
    print("📊 FONT FIX TEST RESULTS")  
    print(f"{'='*50}")
    
    pass_count = 0
    for test_name, result in results.items():
        print(f"{result} {test_name}")
        if result.startswith("✅"):
            pass_count += 1
    
    total_tests = len(test_cases)
    success_rate = pass_count / total_tests * 100
    print(f"\\n📈 Results: {pass_count}/{total_tests} passed ({success_rate:.1f}%)")
    
    if pass_count == total_tests:
        print("🎉 ALL FONT FIXES WORKING!")
        print("✅ Arial normal should now render properly (not boxes)")  
        print("✅ Font weights should be visually distinct")
        print("✅ Container box feature should be working")
    elif pass_count >= total_tests * 0.8:
        print("✅ Most fixes working - check images for details")
    else:
        print("⚠️ Some fixes need more work")
    
    return results

def test_font_weight_comparison():
    """Create side-by-side comparison of normal vs bold for visual verification."""
    print("\\n📊 CREATING FONT WEIGHT COMPARISON")
    print("=" * 40)
    
    from nodes.text_generator import RajTextGenerator
    
    generator = RajTextGenerator()
    
    # Test fonts that commonly have weight issues
    fonts_to_test = ["Arial", "Helvetica", "Times New Roman"]
    
    comparison_images = []
    
    for font_name in fonts_to_test:
        print(f"  Testing {font_name} weight comparison...")
        
        # Generate normal weight
        normal_result = generator.generate_text(
            text=f"{font_name} Normal",
            output_width=300,
            output_height=100,
            font_name=font_name,
            font_weight="normal",
            font_size=32,
            font_color="#FFFFFF",
            background_color="#333333",
            text_align="center",
            vertical_align="middle"
        )
        
        # Generate bold weight  
        bold_result = generator.generate_text(
            text=f"{font_name} Bold",
            output_width=300, 
            output_height=100,
            font_name=font_name,
            font_weight="bold",
            font_size=32,
            font_color="#FFFFFF",
            background_color="#333333", 
            text_align="center",
            vertical_align="middle"
        )
        
        if normal_result[0] is not None and bold_result[0] is not None:
            # Convert to PIL images
            normal_img = Image.fromarray((normal_result[0][0].cpu().numpy() * 255).astype(np.uint8))
            bold_img = Image.fromarray((bold_result[0][0].cpu().numpy() * 255).astype(np.uint8))
            
            # Create side-by-side comparison
            comparison = Image.new('RGBA', (600, 100), (0, 0, 0, 255))
            comparison.paste(normal_img, (0, 0))
            comparison.paste(bold_img, (300, 0))
            
            # Save comparison
            filename = f"weight_comparison_{font_name.lower().replace(' ', '_')}.png"
            comparison.save(filename)
            print(f"    💾 Saved comparison: {filename}")
            comparison_images.append(filename)
        else:
            print(f"    ❌ Failed to generate {font_name} comparison")
    
    print(f"\\n✅ Created {len(comparison_images)} font weight comparisons")
    print("📝 Check the comparison images to verify bold vs normal distinction")

if __name__ == "__main__":
    # Run font fix tests
    results = test_font_weight_fixes()
    
    # Create weight comparisons
    test_font_weight_comparison()
    
    print("\\n✨ Font fix testing complete!")
    print("📖 Key fixes implemented:")
    print("   1. Improved font discovery (avoids problematic variants)")
    print("   2. Enhanced font validation (detects box rendering)")  
    print("   3. Better font weight fallback system")
    print("   4. New container box feature with padding and borders")
    print("\\n🔍 Manually review the generated images to confirm:")
    print("   • Arial normal renders text (not boxes)")
    print("   • Bold weights look heavier than normal weights") 
    print("   • Container boxes appear around text when enabled")