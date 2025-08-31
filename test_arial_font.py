#!/usr/bin/env python3
"""
Test Arial font loading specifically.
"""

import os
import sys
from PIL import ImageFont

sys.path.insert(0, os.path.dirname(__file__))

def test_arial_loading():
    """Test Arial font loading."""
    print("🔤 Testing Arial Font Loading")
    print("=" * 40)
    
    # Check for Arial in common locations
    arial_paths = [
        "/System/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/ArialHB.ttc",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux Arial alternative
        "C:\\Windows\\Fonts\\arial.ttf",  # Windows
    ]
    
    print("Searching for Arial font files:")
    found_arial = None
    for path in arial_paths:
        exists = os.path.exists(path)
        print(f"  {path}: {'✅ Found' if exists else '❌ Not found'}")
        if exists and not found_arial:
            found_arial = path
    
    if found_arial:
        print(f"\n✅ Using Arial from: {found_arial}")
        try:
            font = ImageFont.truetype(found_arial, 48)
            bbox = font.getbbox("Test Arial")
            print(f"  Font loaded successfully")
            print(f"  Test bbox: {bbox}")
        except Exception as e:
            print(f"  ❌ Error loading font: {e}")
    else:
        print("\n❌ Arial not found in standard locations")
    
    # Test with RajTextGenerator
    print("\n📝 Testing with RajTextGenerator:")
    from nodes.text_generator import RajTextGenerator
    
    generator = RajTextGenerator()
    
    # Test Arial specifically
    print("\n1. Testing 'Arial' font name:")
    font = generator.get_font("Arial", 48)
    if font:
        print(f"  ✅ Font loaded: {type(font)}")
        test_text = "Arial Test ABC 123"
        bbox = font.getbbox(test_text)
        print(f"  BBox: {bbox}")
        if bbox:
            height = bbox[3] - bbox[1]
            print(f"  Height: {height}px (requested: 48px)")
            if height >= 48 * 0.5:
                print(f"  ✅ Font size is correct")
            else:
                print(f"  ⚠️ Font size seems small")
    else:
        print(f"  ❌ Failed to load Arial")
    
    # Generate actual text with Arial
    print("\n2. Generating text with Arial:")
    try:
        result = generator.generate_text(
            text="Arial Font Test\n1234567890\nABCDEFGHIJKLMNOP",
            output_width=512,
            output_height=256,
            font_name="Arial",
            font_size=36,
            font_color="#FFFFFF",
            background_color="#000000",
            text_align="center",
            vertical_align="middle",
            words_per_line=0,
            max_lines=0,
            line_spacing=1.2,
            letter_spacing=0,
            margin_x=20,
            margin_y=20,
            auto_size=False,
            base_opacity=1.0
        )
        
        if result and result[0] is not None:
            print(f"  ✅ Text generated successfully")
            print(f"  Output shape: {result[0].shape}")
        else:
            print(f"  ❌ Text generation failed")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

def search_for_fonts():
    """Search for all available fonts."""
    print("\n🔍 Searching for all system fonts")
    print("=" * 40)
    
    import platform
    
    system = platform.system()
    if system == "Darwin":  # macOS
        font_dirs = ["/System/Library/Fonts/", "/Library/Fonts/"]
    elif system == "Linux":
        font_dirs = ["/usr/share/fonts/", "/usr/local/share/fonts/"]
    elif system == "Windows":
        font_dirs = ["C:\\Windows\\Fonts\\"]
    else:
        font_dirs = []
    
    font_files = []
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for root, dirs, files in os.walk(font_dir):
                for file in files:
                    if file.endswith(('.ttf', '.ttc', '.otf')):
                        if 'arial' in file.lower():
                            font_files.append(os.path.join(root, file))
    
    print(f"Found {len(font_files)} Arial-related fonts:")
    for font_file in font_files[:10]:  # Show first 10
        print(f"  • {font_file}")
    
    if len(font_files) > 10:
        print(f"  ... and {len(font_files) - 10} more")

if __name__ == "__main__":
    test_arial_loading()
    search_for_fonts()
    
    print("\n✅ Arial font testing complete!")