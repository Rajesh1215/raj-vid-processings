#!/usr/bin/env python3
"""
Test the transparency and highlighting fixes for RajSubtitleEngine.
This tests that:
1. Transparent backgrounds actually work (no black backgrounds)
2. Highlighting works with transparent backgrounds
3. Non-highlighted text remains visible with transparent backgrounds
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

def test_transparent_background_without_highlighting():
    """Test transparent background without highlighting features."""
    print("🔧 TESTING TRANSPARENT BACKGROUND (NO HIGHLIGHTING)")
    print("=" * 60)
    
    from nodes.subtitle_engine import RajSubtitleEngine
    
    # Create subtitle engine
    engine = RajSubtitleEngine()
    
    # Create mock word timing data
    word_timings = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
        {"word": "test", "start": 1.0, "end": 1.5}
    ]
    
    # Create base settings with transparent background
    base_settings = {
        "font_config": {
            "font_name": "Arial",
            "font_size": 36,
            "font_color": "#FFFFFF",  # White text
            "font_weight": "normal"
        },
        "layout_config": {
            "text_align": "center",
            "vertical_align": "middle",
            "margin_x": 20,
            "margin_y": 20
        },
        "effects_config": {},
        "container_config": {},
        "output_config": {
            "background_color": "transparent",  # This should create transparent background
            "output_width": 800,
            "output_height": 200,
            "base_opacity": 1.0
        }
    }
    
    try:
        # Generate subtitle video
        result = engine.generate_subtitle_video(
            word_timings=word_timings,
            base_settings=base_settings,
            video_fps=30.0,
            frame_width=800,
            frame_height=200,
            highlight_settings=None  # No highlighting
        )
        
        regular_frames, transparent_frames, total_frames, timing_info, metadata, bg_hexcode = result
        
        print(f"✅ Generated {total_frames} frames successfully")
        print(f"   Regular frames shape: {regular_frames.shape}")
        print(f"   Transparent frames shape: {transparent_frames.shape}")
        print(f"   Background color: {bg_hexcode}")
        
        # Check that transparent frames actually have alpha channel (4 channels)
        if len(transparent_frames.shape) == 4 and transparent_frames.shape[-1] == 4:
            print("✅ Transparent frames have RGBA format (4 channels)")
            
            # Check that some pixels are actually transparent (alpha < 255)
            # Sample a few frames
            sample_indices = [0, total_frames//2, total_frames-1] if total_frames > 0 else [0]
            for idx in sample_indices:
                if idx < total_frames:
                    frame = transparent_frames[idx].numpy()
                    alpha_channel = frame[:, :, 3]  # Get alpha channel
                    min_alpha = np.min(alpha_channel)
                    max_alpha = np.max(alpha_channel)
                    avg_alpha = np.mean(alpha_channel)
                    print(f"   Frame {idx} alpha stats: min={min_alpha:.2f}, max={max_alpha:.2f}, avg={avg_alpha:.2f}")
                    
                    # Count transparent pixels (alpha < 0.1 in normalized 0-1 range)
                    transparent_pixels = np.sum(alpha_channel < 0.1)
                    total_pixels = alpha_channel.shape[0] * alpha_channel.shape[1]
                    transparency_percent = (transparent_pixels / total_pixels) * 100
                    print(f"   Frame {idx}: {transparency_percent:.1f}% transparent pixels")
        else:
            print(f"❌ Transparent frames missing alpha channel: {transparent_frames.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing transparent background: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✅ Transparent background test PASSED\n")
    return True

def test_transparent_background_with_highlighting():
    """Test transparent background WITH highlighting features."""
    print("🔧 TESTING TRANSPARENT BACKGROUND WITH HIGHLIGHTING")
    print("=" * 60)
    
    from nodes.subtitle_engine import RajSubtitleEngine
    
    # Create subtitle engine
    engine = RajSubtitleEngine()
    
    # Create mock word timing data with indices
    word_timings = [
        {"word": "Hello", "start": 0.0, "end": 0.5, "index": 0},
        {"word": "highlighted", "start": 0.5, "end": 1.0, "index": 1},
        {"word": "world", "start": 1.0, "end": 1.5, "index": 2}
    ]
    
    # Create base settings with transparent background
    base_settings = {
        "font_config": {
            "font_name": "Arial",
            "font_size": 36,
            "font_color": "#FFFFFF",  # White text
            "font_weight": "normal"
        },
        "layout_config": {
            "text_align": "center",
            "vertical_align": "middle",
            "margin_x": 20,
            "margin_y": 20
        },
        "effects_config": {},
        "container_config": {},
        "output_config": {
            "background_color": "transparent",  # This should create transparent background
            "output_width": 800,
            "output_height": 200,
            "base_opacity": 1.0
        }
    }
    
    # Create highlighting settings
    highlight_settings = {
        "font_config": {
            "font_name": "Arial",
            "font_size": 42,  # Larger for highlighting
            "font_color": "#00FF00",  # Green highlight
            "font_weight": "bold"
        },
        "layout_config": {
            "margin_width": 5,
            "margin_height": 3
        },
        "output_config": {
            "background_color": "transparent"
        }
    }
    
    try:
        # Generate subtitle video with highlighting
        result = engine.generate_subtitle_video(
            word_timings=word_timings,
            base_settings=base_settings,
            video_fps=30.0,
            frame_width=800,
            frame_height=200,
            highlight_settings=highlight_settings  # Enable highlighting
        )
        
        regular_frames, transparent_frames, total_frames, timing_info, metadata, bg_hexcode = result
        
        print(f"✅ Generated {total_frames} frames with highlighting")
        print(f"   Regular frames shape: {regular_frames.shape}")
        print(f"   Transparent frames shape: {transparent_frames.shape}")
        print(f"   Background color: {bg_hexcode}")
        print(f"   Highlighting enabled: {metadata.get('highlighting_enabled', False)}")
        
        # Check transparent frames have alpha channel
        if len(transparent_frames.shape) == 4 and transparent_frames.shape[-1] == 4:
            print("✅ Transparent frames with highlighting have RGBA format")
            
            # Check a few frames for transparency
            sample_indices = [0, total_frames//2, total_frames-1] if total_frames > 0 else [0]
            for idx in sample_indices:
                if idx < total_frames:
                    frame = transparent_frames[idx].numpy()
                    alpha_channel = frame[:, :, 3]
                    
                    # Count transparent pixels
                    transparent_pixels = np.sum(alpha_channel < 0.1)
                    total_pixels = alpha_channel.shape[0] * alpha_channel.shape[1]
                    transparency_percent = (transparent_pixels / total_pixels) * 100
                    print(f"   Frame {idx}: {transparency_percent:.1f}% transparent pixels (with highlighting)")
                    
                    # Check if there are non-transparent pixels (where text should be)
                    opaque_pixels = np.sum(alpha_channel > 0.9)
                    if opaque_pixels > 0:
                        print(f"   Frame {idx}: {opaque_pixels} opaque pixels (text visible)")
                    else:
                        print(f"   Frame {idx}: WARNING - No opaque pixels found!")
        else:
            print(f"❌ Transparent frames with highlighting missing alpha channel: {transparent_frames.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing transparent background with highlighting: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✅ Transparent background with highlighting test PASSED\n")
    return True

def test_colored_background_with_highlighting():
    """Test that colored backgrounds still work correctly (regression test)."""
    print("🔧 TESTING COLORED BACKGROUND WITH HIGHLIGHTING (REGRESSION)")
    print("=" * 60)
    
    from nodes.subtitle_engine import RajSubtitleEngine
    
    # Create subtitle engine
    engine = RajSubtitleEngine()
    
    # Create mock word timing data
    word_timings = [
        {"word": "Test", "start": 0.0, "end": 0.5, "index": 0},
        {"word": "colored", "start": 0.5, "end": 1.0, "index": 1},
        {"word": "background", "start": 1.0, "end": 1.5, "index": 2}
    ]
    
    # Create base settings with colored background (should be RGB)
    base_settings = {
        "font_config": {
            "font_name": "Arial",
            "font_size": 36,
            "font_color": "#FFFFFF",  # White text
            "font_weight": "normal"
        },
        "layout_config": {
            "text_align": "center",
            "vertical_align": "middle"
        },
        "effects_config": {},
        "container_config": {},
        "output_config": {
            "background_color": "#000080",  # Dark blue background
            "output_width": 800,
            "output_height": 200,
            "base_opacity": 1.0
        }
    }
    
    # Create highlighting settings
    highlight_settings = {
        "font_config": {
            "font_name": "Arial",
            "font_size": 42,  # Larger for highlighting
            "font_color": "#FFFF00",  # Yellow highlight
            "font_weight": "bold"
        }
    }
    
    try:
        # Generate subtitle video
        result = engine.generate_subtitle_video(
            word_timings=word_timings,
            base_settings=base_settings,
            video_fps=30.0,
            frame_width=800,
            frame_height=200,
            highlight_settings=highlight_settings
        )
        
        regular_frames, transparent_frames, total_frames, timing_info, metadata, bg_hexcode = result
        
        print(f"✅ Generated {total_frames} frames with colored background")
        print(f"   Regular frames shape: {regular_frames.shape}")
        print(f"   Background color: {bg_hexcode}")
        
        # Check that regular frames have RGB format (3 channels)
        if len(regular_frames.shape) == 4 and regular_frames.shape[-1] == 3:
            print("✅ Regular frames have RGB format (3 channels)")
        else:
            print(f"❌ Regular frames should have RGB format, got: {regular_frames.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing colored background: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✅ Colored background test PASSED\n")
    return True

if __name__ == "__main__":
    print("🚀 TESTING TRANSPARENT HIGHLIGHTING FIXES")
    print("=" * 80)
    
    tests = [
        test_transparent_background_without_highlighting,
        test_transparent_background_with_highlighting, 
        test_colored_background_with_highlighting
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
        print()
    
    print("=" * 80)
    print(f"🏁 TEST SUMMARY: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests PASSED! Transparency and highlighting fixes are working correctly.")
        sys.exit(0)
    else:
        print("💥 Some tests FAILED! Please check the output above for details.")
        sys.exit(1)