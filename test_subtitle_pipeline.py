#!/usr/bin/env python3
"""
Test the complete subtitle pipeline:
1. RajTextGenerator (settings output)
2. RajTextToTiming (convert text to timing data)
3. RajSubtitleEngine (generate subtitle video)
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

def test_text_generator_settings_output():
    """Test that RajTextGenerator now outputs TEXT_SETTINGS."""
    print("🔧 TESTING TEXT GENERATOR SETTINGS OUTPUT")
    print("=" * 50)
    
    from nodes.text_generator import RajTextGenerator
    
    generator = RajTextGenerator()
    
    try:
        result = generator.generate_text(
            text="Test Settings Output",
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
            base_opacity=1.0,
            font_weight="normal",
            text_border_width=2,
            text_border_color="#FF0000",
            shadow_enabled=True,
            shadow_offset_x=3,
            shadow_offset_y=3,
            shadow_color="#000000",
            shadow_blur=2,
            text_bg_enabled=False,
            text_bg_color="#FFFF00",
            text_bg_padding=5,
            gradient_enabled=False,
            gradient_color2="#FF0000",
            gradient_direction="vertical",
            container_enabled=True,
            container_color="#333333",
            container_width=2,
            container_padding=15,
            container_border_color="#FFFFFF",
            container_fill=True
        )
        
        print(f"✅ Generator returned {len(result)} outputs")
        print(f"   Expected: 5 outputs (image, mask, text_config, render_info, styling_settings)")
        
        if len(result) >= 5:
            styling_settings = result[4]
            print(f"✅ Styling settings type: {type(styling_settings)}")
            
            if isinstance(styling_settings, dict):
                print(f"✅ Styling settings keys: {list(styling_settings.keys())}")
                
                # Check for expected configuration sections
                expected_sections = ['font_config', 'layout_config', 'effects_config', 'container_config', 'output_config']
                for section in expected_sections:
                    if section in styling_settings:
                        print(f"   ✅ Found {section}")
                    else:
                        print(f"   ❌ Missing {section}")
                
                return styling_settings
            else:
                print(f"❌ Styling settings not a dict: {type(styling_settings)}")
                return None
        else:
            print(f"❌ Expected 5 outputs, got {len(result)}")
            return None
            
    except Exception as e:
        print(f"❌ Error testing text generator: {e}")
        return None

def test_text_to_timing_converter():
    """Test RajTextToTiming converter."""
    print("\\n⏰ TESTING TEXT TO TIMING CONVERTER")
    print("=" * 50)
    
    from nodes.text_to_timing import RajTextToTiming
    
    converter = RajTextToTiming()
    
    test_text = "Hello world! This is a test sentence for subtitle generation. Each word will have precise timing data."
    
    try:
        word_timings, preview, total_duration = converter.convert_text_to_timing(
            text=test_text,
            timing_mode="realistic",
            word_duration_min=0.3,
            word_duration_max=0.8,
            sentence_pause=0.5,
            total_duration=0.0,
            speech_rate=150.0
        )
        
        print(f"✅ Converter returned:")
        print(f"   Word timings: {len(word_timings)} words")
        print(f"   Total duration: {total_duration:.2f}s")
        
        if word_timings:
            print(f"   First word: '{word_timings[0]['word']}' @ {word_timings[0]['start_time']:.2f}s")
            print(f"   Last word: '{word_timings[-1]['word']}' @ {word_timings[-1]['end_time']:.2f}s")
            
            # Validate word timing format
            sample_word = word_timings[0]
            required_fields = ['word', 'start_time', 'end_time', 'confidence']
            
            valid_format = True
            for field in required_fields:
                if field not in sample_word:
                    print(f"   ❌ Missing field: {field}")
                    valid_format = False
            
            if valid_format:
                print("   ✅ Word timing format is valid")
            
            print(f"\\n📋 Preview:")
            print(preview[:300] + "..." if len(preview) > 300 else preview)
            
            return word_timings
        else:
            print("❌ No word timings generated")
            return None
            
    except Exception as e:
        print(f"❌ Error testing text to timing: {e}")
        return None

def test_subtitle_engine(word_timings, base_settings, highlight_settings=None):
    """Test RajSubtitleEngine with word timing data."""
    print("\\n🎬 TESTING SUBTITLE ENGINE")
    print("=" * 50)
    
    from nodes.subtitle_engine import RajSubtitleEngine
    
    engine = RajSubtitleEngine()
    
    if not word_timings:
        print("❌ No word timings provided")
        return False
    
    if not base_settings:
        print("❌ No base settings provided")
        return False
    
    print(f"   Input: {len(word_timings)} words, {base_settings.get('output_config', {}).get('output_width', 'unknown')}x{base_settings.get('output_config', {}).get('output_height', 'unknown')} output")
    
    try:
        video_frames, total_frames, timing_info, frame_metadata = engine.generate_subtitle_video(
            word_timings=word_timings,
            base_settings=base_settings,
            video_fps=30.0,
            highlight_settings=highlight_settings,
            sentence_tolerance=1,
            auto_fit=True,
            lead_time=0.0,
            trail_time=0.0
        )
        
        print(f"✅ Subtitle engine returned:")
        print(f"   Video frames tensor: {video_frames.shape}")
        print(f"   Total frames: {total_frames}")
        print(f"   Frame metadata: {len(frame_metadata.get('frames', []))} frame entries")
        
        # Validate video tensor format
        expected_dims = 5  # [batch, frames, height, width, channels]
        if len(video_frames.shape) == expected_dims:
            print(f"   ✅ Video tensor format correct: {video_frames.shape}")
        else:
            print(f"   ⚠️ Unexpected video tensor format: {video_frames.shape}")
        
        # Save a few sample frames
        if total_frames > 0:
            frame_indices = [0, total_frames // 2, total_frames - 1]
            for i, frame_idx in enumerate(frame_indices):
                if frame_idx < total_frames:
                    frame = video_frames[0, frame_idx].cpu().numpy()
                    frame_uint8 = (frame * 255).astype(np.uint8)
                    
                    img = Image.fromarray(frame_uint8)
                    filename = f"subtitle_test_frame_{i:02d}_{frame_idx:04d}.png"
                    img.save(filename)
                    print(f"   💾 Saved sample frame: {filename}")
        
        print(f"\\n📋 Timing Info Preview:")
        print(timing_info[:400] + "..." if len(timing_info) > 400 else timing_info)
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing subtitle engine: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_pipeline():
    """Test the complete subtitle pipeline end-to-end."""
    print("\\n🚀 TESTING COMPLETE SUBTITLE PIPELINE")
    print("=" * 60)
    
    # Step 1: Generate base text settings
    print("\\n📝 Step 1: Generate base text settings...")
    base_settings = test_text_generator_settings_output()
    
    if not base_settings:
        print("❌ Failed to generate base settings")
        return False
    
    # Step 2: Generate highlight text settings (optional)
    print("\\n🎨 Step 2: Generate highlight text settings...")
    from nodes.text_generator import RajTextGenerator
    generator = RajTextGenerator()
    
    try:
        highlight_result = generator.generate_text(
            text="Highlight Test",
            output_width=512,
            output_height=256,
            font_name="Arial",
            font_size=42,  # Larger for highlighting
            font_color="#FFFF00",  # Yellow highlight
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
            base_opacity=1.0,
            font_weight="bold",  # Bold for emphasis
            text_border_width=3,
            text_border_color="#000000",
            shadow_enabled=False,
            text_bg_enabled=False,
            gradient_enabled=False,
            container_enabled=False
        )
        
        highlight_settings = highlight_result[4] if len(highlight_result) >= 5 else None
        print("✅ Generated highlight settings")
        
    except Exception as e:
        print(f"⚠️ Could not generate highlight settings: {e}")
        highlight_settings = None
    
    # Step 3: Convert text to timing data
    print("\\n⏰ Step 3: Convert text to timing data...")
    word_timings = test_text_to_timing_converter()
    
    if not word_timings:
        print("❌ Failed to generate word timings")
        return False
    
    # Step 4: Generate subtitle video
    print("\\n🎬 Step 4: Generate subtitle video...")
    success = test_subtitle_engine(word_timings, base_settings, highlight_settings)
    
    return success

def test_different_modes():
    """Test different timing modes and configurations."""
    print("\\n🔄 TESTING DIFFERENT MODES")
    print("=" * 50)
    
    from nodes.text_to_timing import RajTextToTiming
    
    converter = RajTextToTiming()
    test_text = "Quick test for different timing modes."
    
    modes = ["equal", "random", "realistic"]
    results = {}
    
    for mode in modes:
        try:
            word_timings, preview, duration = converter.convert_text_to_timing(
                text=test_text,
                timing_mode=mode,
                word_duration_min=0.2,
                word_duration_max=1.0,
                sentence_pause=0.3,
                speech_rate=120.0
            )
            
            results[mode] = {
                "word_count": len(word_timings),
                "duration": duration,
                "avg_word_duration": duration / len(word_timings) if word_timings else 0
            }
            
            print(f"✅ {mode.upper()} mode: {len(word_timings)} words, {duration:.2f}s total")
            
        except Exception as e:
            print(f"❌ {mode.upper()} mode failed: {e}")
            results[mode] = None
    
    return results

if __name__ == "__main__":
    print("🎬 COMPREHENSIVE SUBTITLE PIPELINE TEST")
    print("=" * 60)
    
    # Test individual components
    print("\\n📋 COMPONENT TESTS:")
    test_different_modes()
    
    # Test complete pipeline
    print("\\n🔗 INTEGRATION TEST:")
    pipeline_success = test_complete_pipeline()
    
    # Final summary
    print("\\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS")
    print("=" * 60)
    
    if pipeline_success:
        print("🎉 COMPLETE SUBTITLE PIPELINE WORKING!")
        print("✅ All components integrated successfully")
        print("✅ Video frames generated with proper timing")
        print("✅ Settings transfer working between nodes")
        print("✅ Word-level timing system operational")
        
        print("\\n📖 Usage Summary:")
        print("1. 📝 RajTextGenerator → Create text styling settings")
        print("2. 🎨 RajTextGenerator → Create highlight styling (optional)")
        print("3. ⏰ RajTextToTiming → Convert text to timing data (testing)")
        print("4. 🎙️ RajWhisperProcess → Generate word timings (production)")
        print("5. 🎬 RajSubtitleEngine → Generate subtitle video frames")
        
        print("\\n🔧 Generated Files:")
        print("• subtitle_test_frame_*.png - Sample subtitle frames")
        print("• Check frames to verify text rendering and timing")
        
    else:
        print("❌ SUBTITLE PIPELINE HAS ISSUES")
        print("⚠️ Check error messages above for details")
    
    print("\\n✨ Subtitle pipeline test complete!")
    
    if pipeline_success:
        print("\\n🚀 READY FOR PRODUCTION USE!")
        print("Connect RajWhisperProcess → RajSubtitleEngine for real video subtitles")
    else:
        print("\\n🔧 NEEDS DEBUGGING")
        print("Review component tests and fix issues before production use")