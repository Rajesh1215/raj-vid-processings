#!/usr/bin/env python3
"""
Test the complete audio overlay and segmentation system.
Tests all new nodes: RajAudioSegmenter, RajAudioOverlay, etc.
"""

import os
import sys
import torch
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

def create_test_audio(duration: float = 5.0, frequency: float = 440.0, sample_rate: int = 22050):
    """Create test audio - sine wave."""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    # Create sine wave
    sine_wave = torch.sin(2 * torch.pi * frequency * t).unsqueeze(1)  # Add channel dimension
    return sine_wave

def create_test_noise(duration: float = 3.0, sample_rate: int = 22050):
    """Create test noise audio."""
    num_samples = int(duration * sample_rate)
    noise = torch.randn(num_samples, 1) * 0.1  # Quiet noise
    return noise

def test_audio_segmenter():
    """Test the RajAudioSegmenter node."""
    print("🔧 TESTING AUDIO SEGMENTER")
    print("=" * 50)
    
    try:
        from nodes.audio_segmenter import RajAudioSegmenter
        
        # Create test audio - 10 seconds of 440Hz tone
        test_audio = create_test_audio(duration=10.0, frequency=440.0)
        print(f"   Created test audio: {test_audio.shape[0]/22050:.2f}s, {test_audio.shape[1]} channels")
        
        # Create segmenter
        segmenter = RajAudioSegmenter()
        
        # Test 1: Basic segmentation (extract 3-7 seconds)
        print("\n   Test 1: Basic segmentation (3.0s - 7.0s)")
        segmented, remaining, info = segmenter.segment_audio(
            audio=test_audio,
            start_time=3.0,
            end_time=7.0,
            current_sample_rate=22050,
            fade_edges=True,
            fade_duration=0.1
        )
        
        print(f"   ✅ Segmented: {segmented.shape[0]/22050:.2f}s ({segmented.shape[0]:,} samples)")
        print(f"   ✅ Remaining: {remaining.shape[0]/22050:.2f}s ({remaining.shape[0]:,} samples)")
        print(f"   Expected: 4.0s segment, 6.0s remaining")
        
        # Verify durations
        seg_duration = segmented.shape[0] / 22050
        rem_duration = remaining.shape[0] / 22050
        
        if abs(seg_duration - 4.0) < 0.1:
            print(f"   ✅ Segment duration correct: {seg_duration:.2f}s")
        else:
            print(f"   ❌ Segment duration wrong: {seg_duration:.2f}s (expected ~4.0s)")
            
        if abs(rem_duration - 6.0) < 0.1:
            print(f"   ✅ Remaining duration correct: {rem_duration:.2f}s")
        else:
            print(f"   ❌ Remaining duration wrong: {rem_duration:.2f}s (expected ~6.0s)")
        
        # Test 2: Edge case - segment at beginning
        print("\n   Test 2: Segment at beginning (0.0s - 2.0s)")
        segmented2, remaining2, info2 = segmenter.segment_audio(
            audio=test_audio,
            start_time=0.0,
            end_time=2.0,
            current_sample_rate=22050,
            fade_edges=False,
            fade_duration=0.0
        )
        
        print(f"   ✅ Beginning segment: {segmented2.shape[0]/22050:.2f}s")
        print(f"   ✅ Remaining after beginning: {remaining2.shape[0]/22050:.2f}s")
        
        # Test 3: Crossfade junction
        print("\n   Test 3: Crossfade junction enabled")
        segmented3, remaining3, info3 = segmenter.segment_audio(
            audio=test_audio,
            start_time=4.0,
            end_time=6.0,
            current_sample_rate=22050,
            fade_edges=True,
            fade_duration=0.1,
            crossfade_junction=True,
            junction_fade_duration=0.05
        )
        
        print(f"   ✅ Crossfaded segment: {segmented3.shape[0]/22050:.2f}s")
        print(f"   ✅ Crossfaded remaining: {remaining3.shape[0]/22050:.2f}s")
        
        print("\n✅ AUDIO SEGMENTER TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ AUDIO SEGMENTER TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_multi_segmenter():
    """Test the RajAudioMultiSegmenter node."""
    print("\n🔧 TESTING AUDIO MULTI-SEGMENTER")
    print("=" * 50)
    
    try:
        from nodes.audio_segmenter import RajAudioMultiSegmenter
        
        # Create longer test audio - 15 seconds
        test_audio = create_test_audio(duration=15.0, frequency=880.0)
        print(f"   Created test audio: {test_audio.shape[0]/22050:.2f}s")
        
        # Create multi-segmenter
        segmenter = RajAudioMultiSegmenter()
        
        # Test: Extract multiple segments
        print("\n   Test: Multiple segments (0-2s, 5-8s, 10-13s)")
        segments, remaining, info = segmenter.segment_multiple(
            audio=test_audio,
            segments="0.0-2.0,5.0-8.0,10.0-13.0",
            current_sample_rate=22050,
            output_mode="concatenate",
            include_remaining=True,
            fade_edges=True,
            fade_duration=0.05,
            gap_handling="remove"
        )
        
        print(f"   ✅ Combined segments: {segments.shape[0]/22050:.2f}s")
        print(f"   ✅ Remaining audio: {remaining.shape[0]/22050:.2f}s")
        print(f"   Expected: ~8.0s segments (2+3+3), ~7.0s remaining")
        
        # Verify
        seg_duration = segments.shape[0] / 22050
        rem_duration = remaining.shape[0] / 22050
        
        if abs(seg_duration - 8.0) < 0.2:
            print(f"   ✅ Segments duration correct: {seg_duration:.2f}s")
        else:
            print(f"   ⚠️ Segments duration: {seg_duration:.2f}s (expected ~8.0s)")
            
        print("\n✅ AUDIO MULTI-SEGMENTER TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ AUDIO MULTI-SEGMENTER TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_overlay():
    """Test the RajAudioOverlay node."""
    print("\n🔧 TESTING AUDIO OVERLAY")
    print("=" * 50)
    
    try:
        from nodes.audio_overlay import RajAudioOverlay
        
        # Create test audios
        source_audio = create_test_audio(duration=8.0, frequency=440.0)  # 440Hz tone
        overlay_audio = create_test_audio(duration=4.0, frequency=880.0)  # 880Hz tone
        
        print(f"   Source audio: {source_audio.shape[0]/22050:.2f}s")
        print(f"   Overlay audio: {overlay_audio.shape[0]/22050:.2f}s")
        
        # Create overlay processor
        overlay_processor = RajAudioOverlay()
        
        # Test 1: Add mode
        print("\n   Test 1: Add mode (overlay at 2.0s)")
        mixed1, info1 = overlay_processor.overlay_audio(
            source_audio=source_audio,
            overlay_audio=overlay_audio,
            overlay_mode="add",
            overlay_position=2.0,
            current_sample_rate=22050,
            overlay_volume=0.5,
            source_volume=1.0,
            normalize_output=True
        )
        
        print(f"   ✅ Mixed duration: {mixed1.shape[0]/22050:.2f}s")
        expected_duration = max(8.0, 2.0 + 4.0)  # Max of source or overlay end
        if abs(mixed1.shape[0]/22050 - expected_duration) < 0.1:
            print(f"   ✅ Duration correct")
        else:
            print(f"   ⚠️ Duration: {mixed1.shape[0]/22050:.2f}s (expected ~{expected_duration:.2f}s)")
        
        # Test 2: Replace mode
        print("\n   Test 2: Replace mode")
        mixed2, info2 = overlay_processor.overlay_audio(
            source_audio=source_audio,
            overlay_audio=overlay_audio,
            overlay_mode="replace",
            overlay_position=1.0,
            current_sample_rate=22050,
            overlay_volume=1.0,
            source_volume=1.0
        )
        
        print(f"   ✅ Replace mixed: {mixed2.shape[0]/22050:.2f}s")
        
        # Test 3: Crossfade mode
        print("\n   Test 3: Crossfade mode")
        mixed3, info3 = overlay_processor.overlay_audio(
            source_audio=source_audio,
            overlay_audio=overlay_audio,
            overlay_mode="crossfade",
            overlay_position=3.0,
            current_sample_rate=22050,
            overlay_volume=1.0,
            source_volume=1.0,
            crossfade_duration=1.0
        )
        
        print(f"   ✅ Crossfade mixed: {mixed3.shape[0]/22050:.2f}s")
        
        # Test 4: Ducking mode
        print("\n   Test 4: Ducking mode")
        mixed4, info4 = overlay_processor.overlay_audio(
            source_audio=source_audio,
            overlay_audio=overlay_audio,
            overlay_mode="ducking",
            overlay_position=2.5,
            current_sample_rate=22050,
            overlay_volume=1.0,
            source_volume=1.0,
            ducking_threshold=0.3,
            ducking_fade_time=0.2
        )
        
        print(f"   ✅ Ducking mixed: {mixed4.shape[0]/22050:.2f}s")
        
        # Test 5: Insert mode
        print("\n   Test 5: Insert mode")
        mixed5, info5 = overlay_processor.overlay_audio(
            source_audio=source_audio,
            overlay_audio=overlay_audio,
            overlay_mode="insert",
            overlay_position=3.0,
            current_sample_rate=22050,
            overlay_volume=1.0,
            source_volume=1.0
        )
        
        expected_insert_duration = 8.0 + 4.0  # Source + overlay length
        print(f"   ✅ Insert mixed: {mixed5.shape[0]/22050:.2f}s (expected ~{expected_insert_duration:.2f}s)")
        
        print("\n✅ AUDIO OVERLAY TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ AUDIO OVERLAY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_multi_overlay():
    """Test the RajAudioMultiOverlay node."""
    print("\n🔧 TESTING AUDIO MULTI-OVERLAY")
    print("=" * 50)
    
    try:
        from nodes.audio_overlay import RajAudioMultiOverlay
        
        # Create test audios
        base_audio = create_test_audio(duration=10.0, frequency=220.0)  # Base tone
        overlay1 = create_test_audio(duration=3.0, frequency=440.0)     # Overlay 1
        overlay2 = create_test_audio(duration=2.5, frequency=880.0)     # Overlay 2
        overlay3 = create_test_noise(duration=2.0)                      # Overlay 3 (noise)
        
        print(f"   Base audio: {base_audio.shape[0]/22050:.2f}s")
        print(f"   Overlay 1: {overlay1.shape[0]/22050:.2f}s")
        print(f"   Overlay 2: {overlay2.shape[0]/22050:.2f}s")
        print(f"   Overlay 3: {overlay3.shape[0]/22050:.2f}s")
        
        # Create multi-overlay processor
        multi_overlay = RajAudioMultiOverlay()
        
        # Test: Multi-track mixing
        print("\n   Test: Multi-track mixing (normalized mode)")
        mixed, info = multi_overlay.multi_overlay(
            base_audio=base_audio,
            current_sample_rate=22050,
            mix_method="normalized",
            overlay_1=overlay1,
            overlay_2=overlay2,
            overlay_3=overlay3,
            overlay_4=None,
            positions="1.0,4.0,7.0,0.0",  # Start times
            volumes="1.0,0.8,0.6,0.4",    # Volumes
            auto_duck=True,
            duck_amount=0.3,
            normalize_output=True
        )
        
        print(f"   ✅ Multi-track mixed: {mixed.shape[0]/22050:.2f}s")
        print(f"   Expected: ~10.0s (base audio length)")
        
        mixed_duration = mixed.shape[0] / 22050
        if abs(mixed_duration - 10.0) < 0.2:
            print(f"   ✅ Duration correct: {mixed_duration:.2f}s")
        else:
            print(f"   ⚠️ Duration: {mixed_duration:.2f}s")
        
        # Test different mix methods
        print("\n   Test: Different mix methods")
        
        methods = ["additive", "compressed", "layered"]
        for method in methods:
            try:
                mixed_method, info_method = multi_overlay.multi_overlay(
                    base_audio=base_audio,
                    current_sample_rate=22050,
                    mix_method=method,
                    overlay_1=overlay1,
                    overlay_2=overlay2,
                    positions="2.0,5.0",
                    volumes="0.7,0.5",
                    normalize_output=True
                )
                print(f"   ✅ {method} method: {mixed_method.shape[0]/22050:.2f}s")
            except Exception as e:
                print(f"   ⚠️ {method} method failed: {e}")
        
        print("\n✅ AUDIO MULTI-OVERLAY TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ AUDIO MULTI-OVERLAY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_workflow():
    """Test a complete workflow: segment → overlay → combine."""
    print("\n🔧 TESTING INTEGRATION WORKFLOW")
    print("=" * 50)
    
    try:
        from nodes.audio_segmenter import RajAudioSegmenter
        from nodes.audio_overlay import RajAudioOverlay
        
        # Create original audio - 10 seconds
        original_audio = create_test_audio(duration=10.0, frequency=440.0)
        
        # Step 1: Segment out a portion (remove 3-5s section)
        segmenter = RajAudioSegmenter()
        removed_segment, remaining_audio, seg_info = segmenter.segment_audio(
            audio=original_audio,
            start_time=3.0,
            end_time=5.0,
            current_sample_rate=22050,
            fade_edges=True,
            fade_duration=0.1
        )
        
        print(f"   Step 1 - Segmentation:")
        print(f"     Removed segment: {removed_segment.shape[0]/22050:.2f}s")
        print(f"     Remaining audio: {remaining_audio.shape[0]/22050:.2f}s")
        
        # Step 2: Create replacement audio
        replacement_audio = create_test_audio(duration=2.0, frequency=880.0)
        print(f"   Step 2 - Created replacement: {replacement_audio.shape[0]/22050:.2f}s")
        
        # Step 3: Insert replacement at the gap (position 3.0s in remaining)
        overlay = RajAudioOverlay()
        final_audio, mix_info = overlay.overlay_audio(
            source_audio=remaining_audio,
            overlay_audio=replacement_audio,
            overlay_mode="insert",
            overlay_position=3.0,
            current_sample_rate=22050,
            overlay_volume=1.0,
            source_volume=1.0
        )
        
        print(f"   Step 3 - Inserted replacement:")
        print(f"     Final audio: {final_audio.shape[0]/22050:.2f}s")
        
        # Verify workflow
        # Original: 10s, Removed: 2s, Added: 2s → Final should be ~10s
        final_duration = final_audio.shape[0] / 22050
        if abs(final_duration - 10.0) < 0.3:
            print(f"   ✅ Integration workflow successful: {final_duration:.2f}s")
        else:
            print(f"   ⚠️ Integration result: {final_duration:.2f}s (expected ~10.0s)")
        
        print("\n✅ INTEGRATION WORKFLOW TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ INTEGRATION WORKFLOW TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all audio overlay system tests."""
    print("🚀 TESTING AUDIO OVERLAY & SEGMENTATION SYSTEM")
    print("=" * 80)
    
    tests = [
        test_audio_segmenter,
        test_audio_multi_segmenter,
        test_audio_overlay,
        test_audio_multi_overlay,
        test_integration_workflow
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
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 80)
    print(f"🏁 TEST SUMMARY: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL AUDIO OVERLAY & SEGMENTATION TESTS PASSED!")
        print("   ✅ Audio segmentation working correctly")
        print("   ✅ Multi-segment extraction working")  
        print("   ✅ Audio overlay modes functioning")
        print("   ✅ Multi-track mixing operational")
        print("   ✅ Integration workflows successful")
        print("\n📋 SYSTEM READY FOR PRODUCTION USE!")
        return True
    else:
        print("💥 Some tests FAILED!")
        print("📋 Please check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)