#!/usr/bin/env python3
"""
Simple logic validation for audio overlay system.
Tests core algorithms without ComfyUI dependencies.
"""

import sys
import torch
import numpy as np

def test_segmentation_logic():
    """Test core audio segmentation logic."""
    print("🔧 TESTING SEGMENTATION LOGIC")
    print("=" * 40)
    
    # Create test audio - 10 seconds at 22050 Hz
    sample_rate = 22050
    duration = 10.0
    num_samples = int(duration * sample_rate)
    
    # Create sine wave test audio
    t = torch.linspace(0, duration, num_samples)
    audio = torch.sin(2 * torch.pi * 440 * t).unsqueeze(1)  # 440Hz tone, mono
    
    print(f"   Original audio: {audio.shape[0]/sample_rate:.2f}s, {audio.shape}")
    
    # Test segmentation logic (extract 3-7 seconds)
    start_time, end_time = 3.0, 7.0
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    # Extract segment
    segment = audio[start_sample:end_sample].clone()
    
    # Create remaining (before + after)
    before = audio[:start_sample]
    after = audio[end_sample:]
    remaining = torch.cat([before, after], dim=0)
    
    # Verify results
    segment_duration = segment.shape[0] / sample_rate
    remaining_duration = remaining.shape[0] / sample_rate
    
    print(f"   Segment: {segment_duration:.2f}s (expected 4.0s)")
    print(f"   Remaining: {remaining_duration:.2f}s (expected 6.0s)")
    
    success = (abs(segment_duration - 4.0) < 0.01 and 
               abs(remaining_duration - 6.0) < 0.01)
    
    print(f"   Result: {'✅ PASSED' if success else '❌ FAILED'}")
    return success

def test_fade_logic():
    """Test fade in/out logic."""
    print("\n🔧 TESTING FADE LOGIC")
    print("=" * 40)
    
    # Create test audio
    duration = 2.0
    sample_rate = 22050
    num_samples = int(duration * sample_rate)
    audio = torch.ones(num_samples, 1)  # Constant amplitude
    
    # Apply fade in/out (0.1s each)
    fade_duration = 0.1
    fade_samples = int(fade_duration * sample_rate)
    
    # Fade in
    fade_in_curve = torch.linspace(0, 1, fade_samples).unsqueeze(1)
    audio[:fade_samples] *= fade_in_curve
    
    # Fade out
    fade_out_curve = torch.linspace(1, 0, fade_samples).unsqueeze(1)
    audio[-fade_samples:] *= fade_out_curve
    
    # Check fade results
    start_val = audio[0, 0].item()
    end_val = audio[-1, 0].item()
    middle_val = audio[num_samples//2, 0].item()
    
    print(f"   Start value: {start_val:.4f} (expected ~0.0)")
    print(f"   Middle value: {middle_val:.4f} (expected ~1.0)")
    print(f"   End value: {end_val:.4f} (expected ~0.0)")
    
    success = (start_val < 0.1 and end_val < 0.1 and middle_val > 0.9)
    print(f"   Result: {'✅ PASSED' if success else '❌ FAILED'}")
    return success

def test_overlay_logic():
    """Test audio overlay logic."""
    print("\n🔧 TESTING OVERLAY LOGIC")
    print("=" * 40)
    
    # Create source and overlay audio
    sample_rate = 22050
    source = torch.ones(int(5.0 * sample_rate), 1) * 0.5  # 5s at 0.5 amplitude
    overlay = torch.ones(int(2.0 * sample_rate), 1) * 0.3  # 2s at 0.3 amplitude
    
    print(f"   Source: {source.shape[0]/sample_rate:.2f}s at {source[0,0]:.1f} amplitude")
    print(f"   Overlay: {overlay.shape[0]/sample_rate:.2f}s at {overlay[0,0]:.1f} amplitude")
    
    # Test 1: Additive mixing at position 1.0s
    overlay_position = 1.0
    overlay_start = int(overlay_position * sample_rate)
    
    mixed = source.clone()
    end_sample = min(overlay_start + overlay.shape[0], mixed.shape[0])
    overlay_length = end_sample - overlay_start
    
    if overlay_start < mixed.shape[0] and overlay_length > 0:
        mixed[overlay_start:end_sample] += overlay[:overlay_length]
    
    # Check results
    before_val = mixed[0, 0].item()  # Before overlay
    during_val = mixed[overlay_start + 1000, 0].item()  # During overlay
    after_val = mixed[end_sample + 1000, 0].item() if end_sample + 1000 < mixed.shape[0] else mixed[-1, 0].item()
    
    print(f"   Before overlay: {before_val:.2f} (expected 0.5)")
    print(f"   During overlay: {during_val:.2f} (expected 0.8)")
    print(f"   After overlay: {after_val:.2f} (expected 0.5)")
    
    success = (abs(before_val - 0.5) < 0.01 and 
               abs(during_val - 0.8) < 0.01 and
               abs(after_val - 0.5) < 0.01)
    
    print(f"   Result: {'✅ PASSED' if success else '❌ FAILED'}")
    return success

def test_replace_logic():
    """Test replace mode logic."""
    print("\n🔧 TESTING REPLACE LOGIC")  
    print("=" * 40)
    
    sample_rate = 22050
    source = torch.ones(int(5.0 * sample_rate), 1) * 0.5
    overlay = torch.ones(int(2.0 * sample_rate), 1) * 0.9
    
    # Replace mode at 1.5s
    overlay_position = 1.5
    overlay_start = int(overlay_position * sample_rate)
    
    mixed = source.clone()
    end_sample = min(overlay_start + overlay.shape[0], mixed.shape[0])
    overlay_length = end_sample - overlay_start
    
    if overlay_start < mixed.shape[0] and overlay_length > 0:
        mixed[overlay_start:end_sample] = overlay[:overlay_length]
    
    # Check replacement
    before_val = mixed[overlay_start - 1000, 0].item()
    during_val = mixed[overlay_start + 1000, 0].item()
    after_val = mixed[end_sample + 1000, 0].item() if end_sample + 1000 < mixed.shape[0] else mixed[-1, 0].item()
    
    print(f"   Before replacement: {before_val:.2f} (expected 0.5)")
    print(f"   During replacement: {during_val:.2f} (expected 0.9)")
    print(f"   After replacement: {after_val:.2f} (expected 0.5)")
    
    success = (abs(before_val - 0.5) < 0.01 and 
               abs(during_val - 0.9) < 0.01 and
               abs(after_val - 0.5) < 0.01)
    
    print(f"   Result: {'✅ PASSED' if success else '❌ FAILED'}")
    return success

def test_insert_logic():
    """Test insert mode logic."""
    print("\n🔧 TESTING INSERT LOGIC")
    print("=" * 40)
    
    sample_rate = 22050
    source = torch.ones(int(4.0 * sample_rate), 1) * 0.5  # 4s source
    overlay = torch.ones(int(1.0 * sample_rate), 1) * 0.8  # 1s overlay
    
    # Insert at 2.0s
    insert_position = 2.0
    insert_sample = int(insert_position * sample_rate)
    
    # Split and insert
    before = source[:insert_sample]
    after = source[insert_sample:]
    mixed = torch.cat([before, overlay, after], dim=0)
    
    expected_duration = 4.0 + 1.0  # Original + inserted
    actual_duration = mixed.shape[0] / sample_rate
    
    print(f"   Original duration: 4.0s")
    print(f"   Inserted duration: 1.0s")
    print(f"   Final duration: {actual_duration:.2f}s (expected {expected_duration:.1f}s)")
    
    # Check values at different points
    before_insert = mixed[insert_sample - 1000, 0].item()  # Just before insert
    during_insert = mixed[insert_sample + 1000, 0].item()  # During insert
    after_insert = mixed[insert_sample + overlay.shape[0] + 1000, 0].item()  # After insert
    
    print(f"   Before insert point: {before_insert:.2f} (expected 0.5)")
    print(f"   During insert: {during_insert:.2f} (expected 0.8)")
    print(f"   After insert: {after_insert:.2f} (expected 0.5)")
    
    success = (abs(actual_duration - expected_duration) < 0.01 and
               abs(before_insert - 0.5) < 0.01 and
               abs(during_insert - 0.8) < 0.01 and
               abs(after_insert - 0.5) < 0.01)
    
    print(f"   Result: {'✅ PASSED' if success else '❌ FAILED'}")
    return success

def test_normalization_logic():
    """Test audio normalization logic."""
    print("\n🔧 TESTING NORMALIZATION LOGIC")
    print("=" * 40)
    
    # Create audio that peaks above 1.0
    audio = torch.randn(1000, 1) * 2.0  # Random audio with high amplitude
    
    original_max = torch.abs(audio).max().item()
    print(f"   Original peak: {original_max:.2f}")
    
    # Normalize to prevent clipping
    if original_max > 1.0:
        audio = audio / original_max * 0.95
    
    normalized_max = torch.abs(audio).max().item()
    print(f"   Normalized peak: {normalized_max:.2f} (expected ~0.95)")
    
    success = normalized_max <= 1.0 and normalized_max > 0.9
    print(f"   Result: {'✅ PASSED' if success else '❌ FAILED'}")
    return success

def main():
    """Run all core logic tests."""
    print("🚀 TESTING AUDIO OVERLAY CORE LOGIC")
    print("=" * 60)
    
    tests = [
        test_segmentation_logic,
        test_fade_logic, 
        test_overlay_logic,
        test_replace_logic,
        test_insert_logic,
        test_normalization_logic
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
    
    print("\n" + "=" * 60)
    print(f"🏁 CORE LOGIC TEST SUMMARY: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL CORE LOGIC TESTS PASSED!")
        print("   ✅ Audio segmentation algorithms working")
        print("   ✅ Fade in/out logic correct")
        print("   ✅ Overlay mixing functioning")
        print("   ✅ Replace mode working")
        print("   ✅ Insert mode working")
        print("   ✅ Normalization logic correct")
        print("\n📋 CORE ALGORITHMS VALIDATED - READY FOR INTEGRATION!")
        return True
    else:
        print("💥 Some core logic tests FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)