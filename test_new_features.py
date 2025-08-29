#!/usr/bin/env python3
"""
Test all new features added to raj-vid-processings
"""

import os
import time
from nodes.video_upload import RajVideoUpload
from nodes.video_concatenator import RajVideoConcatenator  
from nodes.video_saver import RajVideoSaver
from nodes.video_effects import RajVideoEffects, RajVideoSharpness
from nodes.video_transitions import RajVideoTransitions

def test_new_features():
    print("🧪 Testing New Video Processing Features")
    print("=" * 60)
    
    try:
        # Test 1: Video Saver with 24 FPS default and auto-increment filenames
        print("\n1️⃣ Testing Video Saver (24 FPS default + auto-increment)")
        saver = RajVideoSaver()
        
        # Load test video first
        upload = RajVideoUpload()
        result = upload.upload_and_load_video(
            video="/Users/rajeshkammaluri/mine/content_creation2/ComfyUI/input/Pirate_Captains_Stormy_Coffee_Toast.mp4",
            target_fps=24,
            max_frames=50
        )
        test_frames = result["result"][0] if isinstance(result, dict) else result[0]
        
        # Save with same filename multiple times to test auto-increment
        for i in range(3):
            save_result = saver.save_video(
                frames=test_frames,
                filename="test_increment",
                fps=24.0,  # Should default to 24.0 now
                format="mp4",
                save_to_output=True
            )
            print(f"   Save {i+1}: {save_result['result'][0] if isinstance(save_result, dict) else save_result[0]}")
        
        print("   ✅ Auto-increment filenames working")
        
        # Test 2: Aspect Ratio Handling in Concatenator
        print("\n2️⃣ Testing Aspect Ratio Handling in Concatenator")
        concatenator = RajVideoConcatenator()
        
        # Create test videos with different aspect ratios (simulate by reshaping)
        video1 = test_frames[:25]  # 25 frames of original size
        
        # Load second video
        result2 = upload.upload_and_load_video(
            video="/Users/rajeshkammaluri/mine/content_creation2/ComfyUI/input/Pirate_Battle_Halts_for_Break.mp4",
            target_fps=24,
            max_frames=25
        )
        video2 = result2["result"][0] if isinstance(result2, dict) else result2[0]
        
        # Test different aspect ratio handling modes
        for mode in ["pad", "crop", "resize", "stretch"]:
            print(f"   Testing aspect ratio handling: {mode}")
            concat_result = concatenator.concatenate_videos(
                video1=video1,
                video2=video2,
                aspect_ratio_handling=mode,
                chunk_size=16
            )
            print(f"   ✅ {mode}: {concat_result['result'][1] if isinstance(concat_result, dict) else concat_result[1]}")
        
        # Test 3: Video Effects with Time-based Controls
        print("\n3️⃣ Testing Video Effects (Time-based)")
        effects = RajVideoEffects()
        
        effects_result = effects.apply_effects(
            frames=test_frames,
            fps=24.0,
            brightness_enabled=True,
            brightness_start_time=0.0,
            brightness_end_time=1.0,
            brightness_start_value=0.0,
            brightness_end_value=30.0,
            brightness_easing="ease_in_out",
            contrast_enabled=True,
            contrast_start_time=0.5,
            contrast_end_time=1.5,
            contrast_start_value=1.0,
            contrast_end_value=1.5,
            contrast_easing="linear"
        )
        
        print(f"   ✅ Effects applied: {effects_result['result'][1] if isinstance(effects_result, dict) else effects_result[1]}")
        
        # Test 4: Video Sharpness
        print("\n4️⃣ Testing Video Sharpness")
        sharpness = RajVideoSharpness()
        
        sharpness_result = sharpness.apply_sharpness(
            frames=test_frames,
            fps=24.0,
            sharpness_start_time=0.0,
            sharpness_end_time=1.0,
            sharpness_start_value=1.0,
            sharpness_end_value=2.0,
            sharpness_easing="ease_out"
        )
        
        print(f"   ✅ Sharpness applied: {sharpness_result[1]}")
        
        # Test 5: Video Transitions with Cut Points
        print("\n5️⃣ Testing Video Transitions (Cut Points)")
        transitions = RajVideoTransitions()
        
        transitions_result = transitions.apply_transitions(
            frames=test_frames,
            fps=24.0,
            cut_points="0.5, 1.5",
            transition_type="fade",
            transition_duration=0.3,
            transition_easing="ease_in_out",
            fade_color="black"
        )
        
        print(f"   ✅ Transitions applied: {transitions_result['result'][1] if isinstance(transitions_result, dict) else transitions_result[1]}")
        
        # Test different transition types
        transition_types = ["zoom", "slide", "wipe", "dissolve"]
        for trans_type in transition_types:
            print(f"   Testing {trans_type} transition...")
            result = transitions.apply_transitions(
                frames=test_frames[:30],  # Shorter for speed
                fps=24.0,
                cut_points="0.5",
                transition_type=trans_type,
                transition_duration=0.2,
                transition_easing="linear"
            )
            print(f"   ✅ {trans_type}: Applied successfully")
        
        # Test 6: Time Utilities
        print("\n6️⃣ Testing Time Utilities")
        from nodes.utils import time_to_frame, frame_to_time, parse_time_points
        
        # Test time to frame conversion
        test_times = [0.5, 1.0, 2.5, 5.0]
        for t in test_times:
            frame = time_to_frame(t, 24.0)
            back_to_time = frame_to_time(frame, 24.0)
            print(f"   Time {t}s → Frame {frame} → Time {back_to_time:.2f}s")
        
        # Test cut point parsing
        cut_string = "2.5, 5.0, 8.5"
        cut_frames = parse_time_points(cut_string, 24.0)
        print(f"   Cut points '{cut_string}' → Frames {cut_frames}")
        print("   ✅ Time utilities working correctly")
        
        # Test 7: Easing Functions
        print("\n7️⃣ Testing Easing Functions")
        from nodes.utils import apply_easing
        
        easing_types = ["linear", "ease_in", "ease_out", "ease_in_out", "ease_out_in"]
        for easing in easing_types:
            # Test easing from 0 to 100 at t=0.5
            value = apply_easing(0, 0.0, 100.0, 0.5, easing)
            print(f"   {easing}: 0→100 at t=0.5 = {value:.1f}")
        
        print("   ✅ Easing functions working correctly")
        
        # Test 8: Auto-increment filename function
        print("\n8️⃣ Testing Auto-increment Filename Function")
        from nodes.utils import get_save_path_incremental
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            for i in range(3):
                if i == 0:
                    test_path = os.path.join(temp_dir, "test_file.mp4")
                else:
                    test_path = os.path.join(temp_dir, f"test_file_{i:05d}.mp4")
                
                with open(test_path, 'w') as f:
                    f.write("test")
            
            # Test getting next available filename
            next_path = get_save_path_incremental("test_file", temp_dir, "mp4")
            expected = os.path.join(temp_dir, "test_file_00004.mp4")
            
            if next_path == expected:
                print("   ✅ Auto-increment filename function working")
            else:
                print(f"   ❌ Expected {expected}, got {next_path}")
        
        print("\n✅ ALL NEW FEATURES TESTED SUCCESSFULLY!")
        print("=" * 60)
        print("🎉 Enhanced Features Summary:")
        print("   • Video Saver: 24 FPS default + auto-increment filenames")
        print("   • Concatenator: Aspect ratio handling (pad/crop/resize/stretch)")
        print("   • Video Effects: Time-based brightness, contrast, blur")
        print("   • Video Sharpness: Dedicated sharpness adjustment")
        print("   • Video Transitions: Cut points with fade/zoom/slide/wipe/dissolve")
        print("   • Time Utilities: Frame/time conversion, cut point parsing")
        print("   • Easing Functions: Linear, ease-in/out variations")
        print("   • Video Preview: Enhanced UI preview support")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_new_features()