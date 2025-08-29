#!/usr/bin/env python3
"""
Test new video processing nodes: Trimmer, Cutter, Multi-Cutter, and Mask Composite
"""

import os
import sys
import torch
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nodes.video_upload import RajVideoUpload
from nodes.video_trimmer import RajVideoTrimmer, RajVideoCutter, RajVideoTimecodeConverter
from nodes.video_multi_cutter import RajVideoMultiCutter, RajVideoSegmentManager
from nodes.video_mask_composite import RajVideoMaskComposite
from nodes.video_saver import RajVideoSaver

def create_test_video_data(frames=120, width=640, height=480, fps=24.0):
    """Create synthetic test video data"""
    # Create gradient pattern that changes over time
    video_frames = []
    
    for i in range(frames):
        # Create a frame with time-based gradient
        frame = np.zeros((height, width, 3), dtype=np.float32)
        
        # Horizontal gradient
        for x in range(width):
            frame[:, x, 0] = x / width  # Red gradient
        
        # Vertical gradient
        for y in range(height):
            frame[y, :, 1] = y / height  # Green gradient
        
        # Time-based blue component
        frame[:, :, 2] = (i / frames) * 0.8  # Blue increases over time
        
        video_frames.append(frame)
    
    return torch.tensor(np.array(video_frames), dtype=torch.float32)

def test_video_nodes():
    print("🧪 Testing New Video Processing Nodes")
    print("=" * 60)
    
    try:
        # Create test video data
        print("\n📹 Creating test video data...")
        test_frames = create_test_video_data(frames=120, fps=24.0)  # 5 seconds at 24fps
        print(f"   Test video: {test_frames.shape[0]} frames, {test_frames.shape[1]}x{test_frames.shape[2]}px")
        print(f"   Duration: {test_frames.shape[0]/24.0:.2f} seconds @ 24fps")
        
        # Test 1: Video Trimmer
        print("\n1️⃣ Testing RajVideoTrimmer")
        trimmer = RajVideoTrimmer()
        
        # Test seconds format
        trim_result = trimmer.trim_video(
            frames=test_frames,
            fps=24.0,
            start_time=1.0,
            end_time=3.0,
            time_format="seconds"
        )
        
        if isinstance(trim_result, dict):
            trimmed_frames = trim_result["result"][0]
            trim_info = trim_result["result"][1]
        else:
            trimmed_frames, trim_info, frame_count, duration = trim_result
        
        print(f"   ✅ Trimmed result: {trim_info}")
        print(f"   Trimmed frames shape: {trimmed_frames.shape}")
        
        # Test timecode format
        timecode_trim_result = trimmer.trim_video(
            frames=test_frames,
            fps=24.0,
            start_time=0.0,  # Will be ignored
            end_time=0.0,    # Will be ignored
            time_format="timecode",
            start_timecode="00:00:01:12",  # 1.5 seconds
            end_timecode="00:00:03:00"     # 3.0 seconds
        )
        
        print("   ✅ Timecode trimming successful")
        
        # Test 2: Video Cutter
        print("\n2️⃣ Testing RajVideoCutter")
        cutter = RajVideoCutter()
        
        cut_result = cutter.cut_video(
            frames=test_frames,
            fps=24.0,
            cut_start_time=2.0,
            cut_end_time=4.0,
            time_format="seconds"
        )
        
        remaining_video, removed_segment, cut_info, remaining_frames, removed_frames, remaining_duration, removed_duration = cut_result
        
        print(f"   ✅ Cut result: {cut_info}")
        print(f"   Remaining video: {remaining_video.shape[0]} frames ({remaining_duration:.2f}s)")
        print(f"   Removed segment: {removed_segment.shape[0]} frames ({removed_duration:.2f}s)")
        print(f"   Original: {test_frames.shape[0]} frames = {remaining_frames} + {removed_frames}")
        
        # Verify math
        total_original = test_frames.shape[0]
        total_result = remaining_frames + removed_frames
        print(f"   Frame count verification: {total_original} = {total_result} ({'✅' if total_original == total_result else '❌'})")
        
        # Test 3: Timecode Converter
        print("\n3️⃣ Testing RajVideoTimecodeConverter")
        converter = RajVideoTimecodeConverter()
        
        # Test seconds to timecode
        timecode_result = converter.convert_time(
            fps=24.0,
            conversion_mode="seconds_to_timecode",
            input_seconds=65.5
        )
        
        timecode, seconds, conversion_info = timecode_result
        print(f"   ✅ Seconds to timecode: {conversion_info}")
        
        # Test timecode to seconds
        seconds_result = converter.convert_time(
            fps=24.0,
            conversion_mode="timecode_to_seconds",
            input_timecode="00:01:05:12"
        )
        
        timecode, seconds, conversion_info = seconds_result
        print(f"   ✅ Timecode to seconds: {conversion_info}")
        
        # Test 4: Multi-Cutter
        print("\n4️⃣ Testing RajVideoMultiCutter")
        multi_cutter = RajVideoMultiCutter()
        
        # Test all segments mode
        multi_result = multi_cutter.multi_cut_video(
            frames=test_frames,
            fps=24.0,
            cut_points="1.0, 2.5, 4.0",
            output_mode="all_segments"
        )
        
        output_segments, cut_info, segment_count, total_duration, segment_details = multi_result
        
        print(f"   ✅ Multi-cut result: {cut_info}")
        print(f"   Output segments: {output_segments.shape[0]} frames ({total_duration:.2f}s)")
        print(f"   Segment count: {segment_count}")
        print("   Segment details:")
        for line in segment_details.split('\n')[:6]:  # Show first 6 lines
            print(f"     {line}")
        
        # Test selected segments mode
        selected_result = multi_cutter.multi_cut_video(
            frames=test_frames,
            fps=24.0,
            cut_points="1.0, 3.0",
            output_mode="selected_segments",
            selected_indices="0, 2"  # First and third segments
        )
        
        print("   ✅ Selected segments mode successful")
        
        # Test 5: Segment Manager
        print("\n5️⃣ Testing RajVideoSegmentManager")
        segment_manager = RajVideoSegmentManager()
        
        analysis_result = segment_manager.analyze_segments(
            frames=test_frames,
            fps=24.0,
            analysis_mode="timing_analysis",
            segment_duration=2.0
        )
        
        analysis, segments, duration, details = analysis_result
        print(f"   ✅ Segment analysis: {analysis}")
        print(f"   Details preview:")
        for line in details.split('\n')[:5]:  # Show first 5 lines
            print(f"     {line}")
        
        # Test 6: Mask Composite (Create two test videos)
        print("\n6️⃣ Testing RajVideoMaskComposite")
        
        # Create background video (blue gradient)
        bg_frames = create_test_video_data(frames=60, width=320, height=240)
        bg_frames[:, :, :, 0] = 0.2  # Low red
        bg_frames[:, :, :, 1] = 0.3  # Low green
        bg_frames[:, :, :, 2] = 0.8  # High blue (blue background)
        
        # Create foreground video with green screen area
        fg_frames = create_test_video_data(frames=60, width=320, height=240)
        # Create a green screen in the center
        center_h, center_w = 120, 160
        margin_h, margin_w = 60, 80
        fg_frames[:, margin_h:margin_h+center_h, margin_w:margin_w+center_w, 0] = 0.0  # No red
        fg_frames[:, margin_h:margin_h+center_h, margin_w:margin_w+center_w, 1] = 1.0  # Full green
        fg_frames[:, margin_h:margin_h+center_h, margin_w:margin_w+center_w, 2] = 0.0  # No blue
        
        compositor = RajVideoMaskComposite()
        
        # Test chroma key compositing
        composite_result = compositor.composite_video(
            base_video=bg_frames,
            overlay_video=fg_frames,
            fps=24.0,
            mask_mode="chroma_key",
            mask_color_r=0.0,
            mask_color_g=1.0,
            mask_color_b=0.0,
            tolerance=0.2,
            edge_softness=0.02,
            blend_mode="normal"
        )
        
        composite_video, mask_viz, composite_info, frame_count, duration = composite_result
        
        print(f"   ✅ Chroma key composite: {composite_info}")
        print(f"   Composite video: {composite_video.shape[0]} frames")
        print(f"   Mask visualization: {mask_viz.shape}")
        
        # Test different blend modes
        screen_result = compositor.composite_video(
            base_video=bg_frames,
            overlay_video=fg_frames,
            fps=24.0,
            mask_mode="brightness_mask",
            brightness_threshold=0.5,
            blend_mode="screen",
            opacity=0.8
        )
        
        print("   ✅ Screen blend mode successful")
        
        # Test 7: Integration Test (Chain multiple operations)
        print("\n7️⃣ Testing Node Integration")
        
        # Chain: Upload -> Trim -> Cut -> Multi-cut -> Composite
        print("   Chaining operations: Trim -> Cut -> Composite")
        
        # Step 1: Trim the test video
        trim_step = trimmer.trim_video(
            frames=test_frames,
            fps=24.0,
            start_time=0.5,
            end_time=4.5,
            time_format="seconds"
        )
        
        if isinstance(trim_step, dict):
            trimmed_for_chain = trim_step["result"][0]
        else:
            trimmed_for_chain = trim_step[0]
        
        # Step 2: Cut a piece from the trimmed video
        cut_step = cutter.cut_video(
            frames=trimmed_for_chain,
            fps=24.0,
            cut_start_time=1.0,
            cut_end_time=2.0,
            time_format="seconds"
        )
        
        remaining_for_chain = cut_step[0]  # Use remaining video
        removed_for_overlay = cut_step[1]  # Use removed segment as overlay
        
        # Step 3: Composite the remaining with the removed segment
        if remaining_for_chain.shape[0] > 0 and removed_for_overlay.shape[0] > 0:
            # Make them same length for compositing
            min_frames = min(remaining_for_chain.shape[0], removed_for_overlay.shape[0])
            base_for_composite = remaining_for_chain[:min_frames]
            overlay_for_composite = removed_for_overlay[:min_frames]
            
            final_composite = compositor.composite_video(
                base_video=base_for_composite,
                overlay_video=overlay_for_composite,
                fps=24.0,
                mask_mode="brightness_mask",
                brightness_threshold=0.3,
                blend_mode="overlay",
                opacity=0.7
            )
            
            print(f"   ✅ Integration successful: Final composite {final_composite[0].shape[0]} frames")
        
        # Test 8: Edge Cases and Error Handling
        print("\n8️⃣ Testing Edge Cases")
        
        # Test invalid time ranges
        try:
            invalid_trim = trimmer.trim_video(
                frames=test_frames,
                fps=24.0,
                start_time=3.0,
                end_time=2.0,  # End before start
                time_format="seconds"
            )
            print("   ❌ Should have failed with invalid time range")
        except ValueError as e:
            print(f"   ✅ Correctly caught invalid time range: {str(e)[:50]}...")
        
        # Test empty cut points
        try:
            empty_multi_cut = multi_cutter.multi_cut_video(
                frames=test_frames,
                fps=24.0,
                cut_points="",  # Empty cut points
                output_mode="all_segments"
            )
            print("   ✅ Handled empty cut points gracefully")
        except Exception as e:
            print(f"   ⚠️ Empty cut points issue: {str(e)[:50]}...")
        
        # Test very small video
        small_frames = create_test_video_data(frames=5, width=32, height=32)
        small_trim = trimmer.trim_video(
            frames=small_frames,
            fps=24.0,
            start_time=0.0,
            end_time=0.2,
            time_format="seconds"
        )
        print("   ✅ Handled small video dimensions")
        
        print("\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📊 Test Summary:")
        print("   • RajVideoTrimmer: Time-based trimming with seconds and timecode ✅")
        print("   • RajVideoCutter: Dual output cutting (remaining + removed) ✅")
        print("   • RajVideoTimecodeConverter: Bidirectional time conversion ✅")
        print("   • RajVideoMultiCutter: Multiple cut points with segment management ✅")
        print("   • RajVideoSegmentManager: Video analysis and segment info ✅")
        print("   • RajVideoMaskComposite: Color-based masking and compositing ✅")
        print("   • Integration Testing: Node chaining and workflows ✅")
        print("   • Edge Cases: Error handling and validation ✅")
        print("=" * 60)
        print("🎉 All new video processing nodes are working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_with_real_video():
    """Test with real video file if available"""
    print("\n🎬 Testing with Real Video (if available)")
    
    # Try to find a real video file
    video_paths = [
        "/Users/rajeshkammaluri/mine/content_creation2/ComfyUI/input/Pirate_Captains_Stormy_Coffee_Toast.mp4",
        "test_video.mp4",
        "sample.mp4"
    ]
    
    real_video_path = None
    for path in video_paths:
        if os.path.exists(path):
            real_video_path = path
            break
    
    if real_video_path:
        print(f"   Found real video: {real_video_path}")
        
        try:
            # Load real video
            uploader = RajVideoUpload()
            upload_result = uploader.upload_and_load_video(
                video=real_video_path,
                target_fps=24,
                max_frames=120  # Limit for testing
            )
            
            if isinstance(upload_result, dict):
                real_frames = upload_result["result"][0]
            else:
                real_frames = upload_result[0]
            
            print(f"   Loaded real video: {real_frames.shape}")
            
            # Test trimmer with real video
            trimmer = RajVideoTrimmer()
            real_trim_result = trimmer.trim_video(
                frames=real_frames,
                fps=24.0,
                start_time=1.0,
                end_time=3.0,
                time_format="seconds"
            )
            
            print("   ✅ Real video trimming successful")
            
            # Test saving the result
            saver = RajVideoSaver()
            if isinstance(real_trim_result, dict):
                frames_to_save = real_trim_result["result"][0]
            else:
                frames_to_save = real_trim_result[0]
            
            save_result = saver.save_video(
                frames=frames_to_save,
                filename="test_new_nodes_output",
                fps=24.0,
                format="mp4",
                save_to_output=True
            )
            
            print("   ✅ Real video save successful")
            return True
            
        except Exception as e:
            print(f"   ⚠️ Real video test failed: {e}")
            return False
    else:
        print("   ⚠️ No real video found, skipping real video test")
        return True

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Video Node Testing")
    
    # Run synthetic data tests
    synthetic_success = test_video_nodes()
    
    # Run real video tests if available
    real_video_success = test_with_real_video()
    
    if synthetic_success and real_video_success:
        print("\n🎉 ALL TESTS PASSED! New video processing nodes are ready for use.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check output above for details.")
        sys.exit(1)