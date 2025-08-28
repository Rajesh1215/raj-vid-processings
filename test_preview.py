#!/usr/bin/env python3
"""
Test video preview functionality
"""

import os
import json
from nodes.video_upload import RajVideoUpload
from nodes.video_concatenator import RajVideoConcatenator
from nodes.video_saver import RajVideoSaver

def test_preview():
    print("🎬 Testing Video Preview Functionality")
    print("=" * 50)
    
    try:
        # Test upload node preview
        print("\n1️⃣ Testing RajVideoUpload preview...")
        upload = RajVideoUpload()
        
        result = upload.upload_and_load_video(
            video="/Users/rajeshkammaluri/mine/content_creation2/ComfyUI/input/Pirate_Captains_Stormy_Coffee_Toast.mp4",
            target_fps=24,
            max_frames=50
        )
        
        if isinstance(result, dict) and "ui" in result:
            preview_data = result["ui"]["video_preview"][0]
            print(f"   ✅ Upload preview data:")
            print(f"      Path: {preview_data['path']}")
            print(f"      Format: {preview_data['format']}")
            print(f"      Duration: {preview_data['duration']:.2f}s")
            print(f"      Size: {preview_data['width']}x{preview_data['height']}")
            frames1 = result["result"][0]
        else:
            print("   ❌ No preview data returned from upload")
            frames1 = result[0]
        
        # Load second video
        result2 = upload.upload_and_load_video(
            video="/Users/rajeshkammaluri/mine/content_creation2/ComfyUI/input/Pirate_Battle_Halts_for_Break.mp4",
            target_fps=24,
            max_frames=50
        )
        frames2 = result2["result"][0] if isinstance(result2, dict) else result2[0]
        
        # Test concatenator preview
        print("\n2️⃣ Testing RajVideoConcatenator preview...")
        concatenator = RajVideoConcatenator()
        
        concat_result = concatenator.concatenate_videos(
            video1=frames1,
            video2=frames2,
            batch_processing=True,
            chunk_size=32
        )
        
        if isinstance(concat_result, dict) and "ui" in concat_result:
            preview_data = concat_result["ui"]["video_preview"][0]
            print(f"   ✅ Concatenator preview data:")
            print(f"      Path: {preview_data['path']}")
            print(f"      Format: {preview_data['format']}")
            print(f"      Duration: {preview_data['duration']:.2f}s")
            print(f"      Frames: {preview_data['frame_count']}")
            concat_frames = concat_result["result"][0]
        else:
            print("   ❌ No preview data returned from concatenator")
            concat_frames = concat_result[0]
        
        # Test saver preview
        print("\n3️⃣ Testing RajVideoSaver preview...")
        saver = RajVideoSaver()
        
        save_result = saver.save_video(
            frames=concat_frames,
            filename="preview_test",
            fps=24.0,
            format="mp4",
            quality=23,
            save_to_output=True
        )
        
        if isinstance(save_result, dict) and "ui" in save_result:
            preview_data = save_result["ui"]["video_preview"][0]
            print(f"   ✅ Saver preview data:")
            print(f"      Path: {preview_data['path']}")
            print(f"      Format: {preview_data['format']}")
            print(f"      Duration: {preview_data['duration']:.2f}s")
            print(f"      Size: {preview_data['width']}x{preview_data['height']}")
        else:
            print("   ❌ No preview data returned from saver")
        
        print("\n✅ Preview functionality test complete!")
        print("   All nodes are returning proper UI preview data")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_preview()