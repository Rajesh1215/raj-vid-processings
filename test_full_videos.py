#!/usr/bin/env python3
"""
Test with full-length videos to reproduce the black video issue
"""

import torch
import numpy as np
from nodes.video_upload import RajVideoUpload
from nodes.video_concatenator import RajVideoConcatenator
from nodes.video_saver import RajVideoSaver

def main():
    print("🎬 Testing Full-Length Video Concatenation")
    print("=" * 50)
    
    try:
        upload = RajVideoUpload()
        
        print("📹 Loading FULL first video (no frame limit)...")
        frames1, info1, count1, fps1 = upload.upload_and_load_video(
            video="/Users/rajeshkammaluri/mine/content_creation2/ComfyUI/input/Pirate_Captains_Stormy_Coffee_Toast.mp4",
            target_fps=24,
            max_frames=0  # No limit - full video
        )
        print(f"   Video 1: {frames1.shape[0]} frames, mean: {frames1.mean().item():.6f}")
        
        print("📹 Loading FULL second video (no frame limit)...")
        frames2, info2, count2, fps2 = upload.upload_and_load_video(
            video="/Users/rajeshkammaluri/mine/content_creation2/ComfyUI/input/Pirate_Battle_Halts_for_Break.mp4",
            target_fps=24,
            max_frames=0  # No limit - full video
        )
        print(f"   Video 2: {frames2.shape[0]} frames, mean: {frames2.mean().item():.6f}")
        
        print("🔗 Concatenating full videos...")
        concatenator = RajVideoConcatenator()
        concat_frames, concat_info, total_frames = concatenator.concatenate_videos(
            video1=frames1,
            video2=frames2,
            batch_processing=True,
            chunk_size=32  # Same settings as before
        )
        print(f"   Concatenated: {concat_frames.shape[0]} frames, mean: {concat_frames.mean().item():.6f}")
        
        print("💾 Saving full concatenated video...")
        saver = RajVideoSaver()
        output_path, save_info, total_frames, duration = saver.save_video(
            frames=concat_frames,
            filename="full_test_concat",
            fps=24.0,
            format="mp4",
            quality=23,
            save_to_output=True
        )
        
        print(f"📄 Saved: {output_path}")
        print(f"ℹ️  Info: {save_info}")
        
        # Check file size
        import os
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"📦 File size: {file_size / 1024:.1f} KB ({file_size / (1024*1024):.2f} MB)")
            
            if file_size < 100000:
                print("❌ PROBLEM: File size too small - this will be black!")
            else:
                print("✅ File size looks good")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()