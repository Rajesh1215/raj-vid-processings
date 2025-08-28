#!/usr/bin/env python3
"""
Debug script to check tensor values in video processing pipeline
"""

import torch
import numpy as np
from nodes.video_upload import RajVideoUpload
from nodes.video_concatenator import RajVideoConcatenator
from nodes.video_saver import RajVideoSaver

def check_tensor_values(tensor, name):
    """Check tensor value ranges"""
    print(f"\n🔍 {name}:")
    print(f"   Shape: {tensor.shape}")
    print(f"   Device: {tensor.device}")
    print(f"   Dtype: {tensor.dtype}")
    print(f"   Min: {tensor.min().item():.6f}")
    print(f"   Max: {tensor.max().item():.6f}")
    print(f"   Mean: {tensor.mean().item():.6f}")
    
    # Check if values are reasonable for video frames
    if tensor.min() < -0.1 or tensor.max() > 1.1:
        print(f"   ⚠️ WARNING: Values outside expected [0,1] range!")
    if tensor.mean() < 0.1:
        print(f"   ⚠️ WARNING: Very low mean - might be too dark!")

def main():
    print("🐛 Debugging Tensor Values in Video Pipeline")
    print("=" * 50)
    
    try:
        # Test with the actual uploaded videos
        upload = RajVideoUpload()
        
        # Load first video
        print("📹 Loading first video...")
        frames1, info1, count1, fps1 = upload.upload_and_load_video(
            video="/Users/rajeshkammaluri/mine/content_creation2/ComfyUI/input/Pirate_Captains_Stormy_Coffee_Toast.mp4",
            target_fps=24,
            max_frames=50  # Limit for testing
        )
        check_tensor_values(frames1, "Video 1 after upload")
        
        # Load second video
        print("📹 Loading second video...")
        frames2, info2, count2, fps2 = upload.upload_and_load_video(
            video="/Users/rajeshkammaluri/mine/content_creation2/ComfyUI/input/Pirate_Battle_Halts_for_Break.mp4",
            target_fps=24,
            max_frames=50  # Limit for testing
        )
        check_tensor_values(frames2, "Video 2 after upload")
        
        # Test concatenation
        print("🔗 Testing concatenation...")
        concatenator = RajVideoConcatenator()
        concat_frames, concat_info, total_frames = concatenator.concatenate_videos(
            video1=frames1,
            video2=frames2,
            batch_processing=True,
            chunk_size=16  # Small for testing
        )
        check_tensor_values(concat_frames, "After concatenation")
        
        # Test saving
        print("💾 Testing save conversion...")
        saver = RajVideoSaver()
        
        # Check the conversion that happens in save
        frames_cpu = concat_frames.cpu().numpy()
        frames_uint8 = (frames_cpu * 255).astype(np.uint8)
        
        print(f"\n🔍 Save conversion:")
        print(f"   CPU numpy shape: {frames_cpu.shape}")
        print(f"   CPU numpy min: {frames_cpu.min():.6f}")
        print(f"   CPU numpy max: {frames_cpu.max():.6f}")
        print(f"   After *255 min: {(frames_cpu * 255).min():.1f}")
        print(f"   After *255 max: {(frames_cpu * 255).max():.1f}")
        print(f"   Final uint8 min: {frames_uint8.min()}")
        print(f"   Final uint8 max: {frames_uint8.max()}")
        
        if frames_uint8.max() < 10:
            print("❌ PROBLEM FOUND: Final uint8 values are too low!")
            print("   This will result in a very dark/black video")
        else:
            print("✅ Final uint8 values look reasonable")
        
        # Test actual video save
        print("💾 Testing actual video save...")
        output_path, save_info, total_frames, duration = saver.save_video(
            frames=concat_frames,
            filename="debug_test",
            fps=24.0,
            format="mp4",
            quality=23,
            save_to_output=True
        )
        
        print(f"📄 Save completed: {output_path}")
        print(f"ℹ️  Save info: {save_info}")
        
        # Check file size
        import os
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"📦 File size: {file_size / 1024:.1f} KB ({file_size / (1024*1024):.2f} MB)")
            
            if file_size < 100000:  # Less than 100KB is suspicious
                print("❌ WARNING: File size is very small - likely encoding issue!")
            else:
                print("✅ File size looks reasonable")
        else:
            print("❌ ERROR: Output file was not created!")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()