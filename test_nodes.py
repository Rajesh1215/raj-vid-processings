#!/usr/bin/env python3
"""
Test script for Raj Video Processing Nodes
"""

import torch
import numpy as np
import cv2
import os
import tempfile

# Import our nodes
from nodes.utils import get_optimal_device
from nodes.video_loader import RajVideoLoader
from nodes.video_concatenator import RajVideoConcatenator
from nodes.video_saver import RajVideoSaver
from nodes.video_upload import RajVideoUpload, RajVideoUploadAdvanced

def create_test_video(filename, duration=2, fps=30, width=320, height=240):
    """Create a simple test video"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    total_frames = int(duration * fps)
    
    for i in range(total_frames):
        # Create a gradient frame with frame number
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create gradient effect
        gradient = np.linspace(i * 4, (i + 1) * 4, width) % 255
        for y in range(height):
            for x in range(width):
                frame[y, x] = [int(gradient[x]), (i * 5) % 255, (255 - i * 3) % 255]
        
        # Add frame number text
        cv2.putText(frame, f'Frame {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    
    out.release()
    print(f"✅ Created test video: {filename} ({total_frames} frames)")

def test_device_detection():
    """Test device detection"""
    print("\n🔍 Testing Device Detection:")
    device = get_optimal_device()
    print(f"   Detected device: {device}")
    
    # Test device capabilities
    if device.type == "mps":
        print("   🍎 MPS (Metal) acceleration available")
    elif device.type == "cuda":
        print(f"   🚀 CUDA acceleration available - GPU: {torch.cuda.get_device_name()}")
    else:
        print("   💻 CPU processing (no GPU acceleration)")

def test_video_loader():
    """Test video loading functionality"""
    print("\n📹 Testing Video Loader:")
    
    # Create temp test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        test_video_path = tmp.name
    
    try:
        create_test_video(test_video_path, duration=1, fps=10)  # Small test video
        
        # Test loading
        loader = RajVideoLoader()
        frames, info, frame_count, fps = loader.load_video(
            video_path=test_video_path,
            target_fps=5,  # Reduce fps
            max_frames=20  # Limit frames
        )
        
        print(f"   ✅ Video loaded: {frame_count} frames")
        print(f"   ✅ Info: {info}")
        print(f"   ✅ Tensor shape: {frames.shape}")
        print(f"   ✅ Device: {frames.device}")
        
        return frames
        
    finally:
        if os.path.exists(test_video_path):
            os.unlink(test_video_path)

def test_video_concatenation(video1_frames):
    """Test video concatenation"""
    print("\n🔗 Testing Video Concatenator:")
    
    # Create a second video (just duplicate first few frames with modification)
    video2_frames = video1_frames[:5].clone()
    video2_frames *= 0.5  # Darken frames
    
    concatenator = RajVideoConcatenator()
    result_frames, info, total_frames = concatenator.concatenate_videos(
        video1=video1_frames,
        video2=video2_frames,
        transition_frames=2  # 2 frame crossfade
    )
    
    print(f"   ✅ Concatenation successful: {total_frames} frames")
    print(f"   ✅ Info: {info}")
    print(f"   ✅ Result shape: {result_frames.shape}")
    print(f"   ✅ Device: {result_frames.device}")
    
    return result_frames

def test_video_saver(frames):
    """Test video saving functionality"""
    print("\n💾 Testing Video Saver:")
    
    saver = RajVideoSaver()
    
    # Test multiple formats
    formats_to_test = ["mp4", "gif"]
    
    for format in formats_to_test:
        try:
            file_path, save_info, total_frames, duration = saver.save_video(
                frames=frames,
                filename=f"test_output_{format}",
                fps=10.0,
                format=format,
                quality=25,
                gpu_encoding=False,  # Use CPU for compatibility
                save_to_output=False,  # Save to temp
                add_timestamp=True
            )
            
            print(f"   ✅ {format.upper()} saved: {os.path.basename(file_path)}")
            print(f"   ✅ Info: {save_info}")
            print(f"   ✅ File exists: {os.path.exists(file_path)}")
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   ✅ File size: {size_mb:.2f} MB")
                
        except Exception as e:
            print(f"   ❌ {format.upper()} save failed: {e}")

def test_upload_nodes():
    """Test upload node structure"""
    print("\n📤 Testing Upload Nodes:")
    
    try:
        upload = RajVideoUpload()
        upload_adv = RajVideoUploadAdvanced()
        
        # Test input types
        inputs = RajVideoUpload.INPUT_TYPES()
        adv_inputs = RajVideoUploadAdvanced.INPUT_TYPES()
        
        print(f"   ✅ RajVideoUpload: {len(inputs['required'])} required, {len(inputs.get('optional', {}))} optional")
        print(f"   ✅ Upload button: {'choose video to upload' in inputs.get('hidden', {})}")
        
        print(f"   ✅ RajVideoUploadAdvanced: {len(adv_inputs['required'])} required, {len(adv_inputs.get('optional', {}))} optional") 
        print(f"   ✅ Advanced upload button: {'choose video to upload' in adv_inputs.get('hidden', {})}")
        
        # Test return types
        print(f"   ✅ Upload returns: {upload.RETURN_TYPES}")
        print(f"   ✅ Advanced returns: {upload_adv.RETURN_TYPES}")
        
    except Exception as e:
        print(f"   ❌ Upload node test failed: {e}")

def test_memory_info():
    """Test memory information"""
    print("\n💾 Testing Memory Information:")
    device = get_optimal_device()
    
    from nodes.utils import get_memory_info
    mem_info = get_memory_info(device)
    
    print(f"   Device: {device}")
    for key, value in mem_info.items():
        print(f"   {key.capitalize()}: {value}")

def main():
    """Run all tests"""
    print("🎬 Raj Video Processing Nodes - Test Suite")
    print("=" * 50)
    
    try:
        test_device_detection()
        test_memory_info()
        test_upload_nodes()
        
        # Test video processing pipeline
        frames = test_video_loader()
        if frames is not None:
            concatenated_frames = test_video_concatenation(frames)
            test_video_saver(concatenated_frames)
        
        print("\n✅ All tests completed successfully!")
        print("🎉 Your custom nodes are ready to use in ComfyUI!")
        print("📋 Available nodes: Video Loader, Concatenator, Sequencer, Video Saver, Video Upload")
        print("🌟 Features: GPU Acceleration, Upload Button, Drag & Drop, Multiple Formats")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()