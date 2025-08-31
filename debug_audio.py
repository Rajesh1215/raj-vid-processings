#!/usr/bin/env python3
"""
Debug script to check audio library availability in ComfyUI environment.
Run this to diagnose MoviePy and other audio library issues.
"""

import sys
import os

def debug_environment():
    """Debug the Python environment and audio library availability."""
    print("🔍 Audio Library Debug Information")
    print("=" * 50)
    
    # Python environment info
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {':'.join(sys.path[:3])}...")
    print()
    
    # Test MoviePy
    print("Testing MoviePy:")
    try:
        import moviepy
        print(f"✅ MoviePy installed: {moviepy.__version__}")
        print(f"   Path: {moviepy.__file__}")
        
        # Test specific imports
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip
            print("✅ VideoFileClip and AudioFileClip imported successfully")
        except ImportError as e:
            print(f"❌ Error importing VideoFileClip/AudioFileClip: {e}")
            
    except ImportError as e:
        print(f"❌ MoviePy not found: {e}")
        print("   Install with: pip install moviepy")
    print()
    
    # Test Librosa
    print("Testing Librosa:")
    try:
        import librosa
        print(f"✅ Librosa installed: {librosa.__version__}")
    except ImportError as e:
        print(f"❌ Librosa not found: {e}")
        print("   Install with: pip install librosa")
    print()
    
    # Test TorchAudio
    print("Testing TorchAudio:")
    try:
        import torchaudio
        print(f"✅ TorchAudio installed: {torchaudio.__version__}")
    except ImportError as e:
        print(f"❌ TorchAudio not found: {e}")
        print("   Install with: pip install torchaudio")
    print()
    
    # Test our AudioProcessor
    print("Testing AudioProcessor:")
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from nodes.audio_utils import AudioProcessor, MOVIEPY_AVAILABLE, LIBROSA_AVAILABLE, TORCHAUDIO_AVAILABLE
        print("✅ AudioProcessor imported successfully")
        print(f"   MoviePy available: {MOVIEPY_AVAILABLE}")
        print(f"   Librosa available: {LIBROSA_AVAILABLE}")
        print(f"   TorchAudio available: {TORCHAUDIO_AVAILABLE}")
        
        # Test creating an instance
        processor = AudioProcessor()
        print("✅ AudioProcessor instance created")
        
    except Exception as e:
        print(f"❌ Error importing AudioProcessor: {e}")
    print()
    
    # Installation recommendations
    print("📦 Installation Recommendations:")
    print("If any libraries are missing, install them with:")
    print(f"   {sys.executable} -m pip install moviepy librosa torchaudio")
    print("Or using the current directory's Python:")
    print("   pip install moviepy librosa torchaudio")
    

def test_audio_extraction():
    """Test audio extraction with a dummy video if available."""
    print("\n🎬 Testing Audio Extraction")
    print("=" * 30)
    
    try:
        from nodes.audio_utils import AudioProcessor
        import torch
        
        # Look for test video files (prioritize ones with audio)
        test_video_paths = [
            "../../../input_vids/sometalk.mp4",
            "../../../input_vids/vid5.mp4",
            "../../../input_vids/vid1.mp4",
            "../../input/test.mp4",
            "test_video.mp4"
        ]
        
        test_video = None
        for path in test_video_paths:
            if os.path.exists(path):
                test_video = path
                break
        
        if test_video:
            print(f"Found test video: {test_video}")
            try:
                audio_tensor, metadata = AudioProcessor.extract_audio_from_video(test_video)
                print(f"✅ Audio extraction successful!")
                print(f"   Audio shape: {audio_tensor.shape}")
                print(f"   Sample rate: {metadata['sample_rate']}")
                print(f"   Duration: {metadata['duration']:.2f}s")
            except Exception as e:
                print(f"❌ Audio extraction failed: {e}")
        else:
            print("No test video found, skipping extraction test")
            
    except Exception as e:
        print(f"❌ Could not test audio extraction: {e}")

if __name__ == "__main__":
    debug_environment()
    test_audio_extraction()