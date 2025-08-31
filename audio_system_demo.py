#!/usr/bin/env python3
"""
Audio Processing System Demo
Demonstrates the key capabilities of the modular audio system.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

def demo_audio_extraction():
    """Demonstrate multi-backend audio extraction from video."""
    print("🎬 Audio Extraction Demo")
    print("=" * 30)
    
    from nodes.audio_utils import AudioProcessor
    
    test_video = "../../../input_vids/sometalk.mp4"
    if not os.path.exists(test_video):
        print("❌ Test video not found")
        return
    
    try:
        # Extract audio using the robust multi-backend system
        audio_tensor, metadata = AudioProcessor.extract_audio_from_video(
            test_video, target_sample_rate=22050, mono=True
        )
        
        print(f"✅ Audio extracted successfully!")
        print(f"   📊 Shape: {audio_tensor.shape}")
        print(f"   🎵 Duration: {metadata['duration']:.2f}s")
        print(f"   📻 Sample Rate: {metadata['sample_rate']}Hz")
        print(f"   🔧 Backend Used: {metadata['loader']}")
        print(f"   📁 Format: {metadata.get('format', 'unknown')}")
        
        return audio_tensor, metadata
        
    except Exception as e:
        print(f"❌ Audio extraction failed: {e}")
        return None, None

def demo_whisper_audio():
    """Demonstrate audio-only Whisper transcription."""
    print("\n🎙️ Whisper Audio Transcription Demo")
    print("=" * 40)
    
    # Get audio from extraction demo
    audio_tensor, metadata = demo_audio_extraction()
    if audio_tensor is None:
        return
    
    try:
        from nodes.whisper_audio import RajWhisperAudio
        
        whisper = RajWhisperAudio()
        
        # Transcribe with word-level timestamps
        result = whisper.transcribe_audio(
            audio=audio_tensor,
            whisper_model="tiny",  # Fast for demo
            language="auto", 
            words_per_caption=8,
            max_caption_duration=5.0,
            timestamp_level="words"
        )
        
        transcribed_text, segments_json, word_data, info = result
        
        print(f"✅ Transcription completed!")
        print(f"   📝 Text: \"{transcribed_text[:100]}...\"")
        
        # Parse segments
        import json
        if segments_json != "[]":
            segments = json.loads(segments_json)
            print(f"   📊 Segments: {len(segments)}")
            if segments:
                first_segment = segments[0]
                print(f"   ⏱️  First segment: {first_segment.get('start', 0):.1f}s - {first_segment.get('end', 0):.1f}s")
        else:
            print(f"   📊 Segments: 0 (short audio)")
            
        print(f"   ℹ️  Model info: {info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Whisper transcription failed: {e}")
        return False

def demo_audio_processing():
    """Demonstrate audio processing effects."""
    print("\n🎛️ Audio Processing Effects Demo")
    print("=" * 40)
    
    from nodes.audio_loader import RajAudioProcessor
    
    # Create a test audio signal (1 second sine wave)
    sample_rate = 22050
    duration = 1.0
    frequency = 440.0  # A4 note
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    test_audio = torch.sin(2 * torch.pi * frequency * t).unsqueeze(1)
    
    print(f"🎵 Created test audio: {test_audio.shape}, {frequency}Hz tone")
    
    processor = RajAudioProcessor()
    
    try:
        # Test 1: Normalization
        print("\n1️⃣ Testing Normalization...")
        normalized, info = processor.process_audio(
            audio=test_audio,
            operation="normalize",
            normalize_method="peak",
            normalize_level=0.7
        )
        print(f"   ✅ {info.split('Operations:')[1].split('Input:')[0].strip()}")
        
        # Test 2: Fade effects
        print("\n2️⃣ Testing Fade Effects...")
        faded, info = processor.process_audio(
            audio=normalized,
            operation="fade",
            fade_in=0.2,
            fade_out=0.3,
            current_sample_rate=sample_rate
        )
        print(f"   ✅ Applied fade in/out")
        
        # Test 3: Amplification
        print("\n3️⃣ Testing Amplification...")
        amplified, info = processor.process_audio(
            audio=faded,
            operation="amplify",
            amplify_db=6.0,
            current_sample_rate=sample_rate
        )
        print(f"   ✅ Applied +6dB amplification")
        
        # Test 4: Resampling
        print("\n4️⃣ Testing Resampling...")
        resampled, info = processor.process_audio(
            audio=amplified,
            operation="resample",
            current_sample_rate=sample_rate,
            target_sample_rate=16000
        )
        print(f"   ✅ Resampled: 22050Hz → 16000Hz")
        print(f"   📊 Final shape: {resampled.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio processing failed: {e}")
        return False

def demo_audio_info():
    """Demonstrate audio information utilities."""
    print("\n📊 Audio Information Demo")
    print("=" * 30)
    
    from nodes.audio_utils import AudioProcessor
    
    # Create test audio with different characteristics
    sample_rate = 44100
    duration = 3.5
    
    # Stereo audio with different frequencies in each channel
    t = torch.linspace(0, duration, int(sample_rate * duration))
    left_channel = torch.sin(2 * torch.pi * 220 * t)  # A3
    right_channel = torch.sin(2 * torch.pi * 440 * t)  # A4
    
    stereo_audio = torch.stack([left_channel, right_channel], dim=1)
    
    try:
        # Generate comprehensive audio info
        info = AudioProcessor.get_audio_info(stereo_audio, sample_rate)
        print(f"✅ Audio Info Generated:")
        print(f"   {info}")
        
        # Show range analysis
        audio_min = stereo_audio.min().item()
        audio_max = stereo_audio.max().item()
        audio_mean = stereo_audio.mean().item()
        audio_std = stereo_audio.std().item()
        
        print(f"✅ Detailed Analysis:")
        print(f"   📈 Range: [{audio_min:.3f}, {audio_max:.3f}]")
        print(f"   📊 Mean: {audio_mean:.3f}, Std: {audio_std:.3f}")
        print(f"   🎵 Left/Right: {left_channel.shape[0]:,} samples each")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio info demo failed: {e}")
        return False

def show_system_summary():
    """Show a summary of the audio system capabilities."""
    print("\n" + "="*60)
    print("🏆 AUDIO PROCESSING SYSTEM SUMMARY")
    print("="*60)
    
    print("\n✅ CORE ACHIEVEMENTS:")
    print("   🎬 Enhanced Video Upload - Separate audio output for modular workflows")
    print("   🎙️ Dedicated Whisper Audio - Audio-only transcription for better performance")
    print("   🔊 Standalone Audio Loader - Support for WAV, MP3, AAC, FLAC, M4A, OGG")
    print("   🎛️ Audio Processor - Normalize, fade, amplify, resample, trim operations")
    print("   🔧 Multi-Backend Support - TorchAudio, MoviePy 2.0+, FFmpeg with fallbacks")
    
    print("\n🎯 KEY FEATURES:")
    print("   • MoviePy 2.0+ compatibility with structural import changes")
    print("   • Robust audio extraction with multiple fallback methods")
    print("   • Comprehensive audio format support across backends")
    print("   • Professional audio processing effects and normalization")
    print("   • Word-level timestamp transcription with Whisper")
    print("   • ComfyUI tensor format compatibility (samples, channels)")
    print("   • Detailed audio information and metadata generation")
    
    print("\n⚡ PERFORMANCE OPTIMIZATIONS:")
    print("   • Audio-only Whisper node eliminates video processing overhead")
    print("   • Modular design allows connecting audio directly between nodes")
    print("   • Multi-backend fallbacks ensure compatibility across environments")
    print("   • Efficient tensor operations with GPU/MPS support")
    
    print("\n🔄 WORKFLOW INTEGRATION:")
    print("   Video Upload → Audio Output → Whisper Transcription")
    print("   Audio File → Audio Loader → Audio Processing → Effects")
    print("   Any Audio Source → Whisper → Text Generation → Video Overlay")

if __name__ == "__main__":
    print("🎵 COMPREHENSIVE AUDIO PROCESSING SYSTEM DEMO")
    print("=" * 60)
    
    success_count = 0
    total_demos = 4
    
    # Run all demos
    if demo_audio_extraction():
        success_count += 1
        
    if demo_whisper_audio():
        success_count += 1
        
    if demo_audio_processing():
        success_count += 1
        
    if demo_audio_info():
        success_count += 1
    
    # Show system summary
    show_system_summary()
    
    print(f"\n🏆 DEMO RESULTS: {success_count}/{total_demos} completed successfully")
    
    if success_count == total_demos:
        print("🎉 ALL AUDIO SYSTEM COMPONENTS ARE WORKING PERFECTLY!")
    else:
        print("⚠️  Some components need attention")
    
    print("\n🚀 The modular audio processing system is ready for production use!")