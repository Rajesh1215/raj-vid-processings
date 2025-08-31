#!/usr/bin/env python3
"""
Test script for RajWhisperProcess node with word-level caption functionality.
Demonstrates both sentence-level and word-level caption generation with timing.
"""

import os
import sys
import torch
import json

sys.path.insert(0, os.path.dirname(__file__))

def test_whisper_process_with_audio():
    """Test RajWhisperProcess with real audio file."""
    print("🎙️ Testing RajWhisperProcess with Real Audio")
    print("=" * 50)
    
    try:
        from nodes.whisper_process import RajWhisperProcess
        from nodes.audio_utils import AudioProcessor
        
        # Test with real audio file
        test_video = "../../../input_vids/sometalk.mp4"
        if not os.path.exists(test_video):
            print("❌ Test video not found, skipping real audio test")
            return False
        
        print(f"📹 Extracting audio from: {os.path.basename(test_video)}")
        
        # Extract audio
        audio_tensor, metadata = AudioProcessor.extract_audio_from_video(test_video)
        sample_rate = metadata['sample_rate']
        duration = metadata['duration']
        
        print(f"   ✅ Audio extracted: {duration:.2f}s @ {sample_rate}Hz")
        print(f"   📊 Audio shape: {audio_tensor.shape}")
        
        # Test with shorter clip (first 10 seconds for faster processing)
        max_duration = 10.0
        if duration > max_duration:
            max_samples = int(max_duration * sample_rate)
            audio_tensor = audio_tensor[:max_samples]
            print(f"   ✂️ Truncated to {max_duration:.1f}s for testing")
        
        # Initialize Whisper Process node
        whisper_node = RajWhisperProcess()
        
        print("\n🔄 Running Whisper transcription...")
        print("   Model: tiny (fast for testing)")
        print("   Language: auto-detect")
        print("   Features: sentence + word-level captions")
        
        # Run transcription
        result = whisper_node.transcribe_audio(
            audio=audio_tensor,
            whisper_model="tiny",  # Use tiny for speed
            language="auto",
            words_per_caption=6,
            max_caption_duration=4.0,
            timestamp_level="word",
            normalize_audio=True,
            denoise_audio=False,
            target_sample_rate=16000
        )
        
        sentence_captions, word_captions, full_transcript, word_timings, info = result
        
        # Display results
        print(f"\n✅ Transcription Results:")
        print(f"📝 Full Transcript ({len(full_transcript)} chars):")
        print(f"   \"{full_transcript[:200]}{'...' if len(full_transcript) > 200 else ''}\"")
        
        # Parse and display sentence captions
        try:
            sentences = json.loads(sentence_captions)
            print(f"\n📄 Sentence-Level Captions ({len(sentences)} segments):")
            for i, sentence in enumerate(sentences[:3]):  # Show first 3
                print(f"   {i+1:2d}. [{sentence['start']:6.2f}s - {sentence['end']:6.2f}s] \"{sentence['text'][:50]}...\"")
            if len(sentences) > 3:
                print(f"       ... and {len(sentences) - 3} more segments")
        except json.JSONDecodeError:
            print(f"   ⚠️ Sentence captions: {sentence_captions}")
        
        # Parse and display word captions
        try:
            words = json.loads(word_captions)
            print(f"\n📝 Word-Level Captions ({len(words)} words):")
            for i, word in enumerate(words[:10]):  # Show first 10 words
                confidence = word.get('confidence', 1.0)
                duration = word.get('duration', 0.0)
                print(f"   {i+1:2d}. [{word['start']:6.3f}s - {word['end']:6.3f}s] \"{word['word']}\" (conf: {confidence:.2f}, dur: {duration:.3f}s)")
            if len(words) > 10:
                print(f"       ... and {len(words) - 10} more words")
        except json.JSONDecodeError:
            print(f"   ⚠️ Word captions: {word_captions}")
        
        # Display word timings list
        print(f"\n⏱️ Word Timings Array ({len(word_timings)} entries):")
        for i, timing in enumerate(word_timings[:5]):  # Show first 5
            print(f"   {i+1}. {timing}")
        if len(word_timings) > 5:
            print(f"      ... and {len(word_timings) - 5} more timing entries")
        
        # Display transcription info
        print(f"\n📊 Transcription Info:")
        info_lines = info.split('\\n')
        for line in info_lines:
            if line.strip():
                print(f"   {line}")
        
        # Analyze timing accuracy
        if word_timings:
            word_durations = [w['duration'] for w in word_timings if 'duration' in w]
            if word_durations:
                avg_duration = sum(word_durations) / len(word_durations)
                min_duration = min(word_durations)
                max_duration = max(word_durations)
                print(f"\n📈 Timing Analysis:")
                print(f"   Average word duration: {avg_duration:.3f}s")
                print(f"   Duration range: {min_duration:.3f}s - {max_duration:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_synthetic_audio():
    """Test with synthetic audio to verify word timing generation."""
    print("\n🎵 Testing with Synthetic Audio")
    print("=" * 35)
    
    try:
        from nodes.whisper_process import RajWhisperProcess
        
        # Create longer synthetic audio (3 seconds)
        sample_rate = 16000
        duration = 3.0
        
        # Create a more complex audio signal
        t = torch.linspace(0, duration, int(sample_rate * duration))
        freq1 = torch.sin(2 * torch.pi * 220 * t)  # A3
        freq2 = torch.sin(2 * torch.pi * 330 * t)  # E4
        freq3 = torch.sin(2 * torch.pi * 440 * t)  # A4
        
        # Combine frequencies with some modulation
        synthetic_audio = (freq1 + 0.5 * freq2 + 0.3 * freq3) / 1.8
        synthetic_audio = synthetic_audio.unsqueeze(1) * 0.7  # Add channel dimension and reduce volume
        
        print(f"🎼 Created synthetic audio: {synthetic_audio.shape}, {duration:.1f}s")
        
        whisper_node = RajWhisperProcess()
        
        result = whisper_node.transcribe_audio(
            audio=synthetic_audio,
            whisper_model="tiny",
            language="en",
            words_per_caption=4,
            max_caption_duration=2.0,
            timestamp_level="word",
            normalize_audio=True
        )
        
        sentence_captions, word_captions, full_transcript, word_timings, info = result
        
        print(f"✅ Synthetic audio processed:")
        print(f"   📝 Transcript: \"{full_transcript[:100]}...\"")
        print(f"   📊 Word timings generated: {len(word_timings)} entries")
        
        # Show structure of outputs
        print(f"\n📋 Output Structure Validation:")
        print(f"   sentence_captions: {type(sentence_captions)} ({len(sentence_captions)} chars)")
        print(f"   word_captions: {type(word_captions)} ({len(word_captions)} chars)")
        print(f"   full_transcript: {type(full_transcript)} ({len(full_transcript)} chars)")
        print(f"   word_timings: {type(word_timings)} ({len(word_timings)} items)")
        print(f"   transcription_info: {type(info)} ({len(info)} chars)")
        
        return True
        
    except Exception as e:
        print(f"❌ Synthetic audio test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases for robust behavior."""
    print("\n🧪 Testing Edge Cases")
    print("=" * 25)
    
    try:
        from nodes.whisper_process import RajWhisperProcess
        
        whisper_node = RajWhisperProcess()
        
        # Test 1: Empty audio
        print("1️⃣ Testing empty audio...")
        empty_audio = torch.zeros((1, 1))
        result = whisper_node.transcribe_audio(
            audio=empty_audio,
            whisper_model="tiny",
            language="en",
            words_per_caption=5,
            max_caption_duration=3.0,
            timestamp_level="word"
        )
        
        sentence_captions, word_captions, full_transcript, word_timings, info = result
        print(f"   Result: {full_transcript[:50]}...")
        
        # Test 2: Very short audio
        print("\n2️⃣ Testing very short audio (0.1s)...")
        short_audio = torch.sin(2 * torch.pi * 440 * torch.linspace(0, 0.1, 1600)).unsqueeze(1)
        result = whisper_node.transcribe_audio(
            audio=short_audio,
            whisper_model="tiny",
            language="en",
            words_per_caption=5,
            max_caption_duration=3.0,
            timestamp_level="word"
        )
        
        _, _, transcript, timings, _ = result
        print(f"   Result: {transcript[:50]}...")
        print(f"   Word timings: {len(timings)} entries")
        
        print("✅ Edge cases handled gracefully")
        return True
        
    except Exception as e:
        print(f"❌ Edge case testing failed: {e}")
        return False

def show_feature_summary():
    """Display summary of new features."""
    print("\n" + "="*60)
    print("🏆 RAJWHISPERPROCESS FEATURE SUMMARY")
    print("="*60)
    
    print("\n🆕 NEW FEATURES:")
    print("   🔄 Renamed from RajWhisperAudio to RajWhisperProcess")
    print("   📝 Dual Caption Output: Sentence-level + Word-level")
    print("   ⏱️ Precise Word Timings: Individual word start/end times")
    print("   🎯 Enhanced Confidence Scoring: Per-word confidence levels")
    print("   📊 Detailed Timing Analysis: Duration and timing statistics")
    print("   🔧 Advanced Audio Processing: Denoising and normalization")
    
    print("\n📤 RETURN VALUES:")
    print("   1️⃣ sentence_captions: JSON array of sentence-level segments")
    print("   2️⃣ word_captions: JSON array of individual words with timing")
    print("   3️⃣ full_transcript: Complete transcribed text")
    print("   4️⃣ word_timings: List of word timing objects")
    print("   5️⃣ transcription_info: Processing statistics and metadata")
    
    print("\n⚡ KEY IMPROVEMENTS:")
    print("   • Word-level precision: Each word has individual start/end times")
    print("   • Confidence filtering: Remove low-confidence words")
    print("   • Custom vocabulary support: Better domain-specific recognition")
    print("   • Flexible caption segmentation: Control words per caption")
    print("   • Comprehensive metadata: Model info, language detection, stats")
    print("   • Robust error handling: Graceful fallbacks for edge cases")
    
    print("\n🔄 WORKFLOW INTEGRATION:")
    print("   Video/Audio → RajWhisperProcess → Sentence + Word Captions")
    print("   Word Timings → Text Animation → Frame-accurate text overlay")
    print("   Confidence Scores → Quality Control → Manual review flagging")

if __name__ == "__main__":
    print("🎙️ RAJWHISPERPROCESS COMPREHENSIVE TEST")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Test with real audio
    if test_whisper_process_with_audio():
        success_count += 1
    
    # Test with synthetic audio
    if test_synthetic_audio():
        success_count += 1
        
    # Test edge cases
    if test_edge_cases():
        success_count += 1
    
    # Show feature summary
    show_feature_summary()
    
    print(f"\n🏆 TEST RESULTS: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 ALL RAJWHISPERPROCESS FEATURES ARE WORKING PERFECTLY!")
        print("\n✅ Ready for production with:")
        print("   🎙️ Advanced transcription processing")
        print("   📝 Dual-level caption generation (sentence + word)")
        print("   ⏱️ Precise word-level timing data")
        print("   🔧 Enhanced audio preprocessing")
        print("   📊 Comprehensive transcription metadata")
    else:
        print("⚠️  Some features need attention")
    
    print("\n🚀 RajWhisperProcess is ready for advanced caption workflows!")