#!/usr/bin/env python3
"""
Word-Level Caption Demo for RajWhisperProcess
Demonstrates the detailed word timing capabilities for frame-accurate video captions.
"""

import os
import sys
import torch
import json

sys.path.insert(0, os.path.dirname(__file__))

def demonstrate_word_level_output():
    """Demonstrate the detailed word-level caption output format."""
    print("📝 WORD-LEVEL CAPTION OUTPUT DEMONSTRATION")
    print("=" * 60)
    
    try:
        from nodes.whisper_process import RajWhisperProcess
        from nodes.audio_utils import AudioProcessor
        
        # Extract audio from test video
        test_video = "../../../input_vids/sometalk.mp4"
        if not os.path.exists(test_video):
            print("❌ Test video not found")
            return False
        
        print(f"📹 Processing: {os.path.basename(test_video)}")
        
        # Extract and limit audio for demo
        audio_tensor, metadata = AudioProcessor.extract_audio_from_video(test_video)
        
        # Use first 8 seconds for detailed analysis
        max_samples = int(8.0 * metadata['sample_rate'])
        if audio_tensor.shape[0] > max_samples:
            audio_tensor = audio_tensor[:max_samples]
        
        print(f"   🎵 Audio: {audio_tensor.shape[0] / metadata['sample_rate']:.1f}s @ {metadata['sample_rate']}Hz")
        
        # Run transcription
        whisper_node = RajWhisperProcess()
        
        result = whisper_node.transcribe_audio(
            audio=audio_tensor,
            whisper_model="tiny",
            language="auto",
            words_per_caption=4,  # Smaller chunks for better demo
            max_caption_duration=3.0,
            timestamp_level="word",
            normalize_audio=True,
            confidence_threshold=0.1  # Lower threshold to show more words
        )
        
        sentence_captions, word_captions, full_transcript, word_timings, info = result
        
        print(f"\n📝 FULL TRANSCRIPT:")
        print(f'   "{full_transcript}"')
        print(f"   📊 Length: {len(full_transcript)} characters, {len(full_transcript.split())} words")
        
        print(f"\n📄 SENTENCE-LEVEL CAPTIONS (Traditional Format):")
        print("   Format: [start-end] text")
        try:
            sentences = json.loads(sentence_captions)
            for i, sentence in enumerate(sentences, 1):
                start = sentence['start']
                end = sentence['end']
                text = sentence['text']
                word_count = sentence['word_count']
                print(f"   {i:2d}. [{start:6.2f}s - {end:6.2f}s] \"{text}\" ({word_count} words)")
        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON Error: {e}")
        
        print(f"\n📝 WORD-LEVEL CAPTIONS (NEW FEATURE):")
        print("   Format: [start-end] word (confidence, duration)")
        try:
            words = json.loads(word_captions)
            for i, word in enumerate(words, 1):
                start = word['start']
                end = word['end']
                text = word['word']
                confidence = word['confidence']
                duration = word['duration']
                segment_id = word.get('segment_id', i-1)
                print(f"   {i:2d}. [{start:7.3f}s - {end:7.3f}s] \"{text:<12}\" (conf: {confidence:.3f}, dur: {duration:.3f}s, seg: {segment_id})")
        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON Error: {e}")
        
        print(f"\n⏱️ WORD TIMINGS ARRAY (List Format):")
        print("   Format: Raw timing data for programmatic use")
        for i, timing in enumerate(word_timings[:12], 1):  # Show first 12 for demo
            word = timing['word']
            start = timing['start']
            end = timing['end']
            duration = timing['duration']
            confidence = timing['confidence']
            print(f"   {i:2d}. {{'word': '{word:<12}', 'start': {start:7.3f}, 'end': {end:7.3f}, 'duration': {duration:.3f}, 'confidence': {confidence:.3f}}}")
        
        if len(word_timings) > 12:
            print(f"       ... and {len(word_timings) - 12} more timing entries")
        
        print(f"\n📊 TIMING ANALYSIS:")
        if word_timings:
            # Calculate statistics
            durations = [w['duration'] for w in word_timings]
            confidences = [w['confidence'] for w in word_timings]
            
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            
            print(f"   📏 Duration Stats:")
            print(f"      Average: {avg_duration:.3f}s")
            print(f"      Range: {min_duration:.3f}s - {max_duration:.3f}s")
            print(f"      Total: {sum(durations):.2f}s")
            
            print(f"   🎯 Confidence Stats:")
            print(f"      Average: {avg_confidence:.3f}")
            print(f"      Minimum: {min_confidence:.3f}")
            print(f"      High confidence words: {sum(1 for c in confidences if c > 0.8)}/{len(confidences)}")
        
        print(f"\n🔧 USE CASES FOR WORD-LEVEL TIMING:")
        print("   1️⃣ Frame-accurate text animation")
        print("   2️⃣ Karaoke-style word highlighting")
        print("   3️⃣ Precise subtitle synchronization")
        print("   4️⃣ Quality control (confidence-based filtering)")
        print("   5️⃣ Advanced text effects (per-word styling)")
        print("   6️⃣ Speech analysis and linguistics research")
        
        return True
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_json_structure_examples():
    """Show the detailed JSON structure of both output formats."""
    print("\n📋 JSON STRUCTURE EXAMPLES")
    print("=" * 40)
    
    print("\n1️⃣ SENTENCE-LEVEL CAPTIONS JSON:")
    sentence_example = [
        {
            "start": 0.00,
            "end": 2.72,
            "text": "Bangalore is a silicon valley of",
            "word_count": 6
        },
        {
            "start": 2.72,
            "end": 5.58,
            "text": "India, known for its thriving technology",
            "word_count": 6
        }
    ]
    
    print(json.dumps(sentence_example, indent=4))
    
    print("\n2️⃣ WORD-LEVEL CAPTIONS JSON:")
    word_example = [
        {
            "start": 0.000,
            "end": 1.240,
            "word": "Bangalore",
            "confidence": 0.912,
            "segment_id": 0,
            "duration": 1.240
        },
        {
            "start": 1.240,
            "end": 1.580,
            "word": "is",
            "confidence": 0.987,
            "segment_id": 1,
            "duration": 0.340
        },
        {
            "start": 1.580,
            "end": 1.700,
            "word": "a",
            "confidence": 0.681,
            "segment_id": 2,
            "duration": 0.120
        }
    ]
    
    print(json.dumps(word_example, indent=4))
    
    print("\n3️⃣ WORD TIMINGS ARRAY (List):")
    timing_example = [
        {
            "word": "Bangalore",
            "start": 0.000,
            "end": 1.240,
            "duration": 1.240,
            "confidence": 0.912,
            "char_start": None,
            "char_end": None
        },
        {
            "word": "is",
            "start": 1.240,
            "end": 1.580,
            "duration": 0.340,
            "confidence": 0.987,
            "char_start": None,
            "char_end": None
        }
    ]
    
    for timing in timing_example:
        print(timing)

def compare_old_vs_new():
    """Compare the old vs new output format."""
    print("\n🔄 OLD vs NEW COMPARISON")
    print("=" * 35)
    
    print("\n❌ OLD RajWhisperAudio Output (4 values):")
    print("   1. caption_data: Sentence-level only")
    print("   2. full_transcript: Complete text")
    print("   3. timestamps: Basic segment timing")
    print("   4. transcription_info: Basic metadata")
    
    print("\n✅ NEW RajWhisperProcess Output (5 values):")
    print("   1. sentence_captions: Enhanced sentence segments")
    print("   2. word_captions: NEW - Individual word timing")
    print("   3. full_transcript: Complete text (unchanged)")
    print("   4. word_timings: NEW - Detailed timing array")
    print("   5. transcription_info: Enhanced metadata")
    
    print("\n🆕 KEY IMPROVEMENTS:")
    print("   • Word-level precision timing")
    print("   • Per-word confidence scores")
    print("   • Dual output format (JSON + List)")
    print("   • Enhanced metadata and statistics")
    print("   • Better error handling and edge cases")
    print("   • Advanced audio preprocessing options")

if __name__ == "__main__":
    print("📝 RAJWHISPERPROCESS WORD-LEVEL CAPTION DEMO")
    print("=" * 60)
    
    # Run demonstration
    if demonstrate_word_level_output():
        print("\n✅ Word-level caption demonstration completed successfully!")
    
    # Show JSON examples
    show_json_structure_examples()
    
    # Show comparison
    compare_old_vs_new()
    
    print("\n🏆 SUMMARY:")
    print("✅ RajWhisperProcess now provides:")
    print("   📝 Sentence-level captions (traditional)")
    print("   📝 Word-level captions (NEW)")
    print("   ⏱️ Precise word timings (NEW)")
    print("   🎯 Confidence scoring (NEW)")
    print("   📊 Enhanced metadata (NEW)")
    
    print("\n🚀 Ready for advanced video caption workflows!")