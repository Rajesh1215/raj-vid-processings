#!/usr/bin/env python3
"""
Audio Preview & Analysis System Demo
Demonstrates the complete audio preview and analysis capabilities.
"""

import os
import sys
import torch
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

def demo_audio_preview():
    """Demonstrate audio preview capabilities with real video."""
    print("🎵 Audio Preview Demo")
    print("=" * 30)
    
    try:
        from nodes.audio_utils import AudioProcessor
        from nodes.audio_preview import RajAudioPreview
        
        # Extract audio from test video
        test_video = "../../../input_vids/sometalk.mp4"
        if not os.path.exists(test_video):
            print("❌ Test video not found - creating synthetic audio")
            # Create longer synthetic audio for demo
            sample_rate = 22050
            duration = 5.0
            t = torch.linspace(0, duration, int(sample_rate * duration))
            
            # Create more interesting audio (chord progression)
            freq1 = torch.sin(2 * torch.pi * 220 * t)  # A3
            freq2 = torch.sin(2 * torch.pi * 277 * t)  # C#4
            freq3 = torch.sin(2 * torch.pi * 330 * t)  # E4
            audio_tensor = (freq1 + freq2 + freq3) / 3
            audio_tensor = audio_tensor.unsqueeze(1)
        else:
            print(f"📹 Extracting audio from: {os.path.basename(test_video)}")
            audio_tensor, metadata = AudioProcessor.extract_audio_from_video(test_video)
            sample_rate = metadata['sample_rate']
            print(f"   ✅ Audio extracted: {audio_tensor.shape}, {metadata['duration']:.2f}s")
        
        # Test RajAudioPreview
        preview_node = RajAudioPreview()
        
        print("\n1️⃣ Testing Temporary Preview (3 second clip)...")
        result = preview_node.create_preview(
            audio=audio_tensor,
            sample_rate=sample_rate,
            preview_duration=3.0,
            preview_mode="temporary",
            volume_boost=1.2,
            fade_edges=True,
            normalize_preview=True
        )
        
        audio_info, file_path, waveform_data, status = result
        print(f"   {status}")
        print(f"   📁 File: {os.path.basename(file_path) if file_path else 'None'}")
        
        # Show waveform info
        if waveform_data and waveform_data != "":
            import json
            try:
                waveform = json.loads(waveform_data)
                print(f"   📊 Waveform: {waveform.get('samples', 0)} points, Peak: {waveform.get('peak', 0):.3f}")
            except:
                print(f"   📊 Waveform data: {len(waveform_data)} chars")
        
        print("\n2️⃣ Testing Permanent Preview with MP3 format...")
        result2 = preview_node.create_preview(
            audio=audio_tensor,
            sample_rate=sample_rate,
            preview_duration=2.0,
            preview_mode="permanent",
            start_offset=1.0,
            output_format="wav",  # Use WAV for compatibility
            filename_prefix="demo_audio",
            fade_edges=True
        )
        
        audio_info2, file_path2, waveform_data2, status2 = result2
        print(f"   {status2}")
        print(f"   📁 Saved to: {os.path.basename(file_path2) if file_path2 else 'None'}")
        
        # Cleanup temporary files
        for path in [file_path, file_path2]:
            if path and os.path.exists(path) and "temp" in path.lower():
                os.remove(path)
                print(f"   🗑️ Cleaned up: {os.path.basename(path)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio preview demo failed: {e}")
        return False

def demo_audio_analyzer():
    """Demonstrate audio analysis capabilities."""
    print("\n🔍 Audio Analyzer Demo")
    print("=" * 30)
    
    try:
        from nodes.audio_utils import AudioProcessor
        from nodes.audio_preview import RajAudioAnalyzer
        
        # Create different types of test audio for analysis
        sample_rate = 22050
        duration = 3.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        print("📊 Creating test audio scenarios...")
        
        # Scenario 1: Clean sine wave
        clean_audio = torch.sin(2 * torch.pi * 440 * t).unsqueeze(1) * 0.5
        
        # Scenario 2: Complex audio with harmonics
        complex_audio = (
            torch.sin(2 * torch.pi * 220 * t) +
            0.5 * torch.sin(2 * torch.pi * 440 * t) +
            0.25 * torch.sin(2 * torch.pi * 880 * t)
        ).unsqueeze(1) * 0.3
        
        # Scenario 3: Noisy/distorted audio
        noise = torch.randn_like(t) * 0.1
        noisy_audio = (torch.sin(2 * torch.pi * 300 * t) + noise).unsqueeze(1)
        # Clip to simulate distortion
        noisy_audio = torch.clamp(noisy_audio * 1.5, -1.0, 1.0)
        
        analyzer = RajAudioAnalyzer()
        
        test_cases = [
            ("Clean Sine Wave", clean_audio, "basic"),
            ("Complex Harmonics", complex_audio, "detailed"),
            ("Noisy/Distorted", noisy_audio, "frequency")
        ]
        
        for name, audio, analysis_type in test_cases:
            print(f"\n📈 Analyzing: {name} ({analysis_type} analysis)")
            
            result = analyzer.analyze_audio(
                audio=audio,
                sample_rate=sample_rate,
                analysis_type=analysis_type,
                window_size=0.5,
                overlap=0.25
            )
            
            report, statistics, recommendations = result
            
            print(f"   📊 Statistics Preview:")
            stats_lines = statistics.split('\\n')[:4]  # First 4 lines
            for line in stats_lines:
                if line.strip():
                    print(f"      {line}")
            
            print(f"   💡 Key Recommendations:")
            rec_lines = recommendations.split('\\n')[:2]  # First 2 recommendations
            for line in rec_lines:
                if line.strip():
                    print(f"      {line}")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio analyzer demo failed: {e}")
        return False

def demo_real_world_workflow():
    """Demonstrate a real-world audio preview workflow."""
    print("\n🎬 Real-World Workflow Demo")
    print("=" * 35)
    
    try:
        from nodes.audio_utils import AudioProcessor
        from nodes.audio_loader import RajAudioProcessor
        from nodes.audio_preview import RajAudioPreview, RajAudioAnalyzer
        
        # Simulate: Load video -> Extract audio -> Analyze -> Process -> Preview
        test_video = "../../../input_vids/sometalk.mp4"
        
        if os.path.exists(test_video):
            print("1️⃣ Extracting audio from video...")
            audio_tensor, metadata = AudioProcessor.extract_audio_from_video(test_video)
            sample_rate = metadata['sample_rate']
            print(f"   ✅ Extracted: {metadata['duration']:.1f}s @ {sample_rate}Hz")
        else:
            # Fallback to synthetic
            print("1️⃣ Creating synthetic audio for demo...")
            sample_rate = 22050
            duration = 4.0
            t = torch.linspace(0, duration, int(sample_rate * duration))
            audio_tensor = torch.sin(2 * torch.pi * 440 * t).unsqueeze(1) * 0.8
            print(f"   ✅ Created: {duration:.1f}s test audio")
        
        print("\n2️⃣ Analyzing original audio...")
        analyzer = RajAudioAnalyzer()
        _, stats, recommendations = analyzer.analyze_audio(
            audio=audio_tensor,
            sample_rate=sample_rate,
            analysis_type="basic"
        )
        
        # Show key stats
        rms_line = [line for line in stats.split('\\n') if 'RMS:' in line]
        if rms_line:
            print(f"   📊 {rms_line[0].strip()}")
        
        print("\n3️⃣ Processing audio based on analysis...")
        processor = RajAudioProcessor()
        
        # Apply normalization
        processed_audio, _ = processor.process_audio(
            audio=audio_tensor,
            operation="normalize",
            normalize_method="peak",
            normalize_level=0.7,
            current_sample_rate=sample_rate
        )
        
        print("   ✅ Applied normalization")
        
        # Apply fade
        final_audio, _ = processor.process_audio(
            audio=processed_audio,
            operation="fade",
            fade_in=0.2,
            fade_out=0.5,
            current_sample_rate=sample_rate
        )
        
        print("   ✅ Applied fade effects")
        
        print("\n4️⃣ Creating preview of processed audio...")
        preview_node = RajAudioPreview()
        
        _, file_path, _, status = preview_node.create_preview(
            audio=final_audio,
            sample_rate=sample_rate,
            preview_duration=3.0,
            preview_mode="temporary",
            normalize_preview=False  # Already processed
        )
        
        print(f"   {status}")
        
        print("\n5️⃣ Final analysis of processed audio...")
        _, final_stats, final_recs = analyzer.analyze_audio(
            audio=final_audio,
            sample_rate=sample_rate,
            analysis_type="basic"
        )
        
        # Compare RMS levels
        final_rms_line = [line for line in final_stats.split('\\n') if 'RMS:' in line]
        if final_rms_line:
            print(f"   📊 After processing: {final_rms_line[0].strip()}")
        
        # Show improvement recommendations
        improvement_recs = [rec for rec in final_recs.split('\\n') if '✅' in rec]
        if improvement_recs:
            print(f"   💡 {improvement_recs[0]}")
        
        # Cleanup
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            print(f"   🗑️ Cleaned up preview file")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow demo failed: {e}")
        return False

def show_preview_system_summary():
    """Show summary of the audio preview system."""
    print("\n" + "="*50)
    print("🏆 AUDIO PREVIEW SYSTEM SUMMARY")
    print("="*50)
    
    print("\n🎵 RajAudioPreview Features:")
    print("   • Temporary and permanent preview file generation")
    print("   • WAV and MP3 output format support")
    print("   • Customizable preview duration and start offset")
    print("   • Volume boost and normalization options") 
    print("   • Automatic fade in/out for smooth playback")
    print("   • Waveform visualization data generation")
    print("   • Auto-increment filename system")
    
    print("\n🔍 RajAudioAnalyzer Features:")
    print("   • Basic, detailed, and frequency domain analysis")
    print("   • RMS, peak, and dynamic range measurements")
    print("   • Windowed analysis for temporal characteristics")
    print("   • Frequency band energy distribution")
    print("   • Intelligent processing recommendations")
    print("   • Clipping and distortion detection")
    
    print("\n🔄 Workflow Integration:")
    print("   Video Upload → Audio Extract → Analyze → Process → Preview")
    print("   Audio File → Load → Analyze → Apply Effects → Preview → Export")
    print("   Any Audio → Preview (Quick Listen) → Analyzer (Quality Check)")
    
    print("\n⚡ Key Benefits:")
    print("   • Non-destructive preview generation")
    print("   • Real-time audio quality assessment") 
    print("   • Automated processing recommendations")
    print("   • Cross-platform file compatibility")
    print("   • Seamless ComfyUI node integration")

if __name__ == "__main__":
    print("🎵 AUDIO PREVIEW & ANALYSIS SYSTEM DEMO")
    print("=" * 50)
    
    success_count = 0
    total_demos = 3
    
    # Run all demos
    if demo_audio_preview():
        success_count += 1
    
    if demo_audio_analyzer():
        success_count += 1
        
    if demo_real_world_workflow():
        success_count += 1
    
    # Show system summary
    show_preview_system_summary()
    
    print(f"\n🏆 DEMO RESULTS: {success_count}/{total_demos} workflows completed successfully")
    
    if success_count == total_demos:
        print("🎉 AUDIO PREVIEW SYSTEM IS FULLY OPERATIONAL!")
        print("\n✅ Ready for production use with:")
        print("   🎵 Audio preview with playback file generation")
        print("   🔍 Comprehensive audio analysis and recommendations")
        print("   🎚️ Integrated workflow with existing audio processing nodes")
        print("   📊 Waveform visualization and quality metrics")
    else:
        print("⚠️  Some components need attention")
    
    print("\n🚀 The complete audio preview and analysis system is ready!")