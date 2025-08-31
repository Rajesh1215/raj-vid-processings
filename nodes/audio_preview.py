import os
import torch
import tempfile
import base64
from .utils import logger
from .audio_utils import AudioProcessor

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    logger.warning("TorchAudio not available for audio preview")

class RajAudioPreview:
    """
    Audio preview node that creates playable audio files and provides waveform visualization.
    Supports both temporary preview files and permanent audio exports.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Input audio tensor to preview"
                }),
                "sample_rate": ("INT", {
                    "default": 22050,
                    "min": 8000,
                    "max": 96000,
                    "step": 1000,
                    "tooltip": "Audio sample rate (Hz)"
                }),
                "preview_duration": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 300.0,
                    "step": 0.5,
                    "tooltip": "Preview duration (0 = full audio)"
                }),
                "preview_mode": (["temporary", "permanent"], {
                    "default": "temporary",
                    "tooltip": "Temporary for quick preview, permanent for saving"
                }),
            },
            "optional": {
                "start_offset": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 300.0,
                    "step": 0.1,
                    "tooltip": "Start preview from this time (seconds)"
                }),
                "volume_boost": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Volume boost multiplier"
                }),
                "fade_edges": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply fade in/out to preview"
                }),
                "output_format": (["wav", "mp3"], {
                    "default": "wav",
                    "tooltip": "Audio output format"
                }),
                "filename_prefix": ("STRING", {
                    "default": "audio_preview",
                    "tooltip": "Filename prefix for permanent files"
                }),
                "normalize_preview": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize audio for consistent playback"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio_info", "file_path", "waveform_data", "preview_status")
    FUNCTION = "create_preview"
    CATEGORY = "Raj Video Processing 🎬/Audio"
    OUTPUT_NODE = True
    
    def create_preview(self, audio, sample_rate, preview_duration, preview_mode,
                      start_offset=0.0, volume_boost=1.0, fade_edges=True,
                      output_format="wav", filename_prefix="audio_preview",
                      normalize_preview=True):
        
        if audio.numel() == 0:
            logger.warning("Input audio is empty")
            return ("Empty audio", "", "", "Error: Empty audio input")
        
        try:
            logger.info(f"🎵 Creating audio preview: {preview_mode} mode")
            
            # Clone audio to avoid modifying original
            preview_audio = audio.clone()
            original_shape = preview_audio.shape
            total_samples = preview_audio.shape[0]
            total_duration = total_samples / sample_rate
            
            # Apply start offset
            if start_offset > 0:
                offset_samples = int(start_offset * sample_rate)
                if offset_samples < total_samples:
                    preview_audio = preview_audio[offset_samples:]
                    logger.info(f"⏭️ Applied start offset: {start_offset:.1f}s")
                else:
                    logger.warning(f"Start offset exceeds audio duration")
                    return ("Error: Start offset too large", "", "", "Error")
            
            # Apply preview duration limit
            if preview_duration > 0:
                duration_samples = int(preview_duration * sample_rate)
                if duration_samples < preview_audio.shape[0]:
                    preview_audio = preview_audio[:duration_samples]
                    logger.info(f"✂️ Limited to {preview_duration:.1f}s")
            
            # Apply volume boost
            if volume_boost != 1.0:
                preview_audio = preview_audio * volume_boost
                # Clip to prevent distortion
                preview_audio = torch.clamp(preview_audio, -1.0, 1.0)
                logger.info(f"🔊 Applied volume boost: {volume_boost:.1f}x")
            
            # Apply normalization
            if normalize_preview and preview_audio.numel() > 0:
                preview_audio = AudioProcessor.normalize_audio(
                    preview_audio, method="peak", target_level=0.8
                )
                logger.info(f"📈 Normalized preview audio")
            
            # Apply fade edges
            if fade_edges and preview_audio.shape[0] > sample_rate * 0.1:  # Only if longer than 0.1s
                fade_samples = min(int(0.05 * sample_rate), preview_audio.shape[0] // 20)  # 50ms or 5% max
                
                # Fade in
                fade_in_curve = torch.linspace(0, 1, fade_samples).unsqueeze(1)
                if preview_audio.dim() == 1:
                    fade_in_curve = fade_in_curve.squeeze(1)
                preview_audio[:fade_samples] *= fade_in_curve
                
                # Fade out
                fade_out_curve = torch.linspace(1, 0, fade_samples).unsqueeze(1)
                if preview_audio.dim() == 1:
                    fade_out_curve = fade_out_curve.squeeze(1)
                preview_audio[-fade_samples:] *= fade_out_curve
                
                logger.info(f"🎭 Applied edge fades: {fade_samples} samples")
            
            # Create output file
            file_path = self._save_preview_audio(
                preview_audio, sample_rate, preview_mode, 
                output_format, filename_prefix
            )
            
            # Generate audio info
            preview_duration_actual = preview_audio.shape[0] / sample_rate
            audio_info = (
                f"Audio Preview Generated\\n"
                f"Original: {total_duration:.2f}s @ {sample_rate}Hz\\n"
                f"Preview: {preview_duration_actual:.2f}s\\n"
                f"Samples: {original_shape[0]:,} → {preview_audio.shape[0]:,}\\n"
                f"Channels: {preview_audio.shape[1] if preview_audio.dim() > 1 else 1}\\n"
                f"Format: {output_format.upper()}\\n"
                f"Mode: {preview_mode}\\n"
                f"Volume: {volume_boost:.1f}x"
                + (f"\\nOffset: {start_offset:.1f}s" if start_offset > 0 else "")
                + (f"\\nDuration Limit: {preview_duration:.1f}s" if preview_duration > 0 else "")
            )
            
            # Generate waveform data for visualization
            waveform_data = self._generate_waveform_data(preview_audio, sample_rate)
            
            # Status message
            status = f"✅ Preview created: {os.path.basename(file_path)} ({preview_duration_actual:.1f}s)"
            
            logger.info(f"✅ Audio preview created: {file_path}")
            
            return (audio_info, file_path, waveform_data, status)
            
        except Exception as e:
            logger.error(f"Error creating audio preview: {e}")
            error_msg = f"Failed to create preview: {str(e)}"
            return (error_msg, "", "", f"❌ Error: {str(e)}")
    
    def _save_preview_audio(self, audio_tensor, sample_rate, preview_mode, output_format, filename_prefix):
        """Save audio tensor to file."""
        
        if not TORCHAUDIO_AVAILABLE:
            raise RuntimeError("TorchAudio is required for audio preview")
        
        # Determine output directory
        if preview_mode == "temporary":
            output_dir = tempfile.gettempdir()
            filename = f"comfyui_audio_preview_{os.getpid()}.{output_format}"
        else:
            # Permanent mode - save to output directory
            try:
                import folder_paths
                output_dir = folder_paths.get_output_directory()
            except ImportError:
                output_dir = "output"
            
            # Create unique filename
            import time
            timestamp = int(time.time())
            filename = f"{filename_prefix}_{timestamp}.{output_format}"
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)
        
        # Ensure audio is in correct format for saving
        if audio_tensor.dim() == 1:
            # Mono audio - add channel dimension
            save_audio = audio_tensor.unsqueeze(0)
        else:
            # Multi-channel - transpose to (channels, samples)
            save_audio = audio_tensor.transpose(0, 1)
        
        # Save using TorchAudio
        if output_format == "mp3":
            # For MP3, we need to ensure we have the right backend
            try:
                torchaudio.save(file_path, save_audio, sample_rate, format="mp3")
            except Exception as e:
                # Fallback to WAV if MP3 fails
                logger.warning(f"MP3 save failed, using WAV: {e}")
                file_path = file_path.replace(".mp3", ".wav")
                torchaudio.save(file_path, save_audio, sample_rate)
        else:
            torchaudio.save(file_path, save_audio, sample_rate)
        
        return file_path
    
    def _generate_waveform_data(self, audio_tensor, sample_rate, max_points=1000):
        """Generate waveform visualization data."""
        try:
            # Convert to mono for waveform
            if audio_tensor.dim() > 1 and audio_tensor.shape[1] > 1:
                waveform = audio_tensor.mean(dim=1)
            else:
                waveform = audio_tensor.squeeze()
            
            # Downsample for visualization if needed
            total_samples = waveform.shape[0]
            if total_samples > max_points:
                step = total_samples // max_points
                waveform = waveform[::step][:max_points]
            
            # Convert to list and create time axis
            waveform_list = waveform.tolist()
            duration = len(waveform_list) / sample_rate * (total_samples / len(waveform_list))
            time_axis = [i * duration / len(waveform_list) for i in range(len(waveform_list))]
            
            # Create simple JSON representation
            waveform_data = {
                "samples": len(waveform_list),
                "duration": duration,
                "sample_rate": sample_rate,
                "waveform": waveform_list[:100],  # Limit data size
                "time_axis": time_axis[:100],
                "peak": max(abs(min(waveform_list)), abs(max(waveform_list))),
                "rms": (sum(x*x for x in waveform_list) / len(waveform_list)) ** 0.5
            }
            
            import json
            return json.dumps(waveform_data, indent=2)
            
        except Exception as e:
            logger.warning(f"Could not generate waveform data: {e}")
            return f'{{"error": "Could not generate waveform: {str(e)}"}}'

class RajAudioAnalyzer:
    """
    Audio analysis node that provides detailed audio characteristics and statistics.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Input audio tensor to analyze"
                }),
                "sample_rate": ("INT", {
                    "default": 22050,
                    "min": 8000,
                    "max": 96000,
                    "step": 1000,
                    "tooltip": "Audio sample rate (Hz)"
                }),
                "analysis_type": (["basic", "detailed", "frequency"], {
                    "default": "basic",
                    "tooltip": "Type of analysis to perform"
                }),
            },
            "optional": {
                "window_size": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Analysis window size (seconds)"
                }),
                "overlap": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 0.9,
                    "step": 0.1,
                    "tooltip": "Window overlap ratio"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("analysis_report", "statistics", "recommendations")
    FUNCTION = "analyze_audio"
    CATEGORY = "Raj Video Processing 🎬/Audio"
    
    def analyze_audio(self, audio, sample_rate, analysis_type, window_size=1.0, overlap=0.5):
        
        if audio.numel() == 0:
            return ("Empty audio input", "No statistics available", "No recommendations")
        
        try:
            logger.info(f"🔍 Analyzing audio: {analysis_type} analysis")
            
            # Basic audio properties
            total_samples = audio.shape[0]
            duration = total_samples / sample_rate
            channels = audio.shape[1] if audio.dim() > 1 else 1
            
            # Convert to mono for analysis
            if channels > 1:
                mono_audio = audio.mean(dim=1)
            else:
                mono_audio = audio.squeeze()
            
            # Calculate basic statistics
            audio_min = mono_audio.min().item()
            audio_max = mono_audio.max().item()
            audio_mean = mono_audio.mean().item()
            audio_std = mono_audio.std().item()
            audio_rms = torch.sqrt(torch.mean(mono_audio ** 2)).item()
            
            # Calculate dynamic range
            dynamic_range_db = 20 * torch.log10(torch.tensor(abs(audio_max) / max(audio_rms, 1e-8))).item()
            
            # Peak detection
            abs_audio = torch.abs(mono_audio)
            peak_threshold = 0.8
            peaks = (abs_audio > peak_threshold).sum().item()
            clipping = (abs_audio >= 0.99).sum().item()
            
            # Generate analysis report
            if analysis_type == "basic":
                report = self._generate_basic_report(
                    duration, channels, sample_rate, audio_min, audio_max, 
                    audio_mean, audio_std, audio_rms, dynamic_range_db
                )
            elif analysis_type == "detailed":
                report = self._generate_detailed_report(
                    mono_audio, sample_rate, duration, channels, window_size, overlap
                )
            else:  # frequency
                report = self._generate_frequency_report(
                    mono_audio, sample_rate, duration
                )
            
            # Generate statistics
            statistics = (
                f"Audio Statistics:\\n"
                f"Duration: {duration:.2f}s\\n"
                f"Samples: {total_samples:,}\\n"
                f"Channels: {channels}\\n"
                f"Sample Rate: {sample_rate:,}Hz\\n"
                f"Range: [{audio_min:.3f}, {audio_max:.3f}]\\n"
                f"Mean: {audio_mean:.3f}\\n"
                f"RMS: {audio_rms:.3f}\\n"
                f"Dynamic Range: {dynamic_range_db:.1f}dB\\n"
                f"Peaks (>80%): {peaks:,}\\n"
                f"Clipping (>99%): {clipping:,}"
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                audio_rms, dynamic_range_db, clipping, peaks, total_samples
            )
            
            logger.info(f"✅ Audio analysis completed: {analysis_type}")
            
            return (report, statistics, recommendations)
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            error_msg = f"Analysis failed: {str(e)}"
            return (error_msg, "Statistics unavailable", "No recommendations")
    
    def _generate_basic_report(self, duration, channels, sample_rate, audio_min, audio_max, 
                              audio_mean, audio_std, audio_rms, dynamic_range_db):
        """Generate basic analysis report."""
        return (
            f"Basic Audio Analysis Report\\n"
            f"{'='*30}\\n"
            f"Duration: {duration:.2f} seconds\\n"
            f"Channels: {channels} ({'Stereo' if channels == 2 else 'Mono' if channels == 1 else f'{channels}-channel'})\\n"
            f"Sample Rate: {sample_rate:,} Hz\\n"
            f"Amplitude Range: {audio_min:.3f} to {audio_max:.3f}\\n"
            f"Mean Amplitude: {audio_mean:.3f}\\n"
            f"RMS Level: {audio_rms:.3f}\\n"
            f"Standard Deviation: {audio_std:.3f}\\n"
            f"Dynamic Range: {dynamic_range_db:.1f} dB\\n"
        )
    
    def _generate_detailed_report(self, audio_tensor, sample_rate, duration, channels, window_size, overlap):
        """Generate detailed windowed analysis."""
        window_samples = int(window_size * sample_rate)
        hop_samples = int(window_samples * (1 - overlap))
        
        windows = []
        for i in range(0, len(audio_tensor) - window_samples + 1, hop_samples):
            window = audio_tensor[i:i+window_samples]
            windows.append({
                'start_time': i / sample_rate,
                'rms': torch.sqrt(torch.mean(window ** 2)).item(),
                'peak': torch.max(torch.abs(window)).item(),
                'energy': torch.sum(window ** 2).item()
            })
        
        if not windows:
            return "Audio too short for windowed analysis"
        
        # Calculate windowed statistics
        rms_values = [w['rms'] for w in windows]
        peak_values = [w['peak'] for w in windows]
        energy_values = [w['energy'] for w in windows]
        
        return (
            f"Detailed Audio Analysis Report\\n"
            f"{'='*35}\\n"
            f"Window Analysis ({len(windows)} windows)\\n"
            f"Window Size: {window_size}s, Overlap: {overlap*100:.0f}%\\n"
            f"\\n"
            f"RMS Analysis:\\n"
            f"  Average: {sum(rms_values)/len(rms_values):.3f}\\n"
            f"  Min: {min(rms_values):.3f}\\n"
            f"  Max: {max(rms_values):.3f}\\n"
            f"\\n"
            f"Peak Analysis:\\n"
            f"  Average: {sum(peak_values)/len(peak_values):.3f}\\n"
            f"  Min: {min(peak_values):.3f}\\n"
            f"  Max: {max(peak_values):.3f}\\n"
            f"\\n"
            f"Energy Distribution:\\n"
            f"  Total Energy: {sum(energy_values):.2e}\\n"
            f"  Energy Variance: {torch.var(torch.tensor(energy_values)).item():.2e}"
        )
    
    def _generate_frequency_report(self, audio_tensor, sample_rate, duration):
        """Generate frequency domain analysis."""
        try:
            # Simple frequency analysis using FFT
            fft = torch.fft.fft(audio_tensor)
            magnitude = torch.abs(fft)
            
            # Find dominant frequencies
            freqs = torch.fft.fftfreq(len(audio_tensor), 1/sample_rate)
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            # Find peaks in frequency domain
            _, peak_indices = torch.topk(positive_magnitude, k=min(5, len(positive_magnitude)//10))
            dominant_freqs = positive_freqs[peak_indices]
            
            return (
                f"Frequency Analysis Report\\n"
                f"{'='*30}\\n"
                f"FFT Length: {len(audio_tensor):,} samples\\n"
                f"Frequency Resolution: {sample_rate/len(audio_tensor):.1f} Hz\\n"
                f"\\n"
                f"Dominant Frequencies:\\n"
                + "\\n".join([f"  {freq:.1f} Hz" for freq in dominant_freqs[:5]]) + "\\n"
                f"\\n"
                f"Frequency Bands:\\n"
                f"  Sub-bass (20-60 Hz): {self._get_band_energy(positive_freqs, positive_magnitude, 20, 60):.3f}\\n"
                f"  Bass (60-250 Hz): {self._get_band_energy(positive_freqs, positive_magnitude, 60, 250):.3f}\\n"
                f"  Mid (250-2000 Hz): {self._get_band_energy(positive_freqs, positive_magnitude, 250, 2000):.3f}\\n"
                f"  High-mid (2-8 kHz): {self._get_band_energy(positive_freqs, positive_magnitude, 2000, 8000):.3f}\\n"
                f"  Treble (8+ kHz): {self._get_band_energy(positive_freqs, positive_magnitude, 8000, sample_rate//2):.3f}"
            )
        except Exception as e:
            return f"Frequency analysis failed: {str(e)}"
    
    def _get_band_energy(self, freqs, magnitude, low_freq, high_freq):
        """Calculate energy in frequency band."""
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        return torch.sum(magnitude[mask]).item() if mask.any() else 0.0
    
    def _generate_recommendations(self, rms, dynamic_range_db, clipping, peaks, total_samples):
        """Generate audio processing recommendations."""
        recommendations = []
        
        # Volume recommendations
        if rms < 0.1:
            recommendations.append("🔊 Audio is quiet - consider amplification or normalization")
        elif rms > 0.7:
            recommendations.append("🔉 Audio is loud - consider reducing volume")
        
        # Dynamic range recommendations
        if dynamic_range_db < 6:
            recommendations.append("📈 Low dynamic range - audio may be over-compressed")
        elif dynamic_range_db > 40:
            recommendations.append("📉 High dynamic range - consider compression for consistent levels")
        
        # Clipping recommendations
        if clipping > 0:
            recommendations.append(f"⚠️ {clipping} clipped samples detected - reduce volume or apply limiting")
        
        # Peak recommendations
        if peaks > total_samples * 0.01:  # More than 1% peaks
            recommendations.append("📊 Many peaks detected - consider applying compression")
        
        # Duration recommendations
        duration = total_samples / 22050  # Assume default sample rate for estimation
        if duration < 0.5:
            recommendations.append("⏱️ Very short audio - consider extending for better processing")
        elif duration > 300:
            recommendations.append("⏱️ Long audio - consider splitting into segments")
        
        if not recommendations:
            recommendations.append("✅ Audio appears to be in good condition")
        
        return "\\n".join(recommendations)

# Test functions
def test_audio_preview():
    """Test audio preview functionality."""
    print("Testing RajAudioPreview...")
    
    # Create test audio
    sample_rate = 22050
    duration = 2.0
    frequency = 440.0
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    test_audio = torch.sin(2 * torch.pi * frequency * t).unsqueeze(1)
    
    preview_node = RajAudioPreview()
    
    # Test temporary preview
    result = preview_node.create_preview(
        audio=test_audio,
        sample_rate=sample_rate,
        preview_duration=1.0,
        preview_mode="temporary"
    )
    
    audio_info, file_path, waveform_data, status = result
    print(f"Preview test result: {status}")
    print(f"File created: {file_path}")
    
    # Cleanup
    if os.path.exists(file_path):
        os.remove(file_path)
        print("Cleaned up test file")

def test_audio_analyzer():
    """Test audio analyzer functionality."""
    print("Testing RajAudioAnalyzer...")
    
    # Create test audio
    sample_rate = 22050
    duration = 1.0
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    test_audio = torch.sin(2 * torch.pi * 440 * t).unsqueeze(1)
    
    analyzer_node = RajAudioAnalyzer()
    
    # Test basic analysis
    result = analyzer_node.analyze_audio(
        audio=test_audio,
        sample_rate=sample_rate,
        analysis_type="basic"
    )
    
    report, stats, recommendations = result
    print("Analysis completed successfully")
    print(f"Recommendations: {recommendations}")

if __name__ == "__main__":
    test_audio_preview()
    test_audio_analyzer()