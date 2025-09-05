import os
import torch
from .utils import logger
from ..utils.audio_utils import AudioProcessor

try:
    import folder_paths
except ImportError:
    # Fallback for testing outside ComfyUI
    class MockFolderPaths:
        @staticmethod
        def get_input_directory():
            return "input"
        
        @staticmethod  
        def get_annotated_filepath(path):
            return path
            
        base_path = "."
    
    folder_paths = MockFolderPaths()

def get_audio_files():
    """Get list of audio files in input directory"""
    input_dir = folder_paths.get_input_directory()
    audio_extensions = ['wav', 'mp3', 'aac', 'flac', 'm4a', 'ogg', 'wma']
    files = []
    
    if os.path.exists(input_dir):
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and file_parts[-1].lower() in audio_extensions:
                    files.append(f)
    
    return sorted(files)

class RajAudioLoader:
    """
    Dedicated audio file loader with comprehensive format support.
    Loads audio files and outputs audio tensors compatible with RajWhisperAudio.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_file": (get_audio_files(), {
                    "audio_upload": True,
                    "tooltip": "Select audio file or click upload button"
                }),
                "target_sample_rate": ("INT", {
                    "default": 22050,
                    "min": 8000,
                    "max": 96000,
                    "step": 1000,
                    "tooltip": "Target sample rate (Hz)"
                }),
                "mono": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Convert to mono (recommended for Whisper)"
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize audio amplitude"
                }),
            },
            "optional": {
                "max_duration": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 1.0,
                    "tooltip": "Maximum duration to load (0 = load all)"
                }),
                "start_offset": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Start offset in seconds"
                }),
                "normalization_method": (["peak", "rms"], {
                    "default": "peak",
                    "tooltip": "Normalization method"
                }),
                "normalization_level": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Target normalization level"
                }),
            },
            "hidden": {
                "choose audio to upload": ("UPLOAD",)  # Creates upload button
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "audio_info", "file_info")
    FUNCTION = "load_audio"
    CATEGORY = "Raj Video Processing 🎬/Audio"
    
    def load_audio(self, audio_file, target_sample_rate, mono, normalize,
                   max_duration=0.0, start_offset=0.0, normalization_method="peak",
                   normalization_level=0.95, **kwargs):
        
        # Get full path to audio file
        audio_path = folder_paths.get_annotated_filepath(audio_file)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"🔊 Loading audio file: {os.path.basename(audio_path)}")
        
        try:
            # Load audio with AudioProcessor
            audio_tensor, metadata = AudioProcessor.load_audio_file(
                audio_path, target_sample_rate, mono
            )
            
            original_duration = metadata['duration']
            original_samples = metadata['samples']
            logger.info(f"📊 Original audio: {original_duration:.2f}s, {metadata['sample_rate']}Hz, {metadata['channels']}ch")
            
            # Apply start offset if specified
            if start_offset > 0.0:
                offset_samples = int(start_offset * metadata['sample_rate'])
                if offset_samples < audio_tensor.shape[0]:
                    audio_tensor = audio_tensor[offset_samples:]
                    logger.info(f"⏭️ Applied start offset: {start_offset:.1f}s")
                else:
                    logger.warning(f"Start offset {start_offset:.1f}s exceeds audio duration")
                    # Return empty audio
                    audio_tensor = torch.zeros((1, 1))
            
            # Apply max duration if specified
            if max_duration > 0.0:
                max_samples = int(max_duration * metadata['sample_rate'])
                if max_samples < audio_tensor.shape[0]:
                    audio_tensor = audio_tensor[:max_samples]
                    logger.info(f"✂️ Trimmed to max duration: {max_duration:.1f}s")
            
            # Apply normalization if requested
            if normalize and audio_tensor.numel() > 1:
                audio_tensor = AudioProcessor.normalize_audio(
                    audio_tensor, normalization_method, normalization_level
                )
                logger.info(f"📈 Normalized audio using {normalization_method} method")
            
            # Update metadata
            final_samples = audio_tensor.shape[0]
            final_duration = final_samples / metadata['sample_rate']
            final_channels = audio_tensor.shape[1]
            
            # Create audio info string
            audio_info = AudioProcessor.get_audio_info(audio_tensor, metadata['sample_rate'])
            
            # Create detailed file info
            file_info = (
                f"File: {os.path.basename(audio_path)}\n"
                f"Format: {metadata.get('format', 'unknown')}\n"
                f"Loader: {metadata.get('loader', 'unknown')}\n"
                f"Original: {original_duration:.2f}s @ {metadata['sample_rate']}Hz\n"
                f"Processed: {final_duration:.2f}s\n"
                f"Samples: {original_samples:,} → {final_samples:,}\n"
                f"Channels: {metadata['channels']} → {final_channels}\n"
                f"Mono conversion: {'Yes' if mono and metadata['channels'] > 1 else 'No'}\n"
                f"Normalization: {'Yes' if normalize else 'No'}"
                + (f" ({normalization_method})" if normalize else "")
            )
            
            logger.info(f"✅ Audio loaded successfully: {final_duration:.2f}s")
            
            return (audio_tensor, audio_info, file_info)
            
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            raise RuntimeError(f"Failed to load audio file: {str(e)}")
    
    @classmethod
    def IS_CHANGED(cls, audio_file, **kwargs):
        """Check if audio file has changed"""
        try:
            audio_path = folder_paths.get_annotated_filepath(audio_file)
            if os.path.exists(audio_path):
                return os.path.getmtime(audio_path)
            return float("inf")
        except:
            return float("inf")

class RajAudioProcessor:
    """
    Advanced audio processing node with filtering, effects, and analysis.
    Works with audio tensors from RajAudioLoader or RajVideoUpload.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Input audio tensor"
                }),
                "operation": (["passthrough", "normalize", "resample", "trim", "fade", "amplify"], {
                    "default": "passthrough",
                    "tooltip": "Audio processing operation"
                }),
            },
            "optional": {
                # Normalization parameters
                "normalize_method": (["peak", "rms"], {
                    "default": "peak",
                    "tooltip": "Normalization method"
                }),
                "normalize_level": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Target normalization level"
                }),
                
                # Resampling parameters  
                "current_sample_rate": ("INT", {
                    "default": 22050,
                    "min": 8000,
                    "max": 96000,
                    "step": 1000,
                    "tooltip": "Current sample rate"
                }),
                "target_sample_rate": ("INT", {
                    "default": 16000,
                    "min": 8000,
                    "max": 96000,
                    "step": 1000,
                    "tooltip": "Target sample rate"
                }),
                
                # Trim parameters
                "trim_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Trim start time (seconds)"
                }),
                "trim_end": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Trim end time (seconds, 0 = end of audio)"
                }),
                
                # Fade parameters
                "fade_in": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Fade in duration (seconds)"
                }),
                "fade_out": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Fade out duration (seconds)"
                }),
                
                # Amplify parameters
                "amplify_db": ("FLOAT", {
                    "default": 0.0,
                    "min": -40.0,
                    "max": 40.0,
                    "step": 0.1,
                    "tooltip": "Amplification in dB"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("processed_audio", "processing_info")
    FUNCTION = "process_audio"
    CATEGORY = "Raj Video Processing 🎬/Audio"
    
    def process_audio(self, audio, operation, normalize_method="peak", normalize_level=0.95,
                     current_sample_rate=22050, target_sample_rate=16000,
                     trim_start=0.0, trim_end=0.0, fade_in=0.1, fade_out=0.1,
                     amplify_db=0.0):
        
        if audio.numel() == 0:
            logger.warning("Input audio is empty")
            return (audio, "Input audio was empty")
        
        processed_audio = audio.clone()
        operations_applied = []
        
        try:
            if operation == "passthrough":
                operations_applied.append("passthrough")
            
            elif operation == "normalize":
                processed_audio = AudioProcessor.normalize_audio(
                    processed_audio, normalize_method, normalize_level
                )
                operations_applied.append(f"normalize ({normalize_method} @ {normalize_level})")
                logger.info(f"🔊 Applied normalization: {normalize_method}")
            
            elif operation == "resample":
                processed_audio = AudioProcessor.resample_audio(
                    processed_audio, current_sample_rate, target_sample_rate
                )
                operations_applied.append(f"resample ({current_sample_rate}Hz → {target_sample_rate}Hz)")
                logger.info(f"🔊 Resampled audio: {current_sample_rate}Hz → {target_sample_rate}Hz")
            
            elif operation == "trim":
                sample_rate = current_sample_rate
                total_samples = processed_audio.shape[0]
                total_duration = total_samples / sample_rate
                
                start_sample = int(trim_start * sample_rate) if trim_start > 0 else 0
                end_sample = int(trim_end * sample_rate) if trim_end > 0 else total_samples
                
                start_sample = max(0, min(start_sample, total_samples))
                end_sample = max(start_sample, min(end_sample, total_samples))
                
                processed_audio = processed_audio[start_sample:end_sample]
                operations_applied.append(f"trim ({trim_start:.1f}s - {trim_end:.1f}s)")
                logger.info(f"✂️ Trimmed audio: {start_sample} - {end_sample} samples")
            
            elif operation == "fade":
                sample_rate = current_sample_rate
                total_samples = processed_audio.shape[0]
                
                # Apply fade in
                if fade_in > 0:
                    fade_in_samples = int(fade_in * sample_rate)
                    fade_in_samples = min(fade_in_samples, total_samples)
                    
                    fade_curve = torch.linspace(0, 1, fade_in_samples).unsqueeze(1)
                    processed_audio[:fade_in_samples] *= fade_curve
                
                # Apply fade out
                if fade_out > 0:
                    fade_out_samples = int(fade_out * sample_rate)
                    fade_out_samples = min(fade_out_samples, total_samples)
                    
                    fade_curve = torch.linspace(1, 0, fade_out_samples).unsqueeze(1)
                    processed_audio[-fade_out_samples:] *= fade_curve
                
                operations_applied.append(f"fade (in: {fade_in:.1f}s, out: {fade_out:.1f}s)")
                logger.info(f"🔊 Applied fade: in {fade_in:.1f}s, out {fade_out:.1f}s")
            
            elif operation == "amplify":
                # Convert dB to linear scale
                linear_gain = 10 ** (amplify_db / 20)
                processed_audio = processed_audio * linear_gain
                
                # Clip to prevent overflow
                processed_audio = torch.clamp(processed_audio, -1.0, 1.0)
                
                operations_applied.append(f"amplify ({amplify_db:+.1f} dB)")
                logger.info(f"🔊 Applied amplification: {amplify_db:+.1f} dB")
            
            # Create processing info
            processing_info = (
                f"Audio Processing Complete\n"
                f"Operations: {', '.join(operations_applied)}\n"
                f"Input: {audio.shape[0]:,} samples, {audio.shape[1]} channels\n"
                f"Output: {processed_audio.shape[0]:,} samples, {processed_audio.shape[1]} channels\n"
                f"Duration change: {audio.shape[0]/current_sample_rate:.2f}s → {processed_audio.shape[0]/current_sample_rate:.2f}s"
            )
            
            logger.info(f"✅ Audio processing completed")
            
            return (processed_audio, processing_info)
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return (audio, f"Processing failed: {str(e)}")


# Test functions
def test_audio_loader():
    """Test audio loader functionality."""
    print("Testing RajAudioLoader...")
    
    # Check for test files
    files = get_audio_files()
    print(f"Available audio files: {files}")
    
    if files:
        loader = RajAudioLoader()
        print("Audio loader initialized successfully")
    else:
        print("No audio files found for testing")

def test_audio_processor():
    """Test audio processor functionality."""
    print("Testing RajAudioProcessor...")
    
    # Create test audio (1 second of sine wave)
    sample_rate = 22050
    duration = 1.0
    frequency = 440.0  # A4 note
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    test_audio = torch.sin(2 * torch.pi * frequency * t).unsqueeze(1)
    
    processor = RajAudioProcessor()
    
    # Test normalization
    normalized_audio, info = processor.process_audio(test_audio, "normalize")
    print(f"Normalization test: {info}")
    
    print("Audio processor test completed")

if __name__ == "__main__":
    test_audio_loader()
    test_audio_processor()