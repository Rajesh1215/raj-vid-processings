import torch
import numpy as np
import tempfile
import os
from typing import Dict, Tuple, Optional, Union
from .utils import logger

# Function to safely import libraries with detailed error info
def safe_import(module_name, from_module=None):
    """Safely import a module and return availability status."""
    try:
        if from_module:
            module = __import__(from_module, fromlist=[module_name])
            getattr(module, module_name)  # Check if the specific class/function exists
        else:
            __import__(module_name)
        return True
    except ImportError as e:
        # Only log detailed error in debug mode
        import sys
        if hasattr(sys, 'ps1'):  # Interactive mode
            print(f"Debug: {module_name} import failed: {e}")
        return False
    except Exception as e:
        if hasattr(sys, 'ps1'):  # Interactive mode
            print(f"Debug: {module_name} other error: {e}")
        return False

# Try to import audio processing libraries
LIBROSA_AVAILABLE = safe_import('librosa')
MOVIEPY_AVAILABLE = safe_import('VideoFileClip', 'moviepy.editor') and safe_import('AudioFileClip', 'moviepy.editor')
TORCHAUDIO_AVAILABLE = safe_import('torchaudio')

# Also try the new MoviePy 2.0+ structure
if not MOVIEPY_AVAILABLE:
    try:
        # Try MoviePy 2.0+ imports
        from moviepy import VideoFileClip, AudioFileClip
        MOVIEPY_AVAILABLE = True
    except:
        try:
            # Try alternative imports for MoviePy 2.0+
            from moviepy.video.io.VideoFileClip import VideoFileClip
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            MOVIEPY_AVAILABLE = True
        except:
            pass

if not LIBROSA_AVAILABLE:
    try:
        import librosa
        LIBROSA_AVAILABLE = True
    except:
        pass

if not TORCHAUDIO_AVAILABLE:
    try:
        import torchaudio
        TORCHAUDIO_AVAILABLE = True
    except:
        pass

class AudioProcessor:
    """
    Comprehensive audio processing utilities for ComfyUI video nodes.
    Handles audio extraction, conversion, and tensor operations.
    """
    
    # Standard audio formats and their metadata
    SUPPORTED_FORMATS = {
        'wav': {'container': 'wav', 'codec': 'pcm'},
        'mp3': {'container': 'mp3', 'codec': 'mp3'},
        'aac': {'container': 'mp4', 'codec': 'aac'},
        'flac': {'container': 'flac', 'codec': 'flac'},
        'm4a': {'container': 'mp4', 'codec': 'aac'},
        'ogg': {'container': 'ogg', 'codec': 'vorbis'}
    }
    
    # Standard sample rates
    SAMPLE_RATES = [8000, 16000, 22050, 44100, 48000]
    DEFAULT_SAMPLE_RATE = 22050  # Good for Whisper
    
    @staticmethod
    def extract_audio_from_video(video_path: str, 
                                target_sample_rate: int = DEFAULT_SAMPLE_RATE,
                                mono: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Extract audio from video file using multiple backends.
        
        Args:
            video_path (str): Path to video file
            target_sample_rate (int): Target sample rate
            mono (bool): Convert to mono if True
            
        Returns:
            Tuple[torch.Tensor, Dict]: Audio tensor and metadata
        """
        logger.info(f"🔊 Extracting audio from video: {os.path.basename(video_path)}")
        
        # Try multiple extraction methods in order of preference
        extraction_methods = [
            ("FFmpeg direct", AudioProcessor._extract_with_ffmpeg),
            ("MoviePy", AudioProcessor._extract_with_moviepy),
            ("OpenCV + FFmpeg", AudioProcessor._extract_with_opencv),
        ]
        
        for method_name, method_func in extraction_methods:
            try:
                logger.info(f"Trying {method_name}...")
                return method_func(video_path, target_sample_rate, mono)
            except Exception as e:
                logger.warning(f"{method_name} failed: {e}")
                continue
        
        # If all methods fail, return empty audio
        logger.error("All audio extraction methods failed")
        empty_audio = torch.zeros((1, 1))
        metadata = {
            'sample_rate': target_sample_rate,
            'channels': 1,
            'duration': 0.0,
            'samples': 1,
            'has_audio': False,
            'format': 'empty',
            'error': 'All extraction methods failed'
        }
        return empty_audio, metadata
    
    @staticmethod
    def _extract_with_ffmpeg(video_path: str, target_sample_rate: int, mono: bool) -> Tuple[torch.Tensor, Dict]:
        """Extract audio using FFmpeg directly."""
        import subprocess
        import tempfile
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_audio_path = temp_file.name
        
        try:
            # Use FFmpeg to extract audio
            ffmpeg_cmd = [
                'ffmpeg', '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', str(target_sample_rate),  # Sample rate
                '-ac', '1' if mono else '2',  # Channels
                '-y',  # Overwrite output
                temp_audio_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            # Load the extracted audio
            if TORCHAUDIO_AVAILABLE:
                import torchaudio
                waveform, sample_rate = torchaudio.load(temp_audio_path)
                waveform = waveform.T  # Convert to (samples, channels)
            else:
                raise RuntimeError("TorchAudio not available for loading extracted audio")
            
            metadata = {
                'sample_rate': sample_rate,
                'channels': waveform.shape[1],
                'duration': waveform.shape[0] / sample_rate,
                'samples': waveform.shape[0],
                'has_audio': True,
                'format': 'wav',
                'loader': 'ffmpeg'
            }
            
            logger.info(f"✅ FFmpeg extraction successful: {metadata['duration']:.2f}s")
            return waveform, metadata
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
    @staticmethod
    def _extract_with_moviepy(video_path: str, target_sample_rate: int, mono: bool) -> Tuple[torch.Tensor, Dict]:
        """Extract audio using MoviePy (if available)."""
        # Try different import patterns for MoviePy 2.0+
        VideoFileClip = None
        
        # Try MoviePy 2.0+ direct import
        try:
            from moviepy import VideoFileClip
            logger.info("MoviePy 2.0+ loaded successfully (direct import)")
        except ImportError:
            try:
                # Try MoviePy 2.0+ submodule import
                from moviepy.video.io.VideoFileClip import VideoFileClip
                logger.info("MoviePy 2.0+ loaded successfully (submodule import)")
            except ImportError:
                try:
                    # Try legacy MoviePy 1.x import (fallback)
                    from moviepy.editor import VideoFileClip
                    logger.info("MoviePy 1.x loaded successfully (legacy import)")
                except ImportError as e:
                    raise RuntimeError(f"MoviePy not available with any import method: {e}")
        
        if VideoFileClip is None:
            raise RuntimeError("Could not import VideoFileClip from MoviePy")
        
        # Load video with audio
        video = VideoFileClip(video_path)
        
        if video.audio is None:
            raise RuntimeError(f"Video has no audio track")
        
        # Extract audio
        audio_clip = video.audio
        duration = audio_clip.duration
        
        # Create temporary file for audio extraction
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Write audio to temporary file
            audio_clip.write_audiofile(
                temp_path,
                fps=target_sample_rate,
                nbytes=2,  # 16-bit
                verbose=False,
                logger=None
            )
            
            # Load audio as tensor
            audio_tensor, metadata = AudioProcessor.load_audio_file(
                temp_path, target_sample_rate, mono
            )
            
            # Update metadata with video info
            metadata.update({
                'source': 'moviepy_extraction',
                'video_duration': video.duration,
                'video_fps': video.fps if hasattr(video, 'fps') else None,
                'has_audio': True,
                'loader': 'moviepy'
            })
            
            logger.info(f"✅ MoviePy extraction successful: {metadata['duration']:.2f}s")
            return audio_tensor, metadata
            
        finally:
            # Cleanup
            audio_clip.close()
            video.close()
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @staticmethod
    def _extract_with_opencv(video_path: str, target_sample_rate: int, mono: bool) -> Tuple[torch.Tensor, Dict]:
        """Extract audio using OpenCV + FFmpeg (fallback method)."""
        import subprocess
        import tempfile
        
        # Check if video has audio using OpenCV
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        # Use FFmpeg to check for audio streams
        probe_cmd = [
            'ffprobe', '-v', 'quiet', '-select_streams', 'a:0',
            '-show_entries', 'stream=index', '-of', 'csv=p=0',
            video_path
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError("No audio stream found in video")
        
        # Extract audio using FFmpeg
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_audio_path = temp_file.name
        
        try:
            # Use FFmpeg to extract audio
            ffmpeg_cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(target_sample_rate),
                '-ac', '1' if mono else '2',
                '-y', temp_audio_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")
            
            # Load the extracted audio
            if TORCHAUDIO_AVAILABLE:
                import torchaudio
                waveform, sample_rate = torchaudio.load(temp_audio_path)
                waveform = waveform.T  # Convert to (samples, channels)
            else:
                raise RuntimeError("TorchAudio not available for loading extracted audio")
            
            metadata = {
                'sample_rate': sample_rate,
                'channels': waveform.shape[1],
                'duration': waveform.shape[0] / sample_rate,
                'samples': waveform.shape[0],
                'has_audio': True,
                'format': 'wav',
                'loader': 'opencv_ffmpeg',
                'video_fps': fps,
                'video_frame_count': frame_count
            }
            
            logger.info(f"✅ OpenCV+FFmpeg extraction successful: {metadata['duration']:.2f}s")
            return waveform, metadata
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
    
    @staticmethod
    def load_audio_file(audio_path: str,
                       target_sample_rate: int = DEFAULT_SAMPLE_RATE,
                       mono: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Load audio file and convert to tensor.
        
        Args:
            audio_path (str): Path to audio file
            target_sample_rate (int): Target sample rate
            mono (bool): Convert to mono if True
            
        Returns:
            Tuple[torch.Tensor, Dict]: Audio tensor and metadata
        """
        try:
            # Try torchaudio first (fastest)
            if TORCHAUDIO_AVAILABLE:
                return AudioProcessor._load_with_torchaudio(audio_path, target_sample_rate, mono)
            
            # Fallback to librosa
            elif LIBROSA_AVAILABLE:
                return AudioProcessor._load_with_librosa(audio_path, target_sample_rate, mono)
            
            # Last resort: MoviePy
            elif MOVIEPY_AVAILABLE:
                return AudioProcessor._load_with_moviepy(audio_path, target_sample_rate, mono)
            
            else:
                raise RuntimeError("No audio processing library available. Install torchaudio, librosa, or moviepy")
                
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            # Return empty audio tensor on error
            empty_audio = torch.zeros((1, 1))
            metadata = {
                'sample_rate': target_sample_rate,
                'channels': 1,
                'duration': 0.0,
                'samples': 1,
                'has_audio': False,
                'format': 'error',
                'error': str(e)
            }
            return empty_audio, metadata
    
    @staticmethod
    def _load_with_torchaudio(audio_path: str, target_sample_rate: int, mono: bool) -> Tuple[torch.Tensor, Dict]:
        """Load audio using torchaudio."""
        import torchaudio
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample if needed
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
        
        # Convert to mono if requested
        if mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Transpose to (samples, channels) format
        waveform = waveform.T
        
        metadata = {
            'sample_rate': sample_rate,
            'channels': waveform.shape[1],
            'duration': waveform.shape[0] / sample_rate,
            'samples': waveform.shape[0],
            'has_audio': True,
            'format': os.path.splitext(audio_path)[1][1:].lower(),
            'loader': 'torchaudio'
        }
        
        return waveform, metadata
    
    @staticmethod
    def _load_with_librosa(audio_path: str, target_sample_rate: int, mono: bool) -> Tuple[torch.Tensor, Dict]:
        """Load audio using librosa."""
        # Load audio
        audio_data, sample_rate = librosa.load(
            audio_path, 
            sr=target_sample_rate, 
            mono=mono
        )
        
        # Convert numpy to tensor
        if mono:
            # Reshape mono audio to (samples, 1)
            waveform = torch.from_numpy(audio_data).unsqueeze(1).float()
        else:
            # Handle stereo: (2, samples) -> (samples, 2)
            waveform = torch.from_numpy(audio_data).T.float()
        
        metadata = {
            'sample_rate': target_sample_rate,
            'channels': waveform.shape[1],
            'duration': waveform.shape[0] / target_sample_rate,
            'samples': waveform.shape[0],
            'has_audio': True,
            'format': os.path.splitext(audio_path)[1][1:].lower(),
            'loader': 'librosa'
        }
        
        return waveform, metadata
    
    @staticmethod
    def _load_with_moviepy(audio_path: str, target_sample_rate: int, mono: bool) -> Tuple[torch.Tensor, Dict]:
        """Load audio using MoviePy."""
        audio_clip = AudioFileClip(audio_path)
        
        # Get audio as numpy array
        audio_array = audio_clip.to_soundarray(fps=target_sample_rate)
        audio_clip.close()
        
        # Convert to tensor
        waveform = torch.from_numpy(audio_array).float()
        
        # Handle mono conversion
        if mono and waveform.shape[1] > 1:
            waveform = torch.mean(waveform, dim=1, keepdim=True)
        
        metadata = {
            'sample_rate': target_sample_rate,
            'channels': waveform.shape[1],
            'duration': waveform.shape[0] / target_sample_rate,
            'samples': waveform.shape[0],
            'has_audio': True,
            'format': os.path.splitext(audio_path)[1][1:].lower(),
            'loader': 'moviepy'
        }
        
        return waveform, metadata
    
    @staticmethod
    def tensor_to_numpy(audio_tensor: torch.Tensor) -> np.ndarray:
        """Convert audio tensor to numpy array."""
        if isinstance(audio_tensor, torch.Tensor):
            return audio_tensor.detach().cpu().numpy()
        return audio_tensor
    
    @staticmethod
    def numpy_to_tensor(audio_array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to audio tensor."""
        return torch.from_numpy(audio_array).float()
    
    @staticmethod
    def resample_audio(audio_tensor: torch.Tensor, 
                      current_rate: int, 
                      target_rate: int) -> torch.Tensor:
        """
        Resample audio tensor to target sample rate.
        
        Args:
            audio_tensor (torch.Tensor): Input audio (samples, channels)
            current_rate (int): Current sample rate
            target_rate (int): Target sample rate
            
        Returns:
            torch.Tensor: Resampled audio tensor
        """
        if current_rate == target_rate:
            return audio_tensor
        
        if TORCHAUDIO_AVAILABLE:
            # Use torchaudio for resampling (most accurate)
            import torchaudio
            
            # Transpose for torchaudio (channels, samples)
            audio_t = audio_tensor.T
            resampler = torchaudio.transforms.Resample(current_rate, target_rate)
            resampled = resampler(audio_t)
            
            # Transpose back to (samples, channels)
            return resampled.T
        
        elif LIBROSA_AVAILABLE:
            # Use librosa for resampling
            audio_np = AudioProcessor.tensor_to_numpy(audio_tensor)
            
            if audio_np.shape[1] == 1:  # Mono
                resampled = librosa.resample(
                    audio_np[:, 0], 
                    orig_sr=current_rate, 
                    target_sr=target_rate
                )
                return AudioProcessor.numpy_to_tensor(resampled.reshape(-1, 1))
            else:  # Stereo
                resampled_channels = []
                for channel in range(audio_np.shape[1]):
                    resampled_ch = librosa.resample(
                        audio_np[:, channel],
                        orig_sr=current_rate,
                        target_sr=target_rate
                    )
                    resampled_channels.append(resampled_ch)
                
                resampled = np.column_stack(resampled_channels)
                return AudioProcessor.numpy_to_tensor(resampled)
        
        else:
            # Simple linear interpolation fallback
            logger.warning("No resampling library available, using linear interpolation")
            ratio = target_rate / current_rate
            new_length = int(audio_tensor.shape[0] * ratio)
            
            # Simple linear interpolation
            indices = torch.linspace(0, audio_tensor.shape[0] - 1, new_length)
            resampled = torch.zeros((new_length, audio_tensor.shape[1]))
            
            for i, idx in enumerate(indices):
                idx_floor = int(idx)
                idx_ceil = min(idx_floor + 1, audio_tensor.shape[0] - 1)
                weight = idx - idx_floor
                
                resampled[i] = (1 - weight) * audio_tensor[idx_floor] + weight * audio_tensor[idx_ceil]
            
            return resampled
    
    @staticmethod
    def normalize_audio(audio_tensor: torch.Tensor, 
                       method: str = "peak",
                       target_level: float = 0.95) -> torch.Tensor:
        """
        Normalize audio tensor.
        
        Args:
            audio_tensor (torch.Tensor): Input audio
            method (str): Normalization method ('peak', 'rms', 'lufs')
            target_level (float): Target level (0.0-1.0 for peak, dB for others)
            
        Returns:
            torch.Tensor: Normalized audio tensor
        """
        if method == "peak":
            # Peak normalization
            max_val = torch.max(torch.abs(audio_tensor))
            if max_val > 0:
                return audio_tensor * (target_level / max_val)
            return audio_tensor
        
        elif method == "rms":
            # RMS normalization
            rms = torch.sqrt(torch.mean(audio_tensor ** 2))
            if rms > 0:
                return audio_tensor * (target_level / rms)
            return audio_tensor
        
        else:
            # Default to peak normalization
            return AudioProcessor.normalize_audio(audio_tensor, "peak", target_level)
    
    @staticmethod
    def convert_to_mono(audio_tensor: torch.Tensor) -> torch.Tensor:
        """Convert stereo audio to mono by averaging channels."""
        if audio_tensor.shape[1] > 1:
            return torch.mean(audio_tensor, dim=1, keepdim=True)
        return audio_tensor
    
    @staticmethod
    def pad_or_trim_audio(audio_tensor: torch.Tensor, 
                         target_length: int, 
                         pad_value: float = 0.0) -> torch.Tensor:
        """
        Pad or trim audio to target length.
        
        Args:
            audio_tensor (torch.Tensor): Input audio
            target_length (int): Target number of samples
            pad_value (float): Value for padding
            
        Returns:
            torch.Tensor: Padded or trimmed audio
        """
        current_length = audio_tensor.shape[0]
        
        if current_length == target_length:
            return audio_tensor
        elif current_length < target_length:
            # Pad
            padding_length = target_length - current_length
            padding = torch.full((padding_length, audio_tensor.shape[1]), pad_value)
            return torch.cat([audio_tensor, padding], dim=0)
        else:
            # Trim
            return audio_tensor[:target_length]
    
    @staticmethod
    def save_audio_tensor(audio_tensor: torch.Tensor, 
                         output_path: str,
                         sample_rate: int,
                         format: str = "wav") -> bool:
        """
        Save audio tensor to file.
        
        Args:
            audio_tensor (torch.Tensor): Audio tensor to save
            output_path (str): Output file path
            sample_rate (int): Sample rate
            format (str): Audio format
            
        Returns:
            bool: Success status
        """
        try:
            if TORCHAUDIO_AVAILABLE:
                import torchaudio
                # Transpose for torchaudio (channels, samples)
                audio_t = audio_tensor.T
                torchaudio.save(output_path, audio_t, sample_rate)
                return True
            
            elif MOVIEPY_AVAILABLE:
                # Convert to numpy and use MoviePy
                audio_np = AudioProcessor.tensor_to_numpy(audio_tensor)
                
                # Create temporary AudioFileClip
                from moviepy.audio.AudioClip import AudioArrayClip
                audio_clip = AudioArrayClip(audio_np, fps=sample_rate)
                audio_clip.write_audiofile(output_path, verbose=False, logger=None)
                audio_clip.close()
                return True
            
            else:
                logger.error("No audio saving library available")
                return False
                
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return False
    
    @staticmethod
    def get_audio_info(audio_tensor: torch.Tensor, 
                      sample_rate: int) -> str:
        """
        Get formatted audio information string.
        
        Args:
            audio_tensor (torch.Tensor): Audio tensor
            sample_rate (int): Sample rate
            
        Returns:
            str: Formatted audio info
        """
        if audio_tensor.numel() == 0:
            return "Audio: Empty/No audio"
        
        samples, channels = audio_tensor.shape
        duration = samples / sample_rate
        
        channel_str = "mono" if channels == 1 else f"{channels}ch"
        
        return (
            f"Audio: {duration:.2f}s @ {sample_rate}Hz | "
            f"{samples:,} samples | {channel_str} | "
            f"Range: [{torch.min(audio_tensor):.3f}, {torch.max(audio_tensor):.3f}]"
        )


# Test functions
def test_audio_processor():
    """Test audio processor functionality."""
    processor = AudioProcessor()
    
    # Test empty audio
    empty_audio = torch.zeros((1000, 1))
    info = processor.get_audio_info(empty_audio, 22050)
    print(f"Empty audio info: {info}")
    
    # Test resampling
    test_audio = torch.randn((2205, 1))  # 0.1 seconds at 22050 Hz
    resampled = processor.resample_audio(test_audio, 22050, 44100)
    print(f"Resampling test: {test_audio.shape} -> {resampled.shape}")
    
    print("Audio processor test completed")


if __name__ == "__main__":
    test_audio_processor()