import torch
import numpy as np
import cv2
import tempfile
import os
from typing import Tuple, Optional, Dict
from .utils import logger, tensor_to_video_frames
from ..utils.audio_utils import AudioProcessor

class RajAudioCombiner:
    """
    Combine video frames with audio track.
    Replaces or adds audio to video, ensuring perfect synchronization.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Video frames tensor"
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Audio track to combine with video"
                }),
                "video_fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Video frame rate"
                }),
                "audio_sample_rate": ("INT", {
                    "default": 22050,
                    "min": 8000,
                    "max": 96000,
                    "step": 1000,
                    "tooltip": "Audio sample rate"
                }),
                "sync_method": (["auto", "manual_offset", "stretch_audio", "trim_to_match"], {
                    "default": "auto",
                    "tooltip": "How to synchronize audio with video"
                }),
            },
            "optional": {
                "audio_offset": ("FLOAT", {
                    "default": 0.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Manual audio offset in seconds (+ delays audio)"
                }),
                "preserve_original_audio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Mix with original audio instead of replacing"
                }),
                "mix_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Mix ratio when preserving original (0=original, 1=new)"
                }),
                "output_format": (["mp4", "mov", "avi"], {
                    "default": "mp4",
                    "tooltip": "Output video format"
                }),
                "audio_codec": (["aac", "mp3", "pcm"], {
                    "default": "aac",
                    "tooltip": "Audio codec for output"
                }),
                "video_codec": (["h264", "h265", "prores"], {
                    "default": "h264",
                    "tooltip": "Video codec for output"
                }),
            }
        }
    
    RETURN_TYPES = ("VIDEO_TENSOR", "STRING", "STRING")
    RETURN_NAMES = ("video_with_audio", "sync_info", "output_path")
    FUNCTION = "combine_audio_video"
    CATEGORY = "Raj Video Processing 🎬/Audio"
    
    def combine_audio_video(self, video_frames: torch.Tensor, audio: torch.Tensor,
                           video_fps: float, audio_sample_rate: int, sync_method: str,
                           audio_offset: float = 0.0, preserve_original_audio: bool = False,
                           mix_ratio: float = 0.5, output_format: str = "mp4",
                           audio_codec: str = "aac", video_codec: str = "h264") -> Tuple[torch.Tensor, str, str]:
        """
        Combine video frames with audio track.
        """
        
        # Validate inputs
        if video_frames.numel() == 0:
            logger.error("Video frames are empty")
            return (video_frames, "Empty video frames", "")
        
        if audio.numel() == 0:
            logger.error("Audio is empty")
            return (video_frames, "Empty audio", "")
        
        # Calculate video duration
        num_frames = video_frames.shape[0]
        video_duration = num_frames / video_fps
        
        # Calculate audio duration
        audio_samples = audio.shape[0]
        audio_duration = audio_samples / audio_sample_rate
        
        logger.info(f"🎬 Combining video ({video_duration:.2f}s) with audio ({audio_duration:.2f}s)")
        
        # Synchronize audio with video based on method
        if sync_method == "auto":
            # Automatically adjust audio to match video duration
            if abs(video_duration - audio_duration) > 0.1:  # More than 100ms difference
                logger.info(f"⚡ Auto-syncing: adjusting audio from {audio_duration:.2f}s to {video_duration:.2f}s")
                target_samples = int(video_duration * audio_sample_rate)
                audio = self._adjust_audio_length(audio, target_samples)
                audio_duration = video_duration
        
        elif sync_method == "manual_offset":
            # Apply manual offset
            if audio_offset != 0.0:
                offset_samples = int(abs(audio_offset) * audio_sample_rate)
                if audio_offset > 0:
                    # Delay audio (add silence at beginning)
                    silence = torch.zeros((offset_samples, audio.shape[1]))
                    audio = torch.cat([silence, audio], dim=0)
                else:
                    # Advance audio (trim from beginning)
                    if offset_samples < audio.shape[0]:
                        audio = audio[offset_samples:]
                    else:
                        audio = torch.zeros((1, audio.shape[1]))
                
                logger.info(f"📍 Applied audio offset: {audio_offset:.2f}s")
        
        elif sync_method == "stretch_audio":
            # Stretch or compress audio to match video duration
            if abs(video_duration - audio_duration) > 0.01:
                stretch_ratio = video_duration / audio_duration
                logger.info(f"🔄 Stretching audio by {stretch_ratio:.2f}x")
                audio = self._time_stretch_audio(audio, stretch_ratio, audio_sample_rate)
        
        elif sync_method == "trim_to_match":
            # Trim longer track to match shorter
            min_duration = min(video_duration, audio_duration)
            target_samples = int(min_duration * audio_sample_rate)
            target_frames = int(min_duration * video_fps)
            
            if audio.shape[0] > target_samples:
                audio = audio[:target_samples]
            if video_frames.shape[0] > target_frames:
                video_frames = video_frames[:target_frames]
            
            logger.info(f"✂️ Trimmed to {min_duration:.2f}s")
        
        # Create temporary files for processing
        temp_video_path = None
        temp_audio_path = None
        output_path = None
        
        try:
            # Save video frames to temporary file
            with tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False) as temp_video:
                temp_video_path = temp_video.name
            
            # Write video frames
            self._write_video_frames(video_frames, temp_video_path, video_fps, video_codec)
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Save audio
            success = AudioProcessor.save_audio_tensor(audio, temp_audio_path, audio_sample_rate)
            if not success:
                raise RuntimeError("Failed to save audio")
            
            # Create output file
            with tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False) as output_file:
                output_path = output_file.name
            
            # Combine using ffmpeg or moviepy
            if self._combine_with_ffmpeg(temp_video_path, temp_audio_path, output_path, 
                                        audio_codec, preserve_original_audio, mix_ratio):
                logger.info(f"✅ Successfully combined video and audio")
            else:
                # Fallback to moviepy
                self._combine_with_moviepy(temp_video_path, temp_audio_path, output_path,
                                          preserve_original_audio, mix_ratio)
            
            # Load the combined video back as tensor
            combined_tensor = self._load_video_as_tensor(output_path)
            
            # Generate sync info
            final_duration = combined_tensor.shape[0] / video_fps
            sync_info = (
                f"Audio-Video Combination Complete\n"
                f"Video Duration: {video_duration:.2f}s ({num_frames} frames @ {video_fps:.1f} fps)\n"
                f"Audio Duration: {audio_duration:.2f}s ({audio_samples:,} samples @ {audio_sample_rate} Hz)\n"
                f"Final Duration: {final_duration:.2f}s\n"
                f"Sync Method: {sync_method}\n"
                f"Audio Offset: {audio_offset:.2f}s\n"
                f"Original Audio: {'Mixed' if preserve_original_audio else 'Replaced'}"
                + (f" (ratio: {mix_ratio:.2f})" if preserve_original_audio else "") + "\n"
                f"Output Format: {output_format} ({video_codec}/{audio_codec})"
            )
            
            logger.info(f"🎉 Audio-video combination complete: {final_duration:.2f}s")
            
            return (combined_tensor, sync_info, output_path)
            
        except Exception as e:
            logger.error(f"Error combining audio and video: {e}")
            return (video_frames, f"Combination failed: {str(e)}", "")
        
        finally:
            # Cleanup temporary files
            for path in [temp_video_path, temp_audio_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass
    
    def _adjust_audio_length(self, audio: torch.Tensor, target_samples: int) -> torch.Tensor:
        """Adjust audio length to target samples."""
        current_samples = audio.shape[0]
        
        if current_samples == target_samples:
            return audio
        elif current_samples < target_samples:
            # Pad with silence
            padding = torch.zeros((target_samples - current_samples, audio.shape[1]))
            return torch.cat([audio, padding], dim=0)
        else:
            # Trim
            return audio[:target_samples]
    
    def _time_stretch_audio(self, audio: torch.Tensor, stretch_ratio: float, 
                           sample_rate: int) -> torch.Tensor:
        """Time stretch audio without changing pitch."""
        try:
            import librosa
            
            # Convert to numpy
            audio_np = audio.numpy()
            
            # Process each channel
            stretched_channels = []
            for ch in range(audio_np.shape[1]):
                channel_data = audio_np[:, ch]
                # Use librosa's time stretching
                stretched = librosa.effects.time_stretch(channel_data, rate=stretch_ratio)
                stretched_channels.append(stretched)
            
            # Stack channels
            if len(stretched_channels) > 1:
                stretched_np = np.stack(stretched_channels, axis=1)
            else:
                stretched_np = stretched_channels[0].reshape(-1, 1)
            
            return torch.from_numpy(stretched_np).float()
            
        except ImportError:
            # Fallback to simple resampling
            logger.warning("Librosa not available, using simple resampling")
            target_samples = int(audio.shape[0] * stretch_ratio)
            return self._adjust_audio_length(audio, target_samples)
    
    def _write_video_frames(self, frames: torch.Tensor, output_path: str, 
                           fps: float, codec: str):
        """Write video frames to file."""
        frames_np = tensor_to_video_frames(frames)
        height, width = frames_np[0].shape[:2]
        
        # Set up video writer
        fourcc_map = {
            'h264': cv2.VideoWriter_fourcc(*'H264'),
            'h265': cv2.VideoWriter_fourcc(*'HEVC'),
            'prores': cv2.VideoWriter_fourcc(*'mp4v'),
        }
        fourcc = fourcc_map.get(codec, cv2.VideoWriter_fourcc(*'mp4v'))
        
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames_np:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        writer.release()
    
    def _combine_with_ffmpeg(self, video_path: str, audio_path: str, output_path: str,
                            audio_codec: str, preserve_original: bool, mix_ratio: float) -> bool:
        """Combine video and audio using ffmpeg."""
        try:
            import subprocess
            
            # Build ffmpeg command
            cmd = ['ffmpeg', '-y']  # Overwrite output
            
            if preserve_original:
                # Mix with original audio
                cmd.extend([
                    '-i', video_path,
                    '-i', audio_path,
                    '-filter_complex', f'[0:a][1:a]amix=inputs=2:duration=longest:weights={1-mix_ratio} {mix_ratio}',
                    '-c:v', 'copy',
                    '-c:a', audio_codec,
                    output_path
                ])
            else:
                # Replace audio
                cmd.extend([
                    '-i', video_path,
                    '-i', audio_path,
                    '-c:v', 'copy',
                    '-c:a', audio_codec,
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    output_path
                ])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            logger.warning(f"FFmpeg combination failed: {e}")
            return False
    
    def _combine_with_moviepy(self, video_path: str, audio_path: str, output_path: str,
                             preserve_original: bool, mix_ratio: float):
        """Combine video and audio using moviepy."""
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip
            
            video = VideoFileClip(video_path)
            new_audio = AudioFileClip(audio_path)
            
            if preserve_original and video.audio is not None:
                # Mix audios
                original_audio = video.audio
                mixed_audio = original_audio.volumex(1 - mix_ratio).overlay(new_audio.volumex(mix_ratio))
                video = video.set_audio(mixed_audio)
            else:
                # Replace audio
                video = video.set_audio(new_audio)
            
            video.write_videofile(output_path, codec='libx264', audio_codec='aac', 
                                verbose=False, logger=None)
            
            video.close()
            new_audio.close()
            
        except Exception as e:
            logger.error(f"MoviePy combination failed: {e}")
            raise
    
    def _load_video_as_tensor(self, video_path: str) -> torch.Tensor:
        """Load video file back as tensor."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        
        # Convert to tensor
        frames_np = np.stack(frames)
        frames_tensor = torch.from_numpy(frames_np).float() / 255.0
        
        return frames_tensor


class RajAudioExtractor:
    """
    Extract audio from video at any point in the workflow.
    Can be used to extract audio from processed video for further manipulation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Video frames to extract audio from"
                }),
                "video_fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Frame rate of the video"
                }),
                "target_sample_rate": ("INT", {
                    "default": 22050,
                    "min": 8000,
                    "max": 96000,
                    "step": 1000,
                    "tooltip": "Target sample rate for extracted audio"
                }),
                "mono": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Convert to mono"
                }),
            },
            "optional": {
                "original_audio": ("AUDIO", {
                    "tooltip": "Optional: Pass through original audio if video has none"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "audio_info")
    FUNCTION = "extract_audio"
    CATEGORY = "Raj Video Processing 🎬/Audio"
    
    def extract_audio(self, video_frames: torch.Tensor, video_fps: float,
                     target_sample_rate: int, mono: bool,
                     original_audio: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, str]:
        """
        Extract or generate audio from video frames.
        """
        
        # In ComfyUI, video frames don't carry audio
        # This node would typically be used with original audio pass-through
        # or to generate silence for the video duration
        
        if original_audio is not None and original_audio.numel() > 0:
            # Pass through original audio
            logger.info("📤 Using provided original audio")
            
            audio_info = (
                f"Audio Extraction\n"
                f"Source: Original audio provided\n"
                f"Samples: {original_audio.shape[0]:,}\n"
                f"Channels: {original_audio.shape[1]}\n"
                f"Duration: {original_audio.shape[0] / target_sample_rate:.2f}s"
            )
            
            return (original_audio, audio_info)
        else:
            # Generate silence matching video duration
            num_frames = video_frames.shape[0]
            video_duration = num_frames / video_fps
            num_samples = int(video_duration * target_sample_rate)
            num_channels = 1 if mono else 2
            
            silence = torch.zeros((num_samples, num_channels))
            
            logger.info(f"🔇 Generated silence for {video_duration:.2f}s video")
            
            audio_info = (
                f"Audio Extraction\n"
                f"Source: Generated silence\n"
                f"Video Duration: {video_duration:.2f}s ({num_frames} frames)\n"
                f"Audio Samples: {num_samples:,}\n"
                f"Sample Rate: {target_sample_rate} Hz\n"
                f"Channels: {num_channels} ({'mono' if mono else 'stereo'})"
            )
            
            return (silence, audio_info)