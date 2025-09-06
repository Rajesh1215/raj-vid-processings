import os
import torch
import tempfile
import time
from typing import Tuple, Optional
from .utils import logger, get_optimal_device

try:
    import folder_paths
except ImportError:
    class MockFolderPaths:
        @staticmethod
        def get_temp_directory():
            return "temp"
    folder_paths = MockFolderPaths()

class RajVideoPreview:
    """
    Real-time video preview node that displays video frames and audio without saving to disk.
    Provides instant preview with playback controls, seeking, and metadata display.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Video frames to preview"
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0, 
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Frame rate for preview"
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Optional audio track to preview with video"
                }),
                "preview_format": (["mp4", "webm"], {
                    "default": "mp4",
                    "tooltip": "Preview video format (MP4 for better compatibility)"
                }),
                "preview_quality": (["high", "medium", "low"], {
                    "default": "medium",
                    "tooltip": "Preview quality (affects file size and loading speed)"
                }),
                "max_preview_duration": ("FLOAT", {
                    "default": 30.0,
                    "min": 5.0,
                    "max": 300.0,
                    "step": 5.0,
                    "tooltip": "Maximum preview duration in seconds (longer videos will be trimmed)"
                }),
                "auto_play": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Automatically start playback when preview loads"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("preview_info", "file_path", "frame_count", "duration", "metadata")
    FUNCTION = "create_preview"
    CATEGORY = "Raj Video Processing 🎬"
    OUTPUT_NODE = True
    
    def create_preview(self, frames, fps=24.0, audio=None, preview_format="mp4", 
                      preview_quality="medium", max_preview_duration=30.0, auto_play=False):
        """
        Create a real-time video preview with optional audio
        """
        
        device = get_optimal_device()
        
        # Handle tensor conversion
        if not isinstance(frames, torch.Tensor):
            frames = torch.tensor(frames, dtype=torch.float32)
        
        frames = frames.to(device)
        
        # Get frame info
        total_frames = frames.shape[0]
        duration = total_frames / fps
        height, width = frames.shape[1:3]
        channels = frames.shape[3] if len(frames.shape) > 3 else 3
        
        logger.info(f"🎬 Creating video preview: {total_frames} frames, {duration:.2f}s at {fps}fps")
        logger.info(f"   Resolution: {width}x{height}, Format: {preview_format}")
        
        # Handle audio processing
        audio_info = ""
        audio_tensor = None
        audio_sr = None
        
        if audio is not None:
            if isinstance(audio, dict):
                audio_tensor = audio["waveform"].squeeze(0).transpose(0, 1)  # Convert to [samples, channels]
                audio_sr = audio["sample_rate"]
                audio_duration = audio_tensor.shape[0] / audio_sr
                audio_channels = audio_tensor.shape[1]
                audio_info = f"Audio: {audio_duration:.2f}s, {audio_sr}Hz, {audio_channels}ch"
                logger.info(f"🔊 {audio_info}")
            else:
                logger.warning("⚠️ Audio format not recognized, skipping audio preview")
        
        # Limit preview duration if needed
        if duration > max_preview_duration:
            max_frames = int(max_preview_duration * fps)
            frames = frames[:max_frames]
            total_frames = max_frames
            duration = max_preview_duration
            
            # Also trim audio if present
            if audio_tensor is not None:
                max_audio_samples = int(max_preview_duration * audio_sr)
                audio_tensor = audio_tensor[:max_audio_samples]
            
            logger.info(f"⏱️ Trimmed to {max_preview_duration}s for preview")
        
        # Adjust quality settings
        quality_settings = {
            "high": {"crf": 18, "preset": "slow", "max_width": 1920},
            "medium": {"crf": 23, "preset": "medium", "max_width": 1280}, 
            "low": {"crf": 28, "preset": "fast", "max_width": 854}
        }
        
        settings = quality_settings[preview_quality]
        
        # Scale down if needed for performance
        if width > settings["max_width"]:
            scale_factor = settings["max_width"] / width
            new_width = settings["max_width"]
            new_height = int(height * scale_factor)
            
            # Ensure even dimensions for video encoding
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            logger.info(f"📐 Scaling preview: {width}x{height} → {new_width}x{new_height}")
            
            # Resize frames
            frames = torch.nn.functional.interpolate(
                frames.permute(0, 3, 1, 2).float(),  # [batch, height, width, channels] → [batch, channels, height, width]
                size=(new_height, new_width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)  # Back to [batch, height, width, channels]
            
            width, height = new_width, new_height
        
        # Create temporary preview file
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        
        timestamp = str(int(time.time()))
        preview_filename = f"raj_preview_{timestamp}.{preview_format}"
        preview_path = os.path.join(temp_dir, preview_filename)
        
        # Generate preview video
        try:
            if preview_format == "mp4":
                file_path = self._create_mp4_preview(
                    frames, preview_path, fps, audio_tensor, audio_sr, settings, device
                )
            elif preview_format == "webm":
                file_path = self._create_webm_preview(
                    frames, preview_path, fps, audio_tensor, audio_sr, settings, device
                )
            else:
                raise ValueError(f"Unsupported preview format: {preview_format}")
                
        except Exception as e:
            logger.error(f"❌ Preview generation failed: {e}")
            # Create a fallback static image preview
            file_path = self._create_fallback_preview(frames, preview_path, device)
        
        # Generate metadata
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else 0
        
        metadata = (
            f"Preview: {preview_format.upper()} | "
            f"Resolution: {width}x{height} | "
            f"Quality: {preview_quality} | "
            f"Size: {file_size_mb:.2f}MB | "
            f"Device: {device}"
        )
        
        if audio_info:
            metadata += f" | {audio_info}"
        
        preview_info = f"Video preview ready: {total_frames} frames, {duration:.2f}s"
        
        # Prepare UI preview data for the JavaScript widget
        preview_data = {
            "filename": preview_filename,
            "subfolder": "",
            "type": "temp",
            "format": f"video/{preview_format}",
            "frame_rate": fps,
            "frame_count": total_frames,
            "auto_play": auto_play,
            "duration": duration
        }
        
        logger.info(f"✅ Video preview created: {file_path}")
        logger.info(f"   {metadata}")
        
        return {
            "ui": {"gifs": [preview_data]},  # Use "gifs" for compatibility with existing preview systems
            "result": (preview_info, file_path, total_frames, duration, metadata)
        }
    
    def _create_mp4_preview(self, frames, output_path, fps, audio_tensor, audio_sr, settings, device):
        """Create MP4 preview using OpenCV and FFmpeg"""
        import cv2
        import subprocess
        
        # Convert frames to numpy
        frames_cpu = frames.cpu().numpy()
        frames_uint8 = (frames_cpu * 255).astype('uint8')
        
        height, width = frames_uint8.shape[1:3]
        
        if audio_tensor is not None:
            # Use FFmpeg for video + audio encoding
            return self._create_video_with_ffmpeg(
                frames_uint8, output_path, fps, audio_tensor, audio_sr, settings, "mp4"
            )
        else:
            # Use OpenCV for video-only encoding
            return self._create_video_with_opencv(frames_uint8, output_path, fps, settings)
    
    def _create_webm_preview(self, frames, output_path, fps, audio_tensor, audio_sr, settings, device):
        """Create WebM preview using FFmpeg"""
        frames_cpu = frames.cpu().numpy()
        frames_uint8 = (frames_cpu * 255).astype('uint8')
        
        return self._create_video_with_ffmpeg(
            frames_uint8, output_path, fps, audio_tensor, audio_sr, settings, "webm"
        )
    
    def _create_video_with_opencv(self, frames_uint8, output_path, fps, settings):
        """Create video using OpenCV (video only)"""
        import cv2
        
        height, width = frames_uint8.shape[1:3]
        
        # Use MP4V codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError("Failed to open video writer")
        
        logger.info(f"🎥 Encoding video with OpenCV: {width}x{height} at {fps}fps")
        
        for frame in frames_uint8:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        return output_path
    
    def _create_video_with_ffmpeg(self, frames_uint8, output_path, fps, audio_tensor, audio_sr, settings, format_type):
        """Create video with optional audio using FFmpeg"""
        import subprocess
        import tempfile
        
        height, width = frames_uint8.shape[1:3]
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y",  # Overwrite output
            "-f", "rawvideo",
            "-vcodec", "rawvideo", 
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-"  # Read video from stdin
        ]
        
        # Add audio input if present
        audio_temp_path = None
        if audio_tensor is not None:
            # Save audio to temporary WAV file
            audio_temp_path = tempfile.mktemp(suffix='.wav')
            self._save_audio_wav(audio_tensor, audio_temp_path, audio_sr)
            cmd.extend(["-i", audio_temp_path])
        
        # Video encoding settings
        if format_type == "mp4":
            cmd.extend([
                "-c:v", "libx264",
                "-crf", str(settings["crf"]),
                "-preset", settings["preset"],
                "-pix_fmt", "yuv420p"
            ])
        elif format_type == "webm":
            cmd.extend([
                "-c:v", "libvpx-vp9",
                "-crf", str(settings["crf"]),
                "-b:v", "0"  # Constant quality mode
            ])
        
        # Audio encoding settings
        if audio_tensor is not None:
            if format_type == "mp4":
                cmd.extend(["-c:a", "aac", "-b:a", "128k"])
            elif format_type == "webm":
                cmd.extend(["-c:a", "libopus", "-b:a", "128k"])
        
        cmd.append(output_path)
        
        logger.info(f"🚀 Encoding with FFmpeg: {format_type.upper()}")
        
        # Run FFmpeg
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Write video frames
        for frame in frames_uint8:
            process.stdin.write(frame.tobytes())
        
        stdout, stderr = process.communicate()
        
        # Clean up temporary audio file
        if audio_temp_path and os.path.exists(audio_temp_path):
            os.remove(audio_temp_path)
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown FFmpeg error"
            raise RuntimeError(f"FFmpeg encoding failed: {error_msg}")
        
        return output_path
    
    def _save_audio_wav(self, audio_tensor, output_path, sample_rate):
        """Save audio tensor as WAV file"""
        try:
            import torchaudio
            # Transpose back to [channels, samples] for torchaudio
            audio_for_save = audio_tensor.transpose(0, 1)
            torchaudio.save(output_path, audio_for_save, sample_rate)
        except ImportError:
            # Fallback using scipy
            try:
                from scipy.io import wavfile
                import numpy as np
                
                # Convert to numpy and ensure proper format
                audio_np = audio_tensor.cpu().numpy()
                if audio_np.shape[1] == 1:
                    audio_np = audio_np.squeeze(1)  # Convert mono to 1D
                
                # Scale to 16-bit integer range
                audio_int16 = (audio_np * 32767).astype(np.int16)
                wavfile.write(output_path, sample_rate, audio_int16)
                
            except ImportError:
                logger.warning("⚠️ Neither torchaudio nor scipy available, skipping audio")
                raise
    
    def _create_fallback_preview(self, frames, output_path, device):
        """Create a static image preview as fallback"""
        from PIL import Image
        import numpy as np
        
        # Use the first frame as preview image
        first_frame = frames[0].cpu().numpy()
        frame_uint8 = (first_frame * 255).astype('uint8')
        
        # Convert to PIL Image and save as JPEG
        image = Image.fromarray(frame_uint8, 'RGB')
        fallback_path = output_path.replace('.mp4', '.jpg').replace('.webm', '.jpg')
        image.save(fallback_path, 'JPEG', quality=85)
        
        logger.info(f"📷 Created fallback image preview: {fallback_path}")
        return fallback_path

class RajVideoPreviewAdvanced:
    """
    Advanced video preview with custom encoding options and multiple format outputs
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Video frames to preview"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Optional audio track"}),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 1920, "step": 8}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 1080, "step": 8}),
                "bitrate": ("STRING", {"default": "2M", "tooltip": "Video bitrate (e.g., '2M', '500k')"}),
                "custom_codec": (["auto", "h264", "h265", "vp9", "av1"], {"default": "auto"}),
                "preview_segments": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 10,
                    "tooltip": "Split preview into segments for large videos"
                }),
                "start_frame": ("INT", {"default": 0, "min": 0, "tooltip": "Starting frame for preview"}),
                "end_frame": ("INT", {"default": -1, "min": -1, "tooltip": "Ending frame (-1 for all)"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("preview_info", "file_paths", "encoding_log")
    FUNCTION = "create_advanced_preview"
    CATEGORY = "Raj Video Processing 🎬"
    OUTPUT_NODE = True
    
    def create_advanced_preview(self, frames, fps, audio=None, custom_width=0, custom_height=0,
                               bitrate="2M", custom_codec="auto", preview_segments=1, 
                               start_frame=0, end_frame=-1):
        """
        Create advanced preview with custom encoding options
        """
        
        device = get_optimal_device()
        frames = frames.to(device)
        
        # Handle frame range selection
        total_frames = frames.shape[0]
        if end_frame == -1:
            end_frame = total_frames
        
        end_frame = min(end_frame, total_frames)
        start_frame = max(0, min(start_frame, end_frame - 1))
        
        frames = frames[start_frame:end_frame]
        selected_frames = frames.shape[0]
        duration = selected_frames / fps
        
        logger.info(f"🎬 Advanced preview: frames {start_frame}-{end_frame} ({selected_frames} frames, {duration:.2f}s)")
        
        # Handle custom resolution
        if custom_width > 0 and custom_height > 0:
            frames = torch.nn.functional.interpolate(
                frames.permute(0, 3, 1, 2).float(),
                size=(custom_height, custom_width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
            logger.info(f"📐 Custom resolution: {custom_width}x{custom_height}")
        
        # Create multiple preview segments if requested
        preview_files = []
        encoding_logs = []
        
        if preview_segments > 1:
            segment_size = selected_frames // preview_segments
            for i in range(preview_segments):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, selected_frames)
                
                segment_frames = frames[start_idx:end_idx]
                segment_path, log = self._create_preview_segment(
                    segment_frames, fps, audio, bitrate, custom_codec, device, i + 1
                )
                
                preview_files.append(segment_path)
                encoding_logs.append(log)
        else:
            # Single preview file
            preview_path, log = self._create_preview_segment(
                frames, fps, audio, bitrate, custom_codec, device, 1
            )
            preview_files.append(preview_path)
            encoding_logs.append(log)
        
        preview_info = f"Advanced preview: {len(preview_files)} segments, {selected_frames} frames"
        file_paths = " | ".join(preview_files)
        encoding_log = "\n---\n".join(encoding_logs)
        
        # Prepare UI preview data (use first segment)
        if preview_files:
            preview_data = {
                "filename": os.path.basename(preview_files[0]),
                "subfolder": "",
                "type": "temp",
                "format": "video/mp4",
                "frame_rate": fps,
                "frame_count": selected_frames
            }
            
            return {
                "ui": {"gifs": [preview_data]},
                "result": (preview_info, file_paths, encoding_log)
            }
        
        return (preview_info, file_paths, encoding_log)
    
    def _create_preview_segment(self, frames, fps, audio, bitrate, codec, device, segment_num):
        """Create a single preview segment"""
        import subprocess
        import tempfile
        
        timestamp = str(int(time.time()))
        segment_filename = f"raj_advanced_preview_{timestamp}_{segment_num}.mp4"
        temp_dir = folder_paths.get_temp_directory()
        segment_path = os.path.join(temp_dir, segment_filename)
        
        frames_cpu = frames.cpu().numpy()
        frames_uint8 = (frames_cpu * 255).astype('uint8')
        height, width = frames_uint8.shape[1:3]
        
        # Build advanced FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24", 
            "-r", str(fps),
            "-i", "-"
        ]
        
        # Codec selection
        if codec == "auto" or codec == "h264":
            cmd.extend(["-c:v", "libx264", "-crf", "23"])
        elif codec == "h265":
            cmd.extend(["-c:v", "libx265", "-crf", "23"])
        elif codec == "vp9":
            cmd.extend(["-c:v", "libvpx-vp9", "-crf", "23"])
        elif codec == "av1":
            cmd.extend(["-c:v", "libaom-av1", "-crf", "23"])
        
        # Bitrate control
        cmd.extend(["-b:v", bitrate])
        cmd.extend(["-pix_fmt", "yuv420p"])
        cmd.append(segment_path)
        
        # Execute FFmpeg
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        for frame in frames_uint8:
            process.stdin.write(frame.tobytes())
        
        stdout, stderr = process.communicate()
        
        encoding_log = f"Segment {segment_num}: {codec} @ {bitrate}\n"
        if stderr:
            encoding_log += stderr.decode()
        
        return segment_path, encoding_log