import os
import torch
import cv2
import numpy as np
import subprocess
import tempfile
from typing import Tuple, Optional, List
from .utils import get_optimal_device, logger

try:
    import folder_paths
except ImportError:
    # Fallback for testing outside ComfyUI
    class MockFolderPaths:
        @staticmethod
        def get_output_directory():
            return "output"
        
        @staticmethod  
        def get_temp_directory():
            return "temp"
            
    folder_paths = MockFolderPaths()

class RajVideoSaver:
    """
    Save video frames to various formats with GPU acceleration support
    Supports MP4, MOV, AVI, WebM with quality controls
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Video frames to save"}),
                "filename": ("STRING", {
                    "default": "output_video", 
                    "multiline": False,
                    "tooltip": "Output filename (without extension)"
                }),
                "fps": ("FLOAT", {
                    "default": 30.0, 
                    "min": 1.0, 
                    "max": 120.0, 
                    "step": 0.1,
                    "tooltip": "Output frame rate"
                }),
                "format": (["mp4", "mov", "avi", "webm", "gif"], {
                    "default": "mp4",
                    "tooltip": "Output video format"
                }),
            },
            "optional": {
                "quality": ("INT", {
                    "default": 23, 
                    "min": 0, 
                    "max": 51, 
                    "step": 1,
                    "tooltip": "Video quality (0=best, 51=worst) for H.264/H.265"
                }),
                "codec": (["auto", "h264", "h265", "vp9", "av1", "prores"], {
                    "default": "auto",
                    "tooltip": "Video codec (auto = format default)"
                }),
                "gpu_encoding": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPU encoding when available"
                }),
                "save_to_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Save to output directory (False = temp directory)"
                }),
                "add_timestamp": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add timestamp to filename"
                }),
                "force_device": (["auto", "cpu", "cuda", "mps"], {
                    "default": "auto",
                    "tooltip": "Force specific device for processing"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("file_path", "save_info", "total_frames", "duration")
    FUNCTION = "save_video"
    CATEGORY = "Raj Video Processing 🎬"
    OUTPUT_NODE = True
    
    def save_video(self, frames, filename, fps=30.0, format="mp4", 
                   quality=23, codec="auto", gpu_encoding=True,
                   save_to_output=True, add_timestamp=False, force_device="auto"):
        
        # Determine device for processing
        if force_device == "auto":
            device = get_optimal_device()
        else:
            device = torch.device(force_device)
        
        # Handle tensor conversion and device
        if not isinstance(frames, torch.Tensor):
            frames = torch.tensor(frames, dtype=torch.float32)
            frames = frames.to(device)
        else:
            # Keep frames on their current device if they're already processed
            # This prevents corruption when moving large tensors between devices
            current_device = frames.device
            logger.info(f"   Frames already on {current_device}, keeping them there")
            device = current_device
        
        # Prepare filename
        if add_timestamp:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename}_{timestamp}"
        
        # Determine output directory
        if save_to_output:
            output_dir = folder_paths.get_output_directory()
        else:
            output_dir = folder_paths.get_temp_directory()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Full output path
        output_path = os.path.join(output_dir, f"{filename}.{format}")
        
        logger.info(f"💾 Saving video: {output_path}")
        logger.info(f"   Device: {device} | Frames: {frames.shape[0]} | FPS: {fps}")
        logger.info(f"   Format: {format} | Quality: {quality} | GPU Encoding: {gpu_encoding}")
        
        # Determine encoding method
        if format == "gif":
            # Special handling for GIF
            file_path = self._save_as_gif(frames, output_path, fps, device)
        else:
            # Video formats
            file_path = self._save_as_video(
                frames, output_path, fps, format, quality, codec, gpu_encoding, device
            )
        
        # Calculate video info
        total_frames = frames.shape[0]
        duration = total_frames / fps
        
        save_info = f"Format: {format.upper()} | " \
                   f"Frames: {total_frames} | " \
                   f"Duration: {duration:.2f}s | " \
                   f"FPS: {fps} | " \
                   f"Device: {device} | " \
                   f"Size: {os.path.getsize(file_path) / (1024*1024):.2f}MB"
        
        logger.info(f"✅ Video saved successfully: {file_path}")
        logger.info(f"   {save_info}")
        
        return (file_path, save_info, total_frames, duration)
    
    def _save_as_gif(self, frames: torch.Tensor, output_path: str, fps: float, device: torch.device) -> str:
        """Save frames as animated GIF"""
        from PIL import Image
        
        # Convert tensor to PIL Images
        frames_cpu = frames.cpu().numpy()
        frames_uint8 = (frames_cpu * 255).astype(np.uint8)
        
        pil_frames = []
        for frame in frames_uint8:
            pil_frame = Image.fromarray(frame, 'RGB')
            pil_frames.append(pil_frame)
        
        # Save as GIF
        duration_ms = int(1000 / fps)
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True
        )
        
        return output_path
    
    def _save_as_video(self, frames: torch.Tensor, output_path: str, fps: float,
                      format: str, quality: int, codec: str, gpu_encoding: bool, device: torch.device) -> str:
        """Save frames as video using OpenCV or FFmpeg"""
        
        # Determine codec
        actual_codec = self._get_codec(format, codec, gpu_encoding, device)
        
        # Convert frames to numpy
        frames_cpu = frames.cpu().numpy()
        frames_uint8 = (frames_cpu * 255).astype(np.uint8)
        
        height, width = frames_uint8.shape[1:3]
        total_frames = frames_uint8.shape[0]
        
        logger.info(f"   Codec: {actual_codec} | Resolution: {width}x{height}")
        
        # Try GPU-accelerated encoding first if available
        if gpu_encoding and self._supports_gpu_encoding(actual_codec, device):
            try:
                return self._save_with_ffmpeg_gpu(
                    frames_uint8, output_path, fps, actual_codec, quality, device
                )
            except Exception as e:
                logger.warning(f"⚠️ GPU encoding failed, falling back to CPU: {e}")
        
        # Fallback to OpenCV/CPU encoding
        return self._save_with_opencv(frames_uint8, output_path, fps, actual_codec)
    
    def _get_codec(self, format: str, codec: str, gpu_encoding: bool, device: torch.device) -> str:
        """Determine the best codec for the format"""
        if codec != "auto":
            return codec
        
        # Auto-select codec based on format and capabilities
        codec_map = {
            "mp4": "h264_nvenc" if gpu_encoding and device.type == "cuda" else "h264",
            "mov": "prores" if format == "mov" else "h264",
            "avi": "h264",
            "webm": "vp9"
        }
        
        return codec_map.get(format, "h264")
    
    def _supports_gpu_encoding(self, codec: str, device: torch.device) -> bool:
        """Check if GPU encoding is supported for the codec"""
        gpu_codecs = {
            "cuda": ["h264_nvenc", "hevc_nvenc", "av1_nvenc"],
            "mps": [],  # MPS doesn't support video encoding yet
            "cpu": []
        }
        
        return codec in gpu_codecs.get(device.type, [])
    
    def _save_with_ffmpeg_gpu(self, frames: np.ndarray, output_path: str, fps: float,
                             codec: str, quality: int, device: torch.device) -> str:
        """Save video using FFmpeg with GPU encoding"""
        height, width = frames.shape[1:3]
        
        # FFmpeg command for GPU encoding
        cmd = [
            "ffmpeg", "-y",  # Overwrite output file
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",  # Read from stdin
            "-c:v", codec,
            "-crf", str(quality),
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        
        # Add GPU-specific options
        if "nvenc" in codec:
            cmd.extend(["-gpu", "0", "-rc", "vbr"])
        
        logger.info(f"🚀 Using FFmpeg GPU encoding: {' '.join(cmd[:-1])}")
        
        # Run FFmpeg process
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Write frames to FFmpeg
        for frame in frames:
            process.stdin.write(frame.tobytes())
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")
        
        return output_path
    
    def _save_with_opencv(self, frames: np.ndarray, output_path: str, fps: float, codec: str) -> str:
        """Save video using OpenCV (CPU encoding)"""
        height, width = frames.shape[1:3]
        
        # Map codec names to OpenCV fourcc
        fourcc_map = {
            "h264": cv2.VideoWriter_fourcc(*'H264'),
            "h265": cv2.VideoWriter_fourcc(*'HEVC'), 
            "vp9": cv2.VideoWriter_fourcc(*'VP90'),
            "prores": cv2.VideoWriter_fourcc(*'ap4h'),
            "av1": cv2.VideoWriter_fourcc(*'AV01'),
        }
        
        fourcc = fourcc_map.get(codec, cv2.VideoWriter_fourcc(*'mp4v'))
        
        logger.info(f"💻 Using OpenCV CPU encoding: {codec}")
        
        # Create video writer
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            # Fallback to mp4v codec
            logger.warning(f"⚠️ Codec {codec} not supported, using mp4v")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames (convert RGB to BGR for OpenCV)
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        return output_path

class RajVideoSaverAdvanced:
    """
    Advanced video saver with custom encoding parameters and batch processing
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Video frames to save"}),
                "filename": ("STRING", {
                    "default": "advanced_output", 
                    "multiline": False
                }),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0}),
                "width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
            },
            "optional": {
                "bitrate": ("STRING", {
                    "default": "5M",
                    "tooltip": "Target bitrate (e.g., '5M', '1000k')"
                }),
                "custom_ffmpeg_args": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Custom FFmpeg arguments"
                }),
                "audio_file": ("STRING", {
                    "default": "",
                    "tooltip": "Optional audio file to merge"
                }),
                "batch_size": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 128,
                    "tooltip": "Batch size for memory optimization"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "encoding_log")
    FUNCTION = "save_video_advanced"
    CATEGORY = "Raj Video Processing 🎬"
    OUTPUT_NODE = True
    
    def save_video_advanced(self, frames, filename, fps, width=0, height=0,
                           bitrate="5M", custom_ffmpeg_args="", audio_file="", batch_size=32):
        
        device = get_optimal_device()
        frames = frames.to(device)
        
        # Resize if specified
        if width > 0 and height > 0:
            frames = torch.nn.functional.interpolate(
                frames.permute(0, 3, 1, 2),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, f"{filename}.mp4")
        
        logger.info(f"🎬 Advanced video encoding: {output_path}")
        
        # Build FFmpeg command
        frames_np = frames.cpu().numpy()
        frames_uint8 = (frames_np * 255).astype(np.uint8)
        h, w = frames_uint8.shape[1:3]
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo", 
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-"
        ]
        
        # Add audio if specified
        if audio_file and os.path.exists(audio_file):
            cmd.extend(["-i", audio_file])
            cmd.extend(["-c:a", "aac"])
        
        # Add encoding parameters
        cmd.extend([
            "-c:v", "libx264",
            "-b:v", bitrate,
            "-pix_fmt", "yuv420p"
        ])
        
        # Add custom FFmpeg arguments
        if custom_ffmpeg_args.strip():
            cmd.extend(custom_ffmpeg_args.strip().split())
        
        cmd.append(output_path)
        
        # Run encoding
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Write frames in batches
        for i in range(0, len(frames_uint8), batch_size):
            batch = frames_uint8[i:i+batch_size]
            for frame in batch:
                process.stdin.write(frame.tobytes())
        
        stdout, stderr = process.communicate()
        
        encoding_log = stderr.decode() if stderr else "Encoding completed successfully"
        
        return (output_path, encoding_log)