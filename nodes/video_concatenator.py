import torch
import numpy as np
from typing import List, Tuple
from .utils import (get_optimal_device, tensor_to_video_frames, optimize_batch_size, 
                   get_memory_info, logger, clear_gpu_cache, estimate_tensor_memory,
                   should_use_cpu_fallback)

class RajVideoConcatenator:
    """
    Concatenate multiple video tensors with GPU acceleration
    Supports MPS (Mac), CUDA (NVIDIA), and CPU fallback
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video1": ("IMAGE", {"tooltip": "First video frames"}),
                "video2": ("IMAGE", {"tooltip": "Second video frames"}),
            },
            "optional": {
                "video3": ("IMAGE", {"tooltip": "Optional third video"}),
                "video4": ("IMAGE", {"tooltip": "Optional fourth video"}),
                "video5": ("IMAGE", {"tooltip": "Optional fifth video"}),
                "transition_frames": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 60, 
                    "step": 1,
                    "tooltip": "Crossfade transition frames (0 = no transition)"
                }),
                "force_device": (["auto", "cpu", "cuda", "mps"], {
                    "default": "auto",
                    "tooltip": "Force specific device (auto = best available)"
                }),
                "batch_processing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use batch processing for large videos"
                }),
                "chunk_size": ("INT", {
                    "default": 32,
                    "min": 8,
                    "max": 128,
                    "step": 8,
                    "tooltip": "Chunk size for memory optimization (lower = less memory)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("concatenated_frames", "process_info", "total_frames")
    FUNCTION = "concatenate_videos"
    CATEGORY = "Raj Video Processing 🎬"
    
    def concatenate_videos(self, video1, video2, video3=None, video4=None, video5=None,
                          transition_frames=0, force_device="auto", batch_processing=True, chunk_size=32):
        
        # Determine initial device
        if force_device == "auto":
            device = get_optimal_device()
        else:
            device = torch.device(force_device)
        
        # Collect all valid video inputs
        videos = [video1, video2]
        for video in [video3, video4, video5]:
            if video is not None:
                videos.append(video)
        
        # Calculate total memory requirement
        total_frames = sum(v.shape[0] for v in videos)
        height = videos[0].shape[1]
        width = videos[0].shape[2]
        channels = videos[0].shape[3] if len(videos[0].shape) > 3 else 3
        
        estimated_memory = estimate_tensor_memory(total_frames, height, width, channels)
        
        # Check if we should use CPU fallback
        if device.type != "cpu" and should_use_cpu_fallback(device, estimated_memory):
            logger.warning(f"⚠️ Large video detected ({total_frames} frames). Using CPU to avoid memory issues.")
            device = torch.device("cpu")
            batch_processing = True  # Force batch processing on CPU
        
        logger.info(f"🔗 Concatenating {len(videos)} videos on {device}")
        logger.info(f"   Total frames: {total_frames}, Size: {width}x{height}")
        logger.info(f"   Estimated memory: {estimated_memory / 1024**3:.2f}GB")
        
        # Clear GPU cache before starting
        clear_gpu_cache(device)
        
        # Process videos
        videos_on_device = []
        for i, video in enumerate(videos):
            if not isinstance(video, torch.Tensor):
                video = torch.tensor(video, dtype=torch.float32)
            
            # For large videos, process in chunks
            if batch_processing and video.shape[0] > chunk_size * 2:
                video_device = self._move_video_chunked(video, device, chunk_size)
            else:
                video_device = video.to(device)
            
            videos_on_device.append(video_device)
            logger.info(f"   Video {i+1}: {video.shape[0]} frames, {video.shape[1]}x{video.shape[2]}")
        
        # Check dimensions consistency
        reference_height = videos_on_device[0].shape[1]
        reference_width = videos_on_device[0].shape[2]
        reference_channels = videos_on_device[0].shape[3]
        
        # Resize videos if needed
        resized_videos = []
        for i, video in enumerate(videos_on_device):
            if (video.shape[1] != reference_height or 
                video.shape[2] != reference_width or 
                video.shape[3] != reference_channels):
                
                logger.info(f"   Resizing video {i+1} from {video.shape[1]}x{video.shape[2]} to {reference_height}x{reference_width}")
                # Resize using bilinear interpolation
                video_resized = torch.nn.functional.interpolate(
                    video.permute(0, 3, 1, 2),  # [B, C, H, W]
                    size=(reference_height, reference_width),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)  # [B, H, W, C]
                resized_videos.append(video_resized)
            else:
                resized_videos.append(video)
        
        # Concatenate videos with optional transitions
        if transition_frames > 0:
            concatenated = self._concatenate_with_transitions(resized_videos, transition_frames, device)
        else:
            # Use chunked concatenation for large videos
            if batch_processing and total_frames > chunk_size * 3:
                concatenated = self._concatenate_chunked(resized_videos, device, chunk_size)
            else:
                # Simple concatenation for small videos
                concatenated = torch.cat(resized_videos, dim=0)
        
        # Clear cache after concatenation
        clear_gpu_cache(device)
        
        # Ensure proper format
        result = tensor_to_video_frames(concatenated)
        
        # Generate process info
        memory_info = get_memory_info(device)
        total_frames = result.shape[0]
        
        info_str = f"Device: {device} | " \
                  f"Videos: {len(videos)} | " \
                  f"Total frames: {total_frames} | " \
                  f"Transitions: {transition_frames} | " \
                  f"Size: {reference_width}x{reference_height}"
        
        if device.type != "cpu":
            info_str += f" | GPU Memory: {memory_info.get('allocated', 'unknown')}"
        
        logger.info(f"✅ Concatenation complete: {total_frames} frames")
        
        # Generate preview file for UI
        preview_data = None
        try:
            import cv2
            import tempfile
            import numpy as np
            
            # Create temporary preview file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                preview_path = tmp_file.name
            
            # Convert frames to numpy for preview
            frames_cpu = result.cpu().numpy()
            frames_uint8 = (frames_cpu * 255).astype(np.uint8)
            
            # Save preview video using OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(preview_path, fourcc, 24.0, (reference_width, reference_height))
            
            for frame in frames_uint8:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            # Create preview data
            preview_data = {
                "video_preview": [{
                    "path": preview_path,
                    "format": "mp4",
                    "fps": 24.0,
                    "duration": total_frames / 24.0,
                    "width": reference_width,
                    "height": reference_height,
                    "frame_count": total_frames
                }]
            }
            logger.info(f"📸 Preview saved: {preview_path}")
        except Exception as e:
            logger.warning(f"Failed to create preview: {e}")
        
        if preview_data:
            return {
                "ui": preview_data,
                "result": (result, info_str, total_frames)
            }
        else:
            return (result, info_str, total_frames)
    
    def _concatenate_with_transitions(self, videos: List[torch.Tensor], 
                                    transition_frames: int, device: torch.device) -> torch.Tensor:
        """
        Concatenate videos with crossfade transitions
        """
        if len(videos) < 2:
            return videos[0] if videos else torch.tensor([])
        
        logger.info(f"🎭 Adding {transition_frames} frame crossfade transitions")
        
        result_parts = []
        
        for i in range(len(videos)):
            current_video = videos[i]
            
            if i == 0:
                # First video: full length minus transition tail
                if len(videos) > 1:
                    result_parts.append(current_video[:-transition_frames])
                else:
                    result_parts.append(current_video)
            
            elif i == len(videos) - 1:
                # Last video: add transition then remaining frames
                prev_video = videos[i-1]
                transition_part = self._create_crossfade_transition(
                    prev_video[-transition_frames:], 
                    current_video[:transition_frames],
                    device
                )
                result_parts.append(transition_part)
                result_parts.append(current_video[transition_frames:])
            
            else:
                # Middle videos: transition + middle part (without tail)
                prev_video = videos[i-1]
                transition_part = self._create_crossfade_transition(
                    prev_video[-transition_frames:], 
                    current_video[:transition_frames],
                    device
                )
                result_parts.append(transition_part)
                result_parts.append(current_video[transition_frames:-transition_frames])
        
        return torch.cat(result_parts, dim=0)
    
    def _create_crossfade_transition(self, end_frames: torch.Tensor, 
                                   start_frames: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Create smooth crossfade between two video segments
        """
        num_frames = min(end_frames.shape[0], start_frames.shape[0])
        
        # Create alpha blending weights
        alpha = torch.linspace(1.0, 0.0, num_frames, device=device)
        alpha = alpha.view(-1, 1, 1, 1)  # Shape for broadcasting
        
        # Crossfade
        transition = alpha * end_frames[:num_frames] + (1 - alpha) * start_frames[:num_frames]
        
        return transition
    
    def _move_video_chunked(self, video: torch.Tensor, device: torch.device, chunk_size: int = 32) -> torch.Tensor:
        """
        Move video to device in chunks to avoid memory spikes
        """
        if video.shape[0] <= chunk_size:
            return video.to(device)
        
        chunks = []
        for i in range(0, video.shape[0], chunk_size):
            chunk = video[i:i+chunk_size].to(device)
            chunks.append(chunk)
            clear_gpu_cache(device)
        
        result = torch.cat(chunks, dim=0)
        return result
    
    def _concatenate_chunked(self, videos: List[torch.Tensor], device: torch.device, chunk_size: int = 32) -> torch.Tensor:
        """
        Concatenate videos in chunks to manage memory usage
        """
        logger.info(f"📦 Using chunked concatenation (chunk size: {chunk_size})")
        
        result_chunks = []
        
        for video in videos:
            # Process each video in chunks
            for i in range(0, video.shape[0], chunk_size):
                chunk = video[i:i+chunk_size]
                result_chunks.append(chunk)
                
                # Clear cache periodically
                if len(result_chunks) % 10 == 0:
                    clear_gpu_cache(device)
        
        # Concatenate all chunks
        result = torch.cat(result_chunks, dim=0)
        clear_gpu_cache(device)
        
        return result
    
    def _optimize_memory_usage(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Optimize memory usage for large tensors
        """
        if device.type == "cpu":
            return tensor
        
        # Check if we need to reduce precision or batch process
        memory_info = get_memory_info(device)
        
        if memory_info.get('free', 0) != 'unknown' and memory_info['free'] < tensor.numel() * 4:
            logger.warning("⚠️ Low GPU memory, processing in batches")
            # Process in smaller chunks if memory is low
            batch_size = optimize_batch_size(device, tensor.shape[1] * tensor.shape[2] * tensor.shape[3])
            
            if batch_size < tensor.shape[0]:
                # Process in batches
                result_chunks = []
                for i in range(0, tensor.shape[0], batch_size):
                    chunk = tensor[i:i+batch_size]
                    result_chunks.append(chunk)
                    clear_gpu_cache(device)
                
                return torch.cat(result_chunks, dim=0)
        
        return tensor

class RajVideoSequencer:
    """
    Advanced video sequencer with precise timing control
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "videos": ("IMAGE", {"tooltip": "Video frames to sequence"}),
                "sequence_config": ("STRING", {
                    "multiline": True,
                    "default": "0,2.5\n2.5,5.0\n5.0,7.5",
                    "tooltip": "Timing configuration: start_time,end_time per line (seconds)"
                }),
                "output_fps": ("FLOAT", {
                    "default": 30.0, 
                    "min": 1.0, 
                    "max": 120.0, 
                    "step": 0.1,
                    "tooltip": "Output frame rate"
                }),
            },
            "optional": {
                "force_device": (["auto", "cpu", "cuda", "mps"], {
                    "default": "auto",
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("sequenced_frames", "sequence_info", "total_frames")
    FUNCTION = "sequence_videos"
    CATEGORY = "Raj Video Processing 🎬"
    
    def sequence_videos(self, videos, sequence_config, output_fps=30.0, force_device="auto"):
        """
        Create video sequence with precise timing
        """
        if force_device == "auto":
            device = get_optimal_device()
        else:
            device = torch.device(force_device)
        
        if not isinstance(videos, torch.Tensor):
            videos = torch.tensor(videos, dtype=torch.float32)
        
        videos = videos.to(device)
        
        # Parse sequence configuration
        sequences = []
        for line in sequence_config.strip().split('\n'):
            if line.strip():
                start, end = map(float, line.strip().split(','))
                sequences.append((start, end))
        
        logger.info(f"🎬 Sequencing video with {len(sequences)} segments")
        
        # Calculate frame indices based on timing
        result_frames = []
        total_input_frames = videos.shape[0]
        input_duration = total_input_frames / output_fps  # Assume input matches output FPS
        
        for start_time, end_time in sequences:
            start_frame = int(start_time * output_fps)
            end_frame = int(end_time * output_fps)
            
            # Map to input frames
            start_input = int((start_time / input_duration) * total_input_frames)
            end_input = int((end_time / input_duration) * total_input_frames)
            
            start_input = max(0, min(start_input, total_input_frames - 1))
            end_input = max(start_input + 1, min(end_input, total_input_frames))
            
            segment_frames = videos[start_input:end_input]
            result_frames.append(segment_frames)
        
        # Concatenate all segments
        if result_frames:
            result = torch.cat(result_frames, dim=0)
        else:
            result = videos
        
        result = tensor_to_video_frames(result)
        
        info_str = f"Device: {device} | Segments: {len(sequences)} | Frames: {result.shape[0]} | FPS: {output_fps}"
        
        return (result, info_str, result.shape[0])