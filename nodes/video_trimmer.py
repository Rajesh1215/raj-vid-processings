"""
RajVideoTrimmer - Precise video trimming with time-based controls
Extract specific time segments from video with frame-accurate precision
"""

import torch
import numpy as np
import os
import tempfile
from typing import Tuple, Optional
from .utils import (
    get_optimal_device, logger, time_to_frame, frame_to_time,
    get_save_path_incremental
)

class RajVideoTrimmer:
    """
    Trim video to extract specific time segment with precise timing controls
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Input video frames"}),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Frames per second for time calculations"
                }),
                "start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Start time in seconds (e.g., 2.5)"
                }),
                "end_time": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.1,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "End time in seconds (e.g., 8.5)"
                }),
                "time_format": (["seconds", "timecode"], {
                    "default": "seconds",
                    "tooltip": "Time format: seconds (5.5) or timecode (00:00:05:12)"
                })
            },
            "optional": {
                "start_timecode": ("STRING", {
                    "default": "00:00:00:00",
                    "tooltip": "Start timecode (HH:MM:SS:FF) - used if time_format is 'timecode'"
                }),
                "end_timecode": ("STRING", {
                    "default": "00:00:05:00",
                    "tooltip": "End timecode (HH:MM:SS:FF) - used if time_format is 'timecode'"
                }),
                "force_device": (["auto", "cpu", "cuda", "mps"], {
                    "default": "auto",
                    "tooltip": "Force specific device for processing"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("trimmed_frames", "trim_info", "frame_count", "duration")
    FUNCTION = "trim_video"
    CATEGORY = "Raj Video Processing 🎬"
    
    def trim_video(self, frames, fps=24.0, start_time=0.0, end_time=5.0, 
                   time_format="seconds", start_timecode="00:00:00:00", 
                   end_timecode="00:00:05:00", force_device="auto"):
        
        # Determine device
        if force_device == "auto":
            device = get_optimal_device()
        else:
            device = torch.device(force_device)
        
        # Convert frames to tensor if needed
        if not isinstance(frames, torch.Tensor):
            frames = torch.tensor(frames, dtype=torch.float32)
        
        frames = frames.to(device)
        total_frames = frames.shape[0]
        total_duration = total_frames / fps
        
        logger.info(f"✂️ Trimming video: {total_frames} frames @ {fps}fps ({total_duration:.2f}s)")
        
        # Parse timing based on format
        if time_format == "timecode":
            actual_start_time = self._parse_timecode(start_timecode, fps)
            actual_end_time = self._parse_timecode(end_timecode, fps)
            logger.info(f"   Timecode format: {start_timecode} to {end_timecode}")
        else:
            actual_start_time = start_time
            actual_end_time = end_time
            logger.info(f"   Time range: {actual_start_time}s to {actual_end_time}s")
        
        # Validate timing
        if actual_start_time >= actual_end_time:
            raise ValueError(f"Start time ({actual_start_time}s) must be less than end time ({actual_end_time}s)")
        
        if actual_start_time >= total_duration:
            raise ValueError(f"Start time ({actual_start_time}s) is beyond video duration ({total_duration:.2f}s)")
        
        # Convert times to frame indices
        start_frame = max(0, time_to_frame(actual_start_time, fps))
        end_frame = min(total_frames, time_to_frame(actual_end_time, fps))
        
        # Ensure we have at least 1 frame
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        
        # Extract trimmed segment
        trimmed_frames = frames[start_frame:end_frame]
        trimmed_count = trimmed_frames.shape[0]
        trimmed_duration = trimmed_count / fps
        
        # Calculate actual times (might be adjusted due to frame boundaries)
        actual_start = frame_to_time(start_frame, fps)
        actual_end = frame_to_time(end_frame, fps)
        
        logger.info(f"   Trimmed: frames {start_frame}-{end_frame} ({trimmed_count} frames, {trimmed_duration:.2f}s)")
        logger.info(f"   Actual times: {actual_start:.2f}s to {actual_end:.2f}s")
        
        # Generate info string
        trim_info = (f"Trimmed: {actual_start:.2f}s-{actual_end:.2f}s | "
                    f"Frames: {start_frame}-{end_frame} ({trimmed_count}) | "
                    f"Duration: {trimmed_duration:.2f}s @ {fps}fps | "
                    f"Original: {total_duration:.2f}s")
        
        # Generate preview file for UI
        preview_data = None
        try:
            # Create temporary preview file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                preview_path = tmp_file.name
            
            # Save preview using OpenCV
            frames_cpu = trimmed_frames.cpu().numpy()
            frames_uint8 = (frames_cpu * 255).astype(np.uint8)
            
            import cv2
            height, width = frames_uint8.shape[1:3]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(preview_path, fourcc, fps, (width, height))
            
            for frame in frames_uint8:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            preview_data = {
                "gifs": [{
                    "filename": os.path.basename(preview_path),
                    "subfolder": "",
                    "type": "temp",
                    "format": "video/mp4"
                }]
            }
            logger.info(f"📸 Trim preview saved: {preview_path}")
        except Exception as e:
            logger.warning(f"Failed to create trim preview: {e}")
        
        logger.info("✅ Video trim complete")
        
        if preview_data:
            return {
                "ui": preview_data,
                "result": (trimmed_frames, trim_info, trimmed_count, trimmed_duration)
            }
        else:
            return (trimmed_frames, trim_info, trimmed_count, trimmed_duration)
    
    def _parse_timecode(self, timecode: str, fps: float) -> float:
        """
        Parse timecode string (HH:MM:SS:FF) to seconds
        """
        try:
            parts = timecode.split(":")
            if len(parts) != 4:
                raise ValueError(f"Invalid timecode format: {timecode}. Expected HH:MM:SS:FF")
            
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            frames = int(parts[3])
            
            # Convert to total seconds
            total_seconds = (hours * 3600) + (minutes * 60) + seconds + (frames / fps)
            
            return total_seconds
            
        except Exception as e:
            logger.error(f"Failed to parse timecode '{timecode}': {e}")
            raise ValueError(f"Invalid timecode format: {timecode}")


class RajVideoCutter:
    """
    Cut video segments and output both remaining video and removed segment
    8s video: cut 2s-6s → remaining: 0-2s+6-8s (4s), removed: 2s-6s (4s)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Input video frames"}),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Frames per second for time calculations"
                }),
                "cut_start_time": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Start time of segment to remove (seconds)"
                }),
                "cut_end_time": ("FLOAT", {
                    "default": 6.0,
                    "min": 0.1,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "End time of segment to remove (seconds)"
                }),
                "time_format": (["seconds", "timecode"], {
                    "default": "seconds",
                    "tooltip": "Time format: seconds (5.5) or timecode (00:00:05:12)"
                })
            },
            "optional": {
                "cut_start_timecode": ("STRING", {
                    "default": "00:00:02:00",
                    "tooltip": "Start timecode to remove (HH:MM:SS:FF)"
                }),
                "cut_end_timecode": ("STRING", {
                    "default": "00:00:06:00",
                    "tooltip": "End timecode to remove (HH:MM:SS:FF)"
                }),
                "force_device": (["auto", "cpu", "cuda", "mps"], {
                    "default": "auto",
                    "tooltip": "Force specific device for processing"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("remaining_video", "removed_segment", "cut_info", "remaining_frames", "removed_frames", "remaining_duration", "removed_duration")
    FUNCTION = "cut_video"
    CATEGORY = "Raj Video Processing 🎬"
    
    def cut_video(self, frames, fps=24.0, cut_start_time=2.0, cut_end_time=6.0,
                  time_format="seconds", cut_start_timecode="00:00:02:00",
                  cut_end_timecode="00:00:06:00", force_device="auto"):
        
        # Determine device
        if force_device == "auto":
            device = get_optimal_device()
        else:
            device = torch.device(force_device)
        
        # Convert frames to tensor if needed
        if not isinstance(frames, torch.Tensor):
            frames = torch.tensor(frames, dtype=torch.float32)
        
        frames = frames.to(device)
        total_frames = frames.shape[0]
        total_duration = total_frames / fps
        
        logger.info(f"✂️ Cutting video: {total_frames} frames @ {fps}fps ({total_duration:.2f}s)")
        
        # Parse timing based on format
        if time_format == "timecode":
            actual_cut_start = self._parse_timecode(cut_start_timecode, fps)
            actual_cut_end = self._parse_timecode(cut_end_timecode, fps)
            logger.info(f"   Cut timecode: {cut_start_timecode} to {cut_end_timecode}")
        else:
            actual_cut_start = cut_start_time
            actual_cut_end = cut_end_time
            logger.info(f"   Cut range: {actual_cut_start}s to {actual_cut_end}s")
        
        # Validate timing
        if actual_cut_start >= actual_cut_end:
            raise ValueError(f"Cut start ({actual_cut_start}s) must be less than cut end ({actual_cut_end}s)")
        
        if actual_cut_end > total_duration:
            logger.warning(f"Cut end time ({actual_cut_end}s) exceeds video duration ({total_duration:.2f}s), adjusting")
            actual_cut_end = total_duration
        
        if actual_cut_start >= total_duration:
            raise ValueError(f"Cut start time ({actual_cut_start}s) is beyond video duration ({total_duration:.2f}s)")
        
        # Convert times to frame indices
        cut_start_frame = max(0, time_to_frame(actual_cut_start, fps))
        cut_end_frame = min(total_frames, time_to_frame(actual_cut_end, fps))
        
        # Ensure we have valid cut range
        if cut_end_frame <= cut_start_frame:
            cut_end_frame = cut_start_frame + 1
        
        logger.info(f"   Cut frames: {cut_start_frame} to {cut_end_frame}")
        
        # Extract segments
        # Removed segment: the part being cut out
        removed_segment = frames[cut_start_frame:cut_end_frame]
        
        # Remaining video: everything except the cut part (concatenated)
        if cut_start_frame > 0 and cut_end_frame < total_frames:
            # Cut from middle: concatenate before + after
            before_cut = frames[:cut_start_frame]
            after_cut = frames[cut_end_frame:]
            remaining_video = torch.cat([before_cut, after_cut], dim=0)
            logger.info(f"   Middle cut: {before_cut.shape[0]} + {after_cut.shape[0]} = {remaining_video.shape[0]} remaining frames")
        elif cut_start_frame == 0:
            # Cut from beginning: keep only after
            remaining_video = frames[cut_end_frame:]
            logger.info(f"   Beginning cut: keeping {remaining_video.shape[0]} frames after cut")
        elif cut_end_frame >= total_frames:
            # Cut from end: keep only before
            remaining_video = frames[:cut_start_frame]
            logger.info(f"   End cut: keeping {remaining_video.shape[0]} frames before cut")
        else:
            # This shouldn't happen, but fallback
            remaining_video = frames
            logger.warning("   No cut applied - invalid range")
        
        # Calculate metrics
        remaining_count = remaining_video.shape[0]
        removed_count = removed_segment.shape[0]
        remaining_duration = remaining_count / fps
        removed_duration = removed_count / fps
        
        # Calculate actual times
        actual_cut_start_time = frame_to_time(cut_start_frame, fps)
        actual_cut_end_time = frame_to_time(cut_end_frame, fps)
        
        logger.info(f"   Result: {remaining_count} remaining + {removed_count} removed frames")
        logger.info(f"   Durations: {remaining_duration:.2f}s remaining + {removed_duration:.2f}s removed")
        
        # Generate info string
        cut_info = (f"Cut {actual_cut_start_time:.2f}s-{actual_cut_end_time:.2f}s | "
                   f"Removed: {removed_count} frames ({removed_duration:.2f}s) | "
                   f"Remaining: {remaining_count} frames ({remaining_duration:.2f}s) | "
                   f"Original: {total_frames} frames ({total_duration:.2f}s)")
        
        # Generate preview files
        try:
            self._create_cut_previews(remaining_video, removed_segment, fps)
        except Exception as e:
            logger.warning(f"Failed to create cut previews: {e}")
        
        logger.info("✅ Video cut complete")
        
        return (remaining_video, removed_segment, cut_info, remaining_count, 
                removed_count, remaining_duration, removed_duration)
    
    def _parse_timecode(self, timecode: str, fps: float) -> float:
        """Parse timecode string (HH:MM:SS:FF) to seconds"""
        try:
            parts = timecode.split(":")
            if len(parts) != 4:
                raise ValueError(f"Invalid timecode format: {timecode}. Expected HH:MM:SS:FF")
            
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            frames = int(parts[3])
            
            total_seconds = (hours * 3600) + (minutes * 60) + seconds + (frames / fps)
            return total_seconds
            
        except Exception as e:
            logger.error(f"Failed to parse timecode '{timecode}': {e}")
            raise ValueError(f"Invalid timecode format: {timecode}")
    
    def _create_cut_previews(self, remaining_video, removed_segment, fps):
        """Create preview files for both outputs"""
        import cv2
        
        for video_data, name in [(remaining_video, "remaining"), (removed_segment, "removed")]:
            if video_data.shape[0] == 0:
                continue
                
            with tempfile.NamedTemporaryFile(suffix=f"_{name}.mp4", delete=False) as tmp_file:
                preview_path = tmp_file.name
            
            frames_cpu = video_data.cpu().numpy()
            frames_uint8 = (frames_cpu * 255).astype(np.uint8)
            
            if len(frames_uint8) > 0:
                height, width = frames_uint8.shape[1:3]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(preview_path, fourcc, fps, (width, height))
                
                for frame in frames_uint8:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                
                out.release()
                logger.info(f"📸 {name.title()} preview saved: {preview_path}")


class RajVideoTimecodeConverter:
    """
    Utility node to convert between seconds and timecode
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Frames per second"
                }),
                "conversion_mode": (["seconds_to_timecode", "timecode_to_seconds"], {
                    "default": "seconds_to_timecode",
                    "tooltip": "Conversion direction"
                })
            },
            "optional": {
                "input_seconds": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 86400.0,  # 24 hours
                    "step": 0.01,
                    "tooltip": "Input time in seconds"
                }),
                "input_timecode": ("STRING", {
                    "default": "00:00:00:00",
                    "tooltip": "Input timecode (HH:MM:SS:FF)"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("timecode", "seconds", "conversion_info")
    FUNCTION = "convert_time"
    CATEGORY = "Raj Video Processing 🎬"
    
    def convert_time(self, fps=24.0, conversion_mode="seconds_to_timecode", 
                    input_seconds=0.0, input_timecode="00:00:00:00"):
        
        if conversion_mode == "seconds_to_timecode":
            # Convert seconds to timecode
            hours = int(input_seconds // 3600)
            minutes = int((input_seconds % 3600) // 60)
            seconds = int(input_seconds % 60)
            frames = int((input_seconds % 1) * fps)
            
            timecode = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"
            conversion_info = f"Converted {input_seconds:.2f}s to {timecode} @ {fps}fps"
            
            return (timecode, input_seconds, conversion_info)
        
        else:
            # Convert timecode to seconds
            try:
                parts = input_timecode.split(":")
                if len(parts) != 4:
                    raise ValueError(f"Invalid timecode format: {input_timecode}")
                
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = int(parts[2])
                frames = int(parts[3])
                
                total_seconds = (hours * 3600) + (minutes * 60) + seconds + (frames / fps)
                conversion_info = f"Converted {input_timecode} to {total_seconds:.2f}s @ {fps}fps"
                
                return (input_timecode, total_seconds, conversion_info)
                
            except Exception as e:
                error_msg = f"Error converting timecode: {e}"
                logger.error(error_msg)
                return (input_timecode, 0.0, error_msg)