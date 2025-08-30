import torch
from .utils import tensor_to_video_frames, time_to_frame, logger

class RajVideoSegmenter:
    """
    Segment video frames by time range and output both segmented and non-segmented portions
    Takes input frames from another node and creates two outputs based on timing
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Input video frames from another node"
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, 
                    "min": 1.0, 
                    "max": 120.0, 
                    "step": 0.1,
                    "tooltip": "Frame rate of the input video for time calculations"
                }),
                "start_time": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 3600.0, 
                    "step": 0.1,
                    "tooltip": "Segment start time in seconds"
                }),
                "end_time": ("FLOAT", {
                    "default": 5.0, 
                    "min": 0.1, 
                    "max": 3600.0, 
                    "step": 0.1,
                    "tooltip": "Segment end time in seconds"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("non_segmented_frames", "segmented_frames", "non_seg_info", "seg_info", "non_seg_count", "seg_count", "fps")
    FUNCTION = "segment_frames"
    CATEGORY = "Raj Video Processing 🎬"
    
    def segment_frames(self, frames, fps, start_time, end_time):
        
        # Validate inputs
        if start_time >= end_time:
            raise ValueError(f"Start time ({start_time}s) must be less than end time ({end_time}s)")
        
        if start_time < 0:
            raise ValueError(f"Start time cannot be negative ({start_time}s)")
        
        if fps <= 0:
            raise ValueError(f"FPS must be positive ({fps})")
        
        # Get input tensor info
        total_frames = frames.shape[0]
        height = frames.shape[1]
        width = frames.shape[2]
        channels = frames.shape[3] if len(frames.shape) > 3 else 3
        
        # Calculate video duration from frame count and FPS
        video_duration = total_frames / fps if fps > 0 else 0
        
        # Validate segment times against video duration
        if end_time > video_duration:
            logger.warning(f"⚠️ End time ({end_time}s) exceeds video duration ({video_duration:.2f}s), clamping to video end")
            end_time = video_duration
        
        if start_time >= video_duration:
            raise ValueError(f"Start time ({start_time}s) exceeds video duration ({video_duration:.2f}s)")
        
        logger.info(f"✂️ Segmenting frames: {total_frames} frames @ {fps:.2f}fps")
        logger.info(f"   Segment: {start_time:.2f}s - {end_time:.2f}s (duration: {end_time - start_time:.2f}s)")
        logger.info(f"   Video duration: {video_duration:.2f}s")
        
        # Convert times to frame indices
        start_frame = time_to_frame(start_time, fps)
        end_frame = time_to_frame(end_time, fps)
        
        # Clamp frame indices to valid range
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        
        logger.info(f"   Frame indices: {start_frame} - {end_frame} (of {total_frames} total)")
        
        # Split tensor into segments
        # Before segment: frames[0:start_frame]
        # Segment: frames[start_frame:end_frame] 
        # After segment: frames[end_frame:]
        
        segment_frames = frames[start_frame:end_frame]
        
        # Create non-segmented frames by concatenating before and after
        if start_frame > 0 and end_frame < total_frames:
            # Both before and after segments exist
            before_frames = frames[0:start_frame]
            after_frames = frames[end_frame:]
            non_segment_frames = torch.cat([before_frames, after_frames], dim=0)
            logger.info(f"   Non-segmented: {before_frames.shape[0]} before + {after_frames.shape[0]} after = {non_segment_frames.shape[0]} total")
        elif start_frame > 0:
            # Only before segment exists (segment goes to end)
            non_segment_frames = frames[0:start_frame]
            logger.info(f"   Non-segmented: {non_segment_frames.shape[0]} before segment only")
        elif end_frame < total_frames:
            # Only after segment exists (segment starts from beginning)
            non_segment_frames = frames[end_frame:]
            logger.info(f"   Non-segmented: {non_segment_frames.shape[0]} after segment only")
        else:
            # Segment covers entire video - create empty non-segment tensor
            non_segment_frames = torch.empty((0, height, width, channels), 
                                           device=frames.device, dtype=frames.dtype)
            logger.info(f"   Non-segmented: empty (segment covers entire video)")
        
        # Verify tensor integrity
        if segment_frames.shape[0] > 0:
            seg_max = torch.max(segment_frames).item()
            if seg_max < 0.001:
                logger.error(f"❌ Segmented frames corrupted (max: {seg_max:.6f})")
                raise ValueError("Segmented frames are corrupted")
        
        if non_segment_frames.shape[0] > 0:
            non_seg_max = torch.max(non_segment_frames).item()
            if non_seg_max < 0.001:
                logger.error(f"❌ Non-segmented frames corrupted (max: {non_seg_max:.6f})")
                raise ValueError("Non-segmented frames are corrupted")
        
        # Ensure tensors are in ComfyUI format (already should be from input)
        segment_frames_comfy = tensor_to_video_frames(segment_frames)
        non_segment_frames_comfy = tensor_to_video_frames(non_segment_frames) if non_segment_frames.shape[0] > 0 else torch.empty((0, height, width, channels), device=frames.device, dtype=frames.dtype)
        
        # Create info strings
        seg_info = f"Segment: {start_time:.2f}s-{end_time:.2f}s | " \
                  f"Frames: {segment_frames.shape[0]} | " \
                  f"FPS: {fps:.2f} | " \
                  f"Size: {width}x{height}"
        
        non_seg_info = f"Non-Segment: Excluded {start_time:.2f}s-{end_time:.2f}s | " \
                      f"Frames: {non_segment_frames.shape[0]} | " \
                      f"FPS: {fps:.2f} | " \
                      f"Size: {width}x{height}"
        
        logger.info(f"✅ Segmentation complete:")
        logger.info(f"   Segmented: {segment_frames.shape[0]} frames")
        logger.info(f"   Non-segmented: {non_segment_frames.shape[0]} frames")
        
        return (
            non_segment_frames_comfy, 
            segment_frames_comfy, 
            non_seg_info, 
            seg_info, 
            non_segment_frames.shape[0], 
            segment_frames.shape[0], 
            fps
        )