"""
RajVideoTransitions - Advanced transition effects with cut points and timing
Apply fade, zoom, slide, wipe, and dissolve transitions at specific time points
"""

import torch
import numpy as np
import cv2
import tempfile
import os
from typing import Tuple, List, Dict, Any
from .utils import (
    get_optimal_device, logger, time_to_frame, frame_to_time, 
    parse_time_points, apply_easing
)

class RajVideoTransitions:
    """
    Apply transitions at specific cut points in video
    Supports fade, zoom, slide, wipe, and dissolve transitions with easing
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
                    "tooltip": "Frames per second (for time calculations)"
                }),
                "cut_points": ("STRING", {
                    "default": "2.5, 5.0",
                    "multiline": False,
                    "tooltip": "Comma-separated cut times in seconds (e.g., '2.5, 5.0, 8.5')"
                }),
                "transition_type": (["fade", "zoom", "slide", "wipe", "dissolve"], {
                    "default": "fade",
                    "tooltip": "Type of transition to apply"
                }),
                "transition_duration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Duration of each transition in seconds"
                }),
                "transition_easing": (["linear", "ease_in", "ease_out", "ease_in_out", "ease_out_in"], {
                    "default": "ease_in_out",
                    "tooltip": "Easing function for transitions"
                }),
            },
            "optional": {
                "slide_direction": (["up", "down", "left", "right"], {
                    "default": "right",
                    "tooltip": "Direction for slide transitions"
                }),
                "zoom_center_x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Zoom center X coordinate (0.0-1.0)"
                }),
                "zoom_center_y": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Zoom center Y coordinate (0.0-1.0)"
                }),
                "zoom_scale": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Maximum zoom scale"
                }),
                "fade_color": (["black", "white"], {
                    "default": "black",
                    "tooltip": "Fade color"
                }),
                "force_device": (["auto", "cpu", "cuda", "mps"], {
                    "default": "auto",
                    "tooltip": "Force specific device for processing"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("processed_frames", "transition_info", "total_frames", "duration")
    FUNCTION = "apply_transitions"
    CATEGORY = "Raj Video Processing 🎬"
    
    def apply_transitions(self, frames, fps=24.0, cut_points="2.5, 5.0", 
                         transition_type="fade", transition_duration=0.5, transition_easing="ease_in_out",
                         slide_direction="right", zoom_center_x=0.5, zoom_center_y=0.5, zoom_scale=1.5,
                         fade_color="black", force_device="auto"):
        
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
        duration = total_frames / fps
        
        logger.info(f"🎭 Applying video transitions: {total_frames} frames @ {fps}fps ({duration:.2f}s)")
        logger.info(f"   Cut points: {cut_points}")
        logger.info(f"   Transition: {transition_type} ({transition_duration}s, {transition_easing})")
        
        # Parse cut points
        try:
            cut_frame_indices = parse_time_points(cut_points, fps)
        except Exception as e:
            logger.error(f"Failed to parse cut points '{cut_points}': {e}")
            return (frames, f"Error: Invalid cut points - {e}", total_frames, duration)
        
        if not cut_frame_indices:
            logger.info("   No cut points specified, returning original frames")
            return (frames, "No transitions applied", total_frames, duration)
        
        # Filter cut points to be within video duration
        valid_cuts = [cut for cut in cut_frame_indices if 0 <= cut < total_frames]
        if len(valid_cuts) != len(cut_frame_indices):
            logger.warning(f"Some cut points are outside video duration, using valid ones: {valid_cuts}")
        cut_frame_indices = valid_cuts
        
        if not cut_frame_indices:
            logger.warning("No valid cut points within video duration")
            return (frames, "No valid cut points", total_frames, duration)
        
        logger.info(f"   Processing {len(cut_frame_indices)} transitions at frames: {cut_frame_indices}")
        
        # Calculate transition frame duration
        transition_frames = max(1, int(transition_duration * fps))
        
        # Process transitions
        processed_frames = frames.clone()
        
        for cut_frame in cut_frame_indices:
            logger.info(f"   Applying {transition_type} transition at frame {cut_frame} ({frame_to_time(cut_frame, fps):.2f}s)")
            
            # Apply transition based on type
            if transition_type == "fade":
                processed_frames = self._apply_fade_transition(
                    processed_frames, cut_frame, transition_frames, transition_easing, fade_color, device
                )
            elif transition_type == "zoom":
                processed_frames = self._apply_zoom_transition(
                    processed_frames, cut_frame, transition_frames, transition_easing, 
                    zoom_center_x, zoom_center_y, zoom_scale, device
                )
            elif transition_type == "slide":
                processed_frames = self._apply_slide_transition(
                    processed_frames, cut_frame, transition_frames, transition_easing, 
                    slide_direction, device
                )
            elif transition_type == "wipe":
                processed_frames = self._apply_wipe_transition(
                    processed_frames, cut_frame, transition_frames, transition_easing, 
                    slide_direction, device
                )
            elif transition_type == "dissolve":
                processed_frames = self._apply_dissolve_transition(
                    processed_frames, cut_frame, transition_frames, transition_easing, device
                )
        
        transition_info = f"Transitions: {len(cut_frame_indices)} {transition_type} transitions | Duration: {duration:.2f}s @ {fps}fps"
        logger.info(f"✅ Video transitions complete")
        
        # Generate preview file for UI
        preview_data = None
        try:
            # Create temporary preview file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                preview_path = tmp_file.name
            
            # Save preview using OpenCV (limit to first 200 frames for speed)
            preview_frames_count = min(200, total_frames)
            frames_cpu = processed_frames[:preview_frames_count].cpu().numpy()
            frames_uint8 = (frames_cpu * 255).astype(np.uint8)
            
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
            logger.info(f"📸 Transitions preview saved: {preview_path}")
        except Exception as e:
            logger.warning(f"Failed to create transitions preview: {e}")
        
        if preview_data:
            return {
                "ui": preview_data,
                "result": (processed_frames, transition_info, total_frames, duration)
            }
        else:
            return (processed_frames, transition_info, total_frames, duration)
    
    def _apply_fade_transition(self, frames: torch.Tensor, cut_frame: int, transition_frames: int, 
                              easing: str, fade_color: str, device: torch.device) -> torch.Tensor:
        """Apply fade transition (fade to color and back)"""
        fade_value = 1.0 if fade_color == "white" else 0.0
        total_frames = frames.shape[0]
        
        # Fade out (before cut) and fade in (after cut)
        fade_out_start = max(0, cut_frame - transition_frames // 2)
        fade_in_end = min(total_frames, cut_frame + transition_frames // 2)
        
        for i in range(fade_out_start, min(cut_frame, total_frames)):
            # Fade out to color
            t = (i - fade_out_start) / max(1, (cut_frame - fade_out_start))
            alpha = apply_easing(0, 0.0, 1.0, t, easing)
            frames[i] = frames[i] * (1 - alpha) + fade_value * alpha
        
        for i in range(cut_frame, fade_in_end):
            # Fade in from color
            t = (i - cut_frame) / max(1, (fade_in_end - cut_frame))
            alpha = apply_easing(0, 1.0, 0.0, t, easing)
            frames[i] = frames[i] * (1 - alpha) + fade_value * alpha
        
        return frames
    
    def _apply_zoom_transition(self, frames: torch.Tensor, cut_frame: int, transition_frames: int,
                              easing: str, center_x: float, center_y: float, scale: float, device: torch.device) -> torch.Tensor:
        """Apply zoom transition"""
        total_frames = frames.shape[0]
        
        zoom_start = max(0, cut_frame - transition_frames // 2)
        zoom_end = min(total_frames, cut_frame + transition_frames // 2)
        
        for i in range(zoom_start, zoom_end):
            # Calculate zoom factor
            t = (i - zoom_start) / max(1, (zoom_end - zoom_start))
            if i < cut_frame:
                # Zoom in
                zoom_factor = apply_easing(0, 1.0, scale, t * 2, easing)
            else:
                # Zoom out
                zoom_factor = apply_easing(0, scale, 1.0, (t - 0.5) * 2, easing)
            
            if zoom_factor != 1.0:
                frame = frames[i].cpu().numpy()
                zoomed_frame = self._zoom_frame(frame, zoom_factor, center_x, center_y)
                frames[i] = torch.tensor(zoomed_frame, dtype=torch.float32, device=device)
        
        return frames
    
    def _apply_slide_transition(self, frames: torch.Tensor, cut_frame: int, transition_frames: int,
                               easing: str, direction: str, device: torch.device) -> torch.Tensor:
        """Apply slide transition"""
        total_frames = frames.shape[0]
        
        slide_start = max(0, cut_frame - transition_frames // 2)
        slide_end = min(total_frames, cut_frame + transition_frames // 2)
        
        for i in range(slide_start, slide_end):
            t = (i - slide_start) / max(1, (slide_end - slide_start))
            slide_progress = apply_easing(0, 0.0, 1.0, t, easing)
            
            frame = frames[i].cpu().numpy()
            slid_frame = self._slide_frame(frame, slide_progress, direction)
            frames[i] = torch.tensor(slid_frame, dtype=torch.float32, device=device)
        
        return frames
    
    def _apply_wipe_transition(self, frames: torch.Tensor, cut_frame: int, transition_frames: int,
                              easing: str, direction: str, device: torch.device) -> torch.Tensor:
        """Apply wipe transition (reveal effect)"""
        total_frames = frames.shape[0]
        
        wipe_start = max(0, cut_frame - transition_frames // 2)
        wipe_end = min(total_frames, cut_frame + transition_frames // 2)
        
        for i in range(wipe_start, wipe_end):
            t = (i - wipe_start) / max(1, (wipe_end - wipe_start))
            wipe_progress = apply_easing(0, 0.0, 1.0, t, easing)
            
            frame = frames[i].cpu().numpy()
            wiped_frame = self._wipe_frame(frame, wipe_progress, direction)
            frames[i] = torch.tensor(wiped_frame, dtype=torch.float32, device=device)
        
        return frames
    
    def _apply_dissolve_transition(self, frames: torch.Tensor, cut_frame: int, transition_frames: int,
                                  easing: str, device: torch.device) -> torch.Tensor:
        """Apply dissolve transition (noise-based reveal)"""
        total_frames = frames.shape[0]
        
        dissolve_start = max(0, cut_frame - transition_frames // 2)
        dissolve_end = min(total_frames, cut_frame + transition_frames // 2)
        
        # Generate noise pattern
        if total_frames > 0:
            height, width = frames.shape[1:3]
            noise = np.random.random((height, width))
        
        for i in range(dissolve_start, dissolve_end):
            t = (i - dissolve_start) / max(1, (dissolve_end - dissolve_start))
            dissolve_progress = apply_easing(0, 0.0, 1.0, t, easing)
            
            frame = frames[i].cpu().numpy()
            dissolved_frame = self._dissolve_frame(frame, dissolve_progress, noise)
            frames[i] = torch.tensor(dissolved_frame, dtype=torch.float32, device=device)
        
        return frames
    
    # Helper methods for frame transformations
    
    def _zoom_frame(self, frame: np.ndarray, zoom_factor: float, center_x: float, center_y: float) -> np.ndarray:
        """Zoom frame around specified center point"""
        height, width = frame.shape[:2]
        
        # Calculate crop region
        crop_width = int(width / zoom_factor)
        crop_height = int(height / zoom_factor)
        
        center_px_x = int(center_x * width)
        center_px_y = int(center_y * height)
        
        x1 = max(0, center_px_x - crop_width // 2)
        y1 = max(0, center_px_y - crop_height // 2)
        x2 = min(width, x1 + crop_width)
        y2 = min(height, y1 + crop_height)
        
        # Crop and resize
        cropped = frame[y1:y2, x1:x2]
        zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        return zoomed
    
    def _slide_frame(self, frame: np.ndarray, progress: float, direction: str) -> np.ndarray:
        """Slide frame in specified direction"""
        height, width = frame.shape[:2]
        
        if direction == "right":
            shift_x = int(width * progress)
            shift_y = 0
        elif direction == "left":
            shift_x = -int(width * progress)
            shift_y = 0
        elif direction == "down":
            shift_x = 0
            shift_y = int(height * progress)
        elif direction == "up":
            shift_x = 0
            shift_y = -int(height * progress)
        else:
            return frame
        
        # Create transformation matrix
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        return shifted
    
    def _wipe_frame(self, frame: np.ndarray, progress: float, direction: str) -> np.ndarray:
        """Apply wipe effect to frame"""
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.float32)
        
        if direction == "right":
            reveal_width = int(width * progress)
            mask[:, :reveal_width] = 1.0
        elif direction == "left":
            reveal_width = int(width * progress)
            mask[:, width-reveal_width:] = 1.0
        elif direction == "down":
            reveal_height = int(height * progress)
            mask[:reveal_height, :] = 1.0
        elif direction == "up":
            reveal_height = int(height * progress)
            mask[height-reveal_height:, :] = 1.0
        
        # Apply mask
        if len(frame.shape) == 3:
            mask = np.stack([mask] * frame.shape[2], axis=2)
        
        return frame * mask
    
    def _dissolve_frame(self, frame: np.ndarray, progress: float, noise: np.ndarray) -> np.ndarray:
        """Apply dissolve effect using noise pattern"""
        # Create mask based on noise and progress
        mask = (noise < progress).astype(np.float32)
        
        if len(frame.shape) == 3 and len(mask.shape) == 2:
            mask = np.stack([mask] * frame.shape[2], axis=2)
        
        return frame * mask


class RajTransitionLibrary:
    """
    Collection of advanced transition effects
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video1": ("IMAGE", {"tooltip": "First video segment"}),
                "video2": ("IMAGE", {"tooltip": "Second video segment"}),
                "transition_type": (["crossfade", "push", "spin", "blur_transition", "scale_fade"], {
                    "default": "crossfade",
                    "tooltip": "Advanced transition type"
                }),
                "transition_duration": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Transition duration in seconds"
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Frames per second"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("combined_video", "transition_info")
    FUNCTION = "create_transition"
    CATEGORY = "Raj Video Processing 🎬"
    
    def create_transition(self, video1, video2, transition_type="crossfade", transition_duration=1.0, fps=24.0):
        """Create advanced transition between two video segments"""
        
        device = get_optimal_device()
        
        if not isinstance(video1, torch.Tensor):
            video1 = torch.tensor(video1, dtype=torch.float32)
        if not isinstance(video2, torch.Tensor):
            video2 = torch.tensor(video2, dtype=torch.float32)
        
        video1 = video1.to(device)
        video2 = video2.to(device)
        
        transition_frames = int(transition_duration * fps)
        
        logger.info(f"🔄 Creating {transition_type} transition: {transition_frames} frames ({transition_duration}s)")
        
        # Generate transition frames
        if transition_type == "crossfade":
            transition_sequence = self._crossfade_transition(video1, video2, transition_frames)
        elif transition_type == "push":
            transition_sequence = self._push_transition(video1, video2, transition_frames)
        elif transition_type == "spin":
            transition_sequence = self._spin_transition(video1, video2, transition_frames)
        elif transition_type == "blur_transition":
            transition_sequence = self._blur_transition(video1, video2, transition_frames)
        elif transition_type == "scale_fade":
            transition_sequence = self._scale_fade_transition(video1, video2, transition_frames)
        else:
            transition_sequence = self._crossfade_transition(video1, video2, transition_frames)
        
        # Combine: video1 + transition + video2
        if len(video1) > transition_frames:
            pre_transition = video1[:-transition_frames]
        else:
            pre_transition = torch.empty(0, *video1.shape[1:], device=device)
        
        if len(video2) > transition_frames:
            post_transition = video2[transition_frames:]
        else:
            post_transition = torch.empty(0, *video2.shape[1:], device=device)
        
        combined = torch.cat([pre_transition, transition_sequence, post_transition], dim=0)
        
        transition_info = f"Transition: {transition_type} | Duration: {transition_duration}s | Total frames: {len(combined)}"
        logger.info(f"✅ Transition complete: {len(combined)} total frames")
        
        return (combined, transition_info)
    
    def _crossfade_transition(self, video1: torch.Tensor, video2: torch.Tensor, frames: int) -> torch.Tensor:
        """Crossfade between two videos"""
        end_frames = video1[-frames:] if len(video1) >= frames else video1
        start_frames = video2[:frames] if len(video2) >= frames else video2
        
        min_frames = min(len(end_frames), len(start_frames))
        transition = torch.zeros(min_frames, *video1.shape[1:], device=video1.device)
        
        for i in range(min_frames):
            alpha = i / max(1, min_frames - 1)
            transition[i] = end_frames[i] * (1 - alpha) + start_frames[i] * alpha
        
        return transition
    
    def _push_transition(self, video1: torch.Tensor, video2: torch.Tensor, frames: int) -> torch.Tensor:
        """Push transition (slide one video off screen)"""
        # Implementation would involve sliding frames
        return self._crossfade_transition(video1, video2, frames)  # Placeholder
    
    def _spin_transition(self, video1: torch.Tensor, video2: torch.Tensor, frames: int) -> torch.Tensor:
        """Spinning transition effect"""
        # Implementation would involve rotating frames
        return self._crossfade_transition(video1, video2, frames)  # Placeholder
    
    def _blur_transition(self, video1: torch.Tensor, video2: torch.Tensor, frames: int) -> torch.Tensor:
        """Blur and reveal transition"""
        # Implementation would blur video1 out and video2 in
        return self._crossfade_transition(video1, video2, frames)  # Placeholder
    
    def _scale_fade_transition(self, video1: torch.Tensor, video2: torch.Tensor, frames: int) -> torch.Tensor:
        """Scale and fade transition"""
        # Implementation would scale and fade
        return self._crossfade_transition(video1, video2, frames)  # Placeholder