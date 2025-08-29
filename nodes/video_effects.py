"""
RajVideoEffects - Time-based video effects with easing functions
Apply brightness, contrast, sharpness, blur, and saturation effects over specific time ranges
"""

import torch
import numpy as np
import cv2
import os
from typing import Tuple, Dict, Any, List
from .utils import (
    get_optimal_device, logger, time_to_frame, frame_to_time, 
    parse_time_points, apply_easing
)

class RajVideoEffects:
    """
    Apply time-based effects to video frames with easing functions
    Supports brightness, contrast, sharpness, blur, and saturation adjustments
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
                
                # Brightness effect
                "brightness_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable brightness adjustment"
                }),
                "brightness_start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Start time for brightness effect (seconds)"
                }),
                "brightness_end_time": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "End time for brightness effect (seconds)"
                }),
                "brightness_start_value": ("FLOAT", {
                    "default": 0.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Starting brightness value (-100 to 100)"
                }),
                "brightness_end_value": ("FLOAT", {
                    "default": 20.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Ending brightness value (-100 to 100)"
                }),
                "brightness_easing": (["linear", "ease_in", "ease_out", "ease_in_out", "ease_out_in", "constant"], {
                    "default": "linear",
                    "tooltip": "Easing function for brightness transition"
                }),
                
                # Contrast effect
                "contrast_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable contrast adjustment"
                }),
                "contrast_start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Start time for contrast effect (seconds)"
                }),
                "contrast_end_time": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "End time for contrast effect (seconds)"
                }),
                "contrast_start_value": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Starting contrast value (0.1 to 5.0, 1.0 = normal)"
                }),
                "contrast_end_value": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Ending contrast value (0.1 to 5.0, 1.0 = normal)"
                }),
                "contrast_easing": (["linear", "ease_in", "ease_out", "ease_in_out", "ease_out_in", "constant"], {
                    "default": "linear",
                    "tooltip": "Easing function for contrast transition"
                }),
                
                # Blur effect
                "blur_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable blur effect"
                }),
                "blur_start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Start time for blur effect (seconds)"
                }),
                "blur_end_time": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "End time for blur effect (seconds)"
                }),
                "blur_start_value": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": "Starting blur radius (0 = no blur)"
                }),
                "blur_end_value": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": "Ending blur radius (0 = no blur)"
                }),
                "blur_easing": (["linear", "ease_in", "ease_out", "ease_in_out", "ease_out_in", "constant"], {
                    "default": "linear",
                    "tooltip": "Easing function for blur transition"
                }),
                
                # Saturation effect
                "saturation_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable saturation adjustment"
                }),
                "saturation_start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Start time for saturation effect (seconds)"
                }),
                "saturation_end_time": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "End time for saturation effect (seconds)"
                }),
                "saturation_start_value": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Starting saturation value (0=grayscale, 1=normal, >1=oversaturated)"
                }),
                "saturation_end_value": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Ending saturation value (0=grayscale, 1=normal, >1=oversaturated)"
                }),
                "saturation_easing": (["linear", "ease_in", "ease_out", "ease_in_out", "ease_out_in", "constant"], {
                    "default": "linear",
                    "tooltip": "Easing function for saturation transition"
                }),
            },
            "optional": {
                "force_device": (["auto", "cpu", "cuda", "mps"], {
                    "default": "auto",
                    "tooltip": "Force specific device for processing"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("processed_frames", "effect_info", "total_frames", "duration")
    FUNCTION = "apply_effects"
    CATEGORY = "Raj Video Processing 🎬"
    
    def apply_effects(self, frames, fps=24.0, 
                     brightness_enabled=False, brightness_start_time=0.0, brightness_end_time=2.0,
                     brightness_start_value=0.0, brightness_end_value=20.0, brightness_easing="linear",
                     contrast_enabled=False, contrast_start_time=0.0, contrast_end_time=2.0,
                     contrast_start_value=1.0, contrast_end_value=1.5, contrast_easing="linear",
                     blur_enabled=False, blur_start_time=0.0, blur_end_time=2.0,
                     blur_start_value=0.0, blur_end_value=5.0, blur_easing="linear",
                     saturation_enabled=False, saturation_start_time=0.0, saturation_end_time=2.0,
                     saturation_start_value=1.0, saturation_end_value=1.5, saturation_easing="linear",
                     force_device="auto"):
        
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
        
        logger.info(f"🎨 Applying video effects: {total_frames} frames @ {fps}fps ({duration:.2f}s)")
        
        # Collect enabled effects
        effects = []
        if brightness_enabled:
            effects.append(f"Brightness ({brightness_start_value}→{brightness_end_value}, {brightness_start_time}s-{brightness_end_time}s, {brightness_easing})")
        if contrast_enabled:
            effects.append(f"Contrast ({contrast_start_value}→{contrast_end_value}, {contrast_start_time}s-{contrast_end_time}s, {contrast_easing})")
        if blur_enabled:
            effects.append(f"Blur ({blur_start_value}→{blur_end_value}, {blur_start_time}s-{blur_end_time}s, {blur_easing})")
        if saturation_enabled:
            effects.append(f"Saturation ({saturation_start_value}→{saturation_end_value}, {saturation_start_time}s-{saturation_end_time}s, {saturation_easing})")
        
        if not effects:
            logger.info("   No effects enabled, returning original frames")
            effect_info = "No effects applied"
            
            # Still need to return preview data
            ui_preview = {
                "video_preview": [{
                    "path": "/tmp/no_effects_preview.mp4",  # Placeholder
                    "format": "mp4",
                    "fps": fps,
                    "duration": duration,
                    "width": frames.shape[2],
                    "height": frames.shape[1],
                    "frame_count": total_frames
                }]
            }
            
            return {
                "ui": ui_preview,
                "result": (frames, effect_info, total_frames, duration)
            }
        
        logger.info(f"   Effects: {', '.join(effects)}")
        
        # Process frames
        processed_frames = frames.clone()
        
        for frame_idx in range(total_frames):
            current_time = frame_to_time(frame_idx, fps)
            frame = processed_frames[frame_idx].cpu().numpy()
            
            # Apply brightness effect
            if brightness_enabled and brightness_start_time <= current_time <= brightness_end_time:
                t = (current_time - brightness_start_time) / (brightness_end_time - brightness_start_time)
                brightness_value = apply_easing(0, brightness_start_value, brightness_end_value, t, brightness_easing)
                frame = self._apply_brightness(frame, brightness_value)
            
            # Apply contrast effect
            if contrast_enabled and contrast_start_time <= current_time <= contrast_end_time:
                t = (current_time - contrast_start_time) / (contrast_end_time - contrast_start_time)
                contrast_value = apply_easing(0, contrast_start_value, contrast_end_value, t, contrast_easing)
                frame = self._apply_contrast(frame, contrast_value)
            
            # Apply blur effect
            if blur_enabled and blur_start_time <= current_time <= blur_end_time:
                t = (current_time - blur_start_time) / (blur_end_time - blur_start_time)
                blur_value = apply_easing(0, blur_start_value, blur_end_value, t, blur_easing)
                frame = self._apply_blur(frame, blur_value)
            
            # Apply saturation effect
            if saturation_enabled and saturation_start_time <= current_time <= saturation_end_time:
                t = (current_time - saturation_start_time) / (saturation_end_time - saturation_start_time)
                saturation_value = apply_easing(0, saturation_start_value, saturation_end_value, t, saturation_easing)
                frame = self._apply_saturation(frame, saturation_value)
            
            # Update processed frame
            processed_frames[frame_idx] = torch.tensor(frame, dtype=torch.float32, device=device)
        
        effect_info = f"Effects Applied: {', '.join(effects)} | Duration: {duration:.2f}s @ {fps}fps"
        logger.info(f"✅ Video effects complete: {total_frames} frames processed")
        
        # Generate preview file for UI
        preview_data = None
        try:
            import tempfile
            
            # Create temporary preview file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                preview_path = tmp_file.name
            
            # Save quick preview using OpenCV (first 100 frames max for speed)
            preview_frames = min(100, total_frames)
            frames_cpu = processed_frames[:preview_frames].cpu().numpy()
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
            logger.info(f"📸 Effects preview saved: {preview_path}")
        except Exception as e:
            logger.warning(f"Failed to create effects preview: {e}")
        
        if preview_data:
            return {
                "ui": preview_data,
                "result": (processed_frames, effect_info, total_frames, duration)
            }
        else:
            return (processed_frames, effect_info, total_frames, duration)
    
    def _apply_brightness(self, frame: np.ndarray, brightness: float) -> np.ndarray:
        """Apply brightness adjustment to frame"""
        # Brightness adjustment: add/subtract value
        adjusted = frame.copy()
        if brightness != 0:
            brightness_factor = brightness / 100.0  # Convert to -1.0 to 1.0 range
            adjusted = np.clip(adjusted + brightness_factor, 0.0, 1.0)
        return adjusted
    
    def _apply_contrast(self, frame: np.ndarray, contrast: float) -> np.ndarray:
        """Apply contrast adjustment to frame"""
        # Contrast adjustment: multiply by factor
        if contrast != 1.0:
            # Center around 0.5 for better contrast adjustment
            adjusted = (frame - 0.5) * contrast + 0.5
            adjusted = np.clip(adjusted, 0.0, 1.0)
            return adjusted
        return frame
    
    def _apply_blur(self, frame: np.ndarray, blur_radius: float) -> np.ndarray:
        """Apply blur effect to frame"""
        if blur_radius <= 0:
            return frame
        
        # Convert to uint8 for OpenCV processing
        frame_uint8 = (frame * 255).astype(np.uint8)
        
        # Apply Gaussian blur
        kernel_size = max(1, int(blur_radius * 2) + 1)  # Ensure odd kernel size
        if kernel_size > 1:
            blurred = cv2.GaussianBlur(frame_uint8, (kernel_size, kernel_size), blur_radius)
        else:
            blurred = frame_uint8
        
        # Convert back to float
        return blurred.astype(np.float32) / 255.0
    
    def _apply_saturation(self, frame: np.ndarray, saturation: float) -> np.ndarray:
        """Apply saturation adjustment to frame"""
        if saturation == 1.0:
            return frame
        
        # Convert RGB to HSV for saturation adjustment
        frame_uint8 = (frame * 255).astype(np.uint8)
        hsv = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Adjust saturation channel (S)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        
        # Convert back to RGB
        hsv = hsv.astype(np.uint8)
        result_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Convert back to float
        return result_rgb.astype(np.float32) / 255.0


class RajVideoSharpness:
    """
    Dedicated sharpness adjustment node
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
                    "tooltip": "Frames per second"
                }),
                "sharpness_start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Start time for sharpness effect (seconds)"
                }),
                "sharpness_end_time": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "End time for sharpness effect (seconds)"
                }),
                "sharpness_start_value": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Starting sharpness value (1.0 = normal)"
                }),
                "sharpness_end_value": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Ending sharpness value (1.0 = normal)"
                }),
                "sharpness_easing": (["linear", "ease_in", "ease_out", "ease_in_out", "ease_out_in", "constant"], {
                    "default": "linear",
                    "tooltip": "Easing function for sharpness transition"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("processed_frames", "effect_info", "total_frames", "duration")
    FUNCTION = "apply_sharpness"
    CATEGORY = "Raj Video Processing 🎬"
    
    def apply_sharpness(self, frames, fps=24.0, 
                       sharpness_start_time=0.0, sharpness_end_time=2.0,
                       sharpness_start_value=1.0, sharpness_end_value=2.0, 
                       sharpness_easing="linear"):
        
        device = get_optimal_device()
        
        if not isinstance(frames, torch.Tensor):
            frames = torch.tensor(frames, dtype=torch.float32)
        
        frames = frames.to(device)
        total_frames = frames.shape[0]
        duration = total_frames / fps
        
        logger.info(f"✨ Applying sharpness effect: {total_frames} frames @ {fps}fps")
        logger.info(f"   Sharpness: {sharpness_start_value}→{sharpness_end_value} from {sharpness_start_time}s to {sharpness_end_time}s ({sharpness_easing})")
        
        processed_frames = frames.clone()
        
        for frame_idx in range(total_frames):
            current_time = frame_to_time(frame_idx, fps)
            
            if sharpness_start_time <= current_time <= sharpness_end_time:
                t = (current_time - sharpness_start_time) / (sharpness_end_time - sharpness_start_time)
                sharpness_value = apply_easing(0, sharpness_start_value, sharpness_end_value, t, sharpness_easing)
                
                if sharpness_value != 1.0:  # Only process if sharpening needed
                    frame = processed_frames[frame_idx].cpu().numpy()
                    sharpened_frame = self._apply_sharpness(frame, sharpness_value)
                    processed_frames[frame_idx] = torch.tensor(sharpened_frame, dtype=torch.float32, device=device)
        
        effect_info = f"Sharpness: {sharpness_start_value}→{sharpness_end_value} ({sharpness_start_time}s-{sharpness_end_time}s, {sharpness_easing}) | Duration: {duration:.2f}s"
        logger.info(f"✅ Sharpness effect complete")
        
        return (processed_frames, effect_info, total_frames, duration)
    
    def _apply_sharpness(self, frame: np.ndarray, sharpness: float) -> np.ndarray:
        """Apply sharpness adjustment using unsharp mask"""
        if sharpness <= 0:
            return frame
        
        # Convert to uint8 for OpenCV processing
        frame_uint8 = (frame * 255).astype(np.uint8)
        
        # Create unsharp mask
        blurred = cv2.GaussianBlur(frame_uint8, (0, 0), 1.0)
        sharpened = cv2.addWeighted(frame_uint8, 1.0 + sharpness, blurred, -sharpness, 0)
        
        # Convert back to float
        return np.clip(sharpened.astype(np.float32) / 255.0, 0.0, 1.0)