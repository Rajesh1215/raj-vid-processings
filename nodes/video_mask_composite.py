"""
RajVideoMaskComposite - Advanced color-based video masking and compositing
Composite multiple videos using color-based masks with advanced blending modes
"""

import torch
import numpy as np
import os
import tempfile
from typing import Tuple, List, Dict, Optional
from .utils import (
    get_optimal_device, logger, time_to_frame, frame_to_time,
    get_save_path_incremental
)

class RajVideoMaskComposite:
    """
    Composite multiple videos using color-based masking
    Supports chroma key, color range masking, and advanced blending
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_video": ("IMAGE", {"tooltip": "Base/background video frames"}),
                "overlay_video": ("IMAGE", {"tooltip": "Overlay/foreground video frames"}),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Frames per second"
                }),
                "mask_mode": (["chroma_key", "color_range", "brightness_mask", "custom_color"], {
                    "default": "chroma_key",
                    "tooltip": "Masking method to use"
                }),
                "mask_color_r": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Mask color Red component (0-1)"
                }),
                "mask_color_g": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Mask color Green component (0-1) - default green screen"
                }),
                "mask_color_b": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Mask color Blue component (0-1)"
                }),
                "tolerance": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.001,
                    "tooltip": "Color matching tolerance"
                }),
                "edge_softness": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.001,
                    "tooltip": "Edge softening for smooth blending"
                }),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "hard_light"], {
                    "default": "normal",
                    "tooltip": "Blending mode for compositing"
                })
            },
            "optional": {
                "mask_video": ("IMAGE", {"tooltip": "Optional mask video (white=show overlay, black=show base)"}),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Overall opacity of overlay"
                }),
                "spill_suppression": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Suppress color spill from key color"
                }),
                "brightness_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Brightness threshold for brightness_mask mode"
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the generated mask"
                }),
                "force_device": (["auto", "cpu", "cuda", "mps"], {
                    "default": "auto",
                    "tooltip": "Force specific device for processing"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("composite_video", "generated_mask", "composite_info", "frame_count", "duration")
    FUNCTION = "composite_video"
    CATEGORY = "Raj Video Processing 🎬"
    
    def composite_video(self, base_video, overlay_video, fps=24.0, mask_mode="chroma_key",
                       mask_color_r=0.0, mask_color_g=1.0, mask_color_b=0.0,
                       tolerance=0.1, edge_softness=0.02, blend_mode="normal",
                       mask_video=None, opacity=1.0, spill_suppression=0.0,
                       brightness_threshold=0.5, invert_mask=False, force_device="auto"):
        
        # Determine device
        if force_device == "auto":
            device = get_optimal_device()
        else:
            device = torch.device(force_device)
        
        # Convert inputs to tensors
        base_video = self._ensure_tensor(base_video, device)
        overlay_video = self._ensure_tensor(overlay_video, device)
        if mask_video is not None:
            mask_video = self._ensure_tensor(mask_video, device)
        
        # Get dimensions
        base_frames = base_video.shape[0]
        overlay_frames = overlay_video.shape[0]
        min_frames = min(base_frames, overlay_frames)
        
        # Trim videos to same length
        base_video = base_video[:min_frames]
        overlay_video = overlay_video[:min_frames]
        if mask_video is not None:
            mask_video = mask_video[:min_frames]
        
        duration = min_frames / fps
        
        logger.info(f"🎭 Compositing videos: {min_frames} frames @ {fps}fps ({duration:.2f}s)")
        logger.info(f"   Mask mode: {mask_mode}, Blend mode: {blend_mode}")
        logger.info(f"   Mask color: RGB({mask_color_r:.2f}, {mask_color_g:.2f}, {mask_color_b:.2f})")
        
        # Ensure videos have same spatial dimensions
        base_video, overlay_video = self._match_dimensions(base_video, overlay_video, device)
        
        # Generate mask based on mode
        if mask_video is not None:
            # Use provided mask video
            mask = self._process_mask_video(mask_video, base_video.shape)
            logger.info("   Using provided mask video")
        elif mask_mode == "chroma_key":
            mask = self._create_chroma_key_mask(overlay_video, mask_color_r, mask_color_g, mask_color_b, tolerance)
        elif mask_mode == "color_range":
            mask = self._create_color_range_mask(overlay_video, mask_color_r, mask_color_g, mask_color_b, tolerance)
        elif mask_mode == "brightness_mask":
            mask = self._create_brightness_mask(overlay_video, brightness_threshold)
        elif mask_mode == "custom_color":
            mask = self._create_custom_color_mask(overlay_video, mask_color_r, mask_color_g, mask_color_b, tolerance)
        else:
            mask = torch.ones_like(overlay_video[:, :, :, 0:1])
        
        # Apply edge softening
        if edge_softness > 0:
            mask = self._apply_edge_softening(mask, edge_softness)
        
        # Invert mask if requested
        if invert_mask:
            mask = 1.0 - mask
            logger.info("   Mask inverted")
        
        # Apply spill suppression
        if spill_suppression > 0:
            overlay_video = self._apply_spill_suppression(overlay_video, mask_color_r, mask_color_g, mask_color_b, spill_suppression)
        
        # Apply opacity
        if opacity < 1.0:
            mask = mask * opacity
        
        # Apply blending
        composite_video = self._apply_blend_mode(base_video, overlay_video, mask, blend_mode)
        
        # Generate info string
        composite_info = (f"Composite: {mask_mode} mask | "
                         f"Blend: {blend_mode} | "
                         f"Frames: {min_frames} ({duration:.2f}s) | "
                         f"Tolerance: {tolerance:.3f}, Softness: {edge_softness:.3f}")
        
        # Create preview
        try:
            self._create_composite_preview(composite_video, mask, fps)
        except Exception as e:
            logger.warning(f"Failed to create composite preview: {e}")
        
        logger.info("✅ Video composite complete")
        
        # Return mask as 3-channel for visualization
        mask_visualization = mask.repeat(1, 1, 1, 3)
        
        return (composite_video, mask_visualization, composite_info, min_frames, duration)
    
    def _ensure_tensor(self, data, device):
        """Ensure data is a tensor on the correct device"""
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        return data.to(device)
    
    def _match_dimensions(self, base_video, overlay_video, device):
        """Ensure both videos have the same spatial dimensions"""
        base_shape = base_video.shape
        overlay_shape = overlay_video.shape
        
        if base_shape[1:3] != overlay_shape[1:3]:
            # Resize overlay to match base
            import torch.nn.functional as F
            target_h, target_w = base_shape[1], base_shape[2]
            
            # Reshape for interpolation: (frames, channels, height, width)
            overlay_reshaped = overlay_video.permute(0, 3, 1, 2)
            overlay_resized = F.interpolate(overlay_reshaped, size=(target_h, target_w), mode='bilinear', align_corners=False)
            overlay_video = overlay_resized.permute(0, 2, 3, 1)
            
            logger.info(f"   Resized overlay from {overlay_shape[1:3]} to {(target_h, target_w)}")
        
        return base_video, overlay_video
    
    def _process_mask_video(self, mask_video, target_shape):
        """Process provided mask video"""
        # Convert to grayscale if needed
        if mask_video.shape[-1] == 3:
            mask = torch.mean(mask_video, dim=-1, keepdim=True)
        else:
            mask = mask_video[:, :, :, 0:1]
        
        # Ensure mask matches target dimensions
        if mask.shape[1:3] != target_shape[1:3]:
            import torch.nn.functional as F
            mask_reshaped = mask.permute(0, 3, 1, 2)
            mask_resized = F.interpolate(mask_reshaped, size=target_shape[1:3], mode='bilinear', align_corners=False)
            mask = mask_resized.permute(0, 2, 3, 1)
        
        return mask
    
    def _create_chroma_key_mask(self, video, r, g, b, tolerance):
        """Create chroma key mask (green screen effect)"""
        key_color = torch.tensor([r, g, b], device=video.device).view(1, 1, 1, 3)
        
        # Calculate color distance
        color_diff = torch.norm(video - key_color, dim=-1, keepdim=True)
        
        # Create mask (1 = keep overlay, 0 = show base)
        mask = (color_diff > tolerance).float()
        
        logger.info(f"   Chroma key mask: {torch.sum(mask == 0).item()} transparent pixels per frame avg")
        return mask
    
    def _create_color_range_mask(self, video, r, g, b, tolerance):
        """Create color range mask"""
        target_color = torch.tensor([r, g, b], device=video.device).view(1, 1, 1, 3)
        
        # Check if each channel is within tolerance
        color_match = torch.all(torch.abs(video - target_color) <= tolerance, dim=-1, keepdim=True)
        
        # Invert for overlay mask (1 = keep overlay, 0 = show base)
        mask = (~color_match).float()
        
        return mask
    
    def _create_brightness_mask(self, video, threshold):
        """Create brightness-based mask"""
        # Convert to grayscale for brightness calculation
        brightness = torch.mean(video, dim=-1, keepdim=True)
        
        # Create mask based on brightness threshold
        mask = (brightness > threshold).float()
        
        logger.info(f"   Brightness mask: threshold {threshold:.2f}")
        return mask
    
    def _create_custom_color_mask(self, video, r, g, b, tolerance):
        """Create custom color-based mask with advanced color space"""
        target_color = torch.tensor([r, g, b], device=video.device).view(1, 1, 1, 3)
        
        # Convert to HSV for better color matching
        video_hsv = self._rgb_to_hsv(video)
        target_hsv = self._rgb_to_hsv(target_color)
        
        # Calculate hue distance (circular)
        hue_diff = torch.abs(video_hsv[:, :, :, 0:1] - target_hsv[:, :, :, 0:1])
        hue_diff = torch.min(hue_diff, 1.0 - hue_diff)  # Handle circular nature of hue
        
        # Calculate saturation and value differences
        sat_diff = torch.abs(video_hsv[:, :, :, 1:2] - target_hsv[:, :, :, 1:2])
        val_diff = torch.abs(video_hsv[:, :, :, 2:3] - target_hsv[:, :, :, 2:3])
        
        # Combined distance in HSV space
        hsv_distance = torch.sqrt(hue_diff**2 + sat_diff**2 + val_diff**2)
        
        # Create mask
        mask = (hsv_distance > tolerance).float()
        
        return mask
    
    def _rgb_to_hsv(self, rgb):
        """Convert RGB to HSV color space"""
        r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
        
        max_val = torch.max(torch.max(r, g), b)
        min_val = torch.min(torch.min(r, g), b)
        delta = max_val - min_val
        
        # Value
        v = max_val
        
        # Saturation
        s = torch.where(max_val != 0, delta / max_val, torch.zeros_like(max_val))
        
        # Hue
        h = torch.zeros_like(max_val)
        
        # Red is max
        mask_r = (max_val == r) & (delta != 0)
        h = torch.where(mask_r, ((g - b) / delta) % 6, h)
        
        # Green is max
        mask_g = (max_val == g) & (delta != 0)
        h = torch.where(mask_g, (b - r) / delta + 2, h)
        
        # Blue is max
        mask_b = (max_val == b) & (delta != 0)
        h = torch.where(mask_b, (r - g) / delta + 4, h)
        
        h = h / 6.0  # Normalize to [0, 1]
        
        return torch.stack([h, s, v], dim=-1)
    
    def _apply_edge_softening(self, mask, softness):
        """Apply Gaussian blur for edge softening"""
        import torch.nn.functional as F
        
        # Create Gaussian kernel
        kernel_size = max(3, int(softness * 100))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        sigma = softness * 10
        kernel = self._create_gaussian_kernel(kernel_size, sigma, mask.device)
        
        # Apply blur to mask
        mask_padded = F.pad(mask.permute(0, 3, 1, 2), 
                           (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), 
                           mode='reflect')
        
        mask_blurred = F.conv2d(mask_padded, kernel.unsqueeze(0).unsqueeze(0), padding=0)
        mask_softened = mask_blurred.permute(0, 2, 3, 1)
        
        return mask_softened
    
    def _create_gaussian_kernel(self, kernel_size, sigma, device):
        """Create Gaussian kernel for blurring"""
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
        coords -= kernel_size // 2
        
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        
        kernel = g.outer(g)
        return kernel
    
    def _apply_spill_suppression(self, video, key_r, key_g, key_b, suppression):
        """Reduce color spill from key color"""
        key_color = torch.tensor([key_r, key_g, key_b], device=video.device).view(1, 1, 1, 3)
        
        # Calculate spill amount
        spill_amount = torch.sum(video * key_color, dim=-1, keepdim=True)
        spill_suppression = torch.clamp(spill_amount * suppression, 0, 1)
        
        # Reduce spill
        suppressed_video = video - (key_color * spill_suppression)
        return torch.clamp(suppressed_video, 0, 1)
    
    def _apply_blend_mode(self, base, overlay, mask, blend_mode):
        """Apply different blending modes"""
        if blend_mode == "normal":
            return base * (1 - mask) + overlay * mask
        elif blend_mode == "multiply":
            blended = base * overlay
            return base * (1 - mask) + blended * mask
        elif blend_mode == "screen":
            blended = 1 - (1 - base) * (1 - overlay)
            return base * (1 - mask) + blended * mask
        elif blend_mode == "overlay":
            blended = torch.where(base < 0.5, 
                                 2 * base * overlay, 
                                 1 - 2 * (1 - base) * (1 - overlay))
            return base * (1 - mask) + blended * mask
        elif blend_mode == "soft_light":
            blended = torch.where(overlay < 0.5,
                                 base - (1 - 2 * overlay) * base * (1 - base),
                                 base + (2 * overlay - 1) * (torch.sqrt(base) - base))
            return base * (1 - mask) + blended * mask
        elif blend_mode == "hard_light":
            blended = torch.where(overlay < 0.5,
                                 2 * base * overlay,
                                 1 - 2 * (1 - base) * (1 - overlay))
            return base * (1 - mask) + blended * mask
        else:
            return base * (1 - mask) + overlay * mask
    
    def _create_composite_preview(self, composite_video, mask, fps):
        """Create preview files for composite and mask"""
        import cv2
        
        # Create composite preview
        with tempfile.NamedTemporaryFile(suffix="_composite.mp4", delete=False) as tmp_file:
            composite_path = tmp_file.name
        
        frames_cpu = composite_video.cpu().numpy()
        frames_uint8 = (frames_cpu * 255).astype(np.uint8)
        
        if len(frames_uint8) > 0:
            height, width = frames_uint8.shape[1:3]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(composite_path, fourcc, fps, (width, height))
            
            for frame in frames_uint8:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            logger.info(f"📸 Composite preview saved: {composite_path}")
        
        # Create mask preview
        with tempfile.NamedTemporaryFile(suffix="_mask.mp4", delete=False) as tmp_file:
            mask_path = tmp_file.name
        
        mask_cpu = mask.repeat(1, 1, 1, 3).cpu().numpy()  # Convert to 3-channel
        mask_uint8 = (mask_cpu * 255).astype(np.uint8)
        
        if len(mask_uint8) > 0:
            out = cv2.VideoWriter(mask_path, fourcc, fps, (width, height))
            
            for frame in mask_uint8:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            logger.info(f"📸 Mask preview saved: {mask_path}")