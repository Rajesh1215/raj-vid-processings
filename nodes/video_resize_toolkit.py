import torch
import torch.nn.functional as F
import numpy as np
from .utils import tensor_to_video_frames, logger

class RajVideoResizeToolkit:
    """
    Professional video resizing toolkit with alpha channel preservation
    Provides center crop, resize, and fill modes with high-quality interpolation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Input video frames (RGB or RGBA)"
                }),
                "target_width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Target width in pixels"
                }),
                "target_height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Target height in pixels"
                }),
                "resize_method": (["center_crop", "only_resize", "resize_and_fill"], {
                    "default": "center_crop",
                    "tooltip": "Resize method to use"
                }),
                "interpolation": (["lanczos", "cubic", "linear", "nearest"], {
                    "default": "lanczos",
                    "tooltip": "Interpolation method for high quality"
                }),
                "preserve_alpha": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Maintain alpha channel if present"
                }),
            },
            "optional": {
                "fill_color_r": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "tooltip": "Fill background red component (resize_and_fill mode)"
                }),
                "fill_color_g": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "tooltip": "Fill background green component (resize_and_fill mode)"
                }),
                "fill_color_b": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "tooltip": "Fill background blue component (resize_and_fill mode)"
                }),
                "fill_alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Fill alpha value (only if input has alpha channel)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("resized_frames", "resize_info", "frame_count", "output_width", "output_height")
    FUNCTION = "resize_video"
    CATEGORY = "Raj Video Processing 🎬"
    
    def resize_video(self, frames, target_width, target_height, resize_method, interpolation, 
                    preserve_alpha, fill_color_r=0, fill_color_g=0, fill_color_b=0, fill_alpha=1.0):
        
        # Get input info
        frame_count = frames.shape[0]
        current_height = frames.shape[1]
        current_width = frames.shape[2]
        channels = frames.shape[3]
        has_alpha = channels == 4
        
        logger.info(f"🔧 Video Resize Toolkit Processing")
        logger.info(f"   Input: {frame_count} frames, {current_width}x{current_height}, {channels} channels")
        logger.info(f"   Target: {target_width}x{target_height}")
        logger.info(f"   Method: {resize_method}, Interpolation: {interpolation}")
        logger.info(f"   Alpha: {'preserve' if has_alpha and preserve_alpha else 'none'}")
        
        # Check if resize is needed
        if current_width == target_width and current_height == target_height:
            logger.info("   No resize needed, dimensions already match")
            resize_info = f"Resize: No change needed | {target_width}x{target_height} | Frames: {frame_count}"
            return (frames, resize_info, frame_count, target_width, target_height)
        
        # Determine interpolation mode for PyTorch
        interp_mode_map = {
            "lanczos": "bicubic",  # PyTorch doesn't have lanczos, use bicubic as best alternative
            "cubic": "bicubic",
            "linear": "bilinear", 
            "nearest": "nearest"
        }
        torch_interp_mode = interp_mode_map[interpolation]
        
        # Convert frames to PyTorch format: [N, C, H, W] 
        frames_torch = frames.permute(0, 3, 1, 2)
        
        if resize_method == "center_crop":
            # Scale to fit larger dimension, then crop center
            result_frames = self._center_crop_resize(
                frames_torch, target_width, target_height, torch_interp_mode
            )
            
        elif resize_method == "only_resize":
            # Scale maintaining aspect ratio to fit within target dimensions
            result_frames = self._aspect_preserving_resize(
                frames_torch, target_width, target_height, torch_interp_mode
            )
            
        elif resize_method == "resize_and_fill":
            # Scale to fit, then pad with background color
            result_frames = self._resize_and_fill(
                frames_torch, target_width, target_height, torch_interp_mode,
                fill_color_r, fill_color_g, fill_color_b, fill_alpha, has_alpha
            )
        
        # Convert back to ComfyUI format: [N, H, W, C]
        result_frames = result_frames.permute(0, 2, 3, 1)
        
        # Handle alpha channel preservation
        if not preserve_alpha and has_alpha:
            logger.info("   Removing alpha channel as requested")
            result_frames = result_frames[:, :, :, :3]
        
        # Ensure ComfyUI format
        result_frames_comfy = tensor_to_video_frames(result_frames)
        
        # Create info string
        final_channels = result_frames.shape[3]
        resize_info = f"Resize: {resize_method} | " \
                     f"{current_width}x{current_height} → {target_width}x{target_height} | " \
                     f"Frames: {frame_count} | " \
                     f"Channels: {channels}→{final_channels} | " \
                     f"Quality: {interpolation}"
        
        logger.info(f"✅ Video resize complete: {target_width}x{target_height}, {frame_count} frames")
        
        return (
            result_frames_comfy,
            resize_info,
            frame_count,
            target_width,
            target_height
        )
    
    def _center_crop_resize(self, frames_torch, target_width, target_height, interp_mode):
        """
        Scale to cover target area completely, then crop center portion
        """
        current_height = frames_torch.shape[2]
        current_width = frames_torch.shape[3]
        
        # Calculate scale to cover target area (larger of the two scales)
        scale_w = target_width / current_width
        scale_h = target_height / current_height
        scale = max(scale_w, scale_h)
        
        # Calculate intermediate dimensions
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)
        
        logger.info(f"   Center crop: scale {scale:.3f} → {new_width}x{new_height}, then crop center")
        
        # Resize to cover target area
        resized = F.interpolate(frames_torch, size=(new_height, new_width), 
                               mode=interp_mode, align_corners=False)
        
        # Crop center portion
        start_y = (new_height - target_height) // 2
        start_x = (new_width - target_width) // 2
        
        cropped = resized[:, :, start_y:start_y+target_height, start_x:start_x+target_width]
        
        return cropped
    
    def _aspect_preserving_resize(self, frames_torch, target_width, target_height, interp_mode):
        """
        Resize maintaining aspect ratio to fit within target dimensions
        """
        current_height = frames_torch.shape[2]
        current_width = frames_torch.shape[3]
        
        # Calculate scale to fit within target area (smaller of the two scales)
        scale_w = target_width / current_width
        scale_h = target_height / current_height
        scale = min(scale_w, scale_h)
        
        # Calculate final dimensions
        final_width = int(current_width * scale)
        final_height = int(current_height * scale)
        
        logger.info(f"   Only resize: scale {scale:.3f} → {final_width}x{final_height}")
        
        # Resize maintaining aspect ratio
        resized = F.interpolate(frames_torch, size=(final_height, final_width), 
                               mode=interp_mode, align_corners=False)
        
        return resized
    
    def _resize_and_fill(self, frames_torch, target_width, target_height, interp_mode,
                        fill_r, fill_g, fill_b, fill_alpha, has_alpha):
        """
        Resize maintaining aspect ratio, then pad with background color to exact target size
        """
        current_height = frames_torch.shape[2]
        current_width = frames_torch.shape[3]
        channels = frames_torch.shape[1]
        
        # Calculate scale to fit within target area
        scale_w = target_width / current_width
        scale_h = target_height / current_height
        scale = min(scale_w, scale_h)
        
        # Calculate intermediate dimensions
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)
        
        logger.info(f"   Resize and fill: scale {scale:.3f} → {new_width}x{new_height}, then pad to {target_width}x{target_height}")
        
        # Resize maintaining aspect ratio
        resized = F.interpolate(frames_torch, size=(new_height, new_width), 
                               mode=interp_mode, align_corners=False)
        
        # Create background with fill color
        fill_color = torch.tensor([fill_r/255.0, fill_g/255.0, fill_b/255.0], 
                                 device=frames_torch.device, dtype=frames_torch.dtype)
        
        if has_alpha:
            fill_alpha_tensor = torch.tensor([fill_alpha], device=frames_torch.device, dtype=frames_torch.dtype)
            fill_color = torch.cat([fill_color, fill_alpha_tensor])
        
        # Create result tensor with fill color
        result = torch.zeros((frames_torch.shape[0], channels, target_height, target_width),
                           device=frames_torch.device, dtype=frames_torch.dtype)
        
        # Fill with background color
        for c in range(channels):
            result[:, c, :, :] = fill_color[c] if c < len(fill_color) else 0.0
        
        # Calculate padding positions (center the resized content)
        pad_y = (target_height - new_height) // 2
        pad_x = (target_width - new_width) // 2
        
        # Place resized content in center
        result[:, :, pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
        
        return result
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # This node's output depends on its inputs, so always recompute
        return float("nan")