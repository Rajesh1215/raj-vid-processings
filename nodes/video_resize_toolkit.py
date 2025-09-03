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
        device = frames.device
        
        logger.info(f"🔧 Video Resize Toolkit Processing")
        logger.info(f"   Input: {frame_count} frames, {current_width}x{current_height}, {channels} channels")
        logger.info(f"   Target: {target_width}x{target_height}")
        logger.info(f"   Method: {resize_method}, Interpolation: {interpolation}")
        logger.info(f"   Alpha: {'preserve' if has_alpha and preserve_alpha else 'none'}")
        logger.info(f"   Device: {device}")
        
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
        
        # Calculate chunk size based on memory constraints
        chunk_size = self._calculate_chunk_size(frame_count, current_width, current_height, 
                                               target_width, target_height, channels, device)
        logger.info(f"   Processing in chunks of {chunk_size} frames")
        
        # Process frames in chunks to avoid memory exhaustion
        result_chunks = []
        
        for i in range(0, frame_count, chunk_size):
            end_idx = min(i + chunk_size, frame_count)
            chunk = frames[i:end_idx]
            
            # Convert chunk to PyTorch format: [N, C, H, W]
            chunk_torch = chunk.permute(0, 3, 1, 2)
            
            # Apply resize method
            if resize_method == "center_crop":
                result_chunk = self._center_crop_resize_chunked(
                    chunk_torch, target_width, target_height, torch_interp_mode, device
                )
            elif resize_method == "only_resize":
                result_chunk = self._aspect_preserving_resize_chunked(
                    chunk_torch, target_width, target_height, torch_interp_mode, device
                )
            elif resize_method == "resize_and_fill":
                result_chunk = self._resize_and_fill_chunked(
                    chunk_torch, target_width, target_height, torch_interp_mode,
                    fill_color_r, fill_color_g, fill_color_b, fill_alpha, has_alpha, device
                )
            
            # Convert back to ComfyUI format: [N, H, W, C]
            result_chunk = result_chunk.permute(0, 2, 3, 1)
            
            # Handle alpha channel preservation  
            if not preserve_alpha and has_alpha:
                result_chunk = result_chunk[:, :, :, :3]
            
            # Ensure values are in valid range without memory spike
            result_chunk = self._safe_clamp(result_chunk, device)
            
            result_chunks.append(result_chunk)
            
            # Clear GPU memory after each chunk
            if device.type == "mps" or device.type == "cuda":
                if hasattr(torch, 'mps') and device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
        
        # Concatenate all chunks
        result_frames = torch.cat(result_chunks, dim=0)
        
        # Create info string
        final_channels = result_frames.shape[3]
        resize_info = f"Resize: {resize_method} | " \
                     f"{current_width}x{current_height} → {target_width}x{target_height} | " \
                     f"Frames: {frame_count} | " \
                     f"Channels: {channels}→{final_channels} | " \
                     f"Quality: {interpolation}"
        
        logger.info(f"✅ Video resize complete: {target_width}x{target_height}, {frame_count} frames")
        
        return (
            result_frames,
            resize_info,
            frame_count,
            target_width,
            target_height
        )
    
    def _calculate_chunk_size(self, frame_count, current_width, current_height, target_width, target_height, channels, device):
        """Calculate optimal chunk size based on memory constraints and video dimensions"""
        
        # Calculate memory requirements
        current_pixels = current_width * current_height
        target_pixels = target_width * target_height
        
        # Estimate peak memory usage (input + intermediate + output)
        memory_multiplier = 3.5  # Conservative estimate for peak usage during resize
        
        if device.type == "mps":
            # MPS has strict memory limits
            if current_pixels >= 1280 * 720:  # >= 720p
                chunk_size = max(1, min(4, frame_count))
            elif current_pixels >= 640 * 480:  # >= VGA
                chunk_size = max(1, min(8, frame_count))
            else:
                chunk_size = max(1, min(16, frame_count))
        elif device.type == "cuda":
            # CUDA typically has more memory
            if current_pixels >= 1920 * 1080:  # >= 1080p
                chunk_size = max(1, min(6, frame_count))
            elif current_pixels >= 1280 * 720:  # >= 720p  
                chunk_size = max(1, min(10, frame_count))
            else:
                chunk_size = max(1, min(20, frame_count))
        else:
            # CPU processing
            chunk_size = max(1, min(32, frame_count))
        
        # Ensure chunk size doesn't exceed total frames
        return min(chunk_size, frame_count)
    
    def _safe_clamp(self, tensor, device):
        """Safely clamp tensor values to [0,1] range without causing memory spikes"""
        
        # For large tensors on MPS/CUDA, avoid operations that cause memory spikes
        if device.type in ["mps", "cuda"]:
            tensor_size = tensor.numel() * 4  # float32 = 4 bytes
            if tensor_size > 50_000_000:  # 50MB threshold
                # Check if clamping is actually needed
                sample_indices = torch.randperm(min(1000, tensor.numel()))[:100]
                sample_values = tensor.view(-1)[sample_indices]
                
                if torch.any(sample_values < 0) or torch.any(sample_values > 1):
                    # Move to CPU, clamp, then move back
                    original_device = tensor.device
                    tensor_cpu = tensor.cpu()
                    tensor_clamped = torch.clamp(tensor_cpu, 0.0, 1.0)
                    tensor = tensor_clamped.to(original_device)
                
                return tensor
        
        # For smaller tensors or CPU, clamp directly
        return torch.clamp(tensor, 0.0, 1.0)
    
    def _center_crop_resize_chunked(self, frames_torch, target_width, target_height, interp_mode, device):
        """Memory-efficient center crop resize for chunked processing"""
        current_height = frames_torch.shape[2]
        current_width = frames_torch.shape[3]
        
        # Calculate scale to cover target area (larger of the two scales)
        scale_w = target_width / current_width
        scale_h = target_height / current_height
        scale = max(scale_w, scale_h)
        
        # Calculate intermediate dimensions
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)
        
        # For very large intermediate sizes on MPS, use CPU processing
        intermediate_pixels = new_width * new_height
        if device.type == "mps" and intermediate_pixels > 1_000_000:  # > 1MP intermediate
            # Process on CPU to avoid MPS memory issues
            original_device = frames_torch.device
            frames_cpu = frames_torch.cpu()
            
            # Resize on CPU
            resized_cpu = F.interpolate(frames_cpu, size=(new_height, new_width), 
                                       mode=interp_mode, align_corners=False)
            
            # Crop on CPU
            start_y = (new_height - target_height) // 2
            start_x = (new_width - target_width) // 2
            cropped_cpu = resized_cpu[:, :, start_y:start_y+target_height, start_x:start_x+target_width]
            
            # Move back to original device
            result = cropped_cpu.to(original_device)
            
            # Cleanup CPU tensors
            del frames_cpu, resized_cpu, cropped_cpu
            
        else:
            # Process on original device
            resized = F.interpolate(frames_torch, size=(new_height, new_width), 
                                   mode=interp_mode, align_corners=False)
            
            # Crop center portion
            start_y = (new_height - target_height) // 2
            start_x = (new_width - target_width) // 2
            result = resized[:, :, start_y:start_y+target_height, start_x:start_x+target_width]
            
            del resized
        
        return result
    
    def _aspect_preserving_resize_chunked(self, frames_torch, target_width, target_height, interp_mode, device):
        """Memory-efficient aspect-preserving resize for chunked processing"""
        current_height = frames_torch.shape[2]
        current_width = frames_torch.shape[3]
        
        # Calculate scale to fit within target area (smaller of the two scales)
        scale_w = target_width / current_width
        scale_h = target_height / current_height
        scale = min(scale_w, scale_h)
        
        # Calculate final dimensions
        final_width = int(current_width * scale)
        final_height = int(current_height * scale)
        
        # Use CPU for very large resize operations on MPS
        if device.type == "mps" and (current_width * current_height > 1_000_000):
            original_device = frames_torch.device
            frames_cpu = frames_torch.cpu()
            resized_cpu = F.interpolate(frames_cpu, size=(final_height, final_width), 
                                       mode=interp_mode, align_corners=False)
            result = resized_cpu.to(original_device)
            del frames_cpu, resized_cpu
        else:
            result = F.interpolate(frames_torch, size=(final_height, final_width), 
                                  mode=interp_mode, align_corners=False)
        
        return result
    
    def _resize_and_fill_chunked(self, frames_torch, target_width, target_height, interp_mode,
                               fill_r, fill_g, fill_b, fill_alpha, has_alpha, device):
        """Memory-efficient resize and fill for chunked processing"""
        current_height = frames_torch.shape[2]
        current_width = frames_torch.shape[3]
        
        # Calculate scale to fit within target area
        scale_w = target_width / current_width
        scale_h = target_height / current_height
        scale = min(scale_w, scale_h)
        
        # Calculate scaled dimensions
        scaled_width = int(current_width * scale)
        scaled_height = int(current_height * scale)
        
        # Use CPU processing for large operations on MPS
        if device.type == "mps" and (current_width * current_height > 1_000_000):
            original_device = frames_torch.device
            frames_cpu = frames_torch.cpu()
            
            # Resize on CPU
            resized_cpu = F.interpolate(frames_cpu, size=(scaled_height, scaled_width), 
                                       mode=interp_mode, align_corners=False)
            
            # Create background on CPU
            batch_size = frames_cpu.shape[0]
            channels = frames_cpu.shape[1]
            
            if has_alpha and channels == 4:
                background_cpu = torch.full((batch_size, 4, target_height, target_width), 
                                           fill_alpha, dtype=frames_cpu.dtype)
                background_cpu[:, 0, :, :] = fill_r * fill_alpha  # Pre-multiply RGB
                background_cpu[:, 1, :, :] = fill_g * fill_alpha
                background_cpu[:, 2, :, :] = fill_b * fill_alpha
            else:
                background_cpu = torch.zeros((batch_size, 3, target_height, target_width), 
                                           dtype=frames_cpu.dtype)
                background_cpu[:, 0, :, :] = fill_r
                background_cpu[:, 1, :, :] = fill_g
                background_cpu[:, 2, :, :] = fill_b
            
            # Center the resized content
            start_y = (target_height - scaled_height) // 2
            start_x = (target_width - scaled_width) // 2
            
            background_cpu[:, :, start_y:start_y+scaled_height, start_x:start_x+scaled_width] = resized_cpu
            
            result = background_cpu.to(original_device)
            del frames_cpu, resized_cpu, background_cpu
            
        else:
            # Process on original device
            resized = F.interpolate(frames_torch, size=(scaled_height, scaled_width), 
                                   mode=interp_mode, align_corners=False)
            
            batch_size = frames_torch.shape[0]
            channels = frames_torch.shape[1]
            
            # Create background
            if has_alpha and channels == 4:
                background = torch.full((batch_size, 4, target_height, target_width), 
                                       fill_alpha, device=device, dtype=frames_torch.dtype)
                background[:, 0, :, :] = fill_r * fill_alpha
                background[:, 1, :, :] = fill_g * fill_alpha  
                background[:, 2, :, :] = fill_b * fill_alpha
            else:
                background = torch.zeros((batch_size, 3, target_height, target_width), 
                                        device=device, dtype=frames_torch.dtype)
                background[:, 0, :, :] = fill_r
                background[:, 1, :, :] = fill_g
                background[:, 2, :, :] = fill_b
            
            # Center the resized content
            start_y = (target_height - scaled_height) // 2
            start_x = (target_width - scaled_width) // 2
            
            background[:, :, start_y:start_y+scaled_height, start_x:start_x+scaled_width] = resized
            
            result = background
            del resized
        
        return result
    
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