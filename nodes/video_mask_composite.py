import torch
import numpy as np
from .utils import tensor_to_video_frames, logger

class RajVideoMaskComposite:
    """
    Composite source video onto destination using colored mask detection
    Detects specific colors in mask video and composites source content at those locations
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_frames": ("IMAGE", {
                    "tooltip": "Video frames containing colored masks"
                }),
                "source_frames": ("IMAGE", {
                    "tooltip": "Source video content to composite"
                }),
                "mask_color_r": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "tooltip": "Red component of mask color (0-255)"
                }),
                "mask_color_g": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "tooltip": "Green component of mask color (0-255)"
                }),
                "mask_color_b": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "tooltip": "Blue component of mask color (0-255)"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Color matching threshold (0.0-1.0)"
                }),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Frame rate for processing info"
                }),
            },
            "optional": {
                "destination_frames": ("IMAGE", {
                    "tooltip": "Optional destination video (if not provided, uses black background)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("composite_frames", "composite_info", "frame_count", "fps")
    FUNCTION = "composite_with_mask"
    CATEGORY = "Raj Video Processing 🎬"
    
    def composite_with_mask(self, mask_frames, source_frames, mask_color_r, mask_color_g, mask_color_b, 
                           threshold, fps, destination_frames=None):
        
        # Validate inputs
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        # Get frame counts and dimensions
        mask_count = mask_frames.shape[0]
        mask_height = mask_frames.shape[1]
        mask_width = mask_frames.shape[2]
        mask_channels = mask_frames.shape[3]
        
        source_count = source_frames.shape[0]
        source_height = source_frames.shape[1] 
        source_width = source_frames.shape[2]
        source_channels = source_frames.shape[3]
        
        logger.info(f"🎭 Starting mask composite processing")
        logger.info(f"   Mask: {mask_count} frames, {mask_width}x{mask_height}")
        logger.info(f"   Source: {source_count} frames, {source_width}x{source_height}")
        logger.info(f"   Target color: RGB({mask_color_r}, {mask_color_g}, {mask_color_b})")
        logger.info(f"   Threshold: {threshold}")
        
        # Validate aspect ratios - mask and source must match exactly
        if mask_width != source_width or mask_height != source_height:
            raise ValueError(
                f"Aspect ratio mismatch between mask and source!\n"
                f"Mask dimensions: {mask_width}x{mask_height}\n"
                f"Source dimensions: {source_width}x{source_height}\n"
                f"Both videos must have identical dimensions for accurate masking."
            )
        
        # Handle frame count differences
        min_frames = min(mask_count, source_count)
        if mask_count != source_count:
            logger.warning(f"⚠️ Frame count mismatch: mask={mask_count}, source={source_count}")
            logger.info(f"   Using first {min_frames} frames from both videos")
            mask_frames = mask_frames[:min_frames]
            source_frames = source_frames[:min_frames]
        
        # Handle destination frames
        dest_frames = None
        if destination_frames is not None:
            dest_count = destination_frames.shape[0]
            dest_height = destination_frames.shape[1]
            dest_width = destination_frames.shape[2]
            
            logger.info(f"   Destination: {dest_count} frames, {dest_width}x{dest_height}")
            
            # Validate destination aspect ratio
            if dest_width != mask_width or dest_height != mask_height:
                raise ValueError(
                    f"Aspect ratio mismatch between mask and destination!\n"
                    f"Mask dimensions: {mask_width}x{mask_height}\n"
                    f"Destination dimensions: {dest_width}x{dest_height}\n"
                    f"All videos must have identical dimensions for accurate compositing."
                )
            
            # Handle destination frame count
            if dest_count != min_frames:
                logger.warning(f"⚠️ Destination frame count mismatch: dest={dest_count}, processing={min_frames}")
                logger.info(f"   Using first {min(dest_count, min_frames)} destination frames")
                min_frames = min(dest_count, min_frames)
                mask_frames = mask_frames[:min_frames]
                source_frames = source_frames[:min_frames]
                destination_frames = destination_frames[:min_frames]
            
            dest_frames = destination_frames
        else:
            logger.info(f"   Destination: Black background")
        
        # Convert mask color to normalized values [0,1]
        target_color = torch.tensor([
            mask_color_r / 255.0,
            mask_color_g / 255.0, 
            mask_color_b / 255.0
        ], device=mask_frames.device, dtype=mask_frames.dtype)
        
        logger.info(f"   Normalized target color: [{target_color[0]:.3f}, {target_color[1]:.3f}, {target_color[2]:.3f}]")
        
        # Process frames in chunks to handle memory efficiently
        chunk_size = 10  # Process 10 frames at a time
        result_chunks = []
        
        for start_idx in range(0, min_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, min_frames)
            
            # Get chunk of frames
            mask_chunk = mask_frames[start_idx:end_idx]
            source_chunk = source_frames[start_idx:end_idx] 
            dest_chunk = dest_frames[start_idx:end_idx] if dest_frames is not None else None
            
            logger.info(f"   Processing frames {start_idx}-{end_idx-1}")
            
            # Process this chunk
            composite_chunk = self._process_frame_chunk(
                mask_chunk, source_chunk, dest_chunk, target_color, threshold
            )
            
            result_chunks.append(composite_chunk)
            
            # Clear cache to prevent memory buildup
            if mask_frames.device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            elif mask_frames.device.type == "cuda":
                torch.cuda.empty_cache()
        
        # Concatenate all chunks
        logger.info(f"🔗 Concatenating {len(result_chunks)} processed chunks...")
        composite_frames = torch.cat(result_chunks, dim=0)
        
        # Ensure output is in ComfyUI format
        composite_frames_comfy = tensor_to_video_frames(composite_frames)
        
        # Create info string
        dest_info = f"over {dest_frames.shape[0]} dest frames" if dest_frames is not None else "over black background"
        composite_info = f"Mask Composite: RGB({mask_color_r},{mask_color_g},{mask_color_b}) | " \
                        f"Threshold: {threshold} | " \
                        f"Frames: {composite_frames.shape[0]} | " \
                        f"Source {dest_info} | " \
                        f"Size: {mask_width}x{mask_height}"
        
        logger.info(f"✅ Mask composite complete: {composite_frames.shape[0]} frames")
        
        return (
            composite_frames_comfy,
            composite_info,
            composite_frames.shape[0],
            fps
        )
    
    def _process_frame_chunk(self, mask_chunk, source_chunk, dest_chunk, target_color, threshold):
        """
        Process a chunk of frames for mask compositing
        
        Args:
            mask_chunk: Tensor of mask frames [N, H, W, C]
            source_chunk: Tensor of source frames [N, H, W, C]  
            dest_chunk: Tensor of destination frames [N, H, W, C] or None
            target_color: Target color tensor [3] in range [0,1]
            threshold: Color matching threshold
        
        Returns:
            Tensor of composited frames [N, H, W, C]
        """
        
        # Create destination chunk if not provided (black background)
        if dest_chunk is None:
            dest_chunk = torch.zeros_like(source_chunk)
        
        # Detect mask for each frame
        # Calculate Euclidean distance in RGB space for each pixel
        # mask_chunk shape: [N, H, W, C], target_color shape: [C]
        # We want to compute distance for each pixel to target color
        
        # Expand target color to match frame dimensions: [1, 1, 1, C] -> [N, H, W, C]
        target_expanded = target_color.view(1, 1, 1, 3).expand_as(mask_chunk)
        
        # Calculate per-pixel distance in RGB space
        # diff shape: [N, H, W, C]
        diff = mask_chunk - target_expanded
        
        # Euclidean distance: sqrt(sum(diff^2)) for each pixel
        # distances shape: [N, H, W]
        distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        
        # Create binary mask: 1 where distance <= threshold, 0 elsewhere
        # mask shape: [N, H, W]
        binary_mask = (distances <= threshold).float()
        
        # Expand mask to have channel dimension: [N, H, W, 1] -> [N, H, W, C]
        mask_expanded = binary_mask.unsqueeze(-1).expand_as(source_chunk)
        
        # Perform compositing: result = destination * (1-mask) + source * mask
        # This puts source content where mask is 1, destination where mask is 0
        composite = dest_chunk * (1.0 - mask_expanded) + source_chunk * mask_expanded
        
        # Log mask statistics for debugging
        mask_pixels = torch.sum(binary_mask).item()
        total_pixels = binary_mask.numel()
        mask_percentage = (mask_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        logger.debug(f"     Mask coverage: {mask_pixels}/{total_pixels} pixels ({mask_percentage:.1f}%)")
        
        return composite