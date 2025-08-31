import torch
import torch.nn.functional as F
import cv2
import numpy as np
from .utils import tensor_to_video_frames, logger

class RajVideoOverlay:
    """
    Professional video overlay with alpha channel support and full/bbox positioning modes
    Supports RGBA overlays with proper alpha compositing
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_frames": ("IMAGE", {
                    "tooltip": "Base video frames (RGB or RGBA)"
                }),
                "overlay_frames": ("IMAGE", {
                    "tooltip": "Overlay video frames (RGB or RGBA with alpha channel)"
                }),
                "overlay_mode": (["full", "bbox"], {
                    "default": "full",
                    "tooltip": "Full screen or bounding box positioning"
                }),
                "resize_mode": (["center_crop", "resize", "fill", "stretch"], {
                    "default": "center_crop",
                    "tooltip": "How to resize overlay to fit target area"
                }),
                "base_opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Overall overlay opacity (0.0-1.0)"
                }),
                "respect_alpha": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use overlay's alpha channel if present"
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
                "bbox_x": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Bounding box X position (pixels, bbox mode only)"
                }),
                "bbox_y": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Bounding box Y position (pixels, bbox mode only)"
                }),
                "bbox_width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Bounding box width (pixels, bbox mode only)"
                }),
                "bbox_height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Bounding box height (pixels, bbox mode only)"
                }),
                "opacity_settings": ("OPACITY_GRADIENT", {
                    "tooltip": "Optional opacity gradient settings from RajVideoOpacityGradient"
                }),
                "chroma_settings": ("CHROMA_KEY", {
                    "tooltip": "Optional chroma key settings from RajVideoChromaKey"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("composite_frames", "overlay_info", "frame_count", "fps")
    FUNCTION = "overlay_videos"
    CATEGORY = "Raj Video Processing 🎬"
    
    def overlay_videos(self, source_frames, overlay_frames, overlay_mode, resize_mode, 
                      base_opacity, respect_alpha, fps, bbox_x=0, bbox_y=0, 
                      bbox_width=512, bbox_height=512, opacity_settings=None, chroma_settings=None):
        
        # Get frame info
        source_count = source_frames.shape[0]
        source_height = source_frames.shape[1]
        source_width = source_frames.shape[2]
        source_channels = source_frames.shape[3]
        
        overlay_count = overlay_frames.shape[0]
        overlay_height = overlay_frames.shape[1]
        overlay_width = overlay_frames.shape[2]
        overlay_channels = overlay_frames.shape[3]
        
        # Detect alpha channels
        source_has_alpha = source_channels == 4
        overlay_has_alpha = overlay_channels == 4
        
        logger.info(f"🎭 Video Overlay Processing")
        logger.info(f"   Source: {source_count} frames, {source_width}x{source_height}, {source_channels} channels")
        logger.info(f"   Overlay: {overlay_count} frames, {overlay_width}x{overlay_height}, {overlay_channels} channels")
        logger.info(f"   Mode: {overlay_mode}, Resize: {resize_mode}")
        logger.info(f"   Alpha channels - Source: {source_has_alpha}, Overlay: {overlay_has_alpha}")
        
        # Synchronize frame counts
        min_frames = min(source_count, overlay_count)
        if source_count != overlay_count:
            logger.warning(f"⚠️ Frame count mismatch: source={source_count}, overlay={overlay_count}")
            
            # Loop the shorter video
            if source_count < overlay_count:
                # Loop source frames
                repeat_factor = (overlay_count // source_count) + 1
                source_frames = source_frames.repeat(repeat_factor, 1, 1, 1)[:overlay_count]
                min_frames = overlay_count
            else:
                # Loop overlay frames  
                repeat_factor = (source_count // overlay_count) + 1
                overlay_frames = overlay_frames.repeat(repeat_factor, 1, 1, 1)[:source_count]
                min_frames = source_count
            
            logger.info(f"   Synchronized to {min_frames} frames")
        
        # Determine target dimensions based on overlay mode
        if overlay_mode == "full":
            target_width = source_width
            target_height = source_height
            pos_x = 0
            pos_y = 0
            logger.info(f"   Full mode: overlay sized to {target_width}x{target_height}")
        else:  # bbox mode
            target_width = bbox_width
            target_height = bbox_height
            pos_x = bbox_x
            pos_y = bbox_y
            
            # Validate bbox is within source bounds
            if pos_x + target_width > source_width or pos_y + target_height > source_height:
                raise ValueError(
                    f"Bounding box exceeds source dimensions!\n"
                    f"Bbox: {pos_x},{pos_y} {target_width}x{target_height}\n"
                    f"Source: {source_width}x{source_height}"
                )
            
            logger.info(f"   Bbox mode: overlay at ({pos_x},{pos_y}) sized to {target_width}x{target_height}")
        
        # Apply chroma key if provided
        processed_overlay = overlay_frames
        if chroma_settings is not None:
            logger.info(f"   Applying chroma key settings")
            processed_overlay = self._apply_chroma_key(overlay_frames, chroma_settings)
            overlay_has_alpha = True  # Chroma key produces alpha
        
        # Resize overlay to target dimensions
        resized_overlay = self._resize_overlay(processed_overlay, target_width, target_height, resize_mode)
        
        # Process in chunks for memory efficiency
        chunk_size = 10
        result_chunks = []
        
        for start_idx in range(0, min_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, min_frames)
            
            source_chunk = source_frames[start_idx:end_idx]
            overlay_chunk = resized_overlay[start_idx:end_idx]
            
            logger.info(f"   Processing frames {start_idx}-{end_idx-1}")
            
            # Apply overlay to this chunk
            composite_chunk = self._composite_chunk(
                source_chunk, overlay_chunk, pos_x, pos_y, 
                base_opacity, respect_alpha, overlay_has_alpha,
                opacity_settings
            )
            
            result_chunks.append(composite_chunk)
            
            # Clear cache
            if source_frames.device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            elif source_frames.device.type == "cuda":
                torch.cuda.empty_cache()
        
        # Concatenate results
        logger.info(f"🔗 Concatenating {len(result_chunks)} processed chunks...")
        composite_frames = torch.cat(result_chunks, dim=0)
        
        # Ensure ComfyUI format
        composite_frames_comfy = tensor_to_video_frames(composite_frames)
        
        # Create info string
        alpha_info = f"Alpha: {overlay_has_alpha}" if respect_alpha else "Alpha: ignored"
        overlay_info = f"Video Overlay: {overlay_mode} mode | " \
                      f"Frames: {min_frames} | " \
                      f"Size: {source_width}x{source_height} | " \
                      f"Opacity: {base_opacity} | " \
                      f"{alpha_info}"
        
        logger.info(f"✅ Video overlay complete: {min_frames} frames")
        
        return (
            composite_frames_comfy,
            overlay_info,
            min_frames,
            fps
        )
    
    def _resize_overlay(self, overlay_frames, target_width, target_height, resize_mode):
        """
        Resize overlay frames to target dimensions based on resize mode
        """
        current_height = overlay_frames.shape[1]
        current_width = overlay_frames.shape[2] 
        channels = overlay_frames.shape[3]
        
        if current_width == target_width and current_height == target_height:
            return overlay_frames  # No resize needed
        
        logger.info(f"   Resizing overlay: {current_width}x{current_height} → {target_width}x{target_height} ({resize_mode})")
        
        # Convert to format expected by F.interpolate: [N, C, H, W]
        overlay_reshaped = overlay_frames.permute(0, 3, 1, 2)
        
        if resize_mode == "center_crop":
            # Scale to fit larger dimension, then crop center
            scale_w = target_width / current_width
            scale_h = target_height / current_height
            scale = max(scale_w, scale_h)  # Scale to cover target area
            
            new_width = int(current_width * scale)
            new_height = int(current_height * scale)
            
            # Resize
            resized = F.interpolate(overlay_reshaped, size=(new_height, new_width), 
                                  mode='bilinear', align_corners=False)
            
            # Crop center
            start_y = (new_height - target_height) // 2
            start_x = (new_width - target_width) // 2
            cropped = resized[:, :, start_y:start_y+target_height, start_x:start_x+target_width]
            
            result = cropped.permute(0, 2, 3, 1)
            
        elif resize_mode == "fill":
            # Scale to fit smaller dimension, then pad
            scale_w = target_width / current_width
            scale_h = target_height / current_height
            scale = min(scale_w, scale_h)  # Scale to fit inside target area
            
            new_width = int(current_width * scale)
            new_height = int(current_height * scale)
            
            # Resize
            resized = F.interpolate(overlay_reshaped, size=(new_height, new_width), 
                                  mode='bilinear', align_corners=False)
            
            # Create padded result
            result_tensor = torch.zeros((overlay_frames.shape[0], channels, target_height, target_width), 
                                      device=overlay_frames.device, dtype=overlay_frames.dtype)
            
            # Calculate padding positions
            pad_y = (target_height - new_height) // 2
            pad_x = (target_width - new_width) // 2
            
            # Place resized overlay in center
            result_tensor[:, :, pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
            
            result = result_tensor.permute(0, 2, 3, 1)
            
        else:  # "resize" or "stretch" 
            # Direct resize to exact dimensions (may distort aspect ratio)
            resized = F.interpolate(overlay_reshaped, size=(target_height, target_width), 
                                  mode='bilinear', align_corners=False)
            result = resized.permute(0, 2, 3, 1)
        
        return result
    
    def _composite_chunk(self, source_chunk, overlay_chunk, pos_x, pos_y, 
                        base_opacity, respect_alpha, overlay_has_alpha, opacity_settings):
        """
        Composite overlay onto source for a chunk of frames
        """
        # Start with source as base
        result = source_chunk.clone()
        
        # Get overlay dimensions
        overlay_height = overlay_chunk.shape[1]
        overlay_width = overlay_chunk.shape[2]
        
        # Extract overlay region from source
        source_region = result[:, pos_y:pos_y+overlay_height, pos_x:pos_x+overlay_width, :]
        
        # Handle alpha channel compositing
        if overlay_has_alpha and respect_alpha:
            # Extract RGB and alpha from overlay
            overlay_rgb = overlay_chunk[:, :, :, :3]
            overlay_alpha = overlay_chunk[:, :, :, 3:4]
            
            # Apply base opacity
            final_alpha = overlay_alpha * base_opacity
            
            # Apply gradient opacity if provided
            if opacity_settings is not None:
                gradient_mask = self._apply_opacity_gradient(opacity_settings, overlay_height, overlay_width)
                final_alpha = final_alpha * gradient_mask.unsqueeze(0).unsqueeze(-1)
            
            # Alpha composite: result = source * (1-alpha) + overlay * alpha
            composited_region = source_region[:, :, :, :3] * (1 - final_alpha) + overlay_rgb * final_alpha
            
            # If source has alpha, blend alpha channels too
            if source_region.shape[-1] == 4:
                source_alpha = source_region[:, :, :, 3:4]
                final_source_alpha = source_alpha * (1 - final_alpha) + final_alpha
                composited_region = torch.cat([composited_region, final_source_alpha], dim=-1)
            
        else:
            # No alpha channel, use traditional opacity blending
            final_opacity = base_opacity
            
            if opacity_settings is not None:
                gradient_mask = self._apply_opacity_gradient(opacity_settings, overlay_height, overlay_width)
                final_opacity = final_opacity * gradient_mask.unsqueeze(0).unsqueeze(-1)
            
            overlay_rgb = overlay_chunk[:, :, :, :3]
            composited_region = source_region[:, :, :, :3] * (1 - final_opacity) + overlay_rgb * final_opacity
            
            # Preserve source alpha if present
            if source_region.shape[-1] == 4:
                composited_region = torch.cat([composited_region, source_region[:, :, :, 3:4]], dim=-1)
        
        # Place composited region back into result
        result[:, pos_y:pos_y+overlay_height, pos_x:pos_x+overlay_width, :] = composited_region
        
        return result
    
    def _apply_chroma_key(self, frames, chroma_settings):
        """
        Apply chroma key settings to frames
        """
        if not chroma_settings.get('enabled', False):
            return frames
        
        logger.info("   Applying chroma key from settings")
        
        target_color = chroma_settings['color']
        tolerance = chroma_settings['tolerance']
        edge_softness = chroma_settings.get('edge_softness', 0.05)
        
        # Apply the same chroma key logic as in RajVideoChromaKey
        target_expanded = target_color.view(1, 1, 1, 3).expand_as(frames)
        diff = frames - target_expanded
        distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        
        # Create alpha mask
        base_mask = 1.0 - torch.clamp(distances / tolerance, 0.0, 1.0)
        
        if edge_softness > 0.0:
            soft_tolerance = tolerance + edge_softness
            soft_mask = 1.0 - torch.clamp(distances / soft_tolerance, 0.0, 1.0)
            edge_blend = torch.clamp((distances - tolerance) / edge_softness, 0.0, 1.0)
            alpha_mask = base_mask * (1.0 - edge_blend) + soft_mask * edge_blend
        else:
            alpha_mask = base_mask
        
        # Apply smooth step
        alpha_mask = alpha_mask * alpha_mask * (3.0 - 2.0 * alpha_mask)
        
        # Add alpha channel to frames
        alpha_mask = alpha_mask.unsqueeze(-1)
        frames_with_alpha = torch.cat([frames, alpha_mask], dim=-1)
        
        return frames_with_alpha
    
    def _apply_opacity_gradient(self, opacity_settings, height, width):
        """
        Apply opacity gradient settings from RajVideoOpacityGradient
        """
        if not isinstance(opacity_settings, dict):
            return torch.ones(height, width, device='cpu')
        
        logger.info(f"   Applying {opacity_settings.get('type', 'unknown')} opacity gradient")
        
        # Get the gradient mask and resize if needed
        gradient_mask = opacity_settings.get('mask')
        if gradient_mask is None:
            return torch.ones(height, width, device='cpu')
        
        # Move to appropriate device
        target_device = opacity_settings.get('device', 'cpu')
        gradient_mask = gradient_mask.to(target_device)
        
        # Resize if dimensions don't match
        if gradient_mask.shape[0] != height or gradient_mask.shape[1] != width:
            logger.info(f"   Resizing gradient from {gradient_mask.shape} to {height}x{width}")
            # Add batch and channel dims for F.interpolate
            gradient_reshaped = gradient_mask.unsqueeze(0).unsqueeze(0)
            gradient_resized = F.interpolate(gradient_reshaped, size=(height, width), 
                                           mode='bilinear', align_corners=False)
            gradient_mask = gradient_resized.squeeze(0).squeeze(0)
        
        return gradient_mask