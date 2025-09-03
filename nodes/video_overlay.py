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
                "chunk_size": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of frames to process at once (lower = less memory)"
                }),
                "force_cpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force CPU processing to avoid GPU memory issues"
                }),
                "start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Overlay start time in seconds (0.0 = beginning)"
                }),
                "duration": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Overlay duration in seconds (-1.0 = full video duration)"
                }),
                "time_unit": (["seconds", "frames"], {
                    "default": "seconds",
                    "tooltip": "Time unit for start_time and duration parameters"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("composite_frames", "overlay_info", "frame_count", "fps")
    FUNCTION = "overlay_videos"
    CATEGORY = "Raj Video Processing 🎬"
    
    def overlay_videos(self, source_frames, overlay_frames, overlay_mode, resize_mode, 
                      base_opacity, respect_alpha, fps, bbox_x=0, bbox_y=0, 
                      bbox_width=512, bbox_height=512, opacity_settings=None, chroma_settings=None,
                      chunk_size=5, force_cpu=False, start_time=0.0, duration=-1.0, time_unit="seconds"):
        
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
        
        # Handle CPU fallback
        if force_cpu and source_frames.device.type != "cpu":
            logger.info(f"🔄 Force CPU mode enabled, moving tensors to CPU...")
            source_frames = source_frames.cpu()
            overlay_frames = overlay_frames.cpu()
        
        logger.info(f"🎭 Video Overlay Processing")
        logger.info(f"   Source: {source_count} frames, {source_width}x{source_height}, {source_channels} channels")
        logger.info(f"   Overlay: {overlay_count} frames, {overlay_width}x{overlay_height}, {overlay_channels} channels")
        logger.info(f"   Mode: {overlay_mode}, Resize: {resize_mode}")
        logger.info(f"   Alpha channels - Source: {source_has_alpha}, Overlay: {overlay_has_alpha}")
        logger.info(f"   Processing on: {source_frames.device.type}, Chunk size: {chunk_size}")
        
        # Calculate time-based frame boundaries
        if time_unit == "frames":
            # Direct frame indices
            start_frame = max(0, int(start_time))
            if duration >= 0:
                end_frame = min(source_count, start_frame + int(duration))
            else:
                end_frame = source_count
        else:
            # Convert seconds to frames
            start_frame = max(0, int(start_time * fps))
            if duration >= 0:
                end_frame = min(source_count, start_frame + int(duration * fps))
            else:
                end_frame = source_count
        
        # Validate time boundaries
        if start_frame >= source_count:
            raise ValueError(f"Start time/frame {start_time} is beyond video duration ({source_count} frames @ {fps} fps)")
        
        if start_frame >= end_frame:
            logger.warning(f"⚠️ Invalid time range: start={start_frame}, end={end_frame}. Using full video.")
            start_frame = 0
            end_frame = source_count
        
        overlay_duration_frames = end_frame - start_frame
        overlay_duration_seconds = overlay_duration_frames / fps
        
        logger.info(f"⏱️ Time-based overlay:")
        logger.info(f"   Start: frame {start_frame} ({start_frame/fps:.2f}s)")
        logger.info(f"   End: frame {end_frame} ({end_frame/fps:.2f}s)")
        logger.info(f"   Duration: {overlay_duration_frames} frames ({overlay_duration_seconds:.2f}s)")
        
        # Early exit if no overlay needed
        if overlay_duration_frames <= 0:
            logger.warning("⚠️ No overlay duration specified, returning original video")
            overlay_info = "Video Overlay: No overlay applied (zero duration)"
            return (source_frames, overlay_info, source_count, fps)
        
        # Synchronize overlay frame count with overlay duration
        if overlay_count < overlay_duration_frames:
            logger.warning(f"⚠️ Overlay video ({overlay_count} frames) shorter than overlay duration ({overlay_duration_frames} frames)")
            # Loop overlay frames to match duration
            repeat_factor = (overlay_duration_frames // overlay_count) + 1
            overlay_frames = overlay_frames.repeat(repeat_factor, 1, 1, 1)[:overlay_duration_frames]
            logger.info(f"   Looped overlay to {overlay_duration_frames} frames")
        elif overlay_count > overlay_duration_frames:
            # Trim overlay to match duration
            overlay_frames = overlay_frames[:overlay_duration_frames]
            logger.info(f"   Trimmed overlay to {overlay_duration_frames} frames")
        
        # Update overlay count after synchronization
        overlay_count = overlay_frames.shape[0]
        
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
        
        # Video segmentation for time-based overlay
        logger.info(f"✂️ Segmenting video for time-based overlay processing")
        
        # Extract video segments (keep non-overlay parts on CPU to save GPU memory)
        original_device = source_frames.device
        pre_overlay_section = None
        post_overlay_section = None
        
        if start_frame > 0:
            pre_overlay_section = source_frames[:start_frame].cpu()  # Move to CPU immediately
            logger.info(f"   Pre-overlay section: {pre_overlay_section.shape[0]} frames (CPU)")
        
        if end_frame < source_count:
            post_overlay_section = source_frames[end_frame:].cpu()  # Move to CPU immediately
            logger.info(f"   Post-overlay section: {post_overlay_section.shape[0]} frames (CPU)")
        
        # Extract the section that needs overlay processing (keep on GPU for processing)
        overlay_section = source_frames[start_frame:end_frame]
        logger.info(f"   Overlay section: {overlay_section.shape[0]} frames (GPU processing)")
        
        # Process overlay section in chunks for memory efficiency
        logger.info(f"⚙️ Processing overlay section with chunk size: {chunk_size}")
        result_chunks = []
        
        # Check available memory and adjust chunk size if needed  
        if original_device.type == "mps":
            try:
                # Calculate total memory requirements for overlay processing
                frame_pixels = source_width * source_height
                estimated_memory_gb = (overlay_duration_frames * frame_pixels * 4 * 4) / (1024**3)  # 4 bytes per pixel, 4x multiplier for processing
                
                logger.info(f"   Estimated overlay memory requirement: {estimated_memory_gb:.2f} GB")
                
                # Apply more conservative chunking for large videos
                if estimated_memory_gb > 2.0:  # Very large video
                    chunk_size = max(1, min(1, chunk_size))  # Single frame processing
                    logger.warning(f"⚠️ Very large overlay section detected ({estimated_memory_gb:.2f} GB), using single-frame processing")
                elif estimated_memory_gb > 1.0:  # Large video
                    chunk_size = max(1, min(2, chunk_size))  # 2 frames max
                    logger.warning(f"⚠️ Large overlay section detected ({estimated_memory_gb:.2f} GB), reducing chunk size to 2")
                elif overlay_duration_frames > 100 and chunk_size > 3:
                    chunk_size = max(1, min(3, chunk_size))  # 3 frames max
                    logger.warning(f"⚠️ Large overlay section detected, reducing chunk size to 3 for memory safety")
            except:
                pass
        
        for start_idx in range(0, overlay_duration_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, overlay_duration_frames)
            
            try:
                # Extract chunks from overlay section (not full source)
                source_chunk = overlay_section[start_idx:end_idx]
                overlay_chunk = resized_overlay[start_idx:end_idx]
                
                logger.info(f"   Processing overlay frames {start_idx}-{end_idx-1}")
                
                # Apply overlay to this chunk
                composite_chunk = self._composite_chunk(
                    source_chunk, overlay_chunk, pos_x, pos_y, 
                    base_opacity, respect_alpha, overlay_has_alpha,
                    opacity_settings
                )
                
                # Move chunk to CPU immediately to free GPU memory
                composite_chunk_cpu = composite_chunk.cpu()
                result_chunks.append(composite_chunk_cpu)
                
                # Explicitly delete GPU tensors to free memory
                del composite_chunk
                del source_chunk
                del overlay_chunk
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"🔴 GPU memory error at frames {start_idx}-{end_idx-1}")
                    
                    # Try to recover by clearing memory and retrying with smaller chunk
                    if original_device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    elif original_device.type == "cuda":
                        torch.cuda.empty_cache()
                    
                    # Retry with single frame processing
                    logger.warning(f"   Retrying frame-by-frame for this chunk...")
                    for frame_idx in range(start_idx, end_idx):
                        source_single = overlay_section[frame_idx:frame_idx+1]
                        overlay_single = resized_overlay[frame_idx:frame_idx+1]
                        
                        composite_single = self._composite_chunk(
                            source_single, overlay_single, pos_x, pos_y,
                            base_opacity, respect_alpha, overlay_has_alpha,
                            opacity_settings
                        )
                        result_chunks.append(composite_single.cpu())
                        del composite_single
                        
                        if original_device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                else:
                    raise  # Re-raise if not memory error
            
            # Aggressively clear cache after each chunk
            if original_device.type == "mps":
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                if hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
            elif original_device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Concatenate processed overlay section on CPU
        logger.info(f"🔗 Concatenating {len(result_chunks)} processed overlay chunks on CPU...")
        processed_overlay_section = torch.cat(result_chunks, dim=0)
        
        # Clear the chunks list to free memory
        del result_chunks
        
        # Combine all video segments: pre + overlay + post
        logger.info(f"🎬 Assembling final video from segments...")
        final_segments = []
        
        if pre_overlay_section is not None:
            final_segments.append(pre_overlay_section)
            logger.info(f"   Added pre-overlay section: {pre_overlay_section.shape[0]} frames")
        
        final_segments.append(processed_overlay_section)
        logger.info(f"   Added processed overlay section: {processed_overlay_section.shape[0]} frames")
        
        if post_overlay_section is not None:
            final_segments.append(post_overlay_section)
            logger.info(f"   Added post-overlay section: {post_overlay_section.shape[0]} frames")
        
        # Final concatenation on CPU
        composite_frames_cpu = torch.cat(final_segments, dim=0)
        logger.info(f"✓ Final video: {composite_frames_cpu.shape[0]} frames")
        
        # Clean up segments
        del final_segments
        if pre_overlay_section is not None:
            del pre_overlay_section
        del processed_overlay_section
        if post_overlay_section is not None:
            del post_overlay_section
        
        # For memory safety, keep final result on CPU and convert directly
        # This avoids the large GPU memory allocation for final result
        logger.info(f"   Converting final result on CPU to avoid GPU memory issues")
        
        # For MPS backend, check if we can safely transfer to GPU or should keep on CPU
        composite_frames_size_gb = (composite_frames_cpu.numel() * 4) / (1024**3)  # float32 = 4 bytes
        
        logger.info(f"   Final tensor size: {composite_frames_size_gb:.2f} GB")
        
        if original_device != torch.device("cpu"):
            # Apply more conservative memory management for MPS
            if original_device.type == "mps" and composite_frames_size_gb > 1.5:
                logger.warning(f"   Large tensor ({composite_frames_size_gb:.2f} GB) detected, keeping on CPU to prevent MPS OOM")
                composite_frames = composite_frames_cpu
            else:
                # Clear GPU memory aggressively before transfer
                if original_device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                elif original_device.type == "cuda":
                    torch.cuda.empty_cache()
                
                try:
                    logger.info(f"   Attempting transfer to {original_device} (size: {composite_frames_size_gb:.2f} GB)")
                    composite_frames = composite_frames_cpu.to(original_device)
                    del composite_frames_cpu
                    
                    # Clear cache immediately after successful transfer
                    if original_device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    elif original_device.type == "cuda":
                        torch.cuda.empty_cache()
                    
                    logger.info(f"   ✓ Final result transferred to {original_device}")
                    
                except RuntimeError as gpu_error:
                    if "out of memory" in str(gpu_error).lower():
                        logger.warning(f"   MPS/GPU OOM transferring {composite_frames_size_gb:.2f} GB tensor, keeping on CPU")
                        composite_frames = composite_frames_cpu
                    else:
                        raise gpu_error
        else:
            composite_frames = composite_frames_cpu
            logger.info(f"   ✓ Final result kept on CPU")
        
        # Create info string with time-based details
        alpha_info = f"Alpha: {overlay_has_alpha}" if respect_alpha else "Alpha: ignored"
        time_info = f"Time: {start_frame/fps:.2f}s-{end_frame/fps:.2f}s ({overlay_duration_frames} frames)"
        overlay_info = f"Video Overlay: {overlay_mode} mode | " \
                      f"Total: {composite_frames.shape[0]} frames | " \
                      f"Size: {source_width}x{source_height} | " \
                      f"Opacity: {base_opacity} | " \
                      f"{time_info} | " \
                      f"{alpha_info}"
        
        logger.info(f"✅ Time-based video overlay complete: {composite_frames.shape[0]} total frames, {overlay_duration_frames} overlay frames")
        logger.info(f"   Final tensor device: {composite_frames.device.type}")
        logger.info(f"   Output tensor shape: {composite_frames.shape}")
        
        return (
            composite_frames,
            overlay_info,
            composite_frames.shape[0],  # Total frame count
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
        Apply chroma key settings to frames with memory-efficient chunked processing
        """
        if not chroma_settings.get('enabled', False):
            return frames
        
        logger.info("   Applying chroma key with memory-efficient processing")
        
        target_color = chroma_settings['color']
        tolerance = chroma_settings['tolerance']
        edge_softness = chroma_settings.get('edge_softness', 0.05)
        
        # Get frame info
        num_frames = frames.shape[0]
        height = frames.shape[1]
        width = frames.shape[2]
        device = frames.device
        
        # Determine chunk size based on available memory
        # Smaller chunks for larger resolutions
        pixels_per_frame = height * width
        if pixels_per_frame > 2073600:  # > 1080p
            chroma_chunk_size = 3
        elif pixels_per_frame > 921600:  # > 720p
            chroma_chunk_size = 5
        else:
            chroma_chunk_size = 10
        
        logger.info(f"   Processing {num_frames} frames in chunks of {chroma_chunk_size}")
        
        # Process chroma key in chunks to avoid memory issues
        result_chunks = []
        
        for start_idx in range(0, num_frames, chroma_chunk_size):
            end_idx = min(start_idx + chroma_chunk_size, num_frames)
            chunk = frames[start_idx:end_idx]
            
            logger.info(f"     Chroma key chunk: frames {start_idx}-{end_idx-1}")
            
            try:
                # Memory-efficient color distance calculation
                # Use broadcasting instead of expand_as to save memory
                target_broadcast = target_color.view(1, 1, 1, 3)
                
                # Calculate difference (in-place operations where possible)
                diff = chunk - target_broadcast
                
                # Square in-place to save memory
                diff_squared = diff.pow(2)
                
                # Sum and sqrt for distance
                distances = torch.sqrt(diff_squared.sum(dim=-1))
                
                # Clean up intermediate tensors immediately
                del diff
                del diff_squared
                
                # Create alpha mask
                base_mask = 1.0 - torch.clamp(distances / tolerance, 0.0, 1.0)
                
                if edge_softness > 0.0:
                    soft_tolerance = tolerance + edge_softness
                    soft_mask = 1.0 - torch.clamp(distances / soft_tolerance, 0.0, 1.0)
                    edge_blend = torch.clamp((distances - tolerance) / edge_softness, 0.0, 1.0)
                    alpha_mask = base_mask * (1.0 - edge_blend) + soft_mask * edge_blend
                    del soft_mask
                    del edge_blend
                else:
                    alpha_mask = base_mask
                
                del base_mask
                del distances
                
                # Apply smooth step (Hermite interpolation)
                alpha_mask = alpha_mask * alpha_mask * (3.0 - 2.0 * alpha_mask)
                
                # Add alpha channel to chunk
                alpha_mask = alpha_mask.unsqueeze(-1)
                chunk_with_alpha = torch.cat([chunk, alpha_mask], dim=-1)
                
                # Move to CPU immediately to free GPU memory
                chunk_cpu = chunk_with_alpha.cpu()
                result_chunks.append(chunk_cpu)
                
                # Clean up
                del chunk_with_alpha
                del alpha_mask
                del chunk
                
                # Clear GPU cache after each chunk
                if device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"⚠️ Chroma key OOM at chunk {start_idx}-{end_idx-1}")
                    
                    # Fallback: process frame by frame
                    logger.info("   Falling back to frame-by-frame processing...")
                    for frame_idx in range(start_idx, end_idx):
                        single_frame = frames[frame_idx:frame_idx+1]
                        
                        # Simple chroma key for single frame
                        target_broadcast = target_color.view(1, 1, 1, 3)
                        diff = single_frame - target_broadcast
                        distances = torch.sqrt((diff * diff).sum(dim=-1))
                        
                        alpha = 1.0 - torch.clamp(distances / tolerance, 0.0, 1.0)
                        alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # Smooth step
                        alpha = alpha.unsqueeze(-1)
                        
                        frame_with_alpha = torch.cat([single_frame, alpha], dim=-1)
                        result_chunks.append(frame_with_alpha.cpu())
                        
                        del frame_with_alpha
                        del alpha
                        del distances
                        del diff
                        
                        if device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                else:
                    raise  # Re-raise if not memory error
        
        # Concatenate all chunks on CPU
        logger.info("   Concatenating chroma key results...")
        frames_with_alpha_cpu = torch.cat(result_chunks, dim=0)
        
        # Clear chunks
        del result_chunks
        
        # Move back to original device if needed
        if device.type != "cpu":
            frames_with_alpha = frames_with_alpha_cpu.to(device)
            del frames_with_alpha_cpu
            
            if device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        else:
            frames_with_alpha = frames_with_alpha_cpu
        
        logger.info(f"   ✓ Chroma key applied: {frames_with_alpha.shape}")
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