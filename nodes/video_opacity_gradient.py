import torch
import torch.nn.functional as F
import math
import numpy as np

try:
    from .utils import logger
except ImportError:
    # Fallback for testing without ComfyUI context
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

class RajVideoOpacityGradient:
    """
    Create advanced opacity gradient masks for video overlays
    Supports various gradient types from constant to complex multi-directional gradients
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Output width (ignored if keyframes provided)"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Output height (ignored if keyframes provided)"
                }),
                "gradient_type": (["constant", "linear", "radial", "corner", "edge", "multi_edge"], {
                    "default": "radial",
                    "tooltip": "Type of gradient pattern to create"
                }),
                "center_opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Opacity at center/focal point (1.0=opaque, 0.0=transparent)"
                }),
                "edge_opacity": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Opacity at edges/outer areas (1.0=opaque, 0.0=transparent)"
                }),
                "gradient_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Gradient curve strength (0.1=gentle, 10.0=sharp transition)"
                }),
                "easing_mode": (["linear", "ease_in", "ease_out", "ease_in_out"], {
                    "default": "linear",
                    "tooltip": "Gradient transition curve type"
                }),
                "gradient_direction": (["center_to_edge", "edge_to_center", "top_to_bottom", "bottom_to_top", 
                                       "left_to_right", "right_to_left", "corner_tl", "corner_tr", "corner_bl", "corner_br"], {
                    "default": "center_to_edge",
                    "tooltip": "Gradient flow direction (center_to_edge/edge_to_center for radial, others for linear/corner/edge)"
                }),
                "blend_mode": (["replace", "multiply", "overlay"], {
                    "default": "replace",
                    "tooltip": "How to combine with existing alpha channels"
                }),
            },
            "optional": {
                "keyframes": ("IMAGE", {
                    "tooltip": "Input video frames to apply opacity gradient (optional - leave empty for mask generation only)"
                }),
                "opacity_level": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Uniform opacity level (only used with 'constant' gradient type)"
                }),
                "center_x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Center X position (0.0=left, 1.0=right, for radial gradients)"
                }),
                "center_y": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Center Y position (0.0=top, 1.0=bottom, for radial gradients)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "OPACITY_GRADIENT", "STRING")
    RETURN_NAMES = ("opacity_result", "inverted_opacity", "opacity_settings", "gradient_info")
    FUNCTION = "create_opacity_gradient"
    CATEGORY = "Raj Video Processing 🎬"
    
    def create_opacity_gradient(self, width, height, gradient_type, center_opacity, edge_opacity, 
                               gradient_strength, easing_mode, gradient_direction, blend_mode, keyframes=None, 
                               opacity_level=0.5, center_x=0.5, center_y=0.5):
        
        # Determine processing mode
        process_keyframes = keyframes is not None
        
        if process_keyframes:
            # Get dimensions from keyframes
            batch_size, frame_height, frame_width, channels = keyframes.shape
            actual_width, actual_height = frame_width, frame_height
            keyframes_device = keyframes.device
            keyframes_dtype = keyframes.dtype
            
            logger.info(f"🌈 Processing {batch_size} keyframes with opacity gradient")
            logger.info(f"   Frame size: {actual_width}x{actual_height}, Channels: {channels}")
            logger.info(f"   Keyframes device: {keyframes_device}")
        else:
            # Use provided dimensions for mask generation
            actual_width, actual_height = width, height
            keyframes_device = torch.device("cpu")
            keyframes_dtype = torch.float32
            
            logger.info(f"🌈 Creating opacity gradient mask")
            logger.info(f"   Size: {actual_width}x{actual_height}")
        
        logger.info(f"   Type: {gradient_type}, Direction: {gradient_direction}")
        if gradient_type == "constant":
            logger.info(f"   Constant opacity: {opacity_level:.2f}")
        else:
            logger.info(f"   Center→Edge opacity: {center_opacity:.2f} → {edge_opacity:.2f}")
        logger.info(f"   Strength: {gradient_strength}, Easing: {easing_mode}, Blend: {blend_mode}")
        
        # Always create gradients on CPU to reduce memory pressure
        gradient_device = torch.device("cpu")
        gradient_dtype = torch.float32
        
        logger.info(f"   Gradient creation: CPU (memory optimization)")
        
        # Create normalized coordinate grids (0 to 1) on CPU
        y_coords = torch.linspace(0, 1, actual_height, device=gradient_device, dtype=gradient_dtype).unsqueeze(1).expand(actual_height, actual_width)
        x_coords = torch.linspace(0, 1, actual_width, device=gradient_device, dtype=gradient_dtype).unsqueeze(0).expand(actual_height, actual_width)
        
        # Generate gradient mask based on type
        if gradient_type == "constant":
            gradient_mask = self._create_constant_mask(actual_width, actual_height, opacity_level, gradient_device, gradient_dtype)
            
        elif gradient_type == "linear":
            gradient_mask = self._create_linear_gradient(
                x_coords, y_coords, gradient_direction, center_opacity, edge_opacity, gradient_strength, easing_mode
            )
            
        elif gradient_type == "radial":
            gradient_mask = self._create_radial_gradient(
                x_coords, y_coords, center_x, center_y, gradient_direction, center_opacity, edge_opacity, gradient_strength, easing_mode
            )
            
        elif gradient_type == "corner":
            gradient_mask = self._create_corner_gradient(
                x_coords, y_coords, gradient_direction, center_opacity, edge_opacity, gradient_strength, easing_mode
            )
            
        elif gradient_type == "edge":
            gradient_mask = self._create_edge_gradient(
                x_coords, y_coords, gradient_direction, center_opacity, edge_opacity, gradient_strength, easing_mode
            )
            
        elif gradient_type == "multi_edge":
            gradient_mask = self._create_multi_edge_gradient(
                x_coords, y_coords, center_opacity, edge_opacity, gradient_strength, easing_mode
            )
        
        # Clamp values to [0, 1]
        gradient_mask = torch.clamp(gradient_mask, 0.0, 1.0)
        
        # Create inverted gradient mask
        inverted_gradient_mask = 1.0 - gradient_mask
        
        if process_keyframes:
            # Process keyframes with gradient opacity using CPU-first hybrid strategy
            opacity_result, inverted_result = self._process_keyframes_hybrid(
                keyframes, gradient_mask, inverted_gradient_mask, blend_mode, 
                keyframes_device, keyframes_dtype
            )
            
            result_info = f"Processed {batch_size} frames"
            logger.info(f"✅ Processed {batch_size} keyframes with opacity gradient: {gradient_type} {actual_width}x{actual_height}")
        else:
            # Create mask-only output (backward compatibility)
            opacity_result = gradient_mask.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
            inverted_result = inverted_gradient_mask.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
            
            result_info = f"Size: {actual_width}x{actual_height}"
            logger.info(f"✅ Opacity gradient masks created: {gradient_type} {actual_width}x{actual_height}")
        
        # Create settings package for RajVideoOverlay
        opacity_settings = {
            'mask': gradient_mask,
            'inverted_mask': inverted_gradient_mask,
            'blend_mode': blend_mode,
            'type': gradient_type,
            'direction': gradient_direction,
            'strength': gradient_strength,
            'device': gradient_device,
            'has_keyframes': process_keyframes
        }
        
        # Create info string
        if gradient_type == "constant":
            opacity_info = f"Level: {opacity_level:.2f}"
        else:
            opacity_info = f"Center→Edge: {center_opacity:.2f}→{edge_opacity:.2f}"
        
        gradient_info = f"Opacity Gradient: {gradient_type} | " \
                       f"Direction: {gradient_direction} | " \
                       f"{opacity_info} | " \
                       f"Strength: {gradient_strength} | " \
                       f"Easing: {easing_mode} | " \
                       f"{result_info} | " \
                       f"Blend: {blend_mode}"
        
        return (
            opacity_result,
            inverted_result,
            opacity_settings,
            gradient_info
        )
    
    def _process_keyframes_hybrid(self, keyframes, gradient_mask, inverted_gradient_mask, blend_mode, target_device, target_dtype):
        """
        Apply gradient opacity using CPU-first hybrid strategy for memory efficiency
        Processes on CPU then transfers to target device to prevent MPS OOM
        """
        batch_size, height, width, channels = keyframes.shape
        
        # Determine processing strategy based on resolution and device
        processing_strategy = self._choose_processing_strategy(height, width, batch_size, target_device)
        
        logger.info(f"   Processing strategy: {processing_strategy}")
        
        # Determine if input has alpha channel
        has_alpha = channels == 4
        
        # Get optimal chunk size for this strategy
        chunk_size = self._calculate_hybrid_chunk_size(batch_size, height, width, channels, target_device, processing_strategy)
        
        logger.info(f"   Chunk size: {chunk_size} frames")
        
        if processing_strategy == "cpu_primary":
            return self._process_cpu_primary(keyframes, gradient_mask, inverted_gradient_mask, blend_mode, has_alpha, chunk_size, target_device, target_dtype)
        elif processing_strategy == "gpu_small_chunks":
            return self._process_gpu_small_chunks(keyframes, gradient_mask, inverted_gradient_mask, blend_mode, has_alpha, chunk_size)
        else:  # cpu_only
            return self._process_cpu_only(keyframes, gradient_mask, inverted_gradient_mask, blend_mode, has_alpha, chunk_size, target_device, target_dtype)
    
    def _apply_overlay_blend(self, base_alpha, gradient_alpha):
        """Apply overlay blend mode between two alpha channels"""
        # Overlay blend formula: if base < 0.5: 2 * base * overlay, else: 1 - 2 * (1-base) * (1-overlay)
        condition = base_alpha < 0.5
        result = torch.where(
            condition,
            2 * base_alpha * gradient_alpha,
            1 - 2 * (1 - base_alpha) * (1 - gradient_alpha)
        )
        return torch.clamp(result, 0.0, 1.0)
    
    def _apply_easing(self, progress, strength, easing_mode):
        """Apply easing curve to progress with specified strength and mode"""
        # Apply strength first (power function)
        if strength != 1.0:
            progress = torch.pow(torch.clamp(progress, 0.0, 1.0), strength)
        
        # Apply easing mode
        if easing_mode == "ease_in":
            # Slow start, fast end
            progress = progress * progress
        elif easing_mode == "ease_out":
            # Fast start, slow end  
            progress = 1.0 - (1.0 - progress) * (1.0 - progress)
        elif easing_mode == "ease_in_out":
            # Slow start and end, fast middle
            condition = progress < 0.5
            progress = torch.where(
                condition,
                2.0 * progress * progress,
                1.0 - 2.0 * (1.0 - progress) * (1.0 - progress)
            )
        # For "linear", no additional transformation needed
        
        return torch.clamp(progress, 0.0, 1.0)
    
    def _choose_processing_strategy(self, height, width, batch_size, target_device):
        """Choose optimal processing strategy based on resolution, batch size, and device"""
        pixel_count = height * width
        total_pixels = pixel_count * batch_size
        
        if target_device.type == "mps":
            # MPS has strict memory limits - prioritize CPU processing for large content
            if pixel_count >= 1280 * 720:  # >= 720p
                return "cpu_primary"
            elif total_pixels > 100_000_000:  # > 100M total pixels
                return "cpu_primary" 
            elif pixel_count >= 640 * 480:  # >= VGA
                return "cpu_only"  # Safer for medium resolutions
            else:
                return "gpu_small_chunks"
        elif target_device.type == "cuda":
            # CUDA generally has more memory - allow more GPU processing
            if pixel_count >= 1920 * 1080:  # >= 1080p
                return "cpu_primary"
            elif total_pixels > 200_000_000:  # > 200M total pixels
                return "cpu_primary"
            else:
                return "gpu_small_chunks"
        else:
            # CPU device - always use CPU processing
            return "cpu_only"
    
    def _calculate_hybrid_chunk_size(self, batch_size, height, width, channels, target_device, strategy):
        """Calculate chunk size based on strategy and hardware constraints"""
        pixel_count = height * width
        
        if strategy == "cpu_primary" or strategy == "cpu_only":
            # CPU processing can handle larger chunks
            if pixel_count >= 1280 * 720:  # >= 720p
                return max(1, min(4, batch_size))
            elif pixel_count >= 640 * 480:  # >= VGA  
                return max(1, min(8, batch_size))
            else:
                return max(1, min(16, batch_size))
        else:  # gpu_small_chunks
            # GPU processing needs smaller chunks for memory safety
            if target_device.type == "mps":
                if pixel_count >= 640 * 480:  # >= VGA
                    return max(1, min(2, batch_size))  # Very conservative for MPS
                else:
                    return max(1, min(4, batch_size))
            elif target_device.type == "cuda":
                if pixel_count >= 1280 * 720:  # >= 720p
                    return max(1, min(2, batch_size))
                else:
                    return max(1, min(6, batch_size))
            else:
                return max(1, min(8, batch_size))
    
    def _process_cpu_primary(self, keyframes, gradient_mask, inverted_gradient_mask, blend_mode, has_alpha, chunk_size, target_device, target_dtype):
        """Process primarily on CPU with minimal GPU transfers"""
        logger.info(f"   🖥️ CPU primary processing (memory-safe)")
        
        batch_size = keyframes.shape[0]
        
        # Move gradients to CPU if not already there
        if gradient_mask.device != torch.device("cpu"):
            gradient_mask = gradient_mask.to("cpu")
            inverted_gradient_mask = inverted_gradient_mask.to("cpu")
        
        processed_chunks = []
        inverted_chunks = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            
            # Move chunk to CPU for processing
            chunk_cpu = keyframes[i:end_idx].to("cpu", dtype=torch.float32)
            
            # Process on CPU
            processed_chunk, inverted_chunk = self._process_chunk_cpu(
                chunk_cpu, gradient_mask, inverted_gradient_mask, blend_mode, has_alpha
            )
            
            # Keep on CPU for now to build result
            processed_chunks.append(processed_chunk)
            inverted_chunks.append(inverted_chunk)
            
            # Cleanup intermediate tensors
            del chunk_cpu
            if torch.cuda.is_available() or hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Concatenate all chunks on CPU
        opacity_result_cpu = torch.cat(processed_chunks, dim=0)
        inverted_result_cpu = torch.cat(inverted_chunks, dim=0)
        
        # Clear chunk lists to free memory
        del processed_chunks, inverted_chunks
        
        # Convert to target device and dtype only at the end
        opacity_result = opacity_result_cpu.to(target_device, dtype=target_dtype)
        inverted_result = inverted_result_cpu.to(target_device, dtype=target_dtype)
        
        # Cleanup CPU tensors
        del opacity_result_cpu, inverted_result_cpu
        
        return opacity_result, inverted_result
    
    def _process_cpu_only(self, keyframes, gradient_mask, inverted_gradient_mask, blend_mode, has_alpha, chunk_size, target_device, target_dtype):
        """Process entirely on CPU for maximum memory safety"""
        logger.info(f"   🖥️ CPU-only processing (maximum safety)")
        
        # Ensure everything is on CPU
        keyframes_cpu = keyframes.to("cpu", dtype=torch.float32)
        if gradient_mask.device != torch.device("cpu"):
            gradient_mask = gradient_mask.to("cpu")
            inverted_gradient_mask = inverted_gradient_mask.to("cpu")
        
        batch_size = keyframes_cpu.shape[0]
        processed_chunks = []
        inverted_chunks = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk_cpu = keyframes_cpu[i:end_idx]
            
            processed_chunk, inverted_chunk = self._process_chunk_cpu(
                chunk_cpu, gradient_mask, inverted_gradient_mask, blend_mode, has_alpha
            )
            
            processed_chunks.append(processed_chunk)
            inverted_chunks.append(inverted_chunk)
        
        # Concatenate on CPU
        opacity_result = torch.cat(processed_chunks, dim=0)
        inverted_result = torch.cat(inverted_chunks, dim=0)
        
        # Convert to target device if needed
        if target_device != torch.device("cpu"):
            opacity_result = opacity_result.to(target_device, dtype=target_dtype)
            inverted_result = inverted_result.to(target_device, dtype=target_dtype)
        
        return opacity_result, inverted_result
    
    def _process_gpu_small_chunks(self, keyframes, gradient_mask, inverted_gradient_mask, blend_mode, has_alpha, chunk_size):
        """Process small chunks on GPU with aggressive memory management"""
        logger.info(f"   🎮 GPU small-chunk processing")
        
        batch_size = keyframes.shape[0]
        target_device = keyframes.device
        
        # Move gradients to target device
        if gradient_mask.device != target_device:
            gradient_mask = gradient_mask.to(target_device)
            inverted_gradient_mask = inverted_gradient_mask.to(target_device)
        
        processed_chunks = []
        inverted_chunks = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk = keyframes[i:end_idx]
            
            # Process chunk on GPU
            processed_chunk, inverted_chunk = self._process_chunk_gpu(
                chunk, gradient_mask, inverted_gradient_mask, blend_mode, has_alpha
            )
            
            processed_chunks.append(processed_chunk)
            inverted_chunks.append(inverted_chunk)
            
            # Aggressive memory cleanup after each chunk
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final concatenation on GPU
        opacity_result = torch.cat(processed_chunks, dim=0)
        inverted_result = torch.cat(inverted_chunks, dim=0)
        
        return opacity_result, inverted_result
    
    def _process_chunk_cpu(self, chunk, gradient_mask, inverted_gradient_mask, blend_mode, has_alpha):
        """Process a chunk on CPU with memory-efficient operations"""
        batch_size, height, width, channels = chunk.shape
        
        # Handle alpha channel creation/separation
        if has_alpha:
            rgb_chunk = chunk[..., :3]
            alpha_chunk = chunk[..., 3:4]
        else:
            rgb_chunk = chunk
            alpha_chunk = torch.ones(batch_size, height, width, 1, dtype=chunk.dtype)
        
        # Expand gradients efficiently
        gradient_expanded = gradient_mask.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 1)
        inverted_gradient_expanded = inverted_gradient_mask.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 1)
        
        # Apply blend mode
        if blend_mode == "replace":
            new_alpha = gradient_expanded.clone()
            inverted_new_alpha = inverted_gradient_expanded.clone()
        elif blend_mode == "multiply":
            new_alpha = alpha_chunk * gradient_expanded
            inverted_new_alpha = alpha_chunk * inverted_gradient_expanded
        elif blend_mode == "overlay":
            new_alpha = self._apply_overlay_blend(alpha_chunk, gradient_expanded)
            inverted_new_alpha = self._apply_overlay_blend(alpha_chunk, inverted_gradient_expanded)
        else:
            new_alpha = alpha_chunk * gradient_expanded
            inverted_new_alpha = alpha_chunk * inverted_gradient_expanded
        
        # Combine RGB with alpha
        processed_chunk = torch.cat([rgb_chunk, new_alpha], dim=-1)
        inverted_chunk = torch.cat([rgb_chunk, inverted_new_alpha], dim=-1)
        
        return processed_chunk, inverted_chunk
    
    def _process_chunk_gpu(self, chunk, gradient_mask, inverted_gradient_mask, blend_mode, has_alpha):
        """Process a chunk on GPU with memory monitoring"""
        batch_size, height, width, channels = chunk.shape
        
        # Handle alpha channel
        if has_alpha:
            rgb_chunk = chunk[..., :3]
            alpha_chunk = chunk[..., 3:4]
        else:
            rgb_chunk = chunk
            alpha_chunk = torch.ones(batch_size, height, width, 1, device=chunk.device, dtype=chunk.dtype)
        
        # Expand gradients
        gradient_expanded = gradient_mask.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 1)
        inverted_gradient_expanded = inverted_gradient_mask.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, 1)
        
        # Apply blend mode
        if blend_mode == "replace":
            new_alpha = gradient_expanded
            inverted_new_alpha = inverted_gradient_expanded
        elif blend_mode == "multiply":
            new_alpha = alpha_chunk * gradient_expanded
            inverted_new_alpha = alpha_chunk * inverted_gradient_expanded
        elif blend_mode == "overlay":
            new_alpha = self._apply_overlay_blend(alpha_chunk, gradient_expanded)
            inverted_new_alpha = self._apply_overlay_blend(alpha_chunk, inverted_gradient_expanded)
        else:
            new_alpha = alpha_chunk * gradient_expanded
            inverted_new_alpha = alpha_chunk * inverted_gradient_expanded
        
        # Combine RGB with alpha
        processed_chunk = torch.cat([rgb_chunk, new_alpha], dim=-1)
        inverted_chunk = torch.cat([rgb_chunk, inverted_new_alpha], dim=-1)
        
        return processed_chunk, inverted_chunk
    
    def _create_constant_mask(self, width, height, opacity, device, dtype):
        """Create uniform opacity mask"""
        return torch.full((height, width), opacity, device=device, dtype=dtype)
    
    def _create_linear_gradient(self, x_coords, y_coords, direction, center_opacity, edge_opacity, strength, easing_mode):
        """Create linear gradient in specified direction (no radial directions)"""
        
        # Linear gradients only support directional flows
        if direction == "top_to_bottom":
            progress = y_coords  # 0 at top, 1 at bottom
        elif direction == "bottom_to_top":
            progress = 1.0 - y_coords  # 1 at top, 0 at bottom
        elif direction == "left_to_right":
            progress = x_coords  # 0 at left, 1 at right
        elif direction == "right_to_left":
            progress = 1.0 - x_coords  # 1 at left, 0 at right
        else:
            # For radial directions, default to top_to_bottom
            progress = y_coords
            logger.warning(f"Linear gradient doesn't support '{direction}', using 'top_to_bottom'")
        
        # Apply easing with strength
        progress = self._apply_easing(progress, strength, easing_mode)
        
        # Interpolate opacity (center_opacity at start, edge_opacity at end)
        gradient = center_opacity + (edge_opacity - center_opacity) * progress
        
        return gradient
    
    def _create_radial_gradient(self, x_coords, y_coords, center_x, center_y, direction, center_opacity, edge_opacity, strength, easing_mode):
        """Create true radial gradient with proper direction support"""
        
        # Calculate distance from center point
        dx = x_coords - center_x
        dy = y_coords - center_y
        distance = torch.sqrt(dx * dx + dy * dy)
        
        # Normalize distance to 0-1 range (max distance is to furthest corner)
        max_distance = math.sqrt(max(center_x, 1-center_x)**2 + max(center_y, 1-center_y)**2)
        normalized_distance = distance / max_distance
        normalized_distance = torch.clamp(normalized_distance, 0.0, 1.0)
        
        # Apply direction logic
        if direction == "center_to_edge":
            # Center gets center_opacity, edges get edge_opacity
            progress = normalized_distance
        elif direction == "edge_to_center":
            # Edges get center_opacity, center gets edge_opacity (inverted)
            progress = 1.0 - normalized_distance
        else:
            # Default to center_to_edge for non-radial directions
            progress = normalized_distance
            logger.warning(f"Radial gradient doesn't support '{direction}', using 'center_to_edge'")
        
        # Apply easing with strength
        progress = self._apply_easing(progress, strength, easing_mode)
        
        # Interpolate opacity
        gradient = center_opacity + (edge_opacity - center_opacity) * progress
        
        return gradient
    
    def _create_corner_gradient(self, x_coords, y_coords, direction, center_opacity, edge_opacity, strength, easing_mode):
        """Create gradient emanating from specific corner"""
        
        # Calculate distance from specified corner
        if direction == "corner_tl":  # Top-left corner
            distance = torch.sqrt(x_coords**2 + y_coords**2)
        elif direction == "corner_tr":  # Top-right corner
            distance = torch.sqrt((1-x_coords)**2 + y_coords**2)
        elif direction == "corner_bl":  # Bottom-left corner
            distance = torch.sqrt(x_coords**2 + (1-y_coords)**2)
        elif direction == "corner_br":  # Bottom-right corner
            distance = torch.sqrt((1-x_coords)**2 + (1-y_coords)**2)
        else:
            # Default to top-left for non-corner directions
            distance = torch.sqrt(x_coords**2 + y_coords**2)
            logger.warning(f"Corner gradient doesn't support '{direction}', using 'corner_tl'")
        
        # Normalize to diagonal distance (corner to opposite corner)
        max_distance = math.sqrt(2.0)
        progress = distance / max_distance
        progress = torch.clamp(progress, 0.0, 1.0)
        
        # Apply easing with strength
        progress = self._apply_easing(progress, strength, easing_mode)
        
        # Interpolate opacity (corner gets center_opacity, far areas get edge_opacity)
        gradient = center_opacity + (edge_opacity - center_opacity) * progress
        
        return gradient
    
    def _create_edge_gradient(self, x_coords, y_coords, direction, center_opacity, edge_opacity, strength, easing_mode):
        """Create gradient flowing from specific edge"""
        
        # Calculate distance from specified edge
        if direction == "top_to_bottom":
            progress = y_coords  # 0 at top edge, 1 at bottom
        elif direction == "bottom_to_top":
            progress = 1.0 - y_coords  # 1 at top, 0 at bottom edge
        elif direction == "left_to_right":
            progress = x_coords  # 0 at left edge, 1 at right
        elif direction == "right_to_left":
            progress = 1.0 - x_coords  # 1 at left, 0 at right edge
        else:
            # Default to top edge for unsupported directions
            progress = y_coords
            logger.warning(f"Edge gradient doesn't support '{direction}', using 'top_to_bottom'")
        
        # Apply easing with strength
        progress = self._apply_easing(progress, strength, easing_mode)
        
        # Interpolate opacity (edge gets center_opacity, far side gets edge_opacity)
        gradient = center_opacity + (edge_opacity - center_opacity) * progress
        
        return gradient
    
    def _create_multi_edge_gradient(self, x_coords, y_coords, center_opacity, edge_opacity, strength, easing_mode):
        """Create gradient that flows from all edges toward center (vignette effect)"""
        
        # Calculate minimum distance from any edge
        dist_from_edges = torch.min(
            torch.min(x_coords, 1.0 - x_coords),  # Distance from left/right edges
            torch.min(y_coords, 1.0 - y_coords)   # Distance from top/bottom edges
        )
        
        # Normalize (max distance from edge to center is 0.5)
        progress = dist_from_edges / 0.5
        progress = torch.clamp(progress, 0.0, 1.0)
        
        # Apply easing with strength
        progress = self._apply_easing(progress, strength, easing_mode)
        
        # Interpolate opacity (edges get edge_opacity, center gets center_opacity)
        # Note: For multi-edge, we swap the logic - edges start with edge_opacity, center gets center_opacity
        gradient = edge_opacity + (center_opacity - edge_opacity) * progress
        
        return gradient
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always regenerate gradients when parameters change
        return float("nan")