import torch
import torch.nn.functional as F
import numpy as np
from .utils import tensor_to_video_frames, logger

class RajVideoChromaKey:
    """
    Professional chroma key (green screen) background removal with alpha channel output
    Removes specific colored backgrounds and creates high-quality alpha channels
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Input video frames (RGB)"
                }),
                "chroma_color_r": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "tooltip": "Red component of chroma key color (0-255)"
                }),
                "chroma_color_g": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "tooltip": "Green component of chroma key color (0-255)"
                }),
                "chroma_color_b": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "tooltip": "Blue component of chroma key color (0-255)"
                }),
                "tolerance": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Color matching tolerance (0.0-1.0)"
                }),
                "edge_softness": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Edge softening amount for natural blending"
                }),
                "spill_suppression": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Color spill reduction on edges"
                }),
                "enable_chroma": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable/disable chroma key effect"
                }),
            },
            "optional": {
                "output_format": (["rgba", "rgb_with_alpha", "alpha_only"], {
                    "default": "rgba",
                    "tooltip": "Output format type"
                }),
                "quality_mode": (["fast", "high"], {
                    "default": "high",
                    "tooltip": "Processing quality vs speed trade-off"
                }),
                "despill_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Strength of color spill removal"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "CHROMA_KEY", "STRING", "INT")
    RETURN_NAMES = ("keyed_frames", "alpha_mask", "chroma_settings", "chroma_info", "frame_count")
    FUNCTION = "chroma_key_removal"
    CATEGORY = "Raj Video Processing 🎬"
    
    def chroma_key_removal(self, frames, chroma_color_r, chroma_color_g, chroma_color_b, 
                          tolerance, edge_softness, spill_suppression, enable_chroma,
                          output_format="rgba", quality_mode="high", despill_strength=0.5):
        
        frame_count = frames.shape[0]
        height = frames.shape[1]
        width = frames.shape[2]
        channels = frames.shape[3]
        
        logger.info(f"🎬 Chroma Key Processing")
        logger.info(f"   Input: {frame_count} frames, {width}x{height}, {channels} channels")
        logger.info(f"   Chroma color: RGB({chroma_color_r}, {chroma_color_g}, {chroma_color_b})")
        logger.info(f"   Tolerance: {tolerance}, Edge softness: {edge_softness}")
        logger.info(f"   Spill suppression: {spill_suppression}, Quality: {quality_mode}")
        logger.info(f"   Enabled: {enable_chroma}")
        
        if not enable_chroma:
            # Return original frames with full alpha
            logger.info("   Chroma key disabled, returning original frames with full alpha")
            alpha_mask = torch.ones(frame_count, height, width, 1, device=frames.device, dtype=frames.dtype)
            keyed_frames = self._ensure_rgba_format(frames, alpha_mask)
            
            chroma_settings = {
                'enabled': False,
                'device': frames.device
            }
            
            chroma_info = f"Chroma Key: Disabled | Frames: {frame_count} | Size: {width}x{height}"
            return (keyed_frames, alpha_mask, chroma_settings, chroma_info, frame_count)
        
        # Convert chroma color to normalized values [0,1]
        target_color = torch.tensor([
            chroma_color_r / 255.0,
            chroma_color_g / 255.0,
            chroma_color_b / 255.0
        ], device=frames.device, dtype=frames.dtype)
        
        logger.info(f"   Normalized chroma color: [{target_color[0]:.3f}, {target_color[1]:.3f}, {target_color[2]:.3f}]")
        
        # Process in chunks for memory efficiency
        chunk_size = 10  # Process 10 frames at a time
        alpha_chunks = []
        keyed_chunks = []
        
        for start_idx in range(0, frame_count, chunk_size):
            end_idx = min(start_idx + chunk_size, frame_count)
            frame_chunk = frames[start_idx:end_idx]
            
            logger.info(f"   Processing frames {start_idx}-{end_idx-1}")
            
            # Generate alpha mask for this chunk
            alpha_chunk = self._generate_alpha_mask(
                frame_chunk, target_color, tolerance, edge_softness, quality_mode
            )
            
            # Apply spill suppression if enabled
            processed_chunk = frame_chunk
            if spill_suppression > 0.0:
                processed_chunk = self._suppress_color_spill(
                    frame_chunk, target_color, alpha_chunk, spill_suppression, despill_strength
                )
            
            # Create keyed frames (RGBA format)
            keyed_chunk = self._create_keyed_frames(processed_chunk, alpha_chunk, output_format)
            
            alpha_chunks.append(alpha_chunk)
            keyed_chunks.append(keyed_chunk)
            
            # Clear cache
            if frames.device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            elif frames.device.type == "cuda":
                torch.cuda.empty_cache()
        
        # Concatenate results
        logger.info(f"🔗 Concatenating {len(alpha_chunks)} processed chunks...")
        final_alpha = torch.cat(alpha_chunks, dim=0)
        final_keyed = torch.cat(keyed_chunks, dim=0)
        
        # Ensure ComfyUI format
        final_alpha_comfy = tensor_to_video_frames(final_alpha)
        final_keyed_comfy = tensor_to_video_frames(final_keyed)
        
        # Create chroma settings for overlay node
        chroma_settings = {
            'enabled': True,
            'color': target_color,
            'tolerance': tolerance,
            'edge_softness': edge_softness,
            'spill_suppression': spill_suppression,
            'device': frames.device
        }
        
        # Create info string
        chroma_info = f"Chroma Key: RGB({chroma_color_r},{chroma_color_g},{chroma_color_b}) | " \
                     f"Tolerance: {tolerance} | Edge: {edge_softness} | " \
                     f"Spill: {spill_suppression} | " \
                     f"Frames: {frame_count} | Size: {width}x{height}"
        
        logger.info(f"✅ Chroma key complete: {frame_count} frames with alpha")
        
        return (
            final_keyed_comfy,
            final_alpha_comfy,
            chroma_settings,
            chroma_info,
            frame_count
        )
    
    def _generate_alpha_mask(self, frames, target_color, tolerance, edge_softness, quality_mode):
        """
        Generate alpha mask by detecting chroma key color
        """
        # Expand target color to match frame dimensions
        target_expanded = target_color.view(1, 1, 1, 3).expand_as(frames)
        
        # Calculate color distance using Euclidean distance in RGB space
        diff = frames - target_expanded
        distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))  # Shape: [N, H, W]
        
        if quality_mode == "high":
            # High quality: smooth falloff with edge softening
            # Create base mask
            base_mask = 1.0 - torch.clamp(distances / tolerance, 0.0, 1.0)
            
            # Apply edge softening using Gaussian-like falloff
            if edge_softness > 0.0:
                # Create soft edge around the base mask
                soft_tolerance = tolerance + edge_softness
                soft_mask = 1.0 - torch.clamp(distances / soft_tolerance, 0.0, 1.0)
                
                # Blend between base and soft mask for smooth edges
                edge_blend = torch.clamp((distances - tolerance) / edge_softness, 0.0, 1.0)
                alpha_mask = base_mask * (1.0 - edge_blend) + soft_mask * edge_blend
            else:
                alpha_mask = base_mask
            
        else:  # fast mode
            # Simple threshold-based mask
            alpha_mask = (distances > tolerance).float()
        
        # Apply smooth step function for better quality
        alpha_mask = self._smooth_step(alpha_mask)
        
        # Add channel dimension: [N, H, W] -> [N, H, W, 1]
        alpha_mask = alpha_mask.unsqueeze(-1)
        
        return alpha_mask
    
    def _suppress_color_spill(self, frames, target_color, alpha_mask, spill_suppression, despill_strength):
        """
        Reduce color spill on semi-transparent edges
        """
        if spill_suppression == 0.0:
            return frames
        
        # Calculate spill amount based on similarity to chroma color
        target_expanded = target_color.view(1, 1, 1, 3).expand_as(frames)
        
        # Use dot product to measure color similarity (different approach from distance)
        similarity = torch.sum(frames * target_expanded, dim=-1, keepdim=True)
        spill_factor = torch.clamp(similarity * despill_strength, 0.0, 1.0)
        
        # Apply spill suppression more strongly on semi-transparent areas
        alpha_influence = 1.0 - alpha_mask.squeeze(-1).unsqueeze(-1)  # Invert alpha for spill areas
        final_spill_factor = spill_factor * alpha_influence * spill_suppression
        
        # Reduce the chroma color component
        spill_reduction = target_expanded * final_spill_factor
        despilled_frames = frames - spill_reduction
        
        # Ensure values stay in valid range
        despilled_frames = torch.clamp(despilled_frames, 0.0, 1.0)
        
        return despilled_frames
    
    def _create_keyed_frames(self, frames, alpha_mask, output_format):
        """
        Create final keyed frames in specified format
        """
        if output_format == "alpha_only":
            # Return only the alpha mask as grayscale
            return alpha_mask
        elif output_format == "rgb_with_alpha":
            # Return RGB and alpha as separate channels (still RGBA but for different use)
            return torch.cat([frames, alpha_mask], dim=-1)
        else:  # rgba (default)
            # Standard RGBA output
            return torch.cat([frames, alpha_mask], dim=-1)
    
    def _ensure_rgba_format(self, frames, alpha_mask):
        """
        Ensure frames are in RGBA format by adding alpha channel
        """
        if frames.shape[-1] == 4:
            return frames  # Already RGBA
        else:
            return torch.cat([frames, alpha_mask], dim=-1)
    
    def _smooth_step(self, x):
        """
        Apply smooth step function for better alpha quality
        Smoothstep: 3x² - 2x³
        """
        x = torch.clamp(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always recompute when parameters change
        return float("nan")