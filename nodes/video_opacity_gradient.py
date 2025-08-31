import torch
import torch.nn.functional as F
import math
import numpy as np
from .utils import logger

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
                    "tooltip": "Output width (should match overlay dimensions)"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Output height (should match overlay dimensions)"
                }),
                "gradient_type": (["constant", "linear", "radial", "corner", "edge", "multi_edge"], {
                    "default": "constant",
                    "tooltip": "Type of gradient to create"
                }),
                "opacity_start": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Starting opacity value"
                }),
                "opacity_end": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Ending opacity value"
                }),
                "gradient_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Gradient transition sharpness (higher = sharper)"
                }),
                "gradient_direction": (["top_to_bottom", "bottom_to_top", "left_to_right", "right_to_left", 
                                       "center_out", "center_in", "corner_tl", "corner_tr", "corner_bl", "corner_br"], {
                    "default": "top_to_bottom",
                    "tooltip": "Gradient direction or focal point"
                }),
                "blend_mode": (["replace", "multiply", "overlay"], {
                    "default": "replace",
                    "tooltip": "How to combine with existing alpha channels"
                }),
            },
            "optional": {
                "center_x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Center X position (0.0-1.0, for radial gradients)"
                }),
                "center_y": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Center Y position (0.0-1.0, for radial gradients)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "OPACITY_GRADIENT", "STRING")
    RETURN_NAMES = ("opacity_mask", "opacity_settings", "gradient_info")
    FUNCTION = "create_opacity_gradient"
    CATEGORY = "Raj Video Processing 🎬"
    
    def create_opacity_gradient(self, width, height, gradient_type, opacity_start, opacity_end, 
                               gradient_strength, gradient_direction, blend_mode, center_x=0.5, center_y=0.5):
        
        logger.info(f"🌈 Creating opacity gradient")
        logger.info(f"   Size: {width}x{height}")
        logger.info(f"   Type: {gradient_type}, Direction: {gradient_direction}")
        logger.info(f"   Opacity: {opacity_start:.2f} → {opacity_end:.2f}")
        logger.info(f"   Strength: {gradient_strength}, Blend: {blend_mode}")
        
        # Create coordinate grids
        device = torch.device("cpu")  # Default to CPU, will be moved to appropriate device when used
        dtype = torch.float32
        
        # Create normalized coordinate grids (0 to 1)
        y_coords = torch.linspace(0, 1, height, device=device, dtype=dtype).unsqueeze(1).expand(height, width)
        x_coords = torch.linspace(0, 1, width, device=device, dtype=dtype).unsqueeze(0).expand(height, width)
        
        # Generate gradient mask based on type
        if gradient_type == "constant":
            gradient_mask = self._create_constant_mask(width, height, opacity_start, device, dtype)
            
        elif gradient_type == "linear":
            gradient_mask = self._create_linear_gradient(
                x_coords, y_coords, gradient_direction, opacity_start, opacity_end, gradient_strength
            )
            
        elif gradient_type == "radial":
            gradient_mask = self._create_radial_gradient(
                x_coords, y_coords, center_x, center_y, opacity_start, opacity_end, gradient_strength
            )
            
        elif gradient_type == "corner":
            gradient_mask = self._create_corner_gradient(
                x_coords, y_coords, gradient_direction, opacity_start, opacity_end, gradient_strength
            )
            
        elif gradient_type == "edge":
            gradient_mask = self._create_edge_gradient(
                x_coords, y_coords, gradient_direction, opacity_start, opacity_end, gradient_strength
            )
            
        elif gradient_type == "multi_edge":
            gradient_mask = self._create_multi_edge_gradient(
                x_coords, y_coords, opacity_start, opacity_end, gradient_strength
            )
        
        # Clamp values to [0, 1]
        gradient_mask = torch.clamp(gradient_mask, 0.0, 1.0)
        
        # Convert to image format [1, H, W, 1] (single frame, grayscale)
        opacity_image = gradient_mask.unsqueeze(0).unsqueeze(-1)
        
        # Create settings package for RajVideoOverlay
        opacity_settings = {
            'mask': gradient_mask,
            'blend_mode': blend_mode,
            'type': gradient_type,
            'direction': gradient_direction,
            'strength': gradient_strength,
            'device': device
        }
        
        # Create info string
        gradient_info = f"Opacity Gradient: {gradient_type} | " \
                       f"Direction: {gradient_direction} | " \
                       f"Range: {opacity_start:.2f}→{opacity_end:.2f} | " \
                       f"Size: {width}x{height} | " \
                       f"Blend: {blend_mode}"
        
        logger.info(f"✅ Opacity gradient created: {gradient_type} {width}x{height}")
        
        return (
            opacity_image,
            opacity_settings,
            gradient_info
        )
    
    def _create_constant_mask(self, width, height, opacity, device, dtype):
        """Create uniform opacity mask"""
        return torch.full((height, width), opacity, device=device, dtype=dtype)
    
    def _create_linear_gradient(self, x_coords, y_coords, direction, start_opacity, end_opacity, strength):
        """Create linear gradient in specified direction"""
        
        if direction == "top_to_bottom":
            progress = y_coords
        elif direction == "bottom_to_top":
            progress = 1.0 - y_coords
        elif direction == "left_to_right":
            progress = x_coords
        elif direction == "right_to_left":
            progress = 1.0 - x_coords
        elif direction == "center_out":
            # Distance from center (0.5, 0.5)
            center_x, center_y = 0.5, 0.5
            dist_x = torch.abs(x_coords - center_x) * 2  # Scale to 0-1 range
            dist_y = torch.abs(y_coords - center_y) * 2
            progress = torch.max(dist_x, dist_y)  # Max distance (diamond shape)
        elif direction == "center_in":
            # Inverse of center_out
            center_x, center_y = 0.5, 0.5
            dist_x = torch.abs(x_coords - center_x) * 2
            dist_y = torch.abs(y_coords - center_y) * 2
            progress = 1.0 - torch.max(dist_x, dist_y)
        else:
            progress = y_coords  # Default fallback
        
        # Apply easing based on strength
        if strength != 1.0:
            progress = torch.pow(progress, strength)
        
        # Interpolate between start and end opacity
        gradient = start_opacity + (end_opacity - start_opacity) * progress
        
        return gradient
    
    def _create_radial_gradient(self, x_coords, y_coords, center_x, center_y, start_opacity, end_opacity, strength):
        """Create radial gradient from specified center point"""
        
        # Calculate distance from center
        dx = x_coords - center_x
        dy = y_coords - center_y
        distance = torch.sqrt(dx * dx + dy * dy)
        
        # Normalize distance to 0-1 range (max distance is to corner)
        max_distance = math.sqrt(max(center_x, 1-center_x)**2 + max(center_y, 1-center_y)**2)
        progress = distance / max_distance
        progress = torch.clamp(progress, 0.0, 1.0)
        
        # Apply easing
        if strength != 1.0:
            progress = torch.pow(progress, strength)
        
        # Interpolate opacity
        gradient = start_opacity + (end_opacity - start_opacity) * progress
        
        return gradient
    
    def _create_corner_gradient(self, x_coords, y_coords, direction, start_opacity, end_opacity, strength):
        """Create gradient from specific corner"""
        
        if direction == "corner_tl":  # Top-left
            distance = torch.sqrt(x_coords**2 + y_coords**2)
        elif direction == "corner_tr":  # Top-right
            distance = torch.sqrt((1-x_coords)**2 + y_coords**2)
        elif direction == "corner_bl":  # Bottom-left
            distance = torch.sqrt(x_coords**2 + (1-y_coords)**2)
        elif direction == "corner_br":  # Bottom-right
            distance = torch.sqrt((1-x_coords)**2 + (1-y_coords)**2)
        else:
            distance = torch.sqrt(x_coords**2 + y_coords**2)  # Default to TL
        
        # Normalize to diagonal distance
        max_distance = math.sqrt(2)  # Corner to opposite corner
        progress = distance / max_distance
        progress = torch.clamp(progress, 0.0, 1.0)
        
        # Apply easing
        if strength != 1.0:
            progress = torch.pow(progress, strength)
        
        # Interpolate opacity
        gradient = start_opacity + (end_opacity - start_opacity) * progress
        
        return gradient
    
    def _create_edge_gradient(self, x_coords, y_coords, direction, start_opacity, end_opacity, strength):
        """Create gradient from specific edge"""
        
        if direction.startswith("top"):
            progress = y_coords
        elif direction.startswith("bottom"):
            progress = 1.0 - y_coords
        elif direction.startswith("left"):
            progress = x_coords
        elif direction.startswith("right"):
            progress = 1.0 - x_coords
        else:
            progress = y_coords  # Default
        
        # Apply easing
        if strength != 1.0:
            progress = torch.pow(progress, strength)
        
        # Interpolate opacity
        gradient = start_opacity + (end_opacity - start_opacity) * progress
        
        return gradient
    
    def _create_multi_edge_gradient(self, x_coords, y_coords, start_opacity, end_opacity, strength):
        """Create gradient that fades from all edges toward center"""
        
        # Distance from edges (minimum distance to any edge)
        dist_from_edges = torch.min(
            torch.min(x_coords, 1.0 - x_coords),  # Distance from left/right edges
            torch.min(y_coords, 1.0 - y_coords)   # Distance from top/bottom edges
        )
        
        # Normalize (max distance from edge to center is 0.5)
        progress = dist_from_edges / 0.5
        progress = torch.clamp(progress, 0.0, 1.0)
        
        # Apply easing
        if strength != 1.0:
            progress = torch.pow(progress, strength)
        
        # Interpolate opacity (edges start at start_opacity, center at end_opacity)
        gradient = start_opacity + (end_opacity - start_opacity) * progress
        
        return gradient
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always regenerate gradients when parameters change
        return float("nan")