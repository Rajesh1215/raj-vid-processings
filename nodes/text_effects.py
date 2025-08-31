import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import json
import math
from typing import Dict, List, Tuple, Optional, Union
from .utils import logger

class RajTextEffects:
    """
    Advanced text styling node with opacity maps, shadows, borders, glows, and gradients.
    Works with output from RajTextGenerator to add professional effects.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_image": ("IMAGE", {
                    "tooltip": "Input text image from RajTextGenerator"
                }),
                "border_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable text border/stroke"
                }),
                "border_width": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Border width in pixels"
                }),
                "border_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Border color (HEX)"
                }),
                "border_style": (["solid", "dashed", "dotted", "double", "groove", "ridge", "inset", "outset", "gradient"], {
                    "default": "solid",
                    "tooltip": "Border style type"
                }),
                "shadow_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable drop shadow"
                }),
                "shadow_offset_x": ("INT", {
                    "default": 2,
                    "min": -50,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Shadow X offset"
                }),
                "shadow_offset_y": ("INT", {
                    "default": 2,
                    "min": -50,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Shadow Y offset"
                }),
                "shadow_blur": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Shadow blur radius"
                }),
                "shadow_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Shadow color with alpha"
                }),
                "shadow_opacity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Shadow opacity"
                }),
                "glow_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable glow effect"
                }),
                "glow_radius": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Glow radius"
                }),
                "glow_intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Glow intensity"
                }),
                "glow_color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Glow color"
                }),
                "opacity_mode": (["uniform", "gradient", "radial", "custom_map"], {
                    "default": "uniform",
                    "tooltip": "Opacity distribution mode"
                }),
                "opacity_gradient_type": (["linear", "radial", "angular", "diamond", "corner"], {
                    "default": "linear",
                    "tooltip": "Gradient type for opacity"
                }),
                "opacity_gradient_angle": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 1.0,
                    "tooltip": "Gradient angle (degrees)"
                }),
                "opacity_start": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Gradient start opacity"
                }),
                "opacity_end": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Gradient end opacity"
                }),
            },
            "optional": {
                "alpha_mask": ("MASK", {
                    "tooltip": "Custom opacity mask input"
                }),
                "opacity_map_settings": ("OPACITY_GRADIENT", {
                    "tooltip": "Opacity gradient settings from RajVideoOpacityGradient"
                }),
                "text_config": ("STRING", {
                    "default": "{}",
                    "tooltip": "Text configuration JSON from generator"
                }),
                "bevel_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable bevel and emboss effect"
                }),
                "bevel_depth": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Bevel depth"
                }),
                "bevel_angle": ("FLOAT", {
                    "default": 45.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 1.0,
                    "tooltip": "Bevel lighting angle"
                }),
                "texture_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable texture overlay"
                }),
                "texture_type": (["noise", "fabric", "metal", "wood", "paper"], {
                    "default": "noise",
                    "tooltip": "Texture type"
                }),
                "texture_intensity": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Texture intensity"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("styled_text", "opacity_mask", "text_config", "effect_info")
    FUNCTION = "apply_effects"
    CATEGORY = "Raj Video Processing 🎬/Text"
    
    @staticmethod
    def parse_color(color_str: str) -> Tuple[int, int, int, int]:
        """Parse color string to RGBA tuple."""
        if color_str.lower() == "transparent":
            return (0, 0, 0, 0)
        
        # Remove # if present
        if color_str.startswith("#"):
            color_str = color_str[1:]
        
        # Parse hex color
        if len(color_str) == 6:
            r = int(color_str[0:2], 16)
            g = int(color_str[2:4], 16)
            b = int(color_str[4:6], 16)
            return (r, g, b, 255)
        elif len(color_str) == 8:
            r = int(color_str[0:2], 16)
            g = int(color_str[2:4], 16)
            b = int(color_str[4:6], 16)
            a = int(color_str[6:8], 16)
            return (r, g, b, a)
        
        # Default to white
        return (255, 255, 255, 255)
    
    @staticmethod
    def create_gradient_mask(width: int, height: int, gradient_type: str, 
                           angle: float, start_opacity: float, end_opacity: float) -> Image.Image:
        """Create gradient opacity mask."""
        mask = Image.new('L', (width, height), 255)
        
        if gradient_type == "linear":
            # Create linear gradient
            angle_rad = math.radians(angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            # Calculate gradient direction
            center_x, center_y = width // 2, height // 2
            
            for y in range(height):
                for x in range(width):
                    # Distance along gradient direction
                    dx = x - center_x
                    dy = y - center_y
                    distance = dx * cos_a + dy * sin_a
                    
                    # Normalize to 0-1 range
                    max_distance = math.sqrt(width**2 + height**2) / 2
                    normalized = (distance + max_distance) / (2 * max_distance)
                    normalized = max(0, min(1, normalized))
                    
                    # Apply opacity gradient
                    opacity = start_opacity + (end_opacity - start_opacity) * normalized
                    mask.putpixel((x, y), int(opacity * 255))
        
        elif gradient_type == "radial":
            # Create radial gradient
            center_x, center_y = width // 2, height // 2
            max_distance = math.sqrt(center_x**2 + center_y**2)
            
            for y in range(height):
                for x in range(width):
                    distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                    normalized = distance / max_distance
                    normalized = max(0, min(1, normalized))
                    
                    opacity = start_opacity + (end_opacity - start_opacity) * normalized
                    mask.putpixel((x, y), int(opacity * 255))
        
        elif gradient_type == "angular":
            # Create angular gradient
            center_x, center_y = width // 2, height // 2
            start_angle = math.radians(angle)
            
            for y in range(height):
                for x in range(width):
                    dx = x - center_x
                    dy = y - center_y
                    
                    if dx == 0 and dy == 0:
                        pixel_angle = 0
                    else:
                        pixel_angle = math.atan2(dy, dx)
                    
                    # Normalize angle difference
                    angle_diff = abs(pixel_angle - start_angle)
                    angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                    normalized = angle_diff / math.pi
                    
                    opacity = start_opacity + (end_opacity - start_opacity) * normalized
                    mask.putpixel((x, y), int(opacity * 255))
        
        elif gradient_type == "diamond":
            # Create diamond gradient
            center_x, center_y = width // 2, height // 2
            
            for y in range(height):
                for x in range(width):
                    distance = abs(x - center_x) + abs(y - center_y)
                    max_distance = center_x + center_y
                    normalized = distance / max_distance
                    normalized = max(0, min(1, normalized))
                    
                    opacity = start_opacity + (end_opacity - start_opacity) * normalized
                    mask.putpixel((x, y), int(opacity * 255))
        
        elif gradient_type == "corner":
            # Create corner-to-center gradient
            for y in range(height):
                for x in range(width):
                    # Distance from corners
                    corners = [
                        math.sqrt(x**2 + y**2),  # Top-left
                        math.sqrt((width - x)**2 + y**2),  # Top-right
                        math.sqrt(x**2 + (height - y)**2),  # Bottom-left
                        math.sqrt((width - x)**2 + (height - y)**2)  # Bottom-right
                    ]
                    
                    distance = min(corners)
                    max_distance = math.sqrt(width**2 + height**2) / 2
                    normalized = distance / max_distance
                    normalized = max(0, min(1, normalized))
                    
                    opacity = start_opacity + (end_opacity - start_opacity) * normalized
                    mask.putpixel((x, y), int(opacity * 255))
        
        return mask
    
    @staticmethod
    def create_border_mask(width: int, height: int, text_mask: Image.Image,
                          border_width: int, border_style: str) -> Image.Image:
        """Create border mask from text mask."""
        # Create border by dilating the text mask
        border_mask = Image.new('L', (width, height), 0)
        
        # Simple dilation approach
        text_pixels = set()
        for y in range(height):
            for x in range(width):
                if text_mask.getpixel((x, y)) > 0:
                    text_pixels.add((x, y))
        
        # Create border pixels
        border_pixels = set()
        for x, y in text_pixels:
            for dx in range(-border_width, border_width + 1):
                for dy in range(-border_width, border_width + 1):
                    if 0 <= x + dx < width and 0 <= y + dy < height:
                        distance = math.sqrt(dx**2 + dy**2)
                        if distance <= border_width:
                            if border_style == "solid":
                                border_pixels.add((x + dx, y + dy))
                            elif border_style == "dashed":
                                if (x + dx + y + dy) % 8 < 4:
                                    border_pixels.add((x + dx, y + dy))
                            elif border_style == "dotted":
                                if (x + dx + y + dy) % 4 < 2:
                                    border_pixels.add((x + dx, y + dy))
        
        # Remove original text pixels to create just the border
        border_pixels -= text_pixels
        
        # Draw border
        for x, y in border_pixels:
            border_mask.putpixel((x, y), 255)
        
        return border_mask
    
    @staticmethod
    def create_texture(width: int, height: int, texture_type: str, intensity: float) -> Image.Image:
        """Create texture overlay."""
        texture = Image.new('L', (width, height), 128)
        
        if texture_type == "noise":
            # Random noise texture
            import random
            for y in range(height):
                for x in range(width):
                    noise = random.randint(0, 255)
                    base = 128 + int((noise - 128) * intensity)
                    texture.putpixel((x, y), max(0, min(255, base)))
        
        elif texture_type == "fabric":
            # Fabric-like weave pattern
            for y in range(height):
                for x in range(width):
                    pattern = (math.sin(x * 0.3) + math.sin(y * 0.3)) * 20 * intensity
                    value = 128 + int(pattern)
                    texture.putpixel((x, y), max(0, min(255, value)))
        
        elif texture_type == "metal":
            # Metallic brushed texture
            for y in range(height):
                for x in range(width):
                    pattern = math.sin(x * 0.1) * 30 * intensity
                    value = 128 + int(pattern)
                    texture.putpixel((x, y), max(0, min(255, value)))
        
        elif texture_type == "wood":
            # Wood grain pattern
            for y in range(height):
                for x in range(width):
                    grain = math.sin(y * 0.05) * math.sin(x * 0.02) * 40 * intensity
                    value = 128 + int(grain)
                    texture.putpixel((x, y), max(0, min(255, value)))
        
        elif texture_type == "paper":
            # Paper fiber texture
            import random
            for y in range(height):
                for x in range(width):
                    fiber = (random.random() - 0.5) * 20 * intensity
                    value = 128 + int(fiber)
                    texture.putpixel((x, y), max(0, min(255, value)))
        
        return texture
    
    def apply_effects(self, text_image, border_enabled, border_width, border_color,
                     border_style, shadow_enabled, shadow_offset_x, shadow_offset_y,
                     shadow_blur, shadow_color, shadow_opacity, glow_enabled,
                     glow_radius, glow_intensity, glow_color, opacity_mode,
                     opacity_gradient_type, opacity_gradient_angle, opacity_start,
                     opacity_end, alpha_mask=None, opacity_map_settings=None,
                     text_config="{}", bevel_enabled=False, bevel_depth=3,
                     bevel_angle=45.0, texture_enabled=False, texture_type="noise",
                     texture_intensity=0.3):
        
        # Convert tensor to PIL image
        batch_size, height, width, channels = text_image.shape
        if batch_size > 1:
            logger.warning("Multiple images in batch, using first image only")
        
        image_np = (text_image[0].cpu().numpy() * 255).astype(np.uint8)
        
        if channels == 3:
            # Add alpha channel
            alpha = np.ones((height, width), dtype=np.uint8) * 255
            image_np = np.dstack([image_np, alpha])
        
        input_image = Image.fromarray(image_np, 'RGBA')
        
        # Create working image with larger canvas for effects
        padding = max(shadow_blur + abs(shadow_offset_x), shadow_blur + abs(shadow_offset_y),
                     glow_radius, border_width) + 10
        canvas_width = width + 2 * padding
        canvas_height = height + 2 * padding
        
        canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
        
        # Parse colors
        border_rgba = self.parse_color(border_color)
        shadow_rgba = self.parse_color(shadow_color)
        shadow_rgba = (shadow_rgba[0], shadow_rgba[1], shadow_rgba[2], int(shadow_rgba[3] * shadow_opacity))
        glow_rgba = self.parse_color(glow_color)
        
        # Create text mask for effect generation
        text_alpha = input_image.split()[3]
        text_mask = text_alpha.copy()
        
        # Apply border effect
        if border_enabled:
            border_mask = self.create_border_mask(canvas_width, canvas_height, 
                                                text_mask.resize((canvas_width, canvas_height)), 
                                                border_width, border_style)
            border_layer = Image.new('RGBA', (canvas_width, canvas_height), border_rgba)
            border_layer.putalpha(border_mask)
            canvas = Image.alpha_composite(canvas, border_layer)
        
        # Apply shadow effect
        if shadow_enabled:
            shadow_layer = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
            shadow_text = Image.new('RGBA', (canvas_width, canvas_height), shadow_rgba)
            
            # Position shadow
            shadow_x = padding + shadow_offset_x
            shadow_y = padding + shadow_offset_y
            shadow_mask = text_mask.resize((width, height))
            shadow_layer.paste(shadow_text, (shadow_x, shadow_y), shadow_mask)
            
            # Apply blur
            if shadow_blur > 0:
                shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(shadow_blur))
            
            canvas = Image.alpha_composite(canvas, shadow_layer)
        
        # Apply glow effect
        if glow_enabled:
            glow_layer = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
            glow_text = Image.new('RGBA', (canvas_width, canvas_height), glow_rgba)
            
            # Create glow by multiple blurred copies
            for i in range(glow_radius):
                glow_copy = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
                glow_copy.paste(glow_text, (padding, padding), text_mask)
                glow_copy = glow_copy.filter(ImageFilter.GaussianBlur(i + 1))
                
                # Reduce opacity for outer glow layers
                opacity = int(glow_intensity * 255 / (i + 1))
                glow_array = np.array(glow_copy)
                glow_array[:, :, 3] = (glow_array[:, :, 3] * opacity // 255)
                glow_copy = Image.fromarray(glow_array, 'RGBA')
                
                glow_layer = Image.alpha_composite(glow_layer, glow_copy)
            
            canvas = Image.alpha_composite(canvas, glow_layer)
        
        # Apply bevel effect
        if bevel_enabled:
            # Create highlight and shadow for bevel
            angle_rad = math.radians(bevel_angle)
            highlight_x = int(math.cos(angle_rad) * bevel_depth)
            highlight_y = int(math.sin(angle_rad) * bevel_depth)
            shadow_x = -highlight_x
            shadow_y = -highlight_y
            
            # Highlight
            highlight_layer = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 100))
            highlight_layer.paste((0, 0, 0, 0), (padding + highlight_x, padding + highlight_y), text_mask)
            
            # Shadow
            bevel_shadow_layer = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 100))
            bevel_shadow_layer.paste((0, 0, 0, 0), (padding + shadow_x, padding + shadow_y), text_mask)
            
            canvas = Image.alpha_composite(canvas, highlight_layer)
            canvas = Image.alpha_composite(canvas, bevel_shadow_layer)
        
        # Paste original text on top
        canvas.paste(input_image, (padding, padding), input_image)
        
        # Apply texture if enabled
        if texture_enabled:
            texture = self.create_texture(canvas_width, canvas_height, texture_type, texture_intensity)
            texture_layer = Image.new('RGBA', (canvas_width, canvas_height), (128, 128, 128, 255))
            texture_layer.putalpha(texture)
            
            # Blend texture with text areas only
            text_areas = canvas.split()[3]
            texture_masked = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
            texture_masked.paste(texture_layer, (0, 0), text_areas)
            
            # Apply texture using multiply blend mode (simplified)
            canvas_array = np.array(canvas)
            texture_array = np.array(texture_masked)
            
            for c in range(3):  # RGB channels
                canvas_array[:, :, c] = (canvas_array[:, :, c] * texture_array[:, :, c] // 255)
            
            canvas = Image.fromarray(canvas_array, 'RGBA')
        
        # Apply opacity effects
        if opacity_mode != "uniform":
            if opacity_mode == "custom_map" and alpha_mask is not None:
                # Use provided alpha mask
                mask_np = alpha_mask[0].cpu().numpy() if isinstance(alpha_mask, torch.Tensor) else alpha_mask
                custom_mask = Image.fromarray((mask_np * 255).astype(np.uint8), 'L')
                custom_mask = custom_mask.resize((canvas_width, canvas_height))
                
                canvas_array = np.array(canvas)
                mask_array = np.array(custom_mask)
                canvas_array[:, :, 3] = (canvas_array[:, :, 3] * mask_array // 255)
                canvas = Image.fromarray(canvas_array, 'RGBA')
            
            else:
                # Create gradient mask
                gradient_mask = self.create_gradient_mask(
                    canvas_width, canvas_height, opacity_gradient_type,
                    opacity_gradient_angle, opacity_start, opacity_end
                )
                
                canvas_array = np.array(canvas)
                mask_array = np.array(gradient_mask)
                canvas_array[:, :, 3] = (canvas_array[:, :, 3] * mask_array // 255)
                canvas = Image.fromarray(canvas_array, 'RGBA')
        
        # Crop back to original size
        final_image = canvas.crop((padding, padding, padding + width, padding + height))
        
        # Create final opacity mask
        final_alpha = final_image.split()[3]
        
        # Convert back to tensor
        final_np = np.array(final_image).astype(np.float32) / 255.0
        final_tensor = torch.from_numpy(final_np).unsqueeze(0)
        
        # Create mask tensor
        mask_np = np.array(final_alpha).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        
        # Update text config
        try:
            config = json.loads(text_config) if text_config else {}
        except:
            config = {}
        
        config.update({
            "effects_applied": {
                "border": border_enabled,
                "shadow": shadow_enabled,
                "glow": glow_enabled,
                "bevel": bevel_enabled,
                "texture": texture_enabled,
                "opacity_mode": opacity_mode
            }
        })
        
        # Create effect info
        effects = []
        if border_enabled:
            effects.append(f"Border ({border_style}, {border_width}px)")
        if shadow_enabled:
            effects.append(f"Shadow ({shadow_offset_x},{shadow_offset_y}, blur: {shadow_blur})")
        if glow_enabled:
            effects.append(f"Glow (radius: {glow_radius}, intensity: {glow_intensity})")
        if bevel_enabled:
            effects.append(f"Bevel (depth: {bevel_depth}, angle: {bevel_angle}°)")
        if texture_enabled:
            effects.append(f"Texture ({texture_type}, {texture_intensity:.1f})")
        
        effect_info = f"Effects applied: {', '.join(effects) if effects else 'None'}\n"
        effect_info += f"Opacity mode: {opacity_mode}\n"
        effect_info += f"Final size: {width}x{height}"
        
        return (final_tensor, mask_tensor, json.dumps(config), effect_info)


# Test function
if __name__ == "__main__":
    node = RajTextEffects()
    print("Text Effects node initialized")
    print("Available border styles:", node.INPUT_TYPES()["required"]["border_style"][0])
    print("Available opacity modes:", node.INPUT_TYPES()["required"]["opacity_mode"][0])