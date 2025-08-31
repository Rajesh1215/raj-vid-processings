import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import json
import math
from typing import Dict, List, Tuple, Optional, Union
from .utils import logger

class RajTextAnimator:
    """
    Professional text animation node with motion graphics and opacity animations.
    Creates animated sequences from static text images with various effects.
    """
    
    # Animation types and their descriptions
    ANIMATIONS = {
        "none": "No animation",
        "fade": "Fade in/out with opacity",
        "slide": "Slide from direction",
        "scale": "Scale/zoom effect",
        "rotate": "Rotation animation",
        "typewriter": "Character-by-character reveal",
        "blur": "Blur to focus effect",
        "glitch": "Digital glitch effect",
        "wave": "Wave motion effect",
        "bounce": "Bouncing motion",
        "elastic": "Elastic scaling",
        "shake": "Camera shake effect",
        "spiral": "Spiral motion",
        "flip": "3D flip effect",
        "matrix": "Matrix-style character rain"
    }
    
    # Easing functions
    EASING_FUNCTIONS = {
        "linear": lambda t: t,
        "ease_in": lambda t: t * t,
        "ease_out": lambda t: 1 - (1 - t) * (1 - t),
        "ease_in_out": lambda t: 2 * t * t if t < 0.5 else 1 - 2 * (1 - t) * (1 - t),
        "bounce": lambda t: 1 - abs(math.sin(t * math.pi * 4)) * (1 - t),
        "elastic": lambda t: 1 - math.pow(2, -10 * t) * math.sin((t - 0.1) * 5 * math.pi) if t > 0 else 0,
        "back": lambda t: t * t * (2.7 * t - 1.7),
        "sine": lambda t: 1 - math.cos(t * math.pi / 2),
        "expo": lambda t: 0 if t == 0 else math.pow(2, 10 * (t - 1)),
        "circ": lambda t: 1 - math.sqrt(1 - t * t)
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_image": ("IMAGE", {
                    "tooltip": "Input text image"
                }),
                "animation_type": (list(cls.ANIMATIONS.keys()), {
                    "default": "none",
                    "tooltip": "Animation type"
                }),
                "animation_duration": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Animation duration in seconds"
                }),
                "easing": (list(cls.EASING_FUNCTIONS.keys()), {
                    "default": "ease_in_out",
                    "tooltip": "Animation easing function"
                }),
                "direction": (["left", "right", "up", "down", "center", "top_left", "top_right", "bottom_left", "bottom_right", "random"], {
                    "default": "center",
                    "tooltip": "Animation direction"
                }),
                "opacity_animate": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include opacity in animation"
                }),
                "opacity_start": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Starting opacity"
                }),
                "opacity_end": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Ending opacity"
                }),
                "scale_start": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Starting scale factor"
                }),
                "scale_end": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Ending scale factor"
                }),
                "rotation_start": ("FLOAT", {
                    "default": 0.0,
                    "min": -360.0,
                    "max": 360.0,
                    "step": 1.0,
                    "tooltip": "Starting rotation (degrees)"
                }),
                "rotation_end": ("FLOAT", {
                    "default": 0.0,
                    "min": -360.0,
                    "max": 360.0,
                    "step": 1.0,
                    "tooltip": "Ending rotation (degrees)"
                }),
                "loop": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Loop animation"
                }),
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Animation frame rate"
                }),
            },
            "optional": {
                "opacity_mask": ("MASK", {
                    "tooltip": "Custom opacity mask for animation"
                }),
                "text_config": ("STRING", {
                    "default": "{}",
                    "tooltip": "Text configuration JSON"
                }),
                "custom_path": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Custom animation path (JSON coordinates)"
                }),
                "blur_amount": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Blur amount for blur animations"
                }),
                "wave_amplitude": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Wave animation amplitude"
                }),
                "wave_frequency": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Wave animation frequency"
                }),
                "glitch_intensity": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Glitch effect intensity"
                }),
                "typewriter_speed": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Typewriter reveal speed (seconds per character)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("animated_sequence", "animation_data", "timing_info")
    FUNCTION = "animate_text"
    CATEGORY = "Raj Video Processing 🎬/Text"
    
    @staticmethod
    def apply_easing(t: float, easing_name: str) -> float:
        """Apply easing function to time value."""
        easing_func = RajTextAnimator.EASING_FUNCTIONS.get(easing_name, lambda x: x)
        return max(0.0, min(1.0, easing_func(t)))
    
    @staticmethod
    def create_transformation_matrix(scale: float, rotation: float, 
                                   translate_x: float, translate_y: float) -> np.ndarray:
        """Create 2D transformation matrix."""
        # Scale matrix
        scale_matrix = np.array([
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ])
        
        # Rotation matrix
        angle_rad = math.radians(rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        # Translation matrix
        translation_matrix = np.array([
            [1, 0, translate_x],
            [0, 1, translate_y],
            [0, 0, 1]
        ])
        
        # Combine transformations
        return translation_matrix @ rotation_matrix @ scale_matrix
    
    @staticmethod
    def apply_transform(image: Image.Image, matrix: np.ndarray, 
                       output_size: Tuple[int, int]) -> Image.Image:
        """Apply transformation matrix to image."""
        # Convert matrix to PIL format (flatten and take first 6 elements)
        transform_data = [
            matrix[0, 0], matrix[0, 1], matrix[0, 2],
            matrix[1, 0], matrix[1, 1], matrix[1, 2]
        ]
        
        try:
            transformed = image.transform(
                output_size,
                Image.Transform.AFFINE,
                transform_data,
                resample=Image.Resampling.BILINEAR,
                fillcolor=(0, 0, 0, 0)
            )
            return transformed
        except Exception as e:
            logger.warning(f"Transform failed: {e}, returning original")
            return image
    
    def create_typewriter_effect(self, text_image: Image.Image, progress: float,
                                text_config: Dict) -> Image.Image:
        """Create typewriter effect by revealing characters progressively."""
        # This is a simplified version - would need actual text information
        # for proper character-by-character reveal
        width, height = text_image.size
        
        # Create mask that reveals from left to right
        mask = Image.new('L', (width, height), 0)
        reveal_width = int(width * progress)
        
        if reveal_width > 0:
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rectangle([0, 0, reveal_width, height], fill=255)
        
        # Apply mask to text
        result = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        result.paste(text_image, (0, 0), mask)
        
        return result
    
    def create_glitch_effect(self, text_image: Image.Image, intensity: float) -> Image.Image:
        """Create digital glitch effect."""
        result = text_image.copy()
        width, height = result.size
        
        if intensity <= 0:
            return result
        
        # RGB channel shifting
        r, g, b, a = result.split()
        
        # Shift channels slightly
        shift_amount = int(intensity * 10)
        if shift_amount > 0:
            # Shift red channel
            r_shifted = Image.new('L', (width, height), 0)
            r_shifted.paste(r, (shift_amount, 0))
            
            # Shift blue channel
            b_shifted = Image.new('L', (width, height), 0)
            b_shifted.paste(b, (-shift_amount, 0))
            
            result = Image.merge('RGBA', [r_shifted, g, b_shifted, a])
        
        # Add scanline effect
        if intensity > 0.3:
            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            for y in range(0, height, 4):
                line_opacity = int(intensity * 100)
                draw.line([(0, y), (width, y)], fill=(0, 0, 0, line_opacity))
            
            result = Image.alpha_composite(result, overlay)
        
        return result
    
    def create_wave_effect(self, text_image: Image.Image, amplitude: float,
                          frequency: float, time: float) -> Image.Image:
        """Create wave distortion effect."""
        width, height = text_image.size
        result = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        # Apply wave distortion
        for y in range(height):
            wave_offset = int(amplitude * math.sin(frequency * y * 0.1 + time * 5))
            
            # Copy row with offset
            if 0 <= y < height:
                for x in range(width):
                    source_x = x - wave_offset
                    if 0 <= source_x < width:
                        pixel = text_image.getpixel((source_x, y))
                        result.putpixel((x, y), pixel)
        
        return result
    
    def animate_text(self, text_image, animation_type, animation_duration, easing,
                    direction, opacity_animate, opacity_start, opacity_end,
                    scale_start, scale_end, rotation_start, rotation_end,
                    loop, fps, opacity_mask=None, text_config="{}",
                    custom_path="", blur_amount=5.0, wave_amplitude=10.0,
                    wave_frequency=2.0, glitch_intensity=0.1, typewriter_speed=0.1):
        
        # Parse text config
        try:
            config = json.loads(text_config) if text_config else {}
        except:
            config = {}
        
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
        
        # Calculate frame count
        total_frames = int(animation_duration * fps)
        if total_frames < 1:
            total_frames = 1
        
        frames = []
        animation_data = {
            "type": animation_type,
            "duration": animation_duration,
            "fps": fps,
            "total_frames": total_frames,
            "easing": easing,
            "direction": direction
        }
        
        # Calculate movement distances based on direction
        move_distance = max(width, height)  # Move completely off-screen
        
        direction_offsets = {
            "left": (-move_distance, 0),
            "right": (move_distance, 0),
            "up": (0, -move_distance),
            "down": (0, move_distance),
            "top_left": (-move_distance, -move_distance),
            "top_right": (move_distance, -move_distance),
            "bottom_left": (-move_distance, move_distance),
            "bottom_right": (move_distance, move_distance),
            "center": (0, 0),
            "random": (np.random.randint(-move_distance//2, move_distance//2),
                      np.random.randint(-move_distance//2, move_distance//2))
        }
        
        start_offset = direction_offsets.get(direction, (0, 0))
        
        for frame_idx in range(total_frames):
            # Calculate progress (0.0 to 1.0)
            progress = frame_idx / max(1, total_frames - 1)
            
            # Apply easing
            eased_progress = self.apply_easing(progress, easing)
            
            # Start with original image
            current_frame = input_image.copy()
            
            # Apply animation based on type
            if animation_type == "none":
                pass  # No animation
            
            elif animation_type == "fade":
                if opacity_animate:
                    current_opacity = opacity_start + (opacity_end - opacity_start) * eased_progress
                    # Apply opacity
                    frame_array = np.array(current_frame)
                    frame_array[:, :, 3] = (frame_array[:, :, 3] * current_opacity).astype(np.uint8)
                    current_frame = Image.fromarray(frame_array, 'RGBA')
            
            elif animation_type == "slide":
                # Calculate position
                offset_x = start_offset[0] * (1.0 - eased_progress)
                offset_y = start_offset[1] * (1.0 - eased_progress)
                
                # Create transformation matrix
                transform_matrix = self.create_transformation_matrix(
                    1.0, 0.0, offset_x, offset_y
                )
                current_frame = self.apply_transform(current_frame, transform_matrix, (width, height))
                
                if opacity_animate:
                    current_opacity = opacity_start + (opacity_end - opacity_start) * eased_progress
                    frame_array = np.array(current_frame)
                    frame_array[:, :, 3] = (frame_array[:, :, 3] * current_opacity).astype(np.uint8)
                    current_frame = Image.fromarray(frame_array, 'RGBA')
            
            elif animation_type == "scale":
                # Calculate scale
                current_scale = scale_start + (scale_end - scale_start) * eased_progress
                
                # Create transformation matrix (scale from center)
                center_x, center_y = width // 2, height // 2
                transform_matrix = self.create_transformation_matrix(
                    current_scale, 0.0, 
                    center_x * (1 - current_scale), 
                    center_y * (1 - current_scale)
                )
                current_frame = self.apply_transform(current_frame, transform_matrix, (width, height))
                
                if opacity_animate:
                    current_opacity = opacity_start + (opacity_end - opacity_start) * eased_progress
                    frame_array = np.array(current_frame)
                    frame_array[:, :, 3] = (frame_array[:, :, 3] * current_opacity).astype(np.uint8)
                    current_frame = Image.fromarray(frame_array, 'RGBA')
            
            elif animation_type == "rotate":
                # Calculate rotation
                current_rotation = rotation_start + (rotation_end - rotation_start) * eased_progress
                
                # Create transformation matrix (rotate around center)
                center_x, center_y = width // 2, height // 2
                transform_matrix = self.create_transformation_matrix(
                    1.0, current_rotation, 0, 0
                )
                current_frame = self.apply_transform(current_frame, transform_matrix, (width, height))
            
            elif animation_type == "typewriter":
                current_frame = self.create_typewriter_effect(current_frame, eased_progress, config)
            
            elif animation_type == "blur":
                if eased_progress < 1.0:
                    blur_radius = blur_amount * (1.0 - eased_progress)
                    if blur_radius > 0.1:
                        current_frame = current_frame.filter(ImageFilter.GaussianBlur(blur_radius))
                
                if opacity_animate:
                    current_opacity = opacity_start + (opacity_end - opacity_start) * eased_progress
                    frame_array = np.array(current_frame)
                    frame_array[:, :, 3] = (frame_array[:, :, 3] * current_opacity).astype(np.uint8)
                    current_frame = Image.fromarray(frame_array, 'RGBA')
            
            elif animation_type == "glitch":
                # Vary glitch intensity over time
                current_intensity = glitch_intensity * (1.0 - abs(0.5 - progress) * 2)
                current_frame = self.create_glitch_effect(current_frame, current_intensity)
            
            elif animation_type == "wave":
                current_frame = self.create_wave_effect(
                    current_frame, wave_amplitude, wave_frequency, progress
                )
            
            elif animation_type == "bounce":
                # Bouncing motion
                bounce_height = int(move_distance * 0.3 * abs(math.sin(progress * math.pi * 4)) * (1 - progress))
                transform_matrix = self.create_transformation_matrix(
                    1.0, 0.0, 0, -bounce_height
                )
                current_frame = self.apply_transform(current_frame, transform_matrix, (width, height))
            
            elif animation_type == "elastic":
                # Elastic scaling
                elastic_scale = 1.0 + 0.3 * math.sin(progress * math.pi * 6) * (1 - progress)
                center_x, center_y = width // 2, height // 2
                transform_matrix = self.create_transformation_matrix(
                    elastic_scale, 0.0,
                    center_x * (1 - elastic_scale),
                    center_y * (1 - elastic_scale)
                )
                current_frame = self.apply_transform(current_frame, transform_matrix, (width, height))
            
            elif animation_type == "shake":
                # Camera shake effect
                shake_intensity = 10 * (1 - progress)
                shake_x = np.random.randint(-int(shake_intensity), int(shake_intensity) + 1)
                shake_y = np.random.randint(-int(shake_intensity), int(shake_intensity) + 1)
                
                transform_matrix = self.create_transformation_matrix(
                    1.0, 0.0, shake_x, shake_y
                )
                current_frame = self.apply_transform(current_frame, transform_matrix, (width, height))
            
            elif animation_type == "spiral":
                # Spiral motion
                angle = progress * 360 * 2  # Two full rotations
                radius = move_distance * 0.5 * (1 - progress)
                spiral_x = radius * math.cos(math.radians(angle))
                spiral_y = radius * math.sin(math.radians(angle))
                
                transform_matrix = self.create_transformation_matrix(
                    1.0, angle, spiral_x, spiral_y
                )
                current_frame = self.apply_transform(current_frame, transform_matrix, (width, height))
            
            # Apply custom opacity mask if provided
            if opacity_mask is not None:
                mask_np = opacity_mask[0].cpu().numpy() if isinstance(opacity_mask, torch.Tensor) else opacity_mask
                custom_mask = Image.fromarray((mask_np * 255).astype(np.uint8), 'L')
                custom_mask = custom_mask.resize((width, height))
                
                frame_array = np.array(current_frame)
                mask_array = np.array(custom_mask)
                frame_array[:, :, 3] = (frame_array[:, :, 3] * mask_array // 255).astype(np.uint8)
                current_frame = Image.fromarray(frame_array, 'RGBA')
            
            frames.append(current_frame)
        
        # Convert frames to tensor
        frames_np = np.array([np.array(frame).astype(np.float32) / 255.0 for frame in frames])
        frames_tensor = torch.from_numpy(frames_np)
        
        # Create timing info
        timing_info = (
            f"Animation: {animation_type}\n"
            f"Duration: {animation_duration:.1f}s @ {fps:.1f} FPS\n"
            f"Frames: {total_frames}\n"
            f"Easing: {easing}\n"
            f"Direction: {direction}\n"
            f"Loop: {'Yes' if loop else 'No'}"
        )
        
        return (frames_tensor, json.dumps(animation_data), timing_info)


# Test function
if __name__ == "__main__":
    node = RajTextAnimator()
    print("Text Animator node initialized")
    print(f"Available animations: {list(node.ANIMATIONS.keys())}")
    print(f"Available easing functions: {list(node.EASING_FUNCTIONS.keys())}")