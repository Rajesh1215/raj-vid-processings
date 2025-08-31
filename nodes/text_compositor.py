import torch
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import json
import cv2
from typing import Dict, List, Tuple, Optional, Union
from .utils import logger

class RajTextCompositor:
    """
    Final overlay compositor for text on video with smart positioning,
    opacity blending, and professional compositing modes.
    """
    
    # Blend modes
    BLEND_MODES = [
        "normal", "multiply", "screen", "overlay", "soft_light", "hard_light",
        "color_dodge", "color_burn", "darken", "lighten", "difference", "exclusion"
    ]
    
    # Position presets
    POSITION_PRESETS = [
        "top_left", "top_center", "top_right",
        "middle_left", "center", "middle_right", 
        "bottom_left", "bottom_center", "bottom_right",
        "lower_third", "upper_third", "title_safe"
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_frames": ("IMAGE", {
                    "tooltip": "Background video frames"
                }),
                "text_overlay": ("IMAGE", {
                    "tooltip": "Text image/sequence to overlay"
                }),
                "position_mode": (["manual", "auto", "track", "preset"], {
                    "default": "manual",
                    "tooltip": "Positioning mode"
                }),
                "position_x": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "X position (pixels or percentage)"
                }),
                "position_y": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Y position (pixels or percentage)"
                }),
                "position_preset": (cls.POSITION_PRESETS, {
                    "default": "center",
                    "tooltip": "Position preset"
                }),
                "blend_mode": (cls.BLEND_MODES, {
                    "default": "normal",
                    "tooltip": "Blend mode"
                }),
                "global_opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Global opacity multiplier"
                }),
                "use_text_alpha": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Respect text alpha channel"
                }),
                "auto_contrast": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Auto-adjust text contrast based on background"
                }),
                "safe_zones": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Respect broadcast safe zones"
                }),
            },
            "optional": {
                "opacity_mask": ("MASK", {
                    "tooltip": "Additional opacity mask"
                }),
                "timeline_data": ("STRING", {
                    "default": "{}",
                    "tooltip": "Timeline from sequencer"
                }),
                "tracking_data": ("STRING", {
                    "default": "{}",
                    "tooltip": "Motion tracking data (JSON)"
                }),
                "face_detection": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Avoid placing text over faces"
                }),
                "edge_detection": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use edge detection for smart placement"
                }),
                "background_analysis": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Analyze background for optimal text placement"
                }),
                "margin_percent": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 25.0,
                    "step": 0.5,
                    "tooltip": "Safety margin as percentage of frame size"
                }),
                "scaling_mode": (["none", "fit", "fill", "stretch"], {
                    "default": "none",
                    "tooltip": "Text scaling mode"
                }),
                "motion_blur": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply motion blur for moving text"
                }),
                "motion_blur_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Motion blur intensity"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("composite_video", "composite_info")
    FUNCTION = "composite_text"
    CATEGORY = "Raj Video Processing 🎬/Text"
    
    @staticmethod
    def apply_blend_mode(base: np.ndarray, overlay: np.ndarray, mode: str) -> np.ndarray:
        """Apply blend mode to combine base and overlay images."""
        if mode == "normal":
            return overlay
        
        # Normalize to 0-1 range
        base = base.astype(np.float32) / 255.0
        overlay = overlay.astype(np.float32) / 255.0
        
        if mode == "multiply":
            result = base * overlay
        elif mode == "screen":
            result = 1 - (1 - base) * (1 - overlay)
        elif mode == "overlay":
            result = np.where(base < 0.5, 
                             2 * base * overlay, 
                             1 - 2 * (1 - base) * (1 - overlay))
        elif mode == "soft_light":
            result = np.where(overlay < 0.5,
                             base - (1 - 2 * overlay) * base * (1 - base),
                             base + (2 * overlay - 1) * (np.sqrt(base) - base))
        elif mode == "hard_light":
            result = np.where(overlay < 0.5,
                             2 * base * overlay,
                             1 - 2 * (1 - base) * (1 - overlay))
        elif mode == "color_dodge":
            result = np.where(overlay >= 0.999, 1.0, np.minimum(1.0, base / (1 - overlay)))
        elif mode == "color_burn":
            result = np.where(overlay <= 0.001, 0.0, np.maximum(0.0, 1 - (1 - base) / overlay))
        elif mode == "darken":
            result = np.minimum(base, overlay)
        elif mode == "lighten":
            result = np.maximum(base, overlay)
        elif mode == "difference":
            result = np.abs(base - overlay)
        elif mode == "exclusion":
            result = base + overlay - 2 * base * overlay
        else:
            result = overlay
        
        return (result * 255).astype(np.uint8)
    
    @staticmethod
    def calculate_position_preset(preset: str, frame_width: int, frame_height: int,
                                text_width: int, text_height: int, margin_percent: float) -> Tuple[int, int]:
        """Calculate position based on preset."""
        margin_x = int(frame_width * margin_percent / 100)
        margin_y = int(frame_height * margin_percent / 100)
        
        safe_width = frame_width - 2 * margin_x
        safe_height = frame_height - 2 * margin_y
        
        positions = {
            "top_left": (margin_x, margin_y),
            "top_center": ((frame_width - text_width) // 2, margin_y),
            "top_right": (frame_width - text_width - margin_x, margin_y),
            "middle_left": (margin_x, (frame_height - text_height) // 2),
            "center": ((frame_width - text_width) // 2, (frame_height - text_height) // 2),
            "middle_right": (frame_width - text_width - margin_x, (frame_height - text_height) // 2),
            "bottom_left": (margin_x, frame_height - text_height - margin_y),
            "bottom_center": ((frame_width - text_width) // 2, frame_height - text_height - margin_y),
            "bottom_right": (frame_width - text_width - margin_x, frame_height - text_height - margin_y),
            "lower_third": ((frame_width - text_width) // 2, int(frame_height * 0.75) - text_height // 2),
            "upper_third": ((frame_width - text_width) // 2, int(frame_height * 0.25) - text_height // 2),
            "title_safe": ((frame_width - text_width) // 2, int(frame_height * 0.1))
        }
        
        return positions.get(preset, (0, 0))
    
    @staticmethod
    def analyze_background_region(frame: np.ndarray, x: int, y: int, 
                                 width: int, height: int) -> Dict:
        """Analyze background region for optimal text placement."""
        # Extract region
        region = frame[y:y+height, x:x+width]
        
        # Calculate average brightness
        gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if len(region.shape) == 3 else region
        brightness = np.mean(gray_region)
        
        # Calculate contrast (standard deviation)
        contrast = np.std(gray_region)
        
        # Calculate edge density
        edges = cv2.Canny(gray_region, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        
        return {
            "brightness": brightness / 255.0,
            "contrast": contrast / 255.0,
            "edge_density": edge_density,
            "is_busy": edge_density > 0.1,
            "is_dark": brightness < 100,
            "is_light": brightness > 200
        }
    
    @staticmethod
    def detect_faces(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame (simplified using OpenCV)."""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
            
            # Load face cascade (would need to be installed)
            # For now, return empty list as placeholder
            # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            # return [(x, y, w, h) for x, y, w, h in faces]
            
            return []  # Placeholder - no face detection without cascade file
        except:
            return []
    
    @staticmethod
    def find_optimal_position(frame: np.ndarray, text_width: int, text_height: int,
                            avoid_faces: bool = False, prefer_edges: bool = False) -> Tuple[int, int]:
        """Find optimal position for text placement."""
        frame_height, frame_width = frame.shape[:2]
        
        # Define candidate positions
        positions = [
            (frame_width // 4, frame_height // 4),  # Top-left quadrant
            (3 * frame_width // 4 - text_width, frame_height // 4),  # Top-right quadrant
            (frame_width // 4, 3 * frame_height // 4 - text_height),  # Bottom-left quadrant
            (3 * frame_width // 4 - text_width, 3 * frame_height // 4 - text_height),  # Bottom-right
            (frame_width // 2 - text_width // 2, frame_height // 6),  # Top center
            (frame_width // 2 - text_width // 2, 5 * frame_height // 6 - text_height),  # Bottom center
        ]
        
        # Score positions
        best_position = positions[0]
        best_score = -1
        
        for x, y in positions:
            # Ensure position is within frame
            x = max(0, min(x, frame_width - text_width))
            y = max(0, min(y, frame_height - text_height))
            
            # Analyze region
            analysis = RajTextCompositor.analyze_background_region(frame, x, y, text_width, text_height)
            
            # Calculate score
            score = 0
            
            # Prefer areas with low edge density (less busy)
            score += (1.0 - analysis["edge_density"]) * 0.4
            
            # Prefer areas with moderate contrast
            score += (1.0 - abs(analysis["contrast"] - 0.5)) * 0.3
            
            # Prefer areas that aren't too bright or too dark
            score += (1.0 - abs(analysis["brightness"] - 0.5)) * 0.2
            
            # Distance from center penalty (prefer off-center)
            center_x, center_y = frame_width // 2, frame_height // 2
            distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            score += (distance_from_center / max_distance) * 0.1
            
            if score > best_score:
                best_score = score
                best_position = (x, y)
        
        return best_position
    
    def composite_text(self, source_frames, text_overlay, position_mode, position_x,
                      position_y, position_preset, blend_mode, global_opacity,
                      use_text_alpha, auto_contrast, safe_zones, opacity_mask=None,
                      timeline_data="{}", tracking_data="{}", face_detection=False,
                      edge_detection=False, background_analysis=False, margin_percent=5.0,
                      scaling_mode="none", motion_blur=False, motion_blur_strength=0.5):
        
        # Get frame dimensions
        source_batch, source_height, source_width, source_channels = source_frames.shape
        text_batch, text_height, text_width, text_channels = text_overlay.shape
        
        logger.info(f"Compositing {text_batch} text frames over {source_batch} source frames")
        
        # Handle different batch sizes
        max_frames = max(source_batch, text_batch)
        
        # Parse timeline and tracking data
        try:
            timeline = json.loads(timeline_data) if timeline_data else {}
            tracking = json.loads(tracking_data) if tracking_data else {}
        except:
            timeline = {}
            tracking = {}
        
        # Process frames
        result_frames = []
        
        for frame_idx in range(max_frames):
            # Get source frame
            source_idx = min(frame_idx, source_batch - 1)
            source_frame = source_frames[source_idx].cpu().numpy()
            source_frame = (source_frame * 255).astype(np.uint8)
            
            # Get text frame
            text_idx = min(frame_idx, text_batch - 1)
            text_frame = text_overlay[text_idx].cpu().numpy()
            text_frame = (text_frame * 255).astype(np.uint8)
            
            # Convert to PIL for easier manipulation
            source_pil = Image.fromarray(source_frame[:, :, :3], 'RGB' if source_channels >= 3 else 'L')
            
            # Ensure text frame has alpha
            if text_channels == 3:
                # Add alpha channel
                alpha = np.ones((text_height, text_width), dtype=np.uint8) * 255
                text_frame = np.dstack([text_frame, alpha])
            
            text_pil = Image.fromarray(text_frame, 'RGBA')
            
            # Apply scaling if needed
            if scaling_mode != "none":
                if scaling_mode == "fit":
                    # Scale to fit within source frame
                    scale_x = source_width / text_width
                    scale_y = source_height / text_height
                    scale = min(scale_x, scale_y)
                elif scaling_mode == "fill":
                    # Scale to fill source frame
                    scale_x = source_width / text_width
                    scale_y = source_height / text_height
                    scale = max(scale_x, scale_y)
                else:  # stretch
                    scale_x = source_width / text_width
                    scale_y = source_height / text_height
                    text_pil = text_pil.resize((source_width, source_height), Image.Resampling.LANCZOS)
                
                if scaling_mode in ["fit", "fill"]:
                    new_width = int(text_width * scale)
                    new_height = int(text_height * scale)
                    text_pil = text_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    text_width, text_height = new_width, new_height
            
            # Calculate position
            if position_mode == "preset":
                pos_x, pos_y = self.calculate_position_preset(
                    position_preset, source_width, source_height,
                    text_width, text_height, margin_percent
                )
            elif position_mode == "auto":
                if background_analysis:
                    pos_x, pos_y = self.find_optimal_position(
                        source_frame, text_width, text_height,
                        face_detection, edge_detection
                    )
                else:
                    # Simple centering
                    pos_x = (source_width - text_width) // 2
                    pos_y = (source_height - text_height) // 2
            elif position_mode == "track" and tracking:
                # Use tracking data (placeholder)
                pos_x = position_x + tracking.get('offset_x', 0)
                pos_y = position_y + tracking.get('offset_y', 0)
            else:  # manual
                pos_x = position_x + (source_width - text_width) // 2
                pos_y = position_y + (source_height - text_height) // 2
            
            # Apply safe zones
            if safe_zones:
                margin_x = int(source_width * margin_percent / 100)
                margin_y = int(source_height * margin_percent / 100)
                pos_x = max(margin_x, min(pos_x, source_width - text_width - margin_x))
                pos_y = max(margin_y, min(pos_y, source_height - text_height - margin_y))
            
            # Auto-contrast adjustment
            if auto_contrast:
                # Analyze background region
                region_analysis = self.analyze_background_region(
                    source_frame, pos_x, pos_y, text_width, text_height
                )
                
                # Adjust text opacity/contrast based on background
                if region_analysis["is_light"]:
                    # Darken text or increase contrast for light backgrounds
                    enhancer = ImageEnhance.Brightness(text_pil)
                    text_pil = enhancer.enhance(0.7)
                elif region_analysis["is_dark"]:
                    # Brighten text for dark backgrounds
                    enhancer = ImageEnhance.Brightness(text_pil)
                    text_pil = enhancer.enhance(1.3)
            
            # Apply global opacity
            if global_opacity < 1.0:
                text_array = np.array(text_pil)
                text_array[:, :, 3] = (text_array[:, :, 3] * global_opacity).astype(np.uint8)
                text_pil = Image.fromarray(text_array, 'RGBA')
            
            # Apply additional opacity mask
            if opacity_mask is not None:
                mask_idx = min(frame_idx, opacity_mask.shape[0] - 1)
                mask_data = opacity_mask[mask_idx].cpu().numpy()
                mask_resized = cv2.resize(mask_data, (text_width, text_height))
                
                text_array = np.array(text_pil)
                text_array[:, :, 3] = (text_array[:, :, 3] * mask_resized).astype(np.uint8)
                text_pil = Image.fromarray(text_array, 'RGBA')
            
            # Motion blur for moving text
            if motion_blur and frame_idx > 0:
                # Simple motion blur approximation
                blur_kernel_size = int(motion_blur_strength * 5)
                if blur_kernel_size > 1:
                    text_pil = text_pil.filter(ImageFilter.GaussianBlur(blur_kernel_size))
            
            # Create composite
            composite = source_pil.copy()
            if source_pil.mode != 'RGBA':
                composite = composite.convert('RGBA')
            
            # Apply blending
            if blend_mode != "normal" and use_text_alpha:
                # Extract text and alpha
                text_rgb = np.array(text_pil)[:, :, :3]
                text_alpha = np.array(text_pil)[:, :, 3]
                
                # Get background region
                bg_region = np.array(composite)[pos_y:pos_y+text_height, pos_x:pos_x+text_width, :3]
                
                # Apply blend mode
                blended = self.apply_blend_mode(bg_region, text_rgb, blend_mode)
                
                # Composite with alpha
                composite_array = np.array(composite)
                for c in range(3):
                    composite_array[pos_y:pos_y+text_height, pos_x:pos_x+text_width, c] = (
                        composite_array[pos_y:pos_y+text_height, pos_x:pos_x+text_width, c] * (255 - text_alpha) +
                        blended[:, :, c] * text_alpha
                    ) // 255
                
                composite = Image.fromarray(composite_array, 'RGBA')
            else:
                # Simple alpha composite
                composite.paste(text_pil, (pos_x, pos_y), text_pil if use_text_alpha else None)
            
            # Convert back to RGB for output
            result_frame = np.array(composite.convert('RGB')).astype(np.float32) / 255.0
            result_frames.append(result_frame)
        
        # Convert to tensor
        result_tensor = torch.from_numpy(np.array(result_frames))
        
        # Create composite info
        composite_info = (
            f"Text Composite Complete\n"
            f"Source frames: {source_batch}\n"
            f"Text frames: {text_batch}\n"
            f"Output frames: {len(result_frames)}\n"
            f"Position mode: {position_mode}\n"
            f"Blend mode: {blend_mode}\n"
            f"Global opacity: {global_opacity:.2f}\n"
            f"Final position: ({pos_x}, {pos_y})\n"
            f"Output size: {source_width}x{source_height}"
        )
        
        return (result_tensor, composite_info)


# Test function
if __name__ == "__main__":
    node = RajTextCompositor()
    print("Text Compositor node initialized")
    print(f"Supported blend modes: {node.BLEND_MODES}")
    print(f"Position presets: {node.POSITION_PRESETS}")