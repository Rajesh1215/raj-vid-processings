import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import json
from typing import Dict, List, Tuple, Optional, Union
from .utils import logger
import requests
import zipfile
from pathlib import Path

class RajTextGenerator:
    """
    Core text generation node that creates 512x512 (or custom size) text images using PIL.
    Supports advanced typography, auto-fitting, and opacity control.
    """
    
    # Font management
    FONT_CACHE = {}
    GOOGLE_FONTS_URL = "https://fonts.google.com/download?family="
    
    # Default font list
    DEFAULT_FONTS = [
        "Arial", "Helvetica", "Times New Roman", "Courier New", "Verdana",
        "Georgia", "Palatino", "Garamond", "Bookman", "Comic Sans MS",
        "Trebuchet MS", "Arial Black", "Impact", "Lucida Sans", "Tahoma",
        "Century Gothic", "Lucida Console", "Monaco", "Courier", "System"
    ]
    
    # Google Fonts collection
    GOOGLE_FONTS = [
        "Roboto", "Open Sans", "Lato", "Montserrat", "Oswald",
        "Source Sans Pro", "Raleway", "Poppins", "Noto Sans", "Ubuntu",
        "Playfair Display", "Merriweather", "Lora", "PT Sans", "Nunito",
        "Bebas Neue", "Anton", "Lobster", "Pacifico", "Dancing Script",
        "Shadows Into Light", "Indie Flower", "Amatic SC", "Permanent Marker",
        "Caveat", "Abril Fatface", "Righteous", "Alfa Slab One", "Russo One",
        "Orbitron", "Press Start 2P", "VT323", "Special Elite", "Creepster"
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        all_fonts = cls.DEFAULT_FONTS + cls.GOOGLE_FONTS
        
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Enter text here",
                    "tooltip": "Text content to render"
                }),
                "output_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Output image width in pixels"
                }),
                "output_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Output image height in pixels"
                }),
                "font_name": (all_fonts, {
                    "default": "Arial",
                    "tooltip": "Font family to use"
                }),
                "font_size": ("INT", {
                    "default": 48,
                    "min": 8,
                    "max": 500,
                    "step": 1,
                    "tooltip": "Font size in pixels"
                }),
                "font_color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Text color (HEX or RGB)"
                }),
                "background_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Background color (HEX or 'transparent')"
                }),
                "text_align": (["left", "center", "right", "justify"], {
                    "default": "center",
                    "tooltip": "Text alignment"
                }),
                "vertical_align": (["top", "middle", "bottom"], {
                    "default": "middle",
                    "tooltip": "Vertical alignment"
                }),
                "words_per_line": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Max words per line (0 = auto-fit)"
                }),
                "max_lines": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Max number of lines (0 = auto-fit)"
                }),
                "line_spacing": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Line height multiplier"
                }),
                "letter_spacing": ("INT", {
                    "default": 0,
                    "min": -20,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Letter spacing in pixels"
                }),
                "margin_x": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Horizontal margin"
                }),
                "margin_y": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Vertical margin"
                }),
                "auto_size": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Auto-fit text to container"
                }),
                "base_opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Overall text opacity"
                }),
            },
            "optional": {
                "font_file": ("STRING", {
                    "default": "",
                    "tooltip": "Path to custom font file (.ttf/.otf)"
                }),
                "time_display": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Display duration in seconds (0 = static)"
                }),
                "fade_in": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Fade in duration"
                }),
                "fade_out": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Fade out duration"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "TEXT_CONFIG", "STRING")
    RETURN_NAMES = ("text_image", "alpha_mask", "text_settings", "render_info")
    FUNCTION = "generate_text"
    CATEGORY = "Raj Video Processing 🎬/Text"
    
    # Cache for found system fonts
    SYSTEM_FONT_CACHE = {}
    
    @classmethod
    def find_system_font(cls, font_name: str) -> Optional[str]:
        """Find font file in system directories."""
        # Check cache first
        if font_name in cls.SYSTEM_FONT_CACHE:
            return cls.SYSTEM_FONT_CACHE[font_name]
        
        import platform
        
        system = platform.system()
        font_dirs = []
        
        if system == "Darwin":  # macOS
            font_dirs = [
                "/System/Library/Fonts/",
                "/Library/Fonts/",
                os.path.expanduser("~/Library/Fonts/")
            ]
        elif system == "Linux":
            font_dirs = [
                "/usr/share/fonts/",
                "/usr/local/share/fonts/",
                os.path.expanduser("~/.fonts/")
            ]
        elif system == "Windows":
            font_dirs = [
                "C:\\Windows\\Fonts\\",
                os.path.expanduser("~\\AppData\\Local\\Microsoft\\Windows\\Fonts\\")
            ]
        
        # Search for font file
        found_path = None
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for root, dirs, files in os.walk(font_dir):
                    for file in files:
                        if font_name.lower() in file.lower():
                            if file.endswith(('.ttf', '.ttc', '.otf')):
                                found_path = os.path.join(root, file)
                                break
                    if found_path:
                        break
                if found_path:
                    break
        
        # Cache the result (even if None to avoid repeated searches)
        cls.SYSTEM_FONT_CACHE[font_name] = found_path
        return found_path
    
    @classmethod
    def validate_font(cls, font, font_name: str, font_size: int) -> bool:
        """Validate that font is working correctly."""
        try:
            test_text = "Test"
            bbox = font.getbbox(test_text)
            if bbox:
                height = bbox[3] - bbox[1]
                # Check if font height is reasonable (at least 30% of requested size)
                if height < font_size * 0.3:
                    logger.warning(f"Font {font_name} renders too small: {height}px vs requested {font_size}px")
                    return False
            return True
        except:
            return False
    
    @classmethod
    def get_font(cls, font_name: str, font_size: int, font_file: str = "") -> ImageFont:
        """Get or download font with improved fallback."""
        cache_key = f"{font_name}_{font_size}_{font_file}"
        
        if cache_key in cls.FONT_CACHE:
            return cls.FONT_CACHE[cache_key]
        
        font = None
        
        # Try custom font file first
        if font_file and os.path.exists(font_file):
            try:
                font = ImageFont.truetype(font_file, font_size)
                if cls.validate_font(font, font_name, font_size):
                    cls.FONT_CACHE[cache_key] = font
                    return font
            except Exception as e:
                logger.warning(f"Failed to load custom font: {e}")
        
        # Try direct system font paths
        font_paths = [
            f"/System/Library/Fonts/{font_name}.ttc",
            f"/System/Library/Fonts/{font_name}.ttf",
            f"/Library/Fonts/{font_name}.ttf",
            f"/usr/share/fonts/truetype/{font_name.lower()}/{font_name}.ttf",
            f"C:\\Windows\\Fonts\\{font_name}.ttf",
            f"C:\\Windows\\Fonts\\{font_name.lower()}.ttf",
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, font_size)
                    if cls.validate_font(font, font_name, font_size):
                        cls.FONT_CACHE[cache_key] = font
                        return font
                except:
                    continue
        
        # Try system font discovery
        found_path = cls.find_system_font(font_name)
        if found_path:
            try:
                font = ImageFont.truetype(found_path, font_size)
                if cls.validate_font(font, font_name, font_size):
                    if cache_key not in cls.FONT_CACHE:  # Only log on first load
                        logger.info(f"Found font at: {found_path}")
                    cls.FONT_CACHE[cache_key] = font
                    return font
            except Exception as e:
                logger.warning(f"Failed to load found font: {e}")
        
        # Try to download from Google Fonts
        if font_name in cls.GOOGLE_FONTS:
            font_dir = Path("fonts") / font_name.replace(" ", "_")
            font_dir.mkdir(parents=True, exist_ok=True)
            font_file = font_dir / f"{font_name.replace(' ', '_')}.ttf"
            
            if not font_file.exists():
                try:
                    logger.info(f"Downloading Google Font: {font_name}")
                    # This is a simplified approach - in production, use proper Google Fonts API
                    # For now, fall back to default font
                except Exception as e:
                    logger.warning(f"Could not download font {font_name}: {e}")
        
        # Improved fallback to default font
        logger.warning(f"Font '{font_name}' not found, using fallback")
        try:
            # Try to load a basic system font at requested size
            fallback_fonts = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial Unicode.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "C:\\Windows\\Fonts\\arial.ttf"
            ]
            
            for fallback_path in fallback_fonts:
                if os.path.exists(fallback_path):
                    try:
                        font = ImageFont.truetype(fallback_path, font_size)
                        logger.info(f"Using fallback font: {fallback_path}")
                        break
                    except:
                        continue
            
            if not font:
                # Last resort: use default but warn user
                font = ImageFont.load_default()
                logger.warning(f"Using tiny default font - requested size {font_size}px not available")
        except Exception as e:
            logger.error(f"Failed to load fallback font: {e}")
            font = ImageFont.load_default()
        
        if font:
            cls.FONT_CACHE[cache_key] = font
        
        return font
    
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
    def wrap_text(text: str, font: ImageFont, max_width: int, words_per_line: int = 0) -> List[str]:
        """Wrap text to fit within max width."""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            # Check words per line limit
            if words_per_line > 0 and len(current_line) >= words_per_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                test_line = " ".join(current_line + [word])
                bbox = font.getbbox(test_line)
                width = bbox[2] - bbox[0] if bbox else 0
                
                if width <= max_width or not current_line:
                    current_line.append(word)
                else:
                    lines.append(" ".join(current_line))
                    current_line = [word]
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines
    
    def generate_text(self, text, output_width, output_height, font_name, font_size,
                     font_color, background_color, text_align, vertical_align,
                     words_per_line, max_lines, line_spacing, letter_spacing,
                     margin_x, margin_y, auto_size, base_opacity,
                     font_file="", time_display=0.0, fade_in=0.0, fade_out=0.0):
        
        # Parse colors
        text_color = self.parse_color(font_color)
        bg_color = self.parse_color(background_color)
        
        # Apply base opacity to text color
        text_color = (text_color[0], text_color[1], text_color[2], int(text_color[3] * base_opacity))
        
        # Create image with alpha channel
        image = Image.new('RGBA', (output_width, output_height), bg_color)
        draw = ImageDraw.Draw(image)
        
        # Get font
        font = self.get_font(font_name, font_size, font_file)
        if not font:
            logger.error(f"Could not load font: {font_name}")
            font = ImageFont.load_default()
        
        # Calculate available space
        available_width = output_width - (2 * margin_x)
        available_height = output_height - (2 * margin_y)
        
        # Auto-size font if requested
        if auto_size:
            # Start with large size and reduce until text fits
            test_size = min(output_height // 2, 200)
            while test_size > 8:
                test_font = self.get_font(font_name, test_size, font_file)
                lines = self.wrap_text(text, test_font, available_width, words_per_line)
                
                # Calculate total height using actual font metrics
                if hasattr(test_font, 'getmetrics'):
                    ascent, descent = test_font.getmetrics()
                    line_height = (ascent + descent) * line_spacing
                else:
                    # Fallback for bitmap fonts
                    test_bbox = test_font.getbbox("Ay")
                    line_height = (test_bbox[3] - test_bbox[1]) * line_spacing if test_bbox else test_size * line_spacing
                
                total_height = len(lines) * line_height
                
                if total_height <= available_height and (max_lines == 0 or len(lines) <= max_lines):
                    font = test_font
                    font_size = test_size
                    break
                
                test_size -= 2
        
        # Wrap text
        lines = self.wrap_text(text, font, available_width, words_per_line)
        
        # Apply max lines limit
        if max_lines > 0 and len(lines) > max_lines:
            lines = lines[:max_lines]
            # Add ellipsis to last line if truncated
            if len(lines) == max_lines:
                lines[-1] = lines[-1] + "..."
        
        # Calculate line height and total text height using actual font metrics
        if hasattr(font, 'getmetrics'):
            ascent, descent = font.getmetrics()
            line_height = (ascent + descent) * line_spacing
        else:
            # Fallback for bitmap fonts
            test_bbox = font.getbbox("Ay")
            line_height = (test_bbox[3] - test_bbox[1]) * line_spacing if test_bbox else font_size * line_spacing
        
        total_height = len(lines) * line_height
        
        # Calculate starting Y position based on vertical alignment
        if vertical_align == "top":
            y = margin_y
        elif vertical_align == "bottom":
            y = output_height - margin_y - total_height
        else:  # middle
            y = (output_height - total_height) // 2
        
        # Draw each line with improved positioning
        for line in lines:
            # Get line width with proper metrics
            bbox = font.getbbox(line) if hasattr(font, 'getbbox') else None
            if bbox:
                line_width = bbox[2] - bbox[0]
                # Account for left bearing
                x_offset = -bbox[0] if bbox[0] < 0 else 0
            else:
                # Rough estimate for fallback
                line_width = len(line) * (font_size * 0.6)
                x_offset = 0
            
            # Calculate X position based on alignment
            if text_align == "left":
                x = margin_x
            elif text_align == "right":
                x = output_width - margin_x - line_width
            elif text_align == "center":
                x = (output_width - line_width) // 2
            else:  # justify
                x = margin_x
                # TODO: Implement justify by adjusting word spacing
            
            # Apply letter spacing if needed
            if letter_spacing != 0:
                # Draw each character separately with spacing
                char_x = x + x_offset
                for char in line:
                    draw.text((char_x, y), char, font=font, fill=text_color)
                    char_bbox = font.getbbox(char) if hasattr(font, 'getbbox') else None
                    char_width = char_bbox[2] - char_bbox[0] if char_bbox else font_size * 0.6
                    char_x += char_width + letter_spacing
            else:
                # Draw the entire line with offset
                draw.text((x + x_offset, y), line, font=font, fill=text_color)
            
            y += line_height
        
        # Create alpha mask from the alpha channel
        alpha_mask = image.split()[3]
        
        # Convert to tensor format (B, H, W, C)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        
        # Create mask tensor
        mask_np = np.array(alpha_mask).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        
        # Create text configuration
        text_config = {
            "text": text,
            "font_name": font_name,
            "font_size": font_size,
            "font_color": font_color,
            "background_color": background_color,
            "text_align": text_align,
            "vertical_align": vertical_align,
            "words_per_line": words_per_line,
            "max_lines": max_lines,
            "line_spacing": line_spacing,
            "letter_spacing": letter_spacing,
            "margin_x": margin_x,
            "margin_y": margin_y,
            "auto_size": auto_size,
            "base_opacity": base_opacity,
            "time_display": time_display,
            "fade_in": fade_in,
            "fade_out": fade_out,
            "output_width": output_width,
            "output_height": output_height,
            "actual_lines": len(lines),
            "truncated": len(self.wrap_text(text, font, available_width, words_per_line)) > len(lines)
        }
        
        # Create render info
        render_info = (
            f"Text rendered: {len(lines)} lines\n"
            f"Font: {font_name} @ {font_size}px\n"
            f"Size: {output_width}x{output_height}\n"
            f"Opacity: {base_opacity:.2f}\n"
            f"Timing: {time_display:.1f}s (fade in: {fade_in:.1f}s, out: {fade_out:.1f}s)"
        )
        
        return (image_tensor, mask_tensor, json.dumps(text_config), render_info)


# Test node
if __name__ == "__main__":
    node = RajTextGenerator()
    result = node.generate_text(
        text="Hello World\nThis is a test",
        output_width=512,
        output_height=512,
        font_name="Arial",
        font_size=48,
        font_color="#FFFFFF",
        background_color="#000000",
        text_align="center",
        vertical_align="middle",
        words_per_line=0,
        max_lines=0,
        line_spacing=1.2,
        letter_spacing=0,
        margin_x=20,
        margin_y=20,
        auto_size=False,
        base_opacity=1.0
    )
    print(result[3])  # Print render info