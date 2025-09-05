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
                # Advanced styling options
                "font_weight": (["normal", "bold", "italic", "bold_italic"], {
                    "default": "normal",
                    "tooltip": "Font weight/style"
                }),
                "text_border_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Text border/stroke width in pixels"
                }),
                "text_border_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Text border/stroke color"
                }),
                "shadow_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable text shadow"
                }),
                "shadow_offset_x": ("INT", {
                    "default": 2,
                    "min": -20,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Shadow X offset"
                }),
                "shadow_offset_y": ("INT", {
                    "default": 2,
                    "min": -20,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Shadow Y offset"
                }),
                "shadow_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Shadow color"
                }),
                "shadow_blur": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Shadow blur radius"
                }),
                "text_bg_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable text background highlight"
                }),
                "text_bg_color": ("STRING", {
                    "default": "#FFFF00",
                    "tooltip": "Text background/highlight color"
                }),
                "text_bg_padding": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Text background padding"
                }),
                "gradient_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable gradient text"
                }),
                "gradient_color2": ("STRING", {
                    "default": "#FF0000",
                    "tooltip": "Gradient end color"
                }),
                "gradient_direction": (["vertical", "horizontal", "diagonal"], {
                    "default": "vertical",
                    "tooltip": "Gradient direction"
                }),
                "container_enabled": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable container box around text"
                }),
                "container_color": ("STRING", {
                    "default": "#333333",
                    "tooltip": "Container box background color"
                }),
                "container_width": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Container box border width"
                }),
                "container_padding": ("INT", {
                    "default": 15,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Container box padding around text"
                }),
                "container_border_color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Container box border color"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "TEXT_CONFIG", "STRING", "TEXT_SETTINGS")
    RETURN_NAMES = ("text_image", "transparent_text", "alpha_mask", "text_settings", "render_info", "styling_settings")
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
        
        # Search for font file with improved matching
        found_path = None
        
        # First try exact matches and common variations
        exact_variations = [
            f"{font_name}.ttf",
            f"{font_name}.ttc", 
            f"{font_name}.otf",
            f"{font_name} Regular.ttf",
            f"{font_name}Regular.ttf",
        ]
        
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for root, dirs, files in os.walk(font_dir):
                    # First priority: exact name matches
                    for exact_name in exact_variations:
                        if exact_name in files:
                            found_path = os.path.join(root, exact_name)
                            break
                    
                    # Second priority: case-insensitive exact matches
                    if not found_path:
                        for file in files:
                            if file.lower() in [v.lower() for v in exact_variations]:
                                if file.endswith(('.ttf', '.ttc', '.otf')):
                                    found_path = os.path.join(root, file)
                                    break
                    
                    # Last resort: partial match but exclude problematic fonts
                    if not found_path:
                        for file in files:
                            if (font_name.lower() in file.lower() and 
                                file.endswith(('.ttf', '.ttc', '.otf')) and
                                not any(exclude in file.lower() for exclude in ['hb', 'hebrew', 'symbol', 'wingding'])):
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
        """Validate that font is working correctly and can render English text."""
        try:
            # Test basic English characters
            test_texts = ["Test", "ABC", "abc", "123"]
            
            for test_text in test_texts:
                bbox = font.getbbox(test_text)
                if bbox:
                    height = bbox[3] - bbox[1]
                    width = bbox[2] - bbox[0]
                    
                    # Check if font height is reasonable (at least 30% of requested size)
                    if height < font_size * 0.3:
                        logger.warning(f"Font {font_name} renders too small: {height}px vs requested {font_size}px")
                        return False
                    
                    # Check if font width is reasonable (boxes usually have 0 or very small width)
                    if width < len(test_text) * (font_size * 0.2):
                        logger.warning(f"Font {font_name} may render as boxes - width too small: {width}px for '{test_text}'")
                        return False
                        
                    # Additional validation: check if font appears to be rendering actual glyphs
                    # Different characters should have different widths (unless it's a monospace font)
                    if test_text == "Test":
                        i_bbox = font.getbbox("i")
                        m_bbox = font.getbbox("m") 
                        if i_bbox and m_bbox:
                            i_width = i_bbox[2] - i_bbox[0]
                            m_width = m_bbox[2] - m_bbox[0]
                            # 'm' should typically be wider than 'i', unless monospace
                            if i_width == m_width and i_width < font_size * 0.3:
                                logger.warning(f"Font {font_name} may not be rendering glyphs correctly")
                                return False
                else:
                    logger.warning(f"Font {font_name} failed to get bbox for '{test_text}'")
                    return False
                    
            return True
        except Exception as e:
            logger.error(f"Font validation failed for {font_name}: {e}")
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
    
    @classmethod
    def get_explicit_font_path(cls, font_name: str, font_weight: str) -> str:
        """Get explicit system font paths for common fonts to avoid fallback issues."""
        # Explicit font path mappings for macOS system fonts
        explicit_paths = {
            "Arial": {
                "normal": "/System/Library/Fonts/Arial.ttf",
                "bold": "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "italic": "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
                "bold_italic": "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf"
            },
            "Helvetica": {
                "normal": "/System/Library/Fonts/Helvetica.ttc",
                "bold": "/System/Library/Fonts/Helvetica.ttc",  # Contains multiple weights
                "italic": "/System/Library/Fonts/Helvetica.ttc",
                "bold_italic": "/System/Library/Fonts/Helvetica.ttc"
            },
            "Times New Roman": {
                "normal": "/System/Library/Fonts/Times New Roman.ttf",
                "bold": "/System/Library/Fonts/Times New Roman Bold.ttf", 
                "italic": "/System/Library/Fonts/Times New Roman Italic.ttf",
                "bold_italic": "/System/Library/Fonts/Times New Roman Bold Italic.ttf"
            },
            "Courier New": {
                "normal": "/System/Library/Fonts/Courier New.ttf",
                "bold": "/System/Library/Fonts/Courier New Bold.ttf",
                "italic": "/System/Library/Fonts/Courier New Italic.ttf", 
                "bold_italic": "/System/Library/Fonts/Courier New Bold Italic.ttf"
            }
        }
        
        if font_name in explicit_paths and font_weight in explicit_paths[font_name]:
            path = explicit_paths[font_name][font_weight]
            if os.path.exists(path):
                return path
        
        return None
    
    @classmethod
    def get_font_with_style(cls, font_name: str, font_size: int, font_weight: str, font_file: str = "") -> ImageFont:
        """Get font with specific weight/style."""
        # Try custom font file first
        if font_file:
            return cls.get_font(font_name, font_size, font_file)
        
        # Try explicit system font paths first
        explicit_path = cls.get_explicit_font_path(font_name, font_weight)
        if explicit_path:
            try:
                font = ImageFont.truetype(explicit_path, font_size)
                if cls.validate_font(font, font_name, font_size):
                    logger.info(f"Using explicit font path: {explicit_path}")
                    return font
            except Exception as e:
                logger.warning(f"Failed to load explicit font path {explicit_path}: {e}")
        
        # Map font weights to actual font names - prioritize base font name for normal
        weight_variants = {
            "normal": [font_name, f"{font_name} Regular", f"{font_name}-Regular", f"{font_name}Regular"],
            "bold": [f"{font_name} Bold", f"{font_name}-Bold", f"{font_name}Bold", f"{font_name} Heavy", f"{font_name}-Heavy"],
            "italic": [f"{font_name} Italic", f"{font_name}-Italic", f"{font_name}Italic", f"{font_name} Oblique"],
            "bold_italic": [f"{font_name} Bold Italic", f"{font_name}-BoldItalic", f"{font_name}BoldItalic", f"{font_name} Heavy Italic"]
        }
        
        # Try specific weight variant first
        if font_weight in weight_variants:
            for variant in weight_variants[font_weight]:
                font = cls.get_font(variant, font_size)
                if font and cls.validate_font(font, variant, font_size):
                    return font
        
        # Better fallback system - try alternative strategies
        if font_weight == "bold":
            # For bold, try alternative heavy fonts or use synthetic bold
            alternative_names = [f"{font_name} Heavy", f"{font_name} Black", f"{font_name} Semibold"]
            for alt_name in alternative_names:
                alt_font = cls.get_font(alt_name, font_size)
                if alt_font and cls.validate_font(alt_font, alt_name, font_size):
                    logger.info(f"Using alternative bold font: {alt_name}")
                    return alt_font
        
        # Final fallback to normal font
        font = cls.get_font(font_name, font_size)
        if font_weight != "normal":
            logger.warning(f"Font weight '{font_weight}' not found for {font_name}, using normal. Consider using synthetic bold rendering.")
        
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
    def create_gradient_image(width: int, height: int, color1: Tuple[int, int, int, int], 
                            color2: Tuple[int, int, int, int], direction: str) -> Image:
        """Create a gradient image."""
        gradient = Image.new('RGBA', (width, height))
        
        for i in range(width if direction == "horizontal" else height):
            if direction == "horizontal":
                ratio = i / width
                for j in range(height):
                    color = tuple(int(color1[k] + (color2[k] - color1[k]) * ratio) for k in range(4))
                    gradient.putpixel((i, j), color)
            elif direction == "vertical":
                ratio = i / height
                for j in range(width):
                    color = tuple(int(color1[k] + (color2[k] - color1[k]) * ratio) for k in range(4))
                    gradient.putpixel((j, i), color)
            else:  # diagonal
                for j in range(width):
                    diag_ratio = (j + i) / (width + height - 2)
                    color = tuple(int(color1[k] + (color2[k] - color1[k]) * diag_ratio) for k in range(4))
                    gradient.putpixel((j, i), color)
        
        return gradient
    
    def draw_text_with_effects(self, draw, text, font, x, y, base_color, 
                              border_width=0, border_color=(0,0,0,255),
                              shadow_enabled=False, shadow_offset_x=2, shadow_offset_y=2,
                              shadow_color=(0,0,0,255), shadow_blur=2):
        """Draw text with border and shadow effects."""
        
        # Draw shadow first (if enabled)
        if shadow_enabled:
            shadow_x = x + shadow_offset_x
            shadow_y = y + shadow_offset_y
            
            if shadow_blur > 0:
                # Create shadow with blur
                bbox = font.getbbox(text) if hasattr(font, 'getbbox') else None
                if bbox:
                    shadow_width = int(bbox[2] - bbox[0] + shadow_blur * 2)
                    shadow_height = int(bbox[3] - bbox[1] + shadow_blur * 2)
                    shadow_img = Image.new('RGBA', (shadow_width, shadow_height), (0, 0, 0, 0))
                    shadow_draw = ImageDraw.Draw(shadow_img)
                    shadow_draw.text((shadow_blur, shadow_blur), text, font=font, fill=shadow_color)
                    
                    # Apply blur
                    if shadow_blur > 0:
                        shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
                    
                    # Composite shadow
                    draw._image.paste(shadow_img, (int(shadow_x - shadow_blur), int(shadow_y - shadow_blur)), shadow_img)
            else:
                # Simple shadow without blur
                draw.text((shadow_x, shadow_y), text, font=font, fill=shadow_color)
        
        # Draw border/stroke (if enabled)
        if border_width > 0:
            # Draw text outline by drawing text multiple times around the main position
            for dx in range(-border_width, border_width + 1):
                for dy in range(-border_width, border_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=border_color)
        
        # Draw main text
        draw.text((x, y), text, font=font, fill=base_color)
    
    def draw_text_background(self, draw, text, font, x, y, bg_color, padding):
        """Draw text background highlight."""
        bbox = font.getbbox(text) if hasattr(font, 'getbbox') else None
        if bbox:
            # Calculate background rectangle
            bg_x1 = x + bbox[0] - padding
            bg_y1 = y + bbox[1] - padding
            bg_x2 = x + bbox[2] + padding
            bg_y2 = y + bbox[3] + padding
            
            # Draw background rectangle
            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=bg_color)
    
    def draw_container_box(self, draw, lines, font, start_y, line_height, output_width, output_height, 
                          text_align, margin_x, margin_y, container_color, container_width, container_padding, container_border_color):
        """Draw container box around all text."""
        if not lines:
            return
            
        # Calculate total text bounds
        min_x = output_width
        max_x = 0
        min_y = start_y
        max_y = start_y + len(lines) * line_height
        
        y = start_y
        for line in lines:
            bbox = font.getbbox(line) if hasattr(font, 'getbbox') else None
            if bbox:
                line_width = bbox[2] - bbox[0]
                x_offset = -bbox[0] if bbox[0] < 0 else 0
            else:
                line_width = len(line) * (font.size * 0.6)
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
            
            line_min_x = x + x_offset
            line_max_x = x + x_offset + line_width
            
            min_x = min(min_x, line_min_x)
            max_x = max(max_x, line_max_x)
            y += line_height
        
        # Add container padding
        container_x1 = min_x - container_padding
        container_y1 = min_y - container_padding
        container_x2 = max_x + container_padding  
        container_y2 = max_y + container_padding
        
        # Ensure container stays within image bounds
        container_x1 = max(0, container_x1)
        container_y1 = max(0, container_y1)
        container_x2 = min(output_width, container_x2)
        container_y2 = min(output_height, container_y2)
        
        # Draw container background
        draw.rectangle([container_x1, container_y1, container_x2, container_y2], 
                      fill=container_color)
        
        # Draw container border if width > 0
        if container_width > 0:
            for i in range(container_width):
                draw.rectangle([container_x1 + i, container_y1 + i, 
                              container_x2 - i, container_y2 - i], 
                              outline=container_border_color, width=1)
    
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
                     font_file="", time_display=0.0, fade_in=0.0, fade_out=0.0,
                     font_weight="normal", text_border_width=0, text_border_color="#000000",
                     shadow_enabled=False, shadow_offset_x=2, shadow_offset_y=2,
                     shadow_color="#000000", shadow_blur=2, text_bg_enabled=False,
                     text_bg_color="#FFFF00", text_bg_padding=5, gradient_enabled=False,
                     gradient_color2="#FF0000", gradient_direction="vertical",
                     container_enabled=False, container_color="#333333", 
                     container_width=2, container_padding=15, container_border_color="#FFFFFF",
                     container_fill=True):
        
        # Parse colors
        text_color = self.parse_color(font_color)
        bg_color = self.parse_color(background_color)
        
        # Apply base opacity to text color
        text_color = (text_color[0], text_color[1], text_color[2], int(text_color[3] * base_opacity))
        
        # Create image with alpha channel
        image = Image.new('RGBA', (output_width, output_height), bg_color)
        draw = ImageDraw.Draw(image)
        
        # Get font
        # Parse additional colors
        border_color = self.parse_color(text_border_color)
        shadow_color_parsed = self.parse_color(shadow_color)
        text_bg_color_parsed = self.parse_color(text_bg_color)
        gradient_color2_parsed = self.parse_color(gradient_color2)
        container_color_parsed = self.parse_color(container_color)
        container_border_color_parsed = self.parse_color(container_border_color)
        
        font = self.get_font_with_style(font_name, font_size, font_weight, font_file)
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
                test_font = self.get_font_with_style(font_name, test_size, font_weight, font_file)
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
        
        # Draw container box if enabled (before text)
        start_y = y
        if container_enabled:
            self.draw_container_box(draw, lines, font, start_y, line_height, 
                                  output_width, output_height, text_align, 
                                  margin_x, margin_y, container_color_parsed, 
                                  container_width, container_padding, container_border_color_parsed)
        
        # Handle gradient text color
        if gradient_enabled:
            # Create gradient mask
            gradient_img = self.create_gradient_image(output_width, output_height, 
                                                   text_color, gradient_color2_parsed, gradient_direction)
            # We'll apply this later by masking
            
        # Draw each line with improved positioning and effects
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
            
            # Draw text background if enabled
            if text_bg_enabled:
                self.draw_text_background(draw, line, font, x + x_offset, y, 
                                        text_bg_color_parsed, text_bg_padding)
            
            # Choose text color (gradient or solid)
            final_text_color = text_color
            
            # Apply letter spacing if needed
            if letter_spacing != 0:
                # Draw each character separately with spacing and effects
                char_x = x + x_offset
                for char in line:
                    self.draw_text_with_effects(draw, char, font, char_x, y, final_text_color,
                                              text_border_width, border_color,
                                              shadow_enabled, shadow_offset_x, shadow_offset_y,
                                              shadow_color_parsed, shadow_blur)
                    char_bbox = font.getbbox(char) if hasattr(font, 'getbbox') else None
                    char_width = char_bbox[2] - char_bbox[0] if char_bbox else font_size * 0.6
                    char_x += char_width + letter_spacing
            else:
                # Draw the entire line with effects
                self.draw_text_with_effects(draw, line, font, x + x_offset, y, final_text_color,
                                          text_border_width, border_color,
                                          shadow_enabled, shadow_offset_x, shadow_offset_y,
                                          shadow_color_parsed, shadow_blur)
            
            y += line_height
        
        # Apply gradient text if enabled
        if gradient_enabled:
            # Create a text mask
            text_mask = Image.new('L', (output_width, output_height), 0)
            mask_draw = ImageDraw.Draw(text_mask)
            
            # Draw text on mask
            y = (output_height - total_height) // 2 if vertical_align == "middle" else margin_y
            for line in lines:
                bbox = font.getbbox(line) if hasattr(font, 'getbbox') else None
                line_width = bbox[2] - bbox[0] if bbox else len(line) * (font_size * 0.6)
                x_offset = -bbox[0] if bbox and bbox[0] < 0 else 0
                
                if text_align == "center":
                    x = (output_width - line_width) // 2
                elif text_align == "right":
                    x = output_width - margin_x - line_width
                else:
                    x = margin_x
                
                mask_draw.text((x + x_offset, y), line, font=font, fill=255)
                y += line_height
            
            # Apply gradient using mask
            gradient_img = self.create_gradient_image(output_width, output_height, 
                                                   text_color, gradient_color2_parsed, gradient_direction)
            
            # Clear existing text and apply gradient
            image = Image.new('RGBA', (output_width, output_height), bg_color)
            image = Image.composite(gradient_img, image, text_mask)
        
        # Create alpha mask from the alpha channel
        alpha_mask = image.split()[3]
        
        # Convert to tensor format (B, H, W, C)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        
        # Create transparent text version (no background, only text with alpha)
        transparent_image = Image.new('RGBA', (output_width, output_height), (0, 0, 0, 0))
        if hasattr(image, 'split') and len(image.split()) == 4:
            # Extract RGB channels from original and alpha channel
            r, g, b, a = image.split()
            # Combine with fully transparent background
            transparent_image = Image.merge('RGBA', (r, g, b, a))
            # Make background pixels fully transparent
            transparent_data = list(transparent_image.getdata())
            bg_r, bg_g, bg_b = self.parse_color(background_color)[:3]
            # Convert background pixels to transparent
            for i, (r, g, b, a) in enumerate(transparent_data):
                # If pixel is close to background color, make it transparent
                if abs(r - bg_r) < 10 and abs(g - bg_g) < 10 and abs(b - bg_b) < 10:
                    transparent_data[i] = (r, g, b, 0)
            transparent_image.putdata(transparent_data)
        else:
            transparent_image = image.copy()
        
        transparent_np = np.array(transparent_image).astype(np.float32) / 255.0
        transparent_tensor = torch.from_numpy(transparent_np).unsqueeze(0)
        
        # Create mask tensor
        mask_np = np.array(alpha_mask).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        
        # Create text configuration
        text_config = {
            "text": text,
            "font_name": font_name,
            "font_size": font_size,
            "font_weight": font_weight,
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
            "text_border_width": text_border_width,
            "text_border_color": text_border_color,
            "shadow_enabled": shadow_enabled,
            "text_bg_enabled": text_bg_enabled,
            "gradient_enabled": gradient_enabled,
            "container_enabled": container_enabled,
            "container_color": container_color,
            "container_width": container_width,
            "container_padding": container_padding,
            "container_border_color": container_border_color,
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
        
        # Create styling settings for subtitle engine
        styling_settings = {
            "font_config": {
                "font_name": font_name,
                "font_size": font_size,
                "font_weight": font_weight,
                "font_color": font_color,
                "font_file": font_file
            },
            "layout_config": {
                "text_align": text_align,
                "vertical_align": vertical_align,
                "margin_x": margin_x,
                "margin_y": margin_y,
                "line_spacing": line_spacing,
                "letter_spacing": letter_spacing,
                "words_per_line": words_per_line,
                "max_lines": max_lines,
                "auto_size": auto_size
            },
            "effects_config": {
                "text_border_width": text_border_width,
                "text_border_color": text_border_color,
                "shadow_enabled": shadow_enabled,
                "shadow_offset_x": shadow_offset_x,
                "shadow_offset_y": shadow_offset_y,
                "shadow_color": shadow_color,
                "shadow_blur": shadow_blur,
                "text_bg_enabled": text_bg_enabled,
                "text_bg_color": text_bg_color,
                "text_bg_padding": text_bg_padding,
                "gradient_enabled": gradient_enabled,
                "gradient_color2": gradient_color2,
                "gradient_direction": gradient_direction
            },
            "container_config": {
                "container_enabled": container_enabled,
                "container_color": container_color,
                "container_width": container_width,
                "container_padding": container_padding,
                "container_border_color": container_border_color,
                "container_fill": container_fill
            },
            "output_config": {
                "output_width": output_width,
                "output_height": output_height,
                "base_opacity": base_opacity,
                "background_color": background_color
            },
            "timing_config": {
                "time_display": time_display,
                "fade_in": fade_in,
                "fade_out": fade_out
            }
        }
        
        return (image_tensor, transparent_tensor, mask_tensor, json.dumps(text_config), render_info, styling_settings)


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