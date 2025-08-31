import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from typing import Dict, List, Tuple, Optional, Union
from .utils import logger

class RajTextPresets:
    """
    Industry template node providing one-click professional text presets
    for YouTube, broadcast, cinema, corporate, and social media content.
    """
    
    # Template categories and their presets
    PRESET_CATEGORIES = {
        "youtube": [
            "subscribe_button", "like_bell_reminder", "end_screen_text",
            "channel_name", "video_title", "sponsor_message", "hashtag_display",
            "comment_highlight", "poll_question", "countdown_timer"
        ],
        "broadcast": [
            "news_ticker", "breaking_news", "weather_display", "sports_score",
            "stock_ticker", "election_results", "live_indicator", "location_stamp",
            "reporter_name", "interview_subject"
        ],
        "cinema": [
            "opening_titles", "end_credits", "chapter_title", "location_title",
            "time_stamp", "character_name", "flashback_indicator", "dream_sequence",
            "voice_over_text", "subtitle_style"
        ],
        "corporate": [
            "logo_reveal", "product_name", "price_tag", "feature_highlight",
            "call_to_action", "contact_info", "disclaimer", "testimonial",
            "company_motto", "achievement_badge"
        ],
        "social_media": [
            "instagram_story", "tiktok_caption", "twitter_quote", "facebook_post",
            "linkedin_update", "snapchat_text", "pinterest_overlay", "youtube_short",
            "quote_card", "meme_text"
        ],
        "gaming": [
            "game_title", "player_name", "score_display", "level_indicator",
            "achievement_unlock", "game_over", "high_score", "multiplayer_tag",
            "stream_overlay", "donation_alert"
        ]
    }
    
    # Color schemes
    COLOR_SCHEMES = {
        "default": {"primary": "#FFFFFF", "secondary": "#000000", "accent": "#FF0000"},
        "dark": {"primary": "#FFFFFF", "secondary": "#1A1A1A", "accent": "#00FF00"},
        "light": {"primary": "#000000", "secondary": "#FFFFFF", "accent": "#0066FF"},
        "neon": {"primary": "#00FFFF", "secondary": "#FF00FF", "accent": "#FFFF00"},
        "pastel": {"primary": "#FFB6C1", "secondary": "#E6E6FA", "accent": "#98FB98"},
        "corporate": {"primary": "#333333", "secondary": "#F0F0F0", "accent": "#0066CC"},
        "youtube": {"primary": "#FFFFFF", "secondary": "#FF0000", "accent": "#000000"},
        "instagram": {"primary": "#FFFFFF", "secondary": "#E1306C", "accent": "#405DE6"},
        "tiktok": {"primary": "#FFFFFF", "secondary": "#000000", "accent": "#FF0050"},
        "gaming": {"primary": "#00FF00", "secondary": "#000000", "accent": "#FF6600"}
    }
    
    # Size presets
    SIZE_PRESETS = {
        "small": {"width": 256, "height": 128, "font_size": 24},
        "medium": {"width": 512, "height": 256, "font_size": 48},
        "large": {"width": 1024, "height": 512, "font_size": 96},
        "banner": {"width": 1920, "height": 200, "font_size": 72},
        "square": {"width": 512, "height": 512, "font_size": 64},
        "story": {"width": 1080, "height": 1920, "font_size": 80},
        "thumbnail": {"width": 1280, "height": 720, "font_size": 88}
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available presets for first category
        first_category = list(cls.PRESET_CATEGORIES.keys())[0]
        
        return {
            "required": {
                "preset_category": (list(cls.PRESET_CATEGORIES.keys()), {
                    "default": first_category,
                    "tooltip": "Preset category"
                }),
                "preset_type": (cls.PRESET_CATEGORIES[first_category], {
                    "default": cls.PRESET_CATEGORIES[first_category][0],
                    "tooltip": "Specific preset template"
                }),
                "custom_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Custom text to replace template text"
                }),
                "color_scheme": (list(cls.COLOR_SCHEMES.keys()), {
                    "default": "default",
                    "tooltip": "Color scheme"
                }),
                "size_preset": (list(cls.SIZE_PRESETS.keys()) + ["custom"], {
                    "default": "medium",
                    "tooltip": "Size preset"
                }),
                "animation_preset": (["none", "fade_in", "slide_in", "bounce", "glow", "typewriter", "pulse"], {
                    "default": "none",
                    "tooltip": "Animation preset"
                }),
            },
            "optional": {
                "primary_color": ("STRING", {
                    "default": "#FFFFFF",
                    "tooltip": "Primary text color (overrides scheme)"
                }),
                "secondary_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "Secondary/background color"
                }),
                "accent_color": ("STRING", {
                    "default": "#FF0000",
                    "tooltip": "Accent color for highlights"
                }),
                "custom_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Custom width (when size_preset is 'custom')"
                }),
                "custom_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Custom height (when size_preset is 'custom')"
                }),
                "custom_font_size": ("INT", {
                    "default": 48,
                    "min": 8,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Custom font size"
                }),
                "logo_image": ("IMAGE", {
                    "tooltip": "Logo image for templates that support it"
                }),
                "background_image": ("IMAGE", {
                    "tooltip": "Background image for templates"
                }),
                "template_variables": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "Template variables as JSON (e.g., {\"name\": \"John\", \"score\": \"100\"})"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("preset_text", "text_config", "animation_data", "preset_info")
    FUNCTION = "create_preset"
    CATEGORY = "Raj Video Processing 🎬/Text"
    
    @staticmethod
    def get_preset_config(category: str, preset_type: str) -> Dict:
        """Get configuration for specific preset."""
        configs = {
            # YouTube presets
            ("youtube", "subscribe_button"): {
                "text": "SUBSCRIBE",
                "font": "Arial Black",
                "effects": ["shadow", "glow"],
                "position": "bottom_right",
                "shape": "rounded_rectangle",
                "icon": "bell"
            },
            ("youtube", "like_bell_reminder"): {
                "text": "👍 LIKE & 🔔 SUBSCRIBE",
                "font": "Arial",
                "effects": ["pulse"],
                "position": "bottom_center",
                "shape": "banner"
            },
            ("youtube", "end_screen_text"): {
                "text": "Thanks for Watching!",
                "font": "Open Sans",
                "effects": ["fade_in"],
                "position": "center",
                "shape": "none"
            },
            
            # Broadcast presets
            ("broadcast", "news_ticker"): {
                "text": "BREAKING NEWS",
                "font": "Arial Bold",
                "effects": ["scroll"],
                "position": "bottom",
                "shape": "banner",
                "background_opacity": 0.9
            },
            ("broadcast", "breaking_news"): {
                "text": "🚨 BREAKING NEWS 🚨",
                "font": "Impact",
                "effects": ["flash"],
                "position": "top_center",
                "shape": "alert_banner"
            },
            
            # Cinema presets
            ("cinema", "opening_titles"): {
                "text": "MOVIE TITLE",
                "font": "Cinzel",
                "effects": ["fade_in", "typewriter"],
                "position": "center",
                "shape": "none",
                "background": "black"
            },
            ("cinema", "end_credits"): {
                "text": "Directed by...",
                "font": "Times New Roman",
                "effects": ["scroll_up"],
                "position": "center",
                "shape": "none"
            },
            
            # Corporate presets
            ("corporate", "logo_reveal"): {
                "text": "Company Name",
                "font": "Helvetica",
                "effects": ["scale_in"],
                "position": "center",
                "shape": "none",
                "professional": True
            },
            ("corporate", "call_to_action"): {
                "text": "Call Now: 1-800-XXX-XXXX",
                "font": "Arial",
                "effects": ["pulse"],
                "position": "bottom_center",
                "shape": "call_out"
            },
            
            # Social Media presets
            ("social_media", "instagram_story"): {
                "text": "Your Story Text Here",
                "font": "Helvetica Neue",
                "effects": ["bounce"],
                "position": "middle",
                "shape": "story_card"
            },
            ("social_media", "quote_card"): {
                "text": "\"Inspirational Quote Here\"",
                "font": "Georgia",
                "effects": ["fade_in"],
                "position": "center",
                "shape": "quote_box"
            },
            
            # Gaming presets
            ("gaming", "achievement_unlock"): {
                "text": "🏆 ACHIEVEMENT UNLOCKED!",
                "font": "Orbitron",
                "effects": ["glow", "bounce"],
                "position": "center",
                "shape": "achievement_badge"
            },
            ("gaming", "game_over"): {
                "text": "GAME OVER",
                "font": "Press Start 2P",
                "effects": ["glitch"],
                "position": "center",
                "shape": "none"
            }
        }
        
        return configs.get((category, preset_type), {
            "text": "Default Text",
            "font": "Arial",
            "effects": [],
            "position": "center",
            "shape": "none"
        })
    
    @staticmethod
    def parse_color(color_str: str) -> Tuple[int, int, int, int]:
        """Parse color string to RGBA tuple."""
        if color_str.startswith("#"):
            color_str = color_str[1:]
        
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
        
        return (255, 255, 255, 255)
    
    @staticmethod
    def create_shape_background(width: int, height: int, shape: str, 
                               color: Tuple[int, int, int, int]) -> Image.Image:
        """Create shaped background for text."""
        bg = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(bg)
        
        margin = 20
        
        if shape == "rounded_rectangle":
            # Rounded rectangle
            x1, y1 = margin, margin
            x2, y2 = width - margin, height - margin
            radius = 15
            
            # Draw rounded rectangle (simplified)
            draw.rectangle([x1, y1, x2, y2], fill=color)
            
        elif shape == "banner":
            # Full width banner
            draw.rectangle([0, height//4, width, 3*height//4], fill=color)
            
        elif shape == "alert_banner":
            # Alert banner with warning stripes
            draw.rectangle([0, height//3, width, 2*height//3], fill=color)
            # Add warning stripes
            stripe_color = (255, 255, 0, color[3])
            for x in range(0, width, 40):
                draw.rectangle([x, height//3, x+20, 2*height//3], fill=stripe_color)
            
        elif shape == "call_out":
            # Speech bubble style
            bubble_height = height - 40
            draw.ellipse([margin, 20, width-margin, bubble_height], fill=color)
            # Pointer
            draw.polygon([(width//2-20, bubble_height), (width//2, height-10), 
                         (width//2+20, bubble_height)], fill=color)
            
        elif shape == "story_card":
            # Instagram story style card
            draw.rounded_rectangle([margin, margin, width-margin, height-margin], 
                                 radius=20, fill=color)
            
        elif shape == "quote_box":
            # Quote box with quotation marks
            draw.rectangle([margin, margin, width-margin, height-margin], fill=color)
            # Add quotation marks (simplified)
            quote_font = ImageFont.load_default()
            draw.text((margin+10, margin+10), '"', fill=(255, 255, 255, 255), font=quote_font)
            draw.text((width-margin-20, height-margin-30), '"', fill=(255, 255, 255, 255), font=quote_font)
            
        elif shape == "achievement_badge":
            # Gaming achievement badge
            center_x, center_y = width//2, height//2
            radius = min(width, height)//2 - 10
            draw.ellipse([center_x-radius, center_y-radius, 
                         center_x+radius, center_y+radius], fill=color)
            # Inner circle
            inner_radius = radius - 15
            inner_color = (color[0]//2, color[1]//2, color[2]//2, color[3])
            draw.ellipse([center_x-inner_radius, center_y-inner_radius,
                         center_x+inner_radius, center_y+inner_radius], fill=inner_color)
        
        return bg
    
    def create_preset(self, preset_category, preset_type, custom_text, color_scheme,
                     size_preset, animation_preset, primary_color="#FFFFFF",
                     secondary_color="#000000", accent_color="#FF0000",
                     custom_width=512, custom_height=512, custom_font_size=48,
                     logo_image=None, background_image=None, template_variables="{}"):
        
        # Get preset configuration
        preset_config = self.get_preset_config(preset_category, preset_type)
        
        # Parse template variables
        try:
            variables = json.loads(template_variables) if template_variables else {}
        except:
            variables = {}
        
        # Determine text content
        if custom_text.strip():
            text_content = custom_text
        else:
            text_content = preset_config.get("text", "Sample Text")
        
        # Apply template variables
        for key, value in variables.items():
            text_content = text_content.replace(f"{{{key}}}", str(value))
        
        # Get colors
        if color_scheme != "default":
            scheme_colors = self.COLOR_SCHEMES.get(color_scheme, self.COLOR_SCHEMES["default"])
            if primary_color == "#FFFFFF":  # If using default, use scheme
                primary_color = scheme_colors["primary"]
            if secondary_color == "#000000":
                secondary_color = scheme_colors["secondary"]
            if accent_color == "#FF0000":
                accent_color = scheme_colors["accent"]
        
        # Parse colors
        primary_rgba = self.parse_color(primary_color)
        secondary_rgba = self.parse_color(secondary_color)
        accent_rgba = self.parse_color(accent_color)
        
        # Get dimensions
        if size_preset != "custom":
            size_config = self.SIZE_PRESETS[size_preset]
            width = size_config["width"]
            height = size_config["height"]
            font_size = size_config["font_size"]
        else:
            width = custom_width
            height = custom_height
            font_size = custom_font_size
        
        # Create base image
        if background_image is not None:
            # Use provided background
            bg_tensor = background_image[0].cpu().numpy()
            bg_array = (bg_tensor * 255).astype(np.uint8)
            base_image = Image.fromarray(bg_array[:, :, :3], 'RGB').convert('RGBA')
            base_image = base_image.resize((width, height), Image.Resampling.LANCZOS)
        else:
            # Create background based on shape
            base_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            
            # Add shape background if specified
            shape = preset_config.get("shape", "none")
            if shape != "none":
                shape_bg = self.create_shape_background(width, height, shape, secondary_rgba)
                base_image = Image.alpha_composite(base_image, shape_bg)
        
        # Create text overlay
        draw = ImageDraw.Draw(base_image)
        
        # Try to load specified font
        font_name = preset_config.get("font", "Arial")
        try:
            font = ImageFont.truetype(font_name, font_size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Calculate text position
        if font:
            bbox = draw.textbbox((0, 0), text_content, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width = len(text_content) * font_size // 2
            text_height = font_size
        
        position = preset_config.get("position", "center")
        
        if position == "center":
            x = (width - text_width) // 2
            y = (height - text_height) // 2
        elif position == "top_center":
            x = (width - text_width) // 2
            y = height // 8
        elif position == "bottom_center":
            x = (width - text_width) // 2
            y = height - height // 4
        elif position == "bottom_right":
            x = width - text_width - 20
            y = height - text_height - 20
        else:
            x = (width - text_width) // 2
            y = (height - text_height) // 2
        
        # Apply effects (simplified versions)
        effects = preset_config.get("effects", [])
        
        if "shadow" in effects:
            # Draw shadow
            shadow_offset = 3
            draw.text((x + shadow_offset, y + shadow_offset), text_content, 
                     fill=(0, 0, 0, 128), font=font)
        
        # Draw main text
        draw.text((x, y), text_content, fill=primary_rgba, font=font)
        
        # Add logo if provided and preset supports it
        if logo_image is not None and "logo" in preset_config.get("features", []):
            logo_tensor = logo_image[0].cpu().numpy()
            logo_array = (logo_tensor * 255).astype(np.uint8)
            logo_img = Image.fromarray(logo_array[:, :, :3], 'RGB')
            logo_size = min(width, height) // 4
            logo_img = logo_img.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
            base_image.paste(logo_img, (width - logo_size - 20, 20))
        
        # Convert to tensor
        image_array = np.array(base_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        
        # Create text config
        text_config = {
            "preset_category": preset_category,
            "preset_type": preset_type,
            "text": text_content,
            "font": font_name,
            "font_size": font_size,
            "primary_color": primary_color,
            "secondary_color": secondary_color,
            "accent_color": accent_color,
            "width": width,
            "height": height,
            "effects": effects,
            "shape": preset_config.get("shape", "none"),
            "position": position
        }
        
        # Create animation data
        animation_data = {
            "preset": animation_preset,
            "effects": effects,
            "duration": 2.0,
            "easing": "ease_in_out"
        }
        
        # Create preset info
        preset_info = (
            f"Preset Applied: {preset_category.title()} - {preset_type.replace('_', ' ').title()}\n"
            f"Text: {text_content[:50]}{'...' if len(text_content) > 50 else ''}\n"
            f"Size: {width}x{height}\n"
            f"Font: {font_name} @ {font_size}px\n"
            f"Color Scheme: {color_scheme}\n"
            f"Effects: {', '.join(effects) if effects else 'None'}\n"
            f"Animation: {animation_preset}"
        )
        
        return (image_tensor, json.dumps(text_config), json.dumps(animation_data), preset_info)


# Test function
if __name__ == "__main__":
    node = RajTextPresets()
    print("Text Presets node initialized")
    print(f"Categories: {list(node.PRESET_CATEGORIES.keys())}")
    print(f"YouTube presets: {node.PRESET_CATEGORIES['youtube']}")
    print(f"Color schemes: {list(node.COLOR_SCHEMES.keys())}")