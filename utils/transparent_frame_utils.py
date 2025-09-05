"""
Transparent Frame Utilities for Subtitle Engine
Handles transparent RGBA frame generation with highlighting support
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from .logger import logger
from .subtitle_utils import get_current_highlighted_word


def render_transparent_text_with_settings(text_generator, text: str, settings: Dict, 
                                        output_width: int = None, output_height: int = None, 
                                        frame_padding_left: int = 20, frame_padding_right: int = 20, 
                                        frame_padding_top: int = 20, frame_padding_bottom: int = 20, 
                                        line_gaps: float = 1.2) -> np.ndarray:
    """Render text with transparent background, ensuring RGBA format."""
    try:
        # Create fully transparent background settings
        transparent_settings = settings.copy()
        if "output_config" not in transparent_settings:
            transparent_settings["output_config"] = {}
        transparent_settings["output_config"]["background_color"] = "transparent"
        
        # Extract all parameters from settings
        font_config = transparent_settings.get("font_config", {})
        layout_config = transparent_settings.get("layout_config", {})
        effects_config = transparent_settings.get("effects_config", {})
        container_config = transparent_settings.get("container_config", {})
        output_config = transparent_settings.get("output_config", {})
        
        # Use explicit dimensions if provided, otherwise fall back to settings
        final_width = output_width if output_width is not None else output_config.get("output_width", 512)
        final_height = output_height if output_height is not None else output_config.get("output_height", 256)
        
        # Call text generator with transparent background
        result = text_generator.generate_text(
            text=text,
            output_width=final_width,
            output_height=final_height,
            font_name=font_config.get("font_name", "Arial"),
            font_size=font_config.get("font_size", 24),
            font_color=font_config.get("font_color", "#FFFFFF"),
            background_color="transparent",  # Force transparent background
            text_align=layout_config.get("text_align", "center"),
            vertical_align=layout_config.get("vertical_align", "middle"),
            words_per_line=layout_config.get("words_per_line", 0),
            max_lines=layout_config.get("max_lines", 0),
            line_spacing=line_gaps,
            letter_spacing=layout_config.get("letter_spacing", 0),
            margin_x=frame_padding_left,
            margin_y=frame_padding_top,
            auto_size=layout_config.get("auto_size", False),
            base_opacity=output_config.get("base_opacity", 1.0),
            font_file=font_config.get("font_file", ""),
            font_weight=font_config.get("font_weight", "normal"),
            text_border_width=effects_config.get("text_border_width", 0),
            text_border_color=effects_config.get("text_border_color", "#000000"),
            shadow_enabled=effects_config.get("shadow_enabled", False),
            shadow_offset_x=effects_config.get("shadow_offset_x", 2),
            shadow_offset_y=effects_config.get("shadow_offset_y", 2),
            shadow_color=effects_config.get("shadow_color", "#000000"),
            shadow_blur=effects_config.get("shadow_blur", 2),
            text_bg_enabled=effects_config.get("text_bg_enabled", False),
            text_bg_color=effects_config.get("text_bg_color", "#FFFF00"),
            text_bg_padding=effects_config.get("text_bg_padding", 5),
            gradient_enabled=effects_config.get("gradient_enabled", False),
            gradient_color2=effects_config.get("gradient_color2", "#FF0000"),
            gradient_direction=effects_config.get("gradient_direction", "vertical"),
            container_enabled=container_config.get("container_enabled", False),
            container_color=container_config.get("container_color", "#333333"),
            container_width=container_config.get("container_width", 2),
            container_padding=container_config.get("container_padding", 15),
            container_border_color=container_config.get("container_border_color", "#FFFFFF"),
            container_fill=container_config.get("container_fill", True)
        )
        
        # Extract image tensor and convert to numpy array
        image_tensor = result[0]  # First return value is the image
        image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
        
        # Debug: Log what we got
        logger.info(f"Transparent text generator returned shape: {image_array.shape}")
        
        # Verify it's RGBA format
        if len(image_array.shape) != 3 or image_array.shape[2] != 4:
            logger.error(f"Expected RGBA format from transparent text generator, got: {image_array.shape}")
            # Create a truly transparent RGBA frame as fallback
            rgba_array = np.zeros((final_height, final_width, 4), dtype=np.uint8)
            return rgba_array
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error rendering transparent text: {e}")
        # Return fully transparent RGBA frame as fallback
        fallback_width = output_width if output_width is not None else settings.get("output_config", {}).get("output_width", 512)
        fallback_height = output_height if output_height is not None else settings.get("output_config", {}).get("output_height", 256)
        return np.zeros((fallback_height, fallback_width, 4), dtype=np.uint8)


def render_transparent_mixed_text_with_highlighting(text_generator, create_transparent_settings_func,
                                                  full_text: str, all_words: List[Dict],
                                                  highlighted_word: Optional[Dict], current_time: float,
                                                  base_settings: Dict, highlight_settings: Dict,
                                                  output_width: int, output_height: int,
                                                  frame_padding_left: int = 20, frame_padding_right: int = 20,
                                                  frame_padding_top: int = 20, frame_padding_bottom: int = 20,
                                                  line_gaps: float = 1.2) -> np.ndarray:
    """Render text with highlighting on transparent background, ensuring RGBA format."""
    try:
        # Create transparent settings for both base and highlight
        transparent_base_settings = create_transparent_settings_func(base_settings)
        transparent_highlight_settings = create_transparent_settings_func(highlight_settings)
        
        # For now, use simple approach: render base text with transparent background
        # TODO: Implement proper transparent highlighting in future version
        logger.info(f"Rendering transparent text with highlighting for word: {highlighted_word.get('word', '')}")
        
        return render_transparent_text_with_settings(
            text_generator, full_text, transparent_base_settings, output_width, output_height,
            frame_padding_left, frame_padding_right, frame_padding_top, frame_padding_bottom, line_gaps
        )
        
    except Exception as e:
        logger.error(f"Error rendering transparent mixed text: {e}")
        # Fallback to simple transparent rendering
        return render_transparent_text_with_settings(
            text_generator, full_text, base_settings, output_width, output_height,
            frame_padding_left, frame_padding_right, frame_padding_top, frame_padding_bottom, line_gaps
        )


def ensure_rgba_format(frame: np.ndarray, frame_info: str = "") -> np.ndarray:
    """
    Ensure frame is in RGBA format (4 channels).
    Converts RGB to RGBA by adding full alpha channel if necessary.
    """
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        # Convert RGB to RGBA by adding full alpha channel
        height, width = frame.shape[:2]
        alpha_channel = np.ones((height, width, 1), dtype=frame.dtype) * 255
        rgba_frame = np.concatenate([frame, alpha_channel], axis=2)
        logger.warning(f"Converted {frame_info} from RGB to RGBA - text generator should return RGBA for transparent backgrounds")
        return rgba_frame
    elif len(frame.shape) == 3 and frame.shape[2] == 4:
        # Already RGBA
        return frame
    else:
        raise ValueError(f"Invalid frame format for {frame_info}: {frame.shape}")


def generate_transparent_empty_frame(generate_empty_frame_func, width: int, height: int, settings: Dict) -> np.ndarray:
    """Generate an empty transparent frame with RGBA format."""
    # Generate base empty frame
    frame = generate_empty_frame_func(width, height, settings)
    # Ensure RGBA format
    return ensure_rgba_format(frame, "empty transparent frame")


def generate_transparent_frame_with_groups(text_generator, create_transparent_settings_func, generate_transparent_empty_frame_func,
                                         active_groups: List[Dict], current_time: float,
                                         base_settings: Dict, highlight_settings: Optional[Dict],
                                         output_width: int, output_height: int,
                                         frame_padding_left: int = 20, frame_padding_right: int = 20,
                                         frame_padding_top: int = 20, frame_padding_bottom: int = 20,
                                         line_gaps: float = 1.2) -> np.ndarray:
    """Generate transparent frame with word groups - direct RGBA generation with highlighting support."""
    # Create text lines from active groups
    text_lines = []
    all_words = []
    
    for group in active_groups:
        text_lines.append(group['text'])
        all_words.extend(group['words'])
    
    full_text = '\n'.join(text_lines)
    
    if not full_text.strip():
        return generate_transparent_empty_frame_func(output_width, output_height, base_settings)
    
    # Add word highlighting if enabled
    if highlight_settings and all_words:
        highlighted_word = get_current_highlighted_word(all_words, current_time)
        if highlighted_word:
            # Use mixed text rendering with highlighting for transparent background
            return render_transparent_mixed_text_with_highlighting(
                text_generator, create_transparent_settings_func,
                full_text=full_text, all_words=all_words, highlighted_word=highlighted_word,
                current_time=current_time, base_settings=base_settings, highlight_settings=highlight_settings,
                output_width=output_width, output_height=output_height,
                frame_padding_left=frame_padding_left, frame_padding_right=frame_padding_right,
                frame_padding_top=frame_padding_top, frame_padding_bottom=frame_padding_bottom,
                line_gaps=line_gaps
            )
    
    # No highlighting, use transparent text rendering directly
    return render_transparent_text_with_settings(
        text_generator, full_text, base_settings, output_width, output_height,
        frame_padding_left, frame_padding_right, frame_padding_top, frame_padding_bottom, line_gaps
    )