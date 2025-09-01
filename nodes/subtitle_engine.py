"""
RajSubtitleEngine - Generate subtitle video frames with word-level timing and highlighting.
Main node for the video subtitling system.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json
from typing import List, Dict, Optional, Tuple, Any
from .utils import logger
from .subtitle_utils import (
    parse_whisper_word_data,
    get_total_duration,
    get_active_words_at_time,
    get_current_highlighted_word,
    detect_sentence_boundaries,
    organize_words_into_display_lines,
    create_timing_windows,
    create_word_groups,
    create_timing_windows_grouped,
    create_line_groups,
    create_timing_windows_line_grouped,
    create_area_based_word_groups,
    create_timing_windows_area_based,
    validate_word_timing_data
)
from .text_generator import RajTextGenerator


class RajSubtitleEngine:
    """
    Generate subtitle video frames with precise timing from whisper word data.
    Supports optional word highlighting and intelligent sentence handling.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "word_timings": ("LIST", {
                    "tooltip": "Word timing data from RajWhisperProcess"
                }),
                "base_settings": ("TEXT_SETTINGS", {
                    "tooltip": "Main subtitle styling from RajTextGenerator"
                }),
                "video_fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Video frame rate"
                })
            },
            "optional": {
                "highlight_settings": ("TEXT_SETTINGS", {
                    "tooltip": "Word highlight styling (leave empty for no highlighting)"
                }),
                "max_lines": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 5,
                    "tooltip": "Maximum number of lines to display at once"
                }),
                "use_line_groups": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use line grouping: word groups become lines, multiple lines form frames"
                }),
                "use_area_based_grouping": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use area-based grouping: fit words based on box dimensions instead of max lines"
                }),
                "box_width": ("INT", {
                    "default": 800,
                    "min": 100,
                    "max": 2000,
                    "tooltip": "Width of the subtitle box for area-based grouping"
                }),
                "box_height": ("INT", {
                    "default": 200,
                    "min": 50,
                    "max": 1000,
                    "tooltip": "Height of the subtitle box for area-based grouping"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "DICT")
    RETURN_NAMES = ("subtitle_images", "total_frames", "timing_info", "frame_metadata")
    FUNCTION = "generate_subtitle_video"
    CATEGORY = "Raj Video Processing 🎬/Subtitles"
    
    def __init__(self):
        self.text_generator = RajTextGenerator()
    
    def generate_subtitle_video(self,
                              word_timings: List[Dict],
                              base_settings: Dict,
                              video_fps: float,
                              highlight_settings: Optional[Dict] = None,
                              max_lines: int = 2,
                              use_line_groups: bool = False,
                              use_area_based_grouping: bool = False,
                              box_width: int = 800,
                              box_height: int = 200) -> Tuple[torch.Tensor, int, str, Dict]:
        """
        Generate subtitle video frames with word grouping, line grouping, or area-based grouping.
        
        Args:
            word_timings: Word timing data from RajWhisperProcess
            base_settings: Main subtitle styling settings (includes dimensions and font config)
            video_fps: Video frame rate
            highlight_settings: Optional word highlighting settings
            max_lines: Maximum lines to display simultaneously (for non-area-based modes)
            use_line_groups: If True, use line groups (word groups -> lines -> frames)
            use_area_based_grouping: If True, use area-based grouping (fit words in box dimensions)
            box_width: Width of subtitle box for area-based grouping
            box_height: Height of subtitle box for area-based grouping
            
        Returns:
            Tuple of (video_frames_tensor, total_frames, timing_info, metadata)
        """
        logger.info(f"Starting subtitle video generation with {len(word_timings)} words at {video_fps}fps")
        
        # Validate and parse word timing data
        is_valid, issues = validate_word_timing_data(word_timings)
        if not is_valid:
            logger.error(f"Invalid word timing data: {issues}")
            raise ValueError(f"Word timing validation failed: {'; '.join(issues)}")
        
        parsed_words = parse_whisper_word_data(word_timings)
        if not parsed_words:
            logger.error("No valid words found in timing data")
            raise ValueError("No valid words found in timing data")
        
        # Calculate total duration and frames
        total_duration = get_total_duration(parsed_words)
        total_frames = int(total_duration * video_fps) + 1
        
        logger.info(f"Generating {total_frames} frames for {total_duration:.2f}s duration")
        
        # Get dimensions from base_settings output_config (where text generator stores them)
        output_config = base_settings.get('output_config', {})
        display_width = output_config.get('output_width', 512)
        display_height = output_config.get('output_height', 256)
        
        # Get font and layout configuration
        font_config = base_settings.get('font_config', {})
        layout_config = base_settings.get('layout_config', {})
        margin_x = layout_config.get('margin_x', 20)
        
        logger.info(f"Using dimensions from settings: {display_width}x{display_height}")
        
        # Create grouping based on selected mode
        if use_area_based_grouping:
            # Area-based grouping mode: fit words based on box dimensions
            margin_y = layout_config.get('margin_y', 20)
            line_spacing = layout_config.get('line_spacing', 1.2)
            
            area_groups = create_area_based_word_groups(
                parsed_words,
                box_width,
                box_height,
                font_config,
                margin_x,
                margin_y,
                line_spacing
            )
            timing_windows = create_timing_windows_area_based(area_groups, video_fps)
            logger.info(f"Using area-based grouping mode: {len(area_groups)} area groups for {box_width}x{box_height}px box")
            
            # For compatibility, set word_groups to area_groups
            word_groups = area_groups
            grouping_mode = "area_based"
            
        elif use_line_groups:
            # Line groups mode: word groups -> line groups -> timing windows
            word_groups = create_word_groups(
                parsed_words, 
                display_width, 
                font_config, 
                max_lines, 
                margin_x
            )
            line_groups = create_line_groups(word_groups, max_lines)
            timing_windows = create_timing_windows_line_grouped(line_groups, video_fps)
            logger.info(f"Using line groups mode: {len(line_groups)} line groups from {len(word_groups)} word groups")
            grouping_mode = "line_groups"
            
        else:
            # Original word groups mode
            word_groups = create_word_groups(
                parsed_words, 
                display_width, 
                font_config, 
                max_lines, 
                margin_x
            )
            timing_windows = create_timing_windows_grouped(word_groups, video_fps, max_lines)
            logger.info(f"Using word groups mode: {len(word_groups)} word groups")
            grouping_mode = "word_groups"
        
        # Detect sentences for better organization
        sentences = detect_sentence_boundaries(parsed_words)
        
        # Generate frames
        frames = []
        frame_metadata = {"frames": []}
        
        # Use the dimensions we already extracted
        output_width = display_width
        output_height = display_height
        
        for frame_num in range(total_frames):
            current_time = frame_num / video_fps
            
            # Get active groups for this frame (could be word groups or line groups)
            active_groups = timing_windows.get(frame_num, [])
            
            # Generate subtitle frame
            if active_groups:
                if use_area_based_grouping:
                    # Area-based grouping mode: each active group contains multiple lines fitted to box
                    frame = self._generate_frame_with_area_groups(
                        active_groups,  # area groups
                        current_time,
                        base_settings,
                        highlight_settings,
                        output_width,
                        output_height
                    )
                elif use_line_groups:
                    # Line groups mode: each active group is a line group containing multiple lines
                    frame = self._generate_frame_with_line_groups(
                        active_groups,  # line groups
                        current_time,
                        base_settings,
                        highlight_settings,
                        output_width,
                        output_height
                    )
                else:
                    # Original word groups mode
                    frame = self._generate_frame_with_groups(
                        active_groups,  # word groups
                        current_time,
                        base_settings,
                        highlight_settings,
                        output_width,
                        output_height
                    )
            else:
                # Empty frame
                frame = self._generate_empty_frame(output_width, output_height, base_settings)
            
            frames.append(frame)
            
            # Store metadata
            if use_area_based_grouping and active_groups:
                # Area-based grouping mode: extract text from area groups
                groups_text = [g.get('text', '') for g in active_groups]
            elif use_line_groups and active_groups:
                # Line groups mode: extract text from line groups
                groups_text = [g.get('combined_text', '') for g in active_groups]
            else:
                # Word groups mode: extract text from word groups
                groups_text = [g.get('text', '') for g in active_groups] if active_groups else []
            
            frame_metadata["frames"].append({
                "frame": frame_num,
                "time": current_time,
                "active_groups_count": len(active_groups),
                "groups_text": groups_text,
                "mode": grouping_mode
            })
            
            # Progress logging every 10% of frames
            if frame_num % (total_frames // 10) == 0:
                progress = (frame_num / total_frames) * 100
                logger.info(f"Generated {frame_num}/{total_frames} frames ({progress:.1f}%)")
        
        # Convert frames to image batch tensor
        if frames:
            # Apply standardization to ensure all frames have same format
            standardized_frames = []
            for frame in frames:
                standardized_frame = self._standardize_frame_format(frame)
                standardized_frames.append(standardized_frame)
            
            # Stack all frames into image batch tensor [num_frames, height, width, channels]
            # Convert to float32 and normalize to 0-1 range for ComfyUI compatibility
            frames_tensor = torch.stack([
                torch.from_numpy(frame.astype(np.float32) / 255.0) 
                for frame in standardized_frames
            ])
        else:
            logger.error("No frames generated")
            raise ValueError("No frames were generated")
        
        # Create timing info summary
        timing_info = self._create_timing_info(parsed_words, total_frames, video_fps, sentences)
        
        # Add overall metadata
        metadata = {
            "total_duration": total_duration,
            "total_frames": total_frames,
            "video_fps": video_fps,
            "total_words": len(parsed_words),
            "total_sentences": len(sentences),
            "highlighting_enabled": highlight_settings is not None,
            "word_groups": len(word_groups),
            "max_lines": max_lines,
            "display_width": display_width,
            "display_height": display_height,
            "use_line_groups": use_line_groups,
            "use_area_based_grouping": use_area_based_grouping,
            "grouping_mode": grouping_mode
        }
        
        # Add mode-specific metadata
        if use_area_based_grouping:
            metadata.update({
                "box_width": box_width,
                "box_height": box_height,
                "area_groups": len(area_groups),
                "avg_lines_per_group": sum(g.get('line_count', 0) for g in area_groups) / len(area_groups) if area_groups else 0
            })
        elif use_line_groups:
            metadata["line_groups"] = len(line_groups)
        
        frame_metadata.update(metadata)
        
        logger.info(f"Successfully generated subtitle images: {total_frames} frames, {total_duration:.2f}s")
        
        return (frames_tensor, total_frames, timing_info, frame_metadata)
    
    def _generate_frame_with_groups(self,
                                  active_groups: List[Dict],
                                  current_time: float,
                                  base_settings: Dict,
                                  highlight_settings: Optional[Dict],
                                  output_width: int,
                                  output_height: int) -> np.ndarray:
        """Generate a single frame with word groups."""
        
        # Create text lines from active groups
        text_lines = []
        all_words = []
        
        for group in active_groups:
            text_lines.append(group['text'])
            all_words.extend(group['words'])
        
        full_text = '\n'.join(text_lines)
        
        if not full_text.strip():
            return self._generate_empty_frame(output_width, output_height, base_settings)
        
        # Generate base subtitle frame using text generator
        base_frame = self._render_text_with_settings(full_text, base_settings)
        
        # Add word highlighting if enabled
        if highlight_settings and all_words:
            highlighted_word = get_current_highlighted_word(all_words, current_time)
            if highlighted_word:
                # For now, skip complex highlighting in grouped mode
                # This could be enhanced later to highlight within groups
                pass
        
        return base_frame
    
    def _generate_frame_with_line_groups(self,
                                       active_line_groups: List[Dict],
                                       current_time: float,
                                       base_settings: Dict,
                                       highlight_settings: Optional[Dict],
                                       output_width: int,
                                       output_height: int) -> np.ndarray:
        """Generate a single frame with line groups (each containing multiple lines)."""
        
        if not active_line_groups:
            return self._generate_empty_frame(output_width, output_height, base_settings)
        
        # For line groups mode, we typically show one line group per frame
        # If there are multiple active line groups, take the first one
        line_group = active_line_groups[0]
        
        # Extract the combined text from the line group (already formatted with newlines)
        full_text = line_group.get('combined_text', '')
        
        if not full_text.strip():
            return self._generate_empty_frame(output_width, output_height, base_settings)
        
        # Generate base subtitle frame using text generator
        base_frame = self._render_text_with_settings(full_text, base_settings)
        
        # Add word highlighting if enabled
        if highlight_settings:
            # For line groups, we need to extract all words from all lines in the group
            all_words = []
            for line in line_group.get('lines', []):
                all_words.extend(line.get('words', []))
            
            if all_words:
                highlighted_word = get_current_highlighted_word(all_words, current_time)
                if highlighted_word:
                    # For now, skip complex highlighting in line groups mode
                    # This could be enhanced later to highlight within line groups
                    pass
        
        return base_frame
    
    def _generate_frame_with_area_groups(self,
                                       active_area_groups: List[Dict],
                                       current_time: float,
                                       base_settings: Dict,
                                       highlight_settings: Optional[Dict],
                                       output_width: int,
                                       output_height: int) -> np.ndarray:
        """Generate a single frame with area-based groups (each containing multiple lines fitted to box)."""
        
        if not active_area_groups:
            return self._generate_empty_frame(output_width, output_height, base_settings)
        
        # For area-based groups mode, we typically show one area group per frame
        # If there are multiple active area groups, take the first one
        area_group = active_area_groups[0]
        
        # Extract the text from the area group (already formatted with newlines)
        full_text = area_group.get('text', '')
        
        if not full_text.strip():
            return self._generate_empty_frame(output_width, output_height, base_settings)
        
        # Generate base subtitle frame using text generator
        base_frame = self._render_text_with_settings(full_text, base_settings)
        
        # Add word highlighting if enabled
        if highlight_settings:
            # For area groups, extract all words from the group
            all_words = area_group.get('words', [])
            
            if all_words:
                highlighted_word = get_current_highlighted_word(all_words, current_time)
                if highlighted_word:
                    # For now, skip complex highlighting in area-based groups mode
                    # This could be enhanced later to highlight within area groups
                    pass
        
        return base_frame
    
    def _generate_frame_with_words(self,
                                 active_words: List[Dict],
                                 current_time: float,
                                 base_settings: Dict,
                                 highlight_settings: Optional[Dict],
                                 output_width: int,
                                 output_height: int) -> np.ndarray:
        """Generate a single frame with the given words."""
        
        # Simple organization for legacy mode (one word at a time)
        word_lines = [[word] for word in active_words]
        
        # Create base text for the frame
        text_lines = []
        for line_words in word_lines:
            line_text = ' '.join(word['word'] for word in line_words)
            text_lines.append(line_text)
        
        full_text = '\\n'.join(text_lines)
        
        if not full_text.strip():
            return self._generate_empty_frame(output_width, output_height, base_settings)
        
        # Generate base subtitle frame using text generator
        base_frame = self._render_text_with_settings(full_text, base_settings)
        
        # Add word highlighting if enabled
        if highlight_settings:
            highlighted_word = get_current_highlighted_word(active_words, current_time)
            if highlighted_word:
                base_frame = self._add_word_highlight(
                    base_frame, 
                    highlighted_word, 
                    word_lines, 
                    highlight_settings,
                    base_settings
                )
        
        return base_frame
    
    def _standardize_frame_format(self, frame: np.ndarray) -> np.ndarray:
        """Ensure frame is RGB format with consistent shape."""
        if len(frame.shape) == 3:
            if frame.shape[2] == 4:  # RGBA → RGB
                frame = frame[:, :, :3]
            elif frame.shape[2] != 3:
                raise ValueError(f"Unexpected channel count: {frame.shape[2]}")
        elif len(frame.shape) == 2:  # Grayscale → RGB
            frame = np.stack([frame] * 3, axis=-1)
        
        return frame.astype(np.uint8)
    
    def _render_text_with_settings(self, text: str, settings: Dict) -> np.ndarray:
        """Render text using the text generator with given settings."""
        try:
            # Extract all parameters from settings
            font_config = settings.get('font_config', {})
            layout_config = settings.get('layout_config', {})
            effects_config = settings.get('effects_config', {})
            container_config = settings.get('container_config', {})
            output_config = settings.get('output_config', {})
            
            # Call the text generator
            result = self.text_generator.generate_text(
                text=text,
                output_width=output_config.get('output_width', 512),
                output_height=output_config.get('output_height', 256),
                font_name=font_config.get('font_name', 'Arial'),
                font_size=font_config.get('font_size', 36),
                font_color=font_config.get('font_color', '#FFFFFF'),
                background_color=output_config.get('background_color', '#000000'),
                text_align=layout_config.get('text_align', 'center'),
                vertical_align=layout_config.get('vertical_align', 'middle'),
                words_per_line=layout_config.get('words_per_line', 0),
                max_lines=layout_config.get('max_lines', 0),
                line_spacing=layout_config.get('line_spacing', 1.2),
                letter_spacing=layout_config.get('letter_spacing', 0),
                margin_x=layout_config.get('margin_x', 20),
                margin_y=layout_config.get('margin_y', 20),
                auto_size=layout_config.get('auto_size', False),
                base_opacity=output_config.get('base_opacity', 1.0),
                font_file=font_config.get('font_file', ''),
                font_weight=font_config.get('font_weight', 'normal'),
                text_border_width=effects_config.get('text_border_width', 0),
                text_border_color=effects_config.get('text_border_color', '#000000'),
                shadow_enabled=effects_config.get('shadow_enabled', False),
                shadow_offset_x=effects_config.get('shadow_offset_x', 2),
                shadow_offset_y=effects_config.get('shadow_offset_y', 2),
                shadow_color=effects_config.get('shadow_color', '#000000'),
                shadow_blur=effects_config.get('shadow_blur', 2),
                text_bg_enabled=effects_config.get('text_bg_enabled', False),
                text_bg_color=effects_config.get('text_bg_color', '#FFFF00'),
                text_bg_padding=effects_config.get('text_bg_padding', 5),
                gradient_enabled=effects_config.get('gradient_enabled', False),
                gradient_color2=effects_config.get('gradient_color2', '#FF0000'),
                gradient_direction=effects_config.get('gradient_direction', 'vertical'),
                container_enabled=container_config.get('container_enabled', False),
                container_color=container_config.get('container_color', '#333333'),
                container_width=container_config.get('container_width', 2),
                container_padding=container_config.get('container_padding', 15),
                container_border_color=container_config.get('container_border_color', '#FFFFFF'),
                container_fill=container_config.get('container_fill', True)
            )
            
            # Extract image tensor and convert to numpy array
            image_tensor = result[0]  # First return value is the image
            image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
            
            # Standardize to RGB format
            image_array = self._standardize_frame_format(image_array)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error rendering text with settings: {e}")
            # Return empty frame as fallback
            output_width = settings.get('output_config', {}).get('output_width', 512)
            output_height = settings.get('output_config', {}).get('output_height', 256)
            return self._generate_empty_frame(output_width, output_height, settings)
    
    def _add_word_highlight(self,
                          base_frame: np.ndarray,
                          highlighted_word: Dict,
                          word_lines: List[List[Dict]],
                          highlight_settings: Dict,
                          base_settings: Dict) -> np.ndarray:
        """Add highlighting to a specific word in the frame."""
        
        # Find which line contains the highlighted word
        target_line_idx = -1
        target_word_idx = -1
        
        for line_idx, line_words in enumerate(word_lines):
            for word_idx, word in enumerate(line_words):
                if word.get('index') == highlighted_word.get('index'):
                    target_line_idx = line_idx
                    target_word_idx = word_idx
                    break
            if target_line_idx >= 0:
                break
        
        if target_line_idx < 0:
            logger.warning(f"Could not find highlighted word '{highlighted_word['word']}' in current lines")
            return base_frame
        
        # For simplicity, render the highlighted word as overlay
        # In a more sophisticated implementation, you'd calculate exact word positions
        highlighted_text = highlighted_word['word']
        
        try:
            # Render just the highlighted word with highlight settings
            highlight_frame = self._render_text_with_settings(highlighted_text, highlight_settings)
            
            # For now, just composite the highlight in the center
            # TODO: Implement precise word positioning
            base_pil = Image.fromarray(base_frame)
            highlight_pil = Image.fromarray(highlight_frame)
            
            # Simple center overlay - in production, calculate exact word position
            base_pil = base_pil.convert('RGBA')
            highlight_pil = highlight_pil.convert('RGBA')
            
            # Create a composite
            composite = Image.alpha_composite(base_pil, highlight_pil)
            
            # Convert to numpy and standardize format
            result_array = np.array(composite.convert('RGB'))
            return self._standardize_frame_format(result_array)
            
        except Exception as e:
            logger.warning(f"Error adding word highlight: {e}")
            return base_frame
    
    def _generate_empty_frame(self, width: int, height: int, settings: Dict) -> np.ndarray:
        """Generate an empty frame with background."""
        bg_color = settings.get('output_config', {}).get('background_color', '#000000')
        
        # Parse color
        if bg_color.startswith('#'):
            bg_color = bg_color[1:]
        
        try:
            r = int(bg_color[0:2], 16)
            g = int(bg_color[2:4], 16)  
            b = int(bg_color[4:6], 16)
        except (ValueError, IndexError):
            r, g, b = 0, 0, 0  # Default to black
        
        # Create empty frame
        frame = np.full((height, width, 3), [r, g, b], dtype=np.uint8)
        return frame
    
    def _create_timing_info(self, words: List[Dict], total_frames: int, fps: float, sentences: List[List[Dict]]) -> str:
        """Create human-readable timing information."""
        if not words:
            return "No timing information available"
        
        total_duration = get_total_duration(words)
        
        info = [
            f"Subtitle Video Generated:",
            f"Duration: {total_duration:.2f} seconds",
            f"Frames: {total_frames} @ {fps}fps",
            f"Words: {len(words)}",
            f"Sentences: {len(sentences)}",
            f"First word: '{words[0]['word']}' at {words[0]['start_time']:.2f}s",
            f"Last word: '{words[-1]['word']}' at {words[-1]['end_time']:.2f}s",
            "",
            "Sentence breakdown:"
        ]
        
        for i, sentence in enumerate(sentences[:5]):  # Show first 5 sentences
            sentence_text = ' '.join(word['word'] for word in sentence)
            sentence_start = sentence[0]['start_time']
            sentence_end = sentence[-1]['end_time']
            info.append(f"  {i+1}. \"{sentence_text[:50]}{'...' if len(sentence_text) > 50 else ''}\"")
            info.append(f"     Time: {sentence_start:.2f}s - {sentence_end:.2f}s")
        
        if len(sentences) > 5:
            info.append(f"  ... and {len(sentences) - 5} more sentences")
        
        return "\\n".join(info)