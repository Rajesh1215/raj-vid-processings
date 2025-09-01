"""
RajSubtitleEngine - Generate subtitle video frames with word-level timing and highlighting.
Main node for the video subtitling system.
"""

import torch
import numpy as np
import os
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
    calculate_precise_word_positions,
    find_word_bounds_in_text,
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
                }),
                "frame_padding_width": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 200,
                    "tooltip": "Extra width added to frame for better text spacing (useful for large highlights)"
                }),
                "frame_padding_height": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Extra height added to frame for better text spacing (useful for large highlights)"
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
                              box_height: int = 200,
                              frame_padding_width: int = 0,
                              frame_padding_height: int = 0) -> Tuple[torch.Tensor, int, str, Dict]:
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
            frame_padding_width: Extra width added to frame for better text spacing (0-200px)
            frame_padding_height: Extra height added to frame for better text spacing (0-100px)
            
        Frame Sizing:
            - Base dimensions come from base_settings.output_config
            - Manual padding is added via frame_padding_width/height parameters
            - Auto-padding is applied when highlight text is >30% larger than base text
            - Final frame size = base_size + manual_padding + auto_padding
            
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
        
        logger.info(f"Base dimensions from settings: {display_width}x{display_height}")
        
        # Add smart padding detection for large highlights
        auto_padding_width = 0
        auto_padding_height = 0
        if highlight_settings:
            highlight_font_config = highlight_settings.get('font_config', {})
            base_font_size = font_config.get('font_size', 20)
            highlight_font_size = highlight_font_config.get('font_size', base_font_size)
            
            # Auto-suggest padding for significantly larger highlight text
            if highlight_font_size > base_font_size * 1.3:
                size_diff = highlight_font_size - base_font_size
                auto_padding_width = min(int(size_diff * 2), 50)  # Max 50px auto padding
                auto_padding_height = min(int(size_diff * 1), 25)  # Max 25px auto padding
                logger.info(f"Large highlight detected ({highlight_font_size}px vs {base_font_size}px), suggesting auto padding: {auto_padding_width}x{auto_padding_height}")
        
        # Calculate final frame dimensions with padding
        total_padding_width = frame_padding_width + auto_padding_width
        total_padding_height = frame_padding_height + auto_padding_height
        
        final_width = display_width + total_padding_width
        final_height = display_height + total_padding_height
        
        logger.info(f"Final frame dimensions: {final_width}x{final_height} (added padding: {total_padding_width}x{total_padding_height})")
        
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
        
        # Use the final dimensions with padding
        output_width = final_width
        output_height = final_height
        
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
    
    def generate_subtitle_keyframes(self,
                                  word_timings: List[Dict],
                                  base_settings: Dict,
                                  highlight_settings: Optional[Dict] = None,
                                  max_lines: int = 2,
                                  use_line_groups: bool = False,
                                  use_area_based_grouping: bool = False,
                                  box_width: int = 800,
                                  box_height: int = 200) -> Dict[str, Any]:
        """
        Generate keyframes for subtitle video - only essential frames when content changes.
        Much more efficient than FPS-level generation for testing and development.
        
        Args:
            word_timings: Word timing data from RajWhisperProcess
            base_settings: Main subtitle styling settings
            highlight_settings: Optional word highlighting settings
            max_lines: Maximum lines to display simultaneously (for non-area-based modes)
            use_line_groups: If True, use line groups (word groups -> lines -> frames)
            use_area_based_grouping: If True, use area-based grouping (fit words in box dimensions)
            box_width: Width of subtitle box for area-based grouping
            box_height: Height of subtitle box for area-based grouping
            
        Returns:
            Dictionary containing keyframes, metadata, and timing information
        """
        logger.info(f"Starting keyframe generation with {len(word_timings)} words")
        
        # Validate and parse word timing data
        is_valid, issues = validate_word_timing_data(word_timings)
        if not is_valid:
            logger.error(f"Invalid word timing data: {issues}")
            raise ValueError(f"Word timing validation failed: {'; '.join(issues)}")
        
        parsed_words = parse_whisper_word_data(word_timings)
        if not parsed_words:
            logger.error("No valid words found in timing data")
            raise ValueError("No valid words found in timing data")
        
        # Get dimensions from base_settings
        output_config = base_settings.get('output_config', {})
        display_width = output_config.get('output_width', 512)
        display_height = output_config.get('output_height', 256)
        
        logger.info(f"Keyframe dimensions: {display_width}x{display_height}")
        
        # Get font and layout configuration
        font_config = base_settings.get('font_config', {})
        layout_config = base_settings.get('layout_config', {})
        margin_x = layout_config.get('margin_x', 20)
        
        # Create grouping based on selected mode
        if use_area_based_grouping:
            # Area-based grouping mode
            margin_y = layout_config.get('margin_y', 20)
            line_spacing = layout_config.get('line_spacing', 1.2)
            
            word_groups = create_area_based_word_groups(
                parsed_words,
                box_width,
                box_height,
                font_config,
                margin_x,
                margin_y,
                line_spacing
            )
            grouping_mode = "area_based"
            
        elif use_line_groups:
            # Line groups mode
            word_groups = create_word_groups(
                parsed_words, 
                display_width, 
                font_config, 
                max_lines, 
                margin_x
            )
            line_groups = create_line_groups(word_groups, max_lines)
            word_groups = line_groups  # Use line groups as the main groups
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
            grouping_mode = "word_groups"
        
        logger.info(f"Using {grouping_mode} mode: {len(word_groups)} groups")
        
        # Detect keyframe timestamps
        keyframe_times = self._detect_keyframe_times(parsed_words, word_groups)
        logger.info(f"Detected {len(keyframe_times)} keyframes")
        
        # Generate keyframes
        keyframes = []
        total_duration = get_total_duration(parsed_words)
        
        for i, timestamp in enumerate(keyframe_times):
            # Calculate duration until next keyframe
            if i < len(keyframe_times) - 1:
                duration = keyframe_times[i + 1] - timestamp
            else:
                duration = total_duration - timestamp
            
            # Generate frame for this timestamp
            keyframe_data = self._generate_keyframe_at_time(
                timestamp=timestamp,
                duration=duration,
                parsed_words=parsed_words,
                word_groups=word_groups,
                base_settings=base_settings,
                highlight_settings=highlight_settings,
                output_width=display_width,
                output_height=display_height,
                grouping_mode=grouping_mode,
                use_line_groups=use_line_groups,
                use_area_based_grouping=use_area_based_grouping
            )
            
            if keyframe_data:
                keyframes.append(keyframe_data)
        
        # Calculate compression statistics
        estimated_fps_frames = int(total_duration * 30)  # Assume 30fps
        compression_ratio = ((estimated_fps_frames - len(keyframes)) / estimated_fps_frames) * 100
        
        # Create metadata
        metadata = {
            "total_keyframes": len(keyframes),
            "total_duration": total_duration,
            "estimated_fps_frames": estimated_fps_frames,
            "compression_ratio": f"{compression_ratio:.1f}% reduction vs 30fps",
            "grouping_mode": grouping_mode,
            "settings_used": {
                "base": base_settings,
                "highlight": highlight_settings,
                "max_lines": max_lines,
                "use_line_groups": use_line_groups,
                "use_area_based_grouping": use_area_based_grouping
            },
            "word_groups": len(word_groups),
            "total_words": len(parsed_words),
            "display_dimensions": f"{display_width}x{display_height}"
        }
        
        logger.info(f"Generated {len(keyframes)} keyframes - {compression_ratio:.1f}% reduction from FPS")
        
        return {
            "keyframes": keyframes,
            "metadata": metadata,
            "word_timings": parsed_words,
            "word_groups": word_groups
        }
    
    def _detect_keyframe_times(self, parsed_words: List[Dict], word_groups: List[Dict]) -> List[float]:
        """
        Detect timestamps where keyframes should be generated.
        Only generate frames when content actually changes.
        
        Args:
            parsed_words: List of word dictionaries with timing
            word_groups: List of word groups with timing
            
        Returns:
            Sorted list of timestamps for keyframe generation
        """
        keyframe_times = set()
        
        # Add word highlighting change points
        for word in parsed_words:
            # Add when word starts being highlighted
            keyframe_times.add(word['start_time'])
            # Add when word stops being highlighted
            keyframe_times.add(word['end_time'])
        
        # Add word group visibility change points
        for group in word_groups:
            # Add when group becomes visible
            keyframe_times.add(group['start_time'])
            # Add when group disappears
            keyframe_times.add(group['end_time'])
        
        # Add start and end boundaries
        if parsed_words:
            keyframe_times.add(0.0)  # Start of subtitle sequence
            keyframe_times.add(max(word['end_time'] for word in parsed_words))  # End
        
        # Convert to sorted list and remove very close duplicates
        keyframe_times = sorted(list(keyframe_times))
        
        # Remove duplicates that are too close together (< 0.01s apart)
        filtered_times = []
        for time in keyframe_times:
            if not filtered_times or abs(time - filtered_times[-1]) >= 0.01:
                filtered_times.append(time)
        
        logger.info(f"Keyframe times: {len(keyframe_times)} raw -> {len(filtered_times)} filtered")
        return filtered_times
    
    def _generate_keyframe_at_time(self,
                                 timestamp: float,
                                 duration: float,
                                 parsed_words: List[Dict],
                                 word_groups: List[Dict],
                                 base_settings: Dict,
                                 highlight_settings: Optional[Dict],
                                 output_width: int,
                                 output_height: int,
                                 grouping_mode: str,
                                 use_line_groups: bool,
                                 use_area_based_grouping: bool) -> Optional[Dict]:
        """
        Generate a keyframe at a specific timestamp.
        
        Args:
            timestamp: Time in seconds for this keyframe
            duration: Duration until next keyframe
            parsed_words: All word timing data
            word_groups: All word groups
            base_settings: Base text settings
            highlight_settings: Highlight text settings
            output_width: Frame width
            output_height: Frame height
            grouping_mode: Type of grouping being used
            use_line_groups: Whether line groups mode is active
            use_area_based_grouping: Whether area-based mode is active
            
        Returns:
            Dictionary containing keyframe data or None if no content
        """
        try:
            # Find active word groups at this timestamp
            active_groups = []
            for group in word_groups:
                if group['start_time'] <= timestamp <= group['end_time']:
                    active_groups.append(group)
            
            # Find highlighted word at this timestamp
            highlighted_word = get_current_highlighted_word(parsed_words, timestamp)
            
            # Generate frame based on grouping mode
            if not active_groups:
                # No content at this time - generate empty frame
                frame_array = self._generate_empty_frame(output_width, output_height, base_settings)
                active_text = ""
                group_info = {}
            else:
                # Generate frame with content
                if use_area_based_grouping:
                    frame_array = self._generate_frame_with_area_groups(
                        active_groups, timestamp, base_settings, highlight_settings,
                        output_width, output_height
                    )
                elif use_line_groups:
                    frame_array = self._generate_frame_with_line_groups(
                        active_groups, timestamp, base_settings, highlight_settings,
                        output_width, output_height
                    )
                else:
                    frame_array = self._generate_frame_with_groups(
                        active_groups, timestamp, base_settings, highlight_settings,
                        output_width, output_height
                    )
                
                # Extract active text and group info
                if use_area_based_grouping and active_groups:
                    active_text = active_groups[0].get('text', '')
                    group_info = {
                        'line_count': active_groups[0].get('line_count', 0),
                        'word_count': active_groups[0].get('word_count', 0)
                    }
                elif use_line_groups and active_groups:
                    active_text = active_groups[0].get('combined_text', '')
                    group_info = {
                        'line_count': active_groups[0].get('line_count', 0),
                        'lines': len(active_groups[0].get('lines', []))
                    }
                else:
                    # Regular word groups
                    text_lines = []
                    total_words = 0
                    for group in active_groups:
                        text_lines.append(group.get('text', ''))
                        total_words += len(group.get('words', []))
                    active_text = '\n'.join(text_lines)
                    group_info = {
                        'groups': len(active_groups),
                        'total_words': total_words
                    }
            
            # Create keyframe data structure
            keyframe_data = {
                "timestamp": timestamp,
                "duration": duration,
                "frame_image": frame_array,  # numpy array (height, width, 3)
                "highlighted_word": highlighted_word.get('word', '') if highlighted_word else '',
                "highlighted_word_index": highlighted_word.get('index', -1) if highlighted_word else -1,
                "active_text": active_text,
                "active_groups_count": len(active_groups),
                "grouping_mode": grouping_mode,
                "group_info": group_info,
                "frame_dimensions": f"{output_width}x{output_height}",
                "base_dimensions": f"{display_width}x{display_height}",
                "padding_applied": f"{total_padding_width}x{total_padding_height}",
                "has_highlighting": highlight_settings is not None and highlighted_word is not None
            }
            
            return keyframe_data
            
        except Exception as e:
            logger.error(f"Error generating keyframe at {timestamp}s: {e}")
            return None
    
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
        
        # Add word highlighting if enabled
        if highlight_settings and all_words:
            logger.debug(f"Highlighting enabled at {current_time:.2f}s with {len(all_words)} words")
            highlighted_word = get_current_highlighted_word(all_words, current_time)
            if highlighted_word:
                logger.info(f"Frame {current_time:.2f}s: highlighting word '{highlighted_word.get('word', '')}'")
            else:
                logger.debug(f"Frame {current_time:.2f}s: no word to highlight")
            # Use the new mixed text rendering with precise highlighting
            return self._render_mixed_text_with_highlighting(
                full_text=full_text,
                all_words=all_words,
                highlighted_word=highlighted_word,
                current_time=current_time,
                base_settings=base_settings,
                highlight_settings=highlight_settings,
                output_width=output_width,
                output_height=output_height
            )
        else:
            if not highlight_settings:
                logger.debug(f"Frame {current_time:.2f}s: no highlight settings provided")
            if not all_words:
                logger.debug(f"Frame {current_time:.2f}s: no words data available")
        
        # No highlighting, use standard rendering
        logger.debug(f"Frame {current_time:.2f}s: using standard rendering (no highlighting)")
        return self._render_text_with_settings(full_text, base_settings)
    
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
        
        # Add word highlighting if enabled
        if highlight_settings:
            # For line groups, we need to extract all words from all lines in the group
            all_words = []
            for line in line_group.get('lines', []):
                all_words.extend(line.get('words', []))
            
            if all_words:
                highlighted_word = get_current_highlighted_word(all_words, current_time)
                # Use the new mixed text rendering with precise highlighting
                return self._render_mixed_text_with_highlighting(
                    full_text=full_text,
                    all_words=all_words,
                    highlighted_word=highlighted_word,
                    current_time=current_time,
                    base_settings=base_settings,
                    highlight_settings=highlight_settings,
                    output_width=output_width,
                    output_height=output_height
                )
        
        # No highlighting, use standard rendering
        return self._render_text_with_settings(full_text, base_settings)
    
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
        
        # Add word highlighting if enabled
        if highlight_settings:
            # For area groups, extract all words from the group
            all_words = area_group.get('words', [])
            
            if all_words:
                highlighted_word = get_current_highlighted_word(all_words, current_time)
                # Use the new mixed text rendering with precise highlighting
                return self._render_mixed_text_with_highlighting(
                    full_text=full_text,
                    all_words=all_words,
                    highlighted_word=highlighted_word,
                    current_time=current_time,
                    base_settings=base_settings,
                    highlight_settings=highlight_settings,
                    output_width=output_width,
                    output_height=output_height
                )
        
        # No highlighting, use standard rendering
        return self._render_text_with_settings(full_text, base_settings)
    
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
            # Use the new mixed text rendering with precise highlighting
            return self._render_mixed_text_with_highlighting(
                full_text=full_text,
                all_words=active_words,
                highlighted_word=highlighted_word,
                current_time=current_time,
                base_settings=base_settings,
                highlight_settings=highlight_settings,
                output_width=output_width,
                output_height=output_height
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
    
    def _render_mixed_text_with_highlighting(self,
                                           full_text: str,
                                           all_words: List[Dict],
                                           highlighted_word: Optional[Dict],
                                           current_time: float,
                                           base_settings: Dict,
                                           highlight_settings: Dict,
                                           output_width: int,
                                           output_height: int) -> np.ndarray:
        """Render text with precise word highlighting using calculated positions."""
        
        logger.debug(f"Mixed rendering: text='{full_text[:50]}...', highlighted='{highlighted_word.get('word', '') if highlighted_word else None}'")
        
        if not highlighted_word:
            # No highlighting needed, use standard rendering
            logger.debug("No highlighted word, falling back to standard rendering")
            return self._render_text_with_settings(full_text, base_settings)
        
        try:
            # Use dynamic rendering for clean highlighting without duplication
            logger.info(f"Using dynamic highlighting for word '{highlighted_word.get('word', '')}'")
            
            # Current timestamp is now passed as a parameter
            
            return self._render_text_with_dynamic_highlighting(
                full_text=full_text,
                all_words=all_words,
                highlighted_word=highlighted_word,
                current_time=current_time,
                base_settings=base_settings,
                highlight_settings=highlight_settings,
                output_width=output_width,
                output_height=output_height
            )
            
        except Exception as e:
            logger.warning(f"Error in mixed text highlighting: {e}")
            # Fallback to base rendering
            return self._render_text_with_settings(full_text, base_settings)
    
    def _render_text_with_dynamic_highlighting(self,
                                             full_text: str,
                                             all_words: List[Dict],
                                             highlighted_word: Optional[Dict],
                                             current_time: float,
                                             base_settings: Dict,
                                             highlight_settings: Dict,
                                             output_width: int,
                                             output_height: int) -> np.ndarray:
        """Render text with dynamic word-level highlighting in a single pass.
        
        Enhanced features:
        - Full font weight support (normal, bold, italic, bold_italic) for highlighted words
        - Custom margin control via highlight_settings.layout_config:
          * margin_width: Additional horizontal spacing around highlighted words
          * margin_height: Additional vertical spacing around highlighted words
          * margin_x/margin_y: Override base margins completely
        - Automatic fallback to base settings when highlight settings are incomplete
        
        Example highlight_settings:
        {
            'font_config': {
                'font_name': 'Arial',
                'font_size': 28,        # Larger size for highlights
                'font_weight': 'bold',  # Bold highlighting
                'color': '#0066FF'
            },
            'layout_config': {
                'margin_width': 5,      # Extra horizontal spacing for highlights
                'margin_height': 3      # Extra vertical spacing for highlights
            }
        }
        """
        
        logger.debug(f"Dynamic rendering: {len(all_words)} words, highlighted='{highlighted_word.get('word', '') if highlighted_word else None}'")
        
        try:
            # Extract configurations
            font_config = base_settings.get('font_config', {})
            layout_config = base_settings.get('layout_config', {})
            output_config = base_settings.get('output_config', {})
            
            # Get colors
            base_color = font_config.get('color', '#000000')
            bg_color = output_config.get('bg_color', '#FFFFFF')
            highlight_font_config = highlight_settings.get('font_config', {})
            highlight_color = highlight_font_config.get('color', '#0000FF')
            
            # Parse background color
            if bg_color.startswith('#'):
                bg_hex = bg_color[1:]
                bg_r = int(bg_hex[0:2], 16)
                bg_g = int(bg_hex[2:4], 16) 
                bg_b = int(bg_hex[4:6], 16)
                bg_rgb = (bg_r, bg_g, bg_b)
            else:
                bg_rgb = (255, 255, 255)  # Default white
            
            # Create image
            image = Image.new('RGB', (output_width, output_height), bg_rgb)
            draw = ImageDraw.Draw(image)
            
            # Load fonts with proper weight support
            font_name = font_config.get('font_name', 'Arial')
            font_size = font_config.get('font_size', 20)
            font_weight = font_config.get('font_weight', 'normal')
            
            # Get highlight font properties
            highlight_font_name = highlight_font_config.get('font_name', font_name)
            highlight_font_size = highlight_font_config.get('font_size', font_size)
            highlight_font_weight = highlight_font_config.get('font_weight', font_weight)
            
            # Load fonts with style support using text generator
            base_font = self.text_generator.get_font_with_style(font_name, font_size, font_weight)
            highlight_font = self.text_generator.get_font_with_style(highlight_font_name, highlight_font_size, highlight_font_weight)
            
            # Parse colors
            base_rgb = self._parse_color(base_color)
            highlight_rgb = self._parse_color(highlight_color)
            
            # Split text into lines and words
            lines = full_text.split('\n')
            
            # Build word-to-index mapping for accurate highlighting
            word_index_map = self._build_word_index_map(lines, all_words)
            
            # Get highlighted word index for precise matching
            highlighted_index = highlighted_word.get('index', -1) if highlighted_word else -1
            
            logger.debug(f"Highlighted word index: {highlighted_index}, word: '{highlighted_word.get('word', '') if highlighted_word else None}'")
            
            # Layout configuration - use highlight margins if available
            highlight_layout_config = highlight_settings.get('layout_config', {})
            text_align = layout_config.get('alignment', 'center')
            
            # Use highlight-specific margins if provided, fallback to base margins
            margin_x = highlight_layout_config.get('margin_x', layout_config.get('margin_x', 10))
            margin_y = highlight_layout_config.get('margin_y', layout_config.get('margin_y', 10))
            margin_width = highlight_layout_config.get('margin_width', margin_x)  # Additional margin width for highlights
            margin_height = highlight_layout_config.get('margin_height', margin_y)  # Additional margin height for highlights
            
            line_spacing = layout_config.get('line_spacing', 1.2)
            
            logger.debug(f"Using margins - base: ({margin_x}, {margin_y}), highlight extra: ({margin_width}, {margin_height})")
            
            # Calculate line height
            line_height = int(font_size * line_spacing)
            
            # Calculate starting Y position for vertical centering
            total_text_height = len(lines) * line_height
            start_y = (output_height - total_text_height) // 2
            start_y = max(margin_y, start_y)
            
            # Render each line
            for line_idx, line_text in enumerate(lines):
                if not line_text.strip():
                    continue
                    
                line_words = line_text.split()
                y_pos = start_y + (line_idx * line_height)
                
                # Calculate line width for alignment
                line_width = 0
                for word in line_words:
                    word_width = draw.textbbox((0, 0), word + " ", font=base_font)[2]
                    line_width += word_width
                
                # Calculate starting X position based on alignment
                if text_align == 'center':
                    start_x = (output_width - line_width) // 2
                elif text_align == 'right':
                    start_x = output_width - line_width - margin_x
                else:  # left
                    start_x = margin_x
                
                start_x = max(margin_x, start_x)
                
                # Render each word in the line
                current_x = start_x
                for word_pos, word in enumerate(line_words):
                    # Check if this word should be highlighted using index-based matching
                    is_highlighted = False
                    word_index = word_index_map.get((line_idx, word_pos), -1)
                    
                    if highlighted_index >= 0 and word_index == highlighted_index:
                        # Verify timing as additional check - handle both Whisper and processed formats
                        word_start = highlighted_word.get('start', highlighted_word.get('start_time', 0))
                        word_end = highlighted_word.get('end', highlighted_word.get('end_time', 0))
                        if word_start <= current_time <= word_end:
                            is_highlighted = True
                            logger.debug(f"Highlighting word '{word}' at position ({line_idx}, {word_pos}) with index {word_index} at time {current_time:.2f}s (range: {word_start:.2f}-{word_end:.2f}s)")
                        else:
                            logger.debug(f"Word '{word}' index matches but timing failed: {current_time:.2f}s not in [{word_start:.2f}s, {word_end:.2f}s]")
                    
                    # Choose font and color
                    if is_highlighted:
                        word_font = highlight_font
                        word_color = highlight_rgb
                        # Apply additional margins for highlighted words for better spacing
                        word_x = current_x + margin_width
                        word_y = y_pos - margin_height
                        logger.debug(f"Applying highlight margins: word '{word}' at ({word_x}, {word_y}) with margins (+{margin_width}, -{margin_height})")
                    else:
                        word_font = base_font
                        word_color = base_rgb
                        word_x = current_x
                        word_y = y_pos
                    
                    # Draw the word
                    draw.text((word_x, word_y), word, font=word_font, fill=word_color)
                    
                    # Move to next word position
                    word_width = draw.textbbox((0, 0), word + " ", font=word_font)[2]
                    current_x += word_width
            
            # Convert to numpy array
            result_array = np.array(image)
            return self._standardize_frame_format(result_array)
            
        except Exception as e:
            logger.error(f"Error in dynamic highlighting: {e}")
            # Fallback to base rendering
            return self._render_text_with_settings(full_text, base_settings)
    
    def _load_system_font(self, font_name: str, font_size: int) -> ImageFont.FreeTypeFont:
        """Load a system font or return default."""
        font_paths = {
            'Arial': ['/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'],
            'Helvetica': ['/System/Library/Fonts/Helvetica.ttc', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'],
            'Times': ['/System/Library/Fonts/Supplemental/Times New Roman.ttf', '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf']
        }
        
        for path in font_paths.get(font_name, font_paths['Arial']):
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, font_size)
                except Exception:
                    continue
        
        return ImageFont.load_default()
    
    def _parse_color(self, color_str: str) -> Tuple[int, int, int]:
        """Parse color string to RGB tuple."""
        if color_str.startswith('#'):
            color_hex = color_str[1:]
            try:
                r = int(color_hex[0:2], 16)
                g = int(color_hex[2:4], 16)
                b = int(color_hex[4:6], 16)
                return (r, g, b)
            except (ValueError, IndexError):
                return (0, 0, 0)  # Default black
        else:
            return (0, 0, 0)  # Default black
    
    def _build_word_index_map(self, lines: List[str], all_words: List[Dict]) -> Dict[Tuple[int, int], int]:
        """
        Build mapping from (line_index, word_position) to word timing data index.
        Works with both raw Whisper format and processed format.
        
        Args:
            lines: List of text lines
            all_words: List of word timing data dictionaries (raw Whisper format)
            
        Returns:
            Dict mapping (line_idx, word_pos) tuples to word indices
        """
        word_index_map = {}
        
        # Create a flat list of all words from the text lines
        text_words = []
        for line_idx, line_text in enumerate(lines):
            line_words = line_text.split()
            for word_pos, word in enumerate(line_words):
                text_words.append({
                    'word': word.strip(),
                    'line_idx': line_idx,
                    'word_pos': word_pos
                })
        
        # Match text words with timing data words by order
        logger.debug(f"Mapping {len(text_words)} text words to {len(all_words)} timing words")
        
        for text_idx, text_word in enumerate(text_words):
            if text_idx < len(all_words):
                timing_word = all_words[text_idx]
                # Use array index as the timing index (since raw Whisper data doesn't have index field)
                timing_index = timing_word.get('index', text_idx)
                
                # Map (line_idx, word_pos) -> timing_index  
                key = (text_word['line_idx'], text_word['word_pos'])
                word_index_map[key] = timing_index
                
                # Get word from timing data for verification
                timing_word_text = timing_word.get('word', '').strip()
                
                logger.debug(f"Mapped text '{text_word['word']}' at ({text_word['line_idx']}, {text_word['word_pos']}) -> timing word '{timing_word_text}' with index {timing_index}")
                
                # Warn if words don't match (could indicate alignment issues)
                if text_word['word'].lower() != timing_word_text.lower():
                    logger.warning(f"Word mismatch: text '{text_word['word']}' vs timing '{timing_word_text}' at index {timing_index}")
        
        return word_index_map
    
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