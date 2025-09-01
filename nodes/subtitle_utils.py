"""
Subtitle utilities for timing calculations, word processing, and sentence intelligence.
Used by RajSubtitleEngine and related nodes.
"""

import re
import os
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont
from .utils import logger


def correct_word_timing(word_data: List[Dict]) -> List[Dict]:
    """
    Correct timing issues in word data.
    
    Args:
        word_data: Raw word data that may have timing issues
        
    Returns:
        Corrected word data with valid timing
    """
    if not word_data:
        return []
        
    corrected_words = []
    previous_end = 0.0
    
    for i, word_item in enumerate(word_data):
        if not isinstance(word_item, dict):
            logger.warning(f"Skipping non-dict word item at index {i}: {word_item}")
            continue
            
        word = word_item.copy()  # Don't modify original
        
        # Get timing values with flexible field access
        start_val = word.get('start_time', word.get('start', None))
        end_val = word.get('end_time', word.get('end', None))
        
        if start_val is None or end_val is None:
            logger.warning(f"Word {i} missing timing data, skipping: {word.get('word', 'UNKNOWN')}")
            continue
            
        try:
            start_val = float(start_val)
            end_val = float(end_val)
        except (ValueError, TypeError):
            logger.warning(f"Word {i} has invalid timing values, skipping: {word.get('word', 'UNKNOWN')}")
            continue
        
        # Fix timing issues
        if start_val >= end_val:
            # If start >= end, create a minimum duration
            min_duration = 0.1  # 100ms minimum
            if start_val < previous_end:
                # Overlap with previous word, adjust start
                start_val = previous_end
            end_val = start_val + min_duration
            logger.warning(f"Corrected timing for word {i} '{word.get('word', 'UNKNOWN')}': set duration to {min_duration}s")
        
        # Ensure no overlap with previous word
        if start_val < previous_end:
            start_val = previous_end
            if end_val <= start_val:
                end_val = start_val + 0.1
            logger.warning(f"Corrected overlap for word {i} '{word.get('word', 'UNKNOWN')}': moved start to {start_val:.3f}s")
        
        # Update timing in word data (use both field formats for compatibility)
        word['start_time'] = start_val
        word['end_time'] = end_val
        word['start'] = start_val  # Keep original format too
        word['end'] = end_val
        
        corrected_words.append(word)
        previous_end = end_val
    
    if len(corrected_words) != len(word_data):
        logger.warning(f"Timing correction removed {len(word_data) - len(corrected_words)} words due to invalid data")
    
    return corrected_words


def parse_whisper_word_data(word_data: List[Dict]) -> List[Dict]:
    """
    Parse whisper word-level timing data into standardized format with automatic timing correction.
    
    Expected input format from RajWhisperProcess:
    [{"word": "hello", "start": 0.5, "end": 1.2, "confidence": 0.95}, ...]
    
    Returns standardized format:
    [{"word": "hello", "start_time": 0.5, "end_time": 1.2, "confidence": 0.95}, ...]
    """
    if not word_data:
        logger.warning("Empty word data provided to parse_whisper_word_data")
        return []
    
    # First correct any timing issues
    corrected_data = correct_word_timing(word_data)
    
    parsed_words = []
    for i, word_item in enumerate(corrected_data):
        try:
            # Handle different possible input formats
            if isinstance(word_item, dict):
                word_text = word_item.get('word', '').strip()
                start_time = float(word_item.get('start_time', word_item.get('start', 0)))
                end_time = float(word_item.get('end_time', word_item.get('end', start_time + 0.5)))
                confidence = float(word_item.get('confidence', word_item.get('probability', word_item.get('score', 1.0))))
            else:
                logger.warning(f"Unexpected word item format at index {i}: {word_item}")
                continue
                
            if word_text:  # Only add non-empty words
                parsed_words.append({
                    "word": word_text,
                    "start_time": start_time,
                    "end_time": end_time,
                    "confidence": confidence,
                    "index": i
                })
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing word item at index {i}: {e}")
            continue
    
    logger.info(f"Parsed {len(parsed_words)} words from {len(word_data)} input items (corrected {len(word_data) - len(corrected_data)} timing issues)")
    return parsed_words


def get_total_duration(word_data: List[Dict]) -> float:
    """Extract total duration from word timing data."""
    if not word_data:
        return 0.0
    
    return max(word['end_time'] for word in word_data)


def get_active_words_at_time(word_data: List[Dict], current_time: float) -> List[Dict]:
    """
    Get all words that should be visible at the given time.
    
    Args:
        word_data: Parsed word timing data
        current_time: Current timestamp in seconds
        
    Returns:
        List of word dictionaries that should be displayed at current_time
    """
    active_words = []
    for word in word_data:
        if word['start_time'] <= current_time <= word['end_time']:
            active_words.append(word)
    
    return active_words


def get_current_highlighted_word(word_data: List[Dict], current_time: float) -> Optional[Dict]:
    """
    Get the word that should be highlighted at the current time.
    Usually the word that is currently being spoken.
    
    Args:
        word_data: Parsed word timing data
        current_time: Current timestamp in seconds
        
    Returns:
        Word dictionary to highlight, or None if no word is active
    """
    for word in word_data:
        if word['start_time'] <= current_time <= word['end_time']:
            return word
    
    return None


def detect_sentence_boundaries(word_data: List[Dict], min_gap: float = 0.3) -> List[List[Dict]]:
    """
    Detect sentence boundaries based on timing gaps and punctuation.
    
    Args:
        word_data: Parsed word timing data
        min_gap: Minimum gap in seconds to consider a sentence boundary
        
    Returns:
        List of sentences, where each sentence is a list of word dictionaries
    """
    if not word_data:
        return []
    
    sentences = []
    current_sentence = []
    
    for i, word in enumerate(word_data):
        current_sentence.append(word)
        
        # Check for sentence-ending punctuation
        has_punctuation = bool(re.search(r'[.!?]', word['word']))
        
        # Check for timing gap to next word
        has_gap = False
        if i < len(word_data) - 1:
            next_word = word_data[i + 1]
            gap = next_word['start_time'] - word['end_time']
            has_gap = gap >= min_gap
        
        # End sentence if we have punctuation or significant gap, or it's the last word
        if has_punctuation or has_gap or i == len(word_data) - 1:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
    
    logger.info(f"Detected {len(sentences)} sentences from {len(word_data)} words")
    return sentences


def apply_dual_tolerance_logic(words: List[Dict], 
                             words_per_line: int,
                             sentence_tolerance: int,
                             word_tolerance: int) -> Tuple[List[List[Dict]], str]:
    """
    Apply both sentence and word tolerance, return best result.
    
    Args:
        words: List of word dictionaries
        words_per_line: Target words per line
        sentence_tolerance: Sentence-level tolerance
        word_tolerance: Word-level tolerance
        
    Returns:
        Tuple of (best_organization, method_used)
    """
    if sentence_tolerance == 0 and word_tolerance == 0:
        # No tolerance, use basic organization
        return _organize_words_basic(words, words_per_line), "basic"
    
    results = {}
    
    # Try basic organization
    basic_result = _organize_words_basic(words, words_per_line)
    results["basic"] = basic_result
    
    # Try sentence tolerance if enabled
    if sentence_tolerance > 0:
        sentence_result = _organize_words_with_sentence_tolerance(
            words, words_per_line, sentence_tolerance
        )
        results["sentence"] = sentence_result
    
    # Try word tolerance if enabled
    if word_tolerance > 0:
        word_result = _organize_words_with_word_tolerance(
            words, words_per_line, word_tolerance
        )
        results["word"] = word_result
    
    # Try combined tolerance if both enabled
    if sentence_tolerance > 0 and word_tolerance > 0:
        combined_result = _organize_words_with_combined_tolerance(
            words, words_per_line, sentence_tolerance, word_tolerance
        )
        results["combined"] = combined_result
    
    # Evaluate which result is best
    best_result, best_method = _choose_best_organization(results)
    
    # If all methods produce same result, just use basic
    if len(set(str(result) for result in results.values())) == 1:
        return basic_result, "same_result_skip"
    
    return best_result, best_method


def _organize_words_basic(words: List[Dict], words_per_line: int) -> List[List[Dict]]:
    """Basic word organization without tolerance."""
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        if len(current_line) >= words_per_line:
            lines.append(current_line)
            current_line = []
    
    if current_line:
        lines.append(current_line)
    
    return lines


def _organize_words_with_sentence_tolerance(words: List[Dict], 
                                          words_per_line: int, 
                                          sentence_tolerance: int) -> List[List[Dict]]:
    """Organize words with sentence-ending preference."""
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        
        if len(current_line) >= words_per_line - sentence_tolerance:
            # Look ahead for sentence endings within tolerance
            should_end_line = False
            
            if len(current_line) >= words_per_line + sentence_tolerance:
                should_end_line = True
            elif re.search(r'[.!?]', word['word']):
                should_end_line = True
            elif len(current_line) == words_per_line:
                should_end_line = True
            
            if should_end_line:
                lines.append(current_line)
                current_line = []
    
    if current_line:
        lines.append(current_line)
    
    return lines


def _organize_words_with_word_tolerance(words: List[Dict], 
                                      words_per_line: int, 
                                      word_tolerance: int) -> List[List[Dict]]:
    """Organize words with word-level adjustments."""
    lines = []
    current_line = []
    
    target_min = max(1, words_per_line - word_tolerance)
    target_max = words_per_line + word_tolerance
    
    for word in words:
        current_line.append(word)
        
        if len(current_line) >= target_min:
            # Look for optimal break point within tolerance range
            if len(current_line) >= target_max:
                lines.append(current_line)
                current_line = []
            elif len(current_line) == words_per_line:
                lines.append(current_line)
                current_line = []
    
    if current_line:
        lines.append(current_line)
    
    return lines


def _organize_words_with_combined_tolerance(words: List[Dict], 
                                          words_per_line: int,
                                          sentence_tolerance: int, 
                                          word_tolerance: int) -> List[List[Dict]]:
    """Combine both tolerance types for optimal organization."""
    lines = []
    current_line = []
    
    word_min = max(1, words_per_line - word_tolerance)
    word_max = words_per_line + word_tolerance
    sent_min = max(1, words_per_line - sentence_tolerance) 
    sent_max = words_per_line + sentence_tolerance
    
    # Use the more restrictive bounds
    min_words = max(word_min, sent_min)
    max_words = min(word_max, sent_max)
    
    for word in words:
        current_line.append(word)
        
        if len(current_line) >= min_words:
            should_end_line = False
            
            # Priority 1: Sentence endings
            if re.search(r'[.!?]', word['word']) and len(current_line) <= sent_max:
                should_end_line = True
            # Priority 2: Reached max tolerance
            elif len(current_line) >= max_words:
                should_end_line = True
            # Priority 3: Reached target
            elif len(current_line) == words_per_line:
                should_end_line = True
            
            if should_end_line:
                lines.append(current_line)
                current_line = []
    
    if current_line:
        lines.append(current_line)
    
    return lines


def _choose_best_organization(results: Dict[str, List[List[Dict]]]) -> Tuple[List[List[Dict]], str]:
    """Choose the best word organization from multiple options."""
    # Scoring criteria (lower is better)
    def score_organization(lines: List[List[Dict]]) -> float:
        if not lines:
            return float('inf')
        
        score = 0.0
        
        # Penalty for very uneven line lengths
        line_lengths = [len(line) for line in lines]
        avg_length = sum(line_lengths) / len(line_lengths)
        
        for length in line_lengths:
            deviation = abs(length - avg_length)
            score += deviation * 0.5
        
        # Bonus for lines ending with sentence punctuation
        for line in lines:
            if line and re.search(r'[.!?]', line[-1]['word']):
                score -= 1.0  # Bonus
        
        # Penalty for too many very short lines
        short_lines = sum(1 for length in line_lengths if length < 3)
        score += short_lines * 2.0
        
        return score
    
    best_score = float('inf')
    best_result = None
    best_method = "basic"
    
    for method, organization in results.items():
        score = score_organization(organization)
        if score < best_score:
            best_score = score
            best_result = organization
            best_method = method
    
    return best_result or [], best_method


def organize_words_into_display_lines(words: List[Dict], 
                                    words_per_line: int = 0, 
                                    max_lines: int = 0,
                                    sentence_tolerance: int = 1,
                                    word_tolerance: int = 0,
                                    auto_fit: bool = True) -> List[List[Dict]]:
    """
    Organize words into display lines with dual tolerance intelligence.
    
    Args:
        words: List of word dictionaries to organize
        words_per_line: Target words per line (0 = auto)
        max_lines: Maximum lines to display (0 = unlimited)
        sentence_tolerance: Allow +/- words per line for complete sentences
        word_tolerance: Allow +/- words per line for word boundaries
        auto_fit: Automatically optimize line breaks
        
    Returns:
        List of lines, where each line is a list of word dictionaries
    """
    if not words:
        return []
    
    if words_per_line <= 0:
        # Auto-determine words per line based on total words and max_lines
        if max_lines > 0:
            words_per_line = max(1, len(words) // max_lines)
        else:
            words_per_line = min(8, max(3, len(words) // 3))  # Reasonable default
    
    # Use dual tolerance logic if tolerances are enabled
    if auto_fit and (sentence_tolerance > 0 or word_tolerance > 0):
        lines, method = apply_dual_tolerance_logic(words, words_per_line, sentence_tolerance, word_tolerance)
        logger.info(f"Organized {len(words)} words into {len(lines)} lines using {method}")
    else:
        # Simple organization without tolerance
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            
            if len(current_line) >= words_per_line:
                lines.append(current_line)
                current_line = []
                
                # Check max_lines limit
                if max_lines > 0 and len(lines) >= max_lines:
                    break
        
        # Add remaining words
        if current_line:
            if max_lines > 0 and len(lines) >= max_lines:
                # Merge with last line if we've hit the limit
                if lines:
                    lines[-1].extend(current_line)
            else:
                lines.append(current_line)
        
        logger.info(f"Organized {len(words)} words into {len(lines)} lines (simple mode)")
    
    # Apply max_lines limit if needed
    if max_lines > 0 and len(lines) > max_lines:
        # Truncate to max_lines and merge overflow into last line
        overflow_words = []
        for line in lines[max_lines:]:
            overflow_words.extend(line)
        
        lines = lines[:max_lines]
        if overflow_words and lines:
            lines[-1].extend(overflow_words)
    
    return lines


def calculate_word_positions(words: List[Dict], 
                           base_settings: Dict,
                           highlight_settings: Optional[Dict] = None) -> List[Dict]:
    """
    Calculate screen positions for words, accounting for highlighting size differences.
    
    Args:
        words: List of word dictionaries
        base_settings: Base text styling settings
        highlight_settings: Highlight text styling settings (optional)
        
    Returns:
        List of word dictionaries with added position information
    """
    # This is a simplified implementation - in practice, you'd need PIL/font metrics
    # to calculate exact positions
    
    positioned_words = []
    x_offset = 0
    
    font_size = base_settings.get('font_config', {}).get('font_size', 36)
    letter_spacing = base_settings.get('layout_config', {}).get('letter_spacing', 0)
    
    for word in words:
        word_copy = word.copy()
        
        # Estimate word width (simplified)
        base_width = len(word['word']) * (font_size * 0.6) + letter_spacing
        
        # Adjust for highlighting if this word might be highlighted
        if highlight_settings:
            highlight_font_size = highlight_settings.get('font_config', {}).get('font_size', font_size)
            highlight_width = len(word['word']) * (highlight_font_size * 0.6) + letter_spacing
            max_width = max(base_width, highlight_width)
        else:
            max_width = base_width
        
        word_copy.update({
            'x_position': x_offset,
            'base_width': base_width,
            'max_width': max_width,
            'y_position': 0  # Will be calculated later based on line
        })
        
        positioned_words.append(word_copy)
        x_offset += max_width + 10  # Add some spacing between words
    
    return positioned_words


def calculate_text_width(text: str, font_path: str, font_size: int) -> int:
    """
    Calculate the pixel width of text using PIL.
    
    Args:
        text: Text to measure
        font_path: Path to font file
        font_size: Font size in points
        
    Returns:
        Width in pixels
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
        # Create a temporary image for measurement
        img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        return width
    except Exception as e:
        logger.warning(f"Could not calculate exact text width: {e}")
        # Fallback to estimation
        return len(text) * int(font_size * 0.6)


def create_word_groups(word_data: List[Dict],
                      display_width: int,
                      font_config: Dict,
                      max_lines: int = 2,
                      margin_x: int = 20) -> List[Dict]:
    """
    Group consecutive words based on display width constraints.
    
    Args:
        word_data: List of word dictionaries with timing
        display_width: Available width in pixels
        font_config: Font configuration dictionary
        max_lines: Maximum number of lines to display at once
        margin_x: Horizontal margin on each side
        
    Returns:
        List of word groups, each containing multiple words with combined timing
    """
    if not word_data:
        return []
    
    # Extract font settings
    font_family = font_config.get('font_family', 'Arial')
    font_size = font_config.get('font_size', 36)
    font_weight = font_config.get('font_weight', 'normal')
    
    # Find font path
    font_paths = {
        'Arial': ['/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'],
        'Helvetica': ['/System/Library/Fonts/Helvetica.ttc', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'],
        'Times': ['/System/Library/Fonts/Supplemental/Times New Roman.ttf', '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf']
    }
    
    font_path = None
    for path in font_paths.get(font_family, font_paths['Arial']):
        if os.path.exists(path):
            font_path = path
            break
    
    if not font_path:
        logger.warning(f"Font {font_family} not found, using estimation")
    
    # Calculate available width
    available_width = display_width - (2 * margin_x)
    
    # Group words
    groups = []
    current_group = []
    current_width = 0
    space_width = calculate_text_width(" ", font_path, font_size) if font_path else int(font_size * 0.3)
    
    for word in word_data:
        word_text = word.get('word', '')
        word_width = calculate_text_width(word_text, font_path, font_size) if font_path else len(word_text) * int(font_size * 0.6)
        
        # Check if adding this word would exceed width
        test_width = current_width + (space_width if current_group else 0) + word_width
        
        if test_width > available_width and current_group:
            # Save current group and start new one
            groups.append({
                'words': current_group.copy(),
                'text': ' '.join(w['word'] for w in current_group),
                'start_time': current_group[0].get('start_time', current_group[0].get('start', 0)),
                'end_time': current_group[-1].get('end_time', current_group[-1].get('end', 0)),
                'word_count': len(current_group)
            })
            current_group = [word]
            current_width = word_width
        else:
            # Add word to current group
            current_group.append(word)
            current_width = test_width
    
    # Add final group
    if current_group:
        groups.append({
            'words': current_group.copy(),
            'text': ' '.join(w['word'] for w in current_group),
            'start_time': current_group[0].get('start_time', current_group[0].get('start', 0)),
            'end_time': current_group[-1].get('end_time', current_group[-1].get('end', 0)),
            'word_count': len(current_group)
        })
    
    logger.info(f"Created {len(groups)} word groups from {len(word_data)} words")
    return groups


def debug_word_groups(word_data: List[Dict],
                     display_width: int,
                     font_config: Dict,
                     max_lines: int = 2,
                     margin_x: int = 20,
                     save_images: bool = True) -> List[Dict]:
    """
    Debug word grouping by creating test images for each group.
    
    Args:
        word_data: List of word dictionaries with timing
        display_width: Available width in pixels
        font_config: Font configuration dictionary
        max_lines: Maximum number of lines to display
        margin_x: Horizontal margin on each side
        save_images: Whether to save debug images
        
    Returns:
        List of word groups with debug information
    """
    from PIL import Image, ImageDraw, ImageFont
    import tempfile
    
    # Create word groups
    groups = create_word_groups(word_data, display_width, font_config, max_lines, margin_x)
    
    # Extract font settings
    font_family = font_config.get('font_family', 'Arial')
    font_size = font_config.get('font_size', 36)
    
    # Find font path
    font_paths = {
        'Arial': ['/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'],
        'Helvetica': ['/System/Library/Fonts/Helvetica.ttc', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'],
        'Times': ['/System/Library/Fonts/Supplemental/Times New Roman.ttf', '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf']
    }
    
    font_path = None
    for path in font_paths.get(font_family, font_paths['Arial']):
        if os.path.exists(path):
            font_path = path
            break
    
    available_width = display_width - (2 * margin_x)
    
    logger.info(f"\n=== WORD GROUPING DEBUG ===")
    logger.info(f"Display width: {display_width}px")
    logger.info(f"Available width: {available_width}px (margin: {margin_x}px each side)")
    logger.info(f"Font: {font_family} @ {font_size}px")
    logger.info(f"Font path: {font_path}")
    
    debug_info = []
    
    for i, group in enumerate(groups):
        group_text = group['text']
        
        # Calculate actual text width
        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
                img = Image.new('RGB', (1, 1))
                draw = ImageDraw.Draw(img)
                bbox = draw.textbbox((0, 0), group_text, font=font)
                actual_width = bbox[2] - bbox[0]
                
                # Create test image if requested
                if save_images:
                    # Create image with background and text
                    test_img = Image.new('RGB', (display_width, 100), color='black')
                    test_draw = ImageDraw.Draw(test_img)
                    
                    # Draw available area boundary
                    test_draw.rectangle([margin_x, 10, display_width - margin_x, 90], outline='red', width=2)
                    
                    # Draw text
                    text_x = margin_x
                    text_y = 30
                    test_draw.text((text_x, text_y), group_text, font=font, fill='white')
                    
                    # Save debug image
                    temp_dir = tempfile.gettempdir()
                    img_path = os.path.join(temp_dir, f"word_group_{i+1}.png")
                    test_img.save(img_path)
                    logger.info(f"Saved debug image: {img_path}")
                    
            except Exception as e:
                logger.warning(f"Could not create debug image for group {i+1}: {e}")
                actual_width = len(group_text) * int(font_size * 0.6)  # Fallback
        else:
            actual_width = len(group_text) * int(font_size * 0.6)  # Fallback
        
        # Check if text fits
        fits = actual_width <= available_width
        overflow = actual_width - available_width if not fits else 0
        
        debug_info.append({
            'group_num': i + 1,
            'text': group_text,
            'word_count': group['word_count'],
            'actual_width': actual_width,
            'available_width': available_width,
            'fits': fits,
            'overflow_pixels': overflow,
            'duration': group['end_time'] - group['start_time']
        })
        
        status = "✅ FITS" if fits else f"❌ OVERFLOW ({overflow}px)"
        logger.info(f'Group {i+1}: "{group_text}" - {actual_width}px {status}')
    
    logger.info("=== DEBUG COMPLETE ===\n")
    
    return debug_info


def create_area_based_word_groups(word_data: List[Dict],
                                 box_width: int,
                                 box_height: int,
                                 font_config: Dict,
                                 margin_x: int = 20,
                                 margin_y: int = 20,
                                 line_spacing: float = 1.2) -> List[Dict]:
    """
    Group words based on available box area instead of fixed line limits.
    
    Args:
        word_data: List of word dictionaries with timing
        box_width: Available width of the subtitle box in pixels
        box_height: Available height of the subtitle box in pixels
        font_config: Font configuration dictionary
        margin_x: Horizontal margin on each side
        margin_y: Vertical margin on top and bottom
        line_spacing: Line spacing multiplier
        
    Returns:
        List of word groups that fit within the box dimensions
    """
    if not word_data:
        return []
    
    # Extract font settings
    font_family = font_config.get('font_family', font_config.get('font_name', 'Arial'))
    font_size = font_config.get('font_size', 36)
    font_weight = font_config.get('font_weight', 'normal')
    
    # Find font path
    font_paths = {
        'Arial': ['/System/Library/Fonts/Supplemental/Arial.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'],
        'Helvetica': ['/System/Library/Fonts/Helvetica.ttc', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'],
        'Times': ['/System/Library/Fonts/Supplemental/Times New Roman.ttf', '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf']
    }
    
    font_path = None
    for path in font_paths.get(font_family, font_paths['Arial']):
        if os.path.exists(path):
            font_path = path
            break
    
    if not font_path:
        logger.warning(f"Font {font_family} not found, using estimation")
    
    # Calculate available dimensions
    available_width = box_width - (2 * margin_x)
    available_height = box_height - (2 * margin_y)
    
    # Calculate line height with spacing
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
            # Create temporary image to measure text
            temp_img = Image.new('RGB', (1, 1))
            draw = ImageDraw.Draw(temp_img)
            # Measure a sample text to get line height
            bbox = draw.textbbox((0, 0), "Ag", font=font)  # 'Ag' has ascenders and descenders
            line_height = (bbox[3] - bbox[1]) * line_spacing
        except Exception as e:
            logger.warning(f"Could not load font for measurement: {e}")
            line_height = font_size * line_spacing
    else:
        line_height = font_size * line_spacing
    
    # Calculate maximum number of lines that fit in the box
    max_lines_in_box = max(1, int(available_height / line_height))
    
    logger.info(f"Box dimensions: {box_width}x{box_height}px, Available: {available_width}x{available_height}px")
    logger.info(f"Line height: {line_height:.1f}px, Max lines in box: {max_lines_in_box}")
    
    # Group words to fit within the box area
    groups = []
    current_group_words = []
    current_lines = []  # List of lines in current group
    current_line_words = []  # Words in current line
    current_line_width = 0
    
    space_width = calculate_text_width(" ", font_path, font_size) if font_path else int(font_size * 0.3)
    
    for word in word_data:
        word_text = word.get('word', '')
        word_width = calculate_text_width(word_text, font_path, font_size) if font_path else len(word_text) * int(font_size * 0.6)
        
        # Check if adding this word would exceed line width
        test_width = current_line_width + (space_width if current_line_words else 0) + word_width
        
        if test_width > available_width and current_line_words:
            # Current line is full, move to next line
            current_lines.append({
                'words': current_line_words.copy(),
                'text': ' '.join(w['word'] for w in current_line_words),
                'width': current_line_width
            })
            current_line_words = [word]
            current_line_width = word_width
            
            # Check if we've reached the maximum lines for this group
            if len(current_lines) >= max_lines_in_box:
                # Create a group with current lines
                group_start = min(w['start_time'] for line in current_lines for w in line['words'])
                group_end = max(w['end_time'] for line in current_lines for w in line['words'])
                all_group_words = [w for line in current_lines for w in line['words']]
                
                groups.append({
                    'words': all_group_words,
                    'lines': current_lines.copy(),
                    'text': '\n'.join(line['text'] for line in current_lines),
                    'start_time': group_start,
                    'end_time': group_end,
                    'word_count': len(all_group_words),
                    'line_count': len(current_lines),
                    'fits_in_box': True
                })
                
                # Start new group
                current_lines = []
                current_group_words = []
        else:
            # Add word to current line
            current_line_words.append(word)
            current_line_width = test_width
    
    # Handle remaining words
    if current_line_words:
        current_lines.append({
            'words': current_line_words.copy(),
            'text': ' '.join(w['word'] for w in current_line_words),
            'width': current_line_width
        })
    
    if current_lines:
        # Create final group
        group_start = min(w['start_time'] for line in current_lines for w in line['words'])
        group_end = max(w['end_time'] for line in current_lines for w in line['words'])
        all_group_words = [w for line in current_lines for w in line['words']]
        
        groups.append({
            'words': all_group_words,
            'lines': current_lines.copy(),
            'text': '\n'.join(line['text'] for line in current_lines),
            'start_time': group_start,
            'end_time': group_end,
            'word_count': len(all_group_words),
            'line_count': len(current_lines),
            'fits_in_box': True
        })
    
    logger.info(f"Created {len(groups)} area-based word groups from {len(word_data)} words")
    logger.info(f"Groups have an average of {sum(g['line_count'] for g in groups) / len(groups):.1f} lines each")
    
    return groups


def create_line_groups(word_groups: List[Dict], max_lines: int = 2) -> List[Dict]:
    """
    Group word groups (lines) into line groups based on max_lines parameter.
    Each line group represents a frame and contains multiple lines (word groups).
    
    Args:
        word_groups: List of word groups from create_word_groups (each becomes a line)
        max_lines: Maximum number of lines per line group (frame)
        
    Returns:
        List of line groups, each containing multiple word groups (lines) with combined timing
    """
    if not word_groups:
        return []
    
    if max_lines <= 0:
        max_lines = 2  # Default fallback
    
    line_groups = []
    current_lines = []
    
    for word_group in word_groups:
        current_lines.append(word_group)
        
        # When we reach max_lines, create a line group
        if len(current_lines) >= max_lines:
            # Calculate combined timing for the line group
            start_time = min(line['start_time'] for line in current_lines)
            end_time = max(line['end_time'] for line in current_lines)
            
            # Create line group
            line_group = {
                'lines': current_lines.copy(),  # List of word groups (lines)
                'line_count': len(current_lines),
                'start_time': start_time,
                'end_time': end_time,
                'combined_text': '\n'.join(line['text'] for line in current_lines),
                'total_words': sum(line['word_count'] for line in current_lines)
            }
            
            line_groups.append(line_group)
            current_lines = []
    
    # Handle remaining lines
    if current_lines:
        start_time = min(line['start_time'] for line in current_lines)
        end_time = max(line['end_time'] for line in current_lines)
        
        line_group = {
            'lines': current_lines.copy(),
            'line_count': len(current_lines),
            'start_time': start_time,
            'end_time': end_time,
            'combined_text': '\n'.join(line['text'] for line in current_lines),
            'total_words': sum(line['word_count'] for line in current_lines)
        }
        
        line_groups.append(line_group)
    
    logger.info(f"Created {len(line_groups)} line groups from {len(word_groups)} word groups (lines) with max {max_lines} lines per group")
    return line_groups


def create_timing_windows_grouped(word_groups: List[Dict], 
                                 fps: float,
                                 max_lines: int = 2) -> Dict[int, List[Dict]]:
    """
    Create frame-by-frame timing windows for word groups.
    Shows entire groups for their full duration.
    
    Args:
        word_groups: List of word groups from create_word_groups
        fps: Video frame rate
        max_lines: Maximum lines to display simultaneously
        
    Returns:
        Dictionary mapping frame numbers to lists of word groups to display
    """
    if not word_groups:
        return {}
    
    # Calculate total duration
    total_duration = max(group['end_time'] for group in word_groups)
    total_frames = int(total_duration * fps) + 1
    
    timing_windows = {}
    
    for frame_num in range(total_frames):
        current_time = frame_num / fps
        
        # Find groups that should be displayed at this time
        active_groups = []
        for group in word_groups:
            if group['start_time'] <= current_time <= group['end_time']:
                active_groups.append(group)
        
        # Limit to max_lines most recent groups
        if len(active_groups) > max_lines:
            # Sort by start time and keep the most recent ones
            active_groups = sorted(active_groups, key=lambda g: g['start_time'])[-max_lines:]
        
        timing_windows[frame_num] = active_groups
    
    logger.info(f"Created grouped timing windows for {total_frames} frames at {fps}fps")
    return timing_windows


def create_timing_windows_line_grouped(line_groups: List[Dict], fps: float) -> Dict[int, List[Dict]]:
    """
    Create frame-by-frame timing windows for line groups.
    Each frame shows one complete line group (containing multiple lines).
    
    Args:
        line_groups: List of line groups from create_line_groups
        fps: Video frame rate
        
    Returns:
        Dictionary mapping frame numbers to line groups to display
    """
    if not line_groups:
        return {}
    
    # Calculate total duration
    total_duration = max(group['end_time'] for group in line_groups)
    total_frames = int(total_duration * fps) + 1
    
    timing_windows = {}
    
    for frame_num in range(total_frames):
        current_time = frame_num / fps
        
        # Find line groups that should be displayed at this time
        active_line_groups = []
        for line_group in line_groups:
            if line_group['start_time'] <= current_time <= line_group['end_time']:
                active_line_groups.append(line_group)
        
        # Since line groups are designed to be mutually exclusive frames,
        # we should typically have at most one active line group per frame
        # But if there's overlap, take the most recent one
        if len(active_line_groups) > 1:
            # Sort by start time and take the most recent
            active_line_groups = sorted(active_line_groups, key=lambda g: g['start_time'])[-1:]
        
        timing_windows[frame_num] = active_line_groups
    
    logger.info(f"Created line-grouped timing windows for {total_frames} frames at {fps}fps")
    return timing_windows


def create_timing_windows_area_based(area_groups: List[Dict], fps: float) -> Dict[int, List[Dict]]:
    """
    Create frame-by-frame timing windows for area-based word groups.
    Each frame shows one complete area-based group (containing multiple lines that fit in the box).
    
    Args:
        area_groups: List of area-based word groups from create_area_based_word_groups
        fps: Video frame rate
        
    Returns:
        Dictionary mapping frame numbers to area-based groups to display
    """
    if not area_groups:
        return {}
    
    # Calculate total duration
    total_duration = max(group['end_time'] for group in area_groups)
    total_frames = int(total_duration * fps) + 1
    
    timing_windows = {}
    
    for frame_num in range(total_frames):
        current_time = frame_num / fps
        
        # Find area groups that should be displayed at this time
        active_area_groups = []
        for area_group in area_groups:
            if area_group['start_time'] <= current_time <= area_group['end_time']:
                active_area_groups.append(area_group)
        
        # Since area groups are designed to be mutually exclusive frames,
        # we should typically have at most one active area group per frame
        # But if there's overlap, take the most recent one
        if len(active_area_groups) > 1:
            # Sort by start time and take the most recent
            active_area_groups = sorted(active_area_groups, key=lambda g: g['start_time'])[-1:]
        
        timing_windows[frame_num] = active_area_groups
    
    logger.info(f"Created area-based timing windows for {total_frames} frames at {fps}fps")
    return timing_windows


def create_timing_windows(word_data: List[Dict], 
                         fps: float, 
                         lead_time: float = 0.0,
                         trail_time: float = 0.0) -> Dict[int, List[Dict]]:
    """
    Create frame-by-frame timing windows showing which words are active.
    
    Args:
        word_data: Parsed word timing data
        fps: Video frame rate
        lead_time: Show words this many seconds early
        trail_time: Keep words visible this many seconds after they end
        
    Returns:
        Dictionary mapping frame numbers to lists of active words
    """
    if not word_data:
        return {}
    
    total_duration = get_total_duration(word_data)
    total_frames = int(total_duration * fps) + 1
    
    timing_windows = {}
    
    for frame_num in range(total_frames):
        current_time = frame_num / fps
        
        active_words = []
        for word in word_data:
            # Adjust timing with lead/trail
            adjusted_start = word['start_time'] - lead_time
            adjusted_end = word['end_time'] + trail_time
            
            if adjusted_start <= current_time <= adjusted_end:
                active_words.append(word)
        
        timing_windows[frame_num] = active_words
    
    logger.info(f"Created timing windows for {total_frames} frames at {fps}fps")
    return timing_windows


def validate_word_timing_data(word_data: List[Dict]) -> Tuple[bool, List[str]]:
    """
    Validate word timing data for potential issues.
    Support both Whisper format (start/end) and processed format (start_time/end_time).
    
    Args:
        word_data: Word timing data to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if not word_data:
        issues.append("No word data provided")
        return False, issues
    
    logger.debug(f"Validating {len(word_data)} words. Sample word structure: {word_data[0] if word_data else 'None'}")
    
    # Check for required fields - support both formats
    for i, word in enumerate(word_data):
        if not isinstance(word, dict):
            issues.append(f"Word {i} is not a dictionary: {type(word)}")
            continue
            
        # Debug: Show actual fields for problematic words
        if i == 17 or i >= len(word_data) - 5:  # Log last 5 words and word 17 specifically
            logger.debug(f"Word {i} fields: {list(word.keys())}, values: {word}")
        
        # Check word field
        if 'word' not in word:
            issues.append(f"Word {i} missing 'word' field. Available fields: {list(word.keys())}")
        
        # Check timing fields - accept either start/end or start_time/end_time
        has_start = 'start' in word or 'start_time' in word
        has_end = 'end' in word or 'end_time' in word
        
        if not has_start:
            issues.append(f"Word {i} missing timing field: need 'start' or 'start_time'. Available fields: {list(word.keys())}")
        if not has_end:
            issues.append(f"Word {i} missing timing field: need 'end' or 'end_time'. Available fields: {list(word.keys())}")
        
        # Check timing consistency - use flexible field access with better error handling
        start_val = word.get('start_time', word.get('start', None))
        end_val = word.get('end_time', word.get('end', None))
        
        # Handle None values
        if start_val is None:
            issues.append(f"Word {i} '{word.get('word', 'UNKNOWN')}' has no start timing value")
            continue
        if end_val is None:
            issues.append(f"Word {i} '{word.get('word', 'UNKNOWN')}' has no end timing value")
            continue
            
        # Convert to float and validate
        try:
            start_val = float(start_val)
            end_val = float(end_val)
        except (ValueError, TypeError) as e:
            issues.append(f"Word {i} '{word.get('word', 'UNKNOWN')}' has invalid timing values: start={start_val}, end={end_val} ({e})")
            continue
            
        # Check for invalid timing with tolerance for very short durations
        if start_val >= end_val:
            duration = end_val - start_val
            if abs(duration) < 0.001:  # Less than 1ms - likely a rounding error
                logger.warning(f"Word {i} '{word.get('word', 'UNKNOWN')}' has very short duration ({duration:.6f}s), allowing it")
            else:
                issues.append(f"Word {i} '{word.get('word', 'UNKNOWN')}' has invalid timing: start >= end ({start_val:.3f} >= {end_val:.3f}, duration: {duration:.3f}s)")
    
    # Check for timing overlaps and gaps - use flexible field access
    sorted_words = sorted(word_data, key=lambda w: w.get('start_time', w.get('start', 0)))
    for i in range(len(sorted_words) - 1):
        current_word = sorted_words[i]
        next_word = sorted_words[i + 1]
        
        current_end = current_word.get('end_time', current_word.get('end', 0))
        next_start = next_word.get('start_time', next_word.get('start', 0))
        gap = next_start - current_end
        
        if gap < -0.1:  # Allow small overlaps
            issues.append(f"Significant overlap between words: '{current_word.get('word', '')}' and '{next_word.get('word', '')}'")
    
    is_valid = len(issues) == 0
    if is_valid:
        logger.info(f"Word timing data validation passed for {len(word_data)} words")
    else:
        logger.warning(f"Word timing data validation found {len(issues)} issues")
    
    return is_valid, issues