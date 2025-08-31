"""
Subtitle utilities for timing calculations, word processing, and sentence intelligence.
Used by RajSubtitleEngine and related nodes.
"""

import re
from typing import List, Dict, Tuple, Optional, Any
from .utils import logger


def parse_whisper_word_data(word_data: List[Dict]) -> List[Dict]:
    """
    Parse whisper word-level timing data into standardized format.
    
    Expected input format from RajWhisperProcess:
    [{"word": "hello", "start": 0.5, "end": 1.2, "confidence": 0.95}, ...]
    
    Returns standardized format:
    [{"word": "hello", "start_time": 0.5, "end_time": 1.2, "confidence": 0.95}, ...]
    """
    if not word_data:
        logger.warning("Empty word data provided to parse_whisper_word_data")
        return []
    
    parsed_words = []
    for i, word_item in enumerate(word_data):
        try:
            # Handle different possible input formats
            if isinstance(word_item, dict):
                word_text = word_item.get('word', '').strip()
                start_time = float(word_item.get('start', word_item.get('start_time', 0)))
                end_time = float(word_item.get('end', word_item.get('end_time', start_time + 0.5)))
                confidence = float(word_item.get('confidence', word_item.get('score', 1.0)))
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
    
    logger.info(f"Parsed {len(parsed_words)} words from {len(word_data)} input items")
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
    
    Args:
        word_data: Word timing data to validate
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if not word_data:
        issues.append("No word data provided")
        return False, issues
    
    # Check for required fields
    required_fields = ['word', 'start_time', 'end_time']
    for i, word in enumerate(word_data):
        for field in required_fields:
            if field not in word:
                issues.append(f"Word {i} missing required field: {field}")
        
        # Check timing consistency
        if word.get('start_time', 0) >= word.get('end_time', 0):
            issues.append(f"Word {i} '{word.get('word', '')}' has invalid timing: start >= end")
    
    # Check for timing overlaps and gaps
    sorted_words = sorted(word_data, key=lambda w: w.get('start_time', 0))
    for i in range(len(sorted_words) - 1):
        current_word = sorted_words[i]
        next_word = sorted_words[i + 1]
        
        gap = next_word.get('start_time', 0) - current_word.get('end_time', 0)
        if gap < -0.1:  # Allow small overlaps
            issues.append(f"Significant overlap between words: '{current_word.get('word', '')}' and '{next_word.get('word', '')}'")
    
    is_valid = len(issues) == 0
    if is_valid:
        logger.info(f"Word timing data validation passed for {len(word_data)} words")
    else:
        logger.warning(f"Word timing data validation found {len(issues)} issues")
    
    return is_valid, issues