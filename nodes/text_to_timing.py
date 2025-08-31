"""
RajTextToTiming - Convert plain text to whisper-compatible timing data for testing.
Useful for testing subtitle workflows without needing actual audio files.
"""

import random
import re
from typing import List, Dict, Tuple
from .utils import logger


class RajTextToTiming:
    """
    Convert plain text to whisper-compatible word timing data.
    Useful for testing subtitle systems without audio processing.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello world! This is a test sentence for subtitle generation. Each word will have timing data.",
                    "tooltip": "Text to convert to timed words"
                }),
                "timing_mode": (["equal", "random", "realistic"], {
                    "default": "realistic",
                    "tooltip": "How to distribute word timing"
                })
            },
            "optional": {
                "word_duration_min": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Minimum word duration in seconds"
                }),
                "word_duration_max": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.2,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Maximum word duration in seconds"
                }),
                "sentence_pause": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Pause duration between sentences"
                }),
                "total_duration": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 300.0,
                    "step": 0.1,
                    "tooltip": "Total duration (0 = auto-calculate)"
                }),
                "speech_rate": ("FLOAT", {
                    "default": 150.0,
                    "min": 50.0,
                    "max": 400.0,
                    "step": 10.0,
                    "tooltip": "Words per minute for realistic timing"
                })
            }
        }
    
    RETURN_TYPES = ("LIST", "STRING", "FLOAT")
    RETURN_NAMES = ("word_timings", "timing_preview", "total_duration")
    FUNCTION = "convert_text_to_timing"
    CATEGORY = "Raj Video Processing 🎬/Subtitles"
    
    def convert_text_to_timing(self,
                             text: str,
                             timing_mode: str = "realistic",
                             word_duration_min: float = 0.3,
                             word_duration_max: float = 0.8,
                             sentence_pause: float = 0.5,
                             total_duration: float = 0.0,
                             speech_rate: float = 150.0) -> Tuple[List[Dict], str, float]:
        """
        Convert text to whisper-compatible word timing data.
        
        Args:
            text: Input text to process
            timing_mode: How to distribute timing ('equal', 'random', 'realistic')
            word_duration_min: Minimum word duration
            word_duration_max: Maximum word duration  
            sentence_pause: Pause between sentences
            total_duration: Force total duration (0 = auto)
            speech_rate: Words per minute for realistic mode
            
        Returns:
            Tuple of (word_timings_list, preview_string, calculated_total_duration)
        """
        logger.info(f"Converting text to timing data using {timing_mode} mode")
        
        # Clean and tokenize text
        words = self._tokenize_text(text)
        if not words:
            logger.warning("No words found in input text")
            return [], "No words to process", 0.0
        
        logger.info(f"Processing {len(words)} words")
        
        # Generate timing data based on mode
        if timing_mode == "equal":
            word_timings = self._generate_equal_timing(words, total_duration, sentence_pause)
        elif timing_mode == "random":
            word_timings = self._generate_random_timing(words, word_duration_min, word_duration_max, sentence_pause)
        elif timing_mode == "realistic":
            word_timings = self._generate_realistic_timing(words, speech_rate, sentence_pause)
        else:
            raise ValueError(f"Unknown timing mode: {timing_mode}")
        
        # Calculate actual total duration
        if word_timings:
            calculated_duration = max(word['end_time'] for word in word_timings)
        else:
            calculated_duration = 0.0
        
        # Create preview
        preview = self._create_timing_preview(word_timings, timing_mode)
        
        logger.info(f"Generated timing data for {len(word_timings)} words, total duration: {calculated_duration:.2f}s")
        
        return (word_timings, preview, calculated_duration)
    
    def _tokenize_text(self, text: str) -> List[Dict]:
        """
        Tokenize text into words with sentence boundary detection.
        
        Args:
            text: Input text
            
        Returns:
            List of word dictionaries with text and sentence info
        """
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        words = []
        word_index = 0
        
        # Split into sentences first
        sentences = re.split(r'[.!?]+', text)
        
        for sentence_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Split sentence into words
            sentence_words = sentence.split()
            
            for word_idx, word in enumerate(sentence_words):
                # Clean word of extra punctuation (keep sentence-ending punctuation)
                clean_word = re.sub(r'^[^\w]+|[^\w.!?]+$', '', word)
                
                if clean_word:
                    words.append({
                        'word': clean_word,
                        'sentence_idx': sentence_idx,
                        'word_in_sentence': word_idx,
                        'is_sentence_end': word_idx == len(sentence_words) - 1,
                        'word_index': word_index
                    })
                    word_index += 1
        
        return words
    
    def _generate_equal_timing(self, words: List[Dict], total_duration: float, sentence_pause: float) -> List[Dict]:
        """Generate equal timing distribution."""
        if not words:
            return []
        
        # Calculate total available time
        sentence_count = max(word['sentence_idx'] for word in words) + 1
        total_pause_time = (sentence_count - 1) * sentence_pause
        
        if total_duration > 0:
            available_time = total_duration - total_pause_time
        else:
            # Auto-calculate reasonable duration
            available_time = len(words) * 0.5  # 0.5 seconds per word as default
        
        if available_time <= 0:
            available_time = len(words) * 0.3  # Minimum fallback
        
        word_duration = available_time / len(words)
        
        # Generate timing
        word_timings = []
        current_time = 0.0
        current_sentence = -1
        
        for word in words:
            # Add sentence pause if starting new sentence
            if word['sentence_idx'] != current_sentence and current_sentence >= 0:
                current_time += sentence_pause
            current_sentence = word['sentence_idx']
            
            start_time = current_time
            end_time = current_time + word_duration
            
            word_timings.append({
                'word': word['word'],
                'start_time': start_time,
                'end_time': end_time,
                'confidence': 1.0,
                'start': start_time,  # Alternative format compatibility
                'end': end_time
            })
            
            current_time = end_time
        
        return word_timings
    
    def _generate_random_timing(self, words: List[Dict], min_duration: float, max_duration: float, sentence_pause: float) -> List[Dict]:
        """Generate random timing distribution."""
        if not words:
            return []
        
        word_timings = []
        current_time = 0.0
        current_sentence = -1
        
        for word in words:
            # Add sentence pause if starting new sentence
            if word['sentence_idx'] != current_sentence and current_sentence >= 0:
                current_time += sentence_pause
            current_sentence = word['sentence_idx']
            
            # Random word duration
            word_duration = random.uniform(min_duration, max_duration)
            
            start_time = current_time
            end_time = current_time + word_duration
            
            # Add some randomness to confidence
            confidence = random.uniform(0.85, 0.99)
            
            word_timings.append({
                'word': word['word'],
                'start_time': start_time,
                'end_time': end_time,
                'confidence': confidence,
                'start': start_time,
                'end': end_time
            })
            
            current_time = end_time
        
        return word_timings
    
    def _generate_realistic_timing(self, words: List[Dict], speech_rate: float, sentence_pause: float) -> List[Dict]:
        """Generate realistic timing based on speech rate and word characteristics."""
        if not words:
            return []
        
        # Words per second
        words_per_second = speech_rate / 60.0
        base_duration = 1.0 / words_per_second
        
        word_timings = []
        current_time = 0.0
        current_sentence = -1
        
        for word in words:
            # Add sentence pause if starting new sentence
            if word['sentence_idx'] != current_sentence and current_sentence >= 0:
                current_time += sentence_pause
            current_sentence = word['sentence_idx']
            
            # Adjust duration based on word characteristics
            word_text = word['word']
            duration_multiplier = 1.0
            
            # Longer words take more time
            if len(word_text) > 6:
                duration_multiplier += 0.3
            elif len(word_text) > 3:
                duration_multiplier += 0.1
            elif len(word_text) == 1:
                duration_multiplier -= 0.2
            
            # Punctuation adds slight pause
            if re.search(r'[.!?]', word_text):
                duration_multiplier += 0.2
            elif re.search(r'[,;:]', word_text):
                duration_multiplier += 0.1
            
            # Function words tend to be faster
            function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            if word_text.lower() in function_words:
                duration_multiplier -= 0.15
            
            # Calculate final duration
            word_duration = base_duration * max(0.3, duration_multiplier)  # Minimum 0.3 multiplier
            
            start_time = current_time
            end_time = current_time + word_duration
            
            # Realistic confidence based on word characteristics
            confidence = 0.95
            if len(word_text) == 1:
                confidence -= 0.1
            elif len(word_text) > 8:
                confidence -= 0.05
            
            # Add small random variation
            confidence += random.uniform(-0.05, 0.05)
            confidence = max(0.7, min(0.99, confidence))
            
            word_timings.append({
                'word': word['word'],
                'start_time': start_time,
                'end_time': end_time,
                'confidence': confidence,
                'start': start_time,
                'end': end_time
            })
            
            current_time = end_time
        
        return word_timings
    
    def _create_timing_preview(self, word_timings: List[Dict], timing_mode: str) -> str:
        """Create human-readable timing preview."""
        if not word_timings:
            return "No timing data generated"
        
        total_duration = max(word['end_time'] for word in word_timings)
        word_count = len(word_timings)
        avg_duration = sum(word['end_time'] - word['start_time'] for word in word_timings) / word_count
        
        preview = [
            f"Text-to-Timing Conversion ({timing_mode} mode)",
            f"Total words: {word_count}",
            f"Total duration: {total_duration:.2f} seconds",
            f"Average word duration: {avg_duration:.3f} seconds",
            f"Speech rate: {(word_count / total_duration) * 60:.1f} words per minute",
            "",
            "First 10 words with timing:"
        ]
        
        for word in word_timings[:10]:
            duration = word['end_time'] - word['start_time']
            preview.append(
                f"  {word['start_time']:.2f}s-{word['end_time']:.2f}s: "
                f"'{word['word']}' ({duration:.3f}s, conf: {word['confidence']:.2f})"
            )
        
        if word_count > 10:
            preview.append(f"  ... and {word_count - 10} more words")
        
        return "\\n".join(preview)