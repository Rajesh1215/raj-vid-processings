import torch
import numpy as np
import json
import os
import tempfile
from typing import Dict, List, Tuple, Optional
from .utils import tensor_to_video_frames, logger
import subprocess

# Try to import whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not installed. Install with: pip install openai-whisper")

class RajWhisperCaptions:
    """
    AI-powered transcription node using OpenAI's Whisper model.
    Generates word-level timestamps and caption segments from video audio.
    """
    
    # Whisper model sizes and their descriptions
    WHISPER_MODELS = {
        "tiny": "39M params - Fastest, good for drafts",
        "base": "74M params - Balanced speed/quality",
        "small": "244M params - Better accuracy",
        "medium": "769M params - Professional quality",
        "large": "1550M params - Best accuracy"
    }
    
    # Supported languages (ISO 639-1 codes)
    LANGUAGES = [
        "auto", "en", "zh", "es", "hi", "ar", "pt", "bn", "ru", "ja",
        "pa", "de", "jv", "wu", "ko", "fr", "te", "mr", "ta", "vi",
        "ur", "tr", "it", "th", "gu", "fa", "pl", "uk", "kn", "ml",
        "or", "my", "nl", "si", "el", "he", "ms", "ro", "bg", "hu",
        "sv", "cs", "fi", "da", "no", "sk", "hr", "sr", "lt", "sl",
        "et", "lv", "sq", "mk", "eu", "ca", "cy", "ga", "is", "mt"
    ]
    
    # Model cache
    _model_cache = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        model_list = list(cls.WHISPER_MODELS.keys())
        
        return {
            "required": {
                "video_frames": ("IMAGE", {
                    "tooltip": "Input video frames"
                }),
                "whisper_model": (model_list, {
                    "default": "base",
                    "tooltip": "Whisper model size"
                }),
                "language": (cls.LANGUAGES, {
                    "default": "auto",
                    "tooltip": "Source language (auto-detect or specify)"
                }),
                "words_per_caption": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 30,
                    "step": 1,
                    "tooltip": "Maximum words per caption segment"
                }),
                "max_caption_duration": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Maximum duration per caption (seconds)"
                }),
                "timestamp_level": (["word", "sentence", "paragraph"], {
                    "default": "word",
                    "tooltip": "Timestamp granularity"
                }),
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Frame rate for timing"
                }),
            },
            "optional": {
                "audio_file": ("STRING", {
                    "default": "",
                    "tooltip": "Direct audio file path (if not extracting from video)"
                }),
                "custom_vocabulary": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Custom words/terms for better recognition (one per line)"
                }),
                "translate_to": (["none", "en", "es", "fr", "de", "zh", "ja", "ko"], {
                    "default": "none",
                    "tooltip": "Translate captions to language"
                }),
                "min_silence_duration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Minimum silence duration to split captions"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Minimum confidence for words (0 = keep all)"
                }),
            }
        }
    
    RETURN_TYPES = ("CAPTION_DATA", "STRING", "FLOAT_LIST", "STRING")
    RETURN_NAMES = ("captions", "full_transcript", "timestamps", "caption_info")
    FUNCTION = "generate_captions"
    CATEGORY = "Raj Video Processing 🎬/Text"
    
    @classmethod
    def get_whisper_model(cls, model_name: str):
        """Load or retrieve cached Whisper model."""
        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper is not installed. Please install with: pip install openai-whisper")
        
        if model_name not in cls._model_cache:
            logger.info(f"Loading Whisper model: {model_name}")
            try:
                cls._model_cache[model_name] = whisper.load_model(model_name)
            except Exception as e:
                logger.error(f"Failed to load Whisper model {model_name}: {e}")
                raise
        
        return cls._model_cache[model_name]
    
    @staticmethod
    def extract_audio_from_frames(frames: torch.Tensor, fps: float, output_path: str) -> bool:
        """
        Extract audio from video frames (if video has audio track).
        This is a placeholder - in practice, we need the original video file.
        """
        try:
            # In a real implementation, we would need access to the original video file
            # or audio data passed separately. For now, we'll return False
            # and expect audio_file to be provided
            logger.warning("Audio extraction from frames not implemented. Please provide audio_file.")
            return False
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return False
    
    @staticmethod
    def group_words_into_captions(word_timestamps: List[Dict], 
                                 words_per_caption: int,
                                 max_duration: float,
                                 min_silence: float = 0.5) -> List[Dict]:
        """Group words into caption segments."""
        if not word_timestamps:
            return []
        
        captions = []
        current_words = []
        caption_start = word_timestamps[0]["start"] if word_timestamps else 0
        
        for i, word_data in enumerate(word_timestamps):
            current_words.append(word_data["word"])
            
            # Check if we should end current caption
            caption_duration = word_data["end"] - caption_start
            
            # Check for silence gap to next word
            silence_break = False
            if i < len(word_timestamps) - 1:
                next_word = word_timestamps[i + 1]
                silence_gap = next_word["start"] - word_data["end"]
                if silence_gap >= min_silence:
                    silence_break = True
            
            should_end = (
                len(current_words) >= words_per_caption or
                caption_duration >= max_duration or
                silence_break or
                i == len(word_timestamps) - 1
            )
            
            if should_end and current_words:
                caption_text = " ".join(current_words).strip()
                if caption_text:
                    captions.append({
                        "text": caption_text,
                        "start": caption_start,
                        "end": word_data["end"],
                        "duration": word_data["end"] - caption_start,
                        "word_count": len(current_words),
                        "confidence": sum(w.get("probability", 1.0) for w in word_timestamps[i-len(current_words)+1:i+1]) / len(current_words)
                    })
                
                # Start new caption
                current_words = []
                if i < len(word_timestamps) - 1:
                    caption_start = word_timestamps[i + 1]["start"]
        
        return captions
    
    def generate_captions(self, video_frames, whisper_model, language, 
                         words_per_caption, max_caption_duration, timestamp_level,
                         fps, audio_file="", custom_vocabulary="", translate_to="none",
                         min_silence_duration=0.5, confidence_threshold=0.0):
        
        if not WHISPER_AVAILABLE:
            error_msg = "Whisper is not installed. Please install with: pip install openai-whisper"
            logger.error(error_msg)
            return (json.dumps([]), "", [], error_msg)
        
        # Get frame count and duration
        frame_count = video_frames.shape[0]
        duration = frame_count / fps
        
        # Handle audio extraction
        temp_audio = None
        audio_path = audio_file
        
        if not audio_file:
            # Try to extract audio from video frames (placeholder)
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_path = temp_audio.name
            
            if not self.extract_audio_from_frames(video_frames, fps, audio_path):
                if temp_audio:
                    os.unlink(temp_audio.name)
                error_msg = "No audio file provided and cannot extract from frames. Please provide audio_file parameter."
                logger.error(error_msg)
                return (json.dumps([]), "", [], error_msg)
        
        try:
            # Load Whisper model
            model = self.get_whisper_model(whisper_model)
            
            # Prepare transcription options
            options = {
                "word_timestamps": True if timestamp_level == "word" else False,
                "verbose": False
            }
            
            # Set language if not auto
            if language != "auto":
                options["language"] = language
            
            # Add custom vocabulary as initial prompt if provided
            if custom_vocabulary:
                vocab_words = [w.strip() for w in custom_vocabulary.split("\n") if w.strip()]
                if vocab_words:
                    options["initial_prompt"] = " ".join(vocab_words)
            
            # Transcribe audio
            logger.info(f"Transcribing audio with Whisper {whisper_model} model...")
            result = model.transcribe(audio_path, **options)
            
            # Extract full transcript
            full_transcript = result["text"].strip()
            
            # Process based on timestamp level
            word_timestamps = []
            segments = result.get("segments", [])
            
            if timestamp_level == "word":
                # Extract word-level timestamps
                for segment in segments:
                    for word_data in segment.get("words", []):
                        if confidence_threshold == 0.0 or word_data.get("probability", 1.0) >= confidence_threshold:
                            word_timestamps.append({
                                "word": word_data["word"].strip(),
                                "start": word_data["start"],
                                "end": word_data["end"],
                                "probability": word_data.get("probability", 1.0)
                            })
            
            elif timestamp_level == "sentence":
                # Use segment-level timestamps
                for segment in segments:
                    if confidence_threshold == 0.0 or segment.get("avg_logprob", 0) >= confidence_threshold:
                        word_timestamps.append({
                            "word": segment["text"].strip(),
                            "start": segment["start"],
                            "end": segment["end"],
                            "probability": segment.get("avg_logprob", 0)
                        })
            
            else:  # paragraph
                # Group segments into paragraphs
                if segments:
                    para_start = segments[0]["start"]
                    para_text = []
                    
                    for i, segment in enumerate(segments):
                        para_text.append(segment["text"].strip())
                        
                        # Check if we should end paragraph
                        is_last = i == len(segments) - 1
                        has_long_pause = False
                        
                        if not is_last:
                            next_segment = segments[i + 1]
                            pause = next_segment["start"] - segment["end"]
                            has_long_pause = pause > 1.0
                        
                        if is_last or has_long_pause:
                            word_timestamps.append({
                                "word": " ".join(para_text),
                                "start": para_start,
                                "end": segment["end"],
                                "probability": 1.0
                            })
                            para_text = []
                            if not is_last:
                                para_start = next_segment["start"]
            
            # Group into captions
            captions = self.group_words_into_captions(
                word_timestamps,
                words_per_caption,
                max_caption_duration,
                min_silence_duration
            )
            
            # Handle translation if requested
            if translate_to != "none" and translate_to != language:
                # Note: Whisper can translate to English only
                # For other languages, you'd need a separate translation service
                if translate_to == "en" and language != "en":
                    logger.info(f"Translating to English...")
                    translation_result = model.transcribe(audio_path, task="translate", **options)
                    
                    # Update captions with translations
                    translated_text = translation_result["text"].strip()
                    # Simple approach: replace text in captions proportionally
                    # In production, you'd want more sophisticated alignment
                    for caption in captions:
                        caption["original_text"] = caption["text"]
                        caption["text"] = translated_text  # Simplified - needs proper implementation
                else:
                    logger.warning(f"Translation to {translate_to} not supported. Only English translation is available.")
            
            # Extract timestamps
            timestamps = []
            for caption in captions:
                timestamps.append(caption["start"])
                timestamps.append(caption["end"])
            
            # Create caption data structure
            caption_data = {
                "captions": captions,
                "language": result.get("language", language),
                "duration": duration,
                "frame_count": frame_count,
                "fps": fps,
                "model": whisper_model,
                "word_count": len(word_timestamps),
                "caption_count": len(captions)
            }
            
            # Create info string
            caption_info = (
                f"Transcription complete!\n"
                f"Model: Whisper {whisper_model}\n"
                f"Language: {result.get('language', language)}\n"
                f"Duration: {duration:.1f}s\n"
                f"Words: {len(word_timestamps)}\n"
                f"Captions: {len(captions)}\n"
                f"Avg words/caption: {len(word_timestamps)/max(len(captions), 1):.1f}"
            )
            
            # Cleanup temp audio if created
            if temp_audio:
                os.unlink(temp_audio.name)
            
            return (json.dumps(caption_data), full_transcript, timestamps, caption_info)
            
        except Exception as e:
            logger.error(f"Error generating captions: {e}")
            
            # Cleanup temp audio if created
            if temp_audio and os.path.exists(temp_audio.name):
                os.unlink(temp_audio.name)
            
            return (json.dumps([]), "", [], f"Error: {str(e)}")


# Test function
if __name__ == "__main__":
    if WHISPER_AVAILABLE:
        node = RajWhisperCaptions()
        print("Whisper Captions node initialized")
        print(f"Available models: {list(node.WHISPER_MODELS.keys())}")
        print(f"Supported languages: {len(node.LANGUAGES)}")
    else:
        print("Whisper not available. Install with: pip install openai-whisper")