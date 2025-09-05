import torch
import numpy as np
import json
import os
import tempfile
from typing import Dict, List, Tuple, Optional
from .utils import logger
from ..utils.audio_utils import AudioProcessor

# Try to import whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not installed. Install with: pip install openai-whisper")

class RajWhisperProcess:
    """
    Advanced audio transcription processing node using OpenAI's Whisper model.
    Provides both sentence-level and word-level caption outputs with precise timing.
    Optimized for direct audio input without video processing overhead.
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
                "audio": ("AUDIO", {
                    "tooltip": "Input audio tensor from video upload or audio loader"
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
            },
            "optional": {
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
                "normalize_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize audio before processing"
                }),
                "denoise_audio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply basic denoising"
                }),
                "target_sample_rate": ("INT", {
                    "default": 16000,
                    "min": 8000,
                    "max": 48000,
                    "step": 1000,
                    "tooltip": "Target sample rate for Whisper"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "LIST", "STRING")
    RETURN_NAMES = ("sentence_captions", "word_captions", "full_transcript", "word_timings", "transcription_info")
    FUNCTION = "transcribe_audio"
    CATEGORY = "Raj Video Processing 🎬/Audio"
    
    @classmethod
    def get_whisper_model(cls, model_name: str):
        """Load or retrieve cached Whisper model."""
        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper is not installed. Please install with: pip install openai-whisper")
        
        if model_name not in cls._model_cache:
            logger.info(f"Loading Whisper model: {model_name}")
            try:
                cls._model_cache[model_name] = whisper.load_model(model_name)
                logger.info(f"✅ Whisper {model_name} model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model {model_name}: {e}")
                raise
        
        return cls._model_cache[model_name]
    
    @staticmethod
    def preprocess_audio(audio_tensor: torch.Tensor, 
                        sample_rate: int,
                        target_sample_rate: int = 16000,
                        normalize: bool = True,
                        denoise: bool = False) -> Tuple[torch.Tensor, int]:
        """
        Preprocess audio for optimal Whisper performance.
        
        Args:
            audio_tensor: Input audio tensor (samples, channels)
            sample_rate: Current sample rate
            target_sample_rate: Target sample rate for Whisper
            normalize: Whether to normalize audio
            denoise: Whether to apply basic denoising
            
        Returns:
            Tuple of processed audio tensor and sample rate
        """
        processed_audio = audio_tensor.clone()
        
        # Convert to mono if stereo
        if processed_audio.shape[1] > 1:
            processed_audio = AudioProcessor.convert_to_mono(processed_audio)
        
        # Resample if needed
        if sample_rate != target_sample_rate:
            processed_audio = AudioProcessor.resample_audio(
                processed_audio, sample_rate, target_sample_rate
            )
            logger.info(f"🔄 Resampled audio: {sample_rate}Hz → {target_sample_rate}Hz")
        
        # Normalize audio
        if normalize:
            processed_audio = AudioProcessor.normalize_audio(
                processed_audio, method="peak", target_level=0.8
            )
            logger.info("📈 Audio normalized for Whisper")
        
        # Basic denoising (simple high-pass filter)
        if denoise:
            # Apply simple high-pass filter to reduce low-frequency noise
            processed_audio = cls._apply_high_pass_filter(processed_audio, target_sample_rate)
            logger.info("🔇 Basic denoising applied")
        
        return processed_audio, target_sample_rate
    
    @staticmethod
    def _apply_high_pass_filter(audio: torch.Tensor, sample_rate: int, cutoff: float = 80.0) -> torch.Tensor:
        """Apply simple high-pass filter to reduce low-frequency noise."""
        try:
            # Simple high-pass filter using difference
            filtered = audio.clone()
            alpha = cutoff / (cutoff + sample_rate)
            
            for i in range(1, len(filtered)):
                filtered[i] = alpha * (filtered[i-1] + audio[i] - audio[i-1])
            
            return filtered
        except Exception as e:
            logger.warning(f"High-pass filter failed, using original audio: {e}")
            return audio
    
    def transcribe_audio(self, audio, whisper_model, language, words_per_caption,
                        max_caption_duration, timestamp_level, custom_vocabulary="",
                        translate_to="none", min_silence_duration=0.5, confidence_threshold=0.0,
                        normalize_audio=True, denoise_audio=False, target_sample_rate=16000):
        
        if not WHISPER_AVAILABLE:
            error_msg = "Whisper is not installed. Please install with: pip install openai-whisper"
            logger.error(error_msg)
            return (error_msg, error_msg, error_msg, [], error_msg)
        
        if audio.numel() == 0:
            error_msg = "Input audio is empty"
            logger.warning(error_msg)
            return (error_msg, error_msg, error_msg, [], error_msg)
        
        try:
            logger.info(f"🎙️ Starting Whisper transcription: {whisper_model} model")
            
            # Load Whisper model
            model = self.get_whisper_model(whisper_model)
            
            # Preprocess audio
            processed_audio, processed_sr = self.preprocess_audio(
                audio, target_sample_rate, target_sample_rate, normalize_audio, denoise_audio
            )
            
            # Convert to numpy for Whisper (flatten to 1D)
            audio_np = processed_audio.squeeze().numpy().astype(np.float32)
            
            logger.info(f"🎵 Audio preprocessed: {len(audio_np):,} samples @ {processed_sr}Hz")
            
            # Transcription options
            transcribe_options = {
                "language": None if language == "auto" else language,
                "task": "translate" if translate_to != "none" else "transcribe",
                "word_timestamps": True,  # Always get word timestamps
                "verbose": False,
                "fp16": torch.cuda.is_available(),  # Use fp16 if CUDA available
            }
            
            # Add custom vocabulary if provided
            if custom_vocabulary.strip():
                vocab_words = [w.strip() for w in custom_vocabulary.strip().split('\n') if w.strip()]
                if vocab_words:
                    # Whisper doesn't directly support custom vocabulary, but we can use initial_prompt
                    transcribe_options["initial_prompt"] = " ".join(vocab_words[:20])  # Limit to 20 words
                    logger.info(f"📝 Using custom vocabulary: {len(vocab_words)} words")
            
            # Perform transcription
            logger.info("🔄 Running Whisper transcription...")
            result = model.transcribe(audio_np, **transcribe_options)
            
            # Extract results
            full_transcript = result["text"].strip()
            detected_language = result.get("language", "unknown")
            
            logger.info(f"✅ Transcription completed: {len(full_transcript)} chars, language: {detected_language}")
            
            # Process segments and words
            sentence_captions = self._create_sentence_captions(
                result, words_per_caption, max_caption_duration, min_silence_duration
            )
            
            word_captions, word_timings = self._create_word_captions(
                result, confidence_threshold
            )
            
            # Create transcription info
            total_duration = len(audio_np) / processed_sr
            word_count = len(full_transcript.split()) if full_transcript else 0
            
            transcription_info = (
                f"Whisper Transcription Complete\\n"
                f"Model: {whisper_model} ({self.WHISPER_MODELS.get(whisper_model, 'Unknown')})\\n"
                f"Language: {detected_language}\\n"
                f"Audio Duration: {total_duration:.2f}s\\n"
                f"Transcript Length: {len(full_transcript)} characters\\n"
                f"Word Count: {word_count}\\n"
                f"Sentence Segments: {len(json.loads(sentence_captions)) if sentence_captions != '[]' else 0}\\n"
                f"Word Segments: {len(json.loads(word_captions)) if word_captions != '[]' else 0}\\n"
                f"Processing: {'Normalized' if normalize_audio else 'Raw'}"
                + (f", Denoised" if denoise_audio else "")
                + (f", Custom Vocab" if custom_vocabulary.strip() else "")
            )
            
            logger.info(f"📊 Generated {len(json.loads(sentence_captions))} sentence captions, {len(json.loads(word_captions))} word captions")
            
            return (sentence_captions, word_captions, full_transcript, word_timings, transcription_info)
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            error_msg = f"Transcription failed: {str(e)}"
            return (error_msg, error_msg, error_msg, [], error_msg)
    
    def _create_sentence_captions(self, result: Dict, words_per_caption: int, 
                                 max_duration: float, min_silence: float) -> str:
        """Create sentence-level captions from Whisper result."""
        try:
            segments = result.get("segments", [])
            if not segments:
                return "[]"
            
            captions = []
            
            for segment in segments:
                text = segment.get("text", "").strip()
                if not text:
                    continue
                
                start_time = segment.get("start", 0.0)
                end_time = segment.get("end", start_time + 1.0)
                
                # Split long segments by word count
                words = text.split()
                if len(words) <= words_per_caption and (end_time - start_time) <= max_duration:
                    captions.append({
                        "start": round(start_time, 2),
                        "end": round(end_time, 2),
                        "text": text,
                        "word_count": len(words)
                    })
                else:
                    # Split into smaller chunks
                    word_list = segment.get("words", [])
                    if word_list:
                        current_caption = {"text": "", "words": [], "start": None}
                        
                        for word_info in word_list:
                            word = word_info.get("word", "").strip()
                            if not word:
                                continue
                                
                            word_start = word_info.get("start", 0.0)
                            word_end = word_info.get("end", word_start + 0.1)
                            
                            if current_caption["start"] is None:
                                current_caption["start"] = word_start
                            
                            current_caption["text"] += word + " "
                            current_caption["words"].append(word_info)
                            
                            # Check if we should end this caption
                            word_count = len(current_caption["words"])
                            duration = word_end - current_caption["start"]
                            
                            if (word_count >= words_per_caption or 
                                duration >= max_duration or
                                word == word_list[-1]["word"]):
                                
                                captions.append({
                                    "start": round(current_caption["start"], 2),
                                    "end": round(word_end, 2),
                                    "text": current_caption["text"].strip(),
                                    "word_count": word_count
                                })
                                
                                current_caption = {"text": "", "words": [], "start": None}
                    else:
                        # Fallback: split by word count only
                        chunk_size = words_per_caption
                        duration_per_word = (end_time - start_time) / len(words)
                        
                        for i in range(0, len(words), chunk_size):
                            chunk_words = words[i:i + chunk_size]
                            chunk_start = start_time + i * duration_per_word
                            chunk_end = start_time + min(i + chunk_size, len(words)) * duration_per_word
                            
                            captions.append({
                                "start": round(chunk_start, 2),
                                "end": round(chunk_end, 2),
                                "text": " ".join(chunk_words),
                                "word_count": len(chunk_words)
                            })
            
            return json.dumps(captions, indent=2)
            
        except Exception as e:
            logger.error(f"Error creating sentence captions: {e}")
            return "[]"
    
    def _create_word_captions(self, result: Dict, confidence_threshold: float = 0.0) -> Tuple[str, List]:
        """Create word-level captions from Whisper result."""
        try:
            segments = result.get("segments", [])
            if not segments:
                return "[]", []
            
            word_captions = []
            word_timings = []
            
            for segment in segments:
                words = segment.get("words", [])
                if not words:
                    # Fallback: create word data from segment text
                    text = segment.get("text", "").strip()
                    if text:
                        segment_words = text.split()
                        start_time = segment.get("start", 0.0)
                        end_time = segment.get("end", start_time + 1.0)
                        duration_per_word = (end_time - start_time) / len(segment_words)
                        
                        for i, word in enumerate(segment_words):
                            word_start = start_time + i * duration_per_word
                            word_end = start_time + (i + 1) * duration_per_word
                            
                            word_captions.append({
                                "start": round(word_start, 3),
                                "end": round(word_end, 3),
                                "word": word,
                                "confidence": 1.0,  # Default confidence
                                "segment_id": len(word_captions)
                            })
                            
                            word_timings.append({
                                "word": word,
                                "start": round(word_start, 3),
                                "end": round(word_end, 3),
                                "duration": round(word_end - word_start, 3),
                                "confidence": 1.0
                            })
                    continue
                
                # Process actual word-level data
                for word_info in words:
                    word = word_info.get("word", "").strip()
                    if not word:
                        continue
                    
                    word_start = word_info.get("start", 0.0)
                    word_end = word_info.get("end", word_start + 0.1)
                    confidence = word_info.get("probability", 1.0)  # Whisper uses 'probability'
                    
                    # Apply confidence threshold
                    if confidence < confidence_threshold:
                        logger.debug(f"Skipping low-confidence word: {word} ({confidence:.2f})")
                        continue
                    
                    word_caption = {
                        "start": round(word_start, 3),
                        "end": round(word_end, 3), 
                        "word": word,
                        "confidence": round(confidence, 3),
                        "segment_id": len(word_captions),
                        "duration": round(word_end - word_start, 3)
                    }
                    
                    word_captions.append(word_caption)
                    
                    # Create timing entry
                    word_timings.append({
                        "word": word,
                        "start": round(word_start, 3),
                        "end": round(word_end, 3),
                        "duration": round(word_end - word_start, 3),
                        "confidence": round(confidence, 3),
                        "char_start": None,  # Could be computed if needed
                        "char_end": None
                    })
            
            # Sort by start time
            word_captions.sort(key=lambda x: x["start"])
            word_timings.sort(key=lambda x: x["start"])
            
            logger.info(f"📝 Created {len(word_captions)} word-level captions")
            
            return json.dumps(word_captions, indent=2), word_timings
            
        except Exception as e:
            logger.error(f"Error creating word captions: {e}")
            return "[]", []

# Test function
def test_whisper_process():
    """Test the RajWhisperProcess functionality."""
    print("Testing RajWhisperProcess...")
    
    if not WHISPER_AVAILABLE:
        print("❌ Whisper not available for testing")
        return
    
    # Create test audio (2 seconds of sine wave)
    sample_rate = 16000
    duration = 2.0
    frequency = 440.0
    
    t = torch.linspace(0, duration, int(sample_rate * duration))
    test_audio = torch.sin(2 * torch.pi * frequency * t).unsqueeze(1)
    
    whisper_node = RajWhisperProcess()
    
    try:
        result = whisper_node.transcribe_audio(
            audio=test_audio,
            whisper_model="tiny",
            language="en",
            words_per_caption=5,
            max_caption_duration=3.0,
            timestamp_level="word"
        )
        
        sentence_captions, word_captions, full_transcript, word_timings, info = result
        print(f"✅ Transcription test completed")
        print(f"📝 Transcript: {full_transcript[:100]}...")
        print(f"📊 Word timings: {len(word_timings)} entries")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_whisper_process()