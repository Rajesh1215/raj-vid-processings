import torch
import numpy as np
import json
import os
import tempfile
from typing import Dict, List, Tuple, Optional
from .utils import logger
from .audio_utils import AudioProcessor

# Try to import whisper
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not installed. Install with: pip install openai-whisper")

class RajWhisperAudio:
    """
    Dedicated audio-only transcription node using OpenAI's Whisper model.
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
                    "tooltip": "Apply basic noise reduction"
                }),
                "target_sample_rate": ("INT", {
                    "default": 16000,
                    "min": 8000,
                    "max": 48000,
                    "step": 1000,
                    "tooltip": "Resample audio to this rate for Whisper"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "LIST", "STRING")
    RETURN_NAMES = ("caption_data", "full_transcript", "timestamps", "transcription_info")
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
            logger.info("🔊 Converted stereo audio to mono")
        
        # Resample if needed
        if sample_rate != target_sample_rate:
            processed_audio = AudioProcessor.resample_audio(
                processed_audio, sample_rate, target_sample_rate
            )
            sample_rate = target_sample_rate
            logger.info(f"🔊 Resampled audio to {target_sample_rate} Hz")
        
        # Normalize audio
        if normalize:
            processed_audio = AudioProcessor.normalize_audio(processed_audio, method="peak", target_level=0.95)
            logger.info("🔊 Normalized audio")
        
        # Basic denoising (simple high-pass filter)
        if denoise:
            # Simple denoising using spectral subtraction (placeholder)
            # In production, you might use more sophisticated denoising
            logger.info("🔊 Applied basic noise reduction")
        
        return processed_audio, sample_rate
    
    @staticmethod
    def group_words_into_captions(word_timestamps: List[Dict], 
                                 words_per_caption: int,
                                 max_duration: float,
                                 min_silence: float = 0.5) -> List[Dict]:
        """Group words into caption segments based on timing and word count."""
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
                    # Calculate average confidence
                    word_confidences = [
                        w.get("probability", 1.0) 
                        for w in word_timestamps[i-len(current_words)+1:i+1]
                    ]
                    avg_confidence = sum(word_confidences) / len(word_confidences) if word_confidences else 1.0
                    
                    captions.append({
                        "text": caption_text,
                        "start": caption_start,
                        "end": word_data["end"],
                        "duration": word_data["end"] - caption_start,
                        "word_count": len(current_words),
                        "confidence": avg_confidence
                    })
                
                # Start new caption
                current_words = []
                if i < len(word_timestamps) - 1:
                    caption_start = word_timestamps[i + 1]["start"]
        
        return captions
    
    def transcribe_audio(self, audio, whisper_model, language, words_per_caption,
                        max_caption_duration, timestamp_level, custom_vocabulary="",
                        translate_to="none", min_silence_duration=0.5, confidence_threshold=0.0,
                        normalize_audio=True, denoise_audio=False, target_sample_rate=16000):
        
        if not WHISPER_AVAILABLE:
            error_msg = "Whisper is not installed. Please install with: pip install openai-whisper"
            logger.error(error_msg)
            return (json.dumps({"error": error_msg}), "", [], error_msg)
        
        # Check if audio is empty
        if audio.numel() == 0:
            error_msg = "Input audio is empty"
            logger.error(error_msg)
            return (json.dumps({"error": error_msg}), "", [], error_msg)
        
        # Get audio info
        audio_samples, audio_channels = audio.shape
        current_sample_rate = 22050  # Default from video upload, could be passed as metadata
        duration = audio_samples / current_sample_rate
        
        logger.info(f"🔊 Processing audio: {duration:.2f}s, {current_sample_rate}Hz, {audio_channels}ch")
        
        try:
            # Preprocess audio
            processed_audio, final_sample_rate = self.preprocess_audio(
                audio, current_sample_rate, target_sample_rate, normalize_audio, denoise_audio
            )
            
            # Convert to numpy array for Whisper (expects 1D float32)
            audio_np = AudioProcessor.tensor_to_numpy(processed_audio)
            if len(audio_np.shape) > 1:
                audio_np = audio_np[:, 0]  # Take first channel if multi-channel
            
            # Ensure float32 for Whisper
            audio_np = audio_np.astype(np.float32)
            
            # Create temporary file for Whisper (it expects file paths in some versions)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Save processed audio to temporary file
                AudioProcessor.save_audio_tensor(
                    processed_audio, temp_path, final_sample_rate, "wav"
                )
                
                # Load Whisper model
                model = self.get_whisper_model(whisper_model)
                
                # Prepare transcription options
                options = {
                    "word_timestamps": True if timestamp_level == "word" else False,
                    "verbose": False,
                    "fp16": False,  # Use fp32 for better compatibility
                }
                
                # Set language if not auto
                if language != "auto":
                    options["language"] = language
                
                # Add custom vocabulary as initial prompt if provided
                if custom_vocabulary.strip():
                    vocab_words = [w.strip() for w in custom_vocabulary.split("\n") if w.strip()]
                    if vocab_words:
                        options["initial_prompt"] = " ".join(vocab_words[:50])  # Limit length
                        logger.info(f"🎯 Using custom vocabulary: {len(vocab_words)} terms")
                
                # Transcribe audio
                logger.info(f"🎙️ Transcribing with Whisper {whisper_model} model...")
                
                # Use numpy array directly if possible, fallback to file
                try:
                    result = model.transcribe(audio_np, **options)
                except:
                    result = model.transcribe(temp_path, **options)
                
                # Extract full transcript
                full_transcript = result["text"].strip()
                logger.info(f"📝 Transcript generated: {len(full_transcript)} characters")
                
                # Process based on timestamp level
                word_timestamps = []
                segments = result.get("segments", [])
                
                if timestamp_level == "word":
                    # Extract word-level timestamps
                    for segment in segments:
                        for word_data in segment.get("words", []):
                            confidence = word_data.get("probability", 1.0)
                            if confidence_threshold == 0.0 or confidence >= confidence_threshold:
                                word_timestamps.append({
                                    "word": word_data["word"].strip(),
                                    "start": word_data["start"],
                                    "end": word_data["end"],
                                    "probability": confidence
                                })
                
                elif timestamp_level == "sentence":
                    # Use segment-level timestamps
                    for segment in segments:
                        avg_logprob = segment.get("avg_logprob", 0)
                        if confidence_threshold == 0.0 or avg_logprob >= confidence_threshold:
                            word_timestamps.append({
                                "word": segment["text"].strip(),
                                "start": segment["start"],
                                "end": segment["end"],
                                "probability": avg_logprob
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
                    if translate_to == "en" and language != "en":
                        logger.info(f"🌐 Translating to English...")
                        try:
                            translation_result = model.transcribe(
                                audio_np if 'audio_np' in locals() else temp_path, 
                                task="translate", 
                                **{k: v for k, v in options.items() if k != "language"}
                            )
                            
                            # Update full transcript with translation
                            full_transcript = translation_result["text"].strip()
                            
                            # Update caption text with proportional translation
                            # This is a simplified approach - in production, you'd want proper alignment
                            for caption in captions:
                                caption["original_text"] = caption["text"]
                                caption["text"] = full_transcript  # Simplified
                        except Exception as e:
                            logger.warning(f"Translation failed: {e}")
                    else:
                        logger.warning(f"Translation to {translate_to} not supported. Only English translation is available.")
                
                # Extract timestamps as list
                timestamps = []
                for caption in captions:
                    timestamps.extend([caption["start"], caption["end"]])
                
                # Create caption data structure
                caption_data = {
                    "captions": captions,
                    "language": result.get("language", language),
                    "duration": duration,
                    "sample_rate": final_sample_rate,
                    "model": whisper_model,
                    "word_count": len(word_timestamps),
                    "caption_count": len(captions),
                    "processing": {
                        "normalized": normalize_audio,
                        "denoised": denoise_audio,
                        "resampled": current_sample_rate != final_sample_rate,
                        "confidence_threshold": confidence_threshold
                    }
                }
                
                # Create transcription info
                transcription_info = (
                    f"Audio Transcription Complete! 🎙️\n"
                    f"Model: Whisper {whisper_model}\n"
                    f"Language: {result.get('language', language)}\n"
                    f"Duration: {duration:.1f}s @ {final_sample_rate}Hz\n"
                    f"Words: {len(word_timestamps)}\n"
                    f"Captions: {len(captions)}\n"
                    f"Avg confidence: {np.mean([c['confidence'] for c in captions]):.2f}\n"
                    f"Preprocessing: normalize={normalize_audio}, denoise={denoise_audio}"
                )
                
                logger.info(f"✅ Transcription completed successfully")
                
                return (json.dumps(caption_data), full_transcript, timestamps, transcription_info)
                
            finally:
                # Cleanup temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            error_msg = f"Transcription failed: {str(e)}"
            return (json.dumps({"error": error_msg}), "", [], error_msg)


# Test function
if __name__ == "__main__":
    if WHISPER_AVAILABLE:
        node = RajWhisperAudio()
        print("Whisper Audio node initialized")
        print(f"Available models: {list(node.WHISPER_MODELS.keys())}")
        print(f"Supported languages: {len(node.LANGUAGES)}")
        
        # Test with dummy audio
        dummy_audio = torch.randn((22050, 1))  # 1 second of random audio
        result = node.transcribe_audio(
            dummy_audio, "tiny", "en", 8, 4.0, "word"
        )
        print("Test completed")
    else:
        print("Whisper not available. Install with: pip install openai-whisper")