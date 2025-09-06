import torch
import numpy as np
from typing import Tuple, Optional, Dict
from .utils import logger
from ..utils.audio_utils import AudioProcessor

class RajAudioSegmenter:
    """
    Audio segmentation node that extracts specific segments from audio with dual outputs.
    Similar to RajVideoSegmenter, provides both the extracted segment and remaining audio.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Input audio tensor to segment"
                }),
                "start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Start time of segment in seconds"
                }),
                "end_time": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "End time of segment in seconds"
                }),
                "current_sample_rate": ("INT", {
                    "default": 22050,
                    "min": 8000,
                    "max": 96000,
                    "step": 1000,
                    "tooltip": "Current audio sample rate"
                }),
                "fade_edges": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply fade in/out to avoid clicks"
                }),
                "fade_duration": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Fade duration in seconds"
                }),
            },
            "optional": {
                "crossfade_junction": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply crossfade where segment was removed"
                }),
                "junction_fade_duration": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Crossfade duration at junction point"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("segmented_audio", "remaining_audio", "segment_info")
    FUNCTION = "segment_audio"
    CATEGORY = "Raj Video Processing 🎬/Audio"
    
    def segment_audio(self, audio, start_time: float, end_time: float,
                     current_sample_rate: int, fade_edges: bool, fade_duration: float,
                     crossfade_junction: bool = False, junction_fade_duration: float = 0.05):
        """
        Segment audio into extracted portion and remaining portion.
        
        Returns:
            segmented_audio: The extracted segment
            remaining_audio: Original audio with segment removed
            segment_info: Information about the segmentation
        """
        
        # Extract tensor from ComfyUI AUDIO format (dict with waveform and sample_rate)
        if isinstance(audio, dict):
            audio_tensor = audio["waveform"].squeeze(0).transpose(0, 1)  # [batch, channels, samples] -> [samples, channels]
            audio_sr = audio["sample_rate"]
        else:
            audio_tensor = audio
            audio_sr = current_sample_rate
        
        if audio_tensor.numel() == 0:
            logger.warning("Input audio is empty")
            empty_audio = {"waveform": torch.zeros(1, 1, 1), "sample_rate": audio_sr}
            return (empty_audio, empty_audio, "Input audio was empty")
        
        # Validate times
        if end_time <= start_time:
            logger.error(f"Invalid segment: end_time ({end_time}) must be greater than start_time ({start_time})")
            error_audio = {"waveform": audio_tensor.transpose(0, 1).unsqueeze(0), "sample_rate": audio_sr}
            empty_audio = {"waveform": torch.zeros(1, audio_tensor.shape[1], 1), "sample_rate": audio_sr}
            return (error_audio, empty_audio, "Invalid time range")
        
        # Calculate sample positions
        total_samples = audio_tensor.shape[0]
        total_duration = total_samples / audio_sr
        
        start_sample = int(start_time * audio_sr)
        end_sample = int(end_time * audio_sr)
        
        # Clamp to valid range
        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(start_sample, min(end_sample, total_samples))
        
        if start_sample >= total_samples:
            logger.warning(f"Start time {start_time}s exceeds audio duration {total_duration:.2f}s")
            empty_segment = {"waveform": torch.zeros(1, audio_tensor.shape[1], 1), "sample_rate": audio_sr}
            full_audio = {"waveform": audio_tensor.transpose(0, 1).unsqueeze(0), "sample_rate": audio_sr}
            return (empty_segment, full_audio, f"Start time exceeds duration")
        
        logger.info(f"✂️ Segmenting audio: {start_time:.2f}s - {end_time:.2f}s (samples {start_sample:,} - {end_sample:,})")
        
        try:
            # Extract the segment
            segmented_audio = audio_tensor[start_sample:end_sample].clone()
            
            # Create remaining audio (before + after segment)
            if start_sample > 0 and end_sample < total_samples:
                # There's audio before and after
                before_segment = audio_tensor[:start_sample]
                after_segment = audio_tensor[end_sample:]
                
                if crossfade_junction:
                    # Apply crossfade at junction point
                    remaining_audio = self._crossfade_junction(
                        before_segment, after_segment, 
                        junction_fade_duration, audio_sr
                    )
                else:
                    # Simple concatenation
                    remaining_audio = torch.cat([before_segment, after_segment], dim=0)
            elif start_sample == 0:
                # Segment is at the beginning
                remaining_audio = audio_tensor[end_sample:]
            else:
                # Segment is at the end
                remaining_audio = audio_tensor[:start_sample]
            
            # Apply fade edges if requested
            if fade_edges and segmented_audio.shape[0] > 1:
                segmented_audio = self._apply_fade_edges(
                    segmented_audio, fade_duration, audio_sr
                )
                logger.info(f"🔊 Applied fade in/out of {fade_duration:.2f}s to segment edges")
            
            # Ensure we have valid audio tensors
            if segmented_audio.shape[0] == 0:
                segmented_audio = torch.zeros((1, audio_tensor.shape[1]))
            if remaining_audio.shape[0] == 0:
                remaining_audio = torch.zeros((1, audio_tensor.shape[1]))
            
            # Calculate segment info
            segment_duration = segmented_audio.shape[0] / audio_sr
            remaining_duration = remaining_audio.shape[0] / audio_sr
            
            segment_info = (
                f"Audio Segmentation Complete\n"
                f"Original Duration: {total_duration:.2f}s ({total_samples:,} samples)\n"
                f"Segment Range: {start_time:.2f}s - {end_time:.2f}s\n"
                f"Segment Duration: {segment_duration:.2f}s ({segmented_audio.shape[0]:,} samples)\n"
                f"Remaining Duration: {remaining_duration:.2f}s ({remaining_audio.shape[0]:,} samples)\n"
                f"Sample Rate: {audio_sr} Hz\n"
                f"Channels: {audio_tensor.shape[1]}\n"
                f"Fade Edges: {'Yes' if fade_edges else 'No'}"
                + (f" ({fade_duration:.2f}s)" if fade_edges else "") + "\n"
                f"Crossfade Junction: {'Yes' if crossfade_junction else 'No'}"
                + (f" ({junction_fade_duration:.2f}s)" if crossfade_junction else "")
            )
            
            logger.info(f"✅ Segmentation complete: {segment_duration:.2f}s extracted, {remaining_duration:.2f}s remaining")
            
            # Return in ComfyUI AUDIO format
            segment_result = {
                "waveform": segmented_audio.transpose(0, 1).unsqueeze(0),  # [samples, channels] -> [batch, channels, samples]
                "sample_rate": audio_sr
            }
            remaining_result = {
                "waveform": remaining_audio.transpose(0, 1).unsqueeze(0),  # [samples, channels] -> [batch, channels, samples]  
                "sample_rate": audio_sr
            }
            
            return (segment_result, remaining_result, segment_info)
            
        except Exception as e:
            logger.error(f"Error during audio segmentation: {e}")
            error_audio = {"waveform": audio_tensor.transpose(0, 1).unsqueeze(0), "sample_rate": audio_sr}
            empty_audio = {"waveform": torch.zeros(1, audio_tensor.shape[1], 1), "sample_rate": audio_sr}
            return (error_audio, empty_audio, f"Segmentation failed: {str(e)}")
    
    def _apply_fade_edges(self, audio: torch.Tensor, fade_duration: float, sample_rate: int) -> torch.Tensor:
        """Apply fade in and fade out to audio segment."""
        fade_samples = int(fade_duration * sample_rate)
        total_samples = audio.shape[0]
        
        if fade_samples * 2 >= total_samples:
            # Fade duration too long, adjust
            fade_samples = total_samples // 4
        
        if fade_samples > 0:
            # Apply fade in
            fade_in_curve = torch.linspace(0, 1, fade_samples).unsqueeze(1)
            audio[:fade_samples] *= fade_in_curve.to(audio.device)
            
            # Apply fade out
            fade_out_curve = torch.linspace(1, 0, fade_samples).unsqueeze(1)
            audio[-fade_samples:] *= fade_out_curve.to(audio.device)
        
        return audio
    
    def _crossfade_junction(self, before: torch.Tensor, after: torch.Tensor, 
                           fade_duration: float, sample_rate: int) -> torch.Tensor:
        """Apply crossfade at the junction where segment was removed."""
        fade_samples = int(fade_duration * sample_rate)
        
        if fade_samples > 0 and before.shape[0] >= fade_samples and after.shape[0] >= fade_samples:
            # Apply fade out to end of before
            fade_out_curve = torch.linspace(1, 0, fade_samples).unsqueeze(1)
            before_tail = before[-fade_samples:].clone()
            before_tail *= fade_out_curve.to(before.device)
            
            # Apply fade in to start of after
            fade_in_curve = torch.linspace(0, 1, fade_samples).unsqueeze(1)
            after_head = after[:fade_samples].clone()
            after_head *= fade_in_curve.to(after.device)
            
            # Mix the crossfade region
            crossfade_region = before_tail + after_head
            
            # Concatenate: before (without tail) + crossfade + after (without head)
            result = torch.cat([
                before[:-fade_samples],
                crossfade_region,
                after[fade_samples:]
            ], dim=0)
            
            return result
        else:
            # Can't apply crossfade, just concatenate
            return torch.cat([before, after], dim=0)


class RajAudioMultiSegmenter:
    """
    Extract multiple segments from audio in one operation.
    Supports various output modes including concatenation and separation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Input audio tensor to segment"
                }),
                "segments": ("STRING", {
                    "default": "0.0-5.0,10.0-15.0,20.0-25.0",
                    "multiline": False,
                    "tooltip": "Comma-separated start-end pairs (e.g., '0-5,10-15')"
                }),
                "current_sample_rate": ("INT", {
                    "default": 22050,
                    "min": 8000,
                    "max": 96000,
                    "step": 1000,
                    "tooltip": "Current audio sample rate"
                }),
                "output_mode": (["concatenate", "first_only", "last_only", "longest", "shortest"], {
                    "default": "concatenate",
                    "tooltip": "How to output multiple segments"
                }),
                "include_remaining": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also output non-segmented portions"
                }),
                "fade_edges": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply fade to segment edges"
                }),
                "fade_duration": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Fade duration in seconds"
                }),
                "gap_handling": (["silence", "crossfade", "remove"], {
                    "default": "remove",
                    "tooltip": "How to handle gaps between segments"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("segments_audio", "remaining_audio", "segments_info")
    FUNCTION = "segment_multiple"
    CATEGORY = "Raj Video Processing 🎬/Audio"
    
    def segment_multiple(self, audio: torch.Tensor, segments: str, current_sample_rate: int,
                        output_mode: str, include_remaining: bool, fade_edges: bool,
                        fade_duration: float, gap_handling: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Extract multiple segments from audio.
        """
        
        if audio.numel() == 0:
            logger.warning("Input audio is empty")
            empty = torch.zeros((1, 1))
            return (empty, empty, "Input audio was empty")
        
        # Parse segments string
        segment_list = []
        try:
            for segment_str in segments.split(','):
                segment_str = segment_str.strip()
                if '-' in segment_str:
                    start_str, end_str = segment_str.split('-')
                    start_time = float(start_str.strip())
                    end_time = float(end_str.strip())
                    if end_time > start_time:
                        segment_list.append((start_time, end_time))
                    else:
                        logger.warning(f"Invalid segment: {segment_str}")
        except Exception as e:
            logger.error(f"Error parsing segments: {e}")
            return (audio, torch.zeros((1, audio.shape[1])), f"Failed to parse segments: {str(e)}")
        
        if not segment_list:
            logger.warning("No valid segments found")
            return (audio, torch.zeros((1, audio.shape[1])), "No valid segments found")
        
        # Sort segments by start time
        segment_list.sort(key=lambda x: x[0])
        
        logger.info(f"📋 Processing {len(segment_list)} segments")
        
        # Extract all segments
        extracted_segments = []
        remaining_mask = torch.ones(audio.shape[0], dtype=torch.bool)
        
        for start_time, end_time in segment_list:
            start_sample = int(start_time * current_sample_rate)
            end_sample = int(end_time * current_sample_rate)
            
            # Clamp to valid range
            start_sample = max(0, min(start_sample, audio.shape[0]))
            end_sample = max(start_sample, min(end_sample, audio.shape[0]))
            
            if start_sample < end_sample:
                segment = audio[start_sample:end_sample].clone()
                
                # Apply fade if requested
                if fade_edges and segment.shape[0] > 1:
                    fade_samples = int(fade_duration * current_sample_rate)
                    if fade_samples > 0:
                        # Fade in
                        fade_in_samples = min(fade_samples, segment.shape[0] // 2)
                        if fade_in_samples > 0:
                            fade_in_curve = torch.linspace(0, 1, fade_in_samples).unsqueeze(1)
                            segment[:fade_in_samples] *= fade_in_curve.to(segment.device)
                        
                        # Fade out
                        fade_out_samples = min(fade_samples, segment.shape[0] // 2)
                        if fade_out_samples > 0:
                            fade_out_curve = torch.linspace(1, 0, fade_out_samples).unsqueeze(1)
                            segment[-fade_out_samples:] *= fade_out_curve.to(segment.device)
                
                extracted_segments.append(segment)
                
                # Mark these samples as extracted
                remaining_mask[start_sample:end_sample] = False
                
                logger.info(f"  Segment {len(extracted_segments)}: {start_time:.2f}s - {end_time:.2f}s ({segment.shape[0]:,} samples)")
        
        # Process segments based on output mode
        if not extracted_segments:
            segments_audio = torch.zeros((1, audio.shape[1]))
        elif output_mode == "concatenate":
            if gap_handling == "crossfade" and len(extracted_segments) > 1:
                # Crossfade between segments
                segments_audio = extracted_segments[0]
                for next_segment in extracted_segments[1:]:
                    segments_audio = self._crossfade_segments(
                        segments_audio, next_segment, fade_duration, current_sample_rate
                    )
            else:
                # Simple concatenation
                segments_audio = torch.cat(extracted_segments, dim=0)
        elif output_mode == "first_only":
            segments_audio = extracted_segments[0]
        elif output_mode == "last_only":
            segments_audio = extracted_segments[-1]
        elif output_mode == "longest":
            segments_audio = max(extracted_segments, key=lambda x: x.shape[0])
        elif output_mode == "shortest":
            segments_audio = min(extracted_segments, key=lambda x: x.shape[0])
        else:
            segments_audio = torch.cat(extracted_segments, dim=0)
        
        # Create remaining audio
        if include_remaining:
            remaining_audio = audio[remaining_mask]
            if remaining_audio.shape[0] == 0:
                remaining_audio = torch.zeros((1, audio.shape[1]))
        else:
            remaining_audio = torch.zeros((1, audio.shape[1]))
        
        # Calculate info
        total_duration = audio.shape[0] / current_sample_rate
        segments_duration = segments_audio.shape[0] / current_sample_rate
        remaining_duration = remaining_audio.shape[0] / current_sample_rate
        
        segments_info = (
            f"Multi-Segment Extraction Complete\n"
            f"Original Duration: {total_duration:.2f}s\n"
            f"Segments Processed: {len(extracted_segments)}\n"
            f"Output Mode: {output_mode}\n"
            f"Segments Duration: {segments_duration:.2f}s ({segments_audio.shape[0]:,} samples)\n"
            f"Remaining Duration: {remaining_duration:.2f}s ({remaining_audio.shape[0]:,} samples)\n"
            f"Gap Handling: {gap_handling}\n"
            f"Fade Edges: {'Yes' if fade_edges else 'No'}"
        )
        
        logger.info(f"✅ Multi-segment extraction complete: {len(extracted_segments)} segments processed")
        
        return (segments_audio, remaining_audio, segments_info)
    
    def _crossfade_segments(self, segment1: torch.Tensor, segment2: torch.Tensor, 
                           fade_duration: float, sample_rate: int) -> torch.Tensor:
        """Crossfade between two segments."""
        fade_samples = int(fade_duration * sample_rate)
        
        if fade_samples > 0 and segment1.shape[0] >= fade_samples and segment2.shape[0] >= fade_samples:
            # Apply fade out to end of segment1
            fade_out_curve = torch.linspace(1, 0, fade_samples).unsqueeze(1)
            segment1_tail = segment1[-fade_samples:].clone()
            segment1_tail *= fade_out_curve.to(segment1.device)
            
            # Apply fade in to start of segment2
            fade_in_curve = torch.linspace(0, 1, fade_samples).unsqueeze(1)
            segment2_head = segment2[:fade_samples].clone()
            segment2_head *= fade_in_curve.to(segment2.device)
            
            # Mix the crossfade region
            crossfade_region = segment1_tail + segment2_head
            
            # Concatenate
            result = torch.cat([
                segment1[:-fade_samples],
                crossfade_region,
                segment2[fade_samples:]
            ], dim=0)
            
            return result
        else:
            # Can't apply crossfade, just concatenate
            return torch.cat([segment1, segment2], dim=0)