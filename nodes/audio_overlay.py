import torch
import numpy as np
from typing import Tuple, Optional, List, Dict

# Simple logger replacement for testing
class SimpleLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

try:
    from .utils import logger
except ImportError:
    logger = SimpleLogger()

try:
    from ..utils.audio_utils import AudioProcessor
except ImportError:
    pass  # AudioProcessor not actually used

class RajAudioOverlay:
    """
    Core audio mixing and overlay functionality.
    Supports various mixing modes including add, replace, crossfade, ducking, and insert.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_audio": ("AUDIO", {
                    "tooltip": "Base audio track"
                }),
                "overlay_audio": ("AUDIO", {
                    "tooltip": "Audio to overlay"
                }),
                "overlay_mode": (["add", "replace", "crossfade", "ducking", "insert"], {
                    "default": "add",
                    "tooltip": "How to combine the audio tracks"
                }),
                "overlay_position": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Start time for overlay in seconds"
                }),
                "overlay_end_position": ("FLOAT", {
                    "default": -1.0,
                    "min": -1.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "End position for overlay (-1 for full overlay length)"
                }),
                "current_sample_rate": ("INT", {
                    "default": 22050,
                    "min": 8000,
                    "max": 96000,
                    "step": 1000,
                    "tooltip": "Sample rate of the audio"
                }),
                "overlay_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Volume multiplier for overlay audio"
                }),
                "source_volume": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Source volume during overlay (for ducking)"
                }),
            },
            "optional": {
                "crossfade_duration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Crossfade duration in seconds"
                }),
                "ducking_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Volume level for ducking"
                }),
                "ducking_fade_time": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Fade time for ducking transitions"
                }),
                "trim_to_shortest": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Trim result to shortest audio length"
                }),
                "normalize_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize output to prevent clipping"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("mixed_audio", "status")
    FUNCTION = "overlay_audio"
    CATEGORY = "Raj Video Processing 🎬/Audio"
    
    def overlay_audio(self, source_audio, overlay_audio,
                     overlay_mode: str, overlay_position: float, overlay_end_position: float, current_sample_rate: int,
                     overlay_volume: float, source_volume: float,
                     crossfade_duration: float = 0.5, ducking_threshold: float = 0.3,
                     ducking_fade_time: float = 0.2, trim_to_shortest: bool = False,
                     normalize_output: bool = True):
        """
        Mix two audio tracks with various overlay modes.
        """
        
        # Extract tensors from ComfyUI AUDIO format (dict with waveform and sample_rate)
        if isinstance(source_audio, dict):
            source_tensor = source_audio["waveform"].squeeze(0).transpose(0, 1)  # [batch, channels, samples] -> [samples, channels]
            source_sr = source_audio["sample_rate"]
        else:
            source_tensor = source_audio
            source_sr = current_sample_rate
            
        if isinstance(overlay_audio, dict):
            overlay_tensor = overlay_audio["waveform"].squeeze(0).transpose(0, 1)  # [batch, channels, samples] -> [samples, channels]
            overlay_sr = overlay_audio["sample_rate"]
        else:
            overlay_tensor = overlay_audio
            overlay_sr = current_sample_rate
        
        # Validate inputs
        if source_tensor.numel() == 0 or overlay_tensor.numel() == 0:
            logger.warning("One or both audio inputs are empty")
            return ({"waveform": source_tensor.transpose(0, 1).unsqueeze(0), "sample_rate": source_sr}, "Empty audio input")
        
        # Handle sample rate mismatch
        if source_sr != overlay_sr:
            logger.warning(f"Sample rate mismatch: source {source_sr}Hz vs overlay {overlay_sr}Hz. Using source rate.")
            
        # Ensure same number of channels
        if source_tensor.shape[1] != overlay_tensor.shape[1]:
            # Convert to same channel count (mono or stereo)
            target_channels = max(source_tensor.shape[1], overlay_tensor.shape[1])
            
            if source_tensor.shape[1] < target_channels:
                source_tensor = self._expand_channels(source_tensor, target_channels)
            if overlay_tensor.shape[1] < target_channels:
                overlay_tensor = self._expand_channels(overlay_tensor, target_channels)
        
        # Apply volume adjustments
        source_tensor = source_tensor * source_volume
        overlay_tensor = overlay_tensor * overlay_volume
        
        # Calculate overlay sample positions
        overlay_start = int(overlay_position * source_sr)
        
        # Handle end position - if specified and valid, limit overlay duration
        if overlay_end_position > 0 and overlay_end_position > overlay_position:
            max_overlay_duration = overlay_end_position - overlay_position
            max_overlay_samples = int(max_overlay_duration * source_sr)
            if overlay_tensor.shape[0] > max_overlay_samples:
                overlay_tensor = overlay_tensor[:max_overlay_samples]
                
        overlay_end = overlay_start + overlay_tensor.shape[0]
        
        logger.info(f"🎚️ Overlaying audio: mode={overlay_mode}, position={overlay_position:.2f}s, volumes={overlay_volume:.2f}/{source_volume:.2f}")
        
        # Initialize mixed audio with source
        if trim_to_shortest:
            max_length = min(source_tensor.shape[0], overlay_end)
        else:
            max_length = max(source_tensor.shape[0], overlay_end)
            
        mixed_audio = torch.zeros(max_length, source_tensor.shape[1], dtype=source_tensor.dtype)
        
        # Copy source audio
        source_len = min(source_tensor.shape[0], max_length)
        mixed_audio[:source_len] = source_tensor[:source_len]
        
        # Process based on overlay mode
        if overlay_mode == "add":
            # Simple additive mixing
            end_sample = min(overlay_end, mixed_audio.shape[0])
            overlay_length = end_sample - overlay_start
            if overlay_start < mixed_audio.shape[0] and overlay_length > 0:
                mixed_audio[overlay_start:end_sample] += overlay_tensor[:overlay_length]
                
        elif overlay_mode == "replace":
            # Replace mode - overlay completely replaces source
            end_sample = min(overlay_end, mixed_audio.shape[0])
            overlay_length = end_sample - overlay_start
            if overlay_start < mixed_audio.shape[0] and overlay_length > 0:
                mixed_audio[overlay_start:end_sample] = overlay_tensor[:overlay_length]
                
        elif overlay_mode == "insert":
            # Insert mode - insert overlay and shift source
            before = mixed_audio[:overlay_start]
            after = mixed_audio[overlay_start:]
            mixed_audio = torch.cat([before, overlay_tensor, after], dim=0)
            
        elif overlay_mode == "crossfade":
            # Simple crossfade - fade out source, fade in overlay
            fade_samples = int(crossfade_duration * source_sr)
            end_sample = min(overlay_end, mixed_audio.shape[0])
            overlay_length = end_sample - overlay_start
            
            if overlay_start < mixed_audio.shape[0] and overlay_length > 0:
                # Fade out source at overlay position
                fade_end = min(overlay_start + fade_samples, mixed_audio.shape[0])
                if fade_end > overlay_start:
                    fade_out = torch.linspace(1, 0, fade_end - overlay_start).unsqueeze(1)
                    mixed_audio[overlay_start:fade_end] *= fade_out
                
                # Fade in overlay
                fade_in_end = min(overlay_length, fade_samples)
                fade_in = torch.linspace(0, 1, fade_in_end).unsqueeze(1)
                overlay_faded = overlay_tensor[:overlay_length].clone()
                overlay_faded[:fade_in_end] *= fade_in
                
                mixed_audio[overlay_start:end_sample] += overlay_faded
                
        else:
            logger.warning(f"Unknown overlay mode: {overlay_mode}, using 'add'")
            # Default to add mode
            end_sample = min(overlay_end, mixed_audio.shape[0])
            overlay_length = end_sample - overlay_start
            if overlay_start < mixed_audio.shape[0] and overlay_length > 0:
                mixed_audio[overlay_start:end_sample] += overlay_tensor[:overlay_length]
        
        # Note: trim_to_shortest already handled in max_length calculation above
        
        # Normalize output to prevent clipping
        if normalize_output:
            max_val = torch.abs(mixed_audio).max()
            if max_val > 1.0:
                mixed_audio = mixed_audio / max_val * 0.95
                logger.info(f"📊 Normalized output (peak was {max_val:.2f})")
        
        # Generate status message
        source_duration = source_tensor.shape[0] / source_sr
        overlay_duration = overlay_tensor.shape[0] / source_sr
        mixed_duration = mixed_audio.shape[0] / source_sr
        
        status_msg = (
            f"Audio Overlay Complete - Mode: {overlay_mode}\n"
            f"Source: {source_duration:.2f}s, Overlay: {overlay_duration:.2f}s\n"
            f"Output: {mixed_duration:.2f}s at position {overlay_position:.2f}s"
            + (f" to {overlay_end_position:.2f}s" if overlay_end_position > 0 and overlay_end_position > overlay_position else "")
        )
        
        logger.info(f"✅ Audio overlay completed: {overlay_mode} mode, {mixed_duration:.2f}s output")
        
        # Return in ComfyUI AUDIO format
        result_audio = {
            "waveform": mixed_audio.transpose(0, 1).unsqueeze(0),  # [samples, channels] -> [batch, channels, samples]
            "sample_rate": source_sr
        }
        
        return (result_audio, status_msg)
    
    def _expand_channels(self, audio: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Expand mono to stereo or multi-channel."""
        current_channels = audio.shape[1]
        if current_channels == 1 and target_channels > 1:
            # Duplicate mono to all channels
            return audio.repeat(1, target_channels)
        return audio
    
    def _mix_add(self, source: torch.Tensor, overlay: torch.Tensor, 
                 start_sample: int, source_volume: float) -> torch.Tensor:
        """Additive mixing - overlay is added to source."""
        # Apply source volume
        mixed = source.clone() * source_volume
        
        # Calculate overlay range
        end_sample = min(start_sample + overlay.shape[0], mixed.shape[0])
        overlay_length = end_sample - start_sample
        
        if start_sample < mixed.shape[0] and overlay_length > 0:
            # Add overlay to the appropriate section
            mixed[start_sample:end_sample] += overlay[:overlay_length]
        
        # If overlay extends beyond source, append it
        if start_sample + overlay.shape[0] > mixed.shape[0]:
            remaining_overlay = overlay[mixed.shape[0] - start_sample:]
            mixed = torch.cat([mixed, remaining_overlay], dim=0)
        
        return mixed
    
    def _mix_replace(self, source: torch.Tensor, overlay: torch.Tensor, 
                    start_sample: int) -> torch.Tensor:
        """Replace mode - overlay replaces source at position."""
        mixed = source.clone()
        
        # Calculate replacement range
        end_sample = min(start_sample + overlay.shape[0], mixed.shape[0])
        overlay_length = end_sample - start_sample
        
        if start_sample < mixed.shape[0] and overlay_length > 0:
            # Replace the section with overlay
            mixed[start_sample:end_sample] = overlay[:overlay_length]
        
        # If overlay extends beyond source, append it
        if start_sample + overlay.shape[0] > mixed.shape[0]:
            remaining_overlay = overlay[mixed.shape[0] - start_sample:]
            mixed = torch.cat([mixed, remaining_overlay], dim=0)
        
        return mixed
    
    def _mix_crossfade(self, source: torch.Tensor, overlay: torch.Tensor,
                      start_sample: int, fade_duration: float, sample_rate: int) -> torch.Tensor:
        """Crossfade between source and overlay."""
        mixed = source.clone()
        fade_samples = int(fade_duration * sample_rate)
        
        # Calculate overlay range
        end_sample = min(start_sample + overlay.shape[0], mixed.shape[0])
        overlay_length = end_sample - start_sample
        
        if start_sample < mixed.shape[0] and overlay_length > 0:
            # Apply crossfade
            actual_fade_samples = min(fade_samples, overlay_length, mixed.shape[0] - start_sample)
            
            if actual_fade_samples > 0:
                # Fade out source
                fade_out_curve = torch.linspace(1, 0, actual_fade_samples).unsqueeze(1)
                mixed[start_sample:start_sample + actual_fade_samples] *= fade_out_curve.to(mixed.device)
                
                # Fade in overlay
                fade_in_curve = torch.linspace(0, 1, actual_fade_samples).unsqueeze(1)
                overlay_faded = overlay[:actual_fade_samples] * fade_in_curve.to(overlay.device)
                
                # Mix the crossfade region
                mixed[start_sample:start_sample + actual_fade_samples] += overlay_faded
                
                # Add remaining overlay without fade
                if overlay_length > actual_fade_samples:
                    mixed[start_sample + actual_fade_samples:end_sample] = overlay[actual_fade_samples:overlay_length]
            else:
                # No fade, just replace
                mixed[start_sample:end_sample] = overlay[:overlay_length]
        
        # If overlay extends beyond source, append it
        if start_sample + overlay.shape[0] > mixed.shape[0]:
            remaining_overlay = overlay[mixed.shape[0] - start_sample:]
            mixed = torch.cat([mixed, remaining_overlay], dim=0)
        
        return mixed
    
    def _mix_ducking(self, source: torch.Tensor, overlay: torch.Tensor,
                    start_sample: int, threshold: float, fade_time: float, sample_rate: int) -> torch.Tensor:
        """Ducking - reduce source volume when overlay is present."""
        mixed = source.clone()
        fade_samples = int(fade_time * sample_rate)
        
        # Calculate overlay range
        end_sample = min(start_sample + overlay.shape[0], mixed.shape[0])
        overlay_length = end_sample - start_sample
        
        if start_sample < mixed.shape[0] and overlay_length > 0:
            # Apply ducking envelope to source
            duck_start = max(0, start_sample - fade_samples)
            duck_end = min(mixed.shape[0], end_sample + fade_samples)
            
            # Fade down
            if start_sample > duck_start:
                fade_down_samples = start_sample - duck_start
                fade_down_curve = torch.linspace(1, threshold, fade_down_samples).unsqueeze(1)
                mixed[duck_start:start_sample] *= fade_down_curve.to(mixed.device)
            
            # Apply ducking level
            mixed[start_sample:end_sample] *= threshold
            
            # Fade up
            if duck_end > end_sample:
                fade_up_samples = duck_end - end_sample
                fade_up_curve = torch.linspace(threshold, 1, fade_up_samples).unsqueeze(1)
                mixed[end_sample:duck_end] *= fade_up_curve.to(mixed.device)
            
            # Add overlay
            mixed[start_sample:end_sample] += overlay[:overlay_length]
        
        # If overlay extends beyond source, append it
        if start_sample + overlay.shape[0] > mixed.shape[0]:
            remaining_overlay = overlay[mixed.shape[0] - start_sample:]
            mixed = torch.cat([mixed, remaining_overlay], dim=0)
        
        return mixed
    
    def _mix_insert(self, source: torch.Tensor, overlay: torch.Tensor,
                   start_sample: int) -> torch.Tensor:
        """Insert mode - insert overlay and push source forward."""
        if start_sample >= source.shape[0]:
            # Insert at end is same as concatenate
            return torch.cat([source, overlay], dim=0)
        elif start_sample <= 0:
            # Insert at beginning
            return torch.cat([overlay, source], dim=0)
        else:
            # Split source and insert overlay
            before = source[:start_sample]
            after = source[start_sample:]
            return torch.cat([before, overlay, after], dim=0)


class RajAudioMultiOverlay:
    """
    Advanced multi-track audio composition.
    Supports up to 4 overlay tracks with individual positioning and volume control.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_audio": ("AUDIO", {
                    "tooltip": "Primary audio track"
                }),
                "current_sample_rate": ("INT", {
                    "default": 22050,
                    "min": 8000,
                    "max": 96000,
                    "step": 1000,
                    "tooltip": "Sample rate of the audio"
                }),
                "mix_method": (["additive", "normalized", "compressed", "layered"], {
                    "default": "normalized",
                    "tooltip": "How to combine multiple overlays"
                }),
            },
            "optional": {
                "overlay_1": ("AUDIO", {
                    "tooltip": "First overlay track (optional)"
                }),
                "overlay_2": ("AUDIO", {
                    "tooltip": "Second overlay track (optional)"
                }),
                "overlay_3": ("AUDIO", {
                    "tooltip": "Third overlay track (optional)"
                }),
                "overlay_4": ("AUDIO", {
                    "tooltip": "Fourth overlay track (optional)"
                }),
                "positions": ("STRING", {
                    "default": "0.0,5.0,10.0,15.0",
                    "tooltip": "Comma-separated start times in seconds"
                }),
                "volumes": ("STRING", {
                    "default": "1.0,0.8,0.6,0.4",
                    "tooltip": "Comma-separated volume levels (0.0-2.0)"
                }),
                "auto_duck": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Automatically duck base when overlays are present"
                }),
                "duck_amount": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How much to duck the base audio"
                }),
                "normalize_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize output to prevent clipping"
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("mixed_audio", "track_info")
    FUNCTION = "multi_overlay"
    CATEGORY = "Raj Video Processing 🎬/Audio"
    
    def multi_overlay(self, base_audio: torch.Tensor, current_sample_rate: int, mix_method: str,
                     overlay_1: Optional[torch.Tensor] = None, overlay_2: Optional[torch.Tensor] = None,
                     overlay_3: Optional[torch.Tensor] = None, overlay_4: Optional[torch.Tensor] = None,
                     positions: str = "0.0,5.0,10.0,15.0", volumes: str = "1.0,0.8,0.6,0.4",
                     auto_duck: bool = False, duck_amount: float = 0.5,
                     normalize_output: bool = True) -> Tuple[torch.Tensor, str]:
        """
        Mix multiple audio tracks with timeline-based positioning.
        """
        
        if base_audio.numel() == 0:
            logger.warning("Base audio is empty")
            return (base_audio, "Base audio was empty")
        
        # Parse positions and volumes
        try:
            position_list = [float(p.strip()) for p in positions.split(',')]
            volume_list = [float(v.strip()) for v in volumes.split(',')]
        except Exception as e:
            logger.error(f"Error parsing positions/volumes: {e}")
            return (base_audio, f"Failed to parse parameters: {str(e)}")
        
        # Collect active overlays
        overlays = []
        if overlay_1 is not None and overlay_1.numel() > 0:
            overlays.append((overlay_1, 0))
        if overlay_2 is not None and overlay_2.numel() > 0:
            overlays.append((overlay_2, 1))
        if overlay_3 is not None and overlay_3.numel() > 0:
            overlays.append((overlay_3, 2))
        if overlay_4 is not None and overlay_4.numel() > 0:
            overlays.append((overlay_4, 3))
        
        if not overlays:
            logger.info("No overlay tracks provided")
            return (base_audio, "No overlays to mix")
        
        logger.info(f"🎛️ Multi-overlay: {len(overlays)} tracks, method={mix_method}")
        
        # Ensure all audio has same channels
        target_channels = base_audio.shape[1]
        for i, (overlay, _) in enumerate(overlays):
            if overlay.shape[1] != target_channels:
                if overlay.shape[1] == 1 and target_channels > 1:
                    overlays[i] = (overlay.repeat(1, target_channels), overlays[i][1])
        
        # Start with base audio
        mixed = base_audio.clone()
        
        # Calculate maximum length needed
        max_length = base_audio.shape[0]
        for overlay, idx in overlays:
            if idx < len(position_list):
                position_samples = int(position_list[idx] * current_sample_rate)
                potential_length = position_samples + overlay.shape[0]
                max_length = max(max_length, potential_length)
        
        # Extend mixed audio if needed
        if max_length > mixed.shape[0]:
            padding = torch.zeros((max_length - mixed.shape[0], mixed.shape[1]))
            mixed = torch.cat([mixed, padding], dim=0)
        
        # Apply auto-ducking if requested
        if auto_duck:
            # Create ducking envelope for base audio
            duck_envelope = torch.ones(mixed.shape[0])
            
            for overlay, idx in overlays:
                if idx < len(position_list):
                    start_sample = int(position_list[idx] * current_sample_rate)
                    end_sample = min(start_sample + overlay.shape[0], mixed.shape[0])
                    
                    # Apply ducking during overlay
                    duck_envelope[start_sample:end_sample] *= (1.0 - duck_amount)
            
            # Smooth the ducking envelope
            duck_envelope = self._smooth_envelope(duck_envelope, int(0.05 * current_sample_rate))
            mixed *= duck_envelope.unsqueeze(1).to(mixed.device)
        
        # Mix overlays based on method
        if mix_method == "additive":
            # Simple addition
            for overlay, idx in overlays:
                if idx < len(position_list) and idx < len(volume_list):
                    start_sample = int(position_list[idx] * current_sample_rate)
                    volume = volume_list[idx]
                    end_sample = min(start_sample + overlay.shape[0], mixed.shape[0])
                    
                    if start_sample < mixed.shape[0]:
                        overlay_length = end_sample - start_sample
                        mixed[start_sample:end_sample] += overlay[:overlay_length] * volume
        
        elif mix_method == "normalized":
            # Track number of overlapping sources for normalization
            overlap_count = torch.ones(mixed.shape[0])
            
            for overlay, idx in overlays:
                if idx < len(position_list) and idx < len(volume_list):
                    start_sample = int(position_list[idx] * current_sample_rate)
                    volume = volume_list[idx]
                    end_sample = min(start_sample + overlay.shape[0], mixed.shape[0])
                    
                    if start_sample < mixed.shape[0]:
                        overlay_length = end_sample - start_sample
                        mixed[start_sample:end_sample] += overlay[:overlay_length] * volume
                        overlap_count[start_sample:end_sample] += 1
            
            # Normalize by overlap count
            mixed /= overlap_count.unsqueeze(1).to(mixed.device)
        
        elif mix_method == "compressed":
            # Apply compression-like mixing
            for overlay, idx in overlays:
                if idx < len(position_list) and idx < len(volume_list):
                    start_sample = int(position_list[idx] * current_sample_rate)
                    volume = volume_list[idx]
                    end_sample = min(start_sample + overlay.shape[0], mixed.shape[0])
                    
                    if start_sample < mixed.shape[0]:
                        overlay_length = end_sample - start_sample
                        # Compress the sum
                        section = mixed[start_sample:end_sample]
                        overlay_section = overlay[:overlay_length] * volume
                        mixed[start_sample:end_sample] = self._soft_clip(section + overlay_section)
        
        elif mix_method == "layered":
            # Layer with decreasing influence
            layer_factor = 1.0
            for overlay, idx in overlays:
                if idx < len(position_list) and idx < len(volume_list):
                    start_sample = int(position_list[idx] * current_sample_rate)
                    volume = volume_list[idx] * layer_factor
                    end_sample = min(start_sample + overlay.shape[0], mixed.shape[0])
                    
                    if start_sample < mixed.shape[0]:
                        overlay_length = end_sample - start_sample
                        mixed[start_sample:end_sample] += overlay[:overlay_length] * volume
                    
                    layer_factor *= 0.75  # Each layer has less influence
        
        # Normalize output if requested
        if normalize_output:
            max_val = torch.abs(mixed).max()
            if max_val > 1.0:
                mixed = mixed / max_val * 0.95
                logger.info(f"📊 Normalized output (peak was {max_val:.2f})")
        
        # Generate track info
        base_duration = base_audio.shape[0] / current_sample_rate
        mixed_duration = mixed.shape[0] / current_sample_rate
        
        track_info_lines = [
            f"Multi-Track Mix Complete",
            f"Mix Method: {mix_method}",
            f"Base Duration: {base_duration:.2f}s",
            f"Mixed Duration: {mixed_duration:.2f}s",
            f"Tracks Mixed: {len(overlays) + 1} (base + {len(overlays)} overlays)",
            f"Auto Duck: {'Yes' if auto_duck else 'No'}"
        ]
        
        for i, (overlay, idx) in enumerate(overlays):
            if idx < len(position_list) and idx < len(volume_list):
                overlay_duration = overlay.shape[0] / current_sample_rate
                track_info_lines.append(
                    f"Track {i+2}: Position={position_list[idx]:.2f}s, "
                    f"Duration={overlay_duration:.2f}s, Volume={volume_list[idx]:.2f}"
                )
        
        track_info = "\n".join(track_info_lines)
        
        logger.info(f"✅ Multi-track overlay complete: {len(overlays) + 1} tracks mixed")
        
        return (mixed, track_info)
    
    def _smooth_envelope(self, envelope: torch.Tensor, window_size: int) -> torch.Tensor:
        """Smooth an envelope using a moving average."""
        if window_size <= 1:
            return envelope
        
        # Simple moving average
        kernel = torch.ones(window_size) / window_size
        # Pad the envelope
        padded = torch.nn.functional.pad(envelope, (window_size//2, window_size//2), mode='reflect')
        # Convolve
        smoothed = torch.nn.functional.conv1d(
            padded.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=0
        ).squeeze()
        
        return smoothed[:envelope.shape[0]]
    
    def _soft_clip(self, audio: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
        """Apply soft clipping to prevent harsh distortion."""
        # Soft clipping using tanh
        return threshold * torch.tanh(audio / threshold)