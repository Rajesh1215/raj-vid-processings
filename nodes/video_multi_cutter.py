"""
RajVideoMultiCutter - Advanced multi-point video cutting with segment management
Cut video at multiple points and manage segments efficiently
"""

import torch
import numpy as np
import os
import tempfile
from typing import Tuple, List, Dict
from .utils import (
    get_optimal_device, logger, time_to_frame, frame_to_time,
    get_save_path_incremental
)

class RajVideoMultiCutter:
    """
    Cut video at multiple time points and manage all segments
    Example: 10s video with cuts at [2s, 5s, 8s] → 4 segments: 0-2s, 2-5s, 5-8s, 8-10s
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Input video frames"}),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Frames per second for time calculations"
                }),
                "cut_points": ("STRING", {
                    "default": "2.0, 5.0, 8.0",
                    "tooltip": "Comma-separated cut times in seconds (e.g., '2.0, 5.5, 8.2')"
                }),
                "output_mode": (["all_segments", "selected_segments", "excluded_segments"], {
                    "default": "all_segments",
                    "tooltip": "Output mode: all segments, only selected, or exclude selected"
                })
            },
            "optional": {
                "selected_indices": ("STRING", {
                    "default": "0, 2",
                    "tooltip": "Comma-separated segment indices to select/exclude (0-based)"
                }),
                "timecode_format": (["seconds", "timecode"], {
                    "default": "seconds",
                    "tooltip": "Input format: seconds (5.5) or timecode (00:00:05:12)"
                }),
                "cut_timecodes": ("STRING", {
                    "default": "00:00:02:00, 00:00:05:00, 00:00:08:00",
                    "tooltip": "Comma-separated timecodes (HH:MM:SS:FF)"
                }),
                "force_device": (["auto", "cpu", "cuda", "mps"], {
                    "default": "auto",
                    "tooltip": "Force specific device for processing"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("output_segments", "cut_info", "segment_count", "total_duration", "segment_details")
    FUNCTION = "multi_cut_video"
    CATEGORY = "Raj Video Processing 🎬"
    
    def multi_cut_video(self, frames, fps=24.0, cut_points="2.0, 5.0, 8.0", 
                       output_mode="all_segments", selected_indices="0, 2",
                       timecode_format="seconds", cut_timecodes="00:00:02:00, 00:00:05:00, 00:00:08:00",
                       force_device="auto"):
        
        # Determine device
        if force_device == "auto":
            device = get_optimal_device()
        else:
            device = torch.device(force_device)
        
        # Convert frames to tensor if needed
        if not isinstance(frames, torch.Tensor):
            frames = torch.tensor(frames, dtype=torch.float32)
        
        frames = frames.to(device)
        total_frames = frames.shape[0]
        total_duration = total_frames / fps
        
        logger.info(f"✂️ Multi-cutting video: {total_frames} frames @ {fps}fps ({total_duration:.2f}s)")
        
        # Parse cut points
        if timecode_format == "timecode":
            cut_times = self._parse_timecodes(cut_timecodes, fps)
            logger.info(f"   Cut timecodes: {cut_timecodes}")
        else:
            cut_times = self._parse_cut_points(cut_points)
            logger.info(f"   Cut points: {cut_points}")
        
        # Validate and sort cut times
        cut_times = [t for t in cut_times if 0 < t < total_duration]
        cut_times = sorted(set(cut_times))  # Remove duplicates and sort
        
        logger.info(f"   Valid cut times: {cut_times}")
        
        # Convert to frame indices
        cut_frames = [time_to_frame(t, fps) for t in cut_times]
        
        # Create segment boundaries (start_frame, end_frame)
        segment_boundaries = []
        start_frame = 0
        
        for cut_frame in cut_frames:
            if cut_frame > start_frame:
                segment_boundaries.append((start_frame, cut_frame))
            start_frame = cut_frame
        
        # Add final segment if there are remaining frames
        if start_frame < total_frames:
            segment_boundaries.append((start_frame, total_frames))
        
        logger.info(f"   Created {len(segment_boundaries)} segments")
        
        # Extract segments
        all_segments = []
        segment_info = []
        
        for i, (start_frame, end_frame) in enumerate(segment_boundaries):
            segment = frames[start_frame:end_frame]
            segment_duration = (end_frame - start_frame) / fps
            start_time = frame_to_time(start_frame, fps)
            end_time = frame_to_time(end_frame, fps)
            
            all_segments.append(segment)
            segment_info.append({
                'index': i,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_time,
                'end_time': end_time,
                'duration': segment_duration,
                'frame_count': segment.shape[0]
            })
            
            logger.info(f"   Segment {i}: {start_time:.2f}s-{end_time:.2f}s ({segment.shape[0]} frames)")
        
        # Apply output mode filtering
        if output_mode == "selected_segments":
            selected_idx = self._parse_indices(selected_indices, len(all_segments))
            output_segments = [all_segments[i] for i in selected_idx if i < len(all_segments)]
            selected_info = [segment_info[i] for i in selected_idx if i < len(all_segments)]
            logger.info(f"   Selected segments: {selected_idx}")
        elif output_mode == "excluded_segments":
            excluded_idx = self._parse_indices(selected_indices, len(all_segments))
            output_segments = [all_segments[i] for i in range(len(all_segments)) if i not in excluded_idx]
            selected_info = [segment_info[i] for i in range(len(segment_info)) if i not in excluded_idx]
            logger.info(f"   Excluded segments: {excluded_idx}")
        else:  # all_segments
            output_segments = all_segments
            selected_info = segment_info
        
        # Concatenate selected segments for output
        if output_segments:
            final_output = torch.cat(output_segments, dim=0)
            output_duration = sum(info['duration'] for info in selected_info)
            output_frame_count = final_output.shape[0]
        else:
            # Return empty tensor with same dimensions as input
            final_output = torch.empty((0,) + frames.shape[1:], dtype=frames.dtype, device=device)
            output_duration = 0.0
            output_frame_count = 0
        
        # Generate info strings
        cut_info = (f"Multi-cut: {len(cut_times)} cut points → {len(segment_boundaries)} segments | "
                   f"Output mode: {output_mode} | "
                   f"Final output: {output_frame_count} frames ({output_duration:.2f}s)")
        
        segment_details = self._generate_segment_details(selected_info, output_mode)
        
        # Generate preview files
        try:
            self._create_multi_cut_previews(output_segments, selected_info, fps)
        except Exception as e:
            logger.warning(f"Failed to create multi-cut previews: {e}")
        
        logger.info("✅ Multi-cut video complete")
        
        return (final_output, cut_info, len(selected_info), output_duration, segment_details)
    
    def _parse_cut_points(self, cut_points_str: str) -> List[float]:
        """Parse comma-separated cut points string"""
        try:
            points = []
            for point in cut_points_str.split(','):
                point = point.strip()
                if point:
                    points.append(float(point))
            return points
        except Exception as e:
            logger.error(f"Failed to parse cut points '{cut_points_str}': {e}")
            return []
    
    def _parse_timecodes(self, timecodes_str: str, fps: float) -> List[float]:
        """Parse comma-separated timecodes string"""
        try:
            times = []
            for timecode in timecodes_str.split(','):
                timecode = timecode.strip()
                if timecode:
                    times.append(self._parse_timecode(timecode, fps))
            return times
        except Exception as e:
            logger.error(f"Failed to parse timecodes '{timecodes_str}': {e}")
            return []
    
    def _parse_timecode(self, timecode: str, fps: float) -> float:
        """Parse single timecode (HH:MM:SS:FF) to seconds"""
        try:
            parts = timecode.split(":")
            if len(parts) != 4:
                raise ValueError(f"Invalid timecode format: {timecode}")
            
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            frames = int(parts[3])
            
            return (hours * 3600) + (minutes * 60) + seconds + (frames / fps)
        except Exception as e:
            logger.error(f"Failed to parse timecode '{timecode}': {e}")
            raise ValueError(f"Invalid timecode format: {timecode}")
    
    def _parse_indices(self, indices_str: str, max_count: int) -> List[int]:
        """Parse comma-separated indices string"""
        try:
            indices = []
            for idx in indices_str.split(','):
                idx = idx.strip()
                if idx:
                    index = int(idx)
                    if 0 <= index < max_count:
                        indices.append(index)
            return indices
        except Exception as e:
            logger.error(f"Failed to parse indices '{indices_str}': {e}")
            return []
    
    def _generate_segment_details(self, segment_info: List[Dict], output_mode: str) -> str:
        """Generate detailed segment information string"""
        details = [f"Multi-cut output mode: {output_mode}"]
        details.append(f"Total segments: {len(segment_info)}")
        details.append("Segment details:")
        
        for info in segment_info:
            details.append(f"  [{info['index']}] {info['start_time']:.2f}s-{info['end_time']:.2f}s "
                          f"({info['frame_count']} frames, {info['duration']:.2f}s)")
        
        return "\n".join(details)
    
    def _create_multi_cut_previews(self, segments: List[torch.Tensor], segment_info: List[Dict], fps: float):
        """Create preview files for selected segments"""
        import cv2
        
        for i, (segment, info) in enumerate(zip(segments, segment_info)):
            if segment.shape[0] == 0:
                continue
                
            with tempfile.NamedTemporaryFile(suffix=f"_segment_{info['index']}.mp4", delete=False) as tmp_file:
                preview_path = tmp_file.name
            
            frames_cpu = segment.cpu().numpy()
            frames_uint8 = (frames_cpu * 255).astype(np.uint8)
            
            if len(frames_uint8) > 0:
                height, width = frames_uint8.shape[1:3]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(preview_path, fourcc, fps, (width, height))
                
                for frame in frames_uint8:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                
                out.release()
                logger.info(f"📸 Segment {info['index']} preview saved: {preview_path}")


class RajVideoSegmentManager:
    """
    Utility node for managing and analyzing video segments
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Input video frames"}),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Frames per second"
                }),
                "analysis_mode": (["segment_info", "timing_analysis", "frame_distribution"], {
                    "default": "segment_info",
                    "tooltip": "Analysis type to perform"
                })
            },
            "optional": {
                "segment_duration": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 60.0,
                    "step": 0.1,
                    "tooltip": "Target segment duration for analysis"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("analysis_result", "total_segments", "total_duration", "detailed_info")
    FUNCTION = "analyze_segments"
    CATEGORY = "Raj Video Processing 🎬"
    
    def analyze_segments(self, frames, fps=24.0, analysis_mode="segment_info", segment_duration=2.0):
        
        if not isinstance(frames, torch.Tensor):
            frames = torch.tensor(frames, dtype=torch.float32)
        
        total_frames = frames.shape[0]
        total_duration = total_frames / fps
        
        logger.info(f"📊 Analyzing video segments: {total_frames} frames @ {fps}fps ({total_duration:.2f}s)")
        
        if analysis_mode == "segment_info":
            result = self._analyze_basic_info(frames, fps, total_duration)
        elif analysis_mode == "timing_analysis":
            result = self._analyze_timing(frames, fps, total_duration, segment_duration)
        elif analysis_mode == "frame_distribution":
            result = self._analyze_frame_distribution(frames, fps, total_duration)
        else:
            result = {"analysis": "Unknown analysis mode", "segments": 0, "duration": total_duration, "details": ""}
        
        return (result["analysis"], result["segments"], result["duration"], result["details"])
    
    def _analyze_basic_info(self, frames, fps, total_duration):
        height, width = frames.shape[1], frames.shape[2]
        frame_count = frames.shape[0]
        
        analysis = (f"Video Info: {frame_count} frames, {width}x{height}px, "
                   f"{total_duration:.2f}s @ {fps}fps")
        
        details = f"Frame dimensions: {width}x{height}\nTotal frames: {frame_count}\nDuration: {total_duration:.2f}s\nFPS: {fps}"
        
        return {
            "analysis": analysis,
            "segments": 1,
            "duration": total_duration,
            "details": details
        }
    
    def _analyze_timing(self, frames, fps, total_duration, segment_duration):
        segments_count = int(np.ceil(total_duration / segment_duration))
        
        analysis = f"Timing Analysis: {segments_count} segments of {segment_duration}s each"
        
        details = [f"Target segment duration: {segment_duration}s"]
        details.append(f"Total segments needed: {segments_count}")
        
        for i in range(segments_count):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, total_duration)
            actual_duration = end_time - start_time
            details.append(f"Segment {i}: {start_time:.2f}s-{end_time:.2f}s ({actual_duration:.2f}s)")
        
        return {
            "analysis": analysis,
            "segments": segments_count,
            "duration": total_duration,
            "details": "\n".join(details)
        }
    
    def _analyze_frame_distribution(self, frames, fps, total_duration):
        frame_count = frames.shape[0]
        avg_frames_per_second = frame_count / total_duration if total_duration > 0 else 0
        
        # Analyze frame intensity distribution
        frame_means = torch.mean(frames.view(frame_count, -1), dim=1)
        min_intensity = torch.min(frame_means).item()
        max_intensity = torch.max(frame_means).item()
        avg_intensity = torch.mean(frame_means).item()
        
        analysis = f"Frame Distribution: avg={avg_intensity:.3f}, range={min_intensity:.3f}-{max_intensity:.3f}"
        
        details = [f"Frame count: {frame_count}"]
        details.append(f"Average frames per second: {avg_frames_per_second:.2f}")
        details.append(f"Average frame intensity: {avg_intensity:.3f}")
        details.append(f"Intensity range: {min_intensity:.3f} - {max_intensity:.3f}")
        details.append(f"Intensity variation: {max_intensity - min_intensity:.3f}")
        
        return {
            "analysis": analysis,
            "segments": frame_count,
            "duration": total_duration,
            "details": "\n".join(details)
        }