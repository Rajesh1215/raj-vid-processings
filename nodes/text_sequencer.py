import torch
import numpy as np
from PIL import Image, ImageDraw
import json
import math
from typing import Dict, List, Tuple, Optional, Union
from .utils import logger

class RajTextSequencer:
    """
    Timeline control node for precise text timing and sequencing.
    Handles multiple text elements with frame-accurate positioning and opacity curves.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sequence_mode": (["manual", "auto_duration", "whisper_sync", "beat_sync"], {
                    "default": "manual",
                    "tooltip": "Sequencing mode"
                }),
                "total_duration": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.1,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Total sequence duration in seconds"
                }),
                "fps": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Output frame rate"
                }),
                "output_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Output sequence width"
                }),
                "output_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Output sequence height"
                }),
                "background_color": ("STRING", {
                    "default": "transparent",
                    "tooltip": "Background color (HEX or 'transparent')"
                }),
            },
            "optional": {
                # Text element 1
                "text_image_1": ("IMAGE", {"tooltip": "First text element"}),
                "start_time_1": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "Start time for element 1 (seconds)"
                }),
                "end_time_1": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "tooltip": "End time for element 1 (seconds)"
                }),
                "position_x_1": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "X position for element 1"
                }),
                "position_y_1": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Y position for element 1"
                }),
                "opacity_curve_1": (["constant", "fade_in", "fade_out", "fade_in_out", "pulse", "custom"], {
                    "default": "constant",
                    "tooltip": "Opacity curve for element 1"
                }),
                
                # Text element 2
                "text_image_2": ("IMAGE", {"tooltip": "Second text element"}),
                "start_time_2": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1
                }),
                "end_time_2": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1
                }),
                "position_x_2": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "position_y_2": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "opacity_curve_2": (["constant", "fade_in", "fade_out", "fade_in_out", "pulse", "custom"], {
                    "default": "constant"
                }),
                
                # Text element 3
                "text_image_3": ("IMAGE", {"tooltip": "Third text element"}),
                "start_time_3": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "end_time_3": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "position_x_3": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "position_y_3": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "opacity_curve_3": (["constant", "fade_in", "fade_out", "fade_in_out", "pulse", "custom"], {
                    "default": "constant"
                }),
                
                # Text element 4
                "text_image_4": ("IMAGE", {"tooltip": "Fourth text element"}),
                "start_time_4": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "end_time_4": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "position_x_4": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "position_y_4": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "opacity_curve_4": (["constant", "fade_in", "fade_out", "fade_in_out", "pulse", "custom"], {
                    "default": "constant"
                }),
                
                # Text element 5
                "text_image_5": ("IMAGE", {"tooltip": "Fifth text element"}),
                "start_time_5": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "end_time_5": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "position_x_5": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "position_y_5": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                "opacity_curve_5": (["constant", "fade_in", "fade_out", "fade_in_out", "pulse", "custom"], {
                    "default": "constant"
                }),
                
                # Caption data from Whisper
                "caption_data": ("STRING", {
                    "default": "{}",
                    "tooltip": "Caption timing JSON from RajWhisperCaptions"
                }),
                
                # Custom opacity keyframes
                "opacity_keyframes": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Custom opacity keyframes (format: time1:value1,time2:value2)"
                }),
                
                # Beat sync parameters
                "bpm": ("FLOAT", {
                    "default": 120.0,
                    "min": 60.0,
                    "max": 200.0,
                    "step": 1.0,
                    "tooltip": "Beats per minute for beat sync mode"
                }),
                
                # Transition settings
                "transition_duration": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Crossfade transition duration between elements"
                }),
                
                # Global settings
                "global_opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Global opacity multiplier for all elements"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("sequenced_text", "timeline_data", "sequence_info")
    FUNCTION = "sequence_text"
    CATEGORY = "Raj Video Processing 🎬/Text"
    
    @staticmethod
    def parse_color(color_str: str) -> Tuple[int, int, int, int]:
        """Parse color string to RGBA tuple."""
        if color_str.lower() == "transparent":
            return (0, 0, 0, 0)
        
        if color_str.startswith("#"):
            color_str = color_str[1:]
        
        if len(color_str) == 6:
            r = int(color_str[0:2], 16)
            g = int(color_str[2:4], 16)
            b = int(color_str[4:6], 16)
            return (r, g, b, 255)
        elif len(color_str) == 8:
            r = int(color_str[0:2], 16)
            g = int(color_str[2:4], 16)
            b = int(color_str[4:6], 16)
            a = int(color_str[6:8], 16)
            return (r, g, b, a)
        
        return (0, 0, 0, 0)
    
    @staticmethod
    def calculate_opacity(curve_type: str, time_progress: float, 
                         transition_duration: float = 0.3) -> float:
        """Calculate opacity based on curve type and time progress."""
        if curve_type == "constant":
            return 1.0
        
        elif curve_type == "fade_in":
            fade_progress = min(1.0, time_progress / transition_duration)
            return fade_progress
        
        elif curve_type == "fade_out":
            fade_progress = min(1.0, (1.0 - time_progress) / transition_duration)
            return fade_progress
        
        elif curve_type == "fade_in_out":
            if time_progress < transition_duration:
                return time_progress / transition_duration
            elif time_progress > (1.0 - transition_duration):
                return (1.0 - time_progress) / transition_duration
            else:
                return 1.0
        
        elif curve_type == "pulse":
            return 0.5 + 0.5 * math.sin(time_progress * math.pi * 6)
        
        return 1.0
    
    @staticmethod
    def parse_keyframes(keyframe_str: str) -> List[Tuple[float, float]]:
        """Parse custom keyframe string."""
        keyframes = []
        if not keyframe_str.strip():
            return keyframes
        
        try:
            pairs = keyframe_str.strip().split(',')
            for pair in pairs:
                time_str, value_str = pair.split(':')
                time = float(time_str.strip())
                value = float(value_str.strip())
                keyframes.append((time, value))
        except Exception as e:
            logger.warning(f"Error parsing keyframes: {e}")
        
        return sorted(keyframes, key=lambda x: x[0])
    
    @staticmethod
    def interpolate_keyframes(keyframes: List[Tuple[float, float]], time: float) -> float:
        """Interpolate opacity value from keyframes."""
        if not keyframes:
            return 1.0
        
        # If before first keyframe
        if time <= keyframes[0][0]:
            return keyframes[0][1]
        
        # If after last keyframe
        if time >= keyframes[-1][0]:
            return keyframes[-1][1]
        
        # Find surrounding keyframes
        for i in range(len(keyframes) - 1):
            t1, v1 = keyframes[i]
            t2, v2 = keyframes[i + 1]
            
            if t1 <= time <= t2:
                # Linear interpolation
                progress = (time - t1) / (t2 - t1)
                return v1 + (v2 - v1) * progress
        
        return 1.0
    
    def process_caption_data(self, caption_data_str: str, total_duration: float) -> List[Dict]:
        """Process caption data from Whisper node."""
        try:
            caption_data = json.loads(caption_data_str) if caption_data_str else {}
            captions = caption_data.get('captions', [])
            
            # Convert captions to text elements
            text_elements = []
            for i, caption in enumerate(captions):
                if caption.get('start', 0) < total_duration:
                    text_elements.append({
                        'index': i + 1,
                        'start_time': caption.get('start', 0),
                        'end_time': min(caption.get('end', caption.get('start', 0) + 2), total_duration),
                        'text': caption.get('text', ''),
                        'position_x': 0,
                        'position_y': 0,
                        'opacity_curve': 'fade_in_out'
                    })
            
            return text_elements
        except Exception as e:
            logger.error(f"Error processing caption data: {e}")
            return []
    
    def sequence_text(self, sequence_mode, total_duration, fps, output_width, output_height,
                     background_color, **kwargs):
        
        # Parse background color
        bg_color = self.parse_color(background_color)
        
        # Calculate total frames
        total_frames = int(total_duration * fps)
        
        # Collect text elements
        text_elements = []
        
        # Process based on sequence mode
        if sequence_mode == "whisper_sync":
            # Use caption data
            caption_data = kwargs.get('caption_data', '{}')
            text_elements = self.process_caption_data(caption_data, total_duration)
        
        elif sequence_mode == "beat_sync":
            # Sync to BPM
            bpm = kwargs.get('bpm', 120.0)
            beat_duration = 60.0 / bpm
            
            # Create beat-synced elements from available text images
            for i in range(1, 6):  # Up to 5 text elements
                text_key = f'text_image_{i}'
                if text_key in kwargs and kwargs[text_key] is not None:
                    start_time = (i - 1) * beat_duration * 4  # 4 beats per element
                    end_time = start_time + beat_duration * 2
                    
                    if start_time < total_duration:
                        text_elements.append({
                            'index': i,
                            'image': kwargs[text_key],
                            'start_time': start_time,
                            'end_time': min(end_time, total_duration),
                            'position_x': kwargs.get(f'position_x_{i}', 0),
                            'position_y': kwargs.get(f'position_y_{i}', 0),
                            'opacity_curve': kwargs.get(f'opacity_curve_{i}', 'fade_in_out')
                        })
        
        else:
            # Manual or auto_duration mode
            for i in range(1, 6):  # Up to 5 text elements
                text_key = f'text_image_{i}'
                start_key = f'start_time_{i}'
                end_key = f'end_time_{i}'
                
                if text_key in kwargs and kwargs[text_key] is not None:
                    start_time = kwargs.get(start_key, 0.0)
                    end_time = kwargs.get(end_key, 2.0)
                    
                    if sequence_mode == "auto_duration":
                        # Distribute elements evenly
                        element_duration = total_duration / 5
                        start_time = (i - 1) * element_duration
                        end_time = start_time + element_duration * 0.8  # 80% overlap
                    
                    if start_time < total_duration:
                        text_elements.append({
                            'index': i,
                            'image': kwargs[text_key],
                            'start_time': start_time,
                            'end_time': min(end_time, total_duration),
                            'position_x': kwargs.get(f'position_x_{i}', 0),
                            'position_y': kwargs.get(f'position_y_{i}', 0),
                            'opacity_curve': kwargs.get(f'opacity_curve_{i}', 'constant')
                        })
        
        # Parse custom opacity keyframes
        opacity_keyframes_str = kwargs.get('opacity_keyframes', '')
        global_keyframes = self.parse_keyframes(opacity_keyframes_str)
        
        # Get global settings
        transition_duration = kwargs.get('transition_duration', 0.3)
        global_opacity = kwargs.get('global_opacity', 1.0)
        
        # Generate sequence frames
        frames = []
        timeline_events = []
        
        for frame_idx in range(total_frames):
            current_time = frame_idx / fps
            
            # Create background frame
            frame = Image.new('RGBA', (output_width, output_height), bg_color)
            
            # Process each text element
            active_elements = []
            
            for element in text_elements:
                start_time = element['start_time']
                end_time = element['end_time']
                
                # Check if element is active at current time
                if start_time <= current_time <= end_time:
                    duration = end_time - start_time
                    time_progress = (current_time - start_time) / max(duration, 0.001)
                    
                    # Calculate opacity
                    curve_opacity = self.calculate_opacity(
                        element['opacity_curve'], time_progress, transition_duration
                    )
                    
                    # Apply global keyframes if present
                    if global_keyframes:
                        keyframe_opacity = self.interpolate_keyframes(global_keyframes, current_time)
                        curve_opacity *= keyframe_opacity
                    
                    # Apply global opacity
                    final_opacity = curve_opacity * global_opacity
                    
                    if final_opacity > 0.01:  # Only process visible elements
                        # Get text image
                        text_image_tensor = element['image']
                        if text_image_tensor is not None:
                            # Convert tensor to PIL image
                            img_np = (text_image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
                            if img_np.shape[2] == 3:
                                alpha = np.ones((img_np.shape[0], img_np.shape[1]), dtype=np.uint8) * 255
                                img_np = np.dstack([img_np, alpha])
                            
                            text_img = Image.fromarray(img_np, 'RGBA')
                            
                            # Apply opacity
                            img_array = np.array(text_img)
                            img_array[:, :, 3] = (img_array[:, :, 3] * final_opacity).astype(np.uint8)
                            text_img = Image.fromarray(img_array, 'RGBA')
                            
                            # Calculate position
                            pos_x = element['position_x'] + (output_width - text_img.width) // 2
                            pos_y = element['position_y'] + (output_height - text_img.height) // 2
                            
                            # Composite onto frame
                            if text_img.mode == 'RGBA':
                                frame = Image.alpha_composite(frame, Image.new('RGBA', frame.size, (0, 0, 0, 0)))
                                frame.paste(text_img, (pos_x, pos_y), text_img)
                            else:
                                frame.paste(text_img, (pos_x, pos_y))
                            
                            active_elements.append({
                                'index': element['index'],
                                'opacity': final_opacity,
                                'position': (pos_x, pos_y)
                            })
            
            # Record timeline events
            if active_elements:
                timeline_events.append({
                    'time': current_time,
                    'frame': frame_idx,
                    'active_elements': active_elements
                })
            
            frames.append(frame)
        
        # Convert frames to tensor
        if frames:
            frames_np = np.array([np.array(frame).astype(np.float32) / 255.0 for frame in frames])
            frames_tensor = torch.from_numpy(frames_np)
        else:
            # Empty sequence
            empty_frame = np.zeros((1, output_height, output_width, 4), dtype=np.float32)
            frames_tensor = torch.from_numpy(empty_frame)
        
        # Create timeline data
        timeline_data = {
            "total_duration": total_duration,
            "total_frames": total_frames,
            "fps": fps,
            "sequence_mode": sequence_mode,
            "text_elements": len(text_elements),
            "timeline_events": len(timeline_events),
            "output_size": [output_width, output_height]
        }
        
        # Create sequence info
        sequence_info = (
            f"Text Sequence Generated\n"
            f"Mode: {sequence_mode}\n"
            f"Duration: {total_duration:.1f}s @ {fps:.1f} FPS\n"
            f"Frames: {total_frames}\n"
            f"Text elements: {len(text_elements)}\n"
            f"Timeline events: {len(timeline_events)}\n"
            f"Output: {output_width}x{output_height}"
        )
        
        return (frames_tensor, json.dumps(timeline_data), sequence_info)


# Test function
if __name__ == "__main__":
    node = RajTextSequencer()
    print("Text Sequencer node initialized")
    print("Supported sequence modes:", node.INPUT_TYPES()["required"]["sequence_mode"][0])