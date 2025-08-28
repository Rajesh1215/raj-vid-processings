import os
import torch
import cv2
from .utils import get_optimal_device, video_to_tensor, tensor_to_video_frames

try:
    import folder_paths
except ImportError:
    # Fallback for testing outside ComfyUI
    class MockFolderPaths:
        @staticmethod
        def get_input_directory():
            return "input"
        
        @staticmethod  
        def get_base_path():
            return "."
            
        base_path = "."
    
    folder_paths = MockFolderPaths()

class RajVideoLoader:
    """
    Load video files with GPU acceleration (MPS/CUDA/CPU)
    Optimized for cross-platform performance
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "input/video.mp4", 
                    "multiline": False,
                    "tooltip": "Path to video file (relative to ComfyUI directory)"
                }),
                "target_fps": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 120.0, 
                    "step": 0.1,
                    "tooltip": "Target frame rate (0 = keep original)"
                }),
                "max_frames": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 10000, 
                    "step": 1,
                    "tooltip": "Maximum frames to load (0 = load all)"
                }),
            },
            "optional": {
                "target_width": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 4096, 
                    "step": 8,
                    "tooltip": "Target width (0 = keep original)"
                }),
                "target_height": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 4096, 
                    "step": 8,
                    "tooltip": "Target height (0 = keep original)"
                }),
                "force_device": (["auto", "cpu", "cuda", "mps"], {
                    "default": "auto",
                    "tooltip": "Force specific device (auto = best available)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("frames", "video_info", "frame_count", "fps")
    FUNCTION = "load_video"
    CATEGORY = "Raj Video Processing 🎬"
    
    def load_video(self, video_path, target_fps=0.0, max_frames=0, 
                   target_width=0, target_height=0, force_device="auto"):
        
        # Determine device
        if force_device == "auto":
            device = get_optimal_device()
        else:
            device = torch.device(force_device)
        
        # Resolve path
        if not os.path.isabs(video_path):
            # Try to find in ComfyUI directory structure
            full_path = os.path.join(folder_paths.base_path, video_path)
            if not os.path.exists(full_path):
                # Try in input directory
                input_dir = folder_paths.get_input_directory()
                full_path = os.path.join(input_dir, os.path.basename(video_path))
        else:
            full_path = video_path
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Video file not found: {full_path}")
        
        # Determine target size
        target_size = None
        if target_width > 0 and target_height > 0:
            target_size = (target_width, target_height)
        elif target_width > 0:
            # Calculate height from aspect ratio
            cap = cv2.VideoCapture(full_path)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            aspect_ratio = original_height / original_width
            target_size = (target_width, int(target_width * aspect_ratio))
        elif target_height > 0:
            # Calculate width from aspect ratio
            cap = cv2.VideoCapture(full_path)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            aspect_ratio = original_width / original_height
            target_size = (int(target_height * aspect_ratio), target_height)
        
        # Load video
        try:
            frames_tensor, video_info = video_to_tensor(
                full_path, 
                device, 
                target_fps, 
                max_frames, 
                target_size
            )
            
            # Convert to ComfyUI format
            frames_comfy = tensor_to_video_frames(frames_tensor)
            
            # Create info string
            info_str = f"Device: {video_info['device']} | " \
                      f"Frames: {video_info['total_frames']} | " \
                      f"FPS: {video_info['fps']:.2f} | " \
                      f"Size: {video_info['width']}x{video_info['height']}"
            
            return (frames_comfy, info_str, video_info['total_frames'], video_info['fps'])
            
        except Exception as e:
            raise RuntimeError(f"Failed to load video: {str(e)}")

class RajVideoLoaderPath:
    """
    Load video from external path with GPU acceleration
    Similar to RajVideoLoader but with file browser support
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_file": ("STRING", {
                    "default": "/path/to/video.mp4",
                    "tooltip": "Full path to video file"
                }),
                "target_fps": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 120.0, 
                    "step": 0.1,
                    "tooltip": "Target frame rate (0 = keep original)"
                }),
                "max_frames": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 10000, 
                    "step": 1,
                    "tooltip": "Maximum frames to load (0 = load all)"
                }),
            },
            "optional": {
                "target_width": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 4096, 
                    "step": 8,
                    "tooltip": "Target width (0 = keep original)"
                }),
                "target_height": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 4096, 
                    "step": 8,
                    "tooltip": "Target height (0 = keep original)"
                }),
                "force_device": (["auto", "cpu", "cuda", "mps"], {
                    "default": "auto",
                    "tooltip": "Force specific device (auto = best available)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("frames", "video_info", "frame_count", "fps")
    FUNCTION = "load_video_path"
    CATEGORY = "Raj Video Processing 🎬"
    
    def load_video_path(self, video_file, target_fps=0.0, max_frames=0, 
                       target_width=0, target_height=0, force_device="auto"):
        
        # Determine device
        if force_device == "auto":
            device = get_optimal_device()
        else:
            device = torch.device(force_device)
        
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video file not found: {video_file}")
        
        # Determine target size (same logic as RajVideoLoader)
        target_size = None
        if target_width > 0 and target_height > 0:
            target_size = (target_width, target_height)
        elif target_width > 0:
            import cv2
            cap = cv2.VideoCapture(video_file)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            aspect_ratio = original_height / original_width
            target_size = (target_width, int(target_width * aspect_ratio))
        elif target_height > 0:
            import cv2
            cap = cv2.VideoCapture(video_file)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            aspect_ratio = original_width / original_height
            target_size = (int(target_height * aspect_ratio), target_height)
        
        # Load video
        try:
            frames_tensor, video_info = video_to_tensor(
                video_file, 
                device, 
                target_fps, 
                max_frames, 
                target_size
            )
            
            # Convert to ComfyUI format
            frames_comfy = tensor_to_video_frames(frames_tensor)
            
            # Create info string
            info_str = f"Device: {video_info['device']} | " \
                      f"Frames: {video_info['total_frames']} | " \
                      f"FPS: {video_info['fps']:.2f} | " \
                      f"Size: {video_info['width']}x{video_info['height']}"
            
            return (frames_comfy, info_str, video_info['total_frames'], video_info['fps'])
            
        except Exception as e:
            raise RuntimeError(f"Failed to load video: {str(e)}")