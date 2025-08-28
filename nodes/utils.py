import torch
import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_optimal_device() -> torch.device:
    """
    Auto-detect the best available device across platforms
    Priority: MPS (Mac) > CUDA (NVIDIA) > CPU (Fallback)
    """
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info(f"🍎 Using MPS (Metal Performance Shaders) on Mac")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info(f"💻 Using CPU (no GPU acceleration available)")
    
    return device

def video_to_tensor(video_path: str, 
                   device: torch.device,
                   target_fps: float = 0,
                   max_frames: int = 0,
                   target_size: Tuple[int, int] = None) -> Tuple[torch.Tensor, dict]:
    """
    Load video file and convert to PyTorch tensor with GPU support
    
    Args:
        video_path: Path to video file
        device: Target device (mps/cuda/cpu)
        target_fps: Target frame rate (0 = original)
        max_frames: Maximum frames to load (0 = all)
        target_size: (width, height) to resize to (None = original)
    
    Returns:
        (frames_tensor, video_info)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frame sampling
    if target_fps > 0 and original_fps > 0:
        frame_skip = max(1, int(original_fps / target_fps))
    else:
        frame_skip = 1
        target_fps = original_fps
    
    # Determine final dimensions
    if target_size is not None:
        final_width, final_height = target_size
    else:
        final_width, final_height = width, height
    
    frames = []
    frame_count = 0
    read_count = 0
    
    logger.info(f"📹 Loading video: {video_path}")
    logger.info(f"   Original: {width}x{height} @ {original_fps:.2f}fps, {total_frames} frames")
    logger.info(f"   Target: {final_width}x{final_height} @ {target_fps:.2f}fps")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames based on target FPS
        if read_count % frame_skip != 0:
            read_count += 1
            continue
        
        # Resize if needed
        if target_size is not None:
            frame = cv2.resize(frame, target_size)
        
        # Convert BGR to RGB and normalize to [0,1]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        frames.append(frame_normalized)
        frame_count += 1
        read_count += 1
        
        # Check max frames limit
        if max_frames > 0 and frame_count >= max_frames:
            break
    
    cap.release()
    
    if not frames:
        raise ValueError("No frames loaded from video")
    
    # Convert to tensor and move to device
    # Shape: [batch, height, width, channels]
    frames_array = np.stack(frames, axis=0)
    frames_tensor = torch.from_numpy(frames_array).to(device)
    
    video_info = {
        'fps': target_fps,
        'total_frames': frame_count,
        'width': final_width,
        'height': final_height,
        'original_fps': original_fps,
        'original_total_frames': total_frames,
        'device': str(device)
    }
    
    logger.info(f"✅ Loaded {frame_count} frames on {device}")
    return frames_tensor, video_info

def tensor_to_video_frames(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is in correct format for ComfyUI
    ComfyUI expects: [batch, height, width, channels] with values in [0,1]
    """
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {tensor.dim()}D")
    
    # Ensure values are in [0,1] range - use in-place operation to save memory
    tensor.clamp_(0.0, 1.0)
    
    return tensor

def get_memory_info(device: torch.device) -> dict:
    """Get memory information for the device"""
    if device.type == "cuda":
        return {
            'total': torch.cuda.get_device_properties(device).total_memory,
            'allocated': torch.cuda.memory_allocated(device),
            'free': torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
        }
    elif device.type == "mps":
        return {
            'total': torch.mps.driver_allocated_memory() if hasattr(torch.mps, 'driver_allocated_memory') else 0,
            'allocated': torch.mps.current_allocated_memory() if hasattr(torch.mps, 'current_allocated_memory') else 0,
            'free': 'unknown'
        }
    else:
        return {'total': 'unknown', 'allocated': 'unknown', 'free': 'unknown'}

def optimize_batch_size(device: torch.device, frame_size: int, target_memory_usage: float = 0.8) -> int:
    """
    Calculate optimal batch size based on available GPU memory
    """
    if device.type == "cpu":
        return 32  # Conservative default for CPU
    
    try:
        memory_info = get_memory_info(device)
        if memory_info['free'] == 'unknown':
            return 16  # Safe default
        
        # Estimate memory per frame (rough calculation)
        bytes_per_frame = frame_size * 4  # float32 = 4 bytes
        available_memory = memory_info['free'] * target_memory_usage
        
        batch_size = max(1, int(available_memory / bytes_per_frame))
        return min(batch_size, 64)  # Cap at reasonable maximum
        
    except Exception as e:
        logger.warning(f"Could not calculate optimal batch size: {e}")
        return 16  # Safe fallback

def clear_gpu_cache(device: torch.device):
    """Clear GPU cache to free memory"""
    if device.type == "mps":
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
            logger.debug("🧹 Cleared MPS cache")
    elif device.type == "cuda":
        torch.cuda.empty_cache()
        logger.debug("🧹 Cleared CUDA cache")

def estimate_tensor_memory(frames: int, height: int, width: int, channels: int = 3) -> int:
    """
    Estimate memory usage for a tensor in bytes
    Args:
        frames: Number of frames
        height: Frame height  
        width: Frame width
        channels: Number of channels (default 3 for RGB)
    Returns:
        Estimated memory in bytes
    """
    # float32 = 4 bytes per value
    return frames * height * width * channels * 4

def should_use_cpu_fallback(device: torch.device, estimated_memory: int, safety_factor: float = 0.7) -> bool:
    """
    Check if we should fallback to CPU based on memory requirements
    """
    if device.type == "cpu":
        return False
        
    memory_info = get_memory_info(device)
    
    if device.type == "mps":
        # MPS has a hard limit around 9GB
        max_memory = 9 * 1024 * 1024 * 1024  # 9GB in bytes
        available = max_memory - (memory_info.get('allocated', 0) or 0)
    elif device.type == "cuda":
        available = memory_info.get('free', float('inf'))
    else:
        return False
    
    # Check if estimated memory exceeds available (with safety margin)
    if estimated_memory > available * safety_factor:
        logger.warning(f"⚠️ Estimated memory ({estimated_memory / 1024**3:.2f}GB) exceeds available ({available / 1024**3:.2f}GB)")
        logger.info("🔄 Switching to CPU to avoid out of memory error")
        return True
    
    return False