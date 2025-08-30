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

def video_to_tensor_chunked(video_path: str, 
                           device: torch.device,
                           target_fps: float = 0,
                           max_frames: int = 0,
                           target_size: Tuple[int, int] = None,
                           chunk_size: int = 50) -> Tuple[torch.Tensor, dict]:
    """
    Load video file with chunked processing to handle memory constraints
    
    Args:
        video_path: Path to video file
        device: Target device (mps/cuda/cpu)
        target_fps: Target frame rate (0 = original)
        max_frames: Maximum frames to load (0 = all)
        target_size: (width, height) to resize to (None = original)
        chunk_size: Number of frames to process at once
    
    Returns:
        (frames_tensor, video_info)
    """
    import psutil
    import gc
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Fix FPS conversion for near-identical framerates
    if target_fps > 0 and original_fps > 0:
        fps_ratio = original_fps / target_fps
        # If framerates are very close (within 1%), treat as identical
        if abs(fps_ratio - 1.0) < 0.01:
            logger.info(f"⚡ FPS rates very close ({original_fps:.2f} → {target_fps:.2f}), using 1:1 mapping")
            frame_skip = 1
            need_frame_interpolation = False
        else:
            frame_skip = max(1, int(fps_ratio))
            need_frame_interpolation = (fps_ratio < 1.0)  # Need to duplicate frames
    else:
        frame_skip = 1
        target_fps = original_fps
        need_frame_interpolation = False
    
    # Determine final dimensions
    if target_size is not None:
        final_width, final_height = target_size
    else:
        final_width, final_height = width, height
    
    # Calculate memory requirements
    frame_memory = final_width * final_height * 3 * 4  # RGB float32
    estimated_memory = (total_frames // frame_skip) * frame_memory
    available_memory = psutil.virtual_memory().available
    
    # Adjust chunk size based on memory
    if estimated_memory > available_memory * 0.3:  # Using more than 30% of RAM
        chunk_size = min(chunk_size, max(10, int(available_memory * 0.1 / frame_memory)))
        logger.info(f"⚠️ Large video detected, reducing chunk size to {chunk_size} frames")
    
    logger.info(f"📹 Loading video (chunked): {video_path}")
    logger.info(f"   Original: {width}x{height} @ {original_fps:.2f}fps, {total_frames} frames")
    logger.info(f"   Target: {final_width}x{final_height} @ {target_fps:.2f}fps")
    logger.info(f"   Memory: {estimated_memory/(1024**3):.2f}GB estimated, chunk size: {chunk_size}")
    if need_frame_interpolation:
        logger.info(f"   🔄 Frame interpolation enabled ({original_fps:.2f} → {target_fps:.2f})")
    
    # Process video in chunks
    all_frame_chunks = []
    frames_buffer = []
    frame_count = 0
    read_count = 0
    
    # Determine optimal processing device
    original_device = device
    use_cpu_for_large = (device.type == "mps" and estimated_memory > 500_000_000)  # 500MB threshold
    processing_device = torch.device("cpu") if use_cpu_for_large else device
    
    if use_cpu_for_large:
        logger.info(f"⚠️ Using CPU processing due to large video size")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames based on target FPS
        should_include_frame = True
        if frame_skip > 1 and read_count % frame_skip != 0:
            should_include_frame = False
        
        if should_include_frame:
            # Resize if needed
            if target_size is not None:
                frame = cv2.resize(frame, target_size)
            
            # Convert BGR to RGB and normalize to [0,1]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            
            # Add frame interpolation for upsampling
            if need_frame_interpolation and frame_count > 0:
                # Simple frame duplication for FPS upsampling
                interp_count = int(target_fps / original_fps) - 1
                for _ in range(interp_count):
                    frames_buffer.append(frame_normalized.copy())
                    frame_count += 1
                    if len(frames_buffer) >= chunk_size:
                        all_frame_chunks.append(process_frame_chunk(frames_buffer, processing_device))
                        frames_buffer = []
                        gc.collect()  # Force garbage collection
            
            frames_buffer.append(frame_normalized)
            frame_count += 1
            
            # Process chunk when buffer is full
            if len(frames_buffer) >= chunk_size:
                chunk_tensor = process_frame_chunk(frames_buffer, processing_device)
                all_frame_chunks.append(chunk_tensor)
                frames_buffer = []
                
                # Clear memory and caches
                gc.collect()
                if processing_device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                elif processing_device.type == "cuda":
                    torch.cuda.empty_cache()
                
                # Monitor memory usage
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 85:
                    logger.warning(f"⚠️ High memory usage ({memory_percent:.1f}%), consider reducing chunk size")
        
        read_count += 1
        
        # Check max frames limit
        if max_frames > 0 and frame_count >= max_frames:
            break
    
    # Process remaining frames
    if frames_buffer:
        chunk_tensor = process_frame_chunk(frames_buffer, processing_device)
        all_frame_chunks.append(chunk_tensor)
    
    cap.release()
    
    if not all_frame_chunks:
        raise ValueError("No frames loaded from video")
    
    # Concatenate all chunks
    logger.info(f"🔗 Concatenating {len(all_frame_chunks)} chunks...")
    frames_tensor = torch.cat(all_frame_chunks, dim=0)
    
    # Final integrity check
    tensor_max = torch.max(frames_tensor).item()
    tensor_min = torch.min(frames_tensor).item()
    logger.info(f"✅ Final tensor stats: min={tensor_min:.3f}, max={tensor_max:.3f}")
    
    if tensor_max < 0.001:
        raise ValueError(f"Final tensor corrupted (max: {tensor_max:.6f})")
    
    # Try to move to original device if we used CPU for processing
    if processing_device.type == "cpu" and original_device.type == "mps":
        try:
            # Clear caches
            gc.collect()
            torch.mps.empty_cache()
            
            # Test with small chunk first
            test_chunk = frames_tensor[:min(10, frames_tensor.shape[0])].to(original_device)
            if torch.max(test_chunk).item() > 0.001:
                frames_tensor = frames_tensor.to(original_device)
                processing_device = original_device
                logger.info(f"✅ Successfully moved to {original_device}")
            else:
                logger.warning(f"⚠️ MPS corruption detected, staying on CPU")
        except Exception as e:
            logger.warning(f"⚠️ Could not move to MPS: {e}")
    
    video_info = {
        'fps': target_fps,
        'total_frames': frame_count,
        'width': final_width,
        'height': final_height,
        'original_fps': original_fps,
        'original_total_frames': total_frames,
        'device': str(processing_device),
        'chunks_processed': len(all_frame_chunks),
        'memory_optimized': use_cpu_for_large
    }
    
    logger.info(f"✅ Loaded {frame_count} frames on {processing_device} ({len(all_frame_chunks)} chunks)")
    return frames_tensor, video_info

def process_frame_chunk(frames_list: list, device: torch.device) -> torch.Tensor:
    """Process a chunk of frames with corruption detection"""
    if not frames_list:
        return torch.empty((0, 0, 0, 3), device=device)
    
    # Convert to numpy array
    frames_array = np.stack(frames_list, axis=0)
    
    # Check numpy array integrity
    array_max = np.max(frames_array)
    if array_max < 0.001:
        logger.warning(f"⚠️ Chunk has corrupted numpy data (max: {array_max:.6f})")
        return torch.zeros((len(frames_list), frames_list[0].shape[0], frames_list[0].shape[1], 3), device=device)
    
    # Convert to tensor
    try:
        chunk_tensor = torch.from_numpy(frames_array).to(device)
        
        # Verify tensor integrity
        tensor_max = torch.max(chunk_tensor).item()
        if tensor_max < 0.001:
            logger.warning(f"⚠️ Tensor corrupted after creation (max: {tensor_max:.6f})")
            if device.type != "cpu":
                # Fallback to CPU for this chunk
                chunk_tensor = torch.from_numpy(frames_array).to(torch.device("cpu"))
                logger.info(f"✅ Chunk fallback to CPU successful")
        
        return chunk_tensor
        
    except Exception as e:
        logger.error(f"❌ Chunk processing failed: {e}")
        # Return CPU tensor as fallback
        return torch.from_numpy(frames_array).to(torch.device("cpu"))

def video_to_tensor(video_path: str, 
                   device: torch.device,
                   target_fps: float = 0,
                   max_frames: int = 0,
                   target_size: Tuple[int, int] = None) -> Tuple[torch.Tensor, dict]:
    """
    Load video file and convert to PyTorch tensor with automatic chunking for large videos
    
    This is a wrapper that automatically chooses between standard and chunked loading
    based on video size and available memory.
    """
    import psutil
    
    # Quick check of video properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Estimate memory requirements
    frame_size = width * height * 3 * 4  # RGB float32
    estimated_frames = total_frames if max_frames <= 0 else min(max_frames, total_frames)
    estimated_memory = estimated_frames * frame_size
    available_memory = psutil.virtual_memory().available
    
    # Use chunked processing for large videos or low memory situations
    use_chunked = (
        estimated_memory > available_memory * 0.2 or  # More than 20% of available RAM
        total_frames > 200 or  # Large frame count
        (device.type == "mps" and estimated_memory > 100_000_000) or  # 100MB threshold for MPS
        psutil.virtual_memory().percent > 70  # High memory usage already
    )
    
    if use_chunked:
        logger.info(f"📊 Using chunked processing (est. {estimated_memory/(1024**3):.2f}GB, {available_memory/(1024**3):.2f}GB available)")
        return video_to_tensor_chunked(video_path, device, target_fps, max_frames, target_size)
    else:
        logger.info(f"📊 Using standard processing (est. {estimated_memory/(1024**2):.1f}MB)")
        return video_to_tensor_standard(video_path, device, target_fps, max_frames, target_size)

def video_to_tensor_standard(video_path: str, 
                           device: torch.device,
                           target_fps: float = 0,
                           max_frames: int = 0,
                           target_size: Tuple[int, int] = None) -> Tuple[torch.Tensor, dict]:
    """Standard video loading for smaller videos (legacy function)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frame sampling with improved FPS handling
    if target_fps > 0 and original_fps > 0:
        fps_ratio = original_fps / target_fps
        if abs(fps_ratio - 1.0) < 0.01:  # Very close framerates
            frame_skip = 1
            logger.info(f"⚡ FPS rates very close ({original_fps:.2f} → {target_fps:.2f}), using 1:1 mapping")
        else:
            frame_skip = max(1, int(fps_ratio))
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
    
    # Convert to tensor with corruption detection
    frames_array = np.stack(frames, axis=0)
    array_max = np.max(frames_array)
    
    if array_max < 0.001:
        raise ValueError(f"Video frames are corrupted (max: {array_max:.6f})")
    
    # Process on CPU first for stability, then move to target device
    frames_tensor = torch.from_numpy(frames_array).to(torch.device("cpu"))
    
    # Move to target device if different
    if device.type != "cpu":
        try:
            frames_tensor = frames_tensor.to(device)
            # Verify after move
            if torch.max(frames_tensor).item() < 0.001:
                logger.warning(f"⚠️ Corruption after moving to {device}, staying on CPU")
                frames_tensor = frames_tensor.to(torch.device("cpu"))
                device = torch.device("cpu")
        except Exception as e:
            logger.warning(f"⚠️ Could not move to {device}: {e}, using CPU")
            device = torch.device("cpu")
    
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
    
    # Check input integrity first
    input_max = torch.max(tensor).item()
    input_min = torch.min(tensor).item()
    
    if input_max < 0.001:
        logger.warning(f"⚠️ tensor_to_video_frames received corrupted input (max: {input_max:.6f})")
        return tensor  # Return as-is to avoid further corruption
    
    # Ensure values are in [0,1] range with corruption detection
    # NOTE: Even out-of-place operations on MPS can corrupt large tensors
    if tensor.device.type == "mps":
        # For MPS, avoid clamp operations entirely for large tensors to prevent corruption
        tensor_size = tensor.numel() * 4  # float32 = 4 bytes
        if tensor_size > 100_000_000:  # 100MB threshold
            logger.info(f"⚠️ Large MPS tensor ({tensor_size/(1024**2):.1f}MB), skipping clamp to prevent corruption")
            # Only clamp if values are actually outside [0,1] range
            if input_min < -0.001 or input_max > 1.001:
                logger.info(f"   Values outside [0,1] range ({input_min:.3f}, {input_max:.3f}), applying careful clamp")
                # Move to CPU, clamp, then move back
                device = tensor.device
                tensor_cpu = tensor.cpu()
                tensor_clamped = torch.clamp(tensor_cpu, 0.0, 1.0)
                tensor = tensor_clamped.to(device)
                
                # Verify clamp didn't corrupt
                if torch.max(tensor).item() < 0.001:
                    logger.warning(f"⚠️ Clamp operation corrupted tensor, reverting to original")
                    tensor = tensor_cpu.to(device)  # Use unclamped version
        else:
            # Small tensor, safe to clamp on MPS
            tensor_clamped = torch.clamp(tensor, 0.0, 1.0)
            
            # Verify clamp didn't corrupt
            if torch.max(tensor_clamped).item() < 0.001:
                logger.warning(f"⚠️ MPS clamp corrupted tensor, keeping original values")
            else:
                tensor = tensor_clamped
    else:
        # Use in-place operation for CPU/CUDA to save memory
        tensor.clamp_(0.0, 1.0)
    
    # Final integrity check
    output_max = torch.max(tensor).item()
    if output_max < 0.001 and input_max > 0.01:
        logger.error(f"❌ tensor_to_video_frames corrupted tensor: {input_max:.3f} → {output_max:.6f}")
    
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

# ============================================================================
# File naming utilities
# ============================================================================

def get_save_path_incremental(filename: str, directory: str, extension: str) -> str:
    """
    Get a save path with auto-increment if file exists.
    Similar to VideoHelperSuite behavior: filename_00001.ext, filename_00002.ext, etc.
    
    Args:
        filename: Base filename without extension
        directory: Directory to save in
        extension: File extension (without dot)
    
    Returns:
        Full path with incremented filename if needed
    """
    import os
    import re
    
    # Clean the filename
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Build base path
    base_path = os.path.join(directory, f"{filename}.{extension}")
    
    # If doesn't exist, return as is
    if not os.path.exists(base_path):
        return base_path
    
    # Find the next available number
    counter = 1
    while True:
        numbered_filename = f"{filename}_{counter:05d}"
        numbered_path = os.path.join(directory, f"{numbered_filename}.{extension}")
        
        if not os.path.exists(numbered_path):
            return numbered_path
        
        counter += 1
        
        # Safety check to prevent infinite loop
        if counter > 99999:
            raise RuntimeError(f"Could not find available filename after 99999 attempts")

# ============================================================================
# Time and frame utilities
# ============================================================================

def time_to_frame(time_seconds: float, fps: float = 24.0) -> int:
    """Convert time in seconds to frame number at given FPS"""
    return int(time_seconds * fps)

def frame_to_time(frame_num: int, fps: float = 24.0) -> float:
    """Convert frame number to time in seconds at given FPS"""
    return frame_num / fps

def parse_time_points(time_string: str, fps: float = 24.0) -> List[int]:
    """
    Parse comma-separated time points and convert to frame indices
    Example: "2.5, 5.0, 8.5" -> [60, 120, 204] at 24fps
    """
    if not time_string or time_string.strip() == "":
        return []
    
    times = [float(t.strip()) for t in time_string.split(",")]
    frames = [time_to_frame(t, fps) for t in times]
    return frames

# ============================================================================
# Easing functions for animations
# ============================================================================

def linear(t: float) -> float:
    """Linear interpolation (no easing)"""
    return t

def ease_in_quad(t: float) -> float:
    """Quadratic ease-in (slow start)"""
    return t * t

def ease_out_quad(t: float) -> float:
    """Quadratic ease-out (slow end)"""
    return 1 - (1 - t) ** 2

def ease_in_out_quad(t: float) -> float:
    """Quadratic ease-in-out (slow start and end)"""
    if t < 0.5:
        return 2 * t * t
    return 1 - pow(-2 * t + 2, 2) / 2

def ease_out_in_quad(t: float) -> float:
    """Quadratic ease-out-in (fast middle)"""
    if t < 0.5:
        return ease_out_quad(t * 2) * 0.5
    return ease_in_quad((t - 0.5) * 2) * 0.5 + 0.5

def apply_easing(value: float, start: float, end: float, t: float, easing: str = "linear") -> float:
    """
    Apply easing function to interpolate between start and end values
    
    Args:
        value: Current value (unused, kept for compatibility)
        start: Start value
        end: End value
        t: Normalized time (0.0 to 1.0)
        easing: Easing function name
    
    Returns:
        Interpolated value with easing applied
    """
    easing_functions = {
        "linear": linear,
        "ease_in": ease_in_quad,
        "ease_out": ease_out_quad,
        "ease_in_out": ease_in_out_quad,
        "ease_out_in": ease_out_in_quad,
        "constant": lambda x: 0 if x < 1.0 else 1
    }
    
    ease_func = easing_functions.get(easing, linear)
    eased_t = ease_func(t)
    return start + (end - start) * eased_t

# ============================================================================
# Aspect ratio utilities
# ============================================================================

def calculate_aspect_ratio(width: int, height: int) -> Tuple[int, int]:
    """Calculate simplified aspect ratio"""
    import math
    gcd = math.gcd(width, height)
    return width // gcd, height // gcd

def resize_with_padding(frame: np.ndarray, target_width: int, target_height: int, 
                       pad_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Resize frame maintaining aspect ratio and add padding if needed
    """
    h, w = frame.shape[:2]
    aspect = w / h
    target_aspect = target_width / target_height
    
    if aspect > target_aspect:
        # Frame is wider - fit to width
        new_width = target_width
        new_height = int(target_width / aspect)
    else:
        # Frame is taller - fit to height
        new_height = target_height
        new_width = int(target_height * aspect)
    
    # Resize
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Create padded frame
    padded = np.full((target_height, target_width, 3), pad_color, dtype=frame.dtype)
    
    # Calculate padding
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    # Place resized frame in center
    padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return padded

def resize_with_crop(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resize frame maintaining aspect ratio and crop if needed (center crop)
    """
    h, w = frame.shape[:2]
    aspect = w / h
    target_aspect = target_width / target_height
    
    if aspect > target_aspect:
        # Frame is wider - fit to height and crop width
        new_height = target_height
        new_width = int(target_height * aspect)
    else:
        # Frame is taller - fit to width and crop height
        new_width = target_width
        new_height = int(target_width / aspect)
    
    # Resize
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Center crop
    y_start = (new_height - target_height) // 2
    x_start = (new_width - target_width) // 2
    
    cropped = resized[y_start:y_start+target_height, x_start:x_start+target_width]
    
    return cropped