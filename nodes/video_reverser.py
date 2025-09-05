import torch
import numpy as np
from .utils import logger

class RajVideoReverser:
    """
    Simple video reverser node that reverses the order of video frames
    Memory-efficient processing with device-aware chunking
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Input video frames to reverse"
                }),
            },
            "optional": {
                "chunk_size": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 500,
                    "step": 1,
                    "tooltip": "Number of frames to process at once (memory management)"
                }),
                "reverse_type": (["full_reverse", "time_reverse"], {
                    "default": "full_reverse",
                    "tooltip": "full_reverse: flip all frames, time_reverse: reverse temporal order"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("reversed_frames", "frame_count", "reverse_info")
    FUNCTION = "reverse_video"
    CATEGORY = "Raj Video Processing 🎬"
    
    def reverse_video(self, frames, chunk_size=50, reverse_type="full_reverse"):
        """
        Reverse video frames efficiently
        """
        device = frames.device
        dtype = frames.dtype
        original_frame_count = frames.shape[0]
        height = frames.shape[1]
        width = frames.shape[2]
        channels = frames.shape[3]
        
        logger.info(f"🔄 Video Reverser Processing")
        logger.info(f"   Input: {original_frame_count} frames, {width}x{height}, {channels} channels")
        logger.info(f"   Device: {device}, Type: {reverse_type}")
        logger.info(f"   Chunk size: {chunk_size}")
        
        if original_frame_count == 0:
            logger.warning("   No frames to reverse")
            reverse_info = "No frames provided"
            return (frames, 0, reverse_info)
        
        # Calculate optimal chunk size based on memory and device
        if device.type == "mps":
            # More conservative chunking for MPS
            total_pixels = width * height * channels
            if total_pixels > 1280 * 720 * 3:
                chunk_size = min(chunk_size, 20)
            elif total_pixels > 640 * 480 * 3:
                chunk_size = min(chunk_size, 40)
        elif device.type == "cuda":
            # CUDA can handle larger chunks
            chunk_size = min(chunk_size, 100)
        else:
            # CPU processing
            chunk_size = min(chunk_size, 30)
        
        if reverse_type == "full_reverse":
            # Simply reverse the frame order using PyTorch's flip
            logger.info(f"   Using torch.flip for full frame reversal")
            reversed_frames = torch.flip(frames, [0])
            
        else:  # time_reverse - more memory efficient for very large videos
            logger.info(f"   Using chunked time reversal processing")
            
            # Process in chunks and reverse order
            reversed_chunks = []
            total_chunks = (original_frame_count + chunk_size - 1) // chunk_size
            
            # Process chunks in reverse order
            for chunk_idx in range(total_chunks - 1, -1, -1):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, original_frame_count)
                
                logger.info(f"   Processing chunk {total_chunks - chunk_idx}/{total_chunks}: frames {start_idx}-{end_idx-1}")
                
                # Get chunk and reverse it internally
                chunk = frames[start_idx:end_idx]
                reversed_chunk = torch.flip(chunk, [0])
                reversed_chunks.append(reversed_chunk)
                
                # Memory cleanup
                if device.type == "mps" and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
            
            # Concatenate all reversed chunks
            logger.info(f"🔗 Concatenating {len(reversed_chunks)} reversed chunks...")
            reversed_frames = torch.cat(reversed_chunks, dim=0)
        
        # Verify the reversal worked
        final_frame_count = reversed_frames.shape[0]
        if final_frame_count != original_frame_count:
            logger.error(f"   Frame count mismatch: {original_frame_count} -> {final_frame_count}")
        
        # Create info string
        reverse_info = f"Video Reversed: {reverse_type} | " \
                      f"Frames: {original_frame_count} -> {final_frame_count} | " \
                      f"Size: {width}x{height} | Device: {device}"
        
        logger.info(f"✅ Video reversal complete: {final_frame_count} frames")
        logger.info(f"   Dimensions: {reversed_frames.shape}")
        
        return (reversed_frames, final_frame_count, reverse_info)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always recompute when input changes
        return float("nan")


# Test the node
if __name__ == "__main__":
    # Create test frames
    test_frames = torch.randn(10, 64, 64, 3)
    
    node = RajVideoReverser()
    result = node.reverse_video(test_frames)
    
    print(f"Original shape: {test_frames.shape}")
    print(f"Reversed shape: {result[0].shape}")
    print(f"Info: {result[2]}")
    
    # Verify reversal - first frame should become last frame
    print(f"First frame matches last reversed: {torch.allclose(test_frames[0], result[0][-1])}")
    print(f"Last frame matches first reversed: {torch.allclose(test_frames[-1], result[0][0])}")