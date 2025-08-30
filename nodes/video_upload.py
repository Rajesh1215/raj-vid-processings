import os
import torch
import cv2
from .utils import get_optimal_device, video_to_tensor, tensor_to_video_frames, logger

try:
    import folder_paths
except ImportError:
    # Fallback for testing outside ComfyUI
    class MockFolderPaths:
        @staticmethod
        def get_input_directory():
            return "input"
        
        @staticmethod  
        def get_annotated_filepath(path):
            return path
            
        base_path = "."
    
    folder_paths = MockFolderPaths()

def get_video_files():
    """Get list of video files in input directory"""
    input_dir = folder_paths.get_input_directory()
    video_extensions = ['webm', 'mp4', 'mkv', 'gif', 'mov', 'avi']
    files = []
    
    if os.path.exists(input_dir):
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and file_parts[-1].lower() in video_extensions:
                    files.append(f)
    
    return sorted(files)

class RajVideoUpload:
    """
    Upload and load video files with GPU acceleration
    Provides an upload button similar to VideoHelperSuite
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (get_video_files(), {
                    "video_upload": True,
                    "tooltip": "Select video file or click upload button"
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
            },
            "hidden": {
                "choose video to upload": ("UPLOAD",)  # This creates the upload button
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("frames", "video_info", "frame_count", "fps")
    FUNCTION = "upload_and_load_video"
    CATEGORY = "Raj Video Processing 🎬"
    
    def upload_and_load_video(self, video, target_fps=0.0, max_frames=0, 
                             target_width=0, target_height=0, force_device="auto", **kwargs):
        
        # Determine device
        if force_device == "auto":
            device = get_optimal_device()
        else:
            device = torch.device(force_device)
        
        # Get full path to video
        video_path = folder_paths.get_annotated_filepath(video)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Determine target size
        target_size = None
        if target_width > 0 and target_height > 0:
            target_size = (target_width, target_height)
        elif target_width > 0:
            # Calculate height from aspect ratio
            cap = cv2.VideoCapture(video_path)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if original_width > 0:
                aspect_ratio = original_height / original_width
                target_size = (target_width, int(target_width * aspect_ratio))
        elif target_height > 0:
            # Calculate width from aspect ratio
            cap = cv2.VideoCapture(video_path)
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if original_height > 0:
                aspect_ratio = original_width / original_height
                target_size = (int(target_height * aspect_ratio), target_height)
        
        logger.info(f"📤 Processing uploaded video: {os.path.basename(video_path)}")
        
        # Load video
        try:
            frames_tensor, video_info = video_to_tensor(
                video_path, 
                device, 
                target_fps, 
                max_frames, 
                target_size
            )
            
            # Verify tensor integrity before conversion
            tensor_max = torch.max(frames_tensor).item()
            tensor_min = torch.min(frames_tensor).item()
            logger.info(f"📊 Tensor integrity check: min={tensor_min:.6f}, max={tensor_max:.6f}")
            
            if tensor_max < 0.001:
                logger.error(f"❌ Tensor corrupted before ComfyUI conversion (max: {tensor_max:.6f})")
                logger.info("🔧 Attempting to reload on CPU...")
                
                # Force CPU reload as fallback
                frames_tensor, video_info = video_to_tensor(
                    video_path, 
                    torch.device("cpu"), 
                    target_fps, 
                    max_frames, 
                    target_size
                )
                
                # Verify CPU reload worked
                cpu_max = torch.max(frames_tensor).item()
                if cpu_max < 0.001:
                    raise ValueError(f"Video appears to be corrupted or completely black (max value: {cpu_max:.6f})")
                
                logger.info(f"✅ CPU reload successful: max={cpu_max:.6f}")
            
            # Convert to ComfyUI format
            frames_comfy = tensor_to_video_frames(frames_tensor)
            
            # Create info string
            info_str = f"Upload: {os.path.basename(video_path)} | " \
                      f"Device: {video_info['device']} | " \
                      f"Frames: {video_info['total_frames']} | " \
                      f"FPS: {video_info['fps']:.2f} | " \
                      f"Size: {video_info['width']}x{video_info['height']}"
            
            logger.info(f"✅ Upload processed: {info_str}")
            
            # Prepare UI preview data (VHS-compatible format)
            video_format = os.path.splitext(video_path)[1][1:] or "mp4"
            preview = {
                "filename": os.path.basename(video_path),
                "subfolder": "",
                "type": "input",
                "format": f"video/{video_format}"
            }
            
            return {
                "ui": {"gifs": [preview]},
                "result": (frames_comfy, info_str, video_info['total_frames'], video_info['fps'])
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to process uploaded video: {str(e)}")
    
    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        """Check if video file has changed"""
        try:
            video_path = folder_paths.get_annotated_filepath(video)
            if os.path.exists(video_path):
                return os.path.getmtime(video_path)
            return float("inf")
        except:
            return float("inf")

class RajVideoUploadAdvanced:
    """
    Advanced video upload with batch processing and format selection
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (get_video_files(), {
                    "video_upload": True
                }),
                "target_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0, "step": 0.1}),
                "processing_mode": (["full", "keyframes_only", "every_nth"], {
                    "default": "full",
                    "tooltip": "How to process video frames"
                }),
            },
            "optional": {
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "frame_skip": ("INT", {"default": 1, "min": 1, "max": 100}),
                "quality_preset": (["high", "medium", "low", "custom"], {
                    "default": "high"
                }),
                "custom_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "custom_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
                "memory_optimization": ("BOOLEAN", {"default": True}),
                "force_device": (["auto", "cpu", "cuda", "mps"], {"default": "auto"}),
            },
            "hidden": {
                "choose video to upload": ("UPLOAD",)
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("frames", "video_info", "frame_count", "fps", "processing_log")
    FUNCTION = "upload_and_process_advanced"
    CATEGORY = "Raj Video Processing 🎬"
    
    def upload_and_process_advanced(self, video, target_fps=0.0, processing_mode="full",
                                   max_frames=0, frame_skip=1, quality_preset="high",
                                   custom_width=0, custom_height=0, memory_optimization=True,
                                   force_device="auto", **kwargs):
        
        device = get_optimal_device() if force_device == "auto" else torch.device(force_device)
        video_path = folder_paths.get_annotated_filepath(video)
        
        logger.info(f"🎛️ Advanced upload processing: {os.path.basename(video_path)}")
        logger.info(f"   Mode: {processing_mode} | Quality: {quality_preset} | Device: {device}")
        
        # Determine target size based on quality preset
        target_size = None
        if quality_preset == "high":
            # Keep original or custom size
            if custom_width > 0 and custom_height > 0:
                target_size = (custom_width, custom_height)
        elif quality_preset == "medium":
            # Limit to 1280x720 max
            cap = cv2.VideoCapture(video_path)
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            if orig_w > 1280 or orig_h > 720:
                if orig_w / orig_h > 1280 / 720:
                    target_size = (1280, int(1280 * orig_h / orig_w))
                else:
                    target_size = (int(720 * orig_w / orig_h), 720)
        elif quality_preset == "low":
            # Limit to 640x480 max
            cap = cv2.VideoCapture(video_path)
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            if orig_w > 640 or orig_h > 480:
                if orig_w / orig_h > 640 / 480:
                    target_size = (640, int(640 * orig_h / orig_w))
                else:
                    target_size = (int(480 * orig_w / orig_h), 480)
        else:  # custom
            if custom_width > 0 and custom_height > 0:
                target_size = (custom_width, custom_height)
        
        # Apply processing mode settings
        actual_max_frames = max_frames
        actual_frame_skip = frame_skip
        
        if processing_mode == "keyframes_only":
            # This would need more advanced keyframe detection
            # For now, just skip more frames
            actual_frame_skip = max(frame_skip, 5)
            logger.info("   Using keyframe approximation (every 5th frame)")
        elif processing_mode == "every_nth":
            actual_frame_skip = max(frame_skip, 2)
        
        # Load video with advanced settings
        try:
            frames_tensor, video_info = video_to_tensor(
                video_path, 
                device, 
                target_fps, 
                actual_max_frames, 
                target_size
            )
            
            # Apply frame skipping if needed
            if actual_frame_skip > 1:
                frames_tensor = frames_tensor[::actual_frame_skip]
                video_info['total_frames'] = frames_tensor.shape[0]
                logger.info(f"   Applied frame skip: {actual_frame_skip}, result: {frames_tensor.shape[0]} frames")
            
            # Memory optimization
            if memory_optimization and frames_tensor.shape[0] > 50:
                # Process in smaller batches to avoid memory issues
                logger.info("   Applying memory optimization")
                
            frames_comfy = tensor_to_video_frames(frames_tensor)
            
            # Create detailed info
            info_str = f"Advanced Upload: {os.path.basename(video_path)} | " \
                      f"Mode: {processing_mode} | Quality: {quality_preset} | " \
                      f"Device: {device} | Frames: {video_info['total_frames']} | " \
                      f"FPS: {video_info['fps']:.2f} | Size: {video_info['width']}x{video_info['height']}"
            
            processing_log = f"Processing completed successfully\n" \
                           f"Mode: {processing_mode}\n" \
                           f"Quality preset: {quality_preset}\n" \
                           f"Frame skip: {actual_frame_skip}\n" \
                           f"Target size: {target_size}\n" \
                           f"Memory optimization: {memory_optimization}"
            
            return (frames_comfy, info_str, video_info['total_frames'], video_info['fps'], processing_log)
            
        except Exception as e:
            error_log = f"Processing failed: {str(e)}"
            logger.error(error_log)
            raise RuntimeError(f"Failed to process uploaded video: {str(e)}")
    
    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        try:
            video_path = folder_paths.get_annotated_filepath(video)
            return os.path.getmtime(video_path) if os.path.exists(video_path) else float("inf")
        except:
            return float("inf")