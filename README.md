# Raj Video Processing Nodes 🎬

Cross-platform GPU-accelerated video processing nodes for ComfyUI with support for Mac (MPS), Linux/Windows (CUDA), and CPU fallback.

## Features

- **🚀 GPU Acceleration**: Automatic detection of MPS (Mac), CUDA (NVIDIA), or CPU
- **📹 Video Loading**: Efficient video loading with frame rate control and resizing  
- **🔗 Video Concatenation**: GPU-accelerated concatenation with crossfade transitions
- **🎬 Video Sequencing**: Precise timing control for video segments
- **💾 Memory Optimization**: Intelligent batch processing to prevent OOM errors
- **🌍 Cross-Platform**: Works on Mac, Linux, and Windows

## Installation

1. Copy this folder to your ComfyUI custom_nodes directory:
   ```
   ComfyUI/custom_nodes/raj-vid-processings/
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Restart ComfyUI

## Available Nodes

### 📹 Raj Video Loader
Load video files with GPU acceleration and format conversion.

**Features:**
- Automatic device detection (MPS/CUDA/CPU)
- Frame rate control and limiting
- Automatic resizing with aspect ratio preservation
- Memory-efficient loading

**Inputs:**
- `video_path`: Path to video file
- `target_fps`: Target frame rate (0 = original)
- `max_frames`: Maximum frames to load (0 = all)
- `target_width/height`: Resize dimensions (0 = original)
- `force_device`: Override automatic device selection

### 📁 Raj Video Loader (Path)
Similar to Video Loader but accepts full file paths for external videos.

### 🔗 Raj Video Concatenator
Concatenate multiple videos with optional crossfade transitions.

**Features:**
- GPU-accelerated concatenation
- Smooth crossfade transitions
- Automatic dimension matching
- Memory optimization for large videos

**Inputs:**
- `video1-5`: Video input slots
- `transition_frames`: Crossfade duration in frames
- `batch_processing`: Enable memory optimization
- `force_device`: Device selection

### 🎬 Raj Video Sequencer
Advanced video sequencing with precise timing control.

**Features:**
- Define exact start/end times for segments
- Frame-accurate timing
- Multiple segment support

**Inputs:**
- `videos`: Input video frames
- `sequence_config`: Timing configuration (start,end per line)
- `output_fps`: Output frame rate

### 💾 Raj Video Saver
Save video frames to multiple formats with quality controls.

**Features:**
- Multiple formats: MP4, MOV, AVI, WebM, GIF
- GPU-accelerated encoding (NVENC for NVIDIA)
- Quality and compression controls
- Automatic codec selection
- Timestamp support

**Inputs:**
- `frames`: Video frames to save
- `filename`: Output filename (without extension)
- `fps`: Output frame rate
- `format`: Video format (mp4, mov, avi, webm, gif)
- `quality`: Video quality (0-51, lower = better)
- `codec`: Video codec (auto, h264, h265, vp9, av1, prores)
- `gpu_encoding`: Use GPU encoding when available
- `save_to_output`: Save to output directory (vs temp)
- `add_timestamp`: Add timestamp to filename

### 🎛️ Raj Video Saver (Advanced)
Advanced video saving with custom FFmpeg parameters and audio support.

**Features:**
- Custom FFmpeg arguments
- Audio track merging
- Custom bitrate control
- Batch processing optimization
- Resolution override

**Inputs:**
- `frames`: Video frames to save
- `filename`: Output filename
- `fps`: Output frame rate
- `width/height`: Output resolution
- `bitrate`: Target bitrate (e.g., "5M", "1000k")
- `custom_ffmpeg_args`: Custom FFmpeg parameters
- `audio_file`: Optional audio file to merge
- `batch_size`: Memory optimization batch size

### 📤 Raj Video Upload
Upload and load video files with GPU acceleration and upload button.

**Features:**
- Upload button interface (like VideoHelperSuite)
- Drag & drop video file support
- GPU-accelerated processing (MPS/CUDA/CPU)
- Frame rate control and resizing
- Automatic file management

**Inputs:**
- `video`: Video file dropdown + upload button
- `target_fps`: Target frame rate (0 = original)
- `max_frames`: Maximum frames to load (0 = all)
- `target_width/height`: Resize dimensions (0 = original)
- `force_device`: Device selection

**How to Use:**
1. Click the "Choose video to upload" button
2. Select video file from your computer
3. File is automatically uploaded to ComfyUI input directory
4. Video is processed with GPU acceleration

### 🎚️ Raj Video Upload (Advanced)
Advanced video upload with processing modes and quality presets.

**Features:**
- Multiple processing modes (full, keyframes, every_nth)
- Quality presets (high, medium, low, custom)
- Memory optimization
- Advanced frame skipping
- Detailed processing logs

**Inputs:**
- `video`: Video file with upload button
- `target_fps`: Target frame rate
- `processing_mode`: How to process frames
- `max_frames`: Frame limit
- `frame_skip`: Frame skip interval
- `quality_preset`: Automatic quality settings
- `custom_width/height`: Custom dimensions
- `memory_optimization`: Enable memory management

## GPU Support Details

### Mac (MPS)
- Uses Metal Performance Shaders for GPU acceleration
- Automatically detected on Apple Silicon Macs
- Significant performance improvement over CPU

### NVIDIA (CUDA)
- Uses CUDA for GPU acceleration on Linux/Windows
- Requires NVIDIA GPU with CUDA support
- Install PyTorch with CUDA support for best performance

### CPU Fallback
- Automatically used when no GPU is available
- Still optimized with PyTorch operations
- Memory-efficient batch processing

## Performance Tips

1. **Memory Management**: Enable batch processing for large videos
2. **Device Selection**: Use "auto" for optimal device detection
3. **Video Format**: Use common formats (MP4, MOV) for best compatibility
4. **Frame Limits**: Set `max_frames` for very long videos to manage memory

## Troubleshooting

### Import Errors
Make sure all dependencies are installed:
```bash
pip install torch torchvision opencv-python numpy pillow
```

### GPU Not Detected
- **Mac**: Ensure you have Apple Silicon and PyTorch with MPS support
- **NVIDIA**: Install PyTorch with CUDA support
- **Fallback**: CPU processing will work on any system

### Memory Issues
- Enable batch processing
- Reduce `max_frames` for very large videos
- Use lower resolution videos for testing

## Examples

### Basic Video Loading
```
Input: video.mp4
Target FPS: 30
Max Frames: 300
Output: 300 frames at 30fps loaded on GPU
```

### Video Concatenation
```
Video 1: intro.mp4 (100 frames)
Video 2: content.mp4 (200 frames)  
Transition: 10 frames
Output: 300 frames with smooth 10-frame crossfade
```

### Video Sequencing
```
Config:
0.0,2.5
5.0,7.5
10.0,12.0
Output: Three segments with precise timing
```

## Integration

These nodes are designed to work seamlessly with:
- ComfyUI's built-in video nodes
- VideoHelperSuite nodes
- Other custom video processing nodes

The output format is compatible with standard ComfyUI image/video workflows.