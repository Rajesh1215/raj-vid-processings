"""
Raj Video Processing Nodes for ComfyUI
Cross-platform GPU-accelerated video processing with MPS, CUDA, and CPU support

Author: Raj
"""

try:
    from .nodes.video_loader import RajVideoLoader, RajVideoLoaderPath
    from .nodes.video_concatenator import RajVideoConcatenator, RajVideoSequencer
    from .nodes.video_saver import RajVideoSaver, RajVideoSaverAdvanced
    from .nodes.video_upload import RajVideoUpload, RajVideoUploadAdvanced
    from .nodes.video_effects import RajVideoEffects, RajVideoSharpness
    from .nodes.video_transitions import RajVideoTransitions, RajTransitionLibrary
    from .nodes.video_trimmer import RajVideoTrimmer, RajVideoCutter, RajVideoTimecodeConverter
    from .nodes.video_multi_cutter import RajVideoMultiCutter, RajVideoSegmentManager
    from .nodes.video_mask_composite import RajVideoMaskComposite
except ImportError:
    # Fallback for direct testing
    from nodes.video_loader import RajVideoLoader, RajVideoLoaderPath
    from nodes.video_concatenator import RajVideoConcatenator, RajVideoSequencer
    from nodes.video_saver import RajVideoSaver, RajVideoSaverAdvanced
    from nodes.video_upload import RajVideoUpload, RajVideoUploadAdvanced
    from nodes.video_effects import RajVideoEffects, RajVideoSharpness
    from nodes.video_transitions import RajVideoTransitions, RajTransitionLibrary
    from nodes.video_trimmer import RajVideoTrimmer, RajVideoCutter, RajVideoTimecodeConverter
    from nodes.video_multi_cutter import RajVideoMultiCutter, RajVideoSegmentManager
    from nodes.video_mask_composite import RajVideoMaskComposite

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "RajVideoLoader": RajVideoLoader,
    "RajVideoLoaderPath": RajVideoLoaderPath, 
    "RajVideoConcatenator": RajVideoConcatenator,
    "RajVideoSequencer": RajVideoSequencer,
    "RajVideoSaver": RajVideoSaver,
    "RajVideoSaverAdvanced": RajVideoSaverAdvanced,
    "RajVideoUpload": RajVideoUpload,
    "RajVideoUploadAdvanced": RajVideoUploadAdvanced,
    "RajVideoEffects": RajVideoEffects,
    "RajVideoSharpness": RajVideoSharpness,
    "RajVideoTransitions": RajVideoTransitions,
    "RajTransitionLibrary": RajTransitionLibrary,
    "RajVideoTrimmer": RajVideoTrimmer,
    "RajVideoCutter": RajVideoCutter,
    "RajVideoTimecodeConverter": RajVideoTimecodeConverter,
    "RajVideoMultiCutter": RajVideoMultiCutter,
    "RajVideoSegmentManager": RajVideoSegmentManager,
    "RajVideoMaskComposite": RajVideoMaskComposite,
}

# Display names in ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "RajVideoLoader": "📹 Raj Video Loader",
    "RajVideoLoaderPath": "📁 Raj Video Loader (Path)",
    "RajVideoConcatenator": "🔗 Raj Video Concatenator", 
    "RajVideoSequencer": "🎬 Raj Video Sequencer",
    "RajVideoSaver": "💾 Raj Video Saver",
    "RajVideoSaverAdvanced": "🎛️ Raj Video Saver (Advanced)",
    "RajVideoUpload": "📤 Raj Video Upload",
    "RajVideoUploadAdvanced": "🎚️ Raj Video Upload (Advanced)",
    "RajVideoEffects": "🎨 Raj Video Effects",
    "RajVideoSharpness": "✨ Raj Video Sharpness",
    "RajVideoTransitions": "🎭 Raj Video Transitions",
    "RajTransitionLibrary": "🔄 Raj Transition Library",
    "RajVideoTrimmer": "✂️ Raj Video Trimmer",
    "RajVideoCutter": "🔪 Raj Video Cutter",
    "RajVideoTimecodeConverter": "🕐 Raj Timecode Converter",
    "RajVideoMultiCutter": "✂️ Raj Multi-Cutter",
    "RajVideoSegmentManager": "📊 Raj Segment Manager",
    "RajVideoMaskComposite": "🎭 Raj Mask Composite",
}

# Web directory for custom UI components (optional)
WEB_DIRECTORY = "./web"

# Import and register server routes
try:
    from .server import setup_server
    # This will be called by ComfyUI to set up the server routes
    def on_server_start(server):
        setup_server(server)
        return True
except ImportError as e:
    print(f"⚠️ Server routes not loaded: {e}")
    on_server_start = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
if on_server_start:
    __all__.append("on_server_start")

print("🎬 Raj Video Processing Nodes loaded successfully!")
print("   - GPU Support: MPS (Mac), CUDA (NVIDIA), CPU (Fallback)")
print("   - Core Nodes: Video Loader, Concatenator, Sequencer, Video Saver, Video Upload")
print("   - Effects: Time-based Brightness, Contrast, Blur, Saturation with Easing")
print("   - Transitions: Fade, Zoom, Slide, Wipe, Dissolve at Cut Points")
print("   - NEW: Video Trimmer, Video Cutter, Multi-Cutter, Mask Composite")
print("   - Video Formats: MP4, MOV, AVI, WebM, GIF")
print("   - Upload Support: Drag & Drop, Upload Button")
print("   - Trimming: Time-based (seconds) or Timecode (HH:MM:SS:FF)")
print("   - Cutting: Dual outputs (remaining + removed segments)")
print("   - Multi-Cutting: Multiple cut points with segment management")
print("   - Masking: Chroma key, Color range, Brightness, Custom HSV")
print("   - Compositing: Advanced blending modes (Normal, Multiply, Screen, Overlay)")
print("   - Aspect Ratio: Resize, Pad, Crop, Stretch handling")
print("   - Auto-increment filenames (VideoHelperSuite style)")
print("   - Default 24 FPS, Video Preview in Web UI")
print("   - Cross-platform compatibility: Mac, Linux, Windows")