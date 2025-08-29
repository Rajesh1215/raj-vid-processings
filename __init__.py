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
except ImportError:
    # Fallback for direct testing
    from nodes.video_loader import RajVideoLoader, RajVideoLoaderPath
    from nodes.video_concatenator import RajVideoConcatenator, RajVideoSequencer
    from nodes.video_saver import RajVideoSaver, RajVideoSaverAdvanced
    from nodes.video_upload import RajVideoUpload, RajVideoUploadAdvanced
    from nodes.video_effects import RajVideoEffects, RajVideoSharpness
    from nodes.video_transitions import RajVideoTransitions, RajTransitionLibrary

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
print("   - NEW: Video Effects, Sharpness, Transitions, Transition Library")
print("   - Video Formats: MP4, MOV, AVI, WebM, GIF")
print("   - Upload Support: Drag & Drop, Upload Button")
print("   - Effects: Time-based Brightness, Contrast, Blur with Easing")
print("   - Transitions: Fade, Zoom, Slide, Wipe, Dissolve at Cut Points")
print("   - Aspect Ratio: Resize, Pad, Crop, Stretch handling")
print("   - Auto-increment filenames (VideoHelperSuite style)")
print("   - Default 24 FPS, Video Preview in Web UI")
print("   - Cross-platform compatibility: Mac, Linux, Windows")