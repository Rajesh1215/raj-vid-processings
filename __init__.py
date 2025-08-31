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
    from .nodes.video_segmenter import RajVideoSegmenter
    from .nodes.video_mask_composite import RajVideoMaskComposite
    from .nodes.video_overlay import RajVideoOverlay
    from .nodes.video_resize_toolkit import RajVideoResizeToolkit
    from .nodes.video_opacity_gradient import RajVideoOpacityGradient
    from .nodes.video_chroma_key import RajVideoChromaKey
except ImportError:
    # Fallback for direct testing
    from nodes.video_loader import RajVideoLoader, RajVideoLoaderPath
    from nodes.video_concatenator import RajVideoConcatenator, RajVideoSequencer
    from nodes.video_saver import RajVideoSaver, RajVideoSaverAdvanced
    from nodes.video_upload import RajVideoUpload, RajVideoUploadAdvanced
    from nodes.video_effects import RajVideoEffects, RajVideoSharpness
    from nodes.video_transitions import RajVideoTransitions, RajTransitionLibrary
    from nodes.video_segmenter import RajVideoSegmenter
    from nodes.video_mask_composite import RajVideoMaskComposite
    from nodes.video_overlay import RajVideoOverlay
    from nodes.video_resize_toolkit import RajVideoResizeToolkit
    from nodes.video_opacity_gradient import RajVideoOpacityGradient
    from nodes.video_chroma_key import RajVideoChromaKey

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
    "RajVideoSegmenter": RajVideoSegmenter,
    "RajVideoMaskComposite": RajVideoMaskComposite,
    "RajVideoOverlay": RajVideoOverlay,
    "RajVideoResizeToolkit": RajVideoResizeToolkit,
    "RajVideoOpacityGradient": RajVideoOpacityGradient,
    "RajVideoChromaKey": RajVideoChromaKey,
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
    "RajVideoSegmenter": "✂️ Raj Video Segmenter",
    "RajVideoMaskComposite": "🎭 Raj Video Mask Composite",
    "RajVideoOverlay": "🎬 Raj Video Overlay",
    "RajVideoResizeToolkit": "🔧 Raj Video Resize Toolkit",
    "RajVideoOpacityGradient": "🌈 Raj Video Opacity Gradient",
    "RajVideoChromaKey": "🎬 Raj Video Chroma Key",
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
print("   - NEW: Video Segmenter - Split videos by time with dual outputs")
print("   - NEW: Mask Composite - Color-based video compositing with thresholding")
print("   - NEW: Professional Overlay System - Alpha channel, RGBA, Full/BBox modes")
print("   - NEW: Chroma Key - Green screen removal with edge softening")
print("   - NEW: Opacity Gradients - Linear, radial, corner, edge gradient masks")
print("   - NEW: Resize Toolkit - Center crop, fill, stretch with alpha preservation")
print("   - Video Formats: MP4, MOV, AVI, WebM, GIF")
print("   - Upload Support: Drag & Drop, Upload Button")
print("   - Aspect Ratio: Resize, Pad, Crop, Stretch handling")
print("   - Auto-increment filenames (VideoHelperSuite style)")
print("   - Cross-platform compatibility: Mac, Linux, Windows")