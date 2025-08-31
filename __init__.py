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
    # Text Generation Nodes
    from .nodes.text_generator import RajTextGenerator
    from .nodes.whisper_captions import RajWhisperCaptions
    from .nodes.text_effects import RajTextEffects
    from .nodes.text_animator import RajTextAnimator
    from .nodes.text_sequencer import RajTextSequencer
    from .nodes.text_compositor import RajTextCompositor
    from .nodes.text_presets import RajTextPresets
    # Audio Processing Nodes
    from .nodes.whisper_process import RajWhisperProcess
    from .nodes.audio_loader import RajAudioLoader, RajAudioProcessor
    from .nodes.audio_preview import RajAudioPreview, RajAudioAnalyzer
    # Subtitle System Nodes
    from .nodes.subtitle_engine import RajSubtitleEngine
    from .nodes.text_to_timing import RajTextToTiming
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
    # Text Generation Nodes
    from nodes.text_generator import RajTextGenerator
    from nodes.whisper_captions import RajWhisperCaptions
    from nodes.text_effects import RajTextEffects
    from nodes.text_animator import RajTextAnimator
    from nodes.text_sequencer import RajTextSequencer
    from nodes.text_compositor import RajTextCompositor
    from nodes.text_presets import RajTextPresets
    # Audio Processing Nodes
    from nodes.whisper_process import RajWhisperProcess
    from nodes.audio_loader import RajAudioLoader, RajAudioProcessor
    from nodes.audio_preview import RajAudioPreview, RajAudioAnalyzer
    # Subtitle System Nodes
    from nodes.subtitle_engine import RajSubtitleEngine
    from nodes.text_to_timing import RajTextToTiming

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
    # Text Generation Nodes
    "RajTextGenerator": RajTextGenerator,
    "RajWhisperCaptions": RajWhisperCaptions,
    "RajTextEffects": RajTextEffects,
    "RajTextAnimator": RajTextAnimator,
    "RajTextSequencer": RajTextSequencer,
    "RajTextCompositor": RajTextCompositor,
    "RajTextPresets": RajTextPresets,
    # Audio Processing Nodes
    "RajWhisperProcess": RajWhisperProcess,
    "RajAudioLoader": RajAudioLoader,
    "RajAudioProcessor": RajAudioProcessor,
    "RajAudioPreview": RajAudioPreview,
    "RajAudioAnalyzer": RajAudioAnalyzer,
    # Subtitle System Nodes
    "RajSubtitleEngine": RajSubtitleEngine,
    "RajTextToTiming": RajTextToTiming,
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
    # Text Generation Nodes
    "RajTextGenerator": "📝 Raj Text Generator",
    "RajWhisperCaptions": "🎤 Raj Whisper Captions",
    "RajTextEffects": "✨ Raj Text Effects",
    "RajTextAnimator": "🎭 Raj Text Animator",
    "RajTextSequencer": "⏱️ Raj Text Sequencer",
    "RajTextCompositor": "🎬 Raj Text Compositor",
    "RajTextPresets": "📋 Raj Text Presets",
    # Audio Processing Nodes
    "RajWhisperProcess": "🎙️ Raj Whisper Process",
    "RajAudioLoader": "🔊 Raj Audio Loader",
    "RajAudioProcessor": "🎚️ Raj Audio Processor",
    "RajAudioPreview": "🎵 Raj Audio Preview",
    "RajAudioAnalyzer": "🔍 Raj Audio Analyzer",
    # Subtitle System Nodes
    "RajSubtitleEngine": "🎬 Raj Subtitle Engine",
    "RajTextToTiming": "⏰ Raj Text to Timing",
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
print("   - Video Segmenter - Split videos by time with dual outputs")
print("   - Mask Composite - Color-based video compositing with thresholding")
print("   - Professional Overlay System - Alpha channel, RGBA, Full/BBox modes")
print("   - Chroma Key - Green screen removal with edge softening")
print("   - Opacity Gradients - Linear, radial, corner, edge gradient masks")
print("   - Resize Toolkit - Center crop, fill, stretch with alpha preservation")
print("")
print("📝 Professional Text Generation System")
print("   - Text Generator: 512x512 PIL-based text images with 100+ fonts")
print("   - Whisper Captions: AI-powered transcription with word-level timing")
print("   - Text Effects: Shadows, glows, borders, gradients, opacity maps")
print("   - Text Animator: 15+ motion effects with easing functions")
print("   - Text Sequencer: Timeline control with frame-accurate positioning")
print("   - Text Compositor: Smart overlay with blend modes & auto-positioning")
print("   - Text Presets: Industry templates (YouTube, Broadcast, Cinema, Corporate)")
print("")
print("🔊 NEW: Modular Audio Processing System")
print("   - Enhanced Video Upload: Separate audio output for modular workflows")
print("   - Audio Loader: Standalone audio file support (WAV, MP3, AAC, FLAC)")
print("   - Whisper Process: Advanced transcription with sentence & word-level captions")
print("   - Audio Processor: Normalize, resample, trim, fade, amplify operations")
print("   - Audio Preview: Playable audio files with waveform visualization")
print("   - Audio Analyzer: Detailed frequency analysis and processing recommendations")
print("   - Multi-format Support: TorchAudio, Librosa, MoviePy backends")
print("")
print("   - Video Formats: MP4, MOV, AVI, WebM, GIF")
print("   - Upload Support: Drag & Drop, Upload Button")
print("   - Aspect Ratio: Resize, Pad, Crop, Stretch handling")
print("   - Auto-increment filenames (VideoHelperSuite style)")
print("   - Cross-platform compatibility: Mac, Linux, Windows")