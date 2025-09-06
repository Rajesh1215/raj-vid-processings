import { app } from '../../../scripts/app.js'

// Video preview widget for Raj Video Processing nodes
class RajVideoPreviewWidget {
    constructor(node, name, opts) {
        this.node = node;
        this.name = name;
        this.options = opts || {};
        
        // Create the preview container
        this.element = document.createElement("div");
        this.element.className = "raj-video-preview";
        this.element.style.cssText = `
            width: 100%;
            max-width: 400px;
            margin: 5px 0;
            border: 1px solid #444;
            border-radius: 4px;
            background: #1a1a1a;
            padding: 8px;
            box-sizing: border-box;
        `;
        
        this.videoElement = null;
        this.audioElement = null;
        this.currentVideoData = null;
        this.isPlaying = false;
        
        this.setupUI();
    }
    
    setupUI() {
        // Header with title
        const header = document.createElement("div");
        header.style.cssText = `
            color: #fff;
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        `;
        header.innerHTML = `
            <span>🎬 Video Preview</span>
            <span id="video-info" style="font-size: 10px; color: #888;"></span>
        `;
        this.element.appendChild(header);
        
        // Video container
        this.videoContainer = document.createElement("div");
        this.videoContainer.style.cssText = `
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 4px;
            overflow: hidden;
            min-height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
        `;
        
        // Placeholder text
        this.placeholder = document.createElement("div");
        this.placeholder.style.cssText = `
            color: #666;
            font-size: 14px;
            text-align: center;
        `;
        this.placeholder.textContent = "No video loaded";
        this.videoContainer.appendChild(this.placeholder);
        
        this.element.appendChild(this.videoContainer);
        
        // Controls container
        this.controlsContainer = document.createElement("div");
        this.controlsContainer.style.cssText = `
            margin-top: 8px;
            display: none;
        `;
        this.setupControls();
        this.element.appendChild(this.controlsContainer);
    }
    
    setupControls() {
        // Play/Pause button
        this.playButton = document.createElement("button");
        this.playButton.innerHTML = "▶️";
        this.playButton.style.cssText = `
            background: #333;
            border: 1px solid #555;
            color: white;
            padding: 5px 10px;
            margin-right: 5px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        `;
        this.playButton.onclick = () => this.togglePlayback();
        this.controlsContainer.appendChild(this.playButton);
        
        // Time display
        this.timeDisplay = document.createElement("span");
        this.timeDisplay.style.cssText = `
            color: #ccc;
            font-size: 11px;
            margin: 0 10px;
        `;
        this.timeDisplay.textContent = "00:00 / 00:00";
        this.controlsContainer.appendChild(this.timeDisplay);
        
        // Volume control
        this.volumeSlider = document.createElement("input");
        this.volumeSlider.type = "range";
        this.volumeSlider.min = 0;
        this.volumeSlider.max = 1;
        this.volumeSlider.step = 0.1;
        this.volumeSlider.value = 0.8;
        this.volumeSlider.style.cssText = `
            width: 60px;
            margin-left: 10px;
        `;
        this.volumeSlider.oninput = (e) => {
            if (this.videoElement) {
                this.videoElement.volume = e.target.value;
            }
            if (this.audioElement) {
                this.audioElement.volume = e.target.value;
            }
        };
        this.controlsContainer.appendChild(this.volumeSlider);
        
        // Fullscreen button
        this.fullscreenButton = document.createElement("button");
        this.fullscreenButton.innerHTML = "⛶";
        this.fullscreenButton.style.cssText = `
            background: #333;
            border: 1px solid #555;
            color: white;
            padding: 5px 8px;
            margin-left: 10px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        `;
        this.fullscreenButton.onclick = () => this.toggleFullscreen();
        this.controlsContainer.appendChild(this.fullscreenButton);
    }
    
    updateVideoPreview(previewData) {
        if (!previewData) {
            this.clearPreview();
            return;
        }
        
        console.log('🎬 Updating video preview:', previewData);
        this.currentVideoData = previewData;
        
        // Clear existing video
        this.clearVideo();
        
        // Extract video info
        let videoUrl, filename, format;
        
        if (typeof previewData === 'string') {
            // Simple file path
            videoUrl = `/raj-vid/preview?path=${encodeURIComponent(previewData)}&format=mp4`;
            filename = previewData;
            format = previewData.split('.').pop() || 'mp4';
        } else if (previewData.filename) {
            // VHS-compatible format
            const type = previewData.type || 'output';
            const subfolder = previewData.subfolder || '';
            filename = previewData.filename;
            format = filename.split('.').pop() || 'mp4';
            
            if (subfolder) {
                videoUrl = `/raj-vid/preview?path=${encodeURIComponent(subfolder ? subfolder + '/' + filename : filename)}&format=${format}`;
            } else {
                videoUrl = `/raj-vid/preview?path=${encodeURIComponent(filename)}&format=${format}`;
            }
        } else {
            console.warn('Unknown preview data format:', previewData);
            return;
        }
        
        // Update info display
        const infoElement = this.element.querySelector('#video-info');
        if (infoElement) {
            const fileSize = previewData.frame_count ? `${previewData.frame_count}f` : '';
            const fps = previewData.frame_rate ? `${previewData.frame_rate}fps` : '';
            infoElement.textContent = [filename, fileSize, fps].filter(x => x).join(' | ');
        }
        
        // Create video element
        this.createVideoElement(videoUrl, format);
    }
    
    createVideoElement(videoUrl, format) {
        // Remove placeholder
        if (this.placeholder && this.placeholder.parentNode) {
            this.placeholder.parentNode.removeChild(this.placeholder);
        }
        
        // Create video element
        this.videoElement = document.createElement("video");
        this.videoElement.style.cssText = `
            width: 100%;
            height: auto;
            max-height: 300px;
            background: #000;
        `;
        this.videoElement.controls = false; // We'll use custom controls
        this.videoElement.muted = false;
        this.videoElement.volume = this.volumeSlider.value;
        this.videoElement.preload = "metadata";
        
        // Set up event listeners
        this.videoElement.onloadedmetadata = () => {
            console.log('✅ Video metadata loaded');
            this.updateTimeDisplay();
            this.controlsContainer.style.display = 'block';
        };
        
        this.videoElement.ontimeupdate = () => {
            this.updateTimeDisplay();
        };
        
        this.videoElement.onended = () => {
            this.playButton.innerHTML = "▶️";
            this.isPlaying = false;
        };
        
        this.videoElement.onerror = (e) => {
            console.error('❌ Video load error:', e);
            this.showError('Failed to load video');
        };
        
        this.videoElement.onloadstart = () => {
            console.log('🔄 Video loading started');
        };
        
        // Set video source
        this.videoElement.src = videoUrl;
        this.videoContainer.appendChild(this.videoElement);
        
        // Try to load the video
        this.videoElement.load();
    }
    
    togglePlayback() {
        if (!this.videoElement) return;
        
        if (this.isPlaying) {
            this.videoElement.pause();
            this.playButton.innerHTML = "▶️";
            this.isPlaying = false;
        } else {
            this.videoElement.play().then(() => {
                this.playButton.innerHTML = "⏸️";
                this.isPlaying = true;
            }).catch(e => {
                console.error('❌ Video play error:', e);
                this.showError('Failed to play video');
            });
        }
    }
    
    toggleFullscreen() {
        if (!this.videoElement) return;
        
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            this.videoElement.requestFullscreen().catch(e => {
                console.error('❌ Fullscreen error:', e);
            });
        }
    }
    
    updateTimeDisplay() {
        if (!this.videoElement) return;
        
        const current = this.videoElement.currentTime || 0;
        const duration = this.videoElement.duration || 0;
        
        const formatTime = (seconds) => {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        };
        
        this.timeDisplay.textContent = `${formatTime(current)} / ${formatTime(duration)}`;
    }
    
    showError(message) {
        this.clearVideo();
        
        const errorDiv = document.createElement("div");
        errorDiv.style.cssText = `
            color: #ff6b6b;
            font-size: 12px;
            text-align: center;
            padding: 20px;
        `;
        errorDiv.innerHTML = `❌ ${message}`;
        this.videoContainer.appendChild(errorDiv);
    }
    
    clearVideo() {
        if (this.videoElement) {
            this.videoElement.pause();
            this.videoElement.src = '';
            if (this.videoElement.parentNode) {
                this.videoElement.parentNode.removeChild(this.videoElement);
            }
            this.videoElement = null;
        }
        
        if (this.audioElement) {
            this.audioElement.pause();
            this.audioElement.src = '';
            this.audioElement = null;
        }
        
        this.controlsContainer.style.display = 'none';
        this.isPlaying = false;
        this.playButton.innerHTML = "▶️";
    }
    
    clearPreview() {
        this.clearVideo();
        
        if (!this.placeholder.parentNode) {
            this.videoContainer.appendChild(this.placeholder);
        }
        
        const infoElement = this.element.querySelector('#video-info');
        if (infoElement) {
            infoElement.textContent = '';
        }
    }
}

// Register the video preview extension
app.registerExtension({
    name: "RajVideoProcessing.VideoPreview",
    
    nodeCreated(node) {
        // Add preview widget to nodes that output video
        if (node.comfyClass === "RajVideoSaver" || 
            node.comfyClass === "RajVideoSaverAdvanced" ||
            node.comfyClass === "RajVideoUpload" ||
            node.comfyClass === "RajVideoUploadAdvanced" ||
            node.comfyClass === "RajVideoPreview") {
            
            console.log(`🎬 Adding video preview to ${node.comfyClass}`);
            
            // Create preview widget
            const previewWidget = new RajVideoPreviewWidget(node, "video_preview", {});
            
            // Add to node
            if (!node.widgets) node.widgets = [];
            node.widgets.push({
                name: "video_preview",
                type: "video_preview",
                value: null,
                element: previewWidget.element,
                widget: previewWidget
            });
            
            // Override the node's onExecuted to handle video preview updates
            const originalOnExecuted = node.onExecuted;
            node.onExecuted = function(message) {
                if (originalOnExecuted) {
                    originalOnExecuted.call(this, message);
                }
                
                // Check if the message contains video preview data
                if (message && message.gifs && message.gifs.length > 0) {
                    console.log('🎬 Received video preview data:', message.gifs[0]);
                    previewWidget.updateVideoPreview(message.gifs[0]);
                } else if (message && message.videos && message.videos.length > 0) {
                    console.log('🎬 Received video data:', message.videos[0]);
                    previewWidget.updateVideoPreview(message.videos[0]);
                }
            };
            
            // Adjust node size to accommodate preview
            const originalComputeSize = node.computeSize;
            node.computeSize = function(out) {
                let size = originalComputeSize ? originalComputeSize.call(this, out) : [200, 100];
                
                // Add space for video preview
                if (previewWidget && previewWidget.element) {
                    size[1] += 280; // Height for video preview
                    size[0] = Math.max(size[0], 350); // Minimum width
                }
                
                return size;
            };
        }
    }
});

console.log('🎬 Raj Video Preview widget loaded successfully!');