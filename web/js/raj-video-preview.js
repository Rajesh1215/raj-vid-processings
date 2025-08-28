/**
 * Raj Video Preview Widget for ComfyUI
 * Provides video preview functionality for raj-vid-processings nodes
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Register the video preview widget extension
app.registerExtension({
    name: "RajVid.VideoPreview",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Add preview widget to nodes that output video
        if (nodeData.name === "RajVideoSaver" || 
            nodeData.name === "RajVideoUpload" ||
            nodeData.name === "RajVideoConcatenator") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const ret = onNodeCreated?.apply(this);
                
                // Add video preview widget
                const widget = this.addCustomWidget(createVideoPreviewWidget(this));
                widget.name = "video_preview";
                
                return ret;
            };
            
            // Handle execution results
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);
                
                // Update preview if video data is present
                if (message?.video_preview) {
                    updateVideoPreview(this, message.video_preview);
                }
            };
        }
    },
    
    async nodeCreated(node) {
        // Listen for preview updates via WebSocket
        if (node.comfyClass === "RajVideoUpload" || 
            node.comfyClass === "RajVideoSaver" ||
            node.comfyClass === "RajVideoConcatenator") {
            
            node.onRemoved = function() {
                // Clean up video element when node is removed
                const widget = this.widgets?.find(w => w.name === "video_preview");
                if (widget?.videoElement) {
                    widget.videoElement.pause();
                    widget.videoElement.src = "";
                }
            };
        }
    }
});

/**
 * Create a video preview widget
 */
function createVideoPreviewWidget(node) {
    const widget = {
        type: "raj_video_preview",
        name: "video_preview",
        draw: function(ctx, node, widget_width, y, H) {
            // Widget background
            ctx.fillStyle = "#222";
            ctx.fillRect(0, y, widget_width, H);
            
            if (!this.videoElement) {
                // Show placeholder text
                ctx.fillStyle = "#666";
                ctx.font = "12px Arial";
                ctx.textAlign = "center";
                ctx.fillText("No video loaded", widget_width / 2, y + H / 2);
            }
            
            return H;
        },
        computeSize: function() {
            return [0, 200]; // Height for video preview
        },
        onRemove: function() {
            if (this.videoElement) {
                this.videoElement.pause();
                this.videoElement.src = "";
                this.videoElement = null;
            }
        }
    };
    
    // Create video element
    const createVideoElement = () => {
        const container = document.createElement("div");
        container.style.position = "relative";
        container.style.width = "100%";
        container.style.height = "200px";
        container.style.backgroundColor = "#000";
        container.style.borderRadius = "4px";
        container.style.overflow = "hidden";
        
        const video = document.createElement("video");
        video.style.width = "100%";
        video.style.height = "100%";
        video.style.objectFit = "contain";
        video.controls = true;
        video.loop = true;
        video.muted = true; // Auto-play requires muted
        
        // Add loading indicator
        const loadingDiv = document.createElement("div");
        loadingDiv.style.position = "absolute";
        loadingDiv.style.top = "50%";
        loadingDiv.style.left = "50%";
        loadingDiv.style.transform = "translate(-50%, -50%)";
        loadingDiv.style.color = "#fff";
        loadingDiv.style.fontSize = "14px";
        loadingDiv.style.display = "none";
        loadingDiv.textContent = "Loading video...";
        
        container.appendChild(video);
        container.appendChild(loadingDiv);
        
        widget.videoElement = video;
        widget.loadingElement = loadingDiv;
        widget.containerElement = container;
        
        return container;
    };
    
    // Override serialize to handle video state
    widget.serialize = function() {
        return {
            videoSrc: this.videoElement?.src || null,
            currentTime: this.videoElement?.currentTime || 0
        };
    };
    
    widget.deserialize = function(data) {
        if (data?.videoSrc && this.videoElement) {
            this.videoElement.src = data.videoSrc;
            this.videoElement.currentTime = data.currentTime || 0;
        }
    };
    
    // Add DOM element to widget
    widget.element = createVideoElement();
    
    return widget;
}

/**
 * Update video preview with new video data
 */
function updateVideoPreview(node, videoData) {
    const widget = node.widgets?.find(w => w.name === "video_preview");
    if (!widget || !widget.videoElement) return;
    
    const { path, format, fps, duration, width, height, frame_count } = videoData;
    
    // Show loading indicator
    if (widget.loadingElement) {
        widget.loadingElement.style.display = "block";
    }
    
    // Build video URL
    const videoUrl = `/raj-vid/preview?path=${encodeURIComponent(path)}&format=${format}`;
    
    // Update video element
    widget.videoElement.src = videoUrl;
    
    // Add metadata display
    const updateMetadata = () => {
        if (widget.loadingElement) {
            widget.loadingElement.style.display = "none";
        }
        
        // Create or update metadata overlay
        let metaDiv = widget.containerElement.querySelector(".video-metadata");
        if (!metaDiv) {
            metaDiv = document.createElement("div");
            metaDiv.className = "video-metadata";
            metaDiv.style.position = "absolute";
            metaDiv.style.bottom = "40px";
            metaDiv.style.left = "5px";
            metaDiv.style.color = "#fff";
            metaDiv.style.fontSize = "11px";
            metaDiv.style.backgroundColor = "rgba(0,0,0,0.7)";
            metaDiv.style.padding = "3px 6px";
            metaDiv.style.borderRadius = "3px";
            metaDiv.style.pointerEvents = "none";
            widget.containerElement.appendChild(metaDiv);
        }
        
        metaDiv.innerHTML = `
            ${width}x${height} | ${fps}fps | ${frame_count} frames | ${duration.toFixed(2)}s
        `;
    };
    
    // Handle video load events
    widget.videoElement.onloadedmetadata = () => {
        updateMetadata();
        // Auto-play on load
        widget.videoElement.play().catch(() => {
            // Auto-play might be blocked
            console.log("Auto-play blocked, user interaction required");
        });
    };
    
    widget.videoElement.onerror = () => {
        if (widget.loadingElement) {
            widget.loadingElement.textContent = "Error loading video";
            widget.loadingElement.style.display = "block";
        }
    };
    
    // Update node size if needed
    node.setSize([node.size[0], node.size[1]]);
    app.graph.setDirtyCanvas(true);
}

/**
 * Handle real-time preview updates during processing
 */
api.addEventListener("raj-vid-preview", (event) => {
    const { node_id, preview_data } = event.detail;
    const node = app.graph.getNodeById(node_id);
    
    if (node && preview_data) {
        updateVideoPreview(node, preview_data);
    }
});

// Add custom widget registration
app.canvas.onDrawNode = function(node) {
    const widget = node.widgets?.find(w => w.type === "raj_video_preview");
    if (widget?.element && !widget.element.parentNode) {
        // Attach video element to DOM when node is drawn
        const nodeElement = app.canvas.node_widget?.get(node);
        if (nodeElement) {
            nodeElement.appendChild(widget.element);
        }
    }
};