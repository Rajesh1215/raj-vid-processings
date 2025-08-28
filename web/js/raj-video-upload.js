import { app } from '../../../scripts/app.js'
import { ComfyWidgets } from '../../../scripts/widgets.js'
import './raj-video-preview.js' // Import preview widget

// Video file extensions supported by Raj Video Upload
const SUPPORTED_VIDEO_EXTENSIONS = ['mp4', 'mov', 'avi', 'mkv', 'webm', 'gif'];

function isVideoFile(filename) {
    if (!filename) return false;
    const ext = filename.toLowerCase().split('.').pop();
    return SUPPORTED_VIDEO_EXTENSIONS.includes(ext);
}

// Create upload widget similar to VideoHelperSuite
function createVideoUploadWidget(node, inputName, inputData, app) {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = SUPPORTED_VIDEO_EXTENSIONS.map(ext => `.${ext}`).join(',');
    fileInput.style.display = "none";
    fileInput.multiple = false;
    
    document.body.appendChild(fileInput);
    
    // Create the upload button widget
    const uploadWidget = ComfyWidgets.STRING(node, inputName, ["STRING", inputData[1]], app).widget;
    uploadWidget.value = "Choose video to upload";
    
    // Override the callback to open file picker
    const originalCallback = uploadWidget.callback;
    uploadWidget.callback = function() {
        fileInput.click();
    };
    
    // Handle file selection
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file && isVideoFile(file.name)) {
            uploadFile(file, updateVideoList);
        } else if (file) {
            alert('Please select a supported video file: ' + SUPPORTED_VIDEO_EXTENSIONS.join(', '));
        }
    });
    
    return { widget: uploadWidget, minWidth: 200, minHeight: 20 };
}

// Upload file to ComfyUI input directory
async function uploadFile(file, callback) {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('subfolder', '');
    formData.append('type', 'input');
    
    try {
        const response = await fetch('/upload/image', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log('✅ Video uploaded successfully:', result.name);
            
            // Update video list in all Raj Video Upload nodes
            if (callback) callback();
            
            // Show success notification
            showNotification(`Video uploaded: ${result.name}`, 'success');
        } else {
            console.error('❌ Upload failed:', response.statusText);
            showNotification('Upload failed: ' + response.statusText, 'error');
        }
    } catch (error) {
        console.error('❌ Upload error:', error);
        showNotification('Upload error: ' + error.message, 'error');
    }
}

// Update video dropdown lists in all nodes
function updateVideoList() {
    // Find all Raj Video Upload nodes and refresh their video lists
    app.graph._nodes.forEach(node => {
        if (node.type === "RajVideoUpload" || node.type === "RajVideoUploadAdvanced") {
            // Refresh the video dropdown
            const videoWidget = node.widgets.find(w => w.name === "video");
            if (videoWidget && videoWidget.options) {
                // Trigger a refresh of the video list
                // This will cause the node to re-scan the input directory
                node.onResize?.(node.size);
            }
        }
    });
}

// Show notification to user
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 4px;
        color: white;
        font-weight: bold;
        z-index: 10000;
        max-width: 300px;
        word-wrap: break-word;
    `;
    
    switch(type) {
        case 'success':
            notification.style.backgroundColor = '#4CAF50';
            break;
        case 'error':
            notification.style.backgroundColor = '#f44336';
            break;
        default:
            notification.style.backgroundColor = '#2196F3';
    }
    
    document.body.appendChild(notification);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 3000);
}

// Enhanced file handling for video files
let originalHandleFile = app.handleFile;
app.handleFile = async function(file) {
    // Handle video files dropped onto the interface
    if (file?.type?.startsWith("video/") || isVideoFile(file?.name)) {
        console.log('🎬 Handling video file:', file.name);
        
        // Upload the video file
        await uploadFile(file, updateVideoList);
        return;
    }
    
    // Fallback to original handler
    return await originalHandleFile.apply(this, arguments);
};

// Register the upload widget for our nodes
app.registerExtension({
    name: "RajVideoProcessing.VideoUpload",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Handle RajVideoUpload node
        if (nodeData.name === "RajVideoUpload" || nodeData.name === "RajVideoUploadAdvanced") {
            console.log(`🎬 Registering upload widget for ${nodeData.name}`);
            
            const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                const r = origGetExtraMenuOptions?.apply(this, arguments);
                
                // Add refresh option to context menu
                options.push({
                    content: "🔄 Refresh video list",
                    callback: () => {
                        updateVideoList();
                        showNotification('Video list refreshed', 'info');
                    }
                });
                
                return r;
            };
        }
    },
    
    nodeCreated(node) {
        // Add upload functionality when node is created
        if (node.comfyClass === "RajVideoUpload" || node.comfyClass === "RajVideoUploadAdvanced") {
            console.log(`🎬 Node created: ${node.comfyClass}`);
            
            // Find the hidden upload input and replace with upload widget
            for (const input of node.inputs || []) {
                if (input.name === "choose video to upload") {
                    // Hide the original input
                    input.widget = createVideoUploadWidget(node, input.name, ["UPLOAD", {}], app);
                    break;
                }
            }
        }
    }
});

// Add drag and drop support for video files
document.addEventListener('DOMContentLoaded', function() {
    // Enhance the existing file input to accept video files
    const fileInput = document.getElementById("comfy-file-input");
    if (fileInput) {
        const originalAccept = fileInput.accept;
        const videoAccept = SUPPORTED_VIDEO_EXTENSIONS.map(ext => `video/${ext === 'mov' ? 'quicktime' : ext}`).join(',');
        fileInput.accept = originalAccept + ',' + videoAccept;
        console.log('🎬 Enhanced file input to accept video files');
    }
});

console.log('🎬 Raj Video Upload extension loaded successfully!');