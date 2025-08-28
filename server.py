"""
Server routes for raj-vid-processings video preview functionality
"""

import os
import json
import mimetypes
from aiohttp import web
from pathlib import Path
import folder_paths
import logging

logger = logging.getLogger(__name__)

# Video MIME types
VIDEO_MIMETYPES = {
    "mp4": "video/mp4",
    "webm": "video/webm",
    "avi": "video/x-msvideo",
    "mov": "video/quicktime",
    "mkv": "video/x-matroska",
    "gif": "image/gif"
}

class RajVideoServer:
    """
    HTTP server routes for video preview and metadata
    """
    
    @staticmethod
    def add_routes(server):
        """
        Add video preview routes to ComfyUI server
        """
        routes = web.RouteTableDef()
        
        @routes.get("/raj-vid/preview")
        async def preview_video(request):
            """
            Stream video file for preview
            Supports range requests for video scrubbing
            """
            try:
                # Get video path from query params
                path = request.query.get("path", "")
                format_type = request.query.get("format", "mp4")
                
                if not path:
                    return web.Response(text="No path provided", status=400)
                
                # Resolve full path
                if not os.path.isabs(path):
                    # Check in output directory first
                    output_dir = folder_paths.get_output_directory()
                    full_path = os.path.join(output_dir, path)
                    if not os.path.exists(full_path):
                        # Check in input directory
                        input_dir = folder_paths.get_input_directory()
                        full_path = os.path.join(input_dir, path)
                else:
                    full_path = path
                
                # Security check - ensure path is within allowed directories
                output_dir = folder_paths.get_output_directory()
                input_dir = folder_paths.get_input_directory()
                temp_dir = folder_paths.get_temp_directory()
                
                allowed_dirs = [output_dir, input_dir, temp_dir]
                if not any(os.path.abspath(full_path).startswith(os.path.abspath(d)) for d in allowed_dirs):
                    logger.warning(f"Attempted to access file outside allowed directories: {full_path}")
                    return web.Response(text="Access denied", status=403)
                
                if not os.path.exists(full_path):
                    return web.Response(text="File not found", status=404)
                
                # Get file stats
                file_size = os.path.getsize(full_path)
                
                # Handle range requests for video scrubbing
                range_header = request.headers.get("Range")
                if range_header:
                    # Parse range header (e.g., "bytes=0-1024")
                    range_match = range_header.replace("bytes=", "").split("-")
                    start = int(range_match[0]) if range_match[0] else 0
                    end = int(range_match[1]) if range_match[1] else file_size - 1
                    
                    # Read requested range
                    with open(full_path, "rb") as f:
                        f.seek(start)
                        data = f.read(end - start + 1)
                    
                    # Return partial content
                    headers = {
                        "Content-Type": VIDEO_MIMETYPES.get(format_type, "video/mp4"),
                        "Content-Range": f"bytes {start}-{end}/{file_size}",
                        "Accept-Ranges": "bytes",
                        "Content-Length": str(len(data))
                    }
                    return web.Response(body=data, status=206, headers=headers)
                else:
                    # Return full file
                    with open(full_path, "rb") as f:
                        data = f.read()
                    
                    headers = {
                        "Content-Type": VIDEO_MIMETYPES.get(format_type, "video/mp4"),
                        "Accept-Ranges": "bytes",
                        "Content-Length": str(file_size)
                    }
                    return web.Response(body=data, status=200, headers=headers)
                
            except Exception as e:
                logger.error(f"Error serving video preview: {e}")
                return web.Response(text=str(e), status=500)
        
        @routes.get("/raj-vid/metadata")
        async def get_video_metadata(request):
            """
            Return video metadata as JSON
            """
            try:
                path = request.query.get("path", "")
                
                if not path:
                    return web.json_response({"error": "No path provided"}, status=400)
                
                # Resolve full path
                if not os.path.isabs(path):
                    output_dir = folder_paths.get_output_directory()
                    full_path = os.path.join(output_dir, path)
                    if not os.path.exists(full_path):
                        input_dir = folder_paths.get_input_directory()
                        full_path = os.path.join(input_dir, path)
                else:
                    full_path = path
                
                if not os.path.exists(full_path):
                    return web.json_response({"error": "File not found"}, status=404)
                
                # Get video metadata using OpenCV
                import cv2
                cap = cv2.VideoCapture(full_path)
                
                if not cap.isOpened():
                    return web.json_response({"error": "Failed to open video"}, status=500)
                
                metadata = {
                    "path": path,
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": cap.get(cv2.CAP_PROP_FPS),
                    "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
                    "format": os.path.splitext(path)[1][1:],
                    "size_bytes": os.path.getsize(full_path)
                }
                
                cap.release()
                
                return web.json_response(metadata)
                
            except Exception as e:
                logger.error(f"Error getting video metadata: {e}")
                return web.json_response({"error": str(e)}, status=500)
        
        @routes.get("/raj-vid/thumbnail") 
        async def get_video_thumbnail(request):
            """
            Generate and return video thumbnail
            """
            try:
                path = request.query.get("path", "")
                frame_num = int(request.query.get("frame", 0))
                
                if not path:
                    return web.Response(text="No path provided", status=400)
                
                # Resolve full path
                if not os.path.isabs(path):
                    output_dir = folder_paths.get_output_directory()
                    full_path = os.path.join(output_dir, path)
                    if not os.path.exists(full_path):
                        input_dir = folder_paths.get_input_directory()
                        full_path = os.path.join(input_dir, path)
                else:
                    full_path = path
                
                if not os.path.exists(full_path):
                    return web.Response(text="File not found", status=404)
                
                # Extract frame using OpenCV
                import cv2
                import io
                from PIL import Image
                
                cap = cv2.VideoCapture(full_path)
                if not cap.isOpened():
                    return web.Response(text="Failed to open video", status=500)
                
                # Seek to requested frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    return web.Response(text="Failed to read frame", status=500)
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image and resize for thumbnail
                img = Image.fromarray(frame)
                img.thumbnail((320, 180), Image.Resampling.LANCZOS)
                
                # Save to bytes
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="JPEG", quality=85)
                img_bytes.seek(0)
                
                return web.Response(
                    body=img_bytes.read(),
                    headers={"Content-Type": "image/jpeg"}
                )
                
            except Exception as e:
                logger.error(f"Error generating thumbnail: {e}")
                return web.Response(text=str(e), status=500)
        
        # Register routes with the server
        server.app.router.add_routes(routes)
        logger.info("✅ Raj Video Server routes registered")


# Initialize server routes when module is imported
def setup_server(server):
    """
    Setup function called by ComfyUI to register routes
    """
    RajVideoServer.add_routes(server)
    return True


# Export for ComfyUI
WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}