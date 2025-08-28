# Troubleshooting Guide

## Common Issues and Solutions

### JSON Serialization Error

**Error Message:**
```
TypeError: Object of type function is not JSON serializable
```

**Cause:**
This error occurs when using lambda functions or function references in `INPUT_TYPES` definitions. ComfyUI needs to serialize node definitions to JSON for the frontend, but functions cannot be serialized.

**Solution:**
Replace lambda functions with direct function calls that return values immediately.

❌ **Wrong:**
```python
def INPUT_TYPES(cls):
    return {
        "required": {
            "video": (lambda: get_video_files(), {...})  # Lambda function - NOT serializable
        }
    }
```

✅ **Correct:**
```python
def INPUT_TYPES(cls):
    return {
        "required": {
            "video": (get_video_files(), {...})  # Direct function call - serializable
        }
    }
```

The function `get_video_files()` is called immediately when `INPUT_TYPES` is executed, returning a list that can be serialized to JSON.

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'folder_paths'`

**Cause:** Testing outside of ComfyUI environment

**Solution:** The nodes include fallback imports for testing. When running in ComfyUI, the proper modules will be available.

### GPU Detection Issues

**Mac (MPS):**
- Ensure you have PyTorch with MPS support
- Check with: `torch.backends.mps.is_available()`

**NVIDIA (CUDA):**
- Install PyTorch with CUDA support
- Check with: `torch.cuda.is_available()`

### Video Upload Issues

**Problem:** Upload button not appearing

**Solution:**
1. Ensure the web directory is properly set in `__init__.py`
2. Check that JavaScript files are in `web/js/` directory
3. Restart ComfyUI after adding nodes

**Problem:** Videos not appearing in dropdown after upload

**Solution:**
The video list is refreshed when nodes are created. You may need to:
1. Add a new node
2. Reload the workflow
3. Use the refresh option in node's context menu

### Memory Issues

**Problem:** Out of memory errors with large videos

**Solution:**
1. Enable batch processing in concatenator nodes
2. Use `max_frames` parameter to limit frame count
3. Reduce resolution with target_width/height
4. Use quality presets in upload advanced node

### File Format Issues

**Problem:** Video won't save in certain format

**Solution:**
- Ensure FFmpeg is installed for advanced codecs
- Use fallback formats (MP4 with h264 is most compatible)
- Check codec support with: `ffmpeg -codecs`

## Testing the Nodes

Run the test suite to verify everything works:
```bash
cd ComfyUI/custom_nodes/raj-vid-processings
python test_nodes.py
```

## Debug Logging

The nodes include comprehensive logging. Check the console output for:
- 🍎 MPS detection (Mac)
- 🚀 CUDA detection (NVIDIA)  
- 💻 CPU fallback
- 📹 Video loading status
- 💾 Save operations
- ❌ Error messages

## Getting Help

If you encounter issues not covered here:
1. Check the console/terminal for error messages
2. Run the test suite to identify problems
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Verify ComfyUI is up to date