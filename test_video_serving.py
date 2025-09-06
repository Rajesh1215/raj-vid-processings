#!/usr/bin/env python3
"""
Test video file serving and preview functionality in ComfyUI
"""

import torch
import os
import tempfile
import requests
from urllib.parse import urlencode

def test_video_file_creation():
    """Test if we can create a video file that ComfyUI can serve"""
    print("🎬 Testing video file creation and serving...")
    
    try:
        # Create test frames
        frames = torch.rand(10, 120, 160, 3).float()  # Small test video
        
        from nodes.video_saver import RajVideoSaver
        saver = RajVideoSaver()
        
        result = saver.save_video(
            frames=frames,
            filename="comfyui_preview_test",
            fps=10.0,
            format="mp4",
            quality=28,  # Lower quality for faster encoding
            save_to_output=True,  # Save to output directory
            add_timestamp=True
        )
        
        if "ui" in result:
            ui_data = result["ui"]
            if "gifs" in ui_data and len(ui_data["gifs"]) > 0:
                preview_info = ui_data["gifs"][0]
                print(f"✅ Video file created successfully")
                print(f"   Preview data: {preview_info}")
                
                # Check if file actually exists
                filename = preview_info.get("filename")
                file_type = preview_info.get("type", "output")
                
                if filename:
                    # Try to find the file
                    try:
                        import folder_paths
                        if file_type == "output":
                            output_dir = folder_paths.get_output_directory()
                        elif file_type == "temp":
                            output_dir = folder_paths.get_temp_directory()
                        else:
                            output_dir = folder_paths.get_input_directory()
                        
                        full_path = os.path.join(output_dir, filename)
                        
                        if os.path.exists(full_path):
                            file_size = os.path.getsize(full_path)
                            print(f"✅ File exists: {full_path} ({file_size} bytes)")
                            return True, preview_info, full_path
                        else:
                            print(f"❌ File not found: {full_path}")
                            return False, preview_info, full_path
                    except ImportError:
                        print("⚠️ Running outside ComfyUI - can't check file paths")
                        return True, preview_info, None
                else:
                    print("❌ No filename in preview data")
                    return False, None, None
            else:
                print("❌ No preview data in UI result")
                return False, None, None
        else:
            print("❌ No UI data in result")
            return False, None, None
            
    except Exception as e:
        print(f"❌ Video file creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_comfyui_video_serving(preview_info, comfyui_url="http://127.0.0.1:8188"):
    """Test if ComfyUI can serve the video file"""
    if not preview_info:
        print("❌ No preview info to test")
        return False
        
    print(f"\n🌐 Testing ComfyUI video serving...")
    print(f"   ComfyUI URL: {comfyui_url}")
    
    try:
        # Build the view URL
        params = {
            "filename": preview_info.get("filename"),
            "type": preview_info.get("type", "output"),
            "subfolder": preview_info.get("subfolder", "")
        }
        
        # Remove empty subfolder
        if not params["subfolder"]:
            del params["subfolder"]
        
        view_url = f"{comfyui_url}/view?{urlencode(params)}"
        print(f"   View URL: {view_url}")
        
        # Try to access the video
        response = requests.head(view_url, timeout=5)
        
        if response.status_code == 200:
            print(f"✅ Video accessible via ComfyUI")
            print(f"   Status: {response.status_code}")
            print(f"   Content-Type: {response.headers.get('Content-Type', 'unknown')}")
            print(f"   Content-Length: {response.headers.get('Content-Length', 'unknown')} bytes")
            return True
        else:
            print(f"❌ Video not accessible")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text[:200] if hasattr(response, 'text') else 'N/A'}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("⚠️ ComfyUI not running - can't test video serving")
        print("   Start ComfyUI and try again to test video serving")
        return None
    except Exception as e:
        print(f"❌ Video serving test failed: {e}")
        return False

def check_comfyui_preview_format():
    """Check what the standard ComfyUI preview format should be"""
    print(f"\n🔍 Checking ComfyUI preview format requirements...")
    
    # Standard ComfyUI preview format for images
    standard_image_format = {
        "filename": "example.png",
        "subfolder": "",
        "type": "output"
    }
    
    # What we're using for videos
    our_video_format = {
        "filename": "example.mp4",
        "subfolder": "",
        "type": "output",
        "format": "video/mp4",
        "frame_rate": 24.0,
        "frame_count": 100,
        "width": 1920,
        "height": 1080,
        "duration": 4.17
    }
    
    print(f"📋 Standard image format: {standard_image_format}")
    print(f"📋 Our video format: {our_video_format}")
    
    print(f"\n💡 Key differences:")
    print(f"   - We add 'format' field for MIME type")
    print(f"   - We add video metadata (fps, frames, dimensions)")
    print(f"   - ComfyUI may not recognize video files in 'gifs' array")
    
    return our_video_format

def suggest_fixes():
    """Suggest potential fixes for the preview issue"""
    print(f"\n🔧 POTENTIAL FIXES:")
    
    print(f"\n1. **Change UI return format**:")
    print("   Instead of: {'ui': {'gifs': [preview]}}")
    print("   Try: {'ui': {'videos': [preview]}}")
    print("   Or: {'ui': {'images': [preview]}}")
    
    print("\n2. **Use different file extension**:")
    print("   ComfyUI might expect .gif for 'gifs' array")
    print("   Try saving as animated GIF instead of MP4")
    
    print(f"\n3. **Add JavaScript widget detection**:")
    print(f"   ComfyUI might need custom JS to handle video previews")
    print(f"   Our raj-video-preview.js should handle this")
    
    print(f"\n4. **Check ComfyUI video support**:")
    print(f"   ComfyUI core might not support video previews natively")
    print(f"   May need VideoHelperSuite-style implementation")
    
    print(f"\n5. **Verify file permissions**:")
    print(f"   ComfyUI web server must have read access to video files")
    print(f"   Check output/temp directory permissions")

def main():
    """Run all tests"""
    print("🚀 TESTING VIDEO PREVIEW SERVING")
    print("=" * 50)
    
    # Test 1: Create video file
    success, preview_info, file_path = test_video_file_creation()
    
    if success:
        print(f"✅ Video file creation: PASSED")
        
        # Test 2: Test ComfyUI serving (if ComfyUI is running)
        serving_result = test_comfyui_video_serving(preview_info)
        
        if serving_result is True:
            print(f"✅ ComfyUI video serving: PASSED")
            print(f"\n🎉 Video files are being created and served correctly!")
            print(f"   The issue may be in the JavaScript preview widget")
            print(f"   or ComfyUI's native preview handling")
        elif serving_result is False:
            print(f"❌ ComfyUI video serving: FAILED")
            print(f"\n⚠️ Video files created but ComfyUI can't serve them")
        else:
            print(f"⚠️ ComfyUI video serving: SKIPPED (ComfyUI not running)")
    else:
        print(f"❌ Video file creation: FAILED")
    
    # Test 3: Check format requirements
    check_comfyui_preview_format()
    
    # Show potential fixes
    suggest_fixes()
    
    print(f"\n" + "=" * 50)
    print(f"🏁 DIAGNOSIS COMPLETE")
    
    if success and serving_result is True:
        print(f"🔍 ISSUE: Likely JavaScript preview widget or ComfyUI native video support")
        print(f"📋 NEXT STEPS:")
        print(f"   1. Check browser console for JavaScript errors")
        print(f"   2. Verify raj-video-preview.js is loading")
        print(f"   3. Test with a simple GIF format instead of MP4")
        print(f"   4. Compare with VideoHelperSuite preview format")
    elif success and serving_result is False:
        print(f"🔍 ISSUE: ComfyUI cannot serve video files")
        print(f"📋 NEXT STEPS:")
        print(f"   1. Check ComfyUI file serving permissions")
        print(f"   2. Verify output/temp directory access")
        print(f"   3. Test with image files to confirm serving works")
    else:
        print(f"🔍 ISSUE: Video file creation failed")
        print(f"📋 NEXT STEPS:")
        print(f"   1. Check video encoding dependencies (FFmpeg, OpenCV)")
        print(f"   2. Verify write permissions to output directory")
        print(f"   3. Test with simpler video format")

if __name__ == "__main__":
    main()