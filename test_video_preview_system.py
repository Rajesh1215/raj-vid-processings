#!/usr/bin/env python3
"""
Test the complete video preview system implementation
Tests both the Python nodes and JavaScript widget integration
"""

import torch
import numpy as np
import sys
import os

def test_video_preview_node():
    """Test the RajVideoPreview node functionality"""
    print("🎬 Testing RajVideoPreview node...")
    
    try:
        # Import the node
        sys.path.append('.')
        from nodes.video_preview import RajVideoPreview
        
        # Create test video frames (30 frames of 320x240 RGB)
        frames = torch.rand(30, 240, 320, 3).float()  # 30 frames at 1 second duration
        fps = 30.0
        
        # Create test audio data
        sample_rate = 22050
        duration = 1.0
        audio_samples = int(duration * sample_rate)
        audio_data = torch.sin(2 * torch.pi * 440 * torch.linspace(0, duration, audio_samples)).unsqueeze(1)
        
        audio = {
            "waveform": audio_data.unsqueeze(0),  # Add batch dimension
            "sample_rate": sample_rate
        }
        
        print(f"✅ Test data created:")
        print(f"   Video: {frames.shape} at {fps}fps")
        print(f"   Audio: {audio_data.shape} at {sample_rate}Hz")
        
        # Test the preview node
        preview_node = RajVideoPreview()
        result = preview_node.create_preview(
            frames=frames,
            fps=fps,
            audio=audio,
            preview_format="mp4",
            preview_quality="medium",
            max_preview_duration=30.0,
            auto_play=False
        )
        
        # Check results
        if "ui" in result and "result" in result:
            ui_data = result["ui"]
            result_data = result["result"]
            
            print(f"✅ Preview node execution successful!")
            print(f"   UI data keys: {list(ui_data.keys())}")
            if "gifs" in ui_data and len(ui_data["gifs"]) > 0:
                preview_info = ui_data["gifs"][0]
                print(f"   Preview info: {preview_info}")
                print(f"   Filename: {preview_info.get('filename', 'N/A')}")
                print(f"   Format: {preview_info.get('format', 'N/A')}")
                print(f"   Frame rate: {preview_info.get('frame_rate', 'N/A')}")
                print(f"   Frame count: {preview_info.get('frame_count', 'N/A')}")
                
                # Check if preview file was created
                if 'filename' in preview_info:
                    print(f"   File created: ✅")
                else:
                    print(f"   File created: ❌")
            
            print(f"   Result data: {len(result_data)} items")
            return True
        else:
            print(f"❌ Invalid result format: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Preview node test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_video_saver_preview():
    """Test the updated RajVideoSaver preview functionality"""
    print("\n💾 Testing RajVideoSaver preview...")
    
    try:
        from nodes.video_saver import RajVideoSaver
        
        # Create test frames
        frames = torch.rand(15, 180, 320, 3).float()  # 15 frames
        
        saver_node = RajVideoSaver()
        result = saver_node.save_video(
            frames=frames,
            filename="test_preview_output",
            fps=15.0,
            format="mp4",
            quality=25,
            save_to_output=False,  # Save to temp directory for testing
            add_timestamp=True
        )
        
        if "ui" in result and "gifs" in result["ui"]:
            preview_data = result["ui"]["gifs"][0]
            print(f"✅ Video saver preview successful!")
            print(f"   Preview data: {preview_data}")
            print(f"   Enhanced metadata: {preview_data.get('width', 'N/A')}x{preview_data.get('height', 'N/A')}")
            print(f"   Duration: {preview_data.get('duration', 'N/A')}s")
            print(f"   File size: {preview_data.get('file_size_mb', 'N/A')}MB")
            return True
        else:
            print(f"❌ Video saver preview failed - no UI data")
            return False
            
    except Exception as e:
        print(f"❌ Video saver test failed: {e}")
        return False

def test_video_upload_preview():
    """Test the updated RajVideoUpload preview functionality"""
    print("\n📤 Testing RajVideoUpload preview integration...")
    
    try:
        from nodes.video_upload import RajVideoUpload
        
        # This would normally test with an actual video file
        # For now, just verify the class structure and methods exist
        upload_node = RajVideoUpload()
        
        # Check if the class has the required methods
        if hasattr(upload_node, 'upload_and_load_video'):
            print(f"✅ Upload node structure valid")
            
            # Check INPUT_TYPES has video_upload setting
            input_types = upload_node.INPUT_TYPES()
            if 'required' in input_types and 'video' in input_types['required']:
                video_config = input_types['required']['video'][1]
                if video_config.get('video_upload') == True:
                    print(f"✅ Upload widget configuration valid")
                    return True
                else:
                    print(f"❌ Video upload setting missing")
                    return False
            else:
                print(f"❌ Video input configuration invalid")
                return False
        else:
            print(f"❌ Upload node missing required methods")
            return False
            
    except Exception as e:
        print(f"❌ Video upload test failed: {e}")
        return False

def test_javascript_integration():
    """Test JavaScript file existence and basic structure"""
    print("\n🎯 Testing JavaScript integration...")
    
    js_file = "web/js/raj-video-preview.js"
    
    if os.path.exists(js_file):
        print(f"✅ JavaScript preview widget exists: {js_file}")
        
        # Read and check basic structure
        with open(js_file, 'r') as f:
            content = f.read()
            
        required_elements = [
            "RajVideoPreviewWidget",
            "app.registerExtension",
            "nodeCreated",
            "updateVideoPreview",
            "video_preview"
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if not missing_elements:
            print(f"✅ JavaScript widget structure complete")
            print(f"   Contains: {', '.join(required_elements)}")
            return True
        else:
            print(f"❌ JavaScript widget missing: {', '.join(missing_elements)}")
            return False
    else:
        print(f"❌ JavaScript preview widget not found: {js_file}")
        return False

def test_node_registration():
    """Test that new nodes are properly registered"""
    print("\n📋 Testing node registration...")
    
    try:
        # Import the main module to trigger registration
        import __init__
        
        # Check if NODE_CLASS_MAPPINGS contains our new nodes
        if hasattr(__init__, 'NODE_CLASS_MAPPINGS'):
            mappings = __init__.NODE_CLASS_MAPPINGS
            
            expected_nodes = [
                "RajVideoPreview",
                "RajVideoPreviewAdvanced"
            ]
            
            missing_nodes = []
            for node in expected_nodes:
                if node not in mappings:
                    missing_nodes.append(node)
            
            if not missing_nodes:
                print(f"✅ All new nodes registered successfully")
                print(f"   Registered: {', '.join(expected_nodes)}")
                return True
            else:
                print(f"❌ Missing node registrations: {', '.join(missing_nodes)}")
                return False
        else:
            print(f"❌ NODE_CLASS_MAPPINGS not found")
            return False
            
    except Exception as e:
        print(f"❌ Node registration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 TESTING RAJ VIDEO PREVIEW SYSTEM")
    print("=" * 50)
    
    tests = [
        ("Video Preview Node", test_video_preview_node),
        ("Video Saver Preview", test_video_saver_preview), 
        ("Video Upload Preview", test_video_upload_preview),
        ("JavaScript Integration", test_javascript_integration),
        ("Node Registration", test_node_registration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"💥 {test_name}: CRASHED - {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 VIDEO PREVIEW SYSTEM TEST RESULTS:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("🎬 Video preview system is ready for ComfyUI!")
        print("\n📋 NEXT STEPS:")
        print("1. Restart ComfyUI to load the new nodes")
        print("2. Look for '👁️ Raj Video Preview' in the node menu")
        print("3. Connect IMAGE frames (and optional AUDIO) to see the preview")
        print("4. Video saver and upload nodes now show enhanced previews")
    else:
        print(f"\n⚠️ {failed} test(s) failed - please check the implementation")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)