#!/usr/bin/env python3
"""
Test script for the Video Interpolation Tool

This script creates a simple test video and runs interpolation on it
to verify that the tool works correctly.
"""

import subprocess
import sys
import os
from pathlib import Path


def create_test_video(output_path: str = "test_input.mp4", duration: int = 5) -> bool:
    """
    Create a simple test video using FFmpeg.
    
    Args:
        output_path: Path for the test video
        duration: Duration in seconds
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Creating test video: {output_path}")
    
    # Create a simple test video with moving text
    cmd = [
        "ffmpeg",
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration}:size=640x480:rate=30",
        "-f", "lavfi",
        "-i", f"sine=frequency=1000:duration={duration}",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        "-y",
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Test video created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to create test video: {e.stderr}")
        return False


def test_video_info() -> bool:
    """Test the video information functionality."""
    print("\n--- Testing Video Information ---")
    
    if not os.path.exists("test_input.mp4"):
        print("✗ Test video not found, creating one...")
        if not create_test_video():
            return False
    
    cmd = ["python", "video_interpolator.py", "-i", "test_input.mp4", "--info"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Video information test passed")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Video information test failed: {e.stderr}")
        return False


def test_frame_duplication() -> bool:
    """Test frame duplication interpolation."""
    print("\n--- Testing Frame Duplication ---")
    
    if not os.path.exists("test_input.mp4"):
        print("✗ Test video not found")
        return False
    
    cmd = [
        "python", "video_interpolator.py",
        "-i", "test_input.mp4",
        "-o", "test_output_duplication.mp4",
        "-m", "frame_duplication",
        "-f", "60"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Frame duplication test passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Frame duplication test failed: {e.stderr}")
        return False


def test_temporal_interpolation() -> bool:
    """Test temporal interpolation."""
    print("\n--- Testing Temporal Interpolation ---")
    
    if not os.path.exists("test_input.mp4"):
        print("✗ Test video not found")
        return False
    
    cmd = [
        "python", "video_interpolator.py",
        "-i", "test_input.mp4",
        "-o", "test_output_temporal.mp4",
        "-m", "temporal",
        "-f", "60"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ Temporal interpolation test passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Temporal interpolation test failed: {e.stderr}")
        return False


def cleanup_test_files() -> None:
    """Clean up test files."""
    test_files = [
        "test_input.mp4",
        "test_output_duplication.mp4",
        "test_output_temporal.mp4"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")


def main() -> None:
    """Run all tests."""
    print("Video Interpolation Tool - Test Suite")
    print("=" * 50)
    
    # Check if FFmpeg is available
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        print("✓ FFmpeg is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ FFmpeg not found. Please install FFmpeg first.")
        sys.exit(1)
    
    # Check if the main script exists
    if not os.path.exists("video_interpolator.py"):
        print("✗ video_interpolator.py not found")
        sys.exit(1)
    
    print("✓ video_interpolator.py found")
    
    # Run tests
    tests = [
        ("Video Information", test_video_info),
        ("Frame Duplication", test_frame_duplication),
        ("Temporal Interpolation", test_temporal_interpolation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name} test failed")
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The video interpolation tool is working correctly.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    # Ask if user wants to clean up test files
    if passed > 0:
        response = input("\nDo you want to clean up test files? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            cleanup_test_files()
            print("Test files cleaned up.")
        else:
            print("Test files preserved for manual inspection.")


if __name__ == "__main__":
    main() 