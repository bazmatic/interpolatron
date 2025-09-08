#!/usr/bin/env python3
"""
WebP to MP4 Converter

Converts WebP files (including animated) to MP4 format using PIL/Pillow
as a fallback when FFmpeg fails to decode the WebP file.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple
import argparse

try:
    from PIL import Image, ImageSequence
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/Pillow not available. Install with: pip install Pillow")

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")


def convert_webp_with_pil(input_path: str, output_path: str, fps: int = 25) -> bool:
    """
    Convert WebP to MP4 using PIL/Pillow.
    
    Args:
        input_path: Path to input WebP file
        output_path: Path to output MP4 file
        fps: Target frame rate
        
    Returns:
        True if successful, False otherwise
    """
    if not PIL_AVAILABLE:
        return False
    
    try:
        # Open the WebP file
        with Image.open(input_path) as img:
            # Check if it's animated
            if hasattr(img, 'n_frames') and img.n_frames > 1:
                print(f"Processing animated WebP with {img.n_frames} frames")
                
                # Get frame duration (default to 1/fps if not available)
                try:
                    duration = img.info.get('duration', 1000 // fps)  # duration in ms
                    frame_delay = duration / 1000.0  # convert to seconds
                except:
                    frame_delay = 1.0 / fps
                
                # Extract all frames
                frames = []
                for frame in ImageSequence.Iterator(img):
                    # Convert to RGB if necessary
                    if frame.mode != 'RGB':
                        frame = frame.convert('RGB')
                    frames.append(frame)
                
                if len(frames) == 0:
                    print("No frames found in WebP file")
                    return False
                
                # Use FFmpeg to create video from frames
                temp_dir = Path("temp_webp_frames")
                temp_dir.mkdir(exist_ok=True)
                
                # Save frames as PNG
                for i, frame in enumerate(frames):
                    frame_path = temp_dir / f"frame_{i:04d}.png"
                    frame.save(frame_path, "PNG")
                
                # Create video from frames using FFmpeg
                cmd = [
                    "ffmpeg",
                    "-framerate", str(1.0 / frame_delay),
                    "-i", str(temp_dir / "frame_%04d.png"),
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-y",
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Cleanup
                import shutil
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                
                if result.returncode == 0:
                    print(f"Successfully converted WebP to MP4: {output_path}")
                    return True
                else:
                    print(f"FFmpeg failed: {result.stderr}")
                    return False
                    
            else:
                # Static image - create a short video
                print("Processing static WebP image")
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as temporary PNG
                temp_png = "temp_webp_frame.png"
                img.save(temp_png, "PNG")
                
                # Create a 1-second video
                cmd = [
                    "ffmpeg",
                    "-loop", "1",
                    "-i", temp_png,
                    "-t", "1",
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-y",
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Cleanup
                if os.path.exists(temp_png):
                    os.remove(temp_png)
                
                if result.returncode == 0:
                    print(f"Successfully converted static WebP to MP4: {output_path}")
                    return True
                else:
                    print(f"FFmpeg failed: {result.stderr}")
                    return False
                    
    except Exception as e:
        print(f"Error processing WebP with PIL: {e}")
        return False


def convert_webp_with_opencv(input_path: str, output_path: str, fps: int = 25) -> bool:
    """
    Convert WebP to MP4 using OpenCV.
    
    Args:
        input_path: Path to input WebP file
        output_path: Path to output MP4 file
        fps: Target frame rate
        
    Returns:
        True if successful, False otherwise
    """
    if not OPENCV_AVAILABLE:
        return False
    
    try:
        # Open the WebP file
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print("OpenCV could not open WebP file")
            return False
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if frame_count <= 0:
            print("No frames found in WebP file")
            cap.release()
            return False
        
        print(f"Processing WebP with {frame_count} frames, {width}x{height}")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("Could not create output video writer")
            cap.release()
            return False
        
        # Read and write frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 10 == 0:
                print(f"Processed {frame_idx}/{frame_count} frames")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"Successfully converted WebP to MP4: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing WebP with OpenCV: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Convert WebP files to MP4")
    parser.add_argument("-i", "--input", required=True, help="Input WebP file")
    parser.add_argument("-o", "--output", required=True, help="Output MP4 file")
    parser.add_argument("-f", "--fps", type=int, default=25, help="Target frame rate (default: 25)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Converting {args.input} to {args.output}")
    print(f"Target FPS: {args.fps}")
    
    # Try PIL first (better for animated WebP)
    if PIL_AVAILABLE:
        print("Attempting conversion with PIL/Pillow...")
        if convert_webp_with_pil(args.input, args.output, args.fps):
            print("✓ Conversion successful!")
            sys.exit(0)
    
    # Try OpenCV as fallback
    if OPENCV_AVAILABLE:
        print("Attempting conversion with OpenCV...")
        if convert_webp_with_opencv(args.input, args.output, args.fps):
            print("✓ Conversion successful!")
            sys.exit(0)
    
    # If both fail, try direct FFmpeg as last resort
    print("Attempting direct FFmpeg conversion...")
    cmd = [
        "ffmpeg",
        "-i", args.input,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-y",
        args.output
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Direct FFmpeg conversion successful!")
        sys.exit(0)
    else:
        print("✗ All conversion methods failed!")
        print(f"FFmpeg error: {result.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main() 