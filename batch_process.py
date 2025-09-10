#!/usr/bin/env python3
"""
Batch Video Processing Script

Processes all videos in the input folder
and saves the results to the output folder.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any


def get_video_files(input_dir: str) -> List[str]:
    """
    Get all video files from the input directory.
    
    Args:
        input_dir: Path to input directory
        
    Returns:
        List of video file paths
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    video_files = []
    
    if not os.path.exists(input_dir):
        print(f"✗ Input directory not found: {input_dir}")
        return []
    
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path):
            file_ext = Path(file).suffix.lower()
            if file_ext in video_extensions:
                video_files.append(file_path)
    
    return sorted(video_files)


def get_video_fps(video_path: str, interpolator_script: str = "video_interpolator.py") -> float:
    """
    Get the FPS of a video file using ffprobe.

    Args:
        video_path: Path to video file
        interpolator_script: Path to the video interpolator script

    Returns:
        FPS as float, or 30.0 as fallback
    """
    try:
        # Use ffprobe directly for better reliability
        cmd = ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        fps_str = result.stdout.strip()

        # Parse fps (format is usually "num/den" like "30/1")
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        return fps if fps > 0 else 30.0
    except (subprocess.CalledProcessError, ValueError, IndexError):
        print(f"   ⚠️  Could not determine FPS for {video_path}, using default 30.0")
        return 30.0


def process_video(input_path: str, output_path: str, speed: float, interpolation_method: str, interpolator_script: str = "video_interpolator.py", target_fps: float = None) -> bool:
    """
    Process a single video with specified speed and interpolation method.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        speed: Speed multiplier (e.g., 0.5 for slow motion, 2.0 for fast motion)
        interpolation_method: Interpolation method to use
        interpolator_script: Path to the video interpolator script
        target_fps: Target FPS for interpolation (if None, auto-calculate for optimal interpolation)

    Returns:
        True if successful, False otherwise
    """
    print(f"\n🎬 Processing: {os.path.basename(input_path)}")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Speed: {speed}x")
    print(f"   Method: {interpolation_method}")

    # Command: Custom speed with specified interpolation method (no audio) and frame removal
    # For TLBVFI and advanced methods, use direct interpolation without speed adjustment
    if interpolation_method in ['tlbvfi', 'advanced']:
        # Get input video FPS for smart interpolation targeting
        input_fps = get_video_fps(input_path, interpolator_script)

        # Calculate optimal target FPS: double the input FPS for minimal interpolation
        # This adds one interpolated frame between each original frame
        if target_fps is None:
            target_fps = input_fps * 2
            print(f"   📊 Input FPS: {input_fps:.1f}, Target FPS: {target_fps:.1f} (1 interpolated frame between originals)")
        else:
            print(f"   📊 Input FPS: {input_fps:.1f}, Custom Target FPS: {target_fps:.1f}")

        cmd = [
            "python", interpolator_script,
            "-i", input_path,
            "-o", output_path,
            "--method", interpolation_method,
            "--fps", str(int(target_fps)),  # Convert to int for cleaner output
            "--no-audio",
            "--remove-frames", "3"
        ]

    else:
        # For traditional methods, use --interpolation with --speed
        cmd = [
            "python", interpolator_script,
            "-i", input_path,
            "-o", output_path,
            "--speed", str(speed),
            "--interpolation", interpolation_method,
            "--no-audio",
            "--remove-frames", "3"
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"   ✓ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Failed: {e.stderr}")
        return False


def call_video_info_script(video_path: str, interpolator_script: str = "video_interpolator.py") -> Dict[str, Any]:
    """
    Call the video interpolator script to get video information.
    
    Args:
        video_path: Path to video file
        interpolator_script: Path to the video interpolator script
        
    Returns:
        Dictionary with video information from the script
    """
    cmd = ["python", interpolator_script, "-i", video_path, "--info"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"success": True, "info": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": e.stderr}


def main() -> None:
    """Main function to process all videos in batch."""
    print("🎬 Batch Video Processing - Smart Interpolation with Minimal Artifacts")
    print("   AI Methods: Smart FPS targeting (double input FPS by default)")
    print("   Traditional Methods: Custom speed with interpolation")
    print("=" * 60)
    
    # Configuration
    input_dir = "input"
    output_dir = "output"
    interpolator_script = "video_interpolator.py"
    
    # Check if interpolator script exists
    if not os.path.exists(interpolator_script):
        print(f"✗ Video interpolator script not found: {interpolator_script}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created output directory: {output_dir}")
    
    # Get all video files
    video_files = get_video_files(input_dir)
    
    if not video_files:
        print(f"✗ No video files found in {input_dir}")
        sys.exit(1)
    
    print(f"📁 Found {len(video_files)} video files in {input_dir}")
    print(f"📁 Output will be saved to {output_dir}")
    
    # Ask user for speed setting
    while True:
        try:
            speed_input = input("\n⚙️  Enter speed multiplier (e.g., 0.5 for slow motion, 1.0 for normal, 2.0 for fast): ")
            speed = float(speed_input)
            if speed <= 0:
                print("❌ Speed must be greater than 0. Please try again.")
                continue
            break
        except ValueError:
            print("❌ Invalid input. Please enter a valid number.")
    
    # Ask user for interpolation method
    print("\n🎯 Available interpolation methods:")
    print("   1. frame_duplication - Fastest, duplicates frames")
    print("   2. temporal - Balanced quality using minterpolate filter")
    print("   3. optical_flow - High quality using optical flow")
    print("   4. advanced - Smart FPS targeting, enhanced single-pass (minimal artifacts)")
    print("   5. tlbvfi - AI-powered, state-of-the-art quality with smart FPS (requires PyTorch)")
    
    while True:
        method_input = input("\n⚙️  Choose interpolation method (1-5) or enter method name: ").strip()
        
        # Handle numeric input
        if method_input in ['1', '2', '3', '4', '5']:
            method_map = {
                '1': 'frame_duplication',
                '2': 'temporal', 
                '3': 'optical_flow',
                '4': 'advanced',
                '5': 'tlbvfi'
            }
            interpolation_method = method_map[method_input]
            break
        
        # Handle direct method name input
        elif method_input in ['frame_duplication', 'temporal', 'optical_flow', 'advanced', 'tlbvfi']:
            interpolation_method = method_input
            break
        
        else:
            print("❌ Invalid input. Please enter 1-5 or a valid method name.")
    
    # Special note for TLBVFI
    if interpolation_method == 'tlbvfi':
        print("\n🤖 TLBVFI Selected - AI-Powered Interpolation")
        print("   Note: This method requires PyTorch and may take longer to process.")
        print("   GPU acceleration will be used if available.")
        print("   TLBVFI will use smart FPS targeting (no speed adjustment).")
        print("   If TLBVFI fails, the system will fall back to temporal interpolation.")

    # Ask for custom target FPS for AI methods
    target_fps = None
    if interpolation_method in ['tlbvfi', 'advanced']:
        print("\n🎯 FPS Configuration for AI Methods")
        print("   Default: Smart targeting (double input FPS for minimal interpolation)")
        print("   Example: 8 FPS input → 16 FPS output (one interpolated frame between originals)")

        fps_input = input("   Enter custom target FPS (or press Enter for smart default): ").strip()
        if fps_input:
            try:
                target_fps = float(fps_input)
                if target_fps <= 0:
                    print("   ⚠️  Invalid FPS, using smart default")
                    target_fps = None
                else:
                    print(f"   ✓ Custom target FPS set: {target_fps}")
            except ValueError:
                print("   ⚠️  Invalid input, using smart default")
                target_fps = None

    if interpolation_method in ['tlbvfi', 'advanced']:
        if target_fps:
            print(f"⚙️  Processing settings: {interpolation_method} interpolation to {target_fps:.1f}fps (no audio, remove first 3 frames)")
        else:
            print(f"⚙️  Processing settings: {interpolation_method} interpolation with smart FPS targeting (no audio, remove first 3 frames)")
    else:
        print(f"⚙️  Processing settings: {speed}x speed with {interpolation_method} interpolation (no audio, remove first 3 frames)")
    print("-" * 60)
    
    # Process each video
    successful = 0
    failed = 0
    results = []
    
    for i, input_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing...")
        
        # Create output filename
        input_filename = os.path.basename(input_path)
        name, ext = os.path.splitext(input_filename)

        # Include target FPS in filename for AI methods
        if interpolation_method in ['tlbvfi', 'advanced'] and target_fps:
            output_filename = f"{name}_{int(target_fps)}fps_{interpolation_method}{ext}"
        else:
            output_filename = f"{name}_{speed}x_{interpolation_method}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Process the video
        success = process_video(input_path, output_path, speed, interpolation_method, interpolator_script, target_fps)
        
        if success:
            successful += 1
            results.append({
                "input": input_filename,
                "output": output_filename,
                "status": "success"
            })
        else:
            failed += 1
            results.append({
                "input": input_filename,
                "output": output_filename,
                "status": "failed"
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total videos: {len(video_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful/len(video_files)*100):.1f}%")
    
    if successful > 0:
        print(f"\n✅ Successfully processed videos saved to: {output_dir}")
        
        # Show some sample results
        print("\n📋 Sample Results:")
        for result in results[:5]:  # Show first 5 results
            status_icon = "✓" if result["status"] == "success" else "✗"
            print(f"   {status_icon} {result['input']} → {result['output']}")
        
        if len(results) > 5:
            print(f"   ... and {len(results) - 5} more")
    
    if failed > 0:
        print(f"\n❌ Failed videos:")
        for result in results:
            if result["status"] == "failed":
                print(f"   ✗ {result['input']}")
    
    print("\n🎉 Batch processing completed!")


if __name__ == "__main__":
    main() 