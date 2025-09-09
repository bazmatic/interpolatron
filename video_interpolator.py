#!/usr/bin/env python3
"""
Video Interpolation Tool

A command line tool for interpolating video frames using FFmpeg.
Supports multiple interpolation methods including frame duplication, 
temporal interpolation, and optical flow-based interpolation.
Also supports video speed adjustment (slow motion and fast motion).

Usage:
    python video_interpolator.py --input input.mp4 --output output.mp4 --method frame_duplication
    python video_interpolator.py --input input.mp4 --output output.mp4 --method temporal --fps 60
    python video_interpolator.py --input input.mp4 --output output.mp4 --method optical_flow --fps 60
    python video_interpolator.py --input input.mp4 --output output.mp4 --speed 0.5  # Slow down to 50%
    python video_interpolator.py --input input.mp4 --output output.mp4 --speed 2.0  # Speed up to 200%
"""

import argparse
import subprocess
import sys
import os
from typing import Optional, List, Dict, Any
from pathlib import Path

# TLBVFI integration
try:
    from tlbvfi_wrapper import TLBVFIWrapper
    TLBVFI_AVAILABLE = True
except ImportError as e:
    TLBVFI_AVAILABLE = False


class VideoInterpolator:
    """Video interpolation and speed adjustment tool using FFmpeg."""
    
    def __init__(self, ffmpeg_path: str = "ffmpeg", tlbvfi_model_path: str = "model/vimeo_unet.pth"):
        """
        Initialize the video interpolator.
        
        Args:
            ffmpeg_path: Path to FFmpeg executable
            tlbvfi_model_path: Path to TLBVFI model file
        """
        self.ffmpeg_path = ffmpeg_path
        self.tlbvfi_model_path = tlbvfi_model_path
        self.tlbvfi_wrapper = None
        self._check_ffmpeg()
    
    def _check_ffmpeg(self) -> None:
        """Check if FFmpeg is available."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ FFmpeg found: {result.stdout.split()[2]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ FFmpeg not found. Please install FFmpeg first.")
            print("  Download from: https://ffmpeg.org/download.html")
            sys.exit(1)
    
    def get_video_info(self, input_path: str) -> Dict[str, Any]:
        """
        Get video information using FFmpeg.
        
        Args:
            input_path: Path to input video file
            
        Returns:
            Dictionary containing video information
        """
        cmd = [
            self.ffmpeg_path,
            "-i", input_path,
            "-f", "null",
            "-"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            # Parse FFmpeg output to extract video info
            output = result.stderr
            
            # Extract duration
            duration_match = None
            for line in output.split('\n'):
                if "Duration:" in line:
                    duration_match = line.split("Duration:")[1].split(",")[0].strip()
                    break
            
            # Extract frame rate
            fps_match = None
            for line in output.split('\n'):
                if "fps" in line and "Video:" in line:
                    fps_parts = line.split("fps")[0].split()
                    fps_match = fps_parts[-1]
                    break
            
            return {
                "duration": duration_match,
                "fps": fps_match,
                "raw_output": output
            }
            
        except Exception as e:
            print(f"Error getting video info: {e}")
            return {}
    
    def adjust_speed(self, input_path: str, output_path: str, speed_factor: float, 
                    preserve_audio: bool = True, interpolation_method: Optional[str] = None, 
                    remove_frames: int = 0) -> bool:
        """
        Adjust video speed (slow down or speed up).
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            speed_factor: Speed factor (0.1 = 10% speed, 2.0 = 200% speed)
            preserve_audio: Whether to preserve audio (pitch-corrected)
            interpolation_method: Optional interpolation method for smooth slow motion
            
        Returns:
            True if successful, False otherwise
        """
        if speed_factor <= 0:
            print("✗ Speed factor must be positive")
            return False
        
        # Determine if we're slowing down or speeding up
        is_slow_motion = speed_factor < 1.0
        
        # Use interpolation for both slow motion and fast motion to ensure 60fps
        if interpolation_method:
            if is_slow_motion:
                # For slow motion, we can use interpolatio
                # n for smoother results
                return self._slow_motion_with_interpolation(
                    input_path, output_path, speed_factor, preserve_audio, interpolation_method, remove_frames
                )
            else:
                # For fast motion, interpolate to 60fps first, then adjust speed
                return self._fast_motion_with_interpolation(
                    input_path, output_path, speed_factor, preserve_audio, interpolation_method, remove_frames
                )
        else:
            # Standard speed adjustment (for normal speed or fast motion)
            # Interpolation is not used for speeding up videos
            return self._standard_speed_adjustment(input_path, output_path, speed_factor, preserve_audio, remove_frames)
    
    def _standard_speed_adjustment(self, input_path: str, output_path: str, 
                                 speed_factor: float, preserve_audio: bool, remove_frames: int = 0) -> bool:
        """Perform standard speed adjustment."""
        # Create video filter with frame removal if needed
        video_filter = f"setpts={1/speed_factor}*PTS"
        if remove_frames > 0:
            video_filter = f"select='gt(n,{remove_frames-1})',{video_filter}"
        
        if preserve_audio:
            # Try with audio first
            audio_filter = self._create_audio_filter(speed_factor)
            
            cmd = [
                self.ffmpeg_path,
                "-i", input_path,
                "-filter:v", video_filter,
                "-filter:a", audio_filter,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "medium",
                "-crf", "23",
                "-y",
                output_path
            ]
            
            operation_name = f"Speed adjustment with audio ({speed_factor}x)"
            success = self._run_ffmpeg_command(cmd, operation_name)
            
            # If audio processing fails, try without audio
            if not success:
                print(f"Audio processing failed for speed {speed_factor}x, trying without audio...")
                return self._standard_speed_adjustment(input_path, output_path, speed_factor, preserve_audio=False)
            
            return success
        else:
            # No audio preservation
            cmd = [
                self.ffmpeg_path,
                "-i", input_path,
                "-filter:v", video_filter,
                "-an",  # No audio
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-y",
                output_path
            ]
            
            operation_name = f"Speed adjustment without audio ({speed_factor}x)"
        
            # Get input duration for validation
            input_info = self.get_video_info(input_path)
            expected_duration = None
            if input_info and 'duration' in input_info:
                try:
                    duration_str = input_info['duration']
                    time_parts = duration_str.split(':')
                    if len(time_parts) == 3:
                        hours = int(time_parts[0])
                        minutes = int(time_parts[1])
                        seconds = float(time_parts[2])
                        input_duration = hours * 3600 + minutes * 60 + seconds
                        expected_duration = input_duration / speed_factor
                except (ValueError, IndexError):
                    pass
            
            return self._run_ffmpeg_command(cmd, operation_name, expected_duration=expected_duration, output_path=output_path)
    
    def _create_audio_filter(self, speed_factor: float) -> str:
        """Create audio filter for speed adjustment."""
        if speed_factor <= 0:
            return "anull"  # No audio
        
        # FFmpeg atempo filter supports 0.5x to 2.0x
        if 0.5 <= speed_factor <= 2.0:
            return f"atempo={speed_factor}"
        
        # For speeds outside the supported range, we need to chain filters
        if speed_factor < 0.5:
            # For very slow speeds, chain atempo=0.5 filters
            atempo_chains = []
            remaining_factor = speed_factor
            while remaining_factor < 0.5:
                atempo_chains.append("atempo=0.5")
                remaining_factor /= 0.5
            if remaining_factor != 1.0 and remaining_factor > 0:
                atempo_chains.append(f"atempo={remaining_factor}")
            return ",".join(atempo_chains)
        
        elif speed_factor > 2.0:
            # For very fast speeds, chain atempo=2.0 filters
            atempo_chains = []
            remaining_factor = speed_factor
            while remaining_factor > 2.0:
                atempo_chains.append("atempo=2.0")
                remaining_factor /= 2.0
            if remaining_factor != 1.0 and remaining_factor > 0:
                atempo_chains.append(f"atempo={remaining_factor}")
            return ",".join(atempo_chains)
        
        return "anull"  # Fallback
    
    def _slow_motion_with_interpolation(self, input_path: str, output_path: str, 
                                      speed_factor: float, preserve_audio: bool, 
                                      interpolation_method: str, remove_frames: int = 0) -> bool:
        """Perform slow motion with frame interpolation for smoother results."""
        # First, get original frame rate
        info = self.get_video_info(input_path)
        original_fps = float(info.get('fps', 30))
        
        # Calculate target frame rate for smooth slow motion
        # We want to maintain smooth motion, so we increase frame rate
        target_fps = original_fps / speed_factor
        
        # Create temporary file with interpolated frames
        temp_file = f"temp_interpolated_{os.path.basename(input_path)}"
        
        # Perform interpolation first
        interpolation_success = False
        if interpolation_method == "frame_duplication":
            interpolation_success = self.frame_duplication(input_path, temp_file, int(target_fps))
        elif interpolation_method == "temporal":
            interpolation_success = self.temporal_interpolation(input_path, temp_file, int(target_fps))
        elif interpolation_method == "optical_flow":
            interpolation_success = self.optical_flow_interpolation(input_path, temp_file, int(target_fps))
        else:
            interpolation_success = self.temporal_interpolation(input_path, temp_file, int(target_fps))
        
        if not interpolation_success:
            print("✗ Interpolation failed, falling back to standard speed adjustment")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return self._standard_speed_adjustment(input_path, output_path, speed_factor, preserve_audio, remove_frames)
        
        # Now adjust speed of the interpolated video
        if preserve_audio:
            cmd = [
                self.ffmpeg_path,
                "-i", temp_file,
                "-filter:v", f"setpts={1/speed_factor}*PTS",
                "-filter:a", f"atempo={speed_factor}" if speed_factor >= 0.5 else "atempo=0.5",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "medium",
                "-crf", "23",
                "-y",
                output_path
            ]
        else:
            cmd = [
                self.ffmpeg_path,
                "-i", temp_file,
                "-filter:v", f"setpts={1/speed_factor}*PTS",
                "-an",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-y",
                output_path
            ]
        
        operation_name = f"Slow motion with {interpolation_method} interpolation ({speed_factor}x)"
        success = self._run_ffmpeg_command(cmd, operation_name)
        
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return success
    
    def _fast_motion_with_interpolation(self, input_path: str, output_path: str, 
                                      speed_factor: float, preserve_audio: bool, 
                                      interpolation_method: str, remove_frames: int = 0) -> bool:
        """Perform fast motion with frame interpolation for smoother results."""
        # First, get original frame rate
        info = self.get_video_info(input_path)
        original_fps = float(info.get('fps', 30))
        
        # For fast motion, we interpolate to 60fps first, then adjust speed
        target_fps = 60
        
        # Create temporary file with interpolated frames
        temp_file = f"temp_interpolated_{os.path.basename(input_path)}"
        
        # Perform interpolation first to get 60fps
        interpolation_success = False
        if interpolation_method == "frame_duplication":
            interpolation_success = self.frame_duplication(input_path, temp_file, target_fps)
        elif interpolation_method == "temporal":
            interpolation_success = self.temporal_interpolation(input_path, temp_file, target_fps)
        elif interpolation_method == "optical_flow":
            interpolation_success = self.optical_flow_interpolation(input_path, temp_file, target_fps)
        else:
            interpolation_success = self.temporal_interpolation(input_path, temp_file, target_fps)
        
        if not interpolation_success:
            print("✗ Interpolation failed, falling back to standard speed adjustment")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return self._standard_speed_adjustment(input_path, output_path, speed_factor, preserve_audio, remove_frames)
        
        # Now adjust speed of the interpolated video
        if preserve_audio:
            cmd = [
                self.ffmpeg_path,
                "-i", temp_file,
                "-filter:v", f"setpts={1/speed_factor}*PTS",
                "-filter:a", f"atempo={speed_factor}" if speed_factor <= 2.0 else "atempo=2.0",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "medium",
                "-crf", "23",
                "-y",
                output_path
            ]
        else:
            cmd = [
                self.ffmpeg_path,
                "-i", temp_file,
                "-filter:v", f"setpts={1/speed_factor}*PTS",
                "-an",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-y",
                output_path
            ]
        
        operation_name = f"Fast motion with {interpolation_method} interpolation ({speed_factor}x)"
        success = self._run_ffmpeg_command(cmd, operation_name)
        
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return success
    
    def frame_duplication(self, input_path: str, output_path: str, target_fps: int = 60) -> bool:
        """
        Perform frame duplication interpolation.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            target_fps: Target frame rate
            
        Returns:
            True if successful, False otherwise
        """
        cmd = [
            self.ffmpeg_path,
            "-i", input_path,
            "-filter:v", f"fps=fps={target_fps}:round=up",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-y",
            output_path
        ]
        
        return self._run_ffmpeg_command(cmd, "Frame duplication")
    
    def temporal_interpolation(self, input_path: str, output_path: str, target_fps: int = 60) -> bool:
        """
        Perform temporal interpolation using minterpolate filter.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            target_fps: Target frame rate
            
        Returns:
            True if successful, False otherwise
        """
        cmd = [
            self.ffmpeg_path,
            "-i", input_path,
            "-filter:v", f"minterpolate='fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1'",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-y",
            output_path
        ]
        
        return self._run_ffmpeg_command(cmd, "Temporal interpolation")
    
    def optical_flow_interpolation(self, input_path: str, output_path: str, target_fps: int = 60) -> bool:
        """
        Perform optical flow-based interpolation.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            target_fps: Target frame rate
            
        Returns:
            True if successful, False otherwise
        """
        cmd = [
            self.ffmpeg_path,
            "-i", input_path,
            "-filter:v", f"minterpolate='fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:search_param=32'",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "20",
            "-y",
            output_path
        ]
        
        return self._run_ffmpeg_command(cmd, "Optical flow interpolation")
    
    def tlbvfi_interpolation(self, input_path: str, output_path: str, target_fps: int = 60) -> bool:
        """
        Perform TLBVFI-based interpolation for high-quality frame interpolation.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            target_fps: Target frame rate
            
        Returns:
            True if successful, False otherwise
        """
        if not TLBVFI_AVAILABLE:
            print("✗ TLBVFI not available. Please install required dependencies:")
            print("  pip install torch torchvision opencv-python Pillow numpy")
            print("  Note: TLBVFI now supports CPU fallback - CuPy (CUDA) is optional for acceleration")
            return False
        
        try:
            # Initialize TLBVFI wrapper if not already done
            if self.tlbvfi_wrapper is None:
                self.tlbvfi_wrapper = TLBVFIWrapper(self.tlbvfi_model_path)
            
            print(f"Using TLBVFI interpolation (model: {self.tlbvfi_model_path})")
            return self.tlbvfi_wrapper.interpolate_video(input_path, output_path, target_fps)
            
        except Exception as e:
            print(f"✗ TLBVFI interpolation failed: {e}")
            return False
    
    def advanced_interpolation(self, input_path: str, output_path: str, target_fps: int = 60) -> bool:
        """
        Perform advanced interpolation with multiple passes.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            target_fps: Target frame rate
            
        Returns:
            True if successful, False otherwise
        """
        # First pass: extract frames
        temp_dir = Path("temp_frames")
        temp_dir.mkdir(exist_ok=True)
        
        extract_cmd = [
            self.ffmpeg_path,
            "-i", input_path,
            "-vf", "fps=1",
            "-frame_pts", "1",
            str(temp_dir / "frame_%d.png"),
            "-y"
        ]
        
        if not self._run_ffmpeg_command(extract_cmd, "Frame extraction", silent=True):
            return False
        
        # Second pass: interpolate
        interpolate_cmd = [
            self.ffmpeg_path,
            "-framerate", "1",
            "-i", str(temp_dir / "frame_%d.png"),
            "-filter:v", f"minterpolate='fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1'",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-y",
            output_path
        ]
        
        success = self._run_ffmpeg_command(interpolate_cmd, "Advanced interpolation")
        
        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        return success
    
    def _run_ffmpeg_command(self, cmd: List[str], operation_name: str, silent: bool = False, 
                           expected_duration: Optional[float] = None, output_path: Optional[str] = None) -> bool:
        """
        Run FFmpeg command with error handling.
        
        Args:
            cmd: FFmpeg command list
            operation_name: Name of the operation for logging
            silent: Whether to suppress output
            expected_duration: Expected duration in seconds (for validation)
            output_path: Path to output file (for validation)
            
        Returns:
            True if successful, False otherwise
        """
        if not silent:
            print(f"Running {operation_name}...")
            print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=not silent,
                text=True,
                check=True
            )
            
            if not silent:
                print(f"✓ {operation_name} completed successfully!")
                if result.stdout:
                    print(f"Output: {result.stdout}")
            
            # Validate output duration if expected duration is provided
            if expected_duration is not None and output_path is not None and os.path.exists(output_path):
                output_info = self.get_video_info(output_path)
                if output_info and 'duration' in output_info:
                    try:
                        # Parse duration string (format: HH:MM:SS.ms)
                        duration_str = output_info['duration']
                        time_parts = duration_str.split(':')
                        if len(time_parts) == 3:
                            hours = int(time_parts[0])
                            minutes = int(time_parts[1])
                            seconds = float(time_parts[2])
                            actual_duration = hours * 3600 + minutes * 60 + seconds
                            
                            # Check if duration is reasonable (within 10% tolerance)
                            tolerance = expected_duration * 0.1
                            if abs(actual_duration - expected_duration) > tolerance:
                                if not silent:
                                    print(f"⚠️  Warning: Duration mismatch! Expected: {expected_duration:.2f}s, Actual: {actual_duration:.2f}s")
                                    return False
                            else:
                                if not silent:
                                    print(f"✓ Duration validation passed: {actual_duration:.2f}s")
                    except (ValueError, IndexError) as e:
                        if not silent:
                            print(f"⚠️  Warning: Could not parse output duration: {e}")
                        return False
                else:
                    if not silent:
                        print(f"⚠️  Warning: Could not get output video info")
                    return False
            
            return True
            
        except subprocess.CalledProcessError as e:
            if not silent:
                print(f"✗ {operation_name} failed!")
                print(f"Error: {e.stderr}")
            return False
        except Exception as e:
            if not silent:
                print(f"✗ Unexpected error during {operation_name}: {e}")
            return False


def main() -> None:
    """Main function to handle command line arguments and run interpolation."""
    parser = argparse.ArgumentParser(
        description="Video Interpolation and Speed Adjustment Tool using FFmpeg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Frame duplication (fastest)
  python video_interpolator.py -i input.mp4 -o output.mp4 -m frame_duplication -f 60
  
  # Temporal interpolation (balanced)
  python video_interpolator.py -i input.mp4 -o output.mp4 -m temporal -f 60
  
  # Optical flow interpolation (highest quality)
  python video_interpolator.py -i input.mp4 -o output.mp4 -m optical_flow -f 60
  
  # Advanced interpolation (best quality, slowest)
  python video_interpolator.py -i input.mp4 -o output.mp4 -m advanced -f 60
  
  # TLBVFI interpolation (AI-powered, highest quality)
  python video_interpolator.py -i input.mp4 -o output.mp4 -m tlbvfi -f 60
  
  # Slow down video to 50% speed
  python video_interpolator.py -i input.mp4 -o output.mp4 --speed 0.5
  
  # Slow down video to 25% speed with interpolation
  python video_interpolator.py -i input.mp4 -o output.mp4 --speed 0.25 --interpolation temporal
  
  # Speed up video to 200% speed
  python video_interpolator.py -i input.mp4 -o output.mp4 --speed 2.0
  
  # Get video information
  python video_interpolator.py -i input.mp4 --info
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input video file path"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output video file path"
    )
    
    parser.add_argument(
        "-m", "--method",
        choices=["frame_duplication", "temporal", "optical_flow", "advanced", "tlbvfi"],
        default="temporal",
        help="Interpolation method (default: temporal)"
    )
    
    parser.add_argument(
        "-f", "--fps",
        type=int,
        default=60,
        help="Target frame rate (default: 60)"
    )
    
    parser.add_argument(
        "--speed",
        type=float,
        help="Speed factor (0.1 = 10%% speed, 2.0 = 200%% speed)"
    )
    
    parser.add_argument(
        "--interpolation",
        choices=["frame_duplication", "temporal", "optical_flow"],
        help="Interpolation method for slow motion (only used with --speed < 1.0)"
    )
    
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Remove audio from output (useful for very slow speeds)"
    )
    
    parser.add_argument(
        "--ffmpeg-path",
        default="ffmpeg",
        help="Path to FFmpeg executable (default: ffmpeg)"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show video information only"
    )
    
    parser.add_argument(
        "--remove-frames",
        type=int,
        default=0,
        help="Remove the first N frames from the video (default: 0)"
    )
    
    parser.add_argument(
        "--tlbvfi-model",
        type=str,
        default="model/vimeo_unet.pth",
        help="Path to TLBVFI model file (default: model/vimeo_unet.pth)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"✗ Input file not found: {args.input}")
        sys.exit(1)
    
    # Initialize interpolator
    interpolator = VideoInterpolator(args.ffmpeg_path, args.tlbvfi_model)
    
    # Show video info if requested
    if args.info:
        print(f"Video Information for: {args.input}")
        print("-" * 50)
        info = interpolator.get_video_info(args.input)
        if info:
            print(f"Duration: {info.get('duration', 'Unknown')}")
            print(f"Frame Rate: {info.get('fps', 'Unknown')} fps")
        else:
            print("Could not retrieve video information")
        return
    
    # Validate output path
    if not args.output:
        print("✗ Output path is required when not using --info")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Show video info before processing
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("-" * 50)
    
    info = interpolator.get_video_info(args.input)
    if info:
        print(f"Original FPS: {info.get('fps', 'Unknown')}")
        print(f"Duration: {info.get('duration', 'Unknown')}")
    print("-" * 50)
    
    # Perform operation based on arguments
    success = False
    
    if args.speed is not None:
        # Speed adjustment
        print(f"Speed Factor: {args.speed}x")
        if args.interpolation:
            print(f"Interpolation Method: {args.interpolation}")
        print(f"Preserve Audio: {not args.no_audio}")
        
        success = interpolator.adjust_speed(
            args.input, 
            args.output, 
            args.speed, 
            preserve_audio=not args.no_audio,
            interpolation_method=args.interpolation,
            remove_frames=args.remove_frames
        )
    else:
        # Frame rate interpolation
        print(f"Method: {args.method}")
        print(f"Target FPS: {args.fps}")
        
        if args.method == "frame_duplication":
            success = interpolator.frame_duplication(args.input, args.output, args.fps)
        elif args.method == "temporal":
            success = interpolator.temporal_interpolation(args.input, args.output, args.fps)
        elif args.method == "optical_flow":
            success = interpolator.optical_flow_interpolation(args.input, args.output, args.fps)
        elif args.method == "advanced":
            success = interpolator.advanced_interpolation(args.input, args.output, args.fps)
        elif args.method == "tlbvfi":
            success = interpolator.tlbvfi_interpolation(args.input, args.output, args.fps)
    
    if success:
        print(f"\n✓ Video processing completed successfully!")
        print(f"Output saved to: {args.output}")
        
        # Show final video info
        final_info = interpolator.get_video_info(args.output)
        if final_info:
            print(f"Final FPS: {final_info.get('fps', 'Unknown')}")
            print(f"Final Duration: {final_info.get('duration', 'Unknown')}")
    else:
        print(f"\n✗ Video processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 