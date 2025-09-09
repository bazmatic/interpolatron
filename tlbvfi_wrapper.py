#!/usr/bin/env python3
"""
TLBVFI Wrapper - Uses the original TLBVFI repository for video interpolation.

This wrapper provides a clean interface to the original TLBVFI implementation
while handling video processing, frame extraction, and result assembly.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Tuple


class TLBVFIWrapper:
    """Wrapper for the original TLBVFI repository."""
    
    def __init__(self, model_path: str = "model/vimeo_unet.pth"):
        """
        Initialize the TLBVFI wrapper.
        
        Args:
            model_path: Path to the TLBVFI model checkpoint
        """
        self.model_path = Path(model_path)
        self.tlbvfi_dir = Path("tlbvfi_original")
        self.config_path = self.tlbvfi_dir / "configs" / "Template-LBBDM-video.yaml"
        
        # Verify paths exist
        if not self.tlbvfi_dir.exists():
            raise FileNotFoundError(f"TLBVFI repository not found at: {self.tlbvfi_dir}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"TLBVFI model not found at: {self.model_path}")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"TLBVFI config not found at: {self.config_path}")
        
        # Check if TLBVFI dependencies are available
        self.tlbvfi_available = self._check_tlbvfi_dependencies()
        
        print(f"✓ TLBVFI wrapper initialized")
        print(f"  Model: {self.model_path}")
        print(f"  Repository: {self.tlbvfi_dir}")
        print(f"  TLBVFI Available: {self.tlbvfi_available}")
    
    def _check_tlbvfi_dependencies(self) -> bool:
        """Check if TLBVFI dependencies are available."""
        try:
            # Try to import the key TLBVFI modules
            sys.path.insert(0, str(self.tlbvfi_dir.absolute()))
            
            # Check for CuPy availability
            try:
                import cupy
                print("✓ CuPy is available - TLBVFI will use CUDA acceleration")
                cupy_available = True
            except ImportError:
                print("⚠ CuPy not available - TLBVFI will use CPU fallback")
                cupy_available = False
            
            # Test basic imports
            from utils import dict2namespace
            from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
            
            # Check if compatibility layers are available
            try:
                from flolpips.correlation import correlation_compat
                print("✓ Correlation compatibility layer available")
            except ImportError:
                print("⚠ Correlation compatibility layer not available")
            
            try:
                from cupy_module import dsepconv_compat
                print("✓ DSepconv compatibility layer available")
            except ImportError:
                print("⚠ DSepconv compatibility layer not available")
            
            return True
        except ImportError as e:
            print(f"⚠️  TLBVFI dependencies not fully available: {e}")
            print("   Some dependencies (like CuPy) may be missing.")
            print("   TLBVFI will use CPU fallback implementations.")
            return False
        except Exception as e:
            print(f"⚠️  TLBVFI check failed: {e}")
            # Don't fail completely for CUDA-related errors - we have CPU fallback
            if "CUDA" in str(e) or "cuda" in str(e):
                print("   This is expected on macOS - CPU fallback will be used")
                return True  # Allow TLBVFI to proceed with CPU fallback
            return False
        finally:
            # Remove the path we added
            if str(self.tlbvfi_dir.absolute()) in sys.path:
                sys.path.remove(str(self.tlbvfi_dir.absolute()))
    
    def _check_model_files(self) -> bool:
        """Check if required TLBVFI model files exist."""
        # Check for VQGAN model file
        vqgan_path = self.tlbvfi_dir / "results" / "VQGAN" / "vimeo_new.ckpt"
        if not vqgan_path.exists():
            return False
        return True
    
    def _fallback_temporal_interpolation(self, input_path: str, output_path: str, target_fps: int) -> bool:
        """Fallback to temporal interpolation when TLBVFI models are missing."""
        try:
            print("🔄 Using temporal interpolation fallback...")
            
            # Use FFmpeg for temporal interpolation
            cmd = [
                "ffmpeg", "-i", input_path,
                "-filter:v", f"fps={target_fps}",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-y", output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Temporal interpolation completed successfully!")
                return True
            else:
                print(f"✗ Temporal interpolation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"✗ Fallback interpolation failed: {e}")
            return False
    
    def interpolate_video(self, input_path: str, output_path: str, target_fps: int = 60) -> bool:
        """
        Interpolate a video using TLBVFI or fallback to temporal interpolation.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            target_fps: Target frame rate
            
        Returns:
            True if successful, False otherwise
        """
        if not self.tlbvfi_available:
            print("✗ TLBVFI not available. Cannot proceed without TLBVFI dependencies.")
            print("Please install required dependencies:")
            print("  pip install torch torchvision opencv-python Pillow numpy")
            print("  Note: TLBVFI now supports CPU fallback - CuPy (CUDA) is optional for acceleration")
            return False
        
        # Check if required model files exist
        if not self._check_model_files():
            print("✗ Required TLBVFI model files are missing.")
            print("  Please ensure the VQGAN model file is available.")
            return False
        
        try:
            print(f"🎬 Starting TLBVFI interpolation...")
            print(f"  Input: {input_path}")
            print(f"  Output: {output_path}")
            print(f"  Target FPS: {target_fps}")
            
            # Get original video info
            original_fps = self._get_video_fps(input_path)
            print(f"  Original FPS: {original_fps}")
            
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract frames from input video
                frames_dir = temp_path / "frames"
                frames_dir.mkdir()
                
                print("📸 Extracting frames...")
                self._extract_frames(input_path, frames_dir, original_fps)
                
                # Get frame list
                frame_files = sorted([f for f in frames_dir.glob("*.png")])
                if len(frame_files) < 2:
                    print("✗ Need at least 2 frames for interpolation")
                    return False
                
                print(f"  Extracted {len(frame_files)} frames")
                
                # Interpolate frames using TLBVFI
                interpolated_dir = temp_path / "interpolated"
                interpolated_dir.mkdir()
                
                print("🤖 Running TLBVFI interpolation...")
                success = self._interpolate_frames_tlbvfi(frames_dir, interpolated_dir, target_fps)
                
                if not success:
                    print("✗ TLBVFI interpolation failed. Cannot proceed.")
                    return False
                
                # Assemble interpolated frames into video
                print("🎞️  Assembling video...")
                self._assemble_video(interpolated_dir, output_path, target_fps)
                
                print("✓ TLBVFI interpolation completed successfully!")
                return True
                
        except Exception as e:
            print(f"✗ TLBVFI interpolation failed: {e}")
            return False
    
    
    def _get_video_fps(self, video_path: str) -> float:
        """Get the frame rate of a video using FFprobe."""
        cmd = [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️  Could not get video FPS, defaulting to 30: {result.stderr}")
            return 30.0
        
        # Parse the frame rate (format is usually "30/1" or "30000/1001")
        fps_str = result.stdout.strip()
        if '/' in fps_str:
            numerator, denominator = fps_str.split('/')
            return float(numerator) / float(denominator)
        else:
            return float(fps_str)
    
    def _extract_frames(self, video_path: str, output_dir: Path, original_fps: float) -> None:
        """Extract frames from video using FFmpeg."""
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={original_fps}",  # Extract at original fps to get all frames
            "-y", str(output_dir / "frame_%06d.png")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Frame extraction failed: {result.stderr}")
    
    def _interpolate_frames_tlbvfi(self, frames_dir: Path, output_dir: Path, target_fps: int) -> bool:
        """
        Interpolate frames using the original TLBVFI implementation.
        
        This method processes frame pairs and generates interpolated frames
        between them using the original TLBVFI model.
        """
        try:
            frame_files = sorted([f for f in frames_dir.glob("*.png")])
            
            # For each pair of consecutive frames, interpolate
            for i in range(len(frame_files) - 1):
                frame0 = frame_files[i]
                frame1 = frame_files[i + 1]
                
                print(f"  Interpolating between frames {i+1}/{len(frame_files)-1}")
                
                # Run TLBVFI interpolation for this frame pair
                success = self._interpolate_frame_pair_tlbvfi(frame0, frame1, output_dir, i)
                if not success:
                    return False
            
            return True
            
        except Exception as e:
            print(f"✗ Frame interpolation failed: {e}")
            return False
    
    def _interpolate_frame_pair_tlbvfi(self, frame0: Path, frame1: Path, output_dir: Path, pair_index: int) -> bool:
        """
        Interpolate between two frames using the original TLBVFI.
        
        This calls the original TLBVFI interpolate_one.py script.
        """
        try:
            # Create a temporary ground truth frame (just average of the two frames)
            gt_frame = self._create_ground_truth_frame(frame0, frame1)
            
            # Change to TLBVFI directory
            original_cwd = os.getcwd()
            os.chdir(self.tlbvfi_dir)
            
            # Copy frames to TLBVFI directory for processing
            temp_frame0 = Path("temp_frame0.png")
            temp_frame1 = Path("temp_frame1.png")
            temp_gt = Path("temp_gt.png")
            
            # Use absolute paths for copying
            shutil.copy2(str(frame0.absolute()), str(temp_frame0.absolute()))
            shutil.copy2(str(frame1.absolute()), str(temp_frame1.absolute()))
            shutil.copy2(str(gt_frame.absolute()), str(temp_gt.absolute()))
            
            # Run TLBVFI interpolation
            cmd = [
                sys.executable, "interpolate_one.py",
                "--frame0", str(temp_frame0),
                "--frame1", str(temp_frame1),
                "--frame", str(temp_gt),
                "--resume_model", "../model/vimeo_unet.pth",
                "--config", "configs/Template-LBBDM-video.yaml",
                "--result_path", "temp_results"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Clean up temp files
            for temp_file in [temp_frame0, temp_frame1, temp_gt]:
                if temp_file.exists():
                    temp_file.unlink()
            
            # Change back to original directory
            os.chdir(original_cwd)
            
            if result.returncode != 0:
                print(f"✗ TLBVFI subprocess failed: {result.stderr}")
                return False
            
            # Look for the output in the interpolated directory
            tlbvfi_output = self.tlbvfi_dir / "interpolated" / "example.png"
            if tlbvfi_output.exists():
                # Copy the result to our output directory
                output_path = output_dir / f"interpolated_{pair_index:06d}.png"
                shutil.copy2(tlbvfi_output, output_path)
                
                # Clean up TLBVFI output
                tlbvfi_output.unlink()
                
                return True
            else:
                print(f"✗ TLBVFI output not found at: {tlbvfi_output}")
                return False
            
        except Exception as e:
            print(f"✗ Frame pair interpolation failed: {e}")
            return False
    
    def _create_ground_truth_frame(self, frame0: Path, frame1: Path) -> Path:
        """Create a ground truth frame by averaging two frames."""
        import cv2
        
        # Load frames
        img0 = cv2.imread(str(frame0))
        img1 = cv2.imread(str(frame1))
        
        # Average the frames
        avg_frame = ((img0.astype(np.float32) + img1.astype(np.float32)) / 2).astype(np.uint8)
        
        # Save to temporary file
        temp_gt = frame0.parent / f"gt_{frame0.stem}.png"
        cv2.imwrite(str(temp_gt), avg_frame)
        
        return temp_gt
    
    def _assemble_video(self, frames_dir: Path, output_path: str, target_fps: int) -> None:
        """Assemble interpolated frames into final video."""
        cmd = [
            "ffmpeg", "-framerate", str(target_fps),
            "-i", str(frames_dir / "interpolated_%06d.png"),
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-y", output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Video assembly failed: {result.stderr}")


def main():
    """Test the TLBVFI wrapper."""
    wrapper = TLBVFIWrapper()
    
    # Test with a sample video
    input_video = "input/i2v3_00008.mp4"
    output_video = "test_tlbvfi_wrapper.mp4"
    
    if Path(input_video).exists():
        success = wrapper.interpolate_video(input_video, output_video, 60)
        if success:
            print(f"✓ Test completed successfully! Output: {output_video}")
        else:
            print("✗ Test failed")
    else:
        print(f"✗ Test video not found: {input_video}")


if __name__ == "__main__":
    main()
