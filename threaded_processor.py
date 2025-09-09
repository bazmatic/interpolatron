"""
Threaded processor for TLBVFI that provides better parallelization than the current sequential approach.
"""

import threading
import concurrent.futures
import torch
import numpy as np
import time
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

class ThreadedProcessor:
    def __init__(self, max_threads: Optional[int] = None):
        # Determine optimal number of threads
        self.cpu_count = psutil.cpu_count(logical=True)
        self.physical_cores = psutil.cpu_count(logical=False)

        if max_threads is None:
            # Use all logical cores for threading (better for I/O bound tasks)
            self.max_threads = self.cpu_count
        else:
            self.max_threads = min(max_threads, self.cpu_count)

        print(f"🧵 Threaded Processor: Using {self.max_threads} threads (of {self.cpu_count} logical cores)")

        # Thread synchronization
        self.lock = threading.Lock()

        # GPU management
        self.mps_available = torch.backends.mps.is_available()
        self.cuda_available = torch.cuda.is_available()

        if self.mps_available:
            print("🎯 MPS GPU available for threaded processing")
        if self.cuda_available:
            print("🎯 CUDA GPU available for threaded processing")

    def process_frame_pairs_threaded(self, frame_pairs: List[Tuple[str, str]],
                                   output_dir: Path, config: Dict) -> List[Tuple[int, bool, float]]:
        """
        Process multiple frame pairs using threading.

        Args:
            frame_pairs: List of (frame0_path, frame1_path) tuples
            output_dir: Directory to save interpolated frames
            config: Configuration dictionary

        Returns:
            List of (pair_index, success, processing_time) tuples
        """
        print(f"🧵 Processing {len(frame_pairs)} frame pairs with {self.max_threads} threads...")

        start_time = time.time()

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            # Submit all tasks
            future_to_pair = {}
            for i, (frame0, frame1) in enumerate(frame_pairs):
                future = executor.submit(
                    self._interpolate_pair_threaded,
                    frame0, frame1, output_dir, i, config
                )
                future_to_pair[future] = i

            # Collect results as they complete
            results = []
            completed = 0
            total_pairs = len(frame_pairs)

            for future in as_completed(future_to_pair):
                pair_index = future_to_pair[future]
                try:
                    success, processing_time = future.result()
                    results.append((pair_index, success, processing_time))

                    completed += 1
                    progress = (completed / total_pairs) * 100
                    elapsed = time.time() - start_time
                    eta = elapsed * (total_pairs - completed) / max(completed, 1)

                    with self.lock:
                        if success:
                            print(".2f")
                        else:
                            print(f"❌ Frame pair {pair_index} failed")

                except Exception as e:
                    with self.lock:
                        print(f"💥 Thread error for pair {pair_index}: {e}")
                    results.append((pair_index, False, 0.0))

        # Sort results by pair index
        results.sort(key=lambda x: x[0])

        total_time = time.time() - start_time
        successful_pairs = sum(1 for _, success, _ in results if success)

        print("\n✅ Threaded processing completed!")
        print(f"   Processed: {successful_pairs}/{total_pairs} frame pairs")
        print(".2f")
        if successful_pairs != total_pairs:
            print(f"⚠️  {total_pairs - successful_pairs} frame pairs failed")

        return results

    def _interpolate_pair_threaded(self, frame0: str, frame1: str, output_dir: Path,
                                 pair_index: int, config: Dict) -> Tuple[bool, float]:
        """
        Interpolate a single frame pair in a thread.
        """
        start_time = time.time()

        try:
            # Create output filename
            output_path = output_dir / "04d"

            # Create temporary directory for this thread
            temp_dir = Path(output_dir) / f"temp_thread_{pair_index}"
            temp_dir.mkdir(exist_ok=True)

            # Copy frames to temp directory to avoid conflicts
            temp_frame0 = temp_dir / "04d"
            temp_frame1 = temp_dir / "04d"

            # Copy input frames
            import shutil
            shutil.copy2(frame0, temp_frame0)
            shutil.copy2(frame1, temp_frame1)

            # Run the interpolation using subprocess
            cmd = [
                sys.executable,
                '/Users/barryearsman/projects/personal/sandbox/interpolate/tlbvfi_original/interpolate_one.py',
                '--frame0', str(temp_frame0),
                '--frame1', str(temp_frame1),
                '--frame', str(temp_frame0),  # Use frame0 as ground truth
                '--resume_model', '/Users/barryearsman/projects/personal/sandbox/interpolate/model/vimeo_unet.pth',
                '--config', '/Users/barryearsman/projects/personal/sandbox/interpolate/tlbvfi_original/configs/Template-LBBDM-video.yaml',
                '--result_path', str(temp_dir / "04d")
            ]

            # Run the command
            import subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd='/Users/barryearsman/projects/personal/sandbox/interpolate/tlbvfi_original'
            )

            processing_time = time.time() - start_time

            if result.returncode == 0:
                # Move the result to the correct output location
                result_file = Path('/Users/barryearsman/projects/personal/sandbox/interpolate/tlbvfi_original/interpolated/example.png')
                if result_file.exists():
                    result_file.rename(output_path)

                    # Clean up temp directory
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return True, processing_time
                else:
                    with self.lock:
                        print(f"Result file not found: {result_file}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return False, processing_time
            else:
                with self.lock:
                    print(f"Interpolation failed for pair {pair_index}: {result.stderr[:200]}...")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False, processing_time

        except Exception as e:
            processing_time = time.time() - start_time
            with self.lock:
                print(f"Thread interpolation error for pair {pair_index}: {e}")
            # Clean up on error
            try:
                if 'temp_dir' in locals():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
            return False, processing_time

# Global instance
threaded_processor = ThreadedProcessor()
