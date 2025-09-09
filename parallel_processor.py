"""
Advanced parallel processor for TLBVFI frame interpolation using multi-process architecture.
"""

import multiprocessing as mp
from multiprocessing import Pool, Manager, Lock
import torch
import numpy as np
import os
import time
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import psutil

class ParallelProcessor:
    def __init__(self, max_processes: Optional[int] = None):
        # Determine optimal number of processes
        self.cpu_count = psutil.cpu_count(logical=True)
        self.physical_cores = psutil.cpu_count(logical=False)

        if max_processes is None:
            # Use all physical cores for better performance
            self.max_processes = self.physical_cores
        else:
            self.max_processes = min(max_processes, self.cpu_count)

        print(f"🚀 Parallel Processor: Using {self.max_processes} processes (of {self.cpu_count} logical cores)")
        print(f"💻 System: {self.physical_cores} physical cores, {self.cpu_count} logical cores")

        # Memory management
        self.memory_per_process = self._calculate_memory_per_process()
        print(f"🧠 Memory per process: {self.memory_per_process / (1024**3):.2f} GB")

        # GPU management (if available)
        self.mps_available = torch.backends.mps.is_available()
        self.cuda_available = torch.cuda.is_available()

        if self.mps_available:
            print("🎯 MPS GPU available - will be shared across processes")
        if self.cuda_available:
            print("🎯 CUDA GPU available - will be shared across processes")

        # Initialize process pool
        self.pool: Optional[Pool] = None
        self.manager = Manager()
        self.lock = self.manager.Lock()

    def _calculate_memory_per_process(self) -> int:
        """Calculate optimal memory allocation per process."""
        total_memory = psutil.virtual_memory().total
        available_memory = psutil.virtual_memory().available

        # Reserve 2GB for system and other processes
        usable_memory = available_memory - (2 * 1024**3)

        # Allocate memory per process
        memory_per_process = usable_memory // self.max_processes

        # Cap at 8GB per process for safety
        return min(memory_per_process, 8 * 1024**3)

    def _init_worker(self):
        """Initialize each worker process."""
        # Set process priority
        try:
            os.nice(10)  # Lower priority to be more system-friendly
        except:
            pass

        # Configure PyTorch for this process
        torch.set_num_threads(1)  # Use 1 thread per process for better performance

        # Disable unnecessary warnings
        import warnings
        warnings.filterwarnings("ignore")

        print(f"🔧 Worker {mp.current_process().name} initialized")

    def start_pool(self):
        """Start the multiprocessing pool."""
        if self.pool is None:
            print(f"🏊 Starting process pool with {self.max_processes} workers...")
            self.pool = Pool(
                processes=self.max_processes,
                initializer=self._init_worker,
                maxtasksperchild=5  # Restart processes after 5 tasks to prevent memory leaks
            )
        return self.pool

    def stop_pool(self):
        """Stop the multiprocessing pool."""
        if self.pool is not None:
            print("🛑 Stopping process pool...")
            self.pool.close()
            self.pool.join()
            self.pool = None

    def process_frame_pairs_parallel(self, frame_pairs: List[Tuple[str, str]],
                                   output_dir: Path, config: Dict) -> List[Tuple[int, bool, float]]:
        """
        Process multiple frame pairs in parallel.

        Args:
            frame_pairs: List of (frame0_path, frame1_path) tuples
            output_dir: Directory to save interpolated frames
            config: Configuration dictionary

        Returns:
            List of (pair_index, success, processing_time) tuples
        """
        if not self.pool:
            self.start_pool()

        # Prepare arguments for each task
        tasks = []
        for i, (frame0, frame1) in enumerate(frame_pairs):
            task_config = {
                'frame0': frame0,
                'frame1': frame1,
                'output_dir': str(output_dir),
                'pair_index': i,
                'config': config,
                'lock': self.lock
            }
            tasks.append(task_config)

        print(f"🎬 Processing {len(tasks)} frame pairs with {self.max_processes} parallel workers...")

        # Process tasks in parallel
        start_time = time.time()
        results = self.pool.map_async(self._process_single_pair, tasks)

        # Monitor progress
        total_tasks = len(tasks)
        completed = 0

        while not results.ready():
            time.sleep(1)
            # Get completed tasks (approximate)
            if hasattr(results, '_number_left'):
                remaining = results._number_left
                completed = total_tasks - remaining
            else:
                completed = completed + 1 if completed < total_tasks else total_tasks

            progress = (completed / total_tasks) * 100
            elapsed = time.time() - start_time
            eta = elapsed * (total_tasks - completed) / max(completed, 1)

            print(".1f"
        # Get final results
        final_results = results.get()

        total_time = time.time() - start_time
        print(".2f"
        return final_results

    @staticmethod
    def _process_single_pair(task_config: Dict) -> Tuple[int, bool, float]:
        """
        Process a single frame pair (runs in worker process).

        Args:
            task_config: Configuration for this task

        Returns:
            Tuple of (pair_index, success, processing_time)
        """
        start_time = time.time()

        try:
            frame0 = task_config['frame0']
            frame1 = task_config['frame1']
            output_dir = Path(task_config['output_dir'])
            pair_index = task_config['pair_index']
            config = task_config['config']
            lock = task_config['lock']

            # Import here to avoid issues in worker processes
            import sys
            sys.path.append('/Users/barryearsman/projects/personal/sandbox/interpolate/tlbvfi_original')

            # Process the frame pair
            success = ParallelProcessor._interpolate_pair_worker(frame0, frame1, output_dir, pair_index)

            processing_time = time.time() - start_time

            if success:
                with lock:
                    print(".2f")
            else:
                with lock:
                    print(f"❌ Frame pair {pair_index} failed")

            return pair_index, success, processing_time

        except Exception as e:
            processing_time = time.time() - start_time
            print(f"💥 Worker error for pair {task_config['pair_index']}: {e}")
            return task_config['pair_index'], False, processing_time

    @staticmethod
    def _interpolate_pair_worker(frame0: str, frame1: str, output_dir: Path, pair_index: int) -> bool:
        """
        Worker function to interpolate a single frame pair.
        Uses the existing TLBVFI infrastructure via subprocess for simplicity.
        """
        try:
            # Create output filename
            output_path = output_dir / "04d"

            # Create temporary directory for this worker
            temp_dir = Path(output_dir) / f"temp_worker_{pair_index}"
            temp_dir.mkdir(exist_ok=True)

            # Copy frames to temp directory to avoid conflicts
            temp_frame0 = temp_dir / "04d"
            temp_frame1 = temp_dir / "04d"
            temp_output = temp_dir / "04d"

            # Copy input frames
            import shutil
            shutil.copy2(frame0, temp_frame0)
            shutil.copy2(frame1, temp_frame1)

            # Run the interpolation using the existing interpolate_one.py
            cmd = [
                sys.executable,
                '/Users/barryearsman/projects/personal/sandbox/interpolate/tlbvfi_original/interpolate_one.py',
                '--frame0', str(temp_frame0),
                '--frame1', str(temp_frame1),
                '--frame', str(temp_frame0),  # Use frame0 as ground truth
                '--resume_model', '/Users/barryearsman/projects/personal/sandbox/interpolate/model/vimeo_unet.pth',
                '--config', '/Users/barryearsman/projects/personal/sandbox/interpolate/tlbvfi_original/configs/Template-LBBDM-video.yaml',
                '--result_path', str(temp_output)
            ]

            # Run the command with minimal output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd='/Users/barryearsman/projects/personal/sandbox/interpolate/tlbvfi_original'
            )

            if result.returncode == 0:
                # Move the result to the correct output location
                result_file = Path('/Users/barryearsman/projects/personal/sandbox/interpolate/tlbvfi_original/interpolated/example.png')
                if result_file.exists():
                    result_file.rename(output_path)

                    # Clean up temp directory
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return True
                else:
                    print(f"Result file not found: {result_file}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return False
            else:
                print(f"Interpolation failed for pair {pair_index}: {result.stderr[:200]}...")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False

        except Exception as e:
            print(f"Worker interpolation error for pair {pair_index}: {e}")
            # Clean up on error
            try:
                if 'temp_dir' in locals():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
            return False

    def __enter__(self):
        """Context manager entry."""
        self.start_pool()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_pool()

# Global instance
parallel_processor = ParallelProcessor()
