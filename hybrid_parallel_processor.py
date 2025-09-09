"""
Hybrid parallel processor that combines the best of sequential reliability
with parallel performance using the existing TLBVFI infrastructure.
"""

import threading
import time
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import psutil
import subprocess
import sys

class HybridParallelProcessor:
    def __init__(self, max_concurrent: Optional[int] = None):
        # Determine optimal concurrency
        self.cpu_count = psutil.cpu_count(logical=True)
        self.physical_cores = psutil.cpu_count(logical=False)

        if max_concurrent is None:
            # Use physical cores for better performance
            self.max_concurrent = self.physical_cores
        else:
            self.max_concurrent = min(max_concurrent, self.cpu_count)

        print(f"🔄 Hybrid Parallel Processor: Using {self.max_concurrent} concurrent processes")
        print(f"💻 System: {self.physical_cores} physical cores, {self.cpu_count} logical cores")

        # Thread synchronization
        self.lock = threading.Lock()
        self.completed_count = 0

    def process_frames_hybrid(self, frames_dir: Path, output_dir: Path,
                            total_pairs: int, config: Dict) -> bool:
        """
        Process frames using hybrid parallel approach.
        Divides work into chunks and processes each chunk with a separate
        TLBVFI process for maximum reliability and performance.
        """
        if total_pairs == 0:
            print("⚠️  No frame pairs to process")
            return True

        print(f"🎬 Processing {total_pairs} frame pairs with hybrid parallelism...")

        # Calculate chunk size for optimal division
        chunk_size = max(1, total_pairs // self.max_concurrent)
        if total_pairs % self.max_concurrent != 0:
            chunk_size += 1

        print(f"📦 Dividing into {self.max_concurrent} chunks of ~{chunk_size} pairs each")

        # Create chunks
        chunks = []
        for i in range(0, total_pairs, chunk_size):
            start_pair = i
            end_pair = min(i + chunk_size, total_pairs)
            chunks.append((start_pair, end_pair))

        print(f"🚀 Starting {len(chunks)} parallel chunks...")

        start_time = time.time()
        self.completed_count = 0

        # Process chunks in parallel using threads (each thread runs a subprocess)
        threads = []
        results = []

        for chunk_idx, (start_pair, end_pair) in enumerate(chunks):
            thread = threading.Thread(
                target=self._process_chunk,
                args=(frames_dir, output_dir, start_pair, end_pair, chunk_idx, config, results)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        total_time = time.time() - start_time

        # Check results
        successful_chunks = sum(1 for success, _, _, _ in results if success)
        total_processing_time = sum(time_taken for _, _, _, time_taken in results)

        print("\n✅ Hybrid parallel processing completed!")
        print(f"   Chunks: {successful_chunks}/{len(chunks)} successful")
        print(".2f")
        print(".2f")
        if successful_chunks != len(chunks):
            failed_chunks = len(chunks) - successful_chunks
            print(f"⚠️  {failed_chunks} chunks failed")

        return successful_chunks == len(chunks)

    def _process_chunk(self, frames_dir: Path, output_dir: Path,
                      start_pair: int, end_pair: int, chunk_idx: int,
                      config: Dict, results: List) -> None:
        """
        Process a chunk of frame pairs using a dedicated TLBVFI process.
        """
        try:
            start_time = time.time()

            # Create chunk-specific output directory
            chunk_output_dir = output_dir / "04d"
            chunk_output_dir.mkdir(exist_ok=True)

            # Prepare frame pairs for this chunk
            frame_files = sorted([f for f in frames_dir.glob("*.png")])
            chunk_frame_pairs = []

            for i in range(start_pair, end_pair):
                if i < len(frame_files) - 1:
                    frame0 = frame_files[i]
                    frame1 = frame_files[i + 1]
                    chunk_frame_pairs.append((str(frame0), str(frame1)))

            if not chunk_frame_pairs:
                with self.lock:
                    print(f"⚠️  Chunk {chunk_idx}: No frame pairs to process")
                results.append((False, chunk_idx, 0, time.time() - start_time))
                return

            # Process chunk using sequential batch processing
            success = self._process_chunk_sequential(
                chunk_frame_pairs, chunk_output_dir, chunk_idx, config
            )

            processing_time = time.time() - start_time

            with self.lock:
                if success:
                    print(".2f")
                else:
                    print(f"❌ Chunk {chunk_idx} failed")

            results.append((success, chunk_idx, len(chunk_frame_pairs), processing_time))

        except Exception as e:
            processing_time = time.time() - start_time
            with self.lock:
                print(f"💥 Chunk {chunk_idx} error: {e}")
            results.append((False, chunk_idx, 0, processing_time))

    def _process_chunk_sequential(self, frame_pairs: List[Tuple[str, str]],
                                output_dir: Path, chunk_idx: int, config: Dict) -> bool:
        """
        Process a chunk of frame pairs sequentially using the existing TLBVFI infrastructure.
        This is more reliable than trying to parallelize within the chunk.
        """
        successful_pairs = 0

        for pair_idx, (frame0, frame1) in enumerate(frame_pairs):
            try:
                # Create unique output filename for this frame pair
                global_pair_idx = chunk_idx * 1000 + pair_idx  # Avoid filename conflicts
                output_path = output_dir / "04d"

                # Run interpolation using the existing working method
                success = self._interpolate_single_pair(frame0, frame1, output_path, config)

                if success:
                    successful_pairs += 1
                else:
                    print(f"⚠️  Failed to interpolate pair {global_pair_idx}")

            except Exception as e:
                print(f"⚠️  Error processing pair {global_pair_idx}: {e}")

        return successful_pairs == len(frame_pairs)

    def _interpolate_single_pair(self, frame0: str, frame1: str,
                               output_path: str, config: Dict) -> bool:
        """
        Interpolate a single frame pair using the existing TLBVFI infrastructure.
        """
        try:
            # Run the interpolation using subprocess to avoid complex imports
            cmd = [
                sys.executable,
                '/Users/barryearsman/projects/personal/sandbox/interpolate/tlbvfi_original/interpolate_one.py',
                '--frame0', frame0,
                '--frame1', frame1,
                '--frame', frame0,  # Use frame0 as ground truth
                '--resume_model', '/Users/barryearsman/projects/personal/sandbox/interpolate/model/vimeo_unet.pth',
                '--config', '/Users/barryearsman/projects/personal/sandbox/interpolate/tlbvfi_original/configs/Template-LBBDM-video.yaml',
                '--result_path', '/tmp/tlbvfi_temp_result'  # Use temp result path
            ]

            # Run with minimal output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd='/Users/barryearsman/projects/personal/sandbox/interpolate/tlbvfi_original'
            )

            if result.returncode == 0:
                # Check if result file exists and move it
                temp_result = Path('/Users/barryearsman/projects/personal/sandbox/interpolate/tlbvfi_original/interpolated/example.png')
                if temp_result.exists():
                    temp_result.rename(output_path)
                    return True

            return False

        except Exception as e:
            print(f"Interpolation error: {e}")
            return False

# Global instance
hybrid_processor = HybridParallelProcessor()
