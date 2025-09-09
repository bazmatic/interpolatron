"""
Specialized MPS optimizer for wavelet transform operations in TLBVFI.
"""

import torch
import torch.nn.functional as F
from mps_optimizer import mps_optimizer

class WaveletMPSOptimizer:
    def __init__(self):
        self.mps_available = torch.backends.mps.is_available()
        self.chunk_size = 32768  # Half of MPS limit for safety
        
    def smart_conv1d_flat(self, x, filter, padding):
        """Optimized conv1d_flat that works around MPS limitations."""
        if not self.mps_available:
            return F.conv1d(x.unsqueeze(1), filter, padding=padding).squeeze(1)
        
        # Check if we can use MPS directly
        if x.shape[1] <= self.chunk_size:
            try:
                device = torch.device('mps')
                x_mps = x.to(device)
                filter_mps = filter.to(device)
                result = F.conv1d(x_mps.unsqueeze(1), filter_mps, padding=padding).squeeze(1)
                return result.to(x.device)
            except RuntimeError as e:
                if "not supported at the MPS device" in str(e):
                    print(f"⚠️  MPS conv1d failed, using CPU: {e}")
                    return F.conv1d(x.unsqueeze(1), filter, padding=padding).squeeze(1)
                else:
                    raise e
        
        # Chunk the operation
        print(f"🔧 Chunking large conv1d operation (channels: {x.shape[1]})")
        chunks = []
        chunk_size = self.chunk_size
        
        for i in range(0, x.shape[1], chunk_size):
            end_idx = min(i + chunk_size, x.shape[1])
            x_chunk = x[:, i:end_idx, :]
            filter_chunk = filter[i:end_idx, :, :]
            
            # Try MPS for each chunk
            try:
                device = torch.device('mps')
                x_mps = x_chunk.to(device)
                filter_mps = filter_chunk.to(device)
                chunk_result = F.conv1d(x_mps.unsqueeze(1), filter_mps, padding=padding).squeeze(1)
                chunks.append(chunk_result.to(x.device))
            except RuntimeError:
                # Fallback to CPU for this chunk
                chunk_result = F.conv1d(x_chunk.unsqueeze(1), filter_chunk, padding=padding).squeeze(1)
                chunks.append(chunk_result)
        
        return torch.cat(chunks, dim=1)
    
    def smart_wavelet_transform(self, vid, low_filter, high_filter):
        """Optimized wavelet transform that uses MPS when possible."""
        if not self.mps_available:
            return self._cpu_wavelet_transform(vid, low_filter, high_filter)
        
        try:
            # Try MPS for the entire operation
            device = torch.device('mps')
            vid_mps = vid.to(device)
            low_filter_mps = low_filter.to(device)
            high_filter_mps = high_filter.to(device)
            
            # Apply filters
            low_freq = self.smart_conv1d_flat(vid_mps, low_filter_mps, 0)
            high_freq = self.smart_conv1d_flat(vid_mps, high_filter_mps, 0)
            
            return low_freq.to(vid.device), high_freq.to(vid.device)
            
        except RuntimeError as e:
            if "not supported at the MPS device" in str(e):
                print(f"⚠️  MPS wavelet transform failed, using CPU: {e}")
                return self._cpu_wavelet_transform(vid, low_filter, high_filter)
            else:
                raise e
    
    def _cpu_wavelet_transform(self, vid, low_filter, high_filter):
        """CPU fallback for wavelet transform."""
        low_freq = F.conv1d(vid.unsqueeze(1), low_filter, padding=0).squeeze(1)
        high_freq = F.conv1d(vid.unsqueeze(1), high_filter, padding=0).squeeze(1)
        return low_freq, high_freq

# Global instance
wavelet_optimizer = WaveletMPSOptimizer()
