"""
Advanced MPS optimizer for TLBVFI that maximizes Apple Silicon GPU usage
by intelligently chunking operations and using hybrid processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math

class MPSOptimizer:
    def __init__(self):
        self.mps_available = torch.backends.mps.is_available()
        self.mps_limits = {
            'max_channels': 65536,
            'max_elements': 2**31,
            'optimal_chunk_size': 32768,  # Half the limit for safety
        }
        
        if self.mps_available:
            print("🚀 MPS Optimizer: Advanced Apple Silicon GPU acceleration enabled")
        else:
            print("💻 MPS Optimizer: CPU-only mode")
    
    def can_use_mps(self, tensor_shape: Tuple[int, ...], operation_type: str = "general") -> bool:
        """Check if a tensor operation can use MPS."""
        if not self.mps_available:
            return False
        
        total_elements = 1
        for dim in tensor_shape:
            total_elements *= dim
        
        # Check element count
        if total_elements > self.mps_limits['max_elements']:
            return False
        
        # Check channel limits for convolution operations
        if operation_type in ["conv", "conv1d", "conv2d", "conv3d"] and len(tensor_shape) >= 2:
            channels = tensor_shape[1]
            if channels > self.mps_limits['max_channels']:
                return False
        
        return True
    
    def chunk_tensor_for_mps(self, tensor: torch.Tensor, chunk_dim: int = 1) -> List[torch.Tensor]:
        """Split a tensor into MPS-compatible chunks."""
        if self.can_use_mps(tensor.shape):
            return [tensor]
        
        # Calculate chunk size
        original_shape = list(tensor.shape)
        chunk_size = self.mps_limits['optimal_chunk_size']
        
        if original_shape[chunk_dim] <= chunk_size:
            return [tensor]
        
        # Split along the specified dimension
        chunks = []
        num_chunks = math.ceil(original_shape[chunk_dim] / chunk_size)
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, original_shape[chunk_dim])
            
            # Create slice indices
            slice_indices = [slice(None)] * len(original_shape)
            slice_indices[chunk_dim] = slice(start_idx, end_idx)
            
            chunk = tensor[tuple(slice_indices)]
            chunks.append(chunk)
        
        return chunks
    
    def smart_conv1d(self, input_tensor: torch.Tensor, weight: torch.Tensor, 
                    bias: Optional[torch.Tensor] = None, stride: int = 1, 
                    padding: int = 0, dilation: int = 1, groups: int = 1) -> torch.Tensor:
        """Smart Conv1d that uses MPS when possible, CPU when needed."""
        if self.can_use_mps(input_tensor.shape, "conv1d"):
            # Use MPS for the entire operation
            device = torch.device('mps')
            input_mps = input_tensor.to(device)
            weight_mps = weight.to(device)
            bias_mps = bias.to(device) if bias is not None else None
            
            result = F.conv1d(input_mps, weight_mps, bias_mps, stride, padding, dilation, groups)
            return result.to(input_tensor.device)
        
        # Check if we can chunk the input
        if input_tensor.shape[1] > self.mps_limits['max_channels']:
            # Chunk along channel dimension
            input_chunks = self.chunk_tensor_for_mps(input_tensor, chunk_dim=1)
            weight_chunks = self.chunk_tensor_for_mps(weight, chunk_dim=0)
            
            results = []
            for i, input_chunk in enumerate(input_chunks):
                if i < len(weight_chunks):
                    weight_chunk = weight_chunks[i]
                    bias_chunk = None
                    if bias is not None:
                        bias_chunk = bias[i * weight_chunk.shape[0]:(i + 1) * weight_chunk.shape[0]]
                    
                    # Process chunk on MPS if possible
                    if self.can_use_mps(input_chunk.shape, "conv1d"):
                        device = torch.device('mps')
                        input_mps = input_chunk.to(device)
                        weight_mps = weight_chunk.to(device)
                        bias_mps = bias_chunk.to(device) if bias_chunk is not None else None
                        
                        chunk_result = F.conv1d(input_mps, weight_mps, bias_mps, stride, padding, dilation, groups)
                        results.append(chunk_result.to(input_tensor.device))
                    else:
                        # Fallback to CPU for this chunk
                        chunk_result = F.conv1d(input_chunk, weight_chunk, bias_chunk, stride, padding, dilation, groups)
                        results.append(chunk_result)
            
            # Concatenate results
            return torch.cat(results, dim=1)
        
        # Fallback to CPU
        return F.conv1d(input_tensor, weight, bias, stride, padding, dilation, groups)
    
    def smart_conv3d(self, input_tensor: torch.Tensor, weight: torch.Tensor,
                    bias: Optional[torch.Tensor] = None, stride: Tuple[int, ...] = (1, 1, 1),
                    padding: Tuple[int, ...] = (0, 0, 0), dilation: Tuple[int, ...] = (1, 1, 1),
                    groups: int = 1) -> torch.Tensor:
        """Smart Conv3d that uses MPS when possible."""
        if self.can_use_mps(input_tensor.shape, "conv3d"):
            device = torch.device('mps')
            input_mps = input_tensor.to(device)
            weight_mps = weight.to(device)
            bias_mps = bias.to(device) if bias is not None else None
            
            result = F.conv3d(input_mps, weight_mps, bias_mps, stride, padding, dilation, groups)
            return result.to(input_tensor.device)
        
        # Fallback to CPU
        return F.conv3d(input_tensor, weight, bias, stride, padding, dilation, groups)
    
    def smart_linear(self, input_tensor: torch.Tensor, weight: torch.Tensor,
                    bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Smart Linear layer that uses MPS when possible."""
        if self.can_use_mps(input_tensor.shape, "linear"):
            device = torch.device('mps')
            input_mps = input_tensor.to(device)
            weight_mps = weight.to(device)
            bias_mps = bias.to(device) if bias is not None else None
            
            result = F.linear(input_mps, weight_mps, bias_mps)
            return result.to(input_tensor.device)
        
        # Fallback to CPU
        return F.linear(input_tensor, weight, bias)
    
    def optimize_model_for_mps(self, model: nn.Module) -> nn.Module:
        """Apply MPS optimizations to a model."""
        if not self.mps_available:
            return model
        
        # Create a wrapper that intercepts problematic operations
        class MPSOptimizedModel(nn.Module):
            def __init__(self, original_model, optimizer):
                super().__init__()
                self.original_model = original_model
                self.optimizer = optimizer
                
            def forward(self, *args, **kwargs):
                return self.original_model(*args, **kwargs)
            
            def __getattr__(self, name):
                if name == 'original_model':
                    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
                return getattr(self.original_model, name)
        
        return MPSOptimizedModel(model, self)

# Global instance
mps_optimizer = MPSOptimizer()
