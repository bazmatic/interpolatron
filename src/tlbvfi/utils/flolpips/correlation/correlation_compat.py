#!/usr/bin/env python

import torch
import sys
import warnings

# Try to import CuPy
try:
    import cupy
    CUPY_AVAILABLE = True
    print("✓ CuPy is available - using CUDA kernels for correlation")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠ CuPy not available - using CPU fallback for correlation")

# Import the original CuPy-based correlation
if CUPY_AVAILABLE:
    try:
        from .correlation import FunctionCorrelation as FunctionCorrelationCupy
        from .correlation import ModuleCorrelation as ModuleCorrelationCupy
        CUPY_CORRELATION_AVAILABLE = True
    except ImportError as e:
        print(f"⚠ Could not import CuPy correlation: {e}")
        CUPY_CORRELATION_AVAILABLE = False
else:
    CUPY_CORRELATION_AVAILABLE = False

# Import CPU fallback
try:
    from .correlation_cpu import FunctionCorrelationCPU, ModuleCorrelationCPU
    from .correlation_cpu import FunctionCorrelationOptimized, ModuleCorrelationOptimized
    CPU_CORRELATION_AVAILABLE = True
except ImportError as e:
    print(f"✗ Could not import CPU correlation fallback: {e}")
    CPU_CORRELATION_AVAILABLE = False
    sys.exit(1)


def FunctionCorrelation(tenFirst, tenSecond):
    """
    Compatibility function that automatically chooses between CuPy and CPU implementations.
    
    Args:
        tenFirst: First input tensor [B, C, H, W]
        tenSecond: Second input tensor [B, C, H, W]
        
    Returns:
        Correlation output [B, 81, H, W]
    """
    # Check if tensors are on CUDA and CuPy is available
    if (tenFirst.is_cuda and tenSecond.is_cuda and 
        CUPY_AVAILABLE and CUPY_CORRELATION_AVAILABLE):
        try:
            return FunctionCorrelationCupy(tenFirst, tenSecond)
        except Exception as e:
            print(f"⚠ CuPy correlation failed, falling back to CPU: {e}")
            # Fall back to CPU implementation
            tenFirst_cpu = tenFirst.cpu()
            tenSecond_cpu = tenSecond.cpu()
            result = FunctionCorrelationOptimized(tenFirst_cpu, tenSecond_cpu)
            return result.to(tenFirst.device)
    else:
        # Use CPU implementation
        if tenFirst.is_cuda or tenSecond.is_cuda:
            # Move to CPU for computation
            tenFirst_cpu = tenFirst.cpu()
            tenSecond_cpu = tenSecond.cpu()
            result = FunctionCorrelationOptimized(tenFirst_cpu, tenSecond_cpu)
            return result.to(tenFirst.device)
        else:
            # Already on CPU
            return FunctionCorrelationOptimized(tenFirst, tenSecond)


class ModuleCorrelation(torch.nn.Module):
    """
    Compatibility module that automatically chooses between CuPy and CPU implementations.
    """
    
    def __init__(self):
        super(ModuleCorrelation, self).__init__()
        self._cupy_available = CUPY_AVAILABLE and CUPY_CORRELATION_AVAILABLE
    
    def forward(self, tenFirst, tenSecond):
        return FunctionCorrelation(tenFirst, tenSecond)


# Export the compatibility functions
__all__ = ['FunctionCorrelation', 'ModuleCorrelation', 'CUPY_AVAILABLE', 'CPU_CORRELATION_AVAILABLE']
