import torch
import sys
import warnings

# Try to import CuPy
try:
    import cupy
    CUPY_AVAILABLE = True
    print("✓ CuPy is available - using CUDA kernels for dsepconv")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠ CuPy not available - using CPU fallback for dsepconv")

# Import the original CuPy-based dsepconv
if CUPY_AVAILABLE:
    try:
        from .dsepconv import FunctionDSepconv as FunctionDSepconvCupy
        from .dsepconv import ModuleDSepconv as ModuleDSepconvCupy
        CUPY_DSEPCONV_AVAILABLE = True
    except ImportError as e:
        print(f"⚠ Could not import CuPy dsepconv: {e}")
        CUPY_DSEPCONV_AVAILABLE = False
else:
    CUPY_DSEPCONV_AVAILABLE = False

# Import CPU fallback
try:
    from .dsepconv_cpu import FunctionDSepconvCPU, ModuleDSepconvCPU
    from .dsepconv_cpu import FunctionDSepconvOptimized, ModuleDSepconvOptimized
    CPU_DSEPCONV_AVAILABLE = True
except ImportError as e:
    print(f"✗ Could not import CPU dsepconv fallback: {e}")
    CPU_DSEPCONV_AVAILABLE = False
    sys.exit(1)


def FunctionDSepconv(tensorInput, tensorVertical, tensorHorizontal, tensorOffsetX, tensorOffsetY, tensorMask):
    """
    Compatibility function that automatically chooses between CuPy and CPU implementations.
    
    Args:
        tensorInput: Input tensor [B, C, H, W]
        tensorVertical: Vertical filter [B, F, H, W]
        tensorHorizontal: Horizontal filter [B, F, H, W]
        tensorOffsetX: X offset [B, F*F, H, W]
        tensorOffsetY: Y offset [B, F*F, H, W]
        tensorMask: Mask [B, F*F, H, W]
        
    Returns:
        Output tensor [B, C, H, W]
    """
    # Check if tensors are on CUDA and CuPy is available
    if (tensorInput.is_cuda and CUPY_AVAILABLE and CUPY_DSEPCONV_AVAILABLE):
        try:
            return FunctionDSepconvCupy(tensorInput, tensorVertical, tensorHorizontal, 
                                      tensorOffsetX, tensorOffsetY, tensorMask)
        except Exception as e:
            print(f"⚠ CuPy dsepconv failed, falling back to CPU: {e}")
            # Fall back to CPU implementation
            tensorInput_cpu = tensorInput.cpu()
            tensorVertical_cpu = tensorVertical.cpu()
            tensorHorizontal_cpu = tensorHorizontal.cpu()
            tensorOffsetX_cpu = tensorOffsetX.cpu()
            tensorOffsetY_cpu = tensorOffsetY.cpu()
            tensorMask_cpu = tensorMask.cpu()
            
            result = FunctionDSepconvOptimized(tensorInput_cpu, tensorVertical_cpu, tensorHorizontal_cpu,
                                             tensorOffsetX_cpu, tensorOffsetY_cpu, tensorMask_cpu)
            return result.to(tensorInput.device)
    else:
        # Use CPU implementation
        if tensorInput.is_cuda:
            # Move to CPU for computation
            tensorInput_cpu = tensorInput.cpu()
            tensorVertical_cpu = tensorVertical.cpu()
            tensorHorizontal_cpu = tensorHorizontal.cpu()
            tensorOffsetX_cpu = tensorOffsetX.cpu()
            tensorOffsetY_cpu = tensorOffsetY.cpu()
            tensorMask_cpu = tensorMask.cpu()
            
            result = FunctionDSepconvOptimized(tensorInput_cpu, tensorVertical_cpu, tensorHorizontal_cpu,
                                             tensorOffsetX_cpu, tensorOffsetY_cpu, tensorMask_cpu)
            return result.to(tensorInput.device)
        else:
            # Already on CPU
            return FunctionDSepconvOptimized(tensorInput, tensorVertical, tensorHorizontal,
                                           tensorOffsetX, tensorOffsetY, tensorMask)


class ModuleDSepconv(torch.nn.Module):
    """
    Compatibility module that automatically chooses between CuPy and CPU implementations.
    """
    
    def __init__(self):
        super(ModuleDSepconv, self).__init__()
        self._cupy_available = CUPY_AVAILABLE and CUPY_DSEPCONV_AVAILABLE
    
    def forward(self, tensorInput, tensorVertical, tensorHorizontal, tensorOffsetX, tensorOffsetY, tensorMask):
        return FunctionDSepconv(tensorInput, tensorVertical, tensorHorizontal, tensorOffsetX, tensorOffsetY, tensorMask)


# Export the compatibility functions
__all__ = ['FunctionDSepconv', 'ModuleDSepconv', 'CUPY_AVAILABLE', 'CPU_DSEPCONV_AVAILABLE']
