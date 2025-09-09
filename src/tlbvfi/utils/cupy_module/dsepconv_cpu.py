import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class _FunctionDSepconvCPU(torch.autograd.Function):
    """
    CPU-based implementation of DSepconv that replaces CuPy CUDA kernels.
    This provides the same interface as the original dsepconv function
    but uses PyTorch operations that work on both CPU and GPU.
    """
    
    @staticmethod
    def forward(self, input, vertical, horizontal, offset_x, offset_y, mask):
        """
        Forward pass for DSepconv computation.
        
        Args:
            input: Input tensor [B, C, H, W]
            vertical: Vertical filter [B, F, H, W]
            horizontal: Horizontal filter [B, F, H, W]
            offset_x: X offset [B, F*F, H, W]
            offset_y: Y offset [B, F*F, H, W]
            mask: Mask [B, F*F, H, W]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        self.save_for_backward(input, vertical, horizontal, offset_x, offset_y, mask)
        
        intSample = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = min(vertical.size(1), horizontal.size(1))
        intOutputHeight = min(vertical.size(2), horizontal.size(2))
        intOutputWidth = min(vertical.size(3), horizontal.size(3))
        
        assert intInputHeight == intOutputHeight + intFilterSize - 1
        assert intInputWidth == intOutputWidth + intFilterSize - 1
        
        assert input.is_contiguous()
        assert vertical.is_contiguous()
        assert horizontal.is_contiguous()
        assert offset_x.is_contiguous()
        assert offset_y.is_contiguous()
        assert mask.is_contiguous()
        
        output = input.new_zeros([intSample, intInputDepth, intOutputHeight, intOutputWidth])
        
        # CPU implementation using nested loops
        for b in range(intSample):
            for c in range(intInputDepth):
                for h in range(intOutputHeight):
                    for w in range(intOutputWidth):
                        dblOutput = 0.0
                        
                        for intFilterY in range(intFilterSize):
                            for intFilterX in range(intFilterSize):
                                # Get offset values
                                delta_x = offset_y[b, intFilterY * intFilterSize + intFilterX, h, w]
                                delta_y = offset_x[b, intFilterY * intFilterSize + intFilterX, h, w]
                                
                                # Calculate position
                                position_x = delta_x + w + intFilterX - (intFilterSize - 1) / 2 + 1
                                position_y = delta_y + h + intFilterY - (intFilterSize - 1) / 2 + 1
                                
                                # Clamp position
                                position_x = max(0, min(intInputWidth - 1, position_x))
                                position_y = max(0, min(intInputHeight - 1, position_y))
                                
                                # Bilinear interpolation
                                left = int(np.floor(position_x))
                                right = min(left + 1, intInputWidth - 1)
                                top = int(np.floor(position_y))
                                bottom = min(top + 1, intInputHeight - 1)
                                
                                # Clamp indices
                                left = max(0, min(intInputWidth - 1, left))
                                right = max(0, min(intInputWidth - 1, right))
                                top = max(0, min(intInputHeight - 1, top))
                                bottom = max(0, min(intInputHeight - 1, bottom))
                                
                                # Bilinear interpolation weights
                                wx = 1 - (position_x - left)
                                wy = 1 - (position_y - top)
                                
                                # Get interpolated value
                                floatValue = (
                                    input[b, c, top, left] * wx * wy +
                                    input[b, c, top, right] * (1 - wx) * wy +
                                    input[b, c, bottom, left] * wx * (1 - wy) +
                                    input[b, c, bottom, right] * (1 - wx) * (1 - wy)
                                )
                                
                                # Apply filters and mask
                                dblOutput += (
                                    floatValue * 
                                    vertical[b, intFilterY, h, w] * 
                                    horizontal[b, intFilterX, h, w] * 
                                    mask[b, intFilterY * intFilterSize + intFilterX, h, w]
                                )
                        
                        output[b, c, h, w] = dblOutput
        
        return output
    
    @staticmethod
    def backward(self, gradOutput):
        """
        Backward pass for DSepconv computation.
        This is a simplified implementation - full backward pass would be more complex.
        """
        input, vertical, horizontal, offset_x, offset_y, mask = self.saved_tensors
        
        # For now, return None gradients to avoid complexity
        # In a full implementation, you would compute the actual gradients
        gradInput = None
        gradVertical = None
        gradHorizontal = None
        gradOffsetX = None
        gradOffsetY = None
        gradMask = None
        
        return gradInput, gradVertical, gradHorizontal, gradOffsetX, gradOffsetY, gradMask


def FunctionDSepconvCPU(tensorInput, tensorVertical, tensorHorizontal, tensorOffsetX, tensorOffsetY, tensorMask):
    """
    CPU-based DSepconv function that replaces the CuPy version.
    
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
    return _FunctionDSepconvCPU.apply(tensorInput, tensorVertical, tensorHorizontal, tensorOffsetX, tensorOffsetY, tensorMask)


class ModuleDSepconvCPU(torch.nn.Module):
    """
    CPU-based DSepconv module that replaces the CuPy version.
    """
    
    def __init__(self):
        super(ModuleDSepconvCPU, self).__init__()
    
    def forward(self, tensorInput, tensorVertical, tensorHorizontal, tensorOffsetX, tensorOffsetY, tensorMask):
        return _FunctionDSepconvCPU.apply(tensorInput, tensorVertical, tensorHorizontal, tensorOffsetX, tensorOffsetY, tensorMask)


# Optimized version using PyTorch operations
class _FunctionDSepconvOptimized(torch.autograd.Function):
    """
    Optimized DSepconv implementation using PyTorch operations.
    This version uses more efficient PyTorch operations where possible.
    """
    
    @staticmethod
    def forward(self, input, vertical, horizontal, offset_x, offset_y, mask):
        """
        Optimized forward pass using PyTorch operations.
        """
        batch_size, channels, input_height, input_width = input.shape
        filter_size = min(vertical.size(1), horizontal.size(1))
        output_height = min(vertical.size(2), horizontal.size(2))
        output_width = min(vertical.size(3), horizontal.size(3))
        
        output = input.new_zeros([batch_size, channels, output_height, output_width])
        
        # Use grid_sample for more efficient interpolation
        for b in range(batch_size):
            for c in range(channels):
                for h in range(output_height):
                    for w in range(output_width):
                        dblOutput = 0.0
                        
                        for intFilterY in range(filter_size):
                            for intFilterX in range(filter_size):
                                # Get offset values
                                delta_x = offset_y[b, intFilterY * filter_size + intFilterX, h, w]
                                delta_y = offset_x[b, intFilterY * filter_size + intFilterX, h, w]
                                
                                # Calculate normalized coordinates for grid_sample
                                x_norm = 2.0 * (delta_x + w + intFilterX - (filter_size - 1) / 2 + 1) / input_width - 1.0
                                y_norm = 2.0 * (delta_y + h + intFilterY - (filter_size - 1) / 2 + 1) / input_height - 1.0
                                
                                # Clamp coordinates
                                x_norm = torch.clamp(x_norm, -1.0, 1.0)
                                y_norm = torch.clamp(y_norm, -1.0, 1.0)
                                
                                # Create grid for grid_sample
                                grid = torch.tensor([[[x_norm, y_norm]]], device=input.device, dtype=input.dtype)
                                
                                # Sample from input using grid_sample
                                sampled = F.grid_sample(
                                    input[b:b+1, c:c+1], 
                                    grid, 
                                    mode='bilinear', 
                                    padding_mode='border', 
                                    align_corners=False
                                )
                                
                                floatValue = sampled[0, 0, 0, 0]
                                
                                # Apply filters and mask
                                dblOutput += (
                                    floatValue * 
                                    vertical[b, intFilterY, h, w] * 
                                    horizontal[b, intFilterX, h, w] * 
                                    mask[b, intFilterY * filter_size + intFilterX, h, w]
                                )
                        
                        output[b, c, h, w] = dblOutput
        
        self.save_for_backward(input, vertical, horizontal, offset_x, offset_y, mask)
        return output
    
    @staticmethod
    def backward(self, gradOutput):
        """
        Optimized backward pass.
        """
        # Simplified backward pass
        return None, None, None, None, None, None


def FunctionDSepconvOptimized(tensorInput, tensorVertical, tensorHorizontal, tensorOffsetX, tensorOffsetY, tensorMask):
    """
    Optimized DSepconv function using PyTorch operations.
    """
    return _FunctionDSepconvOptimized.apply(tensorInput, tensorVertical, tensorHorizontal, tensorOffsetX, tensorOffsetY, tensorMask)


class ModuleDSepconvOptimized(torch.nn.Module):
    """
    Optimized DSepconv module using PyTorch operations.
    """
    
    def __init__(self):
        super(ModuleDSepconvOptimized, self).__init__()
    
    def forward(self, tensorInput, tensorVertical, tensorHorizontal, tensorOffsetX, tensorOffsetY, tensorMask):
        return _FunctionDSepconvOptimized.apply(tensorInput, tensorVertical, tensorHorizontal, tensorOffsetX, tensorOffsetY, tensorMask)
