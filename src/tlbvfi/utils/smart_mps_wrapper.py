"""
Smart MPS wrapper that intercepts operations and optimizes them for Apple Silicon GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mps_optimizer import mps_optimizer
import types

class SmartMPSWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.original_model = model
        self.optimizer = mps_optimizer
        
        # Intercept problematic operations
        self._patch_conv_operations()
        
    def _patch_conv_operations(self):
        """Patch convolution operations to use smart MPS optimization."""
        def patch_module(module):
            for name, child in module.named_children():
                if isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    # Replace with smart version
                    setattr(module, name, SmartConvLayer(child, self.optimizer))
                elif isinstance(child, nn.Linear):
                    # Replace with smart version
                    setattr(module, name, SmartLinearLayer(child, self.optimizer))
                else:
                    # Recursively patch submodules
                    patch_module(child)
        
        patch_module(self.original_model)
    
    def forward(self, *args, **kwargs):
        """Forward pass with MPS optimization."""
        return self.original_model(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original model."""
        if name == 'original_model':
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self.original_model, name)

class SmartConvLayer(nn.Module):
    def __init__(self, original_conv, optimizer):
        super().__init__()
        self.original_conv = original_conv
        self.optimizer = optimizer
        
        # Copy all attributes
        for attr_name in dir(original_conv):
            if not attr_name.startswith('_') and not callable(getattr(original_conv, attr_name)):
                setattr(self, attr_name, getattr(original_conv, attr_name))
    
    def forward(self, x):
        """Smart forward pass that uses MPS when possible."""
        if isinstance(self.original_conv, nn.Conv1d):
            return self.optimizer.smart_conv1d(
                x, self.original_conv.weight, self.original_conv.bias,
                self.original_conv.stride[0], self.original_conv.padding[0],
                self.original_conv.dilation[0], self.original_conv.groups
            )
        elif isinstance(self.original_conv, nn.Conv2d):
            # For Conv2d, try MPS first, fallback to CPU
            if self.optimizer.can_use_mps(x.shape, "conv2d"):
                device = torch.device('mps')
                x_mps = x.to(device)
                weight_mps = self.original_conv.weight.to(device)
                bias_mps = self.original_conv.bias.to(device) if self.original_conv.bias is not None else None
                
                result = F.conv2d(x_mps, weight_mps, bias_mps, self.original_conv.stride,
                                self.original_conv.padding, self.original_conv.dilation,
                                self.original_conv.groups)
                return result.to(x.device)
            else:
                return self.original_conv(x)
        elif isinstance(self.original_conv, nn.Conv3d):
            return self.optimizer.smart_conv3d(
                x, self.original_conv.weight, self.original_conv.bias,
                self.original_conv.stride, self.original_conv.padding,
                self.original_conv.dilation, self.original_conv.groups
            )
        else:
            return self.original_conv(x)

class SmartLinearLayer(nn.Module):
    def __init__(self, original_linear, optimizer):
        super().__init__()
        self.original_linear = original_linear
        self.optimizer = optimizer
        
        # Copy all attributes
        for attr_name in dir(original_linear):
            if not attr_name.startswith('_') and not callable(getattr(original_linear, attr_name)):
                setattr(self, attr_name, getattr(original_linear, attr_name))
    
    def forward(self, x):
        """Smart forward pass that uses MPS when possible."""
        return self.optimizer.smart_linear(x, self.original_linear.weight, self.original_linear.bias)

class HybridModelWrapper(nn.Module):
    """Advanced wrapper that can switch between MPS and CPU during inference."""
    
    def __init__(self, model):
        super().__init__()
        self._original_model = model
        self._optimizer = mps_optimizer
        self._mps_available = torch.backends.mps.is_available()
        
        # Statistics
        self._mps_operations = 0
        self._cpu_operations = 0
        
    def forward(self, *args, **kwargs):
        """Hybrid forward pass with intelligent device switching."""
        try:
            # Try MPS first
            if self._mps_available:
                device = torch.device('mps')
                mps_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        mps_args.append(arg.to(device))
                    else:
                        mps_args.append(arg)
                
                # Move model to MPS temporarily
                original_device = next(self._original_model.parameters()).device
                self._original_model.to(device)
                
                try:
                    result = self._original_model(*mps_args, **kwargs)
                    self._mps_operations += 1
                    return result.to(original_device)
                except RuntimeError as e:
                    if "not supported at the MPS device" in str(e):
                        print(f"⚠️  MPS operation failed, switching to CPU: {e}")
                        self._cpu_operations += 1
                        # Move model back to original device
                        self._original_model.to(original_device)
                        # Fallback to CPU
                        return self._original_model(*args, **kwargs)
                    else:
                        raise e
                finally:
                    # Always move model back to original device
                    self._original_model.to(original_device)
            else:
                # No MPS available, use CPU
                self._cpu_operations += 1
                return self._original_model(*args, **kwargs)
                
        except Exception as e:
            print(f"❌ Hybrid forward failed: {e}")
            self._cpu_operations += 1
            return self._original_model(*args, **kwargs)
    
    def get_stats(self):
        """Get operation statistics."""
        total = self._mps_operations + self._cpu_operations
        if total == 0:
            return "No operations performed"
        
        mps_percent = (self._mps_operations / total) * 100
        return f"MPS: {self._mps_operations} ({mps_percent:.1f}%), CPU: {self._cpu_operations} ({100-mps_percent:.1f}%)"
    
    def get_original_model(self):
        """Access to the original model."""
        return self._original_model
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original model."""
        if name.startswith('_') or name in ['get_original_model', 'get_stats']:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self._original_model, name)
