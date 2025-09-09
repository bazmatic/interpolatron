"""
Smart model wrapper that handles device switching for operations that exceed MPS limits.
"""

import torch
import torch.nn as nn
from hybrid_device import device_manager

class SmartModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self._model = model
        self.original_forward = model.forward
        self.original_sample = getattr(model, 'sample', None)
        
    def forward(self, *args, **kwargs):
        """Smart forward pass with device switching."""
        try:
            return self.original_forward(*args, **kwargs)
        except RuntimeError as e:
            if "not supported at the MPS device" in str(e):
                print(f"⚠️  MPS operation failed, switching to CPU: {e}")
                # Move model to CPU temporarily
                original_device = next(self._model.parameters()).device
                self._model.to('cpu')
                
                # Move inputs to CPU
                cpu_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        cpu_args.append(arg.to('cpu'))
                    else:
                        cpu_args.append(arg)
                
                # Run on CPU
                result = self.original_forward(*cpu_args, **kwargs)
                
                # Move model back to original device
                self._model.to(original_device)
                
                return result
            else:
                raise e
    
    def sample(self, *args, **kwargs):
        """Smart sample method with device switching."""
        if self.original_sample is None:
            return self.forward(*args, **kwargs)
        
        try:
            return self.original_sample(*args, **kwargs)
        except RuntimeError as e:
            if "not supported at the MPS device" in str(e):
                print(f"⚠️  MPS sample operation failed, switching to CPU: {e}")
                # Move model to CPU temporarily
                original_device = next(self._model.parameters()).device
                self._model.to('cpu')
                
                # Move inputs to CPU
                cpu_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        cpu_args.append(arg.to('cpu'))
                    else:
                        cpu_args.append(arg)
                
                # Run on CPU
                result = self.original_sample(*cpu_args, **kwargs)
                
                # Move model back to original device
                self._model.to(original_device)
                
                return result
            else:
                raise e
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped model."""
        # Don't delegate access to our internal attributes
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self._model, name)
