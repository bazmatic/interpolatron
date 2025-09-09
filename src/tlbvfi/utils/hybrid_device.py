"""
Hybrid device manager for TLBVFI that intelligently switches between MPS and CPU
based on tensor sizes and operation compatibility.
"""

import torch
import warnings

class HybridDeviceManager:
    def __init__(self):
        self.mps_available = torch.backends.mps.is_available()
        self.cuda_available = torch.cuda.is_available()
        self.mps_limits = {
            'max_channels': 65536,
            'max_elements': 2**31,  # Conservative limit
        }
        
        # Determine primary device
        if self.mps_available:
            self.primary_device = torch.device('mps')
            self.fallback_device = torch.device('cpu')
            print("🚀 Hybrid device manager: MPS + CPU fallback")
        elif self.cuda_available:
            self.primary_device = torch.device('cuda')
            self.fallback_device = torch.device('cpu')
            print("🚀 Hybrid device manager: CUDA + CPU fallback")
        else:
            self.primary_device = torch.device('cpu')
            self.fallback_device = torch.device('cpu')
            print("💻 Hybrid device manager: CPU only")
    
    def get_device_for_tensor(self, tensor_shape, operation_type="conv"):
        """
        Determine the best device for a tensor operation based on its shape.
        
        Args:
            tensor_shape: Shape of the tensor (list or tuple)
            operation_type: Type of operation ("conv", "linear", "general")
        
        Returns:
            torch.device: Best device for this operation
        """
        if not self.mps_available:
            return self.fallback_device
        
        # Check if tensor would exceed MPS limits
        if self._would_exceed_mps_limits(tensor_shape, operation_type):
            return self.fallback_device
        
        return self.primary_device
    
    def _would_exceed_mps_limits(self, tensor_shape, operation_type):
        """Check if a tensor operation would exceed MPS limits."""
        if not self.mps_available:
            return False
        
        # Calculate total elements
        total_elements = 1
        for dim in tensor_shape:
            total_elements *= dim
        
        # Check element count limit
        if total_elements > self.mps_limits['max_elements']:
            return True
        
        # Check channel limit for convolution operations
        if operation_type in ["conv", "conv1d", "conv2d", "conv3d"] and len(tensor_shape) >= 2:
            channels = tensor_shape[1]  # Second dimension is usually channels
            if channels > self.mps_limits['max_channels']:
                return True
        
        return False
    
    def move_tensor_smart(self, tensor, operation_type="general"):
        """
        Move tensor to the best device for its operation.
        
        Args:
            tensor: PyTorch tensor
            operation_type: Type of operation to be performed
        
        Returns:
            tensor: Tensor moved to appropriate device
        """
        target_device = self.get_device_for_tensor(tensor.shape, operation_type)
        return tensor.to(target_device)
    
    def move_model_smart(self, model, sample_input=None):
        """
        Move model to the best device, with fallback if needed.
        
        Args:
            model: PyTorch model
            sample_input: Optional sample input to test compatibility
        
        Returns:
            model: Model moved to appropriate device
        """
        if not self.mps_available:
            return model.to(self.fallback_device)
        
        try:
            # Try moving to MPS first
            model = model.to(self.primary_device)
            
            # Test with sample input if provided
            if sample_input is not None:
                with torch.no_grad():
                    _ = model(sample_input.to(self.primary_device))
            
            return model
        except Exception as e:
            print(f"⚠️  MPS model test failed: {e}")
            print("🔄 Falling back to CPU for model")
            return model.to(self.fallback_device)
    
    def get_optimal_device(self):
        """Get the optimal device for general operations."""
        return self.primary_device if self.mps_available else self.fallback_device

# Global instance
device_manager = HybridDeviceManager()
