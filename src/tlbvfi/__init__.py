"""
TLBVFI - Temporal-Aware Latent Brownian Bridge Diffusion for Video Frame Interpolation

This is an integrated version of TLBVFI with optimizations for Apple Silicon and cross-platform compatibility.

Original Authors: Zonglin Lyu, Chen Chen
Paper: TLB-VFI: Temporal-Aware Latent Brownian Bridge Diffusion for Video Frame Interpolation
"""

__version__ = "1.0.0"
__author__ = "Zonglin Lyu, Chen Chen (with optimizations by this repository maintainer)"

from .core.interpolate_one import interpolate_one
from .core.interpolate import interpolate

__all__ = ['interpolate_one', 'interpolate']
