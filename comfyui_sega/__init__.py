"""
ComfyUI-SEGA: Spectral-Energy Guided Attention for Resolution Extrapolation
Custom nodes for Graydient.AI / ComfyUI
Based on: https://github.com/rajabi2001/sega
"""

from .nodes import (
    SEGAModelPatch,
    SEGASampler,
    SEGAConfig,
)

NODE_CLASS_MAPPINGS = {
    "SEGAModelPatch": SEGAModelPatch,
    "SEGASampler": SEGASampler,
    "SEGAConfig": SEGAConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SEGAModelPatch": "SEGA Model Patch",
    "SEGASampler": "SEGA Sampler",
    "SEGAConfig": "SEGA Config",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
