from .nodes import KSamplerBatch, KSamplerBatchAdvanced

NODE_CLASS_MAPPINGS = {
    "KSamplerBatch": KSamplerBatch,
    "KSamplerBatchAdvanced": KSamplerBatchAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerBatch": "KSampler (Batch Seeds)",
    "KSamplerBatchAdvanced": "KSampler Advanced (Batch Seeds)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
