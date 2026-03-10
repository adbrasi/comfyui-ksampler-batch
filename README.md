# ComfyUI KSampler Batch Seeds

Custom nodes for ComfyUI that generate multiple images in a single GPU batch with different seeds.

## Nodes

### KSampler (Batch Seeds)

Drop-in replacement for KSampler that adds a `batch_size` parameter. Each image in the batch uses `seed + i`, so every result is independently reproducible.

### KSampler Advanced (Batch Seeds)

Same as above but with start/end step control, noise toggle, and leftover noise options — matching KSampler Advanced's interface.

## How It Works

Instead of generating one image at a time, these nodes:

1. Replicate the input latent `batch_size` times
2. Generate noise with truly different seeds per item (`seed+0`, `seed+1`, `seed+2`, …)
3. Pass the entire batch through the model in a **single forward pass**

This maximizes GPU utilization — if your model uses 2GB and you have 8GB of VRAM, you can generate 2-3 images simultaneously.

## Installation

Clone into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/adolfocesar/comfyui-ksampler-batch.git
```

## Reproducing a Specific Image

If you generated a batch with `seed=42` and `batch_size=4`, and liked image #3 (index 2):
- Use the standard KSampler with `seed=44` (42 + 2) to reproduce that exact image.
