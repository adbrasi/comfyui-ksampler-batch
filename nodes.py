import time
import logging

import torch
import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.utils
import latent_preview

logger = logging.getLogger("KSamplerBatch")

HEADER = "\033[95m[KSampler Batch]\033[0m"


def _repeat_batch(tensor, batch_size):
    """Repeat tensor along dim 0, handling any number of dimensions."""
    repeats = [batch_size] + [1] * (tensor.ndim - 1)
    return tensor.repeat(*repeats)


def _generate_batch_noise(latent_image, seed, batch_size, seed_mode="incremental"):
    """Generate noise with different seeds for each batch item.

    seed_mode:
      - "incremental": seed+0, seed+1, seed+2, … (reproducible individually)
      - "random": each item gets a random seed derived from the base seed
      - "fixed": all items share the same seed (same image, useful for testing)
    """
    total_batch = latent_image.shape[0]
    shape = [1] + list(latent_image.shape[1:])
    target_dtype = latent_image.dtype

    # Pre-compute seeds based on mode
    if seed_mode == "random":
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        item_seeds = [int(torch.randint(0, 2**63, (1,), generator=rng).item()) for _ in range(batch_size)]
        print(f"{HEADER} Seed mode: RANDOM (derived from base seed {seed})")
    elif seed_mode == "fixed":
        item_seeds = [seed] * batch_size
        print(f"{HEADER} Seed mode: FIXED (all items use seed {seed})")
    else:
        item_seeds = [seed + i for i in range(batch_size)]
        print(f"{HEADER} Seed mode: INCREMENTAL (seed, seed+1, seed+2, ...)")

    noises = []
    for i in range(total_batch):
        item_seed = item_seeds[i % batch_size]
        generator = torch.manual_seed(item_seed)
        noise = torch.randn(
            shape,
            dtype=torch.float32,
            layout=latent_image.layout,
            generator=generator,
            device="cpu",
        )
        noises.append(noise)
        print(f"{HEADER} Noise item {i}: seed={item_seed}, shape={list(noise.shape)}, dtype=float32→{target_dtype}")
    result = torch.cat(noises, dim=0).to(dtype=target_dtype)
    print(f"{HEADER} Final noise tensor: shape={list(result.shape)}, dtype={result.dtype}")
    return result


def _log_latent_info(label, latent_dict):
    """Print useful info about a LATENT dict."""
    samples = latent_dict["samples"]
    keys = [k for k in latent_dict.keys() if k != "samples"]
    print(f"{HEADER} {label}: shape={list(samples.shape)}, dtype={samples.dtype}, extra_keys={keys}")


def _log_vram():
    """Print current VRAM usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"{HEADER} VRAM: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.2f}GB total")


class KSamplerBatch:
    """KSampler that generates multiple images in a single GPU batch.

    Each batch item uses a different seed (seed, seed+1, seed+2, …)
    so every result is independently reproducible.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Base seed. Item i uses seed + i.",
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 10000,
                }),
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "round": 0.01,
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "batch_size": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Number of images to generate in parallel on the GPU",
                }),
                "seed_mode": (["incremental", "random", "fixed"], {
                    "default": "incremental",
                    "tooltip": (
                        "incremental: seed+0, seed+1, seed+2… (each image reproducible individually). "
                        "random: each item gets a unique random seed derived from the base seed. "
                        "fixed: all items use the same seed (identical images, useful for testing)."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/batch"
    DESCRIPTION = (
        "Like KSampler but generates batch_size images at once on the GPU. "
        "Each image uses seed+i so results are individually reproducible."
    )

    def sample(
        self, model, seed, steps, cfg, sampler_name, scheduler,
        positive, negative, latent_image, denoise, batch_size, seed_mode,
    ):
        print(f"\n{'='*60}")
        print(f"{HEADER} KSampler Batch — START")
        print(f"{HEADER} Config: batch_size={batch_size}, seed={seed}, steps={steps}, cfg={cfg}")
        print(f"{HEADER} Sampler: {sampler_name}, Scheduler: {scheduler}, Denoise: {denoise}")
        print(f"{HEADER} Seed mode: {seed_mode}")
        _log_latent_info("Input latent", latent_image)
        _log_vram()

        latent = latent_image["samples"]
        latent = comfy.sample.fix_empty_latent_channels(
            model, latent, latent_image.get("downscale_ratio_spacial"),
        )

        # Replicate latent for batch
        if batch_size > 1:
            batched_latent = _repeat_batch(latent, batch_size)
            print(f"{HEADER} Replicated latent: {list(latent.shape)} → {list(batched_latent.shape)}")
        else:
            batched_latent = latent
            print(f"{HEADER} batch_size=1, no replication needed")

        # Generate noise with different seeds per item
        print(f"{HEADER} Generating noise...")
        noise = _generate_batch_noise(batched_latent, seed, batch_size, seed_mode)

        # Replicate noise_mask if present
        noise_mask = None
        if "noise_mask" in latent_image:
            mask = latent_image["noise_mask"]
            if mask.shape[0] < batched_latent.shape[0]:
                noise_mask = _repeat_batch(mask, batch_size)
                print(f"{HEADER} Noise mask replicated: {list(mask.shape)} → {list(noise_mask.shape)}")
            else:
                noise_mask = mask
                print(f"{HEADER} Noise mask: using original {list(mask.shape)}")
        else:
            print(f"{HEADER} No noise_mask (not inpainting)")

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        callback = latent_preview.prepare_callback(model, steps)

        print(f"{HEADER} Starting sampling (disable_noise=True, we handle noise)...")
        _log_vram()
        t_start = time.time()

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, batched_latent,
            denoise=denoise,
            disable_noise=True,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )

        t_elapsed = time.time() - t_start
        print(f"{HEADER} Sampling DONE in {t_elapsed:.2f}s")
        print(f"{HEADER} Output: shape={list(samples.shape)}, dtype={samples.dtype}")
        print(f"{HEADER} Time per image: {t_elapsed / batch_size:.2f}s ({batch_size} images)")
        print(f"{HEADER} vs sequential estimate: {t_elapsed:.2f}s vs ~{t_elapsed / batch_size * batch_size:.0f}s×{batch_size}={t_elapsed / batch_size * batch_size:.0f}s")
        _log_vram()
        print(f"{'='*60}\n")

        out = latent_image.copy()
        out["samples"] = samples
        out.pop("downscale_ratio_spacial", None)
        return (out,)


class KSamplerBatchAdvanced:
    """Advanced KSampler with batch generation and step control.

    Combines batch seed generation with start/end step control
    and noise addition options from KSampler Advanced.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"],),
                "noise_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Base seed. Item i uses seed + i.",
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 10000,
                }),
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "round": 0.01,
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                }),
                "end_at_step": ("INT", {
                    "default": 10000,
                    "min": 0,
                    "max": 10000,
                }),
                "return_with_leftover_noise": (["disable", "enable"],),
                "batch_size": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Number of images to generate in parallel on the GPU",
                }),
                "seed_mode": (["incremental", "random", "fixed"], {
                    "default": "incremental",
                    "tooltip": (
                        "incremental: seed+0, seed+1, seed+2… (each image reproducible individually). "
                        "random: each item gets a unique random seed derived from the base seed. "
                        "fixed: all items use the same seed (identical images, useful for testing)."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/batch"
    DESCRIPTION = (
        "Advanced KSampler with batch generation. "
        "Each image uses noise_seed+i. Supports start/end step control."
    )

    def sample(
        self, model, add_noise, noise_seed, steps, cfg, sampler_name,
        scheduler, positive, negative, latent_image, start_at_step,
        end_at_step, return_with_leftover_noise, batch_size, seed_mode,
    ):
        print(f"\n{'='*60}")
        print(f"{HEADER} KSampler Batch Advanced — START")
        print(f"{HEADER} Config: batch_size={batch_size}, seed={noise_seed}, steps={steps}, cfg={cfg}")
        print(f"{HEADER} Sampler: {sampler_name}, Scheduler: {scheduler}")
        print(f"{HEADER} Steps range: {start_at_step} → {end_at_step}")
        print(f"{HEADER} add_noise={add_noise}, return_leftover_noise={return_with_leftover_noise}")
        print(f"{HEADER} Seed mode: {seed_mode}")
        _log_latent_info("Input latent", latent_image)
        _log_vram()

        latent = latent_image["samples"]
        latent = comfy.sample.fix_empty_latent_channels(
            model, latent, latent_image.get("downscale_ratio_spacial"),
        )

        # Replicate latent for batch
        if batch_size > 1:
            batched_latent = _repeat_batch(latent, batch_size)
            print(f"{HEADER} Replicated latent: {list(latent.shape)} → {list(batched_latent.shape)}")
        else:
            batched_latent = latent
            print(f"{HEADER} batch_size=1, no replication needed")

        # Generate noise
        force_full_denoise = return_with_leftover_noise != "enable"
        disable_noise = add_noise != "enable"

        if disable_noise:
            noise = torch.zeros(
                batched_latent.size(),
                dtype=batched_latent.dtype,
                layout=batched_latent.layout,
                device="cpu",
            )
            print(f"{HEADER} Noise: DISABLED (zeros), shape={list(noise.shape)}")
        else:
            print(f"{HEADER} Generating noise...")
            noise = _generate_batch_noise(batched_latent, noise_seed, batch_size, seed_mode)

        # Replicate noise_mask if present
        noise_mask = None
        if "noise_mask" in latent_image:
            mask = latent_image["noise_mask"]
            if mask.shape[0] < batched_latent.shape[0]:
                noise_mask = _repeat_batch(mask, batch_size)
                print(f"{HEADER} Noise mask replicated: {list(mask.shape)} → {list(noise_mask.shape)}")
            else:
                noise_mask = mask
                print(f"{HEADER} Noise mask: using original {list(mask.shape)}")
        else:
            print(f"{HEADER} No noise_mask (not inpainting)")

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        callback = latent_preview.prepare_callback(model, steps)

        print(f"{HEADER} Starting sampling (disable_noise=True, force_full_denoise={force_full_denoise})...")
        _log_vram()
        t_start = time.time()

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, batched_latent,
            denoise=1.0,
            disable_noise=True,
            start_step=start_at_step,
            last_step=end_at_step,
            force_full_denoise=force_full_denoise,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise_seed,
        )

        t_elapsed = time.time() - t_start
        print(f"{HEADER} Sampling DONE in {t_elapsed:.2f}s")
        print(f"{HEADER} Output: shape={list(samples.shape)}, dtype={samples.dtype}")
        print(f"{HEADER} Time per image: {t_elapsed / batch_size:.2f}s ({batch_size} images)")
        _log_vram()
        print(f"{'='*60}\n")

        out = latent_image.copy()
        out["samples"] = samples
        out.pop("downscale_ratio_spacial", None)
        return (out,)
