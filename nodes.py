import torch
import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.utils
import latent_preview


def _generate_batch_noise(latent_image, seed, batch_size):
    """Generate noise with truly different seeds for each batch item.

    Each item i uses torch.manual_seed(seed + i) so every image is
    independently reproducible.
    """
    shape = [1] + list(latent_image.shape[1:])
    noises = []
    for i in range(batch_size):
        generator = torch.manual_seed(seed + i)
        noise = torch.randn(
            shape,
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            generator=generator,
            device="cpu",
        )
        noises.append(noise)
    return torch.cat(noises, dim=0)


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
        positive, negative, latent_image, denoise, batch_size,
    ):
        latent = latent_image["samples"]
        latent = comfy.sample.fix_empty_latent_channels(
            model, latent, latent_image.get("downscale_ratio_spacial"),
        )

        # Replicate latent for batch
        if batch_size > 1:
            batched_latent = latent.repeat(batch_size, 1, 1, 1)
        else:
            batched_latent = latent

        # Generate noise with different seeds per item
        noise = _generate_batch_noise(batched_latent, seed, batch_size)

        # Replicate noise_mask if present
        noise_mask = None
        if "noise_mask" in latent_image:
            mask = latent_image["noise_mask"]
            if batch_size > 1 and mask.shape[0] < batch_size:
                noise_mask = mask.repeat(batch_size, 1, 1, 1)[:batch_size]
            else:
                noise_mask = mask

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        callback = latent_preview.prepare_callback(model, steps)

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, batched_latent,
            denoise=denoise,
            disable_noise=True,  # We already generated noise
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )

        out = latent_image.copy()
        out["samples"] = samples
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
        end_at_step, return_with_leftover_noise, batch_size,
    ):
        latent = latent_image["samples"]
        latent = comfy.sample.fix_empty_latent_channels(
            model, latent, latent_image.get("downscale_ratio_spacial"),
        )

        # Replicate latent for batch
        if batch_size > 1:
            batched_latent = latent.repeat(batch_size, 1, 1, 1)
        else:
            batched_latent = latent

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
        else:
            noise = _generate_batch_noise(batched_latent, noise_seed, batch_size)

        # Replicate noise_mask if present
        noise_mask = None
        if "noise_mask" in latent_image:
            mask = latent_image["noise_mask"]
            if batch_size > 1 and mask.shape[0] < batch_size:
                noise_mask = mask.repeat(batch_size, 1, 1, 1)[:batch_size]
            else:
                noise_mask = mask

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        callback = latent_preview.prepare_callback(model, steps)

        samples = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, batched_latent,
            denoise=1.0,
            disable_noise=True,  # We handle noise ourselves
            start_step=start_at_step,
            last_step=end_at_step,
            force_full_denoise=force_full_denoise,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise_seed,
        )

        out = latent_image.copy()
        out["samples"] = samples
        return (out,)
