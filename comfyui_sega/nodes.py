"""
ComfyUI-SEGA Nodes
Nodes for Graydient.AI workflow builder
"""

import torch
import comfy.model_management as mm
from .sega_transformer import SEGATransformerPatcher

class SEGAConfig:
    """
    SEGA hyperparameter configuration node.
    Outputs a config dict that controls spectral rescaling behavior.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "mscale_alpha": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mscale_beta": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1}),
                "mscale_min": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.01}),
                "base_mscale_formula": (["power_res", "log_res"], {"default": "power_res"}),
                "base_mscale_coefficient": ("FLOAT", {"default": 0.08, "min": 0.01, "max": 0.5, "step": 0.01}),
                "spread_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "spread_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "spread_alpha": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("SEGA_CONFIG",)
    RETURN_NAMES = ("sega_config",)
    FUNCTION = "make_config"
    CATEGORY = "SEGA"

    def make_config(self, training_resolution, mscale_alpha, mscale_beta, mscale_min,
                    base_mscale_formula, base_mscale_coefficient, spread_min, spread_max, spread_alpha):
        config = {
            'training_resolution': training_resolution,
            'mscale_alpha': mscale_alpha,
            'mscale_beta': mscale_beta,
            'mscale_min': mscale_min,
            'base_mscale_formula': base_mscale_formula,
            'base_mscale_coefficient': base_mscale_coefficient,
            'spread_min': spread_min,
            'spread_max': spread_max,
            'spread_alpha': spread_alpha,
        }
        return (config,)


class SEGAModelPatch:
    """
    Applies SEGA spectral rescaling to a loaded FLUX model.
    This patches the model's transformer forward pass in-place.

    Use this AFTER loading your FLUX model and BEFORE sampling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sega_config": ("SEGA_CONFIG",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_sega"
    CATEGORY = "SEGA"

    def apply_sega(self, model, sega_config):
        # Check if model is FLUX
        model_type = type(model.model).__name__ if hasattr(model, 'model') else "unknown"
        diffusion_model = model.model.diffusion_model if hasattr(model.model, 'diffusion_model') else None

        if diffusion_model is None:
            raise ValueError("[SEGA] Could not find diffusion_model. Make sure you're using a FLUX model.")

        # Verify it's a Flux-like transformer
        if not hasattr(diffusion_model, 'pos_embed'):
            raise ValueError("[SEGA] Model does not have pos_embed. Only FLUX models are supported.")

        # Apply patch
        patcher = SEGATransformerPatcher(model, sega_config)
        patcher.patch()

        # Store patcher reference on model so we can unpatch later if needed
        model._sega_patcher = patcher

        print(f"[SEGA] Applied to model: {model_type}")
        print(f"[SEGA] Training resolution: {sega_config['training_resolution']}")
        print(f"[SEGA] mscale_alpha: {sega_config['mscale_alpha']}, beta: {sega_config['mscale_beta']}")

        return (model,)


class SEGASampler:
    """
    SEGA-aware sampler wrapper.

    NOTE: In most cases you can use a standard KSampler after SEGAModelPatch.
    This node is provided for explicit target resolution control and workflow clarity.

    The SEGA spectral analysis happens automatically in the patched transformer forward pass
    when the latent size exceeds the training resolution. No special sampler is required.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (["euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", 
                                   "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", 
                                   "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
                                   "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm", "ipndm_v",
                                   "deis", "res_multistep", "res_multistep_ancestral", "gflow",
                                   "uniform", "g_ult", "sample_euler", "sample_euler_ancestral",
                                   "sample_heun", "sample_dpm_2", "sample_dpm_2_ancestral"], {"default": "euler"}),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"], {"default": "normal"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "target_width": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 64}),
                "target_height": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 64}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "SEGA"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, sampler_name, scheduler, denoise, target_width, target_height):
        # Store target resolution on the model so the patched forward can access it
        target_resolution = max(target_width, target_height)

        if hasattr(model, '_sega_patcher') and model._sega_patcher is not None:
            model._sega_patcher.config['target_resolution'] = target_resolution
            print(f"[SEGA] Target resolution set to {target_resolution}")

        # Use ComfyUI's built-in sampling function
        # This is the same internal call that KSampler uses
        import comfy.sample

        latent = latent_image
        latent_image_tensor = latent["samples"]

        # Create noise
        noise = torch.randn(
            latent_image_tensor.shape, 
            generator=torch.manual_seed(seed), 
            device=mm.get_torch_device(), 
            dtype=latent_image_tensor.dtype
        )

        # Sample using ComfyUI's internal sampler
        samples = comfy.sample.sample(
            model=model,
            noise=noise,
            positive=positive,
            negative=negative,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            steps=steps,
            denoise=denoise,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            noise_mask=None,
            callback=None,
            disable_pbar=False,
            seed=seed,
        )

        out = latent.copy()
        out["samples"] = samples

        return (out,)
