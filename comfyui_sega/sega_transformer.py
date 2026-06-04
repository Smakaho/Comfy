"""
SEGA-Modified FluxPosEmbed
Wraps the standard ComfyUI/Flux position embedding with SEGA spectral rescaling.
This is a monkey-patch approach — we wrap the existing pos_embed forward.
"""

import math
import torch
import torch.nn as nn
from .sega_spectral import (
    get_1d_rotary_pos_embed,
    compute_spectral_energy_profile,
    compute_axis_spectral_profiles,
    compute_dynamic_spread,
)

class SEGAFluxPosEmbed(nn.Module):
    """
    Drop-in replacement for FluxPosEmbed that adds SEGA spectral rescaling.
    """
    def __init__(self, original_pos_embed, config):
        super().__init__()
        self.original = original_pos_embed
        self.config = config

        # Copy attributes from original
        self.theta = getattr(original_pos_embed, 'theta', 10000)
        self.axes_dim = getattr(original_pos_embed, 'axes_dim', [16, 56, 56])
        self.training_resolution = getattr(original_pos_embed, 'training_resolution', 64)

        # SEGA params from config
        self.mscale_alpha = config.get('mscale_alpha', 0.15)
        self.mscale_beta = config.get('mscale_beta', 1.5)
        self.mscale_min = config.get('mscale_min', 1.0)
        self.base_mscale_formula = config.get('base_mscale_formula', 'power_res')
        self.base_mscale_coefficient = config.get('base_mscale_coefficient', 0.08)
        self.spread_min = config.get('spread_min', 0.0)
        self.spread_max = config.get('spread_max', 1.0)
        self.spread_alpha = config.get('spread_alpha', 1.5)
        self.training_res_pixels = config.get('training_resolution', 1024)

    def forward(self, ids, ntk_factor=1.0, mscale_spread=None,
                energy_profile_h=None, energy_profile_w=None, target_res=None):

        n_axes = ids.shape[-1]
        cos_out, sin_out = [], []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64

        axis_energy = {1: energy_profile_h, 2: energy_profile_w}

        for i in range(n_axes):
            if i > 0 and ntk_factor > 1.0:
                cos, sin = get_1d_rotary_pos_embed(
                    self.axes_dim[i], pos[:, i], theta=self.theta,
                    repeat_interleave_real=True, use_real=True, freqs_dtype=freqs_dtype,
                    ntk_factor=ntk_factor,
                    energy_profile=axis_energy.get(i),
                    mscale_spread=mscale_spread,
                    mscale_alpha=self.mscale_alpha, mscale_beta=self.mscale_beta,
                    mscale_min=self.mscale_min,
                    base_mscale_formula=self.base_mscale_formula,
                    base_mscale_coefficient=self.base_mscale_coefficient,
                    target_res=target_res, training_res=self.training_res_pixels,
                )
            else:
                cos, sin = get_1d_rotary_pos_embed(
                    self.axes_dim[i], pos[:, i], theta=self.theta,
                    repeat_interleave_real=True, use_real=True, freqs_dtype=freqs_dtype,
                )
            cos_out.append(cos)
            sin_out.append(sin)

        return torch.cat(cos_out, dim=-1).to(ids.device), torch.cat(sin_out, dim=-1).to(ids.device)


class SEGATransformerPatcher:
    """
    Patches a ComfyUI Flux model's transformer with SEGA logic.
    Uses monkey-patching to avoid replacing the entire model.
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.original_forward = None
        self.original_pos_embed = None
        self.sega_pos_embed = None
        self._patched = False

    def patch(self):
        """Apply SEGA patches to the model."""
        if self._patched:
            return

        diffusion_model = self.model.model.diffusion_model
        transformer = diffusion_model

        # Store original pos_embed
        if hasattr(transformer, 'pos_embed'):
            self.original_pos_embed = transformer.pos_embed
            self.sega_pos_embed = SEGAFluxPosEmbed(transformer.pos_embed, self.config)
            transformer.pos_embed = self.sega_pos_embed

        # Store original forward
        self.original_forward = transformer.forward

        # Build the SEGA forward wrapper
        config = self.config
        training_resolution = config.get('training_resolution', 1024)
        spread_min = config.get('spread_min', 0.0)
        spread_max = config.get('spread_max', 1.0)
        spread_alpha = config.get('spread_alpha', 1.5)
        _rope_spatial_dim = getattr(transformer, '_rope_spatial_dim', 56)

        def sega_forward(
            hidden_states,
            encoder_hidden_states=None,
            pooled_projections=None,
            timestep=None,
            img_ids=None,
            txt_ids=None,
            guidance=None,
            joint_attention_kwargs=None,
            controlnet_block_samples=None,
            controlnet_single_block_samples=None,
            return_dict=True,
            controlnet_blocks_repeat=False,
            target_resolution=None,
            **kwargs
        ):
            # Call original forward up to the point where pos_embed is used
            # We need to inject the spectral analysis before pos_embed call

            # Standard ComfyUI/Flux forward prep
            if joint_attention_kwargs is not None:
                joint_attention_kwargs = joint_attention_kwargs.copy()
            else:
                joint_attention_kwargs = {}

            hidden_states = transformer.x_embedder(hidden_states)

            timestep = timestep.to(hidden_states.dtype) * 1000
            if guidance is not None:
                guidance = guidance.to(hidden_states.dtype) * 1000

            temb = (
                transformer.time_text_embed(timestep, pooled_projections) if guidance is None
                else transformer.time_text_embed(timestep, guidance, pooled_projections)
            )
            encoder_hidden_states = transformer.context_embedder(encoder_hidden_states)

            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]
            if img_ids.ndim == 3:
                img_ids = img_ids[0]

            # --- SEGA SPECTRAL ANALYSIS ---
            max_h = int(img_ids[:, 1].max().item() - img_ids[:, 1].min().item()) + 1
            max_w = int(img_ids[:, 2].max().item() - img_ids[:, 2].min().item()) + 1

            if target_resolution is None:
                target_resolution = max(max_h, max_w) * 16

            s = target_resolution / training_resolution
            d = _rope_spatial_dim
            current_ntk_factor = s ** (2 * d / (d - 2)) / (1 + 0.1 * math.log(s))

            dynamic_spread = None
            energy_profile_h = None
            energy_profile_w = None

            if current_ntk_factor > 1.0:
                n_spatial = hidden_states.shape[1]
                if max_h * max_w == n_spatial:
                    img_h, img_w = max_h, max_w
                else:
                    aspect = max_h / max_w
                    img_w = int(math.sqrt(n_spatial / aspect))
                    img_h = n_spatial // max(1, img_w)

                n_bins_h = max(img_h // 2, 8)
                n_bins_w = max(img_w // 2, 8)
                energy_profile_h, energy_profile_w = compute_axis_spectral_profiles(
                    hidden_states, img_h, img_w, n_bins_h, n_bins_w,
                )
                iso_profile = compute_spectral_energy_profile(
                    hidden_states, img_h, img_w, n_bins=max(img_h, img_w) // 2,
                )
                dynamic_spread = compute_dynamic_spread(
                    iso_profile, spread_min=spread_min,
                    spread_max=spread_max, alpha=spread_alpha,
                )

            ids = torch.cat((txt_ids, img_ids), dim=0)

            # Use SEGA pos_embed
            image_rotary_emb = transformer.pos_embed(
                ids, ntk_factor=current_ntk_factor, mscale_spread=dynamic_spread,
                energy_profile_h=energy_profile_h, energy_profile_w=energy_profile_w,
                target_res=target_resolution,
            )

            # --- Continue with standard transformer blocks ---
            for index_block, block in enumerate(transformer.transformer_blocks):
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                    temb=temb, image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs,
                )
                if controlnet_block_samples is not None:
                    interval_control = int(math.ceil(len(transformer.transformer_blocks) / len(controlnet_block_samples)))
                    if controlnet_blocks_repeat:
                        hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

            for index_block, block in enumerate(transformer.single_transformer_blocks):
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                    temb=temb, image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs,
                )
                if controlnet_single_block_samples is not None:
                    interval_control = int(math.ceil(len(transformer.single_transformer_blocks) / len(controlnet_single_block_samples)))
                    hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

            hidden_states = transformer.norm_out(hidden_states, temb)
            output = transformer.proj_out(hidden_states)

            if not return_dict:
                return (output,)
            return type('Transformer2DModelOutput', (), {'sample': output})()

        transformer.forward = sega_forward
        self._patched = True
        print("[SEGA] Transformer patched successfully")

    def unpatch(self):
        """Remove SEGA patches."""
        if not self._patched:
            return

        diffusion_model = self.model.model.diffusion_model
        if self.original_pos_embed is not None:
            diffusion_model.pos_embed = self.original_pos_embed
        if self.original_forward is not None:
            diffusion_model.forward = self.original_forward

        self._patched = False
        print("[SEGA] Transformer unpatched")
