"""
SEGA Spectral Analysis Helpers
Ported from https://github.com/rajabi2001/sega/flux_sega/transformer_flux.py
"""

import math
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis,
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 2,
):
    if use_real:
        cos, sin = freqs_cis
        if sequence_dim == 2:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        elif sequence_dim == 1:
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        else:
            raise ValueError(f"sequence_dim={sequence_dim} but should be 1 or 2.")
        cos, sin = cos.to(x.device), sin.to(x.device)
        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"use_real_unbind_dim={use_real_unbind_dim} but should be -1 or -2.")
        return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)

# ---------------------------------------------------------------------------
# base_mscale
# ---------------------------------------------------------------------------

def compute_base_mscale(
    target_res: int,
    training_res: int,
    formula: str = "power_res",
    coefficient = None,
):
    s = max(float(target_res) / float(training_res), 1.0)
    if formula == "power_res":
        c = 0.1 if coefficient is None else float(coefficient)
        return s ** c
    if formula == "log_res":
        c = 0.1 if coefficient is None else float(coefficient)
        return 1.0 + c * math.log(s)
    raise ValueError(f"Unknown base_mscale formula: {formula!r}. Use 'power_res' or 'log_res'.")

# ---------------------------------------------------------------------------
# SEGA mscale allocation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_sega_allocation(
    energy_profile: torch.Tensor,
    freqs: torch.Tensor,
    base_mscale: float,
    spread: float,
    alpha: float = 0.15,
    beta: float = 1.5,
    min_mscale: float = 1.0,
):
    """
    Spectral-Energy Guided Attention: per-dim RoPE mscale (asymmetric band).
    High spectral-energy dims get lower m_k (sharpness-biased).
    """
    D_half = freqs.shape[0]
    eps = 1e-8

    if spread <= 0.0 or alpha <= 0.0:
        return torch.full((D_half,), float(base_mscale), device=freqs.device, dtype=torch.float32)

    # Map each RoPE dim to its FFT bin via log-period
    periods = 2.0 * math.pi / freqs.clamp(min=eps)
    log_periods = torch.log(periods)
    min_lp, max_lp = log_periods.min(), log_periods.max()
    if (max_lp - min_lp).item() > 1e-6:
        lp_norm = (log_periods - min_lp) / (max_lp - min_lp)
    else:
        lp_norm = torch.zeros_like(log_periods)

    n_bins = energy_profile.shape[0]
    bin_pos = (1.0 - lp_norm) * (n_bins - 1)
    j_low = bin_pos.floor().long().clamp(0, n_bins - 1)
    j_high = (j_low + 1).clamp(0, n_bins - 1)
    frac = (bin_pos - j_low.to(bin_pos.dtype)).clamp(0.0, 1.0)

    E = energy_profile.to(freqs.device).clamp(min=eps)
    log_E = torch.log(E)
    raw = log_E[j_low] * (1.0 - frac) + log_E[j_high] * frac

    # Standardise + tanh + re-centre
    z = raw - raw.mean()
    z = z / z.std().clamp(min=eps)
    s = torch.tanh(float(beta) * z)
    s = s - s.mean()

    # direction = -1 => subtract
    m = float(base_mscale) * (1.0 - float(alpha) * float(spread) * s)
    return m.clamp(min=float(min_mscale)).to(torch.float32)

# ---------------------------------------------------------------------------
# 1-D RoPE embedding with SEGA per-dim mscale
# ---------------------------------------------------------------------------

def get_1d_rotary_pos_embed(
    dim: int,
    pos,
    theta: float = 10000.0,
    use_real: bool = True,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
    freqs_dtype=torch.float32,
    energy_profile = None,
    mscale_spread = None,
    mscale_alpha: float = 0.15,
    mscale_beta: float = 1.5,
    mscale_min: float = 1.0,
    base_mscale_formula: str = "power_res",
    base_mscale_coefficient = None,
    target_res = None,
    training_res = None,
):
    import numpy as np
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)

    dim_indices = torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)
    inv_freq = 1.0 / (theta ** (dim_indices / dim))

    effective_mscale = 1.0

    if ntk_factor > 1.0:
        scaled_theta = theta * ntk_factor
        freqs = 1.0 / (scaled_theta ** (dim_indices / dim))

        base_ms = compute_base_mscale(
            target_res=target_res,
            training_res=training_res,
            formula=base_mscale_formula,
            coefficient=base_mscale_coefficient,
        )

        use_sega = (
            energy_profile is not None
            and mscale_spread is not None
            and float(mscale_spread) > 0.0
            and base_ms > 1.0 + 1e-8
        )

        if use_sega:
            effective_mscale = compute_sega_allocation(
                energy_profile=energy_profile,
                freqs=freqs,
                base_mscale=base_ms,
                spread=float(mscale_spread),
                alpha=float(mscale_alpha),
                beta=float(mscale_beta),
                min_mscale=float(mscale_min),
            )
        else:
            effective_mscale = base_ms
    else:
        freqs = inv_freq

    freqs = torch.outer(pos.float(), freqs)

    if freqs.device.type == "npu":
        freqs = freqs.float()

    if isinstance(effective_mscale, torch.Tensor):
        if use_real and repeat_interleave_real:
            ms = effective_mscale.repeat_interleave(2)
            freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float() * ms
            freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float() * ms
        elif use_real:
            ms = torch.cat([effective_mscale, effective_mscale], dim=-1)
            freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float() * ms
            freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float() * ms
        else:
            freqs_cis = torch.polar(effective_mscale.unsqueeze(0).expand_as(freqs), freqs)
            return freqs_cis
    else:
        if use_real and repeat_interleave_real:
            freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float() * effective_mscale
            freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float() * effective_mscale
        elif use_real:
            freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float() * effective_mscale
            freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float() * effective_mscale
        else:
            freqs_cis = torch.polar(torch.ones_like(freqs) * effective_mscale, freqs)
            return freqs_cis

    return freqs_cos, freqs_sin

# ---------------------------------------------------------------------------
# Spectral energy profiles
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_spectral_energy_profile(hidden_states, height, width, n_bins):
    B, S, C = hidden_states.shape
    n_spatial = min(S, height * width)
    spatial = hidden_states[:, :n_spatial].reshape(B, height, width, C)
    spatial_map = spatial.float().mean(dim=(0, -1))
    spatial_map = spatial_map - spatial_map.mean()

    power = torch.fft.fftshift(torch.fft.fft2(spatial_map)).abs().pow(2)

    cy, cx = height / 2.0, width / 2.0
    y = torch.arange(height, device=power.device, dtype=torch.float32) - cy
    x = torch.arange(width, device=power.device, dtype=torch.float32) - cx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius_norm = torch.sqrt(yy ** 2 + xx ** 2)
    radius_norm = radius_norm / (radius_norm.max() + 1e-8)

    bin_idx = (radius_norm * n_bins).long().clamp(0, n_bins - 1).flatten()
    flat_pw = power.flatten()

    energy_sum = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
    energy_cnt = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
    energy_sum.scatter_add_(0, bin_idx, flat_pw)
    energy_cnt.scatter_add_(0, bin_idx, torch.ones_like(flat_pw))
    return energy_sum / (energy_cnt + 1e-8)

@torch.no_grad()
def compute_axis_spectral_profiles(hidden_states, height, width, n_bins_h, n_bins_w):
    B, S, C = hidden_states.shape
    n_spatial = min(S, height * width)
    spatial = hidden_states[:, :n_spatial].reshape(B, height, width, C)
    sm = spatial.float().mean(dim=(0, -1))
    sm = sm - sm.mean()

    def _axis_profile(sm, axis, n_bins, length):
        fft = torch.fft.fft(sm, dim=axis)
        power = fft.abs().pow(2).mean(dim=1 - axis)
        half = length // 2 + 1
        power = power[:half]
        freq_norm = torch.linspace(0.0, 1.0, half, device=power.device)
        bin_idx = (freq_norm * n_bins).long().clamp(0, n_bins - 1)
        energy_sum = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
        energy_cnt = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
        energy_sum.scatter_add_(0, bin_idx, power.float())
        energy_cnt.scatter_add_(0, bin_idx, torch.ones_like(power, dtype=torch.float32))
        return energy_sum / (energy_cnt + 1e-8)

    return (
        _axis_profile(sm, axis=0, n_bins=n_bins_h, length=height),
        _axis_profile(sm, axis=1, n_bins=n_bins_w, length=width),
    )

@torch.no_grad()
def compute_dynamic_spread(energy_profile, spread_min=0.0, spread_max=1.0, alpha=1.5):
    eps = 1e-8
    energy = energy_profile.clamp(min=eps)
    geo_mean = torch.exp(torch.log(energy).mean())
    arith_mean = energy.mean()
    flatness = (geo_mean / (arith_mean + eps)).clamp(0.0, 1.0)
    concentration = 1.0 - flatness.item()
    return spread_min + (spread_max - spread_min) * (1.0 - (1.0 - concentration) ** alpha)
