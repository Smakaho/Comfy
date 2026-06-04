# ComfyUI-SEGA

**Spectral-Energy Guided Attention for Resolution Extrapolation in ComfyUI**

Based on [SEGA by Rajabi et al.](https://github.com/rajabi2001/sega) — a training-free method for generating ultra-high-resolution images with FLUX diffusion transformers.

## What is SEGA?

SEGA dynamically rescales RoPE (Rotary Position Embedding) attention components based on the latent's spatial-frequency content at each denoising step. This allows FLUX models to generate at resolutions far beyond their training resolution (e.g., 4096×4096, 6144×6144) without retraining, new weights, or architecture changes.

## Installation

### Method 1: ComfyUI Manager (if available)
Search for "ComfyUI-SEGA" in ComfyUI Manager and install.

### Method 2: Manual Install
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/Smakaho/Comfy.git
```

### Method 3: Graydient.AI Workflow Builder
1. In your Graydient workflow, add this repo to the **Custom Nodes** field:
   ```
   https://github.com/Smakaho/comfy
   ```
2. Check **Auto-install nodes**
3. No PIP requirements needed — uses only PyTorch and standard libraries

## Nodes

### 1. SEGA Config
Configure SEGA hyperparameters:
- **Training Resolution**: The resolution the model was trained on (default: 1024)
- **mscale_alpha**: Sharpness bias strength (default: 0.15)
- **mscale_beta**: Energy standardization steepness (default: 1.5)
- **mscale_min**: Floor for mscale values (default: 1.0)
- **base_mscale_formula**: "power_res" or "log_res" (default: power_res)
- **base_mscale_coefficient**: Exponent coefficient (default: 0.08)
- **spread_min/spread_max**: Dynamic spread range (default: 0.0 / 1.0)
- **spread_alpha**: Spread concentration (default: 1.5)

### 2. SEGA Model Patch
Apply SEGA to a loaded FLUX model. Place this node **after** your Load Diffusion Model node and **before** your KSampler.

**Inputs:**
- `model`: Loaded FLUX model
- `sega_config`: From SEGA Config node

**Output:**
- `model`: SEGA-patched model (use with any standard sampler)

### 3. SEGA Sampler (Optional)
A wrapper sampler that explicitly sets target resolution. In most cases, you can use a standard **KSampler** after SEGA Model Patch — the spectral analysis happens automatically when the latent size exceeds training resolution.

## Workflow Example

```
[Load Checkpoint] → [SEGA Config] → [SEGA Model Patch] → [KSampler] → [VAEDecode] → [Save Image]
                      ↑                ↑
                   set params      connect model
```

## How It Works

1. **Load FLUX model** normally through ComfyUI
2. **Apply SEGA patch** — this swaps the transformer's position embedding for a SEGA-aware version and wraps the forward pass
3. **Generate at high resolution** — when the latent resolution exceeds the training resolution, SEGA automatically:
   - Computes 2D FFT spectral energy profiles from the latent
   - Calculates per-dimension RoPE mscale values based on energy distribution
   - Applies dynamic NTK factor scaling
   - Rescales attention to preserve sharpness at high frequencies
4. **Standard sampling** — use any sampler (Euler, DPM++, etc.) — SEGA operates transparently

## Recommended Settings

| Target Resolution | Training Resolution | mscale_alpha | base_mscale_coefficient |
|-------------------|---------------------|--------------|-------------------------|
| 1024×1024         | 1024                | 0.15         | 0.08                    |
| 2048×2048         | 1024                | 0.15         | 0.08                    |
| 4096×4096         | 1024                | 0.15         | 0.08                    |
| 6144×6144         | 1024                | 0.12         | 0.06                    |

## Multi-GPU

For ultra-high resolutions (6144×6144+), the original SEGA implementation supports multi-GPU block distribution. This custom node does **not** yet implement multi-GPU — if you need it, run the [official standalone scripts](https://github.com/rajabi2001/sega) instead.

## Citation

```bibtex
@article{rajabi2026sega,
  title={SEGA: Spectral-Energy Guided Attention for Resolution Extrapolation in Diffusion Transformers},
  author={Rajabi, Javad and Shaban, Kimia and Roohi, Koorosh and Lindell, David B and Taati, Babak},
  journal={arXiv preprint arXiv:2605.22668},
  year={2026}
}
```
