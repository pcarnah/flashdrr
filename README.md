# FlashDRR
Fast differentiable DRR raycasting and volume rendering for PyTorch.

`FlashDRR` provides fast, differentiable raycasters that turn CT volumes into
synthesized X‑ray projections (Digitally Reconstructed Radiographs, DRR) while
keeping gradients flowing through both the volume and the camera parameters.
It is designed for deep-learning workloads such as 2D/3D registration,
pose estimation, and synthetic‑image generation from medical volumes.

## Features
* **Differentiable DRR raycasting** — `flashdrr.rendering.VolumeRaycaster`
  renders physically‑based DRRs using the Beer‑Lambert law with fully
  differentiable ray sampling.
* **Fast Triton‑fused renderer** — `flashdrr.rendering.FusedVolumeRenderer`
  runs a CUDA kernel for near‑instant, backprop‑free (or autograd-aware)
  projection at high resolution.
* **C‑arm / camera helpers** — sample realistic C‑arm views
  (`carm_to_camera_params`, `get_random_carm_views`) and build VTK‑compatible
  camera matrices in RAS space (`get_vtk_view_mat`).
* **Transfer functions** — map CT Hounsfield units to density/attenuation via
  `flashdrr.rendering.piecewise_linear_channelwise`.
* **Lightweight volume utilities** — `flashdrr.utils` provides dimension
  helpers (`make_nd`), HU normalization, and voxel‑scale handling.

## Benchmark

The following trends summarize head-to-head measurements of
`VolumeRaycaster` (PyTorch `grid_sample` based) against the Triton-fused
`FusedVolumeRenderer` across resolutions, ray-sample counts, and volume
sizes. All tests performed using an RTX 5070Ti 16GB.

<a href="benchmarks/plots/benchmark_analysis.svg">
  <img src="benchmarks/plots/benchmark_analysis.svg"
       alt="Benchmark comparison of the PyTorch-loop VolumeRaycaster against the Triton-fused FusedVolumeRenderer. Left: peak GPU memory (MiB, log scale) versus ray-sample count across resolutions from 128x128 to 2048x2048, showing linear growth for the PyTorch backend (peaking at ~24.8 GB at 2048x2048, 256 samples) and a near-flat profile below ~430 MiB for Triton. Middle: per-iteration latency (ms, log scale) versus resolution, showing super-linear degradation for the PyTorch backend beyond 1024x1024 (3,690 ms at 2048x2048) and roughly linear scaling for Triton (11.8 ms at 2048x2048, a ~314x speedup). Right: train vs eval memory and latency at 1024x1024 with 256 samples, where Triton uses 92.9 MiB eval / 115.2 MiB train versus 6,254.9 / 6,247.2 MiB for PyTorch, and 4.21 ms eval / 23.50 ms train versus 55.9 / 75.37 ms for PyTorch."
       width="900">
</a>

### Key Performance Trends

1. **Memory Scaling: Linear Allocation vs. Near-Constant Footprint**
   * **Without Triton:** Peak VRAM scales directly with total samples
     ($\text{Rays} \times \text{Samples/Ray}$). At $2048 \times 2048$
     resolution with 256 ray samples, peak memory hits 24.8 GB in
     evaluation mode, leading to out-of-memory (OOM) failures at
     $\geq 384$ samples.
   * **With Triton:** Fused rendering kernels avoid materializing
     intermediate ray-sample tensors in HBM. Memory footprint remains
     below 430 MiB across all tested configurations — a 99.5% memory
     reduction at high workloads.
   * **Volume Size Impact:** Scaling volume grid resolution from
     $128^3$ to $256^3$ adds a fixed baseline memory overhead
     (~100–300 MiB) for tensor storage, but Triton's intermediate
     sample memory remains invariant to volume size.

2. **Time Scaling: Sub-Linear Latency vs. Memory-Wall Thrashing**
   * **Extreme Acceleration:** At $2048 \times 2048$ resolution
     (256 samples, eval), Triton reduces iteration time from
     1,804.32 ms to 9.16 ms — a $197\times$ speedup.
   * **Non-Linear Degradation without Triton:** Beyond
     $1024 \times 1024$, standard PyTorch execution time degrades
     super-linearly due to memory bandwidth limits and cache
     thrashing. Triton maintains predictable linear scaling
     proportional to total rays rendered.

3. **Train vs. Eval Dynamics**
   * **Memory Overhead:** Without Triton, evaluation and training both
      consume massive memory ($\approx 12.4$ GB at $1024\times 1024$ with
      512 samples). With Triton, training memory increases only slightly
      over evaluation (121.6 MiB train vs 110.7 MiB eval) due to lightweight
      gradient buffer retention.
     * **Backward Pass Latency:** Training mode adds a larger relative
       compute penalty in Triton than in standard PyTorch (~$3.05\times$ overhead vs ~$1.31\times$
       at $1024\times 1024$, 512 samples). This stems
       from executing backward gradient kernels for custom operators,
       though Triton train step time (15.4 ms) remains dramatically
       faster than non-Triton eval time (82.0 ms).

### Configuration snapshot (1024×1024, 512 samples)

| Configuration (1024×1024, 512 samples) | Triton OFF | Triton ON | Improvement         |
|----------------------------------------|-----------:| ---: |---------------------|
| Eval Memory (MiB)                      |   12,386.7 | 110.7  | $111.9\times$ less  |
| Train Memory (MiB)                     |   12,409.6 | 115.2 | $102.1\times$ less  |
| Eval Latency (ms/iter)                 |      82.00 | 5.05  | $16.2\times$ faster |
| Train Latency (ms/iter)                |     107.34 | 15.40 | $7.0\times$ faster  |

## Installation Instructions
The project is available via PyPI.

```

pip install flashdrr

```

To get the latest master:
```
pip install git+https://github.com/pcarnah/flashdrr.git@master#egg=flashdrr
```

### CUDA builds
PyTorch ships separate wheels per CUDA version. By default `pip install
flashdrr` pulls the CPU build of `torch`. To select a CUDA build, install the
corresponding extra and point pip at the matching PyTorch index, for example:

```bash
pip install flashdrr[cu128] --extra-index-url https://download.pytorch.org/whl/cu128
pip install flashdrr[cu130] --extra-index-url https://download.pytorch.org/whl/cu130
```

The Triton-fused renderer requires a working Triton installation (the
`triton-windows` fork is used automatically on Windows).

### Using uv
[uv](https://docs.astral.sh/uv/) is supported. CUDA build
selection uses the same `cu128` / `cu130` / `cpu` extras as with pip — pick
one and the matching PyTorch index is
selected automatically via `[tool.uv.sources]`:

```bash
uv sync --extra cu128
uv sync --extra cu130
uv sync --extra cpu
```

Dev-only tooling (e.g. `pytest`) lives in PEP 735 dependency groups, which
must not share names with the CUDA extras — add them with `--group`:

```bash
uv sync --extra cu130 --group test
```

The optional `data` extra (NIfTI file support) is just another extra
and combines with a CUDA extra:

```bash
uv sync --extra cu128 --extra data
```

## Quick example
```python
import torch
import flashdrr.rendering as R
from flashdrr.rendering import carm_to_camera_params, get_vtk_view_mat

# vol: (B, C, D, H, W) attenuation volume in RAS-aligned voxels (e.g. from MONAI)
# ras2ijk: (4, 4) RAS -> IJK affine of the volume
vol = torch.rand(1, 1, 128, 128, 128)

raycaster = R.VolumeRaycaster(ray_samples=256, resolution=(512, 512)).cuda()

center = torch.tensor([64., 64., 64.], device='cuda')
pos, focal, up = carm_to_camera_params(sid=1000.0, ap_angle=0.0, lat_angle=-30.0,
                                       center_ras=center, table_si=0.0)
view_mat = get_vtk_view_mat(pos, focal, up, device='cuda').unsqueeze(0)

# (1, 1, H, W) DRR projection
drr = raycaster(vol.cuda(), view_mat=view_mat, ras2ijk=ras2ijk.cuda())
```

## Optional dependencies
* `nibabel` — required only for loading Nifti format data for tests and example scripts
  (`flashdrr[data]`).

## Attribution
If you use this software in your research, please cite it using the "Cite this repository" button or the provided CITATION.cff.
