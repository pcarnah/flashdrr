# FlashDRR
Fast differentiable DRR raycasting and volume rendering for PyTorch.

`FlashDRR` provides fast, differentiable raycasters that turn CT volumes into
synthesized X‑ray projections (Digitally Reconstructed Radiographs, DRR) while
keeping gradients flowing through both the volume and the camera parameters.
It is designed for deep-learning workloads such as 2D/3D registration,
pose estimation, and synthetic‑image generation from medical volumes.

## [Documentation](https://torchvtk.github.io)

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

## Installation Instructions
The latest GitHub release is pushed to PyPI:
```
pip install flashdrr
```

To get the latest master:
```
pip install git+https://github.com/torchvtk/torchvtk.git@master#egg=flashdrr
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
* `nibabel` — required only for the medical‑decathlon crawler scripts
  (`flashdrr[data]`).

Please refer to the [documentation](https://torchvtk.github.io) for guides and
the full API reference.
