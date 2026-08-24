# Getting Started

This document should get you up and running with `FlashDRR`!

## What is FlashDRR?
`FlashDRR` is a small, focused library for **differentiable DRR
(Digitally Reconstructed Radiograph) raycasting** and volume rendering in
PyTorch. It turns CT volumes into synthetic X-ray projections while keeping
gradients flowing through both the volume and the camera parameters, which
makes it a natural fit for deep-learning pipelines in 2D/3D registration,
pose estimation, and synthetic-image generation.

The pre-0.4.x "data loading framework" functionality (TorchDataset /
TorchQueueDataset, HDF5 & DICOM converters, transforms) has been trimmed.
The current focus is the renderer and its camera geometry.

## Installation
### Latest PyPI release
```
pip install flashdrr
```
### Latest master
```
pip install git+https://github.com/pcarnah/flashdrr.git@master#egg=flashdrr
```
### Developer installation
This will install your local changes to `flashdrr` directly.
```
git clone https://github.com/pcarnah/flashdrr.git
cd flashdrr
pip install -e .
```

## Features
### Differentiable DRR raycasting
The core of the library is `flashdrr.rendering.VolumeRaycaster`. It
integrates attenuation along each ray (Beer–Lambert law by default) and is
fully differentiable with respect to both the volume and the view matrix.

Check out [Rendering](Rendering) for a detailed guide and end-to-end
example.

### Fast Triton-fused rendering
`flashdrr.rendering.FusedVolumeRenderer` is a Triton-fused CUDA kernel that
the `VolumeRaycaster` uses when `triton=True`. For high-resolution or many
view renders it provides an order-of-magnitude speedup over the pure-PyTorch
loop.

### C-arm / camera geometry
Realistic X-ray gantry poses are easy to sample with
`carm_to_camera_params` and `get_random_carm_views`, and VTK-compatible
camera matrices can be built with `get_vtk_view_mat`.

### Volume utilities
`flashdrr.utils` provides small helpers for working with volumetric tensors,
such as `make_nd`, `normalize_hounsfield` and `normalize_voxel_scale`.

## A minimal end-to-end example
The following renders a DRR from a random attenuation volume:

```python
import torch
import flashdrr.rendering as R
from flashdrr.rendering import carm_to_camera_params, get_vtk_view_mat

# (B, C, D, H, W) attenuation volume; ras2ijk: (4, 4) RAS -> IJK affine
vol   = torch.rand(1, 1, 128, 128, 128)
ras2ijk = torch.eye(4)

raycaster = R.VolumeRaycaster(ray_samples=256, resolution=(512, 512))

center   = torch.tensor([64., 64., 64.])
pos, focal, up = carm_to_camera_params(
    sid=1000.0, ap_angle=0.0, lat_angle=-30.0,
    center_ras=center, table_si=0.0)
view_mat = get_vtk_view_mat(pos, focal, up).unsqueeze(0)

drr = raycaster(vol, view_mat=view_mat, ras2ijk=ras2ijk)
# drr: (1, 1, 512, 512) projection
```

## Contributing
The project is small and focused. We are open for all suggestions right on our
[GitHub](https://github.com/pcarnah/flashdrr). Just throw us an issue ;)
