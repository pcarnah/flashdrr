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
The snippet below mirrors the end-to-end `__main__` block in
`flashdrr/rendering/raycast.py`: it loads a CT volume with MONAI, derives the
RAS↔IJK affine, builds a C-arm view matrix with `get_vtk_view_mat`, and
renders a DRR with `VolumeRaycaster`. Two attenuation pipelines are shown
from the same loaded volume — a hand-made 5-keypoint transfer function
mapping HU → attenuation coefficients via `piecewise_linear_channelwise`,
and a simple linear HU → μ map.

```python
import torch
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Spacing, ScaleIntensityRange, EnsureType

import flashdrr.rendering as R

# ---------- 1. Load CT volume and build the RAS<->IJK affine ----------
load_tf = Compose([
    LoadImage(),
    EnsureChannelFirst(),
    Spacing([2.5, 2.5, 3.0]),
    ScaleIntensityRange(-3024, 3024, 0, 1, clip=True),
    EnsureType(),
])
vol = load_tf('CTChest.nii.gz').unsqueeze(0)                    # (1, 1, D, H, W), MetaTensor in HU
ijk2ras = vol.meta['affine']                                    # (4, 4) IJK -> RAS
ras2ijk = torch.linalg.inv(ijk2ras)
vol = vol.cuda()

# ---------- 2. Build a C-arm view matrix in RAS ----------
center_ijk = torch.ones(4, dtype=torch.float64)
center_ijk[:3] = torch.as_tensor(vol.shape[2:]) // 2
center_ras = (ijk2ras @ center_ijk)[:3].float().cuda()
view_mat = R.get_vtk_view_mat(
    cam_pos=(0.0, 1000.0, -130.0),                              # source 1 m in front of the volume
    cam_focal=tuple(center_ras.tolist()),
    cam_viewup=(0.0, 0.0, 1.0),
    device='cuda',
).unsqueeze(0)                                                 # (1, 4, 4)

raycaster = R.VolumeRaycaster(ray_samples=384, resolution=(1024, 1024)).cuda().eval()

# ---------- 3a. Hand-made HU -> attenuation transfer function ----------
# Each keypoint is (HU, attenuation); below -200 HU is air (mu=0),
# soft tissue ramps gently, bone saturates to high attenuation.
tf = torch.tensor([
    [-3500, 0.00],   # air
    [ -200, 0.00],   # lung / soft-tissue boundary
    [  200, 0.05],   # soft tissue
    [ 1535, 0.50],   # cortical bone
    [ 3071, 0.65],   # dense bone / metal
]).cuda()
tf[:, 0] = (tf[:, 0] + 3500) / 7000                            # normalize HU to [0, 1]
xp = tf[:, 0].unsqueeze(0)                                     # (1, K)
yp = tf[:, -1].unsqueeze(0)                                    # (1, K)
mu_tf = R.piecewise_linear_channelwise(vol, xp, yp)            # (1, 1, D, H, W)

drr_tf = raycaster(mu_tf, view_mat=view_mat, ras2ijk=ras2ijk.float().cuda())

# ---------- 3b. Linear HU -> mu map ----------
hu = vol * (3024 - (-3524)) + (-3524)
mu_lin = torch.clamp(0.05 * (1.0 + hu / 800.0), min=0.0)       # (1, 1, D, H, W)

drr_lin = raycaster(mu_lin, view_mat=view_mat, ras2ijk=ras2ijk.float().cuda())
```

Pass `triton=True` to `raycaster(...)` to switch to the fused
CUDA/Triton kernel.

## Contributing
The project is small and focused. We are open for all suggestions right on our
[GitHub](https://github.com/pcarnah/flashdrr). Just throw us an issue ;)
