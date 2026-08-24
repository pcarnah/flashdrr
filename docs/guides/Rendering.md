# Rendering

This guide covers how to project a CT volume into a synthetic X-ray image
(Digitally Reconstructed Radiograph, DRR) with `FlashDRR`, how the camera and
coordinate conventions work, and how to choose between the two renderers.

## Coordinate & camera conventions

* Volumes are **RAS-aligned**, stored as `(B, C, D, H, W)` tensors.
* The volume-to-world relationship is described by an affine `ras2ijk` matrix
  `(4, 4)` in **camera space** (RAS coordinates).
* Cameras are described by a **view matrix** `(4, 4)` in VTK convention
  (camera-to-world). The VTK `+Z` axis points along the projection direction.
* C-arm (X-ray gantry) geometry is expressed via source-to-image distance
  (SID), AP angle, lateral angle and table translation, all in millimetres and
  degrees.

The helper functions in `flashdrr.rendering` build and validate these matrices
for you:

* :func:`get_vtk_view_mat` — build a VTK camera-to-world view matrix from a
  camera position, focal point and view-up vector in RAS coordinates.
* :func:`carm_to_camera_params` — convert C-arm gantry parameters into a
  camera position, look-at point and view-up vector.
* :func:`get_random_carm_views` — sample a batch of random C-arm views
  directly as view matrices.

## The renderers

`FlashDRR` ships two raycasters that share the same camera model:

### VolumeRaycaster

:class:`VolumeRaycaster` is the main, fully differentiable raycaster. It
integrates attenuation along each ray using the Beer–Lambert law by default,
with gradients flowing through both the volume and the view matrix. It
supports:

* batching over both volumes and views (a single view broadcast over a batch,
  or `N` views per volume where `N` is `B * views_per_vol`),
* optional memory checkpointing and tiled rendering for large images,
* an optional learned scatter module (:class:`DepthAwareScatter` plus
  :class:`ASPP` context blocks),
* an optional Poisson noise model.

The rendered image shape follows the forward-pass arguments: passing
`vol (B, C, D, H, W)` and an `N`-row `view_mat (N, 4, 4)` returns an
`(N, C, H, W)` projection.

### FusedVolumeRenderer

:class:`FusedVolumeRenderer` is a Triton-fused CUDA kernel with a matching
autograd wrapper. It requires CUDA and a working Triton installation (the
`triton-windows` fork is used automatically on Windows). For high-resolution or
many-view renders it provides a large speedup over the pure-PyTorch loop.
:class:`VolumeRaycaster` uses it internally when `triton=True`.

## End-to-end example

```python
import torch
import flashdrr.rendering as R
from flashdrr.rendering import carm_to_camera_params, get_vtk_view_mat

# (B, C, D, H, W) attenuation volume; ras2ijk: (4, 4) RAS -> IJK affine
vol = torch.rand(1, 1, 128, 128, 128)
ras2ijk = torch.eye(4)

raycaster = R.VolumeRaycaster(ray_samples=256, resolution=(512, 512))

center = torch.tensor([64.0, 64.0, 64.0])
pos, focal, up = carm_to_camera_params(
    sid=1000.0, ap_angle=0.0, lat_angle=-30.0,
    center_ras=center, table_si=0.0,
)
view_mat = get_vtk_view_mat(pos, focal, up).unsqueeze(0)

# (1, 1, H, W) DRR projection
drr = raycaster(vol, view_mat=view_mat, ras2ijk=ras2ijk)
```

## Transfer functions

CT voxels store Hounsfield units (HU). Before integrating attenuation you
usually map these to density/attenuation values via a transfer function.
:func:`piecewise_linear_channelwise` applies a per-channel piecewise-linear
map that operates on `(B, C, D, H, W)` or `(B, C, H, W)` tensors:

```python
from flashdrr.utils import normalize_hounsfield
from flashdrr.rendering import piecewise_linear_channelwise

# xp keypoints (HU, normalized to [0, 1] via normalize_hounsfield)
xp = torch.tensor([[0.0, 0.2, 0.4, 1.0]])
yp = torch.tensor([[0.0, 0.1, 0.5, 1.0]])

attenuation = piecewise_linear_channelwise(
    normalize_hounsfield(vol), xp, yp
)
drr = raycaster(attenuation, view_mat=view_mat, ras2ijk=ras2ijk)
```

## Rendering many views

Pass `N` rows in `view_mat` to render several projections at once. `N` must
either equal the batch size `B` or be a multiple of it:

```python
views = R.get_random_carm_views(
    n_views=4, sid_range=(900, 1200), ap_range=(-15, 15),
    lat_range=(-45, 45), si_range=(-50, 50), center=center,
)  # (4, 4, 4)

drrs = raycaster(vol, view_mat=views, ras2ijk=ras2ijk)  # (4, 1, H, W)
```

Set `raycaster(..., triton=True)` on CUDA to dispatch to the fast
Triton-fused kernel for this multi-view batch.
