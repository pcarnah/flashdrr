"""Differentiable volume rendering for PyTorch.

The package exposes two renderers and a handful of camera / geometry helpers:

* :class:`VolumeRaycaster` - the main Python-loop differentiable raycaster
  (optionally with the :class:`DepthAwareScatter` and :class:`ASPP` building
  blocks for scatter-based DRR rendering).
* :class:`FusedVolumeRenderer` - a Triton-fused forward kernel for fast
  backprop-free rendering on CUDA.

The camera helpers (:func:`get_view_mat`, :func:`get_vtk_view_mat`,
:func:`get_proj_mat`, :func:`get_random_carm_views`, ...) are the same ones
the renderers use internally and are re-exported for downstream code.
"""

from .raycast import (
    ASPP,
    DepthAwareScatter,
    VolumeRaycaster,
    carm_to_camera_params,
    get_random_carm_views,
    get_vtk_view_mat,
    piecewise_linear_channelwise,
)
from .triton_raycast import FusedVolumeRenderer

__all__ = [
    'ASPP',
    'DepthAwareScatter',
    'FusedVolumeRenderer',
    'VolumeRaycaster',
    'carm_to_camera_params',
    'get_random_carm_views',
    'get_vtk_view_mat',
    'piecewise_linear_channelwise',
]
