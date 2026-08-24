# flashdrr.rendering

Differentiable DRR raycasting and volume rendering.

## Raycasters

The package ships two raycaster back-ends that share the same camera model and
VTK/RAS conventions:

* :class:`VolumeRaycaster` — the main, fully differentiable raycaster. It
  integrates attenuation along each ray (Beer–Lambert by default) and supports
  batching over both volumes and views, optional memory checkpointing and
  tiled rendering, an optional learned scatter module, and an optional
  Poisson noise model.
* :class:`FusedVolumeRenderer` — a Triton-fused CUDA kernel with a matching
  autograd wrapper. It is used internally by :class:`VolumeRaycaster` when
  `triton=True` and can also be used standalone for very fast projection.

## VolumeRaycaster

```{eval-rst}
.. autoclass:: flashdrr.rendering.VolumeRaycaster
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

## FusedVolumeRenderer

```{eval-rst}
.. autoclass:: flashdrr.rendering.FusedVolumeRenderer
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

## Scatter & context blocks

These building blocks are used when a physically motivated scatter term should
be added to the primary Beer–Lambert image.

### DepthAwareScatter

```{eval-rst}
.. autoclass:: flashdrr.rendering.DepthAwareScatter
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

### ASPP

```{eval-rst}
.. autoclass:: flashdrr.rendering.ASPP
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
```

## Camera & geometry helpers

Helper functions to build view/projection matrices and sample realistic
C-arm (X-ray gantry) poses. The helpers re-exported from
`flashdrr.rendering` are listed below; a few additional internal helpers
(e.g. `lookAt`, `homogenize_mat`) live in `flashdrr.rendering.raycast` but
are not part of the public API.

### get_vtk_view_mat

Build a VTK-convention camera-to-world view matrix from a camera position,
focal point and view-up vector in RAS coordinates.

```{eval-rst}
.. autofunction:: flashdrr.rendering.get_vtk_view_mat
```

### get_random_carm_views

Sample random C-arm geometry (SID, AP angle, lateral angle, table translation)
and return a batch of VTK view matrices.

```{eval-rst}
.. autofunction:: flashdrr.rendering.get_random_carm_views
```

### carm_to_camera_params

Convert C-arm gantry parameters into a camera position, look-at point and
view-up vector in RAS coordinates.

```{eval-rst}
.. autofunction:: flashdrr.rendering.carm_to_camera_params
```

## Transfer functions

### piecewise_linear_channelwise

Per-channel piecewise-linear transfer function, typically used to map CT
values (normalized Hounsfield units) to attenuation/density or RGBA before
raycasting.

```{eval-rst}
.. autofunction:: flashdrr.rendering.piecewise_linear_channelwise
```
