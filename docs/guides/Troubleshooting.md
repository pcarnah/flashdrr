## Troubleshooting

### Known Issues

* The Triton-fused `FusedVolumeRenderer` requires CUDA and a working Triton
  installation. On Windows the `triton-windows` fork is used automatically —
  if Triton is unavailable, keep `triton=False` (the default) so rendering
  falls back to the pure-PyTorch `VolumeRaycaster` path.
* `FusedVolumeRenderer` explicitly disables CUDA autocasting; controlling the
  accumulation dtype manually. If you rely on mixed precision, do your
  up/down-casting around the render call.

### Frequently Asked Questions

Go ask us some questions on [GitHub](https://github.com/torchvtk/torchvtk)!
