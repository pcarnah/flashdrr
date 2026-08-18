"""Tests that require a GPU + Triton.

These tests compare the Triton-fused renderer against the Python-loop
``VolumeRaycaster`` the same way the ``__main__`` block in ``raycast.py``
does. Triton is only built for CUDA / ROCm, so every test here is
auto-skipped on CPU-only machines.

What we check:
  * Forward outputs of Triton vs Python match within a tolerance
    (the two paths sample density at slightly different boundaries, so
    a pixel-perfect match is not expected).
  * The output values stay in the physically-meaningful [0, 1] range.
  * Multi-view batch expansion works on both paths.
  * The Triton path's autograd is correct at fp64 precision (the only
    precision where ``torch.autograd.gradcheck`` is strict enough to
    meaningfully test against an atomic-add backward).
  * The ``FusedVolumeRenderer`` module works on its own (not just
    through the ``VolumeRaycaster`` wrapper).

Note on dtypes: the renderer's forward explicitly disables CUDA autocast
(``raycast.py:715``), so all internal math runs at fp32 regardless of
the input's dtype. The "fp16 / bf16 timings" in ``__main__`` are
performance comparisons only — they do not actually exercise lower
precision rendering. These tests therefore only cover fp32.
"""
from __future__ import annotations

import pytest
import torch

from flashdrr.rendering import (
    FusedVolumeRenderer,
    VolumeRaycaster,
    get_vtk_view_mat,
)


pytestmark = [pytest.mark.gpu, pytest.mark.triton]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _identity_ras2ijk(shape_xyz, device):
    """RAS->IJK affine that places the volume centered at the world origin
    with 1mm voxels. shape_xyz = (D, H, W).
    """
    M = torch.eye(4, dtype=torch.float32, device=device)
    M[0, 3] = -(shape_xyz[2] - 1) / 2.0
    M[1, 3] = -(shape_xyz[1] - 1) / 2.0
    M[2, 3] = -(shape_xyz[0] - 1) / 2.0
    return torch.inverse(M)


def _camera(device):
    return get_vtk_view_mat(
        (0.0, 0.0, 5.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    ).to(device).unsqueeze(0)


# ---------------------------------------------------------------------------
# Forward parity: Triton vs Python-loop
# ---------------------------------------------------------------------------
def test_triton_matches_python_fp32(gpu_device):
    """Single-view, 1-channel, fp32 — the canonical comparison from
    ``__main__`` (lines 940-944). Tolerances match the dev's defaults
    there (``atol=1e-4``) but are relaxed slightly to account for the
    fact that the two paths sample density at slightly different
    boundaries.
    """
    torch.manual_seed(0)
    device = gpu_device
    vol = torch.rand(1, 1, 16, 16, 16, device=device, dtype=torch.float32)
    view = _camera(device)
    ras2ijk = _identity_ras2ijk((16, 16, 16), device)

    ren = VolumeRaycaster(resolution=(32, 32), ray_samples=16, use_beer_lambert=True).to(device)
    with torch.no_grad():
        out_py = ren(vol, view_mat=view, ras2ijk=ras2ijk)
        out_triton = ren(vol, view_mat=view, ras2ijk=ras2ijk, triton=True)

    # Shape and dtype contract
    assert out_py.shape == out_triton.shape == (1, 1, 32, 32)
    # Both should be in the Beer-Lambert output range
    assert (out_py >= 0).all() and (out_py < 1.0).all()
    assert (out_triton >= 0).all() and (out_triton < 1.0).all()
    # And the two outputs should agree (same physical scene, two implementations)
    assert torch.allclose(out_py, out_triton, atol=1e-3, rtol=1e-3), (
        f"max diff {(out_py - out_triton).abs().max().item()}, "
        f"mean diff {(out_py - out_triton).abs().mean().item()}"
    )


def test_triton_matches_python_8channels(gpu_device):
    """Match the dev-script's main configuration: 8 channels, large
    resolution. Scaled down to keep test time reasonable.
    """
    torch.manual_seed(0)
    device = gpu_device
    vol = torch.rand(1, 8, 16, 16, 16, device=device, dtype=torch.float32)
    view = _camera(device)
    ras2ijk = _identity_ras2ijk((16, 16, 16), device)

    ren = VolumeRaycaster(resolution=(64, 64), ray_samples=16, use_beer_lambert=True).to(device)
    with torch.no_grad():
        out_py = ren(vol, view_mat=view, ras2ijk=ras2ijk)
        out_triton = ren(vol, view_mat=view, ras2ijk=ras2ijk, triton=True)
    assert out_py.shape == out_triton.shape == (1, 8, 64, 64)
    assert torch.allclose(out_py, out_triton, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Multi-view (N > 1) and batch expansion on both paths
# ---------------------------------------------------------------------------
def test_triton_handles_multi_view(gpu_device):
    """B=1, N=4. Both paths must support views_per_vol > 1."""
    torch.manual_seed(0)
    device = gpu_device
    vol = torch.rand(1, 1, 16, 16, 16, device=device, dtype=torch.float32)
    # 4 cameras, all looking at origin from different distances
    cam = _camera(device)[0].clone()
    views = cam.unsqueeze(0).expand(4, -1, -1).contiguous()
    for i in range(4):
        views[i, :3, 3] = torch.tensor([0.0, 0.0, 5.0 + i * 0.5], device=device)
    ras2ijk = _identity_ras2ijk((16, 16, 16), device)

    ren = VolumeRaycaster(resolution=(32, 32), ray_samples=16, use_beer_lambert=True).to(device)
    with torch.no_grad():
        out_py = ren(vol, view_mat=views, ras2ijk=ras2ijk)
        out_triton = ren(vol, view_mat=views, ras2ijk=ras2ijk, triton=True)
    assert out_py.shape == out_triton.shape == (4, 1, 32, 32)
    assert torch.allclose(out_py, out_triton, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Multi-batch (B > 1, N = B)
# ---------------------------------------------------------------------------
def test_triton_handles_batch(gpu_device):
    """B=2, N=2. Same number of cameras as volumes."""
    torch.manual_seed(0)
    device = gpu_device
    vol = torch.rand(2, 1, 16, 16, 16, device=device, dtype=torch.float32)
    cam = _camera(device)[0]
    view = torch.stack([cam.clone(), cam.clone()], dim=0)
    ras2ijk = _identity_ras2ijk((16, 16, 16), device)

    ren = VolumeRaycaster(resolution=(32, 32), ray_samples=16, use_beer_lambert=True).to(device)
    with torch.no_grad():
        out_py = ren(vol, view_mat=view, ras2ijk=ras2ijk)
        out_triton = ren(vol, view_mat=view, ras2ijk=ras2ijk, triton=True)
    assert out_py.shape == out_triton.shape == (2, 1, 32, 32)
    assert torch.allclose(out_py, out_triton, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Scatter branch (Triton path)
# ---------------------------------------------------------------------------
def test_triton_with_scatter(gpu_device):
    """When scatter is enabled, the renderer's scatter branch is
    implemented purely in PyTorch (not in Triton), so the test mainly
    confirms the dispatch doesn't crash and the output shape is right.
    """
    torch.manual_seed(0)
    device = gpu_device
    # Need scatter_channels extra channels in the volume
    vol = torch.rand(1, 2, 16, 16, 16, device=device, dtype=torch.float32)
    view = _camera(device)
    ras2ijk = _identity_ras2ijk((16, 16, 16), device)

    ren = VolumeRaycaster(
        resolution=(32, 32), ray_samples=16, scatter=1, use_beer_lambert=True
    ).to(device)
    with torch.no_grad():
        out_triton = ren(vol, view_mat=view, ras2ijk=ras2ijk, triton=True)
        out_py = ren(vol, view_mat=view, ras2ijk=ras2ijk)
    assert out_py.shape == out_triton.shape == (1, 2, 32, 32)
    assert torch.isfinite(out_triton).all()
    assert (out_triton >= 0).all()


# ---------------------------------------------------------------------------
# Output is in caller's dtype
# ---------------------------------------------------------------------------
def test_triton_output_dtype_matches_input(gpu_device):
    """The FusedVolumeRenderer is documented to return the caller's dtype
    (see triton_raycast.py:678); verify that contract.
    """
    torch.manual_seed(0)
    device = gpu_device
    vol = torch.rand(1, 1, 16, 16, 16, device=device, dtype=torch.float32)
    view = _camera(device)
    ras2ijk = _identity_ras2ijk((16, 16, 16), device)

    ren = VolumeRaycaster(resolution=(16, 16), ray_samples=8).to(device)
    with torch.no_grad():
        out = ren(vol, view_mat=view, ras2ijk=ras2ijk, triton=True)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# FusedVolumeRenderer exposed directly
# ---------------------------------------------------------------------------
def test_fused_volume_renderer_directly(gpu_device):
    """Exercise the FusedVolumeRenderer module on its own (not through
    VolumeRaycaster's wrapper) to make sure the public API still works
    independent of the Python-loop fallback.
    """
    torch.manual_seed(0)
    device = gpu_device
    B, C, D, H, W = 1, 1, 16, 16, 16
    density = torch.rand(B, C, W, H, D, device=device, dtype=torch.float32)

    # dirs_cam: (H, W, 3) — must be unit-norm camera-space ray directions
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    dirs_cam = torch.stack(
        [x, -y, torch.ones_like(x)], dim=-1
    )
    dirs_cam = torch.nn.functional.normalize(dirs_cam, dim=-1)

    # Single camera looking from +Z
    view = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
    view[:, :3, 3] = torch.tensor([0.0, 0.0, 5.0], device=device)
    view[:, :3, 2] = torch.tensor([0.0, 0.0, -1.0], device=device)
    view[:, :3, 0] = torch.tensor([1.0, 0.0, 0.0], device=device)
    view[:, :3, 1] = torch.tensor([0.0, 1.0, 0.0], device=device)

    # Identity-ish RAS->IJK (centered at origin)
    M = torch.eye(4, device=device)
    M[0, 3] = -(W - 1) / 2.0
    M[1, 3] = -(H - 1) / 2.0
    M[2, 3] = -(D - 1) / 2.0
    ras2ijk = torch.inverse(M)
    vol_shape = torch.tensor([D, H, W], device=device, dtype=torch.float32)
    near = torch.tensor([0.1], device=device)
    far = torch.tensor([10.0], device=device)

    fvr = FusedVolumeRenderer(ray_samples=16)
    with torch.no_grad():
        out = fvr(density, view, near, far, ras2ijk, vol_shape, dirs_cam)
    assert out.shape == (B, C, H, W)
    assert torch.isfinite(out).all()
    assert (out >= 0).all() and (out <= 1.0 + 1e-5).all()


# ---------------------------------------------------------------------------
# gradcheck for the Triton backward
# ---------------------------------------------------------------------------
def test_triton_gradcheck_fp64(gpu_device):
    """Numerical-vs-analytical gradient agreement for the Triton path.

    The Triton backward uses ``tl.atomic_add`` for accumulation, which
    is fundamentally order-nondeterministic — many threads write to
    the same voxel in an unpredictable order, and the floating-point
    sum isn't associative across orderings. So ``gradcheck``'s built-in
    reentrancy check is structurally unsatisfiable for this op.

    Reproduces the gradcheck call from ``__main__`` (lines 1102-1118)
    with a much smaller volume so it finishes in seconds, and bypasses
    the reentrancy check by setting ``nondet_tol=1.0`` (effectively
    disabling it) while keeping the analytical-vs-numerical tolerance
    strict.
    """
    from torch.autograd import gradcheck

    torch.manual_seed(0)
    device = gpu_device
    # ``mu`` is 4D (C, D, H, W) so that gradcheck can perturb each voxel
    # independently. The renderer needs a 5D (B, C, D, H, W) input so we
    # expand below. This matches the gradcheck call from ``__main__``
    # (raycast.py:1103-1117) but scaled down to keep test time low.
    mu = torch.randn(
        2, 6, 6, 6, device=device, dtype=torch.float64, requires_grad=True
    )

    # Build a fp64 camera
    view = get_vtk_view_mat(
        (0.0, 0.0, 5.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    ).to(device=device, dtype=torch.float64).unsqueeze(0)
    M = torch.eye(4, dtype=torch.float64, device=device)
    M[0, 3] = M[1, 3] = M[2, 3] = -2.5
    ras2ijk = torch.inverse(M)

    ren = VolumeRaycaster(
        resolution=(4, 4), ray_samples=4, use_beer_lambert=True
    ).to(device)
    ren.eval()

    # First check: gradient has the right shape, is finite, and two
    # independent backward passes agree to a meaningful extent (the
    # per-element *values* will differ due to atomic-add ordering, but
    # the per-voxel mean and sum should still match).
    out1 = ren(mu.expand(1, -1, -1, -1, -1), view_mat=view, ras2ijk=ras2ijk, triton=True)
    g1 = torch.autograd.grad(out1.sum(), mu, create_graph=False)[0]
    assert g1.shape == mu.shape
    assert torch.isfinite(g1).all()

    out2 = ren(mu.expand(1, -1, -1, -1, -1), view_mat=view, ras2ijk=ras2ijk, triton=True)
    g2 = torch.autograd.grad(out2.sum(), mu, create_graph=False)[0]
    assert abs(g1.mean().item() - g2.mean().item()) < 1e-3
    assert abs(g1.sum().item() - g2.sum().item()) < 1e-2

    # Then run gradcheck with a very large nondet_tol to bypass the
    # reentrancy check. The analytical-vs-numerical tolerance
    # (``atol=1e-4``) is the strict one — that's where correctness
    # issues would actually show up.
    gradcheck(
        lambda m: ren(m.expand(1, -1, -1, -1, -1), view_mat=view, ras2ijk=ras2ijk, triton=True),
        (mu.detach().clone().requires_grad_(True),),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        nondet_tol=1.0,
        fast_mode=True,
    )


# ---------------------------------------------------------------------------
# Backward parity at fp32
# ---------------------------------------------------------------------------
def test_triton_backward_finite(gpu_device):
    """The Triton backward uses atomic adds which have non-deterministic
    ordering, so we don't bit-compare gradients against the Python
    path. We just check that they are finite, non-zero, and have the
    right shape.
    """
    torch.manual_seed(0)
    device = gpu_device
    vol = torch.rand(1, 1, 16, 16, 16, device=device, dtype=torch.float32, requires_grad=True)
    view = _camera(device)
    ras2ijk = _identity_ras2ijk((16, 16, 16), device)

    ren = VolumeRaycaster(resolution=(32, 32), ray_samples=16, use_beer_lambert=True).to(device)
    out = ren(vol, view_mat=view, ras2ijk=ras2ijk, triton=True)
    out.sum().backward()
    assert vol.grad is not None
    assert vol.grad.shape == vol.shape
    assert torch.isfinite(vol.grad).all()
    assert vol.grad.abs().sum() > 0
