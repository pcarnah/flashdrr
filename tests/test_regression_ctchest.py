"""Regression tests for the CTChest.nii.gz sample data.

These tests load the bundled CTChest volume, apply the same transfer
function / mu conversion as the dev's ``__main__`` in ``raycast.py``,
and render with the VolumeRaycaster. The output is compared against a
checked-in golden fixture in ``tests/fixtures/``:

  * ``ctchest_drr_stats.json``         — summary stats (min/max/mean/std)
  * ``ctchest_drr_thumbnail.npy``      — full 128x128 output, quantized to 4 decimals
  * ``ctchest_drr_triton_stats.json``  — same, for the triton path (GPU only)
  * ``ctchest_drr_triton_thumbnail.npy`` — same, for the triton path

All comparisons are float-aware (allclose-style) with tolerances chosen
to absorb minor numerical drift across hardware / cuDNN versions; we do
NOT do exact bit-for-bit / hash matching because small floating-point
variation between platforms would cause spurious failures.

If you intentionally change rendering behaviour and need to refresh
the references, re-run ``tests/_tools/regenerate_ctchest_reference.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from flashdrr.rendering import VolumeRaycaster, get_vtk_view_mat


FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------
def _load_ref(filename: str) -> dict:
    """Load a JSON reference fixture. Skip the test if it's missing."""
    path = FIXTURES / filename
    if not path.exists():
        pytest.skip(f"Reference fixture not found: {path}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reproducible rendering of the canonical CTChest DRR
# ---------------------------------------------------------------------------
def _render_ctchest_drr(device, triton: bool = False):
    """Render the CTChest volume with the dev's ``__main__`` setup.

    Returns the rendered tensor (1, 1, 128, 128) on `device`.
    """
    from monai.transforms import (
        Compose,
        EnsureChannelFirst,
        EnsureType,
        LoadImage,
        ScaleIntensityRange,
    )

    load_tf = Compose(
        [
            LoadImage(),
            EnsureChannelFirst(),
            ScaleIntensityRange(-3024, 3024, 0, 1, clip=True),
            EnsureType(),
        ]
    )
    vol = load_tf(str(Path(__file__).resolve().parent.parent / "CTChest.nii.gz"))
    vol = vol.unsqueeze(0)  # (1, 1, D, H, W)

    hu = vol * (3024.0 - (-3524.0)) + (-3524.0)
    mu = torch.clamp(0.05 * (1.0 + hu / 800.0), min=0.0)

    ijk2ras = vol.meta["affine"]
    ras2ijk = torch.inverse(ijk2ras)

    # Center of the volume in RAS
    center = torch.ones(4, dtype=torch.float64)
    center[:3] = torch.as_tensor(vol.shape[2:]) // 2
    center = ijk2ras @ center

    # The camera the dev used in __main__
    view_mat = get_vtk_view_mat(
        (0.0, 1000.0, -130.0), center[:3], (0.0, 0.0, 1.0)
    ).unsqueeze(0)

    ren = VolumeRaycaster(
        scatter=None, resolution=(128, 128), i0=None, ray_samples=128
    ).to(device).eval()

    mu = mu.to(device)
    view_mat = view_mat.to(device)
    ras2ijk = ras2ijk.to(device)

    with torch.no_grad():
        out = ren(mu, view_mat=view_mat, ras2ijk=ras2ijk, triton=triton)
    return out


# ---------------------------------------------------------------------------
# Python-loop path (CPU-runnable)
# ---------------------------------------------------------------------------
@pytest.mark.regression
class TestCTChestDRR:
    """The dev's ``__main__`` produces a known-good DRR. These tests
    guard against accidental regressions in the rendering pipeline by
    re-running the same setup and comparing against the golden fixture.
    """

    def test_summary_stats_match_reference(self, device):
        out = _render_ctchest_drr(device, triton=False)
        ref = _load_ref("ctchest_drr_stats.json")
        # Exact shape
        assert list(out.shape) == ref["shape"]
        # Float-aware comparison of summary stats. Tolerances are chosen
        # to absorb minor floating-point drift across hardware / cuDNN
        # versions while still catching real regressions.
        stats = {
            "mean": out.mean().item(),
            "std": out.std().item(),
            "min": out.min().item(),
            "max": out.max().item(),
        }
        for name, tol in (("mean", 1e-3), ("std", 1e-3), ("min", 1e-3), ("max", 1e-3)):
            assert abs(stats[name] - ref[name]) < tol, (
                f"{name} drift: {stats[name]} vs ref {ref[name]} (tol {tol})"
            )

    def test_pixel_values_match_thumbnail(self, device):
        """Compare the full output to a quantized thumbnail using
        ``np.allclose`` rather than exact equality. The thumbnail itself
        is rounded to 4 decimals to save space, and the render output
        can vary slightly across hardware, so we compare with a
        tolerance of 1e-3 absolute / 1e-3 relative.
        """
        out = _render_ctchest_drr(device, triton=False)
        path = FIXTURES / "ctchest_drr_thumbnail.npy"
        if not path.exists():
            pytest.skip(f"Thumbnail fixture not found: {path}")
        thumb = np.load(path)
        # Shape contract
        assert out.shape == (1, 1, 128, 128)
        assert thumb.shape == (128, 128)
        # Float-aware pixel-wise comparison against the rounded reference
        np.testing.assert_allclose(
            out[0, 0].cpu().numpy(),
            thumb,
            atol=1e-3,
            rtol=1e-3,
            err_msg="rendered output diverged from reference thumbnail",
        )

    def test_value_range_is_physically_meaningful(self, device):
        """Sanity-only: the DRR of an X-ray-attenuating phantom should
        cover a non-trivial portion of the [0, 1] range and not be all
        one value.
        """
        out = _render_ctchest_drr(device, triton=False)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0
        # The std should be non-trivial
        assert out.std().item() > 0.05


# ---------------------------------------------------------------------------
# Triton path (GPU required)
# ---------------------------------------------------------------------------
@pytest.mark.regression
@pytest.mark.gpu
@pytest.mark.triton
class TestCTChestDRRTriton:
    """Same regression as above, but for the Triton-fused path. Skipped
    on CPU-only machines.
    """

    def test_summary_stats_match_reference(self, gpu_device):
        out = _render_ctchest_drr(gpu_device, triton=True)
        ref = _load_ref("ctchest_drr_triton_stats.json")
        assert list(out.shape) == ref["shape"]
        stats = {
            "mean": out.mean().item(),
            "std": out.std().item(),
            "min": out.min().item(),
            "max": out.max().item(),
        }
        for name, tol in (("mean", 1e-3), ("std", 1e-3), ("min", 1e-3), ("max", 1e-3)):
            assert abs(stats[name] - ref[name]) < tol, (
                f"{name} drift: {stats[name]} vs ref {ref[name]} (tol {tol})"
            )

    def test_pixel_values_match_thumbnail(self, gpu_device):
        out = _render_ctchest_drr(gpu_device, triton=True)
        path = FIXTURES / "ctchest_drr_triton_thumbnail.npy"
        if not path.exists():
            pytest.skip(f"Thumbnail fixture not found: {path}")
        thumb = np.load(path)
        np.testing.assert_allclose(
            out[0, 0].cpu().numpy(),
            thumb,
            atol=1e-3,
            rtol=1e-3,
            err_msg="triton output diverged from reference thumbnail",
        )

    def test_triton_matches_python_within_tolerance(self, gpu_device):
        """The Triton and Python-loop paths should produce the same DRR
        of CTChest to a few times the per-pixel error of either path on
        its own (atomic-add nondeterminism etc).
        """
        out_py = _render_ctchest_drr(gpu_device, triton=False)
        out_triton = _render_ctchest_drr(gpu_device, triton=True)
        diff = (out_py - out_triton).abs()
        assert diff.mean().item() < 5e-4
        assert diff.max().item() < 5e-3
