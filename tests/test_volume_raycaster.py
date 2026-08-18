"""Tests for the VolumeRaycaster (Python-loop) renderer.

These tests cover:
  * forward-pass shape, finiteness, and value-range sanity
  * Beer-Lambert vs. cumprod (alpha compositing) branches
  * scatter branch (DepthAwareScatter)
  * tiling math equivalence to non-tiled
  * multi-view batch expansion
  * backward / autograd sanity
  * compute_clipping_distances helper

The Python-loop renderer is fully implemented in PyTorch + grid_sample, so
all of these run on CPU.
"""
from __future__ import annotations

import pytest
import torch

from flashdrr.rendering import (
    DepthAwareScatter,
    VolumeRaycaster,
    get_vtk_view_mat,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _identity_ras2ijk(shape_xyz):
    """RAS->IJK affine that places the volume centered at the world origin
    with 1mm voxels. shape_xyz = (D, H, W).
    """
    M = torch.eye(4, dtype=torch.float32)
    M[0, 3] = -(shape_xyz[2] - 1) / 2.0
    M[1, 3] = -(shape_xyz[1] - 1) / 2.0
    M[2, 3] = -(shape_xyz[0] - 1) / 2.0
    return torch.inverse(M)


def _camera(eye=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    return get_vtk_view_mat(eye, target, up).unsqueeze(0)


# ---------------------------------------------------------------------------
# Shape & finiteness
# ---------------------------------------------------------------------------
class TestForwardShape:
    def test_single_view_output_shape(self, small_random_volume, simple_view_mat):
        ren = VolumeRaycaster(resolution=(32, 32), ray_samples=16)
        out = ren(small_random_volume, view_mat=simple_view_mat, ras2ijk=_identity_ras2ijk((16, 16, 16)))
        assert out.shape == (1, 1, 32, 32)
        assert torch.isfinite(out).all()

    def test_multi_channel_output(self, small_random_volume_8ch, simple_view_mat):
        ren = VolumeRaycaster(resolution=(16, 16), ray_samples=8)
        out = ren(small_random_volume_8ch, view_mat=simple_view_mat, ras2ijk=_identity_ras2ijk((16, 16, 16)))
        assert out.shape == (1, 8, 16, 16)
        assert torch.isfinite(out).all()

    def test_batch_expansion_views_per_vol(self, small_random_volume, batch_view_mat):
        # Two volumes, three views each -> output should be 6 entries
        vol = small_random_volume.expand(2, -1, -1, -1, -1).contiguous()
        # view_mat has shape (2, 4, 4). Repeat each row 3x to get (6, 4, 4)
        view = batch_view_mat.repeat_interleave(3, dim=0)
        ren = VolumeRaycaster(resolution=(16, 16), ray_samples=8)
        out = ren(vol, view_mat=view, ras2ijk=_identity_ras2ijk((16, 16, 16)))
        assert out.shape == (6, 1, 16, 16)


# ---------------------------------------------------------------------------
# Beer-Lambert vs. cumprod
# ---------------------------------------------------------------------------
class TestRenderingModes:
    def test_beer_lambert_output_range(self, small_random_volume, simple_view_mat):
        # BL output is 1 - exp(-sum(d * step)), so for non-negative d it lives
        # in [0, 1). With a non-empty ray, expect strictly positive values.
        ren = VolumeRaycaster(resolution=(16, 16), ray_samples=16, use_beer_lambert=True)
        out = ren(small_random_volume, view_mat=simple_view_mat, ras2ijk=_identity_ras2ijk((16, 16, 16)))
        assert (out >= 0).all()
        assert (out < 1.0).all()
        assert out.max() > 0.0

    def test_beer_lambert_increases_with_density(self, simple_view_mat):
        # Doubling the density should monotonically (in mean) increase the
        # transmission drop, so BL output should be higher on average.
        torch.manual_seed(0)
        vol_a = torch.rand(1, 1, 16, 16, 16) * 0.1  # low density
        vol_b = vol_a * 4.0  # 4x density
        ras2ijk = _identity_ras2ijk((16, 16, 16))
        ren = VolumeRaycaster(resolution=(16, 16), ray_samples=16, use_beer_lambert=True)
        out_a = ren(vol_a, view_mat=simple_view_mat, ras2ijk=ras2ijk)
        out_b = ren(vol_b, view_mat=simple_view_mat, ras2ijk=ras2ijk)
        assert out_b.mean() > out_a.mean()

    def test_cumprod_branch_runs(self, small_random_volume, simple_view_mat):
        ren = VolumeRaycaster(
            resolution=(16, 16),
            ray_samples=8,
            use_beer_lambert=False,
            density_factor=50.0,
        )
        out = ren(small_random_volume, view_mat=simple_view_mat, ras2ijk=_identity_ras2ijk((16, 16, 16)))
        assert out.shape == (1, 1, 16, 16)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Scatter branch
# ---------------------------------------------------------------------------
class TestScatterBranch:
    def test_scatter_with_extra_channels(self):
        # Use 2 channels: 1 main + 1 scatter. The renderer expects the scatter
        # channels to be the last `scatter_channels` entries of the channel dim.
        torch.manual_seed(0)
        vol = torch.rand(1, 2, 16, 16, 16)
        view = _camera()
        ren = VolumeRaycaster(resolution=(16, 16), ray_samples=8, scatter=1, use_beer_lambert=True)
        out = ren(vol, view_mat=view, ras2ijk=_identity_ras2ijk((16, 16, 16)))
        assert out.shape == (1, 2, 16, 16)
        assert torch.isfinite(out).all()
        # Beer-Lambert range still applies
        assert (out >= 0).all()
        assert (out <= 1.0 + 1e-5).all()

    def test_scatter_none_matches_no_scatter(self, small_random_volume, simple_view_mat):
        # When scatter is disabled, behaviour must be identical to a default
        # VolumeRaycaster call.
        torch.manual_seed(0)
        vol = small_random_volume
        ren_none = VolumeRaycaster(resolution=(16, 16), ray_samples=8, scatter=None)
        ren_default = VolumeRaycaster(resolution=(16, 16), ray_samples=8)
        out_none = ren_none(vol, view_mat=simple_view_mat, ras2ijk=_identity_ras2ijk((16, 16, 16)))
        out_default = ren_default(vol, view_mat=simple_view_mat, ras2ijk=_identity_ras2ijk((16, 16, 16)))
        assert torch.allclose(out_none, out_default, atol=1e-6)


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------
class TestTiling:
    def test_tiled_equals_non_tiled(self, small_random_volume, simple_view_mat):
        torch.manual_seed(0)
        vol = small_random_volume
        view = simple_view_mat
        ren = VolumeRaycaster(resolution=(64, 64), ray_samples=16)
        # tile_h=tile_w larger than the image -> single call
        out_single = ren(vol, view_mat=view, ras2ijk=_identity_ras2ijk((16, 16, 16)), tile_h=128, tile_w=128)
        # tile_h=tile_w smaller than the image -> multiple tiles
        out_tiled = ren(vol, view_mat=view, ras2ijk=_identity_ras2ijk((16, 16, 16)), tile_h=16, tile_w=16)
        # Tiling should not change the result (same math, just split)
        assert torch.allclose(out_single, out_tiled, atol=1e-5, rtol=0)
        # And the tiled result should have the same shape
        assert out_tiled.shape == out_single.shape


# ---------------------------------------------------------------------------
# Multi-view handling
# ---------------------------------------------------------------------------
class TestMultiView:
    def test_repeated_views_produce_identical_outputs(self, small_random_volume):
        # view_mat with 2 rows but both rows identical should match the single
        # camera case.
        torch.manual_seed(0)
        vol = small_random_volume
        view = _camera().repeat(2, 1, 1)  # (2, 4, 4) but both views identical
        ren = VolumeRaycaster(resolution=(16, 16), ray_samples=8)
        out = ren(vol, view_mat=view, ras2ijk=_identity_ras2ijk((16, 16, 16)))
        assert out.shape == (2, 1, 16, 16)
        # Both views should produce the same render
        assert torch.allclose(out[0], out[1], atol=1e-6)

    def test_n_must_be_multiple_of_bs(self, small_random_volume):
        # Mismatched N should raise
        torch.manual_seed(0)
        vol = small_random_volume
        # B=1, but N=2 -> 2 must be multiple of 1 (so it works).
        # Conversely: 2 volumes with 3 views -> 3 is not multiple of 2.
        vol2 = small_random_volume.expand(2, -1, -1, -1, -1).contiguous()
        bad_view = _camera().repeat(3, 1, 1)
        ren = VolumeRaycaster(resolution=(8, 8), ray_samples=4)
        with pytest.raises(ValueError):
            ren(vol2, view_mat=bad_view, ras2ijk=_identity_ras2ijk((16, 16, 16)))


# ---------------------------------------------------------------------------
# Backward / autograd
# ---------------------------------------------------------------------------
class TestAutograd:
    def test_backward_produces_finite_grad(self, simple_view_mat):
        torch.manual_seed(0)
        vol = torch.rand(1, 1, 16, 16, 16, requires_grad=True)
        ren = VolumeRaycaster(resolution=(16, 16), ray_samples=8, use_beer_lambert=True)
        out = ren(vol, view_mat=simple_view_mat, ras2ijk=_identity_ras2ijk((16, 16, 16)))
        out.sum().backward()
        assert vol.grad is not None
        assert torch.isfinite(vol.grad).all()
        # Every ray hits at least some volume, so every voxel along any ray
        # should receive some gradient.
        assert vol.grad.abs().sum() > 0

    def test_gradient_well_shaped(self, simple_view_mat):
        torch.manual_seed(0)
        vol = torch.rand(1, 1, 16, 16, 16, requires_grad=True)
        ren = VolumeRaycaster(resolution=(16, 16), ray_samples=8, use_beer_lambert=True)
        out = ren(vol, view_mat=simple_view_mat, ras2ijk=_identity_ras2ijk((16, 16, 16)))
        out.sum().backward()
        assert vol.grad.shape == vol.shape

    def test_checkpointing_eval_matches_no_checkpoint(self, simple_view_mat):
        # The renderer applies checkpointing only in training mode. In eval
        # mode both forward paths should be identical. (The training-mode
        # path uses torch.utils.checkpoint inside a vmapped function, which
        # is unsupported in stock PyTorch — that's a known limitation, not
        # something we test here.)
        torch.manual_seed(0)
        vol = torch.rand(1, 1, 16, 16, 16)
        ren_plain = VolumeRaycaster(resolution=(16, 16), ray_samples=8, use_checkpointing=False)
        ren_ckpt = VolumeRaycaster(resolution=(16, 16), ray_samples=8, use_checkpointing=True)
        ren_ckpt.eval()
        a = ren_plain(vol, view_mat=simple_view_mat, ras2ijk=_identity_ras2ijk((16, 16, 16)))
        b = ren_ckpt(vol, view_mat=simple_view_mat, ras2ijk=_identity_ras2ijk((16, 16, 16)))
        assert torch.allclose(a, b, atol=1e-5)

    def test_i0_enables_poisson_noise(self, simple_view_mat):
        # With i0 set, the renderer should inject noise into transmission.
        torch.manual_seed(0)
        vol = torch.rand(1, 1, 16, 16, 16)
        ren = VolumeRaycaster(resolution=(16, 16), ray_samples=8, i0=1e3, use_beer_lambert=True)
        ren.eval()
        a = ren(vol, view_mat=simple_view_mat, ras2ijk=_identity_ras2ijk((16, 16, 16)))
        b = ren(vol, view_mat=simple_view_mat, ras2ijk=_identity_ras2ijk((16, 16, 16)))
        # Two independent calls should not be bit-identical
        assert not torch.equal(a, b)
        # The Gaussian noise has std = sqrt(transmission / i0); with i0=1e3
        # and transmission in [0, 1] that's at most ~0.032. Allow 5x that as
        # a generous per-element bound, with a per-tensor mean check that the
        # averages still match closely.
        assert (a - b).abs().mean() < 5e-2
        assert (a - b).abs().max() < 0.3
        assert (a.mean() - b.mean()).abs() < 5e-2


# ---------------------------------------------------------------------------
# compute_clipping_distances
# ---------------------------------------------------------------------------
class TestClippingDistances:
    def test_clipping_distances_batched(self):
        # Use a wide volume and cameras outside the volume so the
        # min/max clipping distances are well above the 0.1 floor.
        torch.manual_seed(0)
        shape = (32, 32, 32)
        # RAS->IJK affine: IJK=0 maps to RAS=(-W/2, -H/2, -D/2)
        ijk2ras = torch.eye(4)
        ijk2ras[0, 3] = -shape[2] / 2.0
        ijk2ras[1, 3] = -shape[1] / 2.0
        ijk2ras[2, 3] = -shape[0] / 2.0
        # Two cameras looking at the origin from +Z, at different distances
        V1 = get_vtk_view_mat((0.0, 0.0, 100.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        V2 = get_vtk_view_mat((0.0, 0.0, 200.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        V = torch.stack([V1, V2], dim=0)
        near, far = VolumeRaycaster.compute_clipping_distances(V, shape, ijk2ras)
        assert near.shape == (2,)
        assert far.shape == (2,)
        # Far > Near (always true by construction)
        assert (far > near).all()
        # The further camera should have larger near AND larger far
        assert near[1] > near[0]
        assert far[1] > far[0]
        # And both should be strictly above the 0.1 safety floor
        assert (near > 0.1).all()

    def test_clipping_distances_single(self):
        shape = (32, 32, 32)
        ijk2ras = torch.eye(4)
        V = get_vtk_view_mat((0.0, 0.0, 100.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        near, far = VolumeRaycaster.compute_clipping_distances(V, shape, ijk2ras)
        # Returns Python floats for unbatched input
        assert isinstance(near, float)
        assert isinstance(far, float)
        assert far > near
        assert near > 0.1


# ---------------------------------------------------------------------------
# DepthAwareScatter module directly
# ---------------------------------------------------------------------------
class TestDepthAwareScatter:
    def test_forward_shapes(self):
        torch.manual_seed(0)
        sc = DepthAwareScatter(in_ch=2, base_ch=4, alpha_max=0.1)
        # Input: (B, C, D, H, W)
        mu = torch.rand(1, 2, 8, 8, 8)
        I_out, I_primary, scatter_map, alpha = sc(mu, dz=0.5)
        assert I_out.shape == (1, 2, 8, 8)
        assert I_primary.shape == (1, 2, 8, 8)
        assert scatter_map.shape == (1, 1, 8, 8)
        assert alpha.shape == (1, 1, 8, 8)
        # alpha is bounded by alpha_max
        assert (alpha <= 0.1 + 1e-6).all()
        assert (alpha >= 0.0).all()

    def test_scatter_at_init_is_near_zero(self):
        # At init the scatter head's bias is set so alpha is near 0; the output
        # should therefore be very close to the unscattered primary intensity.
        torch.manual_seed(0)
        sc = DepthAwareScatter(in_ch=1, base_ch=4, alpha_max=0.4)
        mu = torch.rand(1, 1, 8, 8, 8) * 0.5  # low density so the primary is high
        I_out, I_primary, _, _ = sc(mu, dz=0.1)
        # Scattered output should be close to I_primary
        assert torch.allclose(I_out, I_primary, atol=1e-2)
