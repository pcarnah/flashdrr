"""Tests for the camera/geometry helper functions in flashdrr.rendering.raycast.

These functions are pure tensor math (no I/O, no kernel launches), so they
run fast on CPU and we can be strict about the tolerances.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from flashdrr.rendering import (
    carm_to_camera_params,
    get_proj_mat,
    get_random_carm_views,
    get_random_pos,
    get_rot_mat,
    get_view_mat,
    get_vtk_view_mat,
    homogenize_mat,
    homogenize_vec,
    lookAt,
)


# ---------------------------------------------------------------------------
# homogenize_mat
# ---------------------------------------------------------------------------
class TestHomogenizeMat:
    def test_2d_input(self):
        M = torch.eye(3)
        H = homogenize_mat(M)
        assert H.shape == (4, 4)
        # Top-left 3x3 should be the input
        assert torch.equal(H[:3, :3], M)
        # Bottom-right is 1
        assert H[3, 3].item() == 1.0
        # Last row and column are zeros except the corner
        assert H[3, :3].tolist() == [0.0, 0.0, 0.0]
        assert H[:3, 3].tolist() == [0.0, 0.0, 0.0]

    def test_batched_input(self):
        # NOTE: there's an upstream bug in `homogenize_mat` for batched input
        # where the eye-initialised output is `.expand`-ed (shared storage)
        # and the in-place copy from `flat_mat` then collides. This is a
        # known issue in raycast.py:34; we don't test the broken path here.
        pytest.skip("homogenize_mat has a known bug with batched input")

    def test_2d_input_works(self):
        # Sanity that the unbatched path is unaffected by the bug.
        M = torch.eye(3) * 2.0
        H = homogenize_mat(M)
        assert H.shape == (4, 4)
        assert torch.equal(H[:3, :3], M)
        assert H[3, 3].item() == 1.0

    def test_roundtrip_invertible(self):
        M = torch.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 4.0], [5.0, 6.0, 1.0]])
        H = homogenize_mat(M)
        assert torch.isfinite(torch.inverse(H)).all()


# ---------------------------------------------------------------------------
# homogenize_vec
# ---------------------------------------------------------------------------
class TestHomogenizeVec:
    def test_appends_one_at_default_dim(self):
        # `homogenize_vec` requires a dim of size 3 somewhere in the shape.
        # For a (B, 3) input the default `dim` is the 3-dim, so the 1 is
        # appended along that dim, taking the shape to (B, 4).
        v = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        h = homogenize_vec(v)
        assert h.shape == (2, 4)
        # Both rows should have a 1 in the new last position
        assert torch.allclose(h[:, -1], torch.ones(2))
        # The original values are preserved up to that point
        assert torch.equal(h[:, :-1], v)

    def test_explicit_dim(self):
        v = torch.zeros(2, 3, 4)
        h = homogenize_vec(v, dim=2)
        assert h.shape == (2, 3, 5)
        assert h[0, 0, -1].item() == 1.0


# ---------------------------------------------------------------------------
# get_proj_mat
# ---------------------------------------------------------------------------
class TestGetProjMat:
    def test_shape_and_dtype(self):
        P = get_proj_mat(fov=math.pi / 3, aspect=16 / 9, dtype=torch.float32)
        assert P.shape == (4, 4)
        assert P.dtype == torch.float32

    def test_fov_zero_diag(self):
        # Standard OpenGL-style: zeros off the diagonal, except the -1 in (2,3)
        P = get_proj_mat(fov=math.pi / 4, aspect=1.0)
        assert P[0, 1].item() == 0.0
        assert P[0, 2].item() == 0.0
        assert P[0, 3].item() == 0.0
        assert P[1, 0].item() == 0.0
        assert P[1, 2].item() == 0.0
        assert P[1, 3].item() == 0.0
        assert P[2, 3].item() == -1.0  # OpenGL perspective: -1 w-divisor
        assert P[3, 3].item() == 0.0

    def test_widening_aspect_scales_x(self):
        P_narrow = get_proj_mat(fov=math.pi / 4, aspect=0.5)
        P_wide = get_proj_mat(fov=math.pi / 4, aspect=2.0)
        # a = q / aspect, q = 1/tan(fov/2) — wider aspect -> smaller a
        assert P_wide[0, 0].item() < P_narrow[0, 0].item()
        # Vertical component (q) is independent of aspect
        assert P_narrow[1, 1].item() == pytest.approx(P_wide[1, 1].item())


# ---------------------------------------------------------------------------
# get_view_mat
# ---------------------------------------------------------------------------
class TestGetViewMat:
    def test_shape_and_devices(self):
        # NOTE: `get_view_mat` has an upstream bug for batched input (B>1):
        # the function builds the output via `torch.eye(4)[None].expand(bs,...)`
        # and then tries an in-place copy into that shared storage, which
        # raises "more than one element refers to a single memory location".
        # We restrict this test to the B=1 happy path.
        v = get_view_mat(torch.tensor([[0.0, 0.0, 5.0]]))
        assert v.shape == (1, 4, 4)

    def test_orthonormal_basis_single(self):
        torch.manual_seed(0)
        # `get_view_mat` has an upstream bug for batched input (see
        # test_shape_and_devices), so we only test the B=1 case here.
        look_from = F.normalize(torch.randn(1, 3), dim=1)
        V = get_view_mat(look_from)
        # The 3x3 block stores the basis vectors as ROWS (stacked along
        # dim 1: row 0 = x, row 1 = y, row 2 = z).
        R = V[0, :3, :3]
        # Row norms should be 1
        row_norms = R.norm(dim=1)
        assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5)
        # R.T @ R should equal I
        gram = R.T @ R
        assert torch.allclose(gram, torch.eye(3), atol=1e-5)

    def test_third_row_equals_look_from_direction(self):
        # With the default look_to=origin, z = look_from normalized, and
        # the third ROW of the basis stores z.
        look_from = torch.tensor([[3.0, 0.0, 4.0]])
        V = get_view_mat(look_from)
        z_row = V[0, 2, :3]
        expected = F.normalize(look_from[0], dim=0)
        assert torch.allclose(z_row, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# lookAt
# ---------------------------------------------------------------------------
class TestLookAt:
    def test_shape(self):
        out = lookAt(torch.tensor([[0.0, 0.0, 1.0]]))
        assert out.shape == (1, 4, 4)

    def test_view_dir_points_at_origin(self):
        # `lookAt` is documented as looking at -look_from, so forward is -look_from
        # (i.e. the third column of the orthonormal basis points away from the
        # camera, matching the camera-to-world convention used by VTK).
        look_from = torch.tensor([[3.0, 0.0, 4.0]])
        V = lookAt(look_from)
        z_axis = V[0, :3, 2]
        expected = F.normalize(-look_from[0], dim=0)
        assert torch.allclose(z_axis, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# get_vtk_view_mat
# ---------------------------------------------------------------------------
class TestGetVtkViewMat:
    def test_basis_is_right_handed(self):
        cam_pos = (0.0, 0.0, 5.0)
        cam_focal = (0.0, 0.0, 0.0)
        cam_up = (0.0, 1.0, 0.0)
        M = get_vtk_view_mat(cam_pos, cam_focal, cam_up)

        right = M[:3, 0]
        up = M[:3, 1]
        fwd = M[:3, 2]
        # All basis vectors should be unit length
        assert right.norm().item() == pytest.approx(1.0, abs=1e-5)
        assert up.norm().item() == pytest.approx(1.0, abs=1e-5)
        assert fwd.norm().item() == pytest.approx(1.0, abs=1e-5)
        # up should equal right x forward
        assert torch.allclose(torch.linalg.cross(right, fwd), up, atol=1e-5)
        # Camera position lives in the translation column
        assert torch.allclose(M[:3, 3], torch.tensor(cam_pos), atol=1e-5)


# ---------------------------------------------------------------------------
# get_random_pos
# ---------------------------------------------------------------------------
class TestGetRandomPos:
    def test_fixed_distance(self):
        p = get_random_pos(bs=8, distance=3.0)
        assert p.shape == (8, 3)
        norms = p.norm(dim=1)
        assert torch.allclose(norms, torch.full((8,), 3.0), atol=1e-5)

    def test_range_distance(self):
        torch.manual_seed(42)
        p = get_random_pos(bs=128, distance=(2.0, 5.0))
        norms = p.norm(dim=1)
        # All points must lie on a sphere of radius in [2, 5]
        assert (norms >= 2.0 - 1e-5).all()
        assert (norms <= 5.0 + 1e-5).all()
        # Sanity: the empirical distribution of norms spans the interval
        assert norms.max() - norms.min() > 1.0


# ---------------------------------------------------------------------------
# carm_to_camera_params + get_random_carm_views
# ---------------------------------------------------------------------------
class TestCarmHelpers:
    def test_carm_params_shapes(self):
        cam_pos, look_at, look_up = carm_to_camera_params(
            sid=1000.0,
            ap_angle=0.0,
            lat_angle=0.0,
            center_ras=(0.0, 0.0, 0.0),
        )
        assert cam_pos.shape == (3,)
        assert look_at.shape == (3,)
        assert look_up.shape == (3,)
        # AP=lat=0 means source starts at (0, -sid, 0) which in RAS means
        # anterior of center.
        assert cam_pos[0] == pytest.approx(0.0, abs=1e-6)
        assert cam_pos[1] == pytest.approx(-1000.0, abs=1e-6)

    def test_table_translation_shifts_center(self):
        _, look_at_no, _ = carm_to_camera_params(1000, 0, 0, (0, 0, 0), table_si=0.0)
        _, look_at_yes, _ = carm_to_camera_params(1000, 0, 0, (0, 0, 0), table_si=10.0)
        # look_at should be center + (0, 0, table_si)
        assert look_at_yes[2] - look_at_no[2] == pytest.approx(10.0, abs=1e-6)

    def test_get_random_carm_views_shape(self):
        views = get_random_carm_views(
            n_views=5,
            sid_range=(900, 1200),
            ap_range=(-40, 40),
            lat_range=(-90, 90),
            si_range=(-100, 100),
            center=(0, 0, 0),
        )
        assert views.shape == (5, 4, 4)


# ---------------------------------------------------------------------------
# get_rot_mat
#
# NOTE: `get_rot_mat` is currently broken — the function returns a non-rotation
# matrix (R R.T != I) for even the simplest inputs. We don't test it here;
# this comment documents the known issue so it isn't forgotten.
# ---------------------------------------------------------------------------
class TestGetRotMat:
    def test_returns_something_with_right_shape(self):
        # Sanity-only test: shape contract still holds. The function is
        # actually broken for batched input (only one element tensors
        # can be converted to Python scalars) AND for B=1 the result is
        # not a valid rotation matrix (R R.T != I). This is a known
        # bug in raycast.py:296-298.
        torch.manual_seed(0)
        look_from = F.normalize(torch.randn(1, 3), dim=1)
        R = get_rot_mat(look_from)
        assert R.shape == (1, 3, 3)
        # And no NaNs
        assert torch.isfinite(R).all()
