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
    get_random_carm_views,
    get_vtk_view_mat,
)


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

