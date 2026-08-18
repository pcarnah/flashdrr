"""Shared pytest fixtures and helpers for the flashdrr test suite."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
CTCHEST_PATH = REPO_ROOT / "CTChest.nii.gz"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers so pytest -m can filter by them."""
    config.addinivalue_line("markers", "gpu: tests that require a CUDA-capable GPU")
    config.addinivalue_line("markers", "triton: tests that require the triton package")
    config.addinivalue_line(
        "markers",
        "regression: golden-output regression tests against a checked-in fixture",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip @pytest.mark.gpu tests when no GPU is available."""
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA GPU not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def has_triton() -> bool:
    """Return True if the triton package is importable (regardless of GPU)."""
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Default device for tests.

    CPU by default so the suite stays fast and runnable on machines without
    a GPU. Tests that *require* a GPU should depend on ``gpu_device``
    instead, which auto-skips if CUDA isn't available.
    """
    return torch.device("cpu")


@pytest.fixture(scope="session")
def gpu_device() -> torch.device:
    """CUDA device; skips the test if no GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def triton_available() -> bool:
    return has_triton()


# ---------------------------------------------------------------------------
# CTChest fixture
#
# The full CTChest is 1x512x512x139, which is large enough that loading it
# once per session is the friendly thing to do. We normalize intensities
# to [0, 1] the same way raycast.py's __main__ does so the regression tests
# reproduce the values that __main__ was producing.
# ---------------------------------------------------------------------------
def _load_ctchest(path: Path) -> torch.Tensor:
    """Load CTChest.nii.gz, scale to [0, 1], and return as (1, D, H, W) MetaTensor."""
    from monai.transforms import (
        Compose,
        EnsureChannelFirst,
        EnsureType,
        LoadImage,
        ScaleIntensityRange,
    )

    tf = Compose(
        [
            LoadImage(),
            EnsureChannelFirst(),
            ScaleIntensityRange(-3024, 3024, 0, 1, clip=True),
            EnsureType(),
        ]
    )
    vol = tf(str(path))  # (1, D, H, W) MetaTensor
    return vol


@pytest.fixture(scope="session")
def ctchest_volume() -> torch.Tensor:
    """CTChest loaded and intensity-normalized to [0, 1]. Cached for the session."""
    if not CTCHEST_PATH.exists():
        pytest.skip(f"CTChest fixture not found at {CTCHEST_PATH}")
    return _load_ctchest(CTCHEST_PATH)


@pytest.fixture(scope="session")
def ctchest_affine(ctchest_volume: torch.Tensor) -> torch.Tensor:
    """The (4, 4) IJK->RAS affine for the loaded CTChest volume."""
    return ctchest_volume.meta["affine"]  # type: ignore[index]


@pytest.fixture
def ctchest_mu(ctchest_volume: torch.Tensor) -> torch.Tensor:
    """Reproduce the __main__ mu conversion (clamp(0.05 * (1 + hu/800))).

    `ctchest_volume` is in [0, 1]; the original __main__ does
    ``hu = vol * (3024 - (-3524)) + (-3524)`` then
    ``mu = clamp(0.05 * (1 + hu / 800), min=0)``. We do the same here so
    regression tests exercise the same physical values the dev script did.
    """
    hu = ctchest_volume * (3024.0 - (-3524.0)) + (-3524.0)
    return torch.clamp(0.05 * (1.0 + hu / 800.0), min=0.0)


# ---------------------------------------------------------------------------
# Random synthetic volumes
# ---------------------------------------------------------------------------
@pytest.fixture
def small_random_volume() -> torch.Tensor:
    """A small (1, 1, 16, 16, 16) random volume for fast unit tests."""
    torch.manual_seed(0)
    return torch.rand(1, 1, 16, 16, 16, dtype=torch.float32)


@pytest.fixture
def small_random_volume_8ch() -> torch.Tensor:
    """An 8-channel small random volume; matches the __main__ config (C=8)."""
    torch.manual_seed(0)
    return torch.rand(1, 8, 16, 16, 16, dtype=torch.float32)


@pytest.fixture
def identity_ras2ijk() -> torch.Tensor:
    """A RAS->IJK affine that is a pure 1mm isotropic identity (no rotation).

    Useful when the test only cares about the rendering math, not real
    medical-imaging geometry.
    """
    M = torch.eye(4, dtype=torch.float32)
    # Center the volume at the origin: IJK=0 maps to RAS=-(shape-1)/2
    M[0, 3] = -(16 - 1) / 2.0
    M[1, 3] = -(16 - 1) / 2.0
    M[2, 3] = -(16 - 1) / 2.0
    return torch.inverse(M)


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_view_mat(device: torch.device) -> torch.Tensor:
    """A single camera looking at the origin from a few units back.

    Returns a ``(1, 4, 4)`` tensor as the renderer expects a batched view
    matrix. The orthonormal basis is the one produced by
    ``get_vtk_view_mat``: right, up, forward, camera position.
    """
    cam_pos = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float32)
    cam_focal = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    cam_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    forward = torch.nn.functional.normalize(cam_focal - cam_pos, dim=0)
    right = torch.nn.functional.normalize(torch.linalg.cross(forward, cam_up), dim=0)
    up = torch.linalg.cross(right, forward)
    M = torch.eye(4)
    M[:3, 0] = right
    M[:3, 1] = up
    M[:3, 2] = forward
    M[:3, 3] = cam_pos
    return M.to(device).unsqueeze(0)


@pytest.fixture
def batch_view_mat(device: torch.device) -> torch.Tensor:
    """A batch of two cameras with slightly different positions."""
    from flashdrr.rendering import get_vtk_view_mat

    cam_pos_a = (0.0, 0.0, 5.0)
    cam_pos_b = (0.0, 0.0, 6.0)
    v1 = get_vtk_view_mat(cam_pos_a, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)).to(device)
    v2 = get_vtk_view_mat(cam_pos_b, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)).to(device)
    return torch.stack([v1, v2], dim=0)
