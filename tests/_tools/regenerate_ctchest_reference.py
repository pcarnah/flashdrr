"""Regenerate the CTChest regression golden fixtures.

Run this when you intentionally change the rendering pipeline and need
to update the checked-in references. The script writes:

  * tests/fixtures/ctchest_drr_stats.json
  * tests/fixtures/ctchest_drr_thumbnail.npy
  * tests/fixtures/ctchest_drr_triton_stats.json   (GPU only)
  * tests/fixtures/ctchest_drr_triton_thumbnail.npy (GPU only)

Usage:
    python tests/_tools/regenerate_ctchest_reference.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make the flashdrr package importable when run as a script
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from monai.transforms import (  # noqa: E402
    Compose,
    EnsureChannelFirst,
    EnsureType,
    LoadImage,
    ScaleIntensityRange,
)

from flashdrr.rendering import VolumeRaycaster, get_vtk_view_mat  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
CTCHEST = ROOT / "CTChest.nii.gz"


def render(device, triton: bool = False):
    load_tf = Compose(
        [
            LoadImage(),
            EnsureChannelFirst(),
            ScaleIntensityRange(-3024, 3024, 0, 1, clip=True),
            EnsureType(),
        ]
    )
    vol = load_tf(str(CTCHEST)).unsqueeze(0)  # (1, 1, D, H, W)
    hu = vol * (3024.0 - (-3524.0)) + (-3524.0)
    mu = torch.clamp(0.05 * (1.0 + hu / 800.0), min=0.0)
    ijk2ras = vol.meta["affine"]
    ras2ijk = torch.inverse(ijk2ras)

    center = torch.ones(4, dtype=torch.float64)
    center[:3] = torch.as_tensor(vol.shape[2:]) // 2
    center = ijk2ras @ center

    view_mat = get_vtk_view_mat(
        (0.0, 1000.0, -130.0), center[:3], (0.0, 0.0, 1.0)
    ).unsqueeze(0)

    ren = VolumeRaycaster(
        scatter=None, resolution=(128, 128), i0=None, ray_samples=128
    ).to(device).eval()

    with torch.no_grad():
        out = ren(
            mu.to(device),
            view_mat=view_mat.to(device),
            ras2ijk=ras2ijk.to(device),
            triton=triton,
        )
    return out


def save(out: torch.Tensor, stats_path: Path, thumb_path: Path) -> None:
    arr = out.cpu().numpy()
    stats = {
        "shape": list(out.shape),
        "min": float(out.min()),
        "max": float(out.max()),
        "mean": float(out.mean()),
        "std": float(out.std()),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote {stats_path}")
    np.save(thumb_path, np.round(arr[0, 0], 4))
    print(f"wrote {thumb_path}")


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    print("Rendering with Python-loop renderer (CPU)...")
    out_py = render("cpu", triton=False)
    save(
        out_py,
        FIXTURES / "ctchest_drr_stats.json",
        FIXTURES / "ctchest_drr_thumbnail.npy",
    )
    if torch.cuda.is_available():
        print("Rendering with Triton renderer (GPU)...")
        out_triton = render("cuda", triton=True)
        save(
            out_triton,
            FIXTURES / "ctchest_drr_triton_stats.json",
            FIXTURES / "ctchest_drr_triton_thumbnail.npy",
        )
    else:
        print("CUDA not available, skipping triton reference.")


if __name__ == "__main__":
    main()
