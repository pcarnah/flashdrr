import time
from timeit import default_timer as timer
from typing import Tuple, Optional

import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False

from flashdrr.rendering.raycast import VolumeRaycaster, carm_to_camera_params, get_vtk_view_mat
from flashdrr.rendering.triton_raycast import HAS_TRITON

RESOLUTIONS: Tuple[Tuple[int, int], ...] = (
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
)

RAY_SAMPLES: Tuple[int, ...] = (64, 128, 256, 384, 512)

# A small but representative sample of volume sizes — (B, C, D, H, W).
# The last entry mimics a clinical CT slab and stresses anisotropic D.
VOLUME_SIZES: Tuple[Tuple[int, int, int, int, int], ...] = (
    (1, 1, 128, 128, 128),
    (1, 1, 256, 256, 256),
    # (1, 1, 512, 512, 200),
)

WARMUP_ITERS = 3
TIMED_ITERS = 5


def run_render_nograd(vol, view_mat, ras2ijk, raycaster, triton: bool):
    with torch.no_grad():
        return raycaster(vol, view_mat=view_mat, ras2ijk=ras2ijk, triton=triton)


def run_render_train(vol, view_mat, ras2ijk, raycaster, triton: bool):
    """Forward + backward, with grad only on `vol`. `raycaster` is in train() mode."""
    raycaster.train()
    out = raycaster(vol, view_mat=view_mat, ras2ijk=ras2ijk, triton=triton)
    loss = out.float().pow(2).mean()
    loss.backward()
    # Clear grads to avoid accumulating across iterations.
    if vol.grad is not None:
        vol.grad = None
    for p in raycaster.parameters():
        if p.grad is not None:
            p.grad = None
    return out


def time_render(
        raycaster: VolumeRaycaster,
        vol: torch.Tensor,
        view_mat: torch.Tensor,
        ras2ijk: torch.Tensor,
        triton: bool,
        warmup: int = WARMUP_ITERS,
        iters: int = TIMED_ITERS,
        mode: str = "eval",
) -> Optional[Tuple[float, float]]:
    """Warm up, then time `iters` forward passes.

    ``mode="eval"`` runs the renderer under ``torch.no_grad``; ``mode="train"``
    runs forward + backward with grad only on the volume.

    Returns ``(ms_per_iter, peak_mem_mib)`` on success, or ``None`` on OOM.
    Peak memory is the maximum GPU memory allocated during the timed loop
    (in MiB); on CPU it is reported as 0.
    """
    device = vol.device
    is_cuda = device.type == "cuda"

    if mode == "eval":
        def _one():
            return run_render_nograd(vol, view_mat, ras2ijk, raycaster, triton)
    elif mode == "train":
        def _one():
            return run_render_train(vol, view_mat, ras2ijk, raycaster, triton)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    try:
        for _ in range(warmup):
            _one()
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                _one()
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / iters
            peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
            return ms, peak_mib
        else:
            t0 = timer()
            for _ in range(iters):
                _one()
            return (timer() - t0) * 1000.0 / iters, 0.0
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        print(f"  [OOM] triton={triton}: {e}")
        return None
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            print(f"  [OOM] triton={triton}: {e}")
            return None
        raise


def build_inputs(
        input_dim: Tuple[int, int, int, int, int],
        device: torch.device,
):
    """Allocate a random volume, identity affine, and a C-arm view matrix."""
    vol = torch.rand(*input_dim, device=device)
    ras2ijk = torch.eye(4, device=device, dtype=torch.float32)
    center = torch.tensor(input_dim[-3:]) // 2
    pos, focal, up = carm_to_camera_params(
        sid=1000.0, ap_angle=0.0, lat_angle=0.0, center_ras=center, table_si=0.0
    )
    view_mat = get_vtk_view_mat(pos, focal, up, device=device).unsqueeze(0)
    return vol, view_mat, ras2ijk


def print_table(rows):
    headers = ("volume (BCDHW)", "mode", "resolution", "ray_samples", "triton", "ms/iter", "peak MiB")
    widths = (28, 8, 12, 12, 8, 14, 12)

    def _fmt(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    print(_fmt(headers))
    print("-" * sum(widths + (2,) * len(widths)))
    for r in rows:
        print(_fmt(r))


def plot_results(rows, out_path: str = "benchmarks/plots/benchmark_analysis.svg") -> None:
    """Save the 2x2 seaborn benchmark figure (memory + latency across resolutions
    and across (mode, triton) at a fixed resolution)."""
    if not HAS_MPL:
        print("[plot] matplotlib not available, skipping plots.")
        return
    try:
        import pandas as pd
        import seaborn as sns
    except Exception as e:  # pragma: no cover
        print(f"[plot] pandas/seaborn unavailable ({e}), skipping plots.")
        return

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    records = []
    for vol_shape, mode, res, rs, use_triton, ms, mem in rows:
        if ms in ("OOM", "SKIP"):
            continue
        d, h, w = vol_shape[-3], vol_shape[-2], vol_shape[-1]
        vol_size = d if d == h == w else None  # only set for isotropic cubes
        records.append(
            {
                "vol_shape": vol_shape,
                "vol_size": vol_size,
                "mode": mode,
                "res_wh": res[0],  # assume square; matches the example's `res_wh`
                "ray_samples": rs,
                "triton": bool(use_triton),
                "ms_per_iter": float(ms),
                "peak_mib": float(mem),
            }
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        print("[plot] no plottable rows, skipping plots.")
        return

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 11})

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Isolate the 128^3 volume + ray_samples=256 to study resolution scaling.
    df_vol128 = df[df["vol_size"] == 128].copy()
    df_rs256 = df_vol128[df_vol128["ray_samples"] == 256].copy()
    df_plot1 = df_rs256[df_rs256["mode"] == "eval"]

    # Plot 1: memory vs resolution (eval, rs=256, 128^3)
    sns.lineplot(
        data=df_plot1, x="res_wh", y="peak_mib", hue="triton",
        marker="o", linewidth=2.5, ax=axes[0, 0],
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title(
        "Peak Memory (MiB) vs Image Resolution\n(Vol: 128^3, Ray Samples: 256, Eval)",
        fontsize=12, fontweight="bold",
    )
    axes[0, 0].set_xlabel("Resolution (Width/Height)")
    axes[0, 0].set_ylabel("Peak Memory (MiB, Log Scale)")
    axes[0, 0].set_xticks([128, 256, 512, 1024, 2048])

    # Plot 2: latency vs resolution (eval, rs=256, 128^3)
    sns.lineplot(
        data=df_plot1, x="res_wh", y="ms_per_iter", hue="triton",
        marker="s", linewidth=2.5, ax=axes[0, 1],
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title(
        "Latency (ms/iter) vs Image Resolution\n(Vol: 128^3, Ray Samples: 256, Eval)",
        fontsize=12, fontweight="bold",
    )
    axes[0, 1].set_xlabel("Resolution (Width/Height)")
    axes[0, 1].set_ylabel("ms / Iteration (Log Scale)")
    axes[0, 1].set_xticks([128, 256, 512, 1024, 2048])

    # Train vs eval at res 1024, vol 128^3.
    df_res1024 = df_vol128[df_vol128["res_wh"] == 1024].copy()
    if not df_res1024.empty:
        df_res1024["series"] = df_res1024.apply(
            lambda x: f"Triton={x['triton']}, {x['mode']}", axis=1
        )

        # Plot 3: memory across ray_samples, grouped by (triton, mode)
        sns.barplot(
            data=df_res1024, x="ray_samples", y="peak_mib", hue="series", ax=axes[1, 0],
        )
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_title(
            "Memory Scaling across Ray Samples & Modes\n(Vol: 128^3, Res: 1024x1024)",
            fontsize=12, fontweight="bold",
        )
        axes[1, 0].set_xlabel("Ray Samples")
        axes[1, 0].set_ylabel("Peak Memory (MiB, Log Scale)")

        # Plot 4: latency across ray_samples, grouped by (triton, mode)
        sns.barplot(
            data=df_res1024, x="ray_samples", y="ms_per_iter", hue="series", ax=axes[1, 1],
        )
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_title(
            "Latency across Ray Samples & Modes\n(Vol: 128^3, Res: 1024x1024)",
            fontsize=12, fontweight="bold",
        )
        axes[1, 1].set_xlabel("Ray Samples")
        axes[1, 1].set_ylabel("ms / Iteration (Log Scale)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(device)}")
    print(f"Triton available: {HAS_TRITON}")
    print(f"Warmup iters: {WARMUP_ITERS},  timed iters: {TIMED_ITERS}")
    print()

    rows = []
    for vol_shape in VOLUME_SIZES:
        # Each (vol_shape, mode) gets its own independent OOM bookkeeping
        # because memory pressure differs substantially between eval and train.
        for mode in ("eval", "train"):
            if mode == "train" and vol_shape != (1, 1, 128, 128, 128):
                continue

            print(f"== volume {vol_shape}  mode={mode} ==")
            vol, view_mat, ras2ijk = build_inputs(vol_shape, device)
            if mode == "train":
                vol.requires_grad_(True)

            oom_python: set = set()
            oom_triton: set = set()

            for res in RESOLUTIONS:
                for rs in RAY_SAMPLES:
                    raycaster = VolumeRaycaster(
                        ray_samples=rs,
                        resolution=res,
                        i0=None,
                        scatter=None,
                    ).to(device)
                    if mode == "eval":
                        raycaster.eval()

                    for use_triton in (False, True):
                        oom_set = oom_triton if use_triton else oom_python
                        if res in oom_set:
                            rows.append((vol_shape, mode, res, rs, use_triton, "OOM", "-"))
                            continue
                        if use_triton and not HAS_TRITON:
                            rows.append((vol_shape, mode, res, rs, use_triton, "SKIP", "-"))
                            continue

                        res_tuple = time_render(
                            raycaster, vol, view_mat, ras2ijk, use_triton, mode=mode
                        )
                        if res_tuple is None:
                            rows.append((vol_shape, mode, res, rs, use_triton, "OOM", "-"))
                            torch.cuda.empty_cache()
                            oom_set.add(res)
                            continue
                        ms, peak_mib = res_tuple
                        rows.append((vol_shape, mode, res, rs, use_triton, f"{ms:.2f}", f"{peak_mib:.1f}"))
                        print(f"  res={res}  rs={rs:>3}  triton={use_triton!s:>5}  {ms:7.2f} ms/iter  peak={peak_mib:8.1f} MiB")

                    del raycaster
                    torch.cuda.empty_cache()

            # Drop the grad-enabled volume before releasing the next mode.
            del vol, view_mat, ras2ijk
            torch.cuda.empty_cache()
            print()

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print_table(rows)
    plot_results(rows)


if __name__ == '__main__':
    main()