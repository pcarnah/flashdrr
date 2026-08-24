# Initially adapted from torchvtk (https://github.com/torchvtk/torchvtk/). Rendering engine completely re-written since.

"""
Differentiable volume rendering for 2D X-ray projection (DRR) synthesis.

This module provides:

* Camera utilities for sampling and building C-arm view matrices in RAS
  space (:func:`get_random_carm_views`, :func:`carm_to_camera_params`,
  :func:`get_vtk_view_mat`).
* A per-channel piecewise-linear transfer function
  (:func:`piecewise_linear_channelwise`).
* Building blocks :class:`ASPP` and :class:`DepthAwareScatter` (prototype).
* :class:`VolumeRaycaster`, the main differentiable raycaster that ties
  these together. It dispatches between a tiled Python-loop backend and a
  fused CUDA/Triton backend (:class:`flashdrr.rendering.FusedVolumeRenderer`).
"""

import math
from timeit import default_timer as timer
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn

from flashdrr.rendering.triton_raycast import FusedVolumeRenderer
from flashdrr.utils import make_2d

__all__ = ['get_vtk_view_mat', 'get_random_carm_views', 'carm_to_camera_params',
           'VolumeRaycaster']

def get_random_carm_views(n_views, sid_range, ap_range, lat_range, si_range, center):
    """
    Randomly sample C-arm view matrices by uniform sampling of the underlying
    geometric parameters.

    For each of ``n_views`` samples, source-to-image distance (SID), AP angle,
    lateral angle, and table SI translation are drawn uniformly from the
    corresponding ranges, converted to a camera (position, focal, up) tuple
    via :func:`carm_to_camera_params`, and assembled into a 4x4 camera-to-
    world view matrix via :func:`get_vtk_view_mat`. The matrices are stacked
    along a new leading dimension.

    Parameters:
    -----------
    n_views : int
        Number of view matrices to sample.
    sid_range : tuple of 2 floats
        (min_sid, max_sid) in mm. Typical range: (900, 1200)
    ap_range : tuple of 2 floats
        (min_ap, max_ap) in degrees. Typical range: (-40, 40) for cranial/caudal
    lat_range : tuple of 2 floats
        (min_lat, max_lat) in degrees. Typical range: (-90, 90) for LAO/RAO
    si_range : tuple of 2 floats
        (min_si, max_si) in mm. Typical range: (-100, 100) for table translation
    center : tuple or list of 3 floats
        Volume center in RAS coordinates (mm), passed to
        :func:`carm_to_camera_params`.

    Returns:
    --------
    views : torch.Tensor (n_views, 4, 4)
        Stacked camera-to-world view matrices in RAS space, on CPU.

    Notes:
    ------
    Distribution choice rationale:

    * SID: uniform is appropriate — mechanical constraint, no preferred distance.
    * AP angle: uniform is reasonable for training, though clinical use shows
      bias toward AP (0°), cranial (15-30°), and steep cranial (>30°).
    * Lateral angle: uniform works, but clinical practice favors AP (0°),
      RAO 30°, and LAO 30-45° views.
    * Table SI: uniform is appropriate — depends on anatomy region of interest.

    For more realistic clinical distributions, consider adding bias toward
    common views (0°, ±30°, ±45°) or using a mixture of uniforms/peaked
    distributions.
    """
    import random

    views = []
    for _ in range(n_views):
        sid = random.uniform(sid_range[0], sid_range[1])
        ap_angle = random.uniform(ap_range[0], ap_range[1])
        lat_angle = random.uniform(lat_range[0], lat_range[1])
        table_si = random.uniform(si_range[0], si_range[1])

        pos, focal, up = carm_to_camera_params(sid, ap_angle, lat_angle, center, table_si)
        views.append(get_vtk_view_mat(pos, focal, up))

    views = torch.stack(views)

    return views


def carm_to_camera_params(sid, ap_angle, lat_angle, center_ras, table_si=0.0):
    """
    Convert C-arm position parameters to camera parameters.

    Parameters:
    -----------
    sid : float
        Source-to-Image Distance in mm (distance from X-ray source to detector)
    ap_angle : float
        Anteroposterior (AP) angle in degrees
        - 0° = AP view (source in front, looking posterior)
        - Positive = caudal angulation (source tilts inferiorly)
        - Negative = cranial angulation (source tilts superiorly)
    lat_angle : float
        Lateral angle in degrees
        - 0° = AP view
        - Positive = RAO (source moves to patient's right)
        - Negative = LAO (source moves to patient's left)
    center_ras : tuple or list of 3 floats
        Center point of the CT scan in RAS coordinates (mm)
        RAS = Right, Anterior, Superior
    table_si : float, optional
        Table translation in superior-inferior direction in mm (default: 0.0)
        - Positive = table moves superior (head up)
        - Negative = table moves inferior (head down)

    Returns:
    --------
    cam_pos : numpy array (3,)
        Camera/source position in RAS coordinates
    look_at : numpy array (3,)
        Look-at point (focal point) in RAS coordinates
    look_up : numpy array (3,)
        Up vector for camera orientation
    """

    # Convert angles to radians
    ap_rad = np.deg2rad(ap_angle)
    lat_rad = np.deg2rad(lat_angle)

    # Center point with table translation
    # Table moves in superior-inferior direction (Z-axis in RAS)
    center = np.array(center_ras) + np.array([0, 0, table_si])

    # Initial source position at SID distance along negative Y axis (anterior)
    source_local = np.array([0, -sid, 0])

    # Rotation around Z-axis (superior) for lateral angulation
    R_lat = np.array([
        [np.cos(lat_rad), -np.sin(lat_rad), 0],
        [np.sin(lat_rad), np.cos(lat_rad), 0],
        [0, 0, 1]
    ])

    # Rotation around X-axis (right) for cranial/caudal angulation
    R_ap = np.array([
        [1, 0, 0],
        [0, np.cos(ap_rad), -np.sin(ap_rad)],
        [0, np.sin(ap_rad), np.cos(ap_rad)]
    ])

    # Apply rotations: first lateral, then AP
    R_total = R_ap @ R_lat
    source_rotated = R_total @ source_local

    # Camera position in world coordinates
    cam_pos = source_rotated + center

    # Look-at point is the center
    look_at = center

    # Up vector: start with superior direction and apply same rotations
    up_local = np.array([0, 0, 1])
    look_up = R_total @ up_local

    return cam_pos, look_at, look_up


def get_vtk_view_mat(cam_pos: Tuple[float],  # (3,) camera center in RAS
                     cam_focal: Tuple[float],  # (3,) camera focal point in RAS
                     cam_viewup: Tuple[float],  # (3,) view-up vector in RAS)
                     device: str = 'cpu'):
    """
    Build a VTK-compatible camera-to-world view matrix (4x4) from camera position,
    focal point, and view-up direction in RAS coordinates.

    The resulting matrix maps camera-space points to world (RAS) space. The
    top-left 3x3 block has the camera axes as its columns, ordered
    ``[right, up, forward]``; the fourth column is the camera position
    ``cam_pos``; the bottom row is ``[0, 0, 0, 1]``. ``forward`` points
    from ``cam_pos`` toward ``cam_focal``.

    Args:
        cam_pos: Camera/source position in RAS (mm), shape (3,).
        cam_focal: Camera focal (look-at) point in RAS (mm), shape (3,).
        cam_viewup: World-space up direction in RAS, shape (3,). Need not be unit length.
        device: Device on which to allocate the returned tensor. Defaults to ``'cpu'``.

    Returns:
        torch.Tensor: (4, 4) camera-to-world view matrix as described above.
    """
    cam_pos = torch.as_tensor(cam_pos, dtype=torch.float32)
    cam_focal = torch.as_tensor(cam_focal, dtype=torch.float32)
    cam_viewup = torch.as_tensor(cam_viewup, dtype=torch.float32)

    # Construct VTK-compatible camera axes
    forward = torch.nn.functional.normalize(cam_focal - cam_pos, dim=0)  # +Z axis (direction of projection in VTK)
    right = torch.nn.functional.normalize(torch.linalg.cross(forward, cam_viewup), dim=0)
    up = torch.linalg.cross(right, forward)  # True up direction

    # View matrix (camera-to-world)
    view_mat = torch.zeros(4, 4, device=device)
    view_mat[:3, 0] = right
    view_mat[:3, 1] = up
    view_mat[:3, 2] = forward
    view_mat[:3, 3] = cam_pos
    view_mat[3, 3] = 1.0

    return view_mat


def piecewise_linear_channelwise(x, xp, yp):
    """
    Apply a per-channel piecewise linear function to input x.
    Handles extrapolation beyond xp bounds by extending the first/last segment.

    Args:
        x (Tensor): Input (B, C, H, W) or (B, C, D, H, W)
        xp (Tensor): (C, K) sorted x keypoints per channel
        yp (Tensor): (C, K) y values per channel at keypoints

    Returns:
        Tensor: Output of same shape as x
    """
    if xp.ndim != 2 or yp.ndim != 2 or xp.shape != yp.shape:
        print(xp.shape, yp.shape)
        raise ValueError("xp and yp must have shape (C, K)")
    B, C = x.shape[:2]
    x_flat = x.view(B, C, -1)  # (B, C, N)

    K = xp.shape[1]
    xp = xp.unsqueeze(0).expand(B, -1, -1)  # (B, C, K)
    yp = yp.unsqueeze(0).expand(B, -1, -1)

    x_unsq = x_flat.unsqueeze(-1)  # (B, C, N, 1)
    xp_left = xp.unsqueeze(2)[:, :, :, :-1]  # (B, C, 1, K-1)
    xp_right = xp.unsqueeze(2)[:, :, :, 1:]

    # Mask to identify correct segment
    mask = (x_unsq >= xp_left) & (x_unsq < xp_right)

    # If no valid segment (i.e. x >= xp[-1]), assign to last segment
    none_selected = ~mask.any(dim=-1)
    idx = mask.float().argmax(dim=-1)  # (B, C, N)
    idx[none_selected] = K - 2  # assign last interval

    # Safe gather
    gather_idx = idx.unsqueeze(-1)
    xp_l = torch.gather(xp, 2, idx)
    xp_r = torch.gather(xp, 2, idx + 1)
    yp_l = torch.gather(yp, 2, idx)
    yp_r = torch.gather(yp, 2, idx + 1)

    # Linear interpolation
    denom = xp_r - xp_l
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)  # avoid div0
    xval = x_flat
    yval = yp_l + (yp_r - yp_l) * ((xval - xp_l) / denom)

    return yval.view_as(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module.

    Applies parallel 2D atrous (dilated) convolutions with multiple dilation
    rates, concatenates their ReLU-activated outputs, and projects the
    concatenated feature map back to ``out_ch`` channels with a 1x1 conv.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels for each branch and the final projection.
        rates: Tuple of dilation rates for the parallel ``Conv2d`` branches.
            Each branch uses a 3x3 kernel with padding equal to its dilation.
    """

    def __init__(self, in_ch, out_ch, rates=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r) for r in rates
        ])
        self.proj = nn.Conv2d(out_ch * len(rates), out_ch, 1)

        # init
        for m in self.branches:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        """
        Run the ASPP branches and project to ``out_ch`` channels.

        Args:
            x: Input feature map of shape ``(B, in_ch, H, W)``.

        Returns:
            torch.Tensor: Output feature map of shape ``(B, out_ch, H, W)``.
        """
        feats = [F.relu(branch(x)) for branch in self.branches]
        return self.proj(torch.cat(feats, dim=1))


class DepthAwareScatter(nn.Module):
    """Prototype depth-aware scatter estimator for DRR rendering.

    .. warning::
        **Prototype, not fully functional.** This module is an experimental
        design and is not yet validated end-to-end. The scattering model,
        gating, and integration with the main raycaster are subject to
        change. Do not rely on its outputs for production use.

    Given per-sample attenuation ``mu`` along each ray, it computes a
    Beer-Lambert primary transmission, accumulates a depth-dependent scatter
    source, mixes it with an ASPP-conditioned gate, and adds a bounded
    scatter contribution to the primary image.

    Args:
        in_ch: Number of scatter channels (``C``) in the input volume.
        base_ch: Hidden channel width of the conditioning/ASPP branch.
        alpha_max: Upper bound on the learned scatter gating weight applied
            to the scatter map before it is added to the primary image.
    """

    def __init__(self, in_ch, base_ch=32, alpha_max=0.4):
        super().__init__()
        self.alpha_max = alpha_max

        # depth mixing: causal 1D conv over D
        self.depth_conv = nn.Conv1d(in_ch, in_ch, kernel_size=5, padding=4, dilation=2)
        nn.init.kaiming_normal_(self.depth_conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.depth_conv.bias)

        # conditioning features → ASPP
        self.enc = nn.Conv2d(in_ch * 2, base_ch, 1)
        self.aspp = ASPP(base_ch, base_ch)
        self.alpha_head = nn.Conv2d(base_ch, 1, 1)
        nn.init.kaiming_normal_(self.alpha_head.weight, nonlinearity="relu")
        nn.init.constant_(self.alpha_head.bias, -3.0)  # α ≈ 0 at init

        # lateral scatter mixing
        self.scatter_conv = nn.Conv2d(in_ch + base_ch, 1, 3, padding=1)
        nn.init.kaiming_normal_(self.scatter_conv.weight, nonlinearity="relu")  # no scatter initially
        nn.init.zeros_(self.scatter_conv.bias)

    def forward(self, mu, dz=1.0):
        """
        Estimate primary and scatter contributions along rays.

        Args:
            mu: Per-sample attenuation, shape ``(B, C, D, H, W)`` where ``D`` is
                the number of samples along each ray.
            dz: Scalar sample spacing along the ray (same units as volume
                voxel sizes, e.g. mm). Defaults to ``1.0``.

        Returns:
            Tuple of:

            * **I_out** (``(B, C, H, W)``): Primary transmission plus the
              bounded, gated scatter map.
            * **I_primary** (``(B, C, H, W)``): Beer-Lambert primary
              transmission at the exit of the ray (pre-scatter).
            * **scatter_map** (``(B, 1, H, W)``): Learned per-pixel scatter
              contribution before the alpha gate.
            * **alpha** (``(B, 1, H, W)``): Per-pixel gating weight in
              ``[0, alpha_max]`` applied to ``scatter_map``.
        """
        # mu: [B,C,D,H,W], I0 = 1
        B, C, D, H, W = mu.shape

        # integrate attenuation
        tau_z = torch.cumsum(mu * dz, dim=2)  # [B,C,D,H,W]
        I_z = torch.exp(-tau_z)  # [B,C,D,H,W]
        I_primary = I_z[:, :, -1]  # [B,C,H,W]

        # scatter source per depth
        S_z = mu * dz * I_z  # [B,C,D,H,W]

        # depth-aware weighting (causal conv along D)
        # reshape to [B*H*W, C, D]
        S_z_perm = S_z.permute(0, 3, 4, 1, 2).contiguous()  # [B,H,W,C,D]
        S_z_flat = S_z_perm.view(-1, C, D)  # [B*H*W, C, D]
        S_w = self.depth_conv(S_z_flat)  # [B*H*W, C, D]
        S_w = S_w.view(B, H, W, C, D).permute(0, 3, 4, 1, 2)  # [B,C,D,H,W]
        S = S_w.sum(dim=2)  # [B,C,H,W]

        # conditioning features
        tau_exit = tau_z[:, :, -1]  # [B,C,H,W]
        feats = torch.cat([I_primary, tau_exit], dim=1)  # [B,2C,H,W]
        F0 = F.relu(self.enc(feats))  # [B,base,H,W]

        # ASPP context
        F_aspp = self.aspp(F0)  # [B,base,H,W]

        # scatter map
        scatter_in = torch.cat([S, F_aspp], dim=1)  # [B,C+base,H,W]
        scatter_map = F.relu(self.scatter_conv(scatter_in))  # [B,1,H,W]

        # alpha gate
        alpha = torch.sigmoid(self.alpha_head(F_aspp)) * self.alpha_max

        # output
        I_out = I_primary + alpha * scatter_map
        return I_out, I_primary, scatter_map, alpha


# %%
class VolumeRaycaster(nn.Module):
    """
    Differentiable C-arm volume raycaster for digitally reconstructed
    radiographs (DRRs).

    Given a 3D attenuation volume in IJK space and one or more camera view
    matrices in RAS space, ``VolumeRaycaster`` samples along each camera
    ray, integrates the attenuation using either Beer-Lambert or alpha
    compositing, and produces a 2D projection per view. Rendering is
    differentiable with respect to the input volume and view matrices.

    Two backends are supported:

    * A tiled Python-loop renderer using ``torch.grid_sample`` (default).
    * A fused CUDA/Triton renderer (``flashdrr.rendering.FusedVolumeRenderer``),
      selected per-call via ``forward(..., triton=True)``.

    A lightweight, prototype depth-aware scatter model
    (:class:`DepthAwareScatter`) can be attached for the last
    ``scatter`` channels of the volume.
    """

    def __init__(
            self,
            density_factor: float = 100.0,
            ray_samples: int = 256,
            resolution: Tuple[int, int] = (224, 224),
            use_checkpointing: bool = False,
            use_beer_lambert: bool = True,
            scatter: Optional[int] = None,  # or int | None if Python 3.10+
            i0: Optional[float] = None,
            fov: Optional[float] = 20.0
    ) -> None:
        """
        Initialize the raycaster.

        Args:
            density_factor: Scalar multiplier applied to volume intensities
                before integration. Used by the alpha-compositing branch of
                the Python-loop backend and passed to the Triton backend.
            ray_samples: Number of samples taken along each camera ray.
            resolution: Output image size as ``(width, height)`` or a single
                int for a square image.
            use_checkpointing: If True, use ``torch.utils.checkpoint`` to
                trade compute for memory in the Python-loop backend (training
                only).
            use_beer_lambert: If True, integrate via the Beer-Lambert law
                and return ``1 - exp(-int(mu * dz))``. If False, use
                front-to-back alpha compositing (over operator).
            scatter: If not ``None``, treat the last ``scatter`` channels of
                the volume as scatter channels and process them with
                :class:`DepthAwareScatter` (prototype; not fully functional);
                the remaining channels are rendered with Beer-Lambert as
                primary. Only has an effect when ``use_beer_lambert`` is
                True.
            i0: Mean photon count at the source for the Gaussian-Poisson
                surrogate noise model. If ``None``, no noise is added.
            fov: Vertical field of view in degrees used to build the per-pixel
                ray directions.
        """
        super().__init__()
        self.density_factor = density_factor
        self.ray_samples = ray_samples
        self.use_checkpointing = use_checkpointing
        self.use_beer_lambert = use_beer_lambert
        self.scatter = None
        self.i0 = i0
        if isinstance(resolution, tuple):
            self.w, self.h = resolution
        else:
            self.w, self.h = resolution, resolution

        # Z = torch.linspace(-1, 1, ray_samples)
        # W = torch.linspace(-1, 1, self.w)
        # H = torch.linspace(-1, 1, self.h)
        # self.samples = self.get_coord_grid(Z, H, W, perspective=True)
        self.register_buffer('dirs_cam', torch.empty(self.h, self.w, 3), persistent=False)
        self.triton_raycaster = FusedVolumeRenderer(self.ray_samples, self.density_factor, self.apply_poisson)

        if scatter is not None:
            self.scatter_channels = scatter
            self.scatter = DepthAwareScatter(scatter)

        self.set_fov(fov)

    def set_fov(self, fov: float) -> None:
        """
        Configure the camera's vertical field of view and refresh the
        per-pixel ray direction buffer ``self.dirs_cam``.

        The buffer is built on CPU in the image plane, normalized, then
        moved to the module's current device.

        Args:
            fov: Vertical field of view in degrees.
        """
        # Get the current device from the buffer
        device = self.dirs_cam.device

        # Do computation on CPU
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, self.h),  # vertical: -1 (bottom) to 1 (top)
            torch.linspace(-1, 1, self.w),  # horizontal: -1 (left) to 1 (right)
            indexing='ij'
        )

        # Vertical FOV in radians
        fov_y = np.deg2rad(fov)
        aspect = self.w / self.h
        # Compute direction vectors in camera coordinates
        px = x * np.tan(fov_y / 2) * aspect
        py = y * np.tan(fov_y / 2)
        pz = torch.ones_like(px)
        dirs_cam = torch.stack([px, -py, pz], dim=-1)  # shape: (H, W, 3)

        # Normalize directions and move to device once at the end
        dirs_cam = F.normalize(dirs_cam, dim=-1).to(device)
        self.dirs_cam = dirs_cam

    @staticmethod
    def compute_clipping_distances(camera_matrix, volume_shape, ijk2ras, margin=30.0):
        """
        Compute near and far clipping distances for ray sampling based on camera matrix
        and volume bounds in IJK space.

        Parameters:
        -----------
        camera_matrix : torch.Tensor (B, 4, 4) or (4, 4)
            Batched or single 4x4 camera view matrix in RAS space
            Can be world-to-camera or camera-to-world (will be auto-detected)
        volume_shape : tuple of 3 ints
            Volume dimensions in IJK space (I, J, K)
        ijk2ras : torch.Tensor (4, 4)
            Transformation matrix from IJK to RAS coordinates
        margin : float, optional
            Safety margin to add to near/far distances in mm (default: 30.0).
            Positive values expand the clipping range.

        Returns:
        --------
        near : torch.Tensor (B,) or float
            Near clipping distance in mm (distance from camera to closest volume point)
        far : torch.Tensor (B,) or float
            Far clipping distance in mm (distance from camera to farthest volume point)
        """

        if not isinstance(camera_matrix, torch.Tensor):
            camera_matrix = torch.tensor(camera_matrix, dtype=torch.float32)
        if not isinstance(ijk2ras, torch.Tensor):
            ijk2ras = torch.tensor(ijk2ras, dtype=torch.float32)

        # Handle both batched and single camera matrices
        is_batched = camera_matrix.ndim == 3
        if not is_batched:
            camera_matrix = camera_matrix.unsqueeze(0)

        batch_size = camera_matrix.shape[0]
        device = camera_matrix.device

        ijk2ras = ijk2ras.to(dtype=camera_matrix.dtype, device=device)

        # Extract camera position from matrix
        # Check if this is camera-to-world (large translation) or world-to-camera (small translation)
        test_pos = camera_matrix[:, :3, 3]  # (B, 3)
        translation_norm = torch.norm(test_pos, dim=1)  # (B,)

        # For matrices with small translation, assume world-to-camera and invert
        needs_invert = translation_norm < 1.0

        cam_pos_ras = torch.zeros(batch_size, 3, device=device)
        view_dir = torch.zeros(batch_size, 3, device=device)

        for i in range(batch_size):
            if needs_invert[i]:
                cam_to_world = torch.inverse(camera_matrix[i])
                cam_pos_ras[i] = cam_to_world[:3, 3]
                view_dir[i] = cam_to_world[:3, 2]
            else:
                cam_pos_ras[i] = camera_matrix[i, :3, 3]
                view_dir[i] = camera_matrix[i, :3, 2]

        # Normalize view directions
        view_dir = view_dir / torch.norm(view_dir, dim=1, keepdim=True)

        # Generate all 8 corners of the volume bounding box in IJK space
        I, J, K = volume_shape
        corners_ijk = torch.tensor([
            [0, 0, 0],
            [I - 1, 0, 0],
            [0, J - 1, 0],
            [I - 1, J - 1, 0],
            [0, 0, K - 1],
            [I - 1, 0, K - 1],
            [0, J - 1, K - 1],
            [I - 1, J - 1, K - 1]
        ], dtype=camera_matrix.dtype, device=device)  # (8, 3)

        # Transform corners to RAS space
        corners_homog = torch.cat([corners_ijk, torch.ones(8, 1, device=device)], dim=1)  # (8, 4)
        corners_ras = (ijk2ras @ corners_homog.T).T[:, :3]  # (8, 3)

        # Compute distances from camera to each corner projected along view direction
        # Broadcast: cam_pos_ras (B, 1, 3), corners_ras (1, 8, 3)
        to_corners = corners_ras.unsqueeze(0) - cam_pos_ras.unsqueeze(1)  # (B, 8, 3)
        distances = torch.sum(to_corners * view_dir.unsqueeze(1), dim=2)  # (B, 8)

        # Near is the minimum distance, far is the maximum distance
        near = torch.min(distances, dim=1).values - margin  # (B,)
        far = torch.max(distances, dim=1).values + margin  # (B,)

        # Ensure near is positive
        near = torch.clamp(near, min=0.1)

        # Return scalar if input was not batched
        if not is_batched:
            return near.item(), far.item()

        return near, far

    def apply_poisson(self, transmission: torch.Tensor) -> torch.Tensor:
        """
        Add a differentiable Gaussian approximation of Poisson noise to a
        transmission map.

        If ``self.i0`` is ``None``, the input is returned unchanged (no
        noise is added). Otherwise, Gaussian noise with standard deviation
        ``sqrt(transmission / i0)`` is added and the result is clamped to
        ``[0, 1]``. This is a smooth surrogate for shot/Poisson noise and is
        fully differentiable.

        Args:
            transmission: Transmission values (post-primary, pre-noise) in
                ``[0, 1]``. Any shape is supported.

        Returns:
            torch.Tensor: Noisy transmission, same shape as ``transmission``,
            clamped to ``[0, 1]``. Returns the input unchanged when ``i0`` is
            ``None``.
        """
        if self.i0 is None:
            return transmission

        # Differentiable Gaussian approximation of Poisson
        transmission = torch.clamp(transmission, min=1e-8)
        std = torch.sqrt(transmission / self.i0)
        epsilon = torch.randn_like(transmission)
        noisy_transmission = transmission + std * epsilon
        return torch.clamp(noisy_transmission, 0, 1)

    def forward(self, vol: torch.Tensor, view_mat: torch.Tensor, ras2ijk: torch.Tensor,
                tile_h: int = 256, tile_w: int = 256, triton: bool = False) -> torch.Tensor:
        """
        Render one or more views of a (batched) 3D attenuation volume.

        Args:
            vol: Attenuation volume of shape ``(B, C, D, H, W)`` in IJK
                voxel space. The affine ``ras2ijk`` is what makes the volume
                RAS-aligned at runtime.
            view_mat: Camera-to-world view matrices in RAS, shape
                ``(B, 4, 4)`` or ``(N, 4, 4)`` where ``N = B * views_per_vol``.
                As a special case, ``N == 1`` is broadcast to ``B``. The
                clipping-distance computation also accepts world-to-camera
                matrices and auto-inverts them.
            ras2ijk: 4x4 affine mapping RAS to IJK coordinates.
            tile_h: Tile height used by the Python-loop backend to bound
                peak memory. Ignored if the image is smaller than
                ``(2*tile_h, 2*tile_w)``.
            tile_w: Tile width used by the Python-loop backend. See
                ``tile_h``.
            triton: If True, use the fused CUDA/Triton renderer
                (:class:`flashdrr.rendering.FusedVolumeRenderer`); otherwise
                use the tiled Python-loop renderer.

        Returns:
            torch.Tensor: Rendered projections of shape ``(N, C, H, W)``.
        """
        with torch.autocast(device_type='cuda', enabled=False):

            bs = vol.shape[0]
            N = view_mat.shape[0]

            if N < bs:
                raise ValueError(f"view_mat has {N} entries but batch size is {bs}")
            if N > bs and N % bs != 0:
                raise ValueError(f"N ({N}) must be a multiple of batch size ({bs})")

            # Expand density to match number of views
            views_per_vol = N // bs
            density = vol.permute(0, 1, 4, 3, 2)  # (B, C, W, H, D)
            if views_per_vol > 1:
                density = density.repeat_interleave(views_per_vol, dim=0)  # (N, C, W, H, D)

            # N==1 broadcast: expand view_mat to bs
            if N == 1:
                view_mat = view_mat.expand(bs, -1, -1)

            near, far = self.compute_clipping_distances(view_mat, vol.shape[2:], torch.inverse(ras2ijk))

            use_tiling = self.h >= 2 * tile_h or self.w >= 2 * tile_w
            _tile_h = tile_h if use_tiling else None
            _tile_w = tile_w if use_tiling else None

            vol_shape = torch.as_tensor(vol.shape[2:], device=vol.device, dtype=torch.float32)
            if triton:
                render_fn = lambda d, v, n, f: self.triton_raycaster(d, v, n, f, ras2ijk, vol_shape, self.dirs_cam)
            else:
                render_fn = torch.vmap(
                    lambda d, v, n, f: self._render_single(d, v, n, f, ras2ijk, vol_shape, _tile_h, _tile_w),
                    in_dims=(0, 0, 0, 0),
                    randomness='different',  # required if apply_poisson uses torch.randn_like
                )

            return render_fn(density, view_mat, near, far)  # (N, C, H, W)

    def _render_single(
            self,
            density: torch.Tensor,  # (C, W, H, D) — single volume, vmapped over batch
            view_single: torch.Tensor,  # (4, 4)
            near: torch.Tensor,  # scalar
            far: torch.Tensor,  # scalar
            ras2ijk: torch.Tensor,  # (4, 4) shared constant
            vol_shape: torch.Tensor,  # (3,) shared constant
            tile_h: int,
            tile_w: int,
    ) -> torch.Tensor:
        """Render a single (volume, view) pair. Vmapped over batch dim.
        Returns (C, H, W).
        """
        # density = density.contiguous()

        step_size = 0.1 * (far - near) / self.ray_samples

        # --- Ray generation (no loop, no B dim) ---
        cam2world = view_single[:3, :3]
        cam_pos = view_single[:3, 3]

        device = view_single.device if isinstance(view_single, torch.Tensor) else "cpu"
        ras2ijk = ras2ijk.to(device, dtype=torch.float32)

        dirs_world = torch.einsum('ij,hwj->hwi', cam2world, self.dirs_cam)  # (H, W, 3)

        R = ras2ijk[:3, :3]
        t = ras2ijk[:3, 3]

        scale = 2.0 / (vol_shape - 1)  # (3,)

        dirs_ijk = dirs_world @ R.T  # (H, W, 3) @ (3, 3) — small, not (D,H,W,3)
        origin_ijk = cam_pos @ R.T + t  # (3,) — scalar op

        depths = near + (far - near) * torch.linspace(0, 1, self.ray_samples, device=density.device)  # (D,)

        ijk = scale * (origin_ijk + dirs_ijk.unsqueeze(0) * depths[:, None, None, None]) - 1

        # (D, H, W, 3) — no loop, fits in memory per-item
        # samples = cam_pos + dirs_world.unsqueeze(0) * depths[:, None, None, None]
        # ijk = ((2 * (samples @ R.T + t)) / (vol_shape - 1)) - 1  # (D, H, W, 3)

        # grid_sample expects (C, W, H, D) input and (1, D, H, W, 3) grid
        density_b = density.unsqueeze(0)  # (1, C, W, H, D)

        def render_tile(coords_tile):
            # coords_tile: (1, D, h, w, 3)
            dens_tile = F.grid_sample(density_b, coords_tile, align_corners=False)  # (1, C, D, h, w)

            if self.use_beer_lambert:
                if self.scatter:
                    I_out_no_scatter = 1.0 - self.apply_poisson(
                        torch.exp(-torch.sum(dens_tile[:, :-self.scatter_channels] * step_size, dim=2))
                    )
                    I_out, _, _, _ = self.scatter(dens_tile[:, -self.scatter_channels:], step_size)
                    return torch.cat([I_out_no_scatter, 1.0 - I_out], dim=1).squeeze(0)  # (C, h, w)
                else:
                    return (1.0 - self.apply_poisson(
                        torch.exp(-torch.sum(dens_tile * step_size, dim=2))
                    )).squeeze(0)
            else:
                d = self.density_factor * dens_tile / self.ray_samples
                transmission = torch.cumprod(1.0 - d, dim=2)
                weight = d * transmission
                w_sum = weight.sum(dim=2)
                alpha = 1.0 - torch.prod(1.0 - d, dim=2)
                return (weight.sum(dim=2) / (w_sum + 1e-6) * alpha).squeeze(0)  # (C, h, w)

        if tile_h is None or (self.h < 2 * tile_h and self.w < 2 * tile_w):
            # No tiling — single call
            grid = ijk.permute(0, 1, 2, 3).unsqueeze(0)  # (1, D, H, W, 3)
            if self.use_checkpointing and self.training:
                out = cp.checkpoint(render_tile, grid, use_reentrant=False)
            else:
                out = render_tile(grid)
        else:
            # Tile over H and W
            row_outputs = []
            for h_start in range(0, self.h, tile_h):
                h_end = min(h_start + tile_h, self.h)
                col_outputs = []
                for w_start in range(0, self.w, tile_w):
                    w_end = min(w_start + tile_w, self.w)
                    grid = ijk[:, h_start:h_end, w_start:w_end, :].unsqueeze(0)  # (1, D, h, w, 3)
                    if self.use_checkpointing and self.training:
                        tile_out = cp.checkpoint(render_tile, grid, use_reentrant=False)
                    else:
                        tile_out = render_tile(grid)
                    col_outputs.append(tile_out)
                row_outputs.append(torch.cat(col_outputs, dim=-1))
            out = torch.cat(row_outputs, dim=-2)  # (C, H, W)

        return out


if __name__ == '__main__':
    from monai.transforms import LoadImage, EnsureType, EnsureChannelFirst, Compose, Spacing, \
        ScaleIntensityRange
    from matplotlib import pyplot as plt
    from torch.profiler import profile, record_function, ProfilerActivity
    from torch.autograd import gradcheck

    import torch._inductor.config as cfg

    cfg.cpp.vec_isa_ok = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    load_tf = Compose([
        LoadImage(),
        EnsureChannelFirst(),
        Spacing([2.5, 2.5, 3.0]),
        # CenterSpatialCrop([152,152,140]),
        ScaleIntensityRange(-3024, 3024, 0, 1, clip=True),
        EnsureType(),
        # RandAffine(1, translate_range=((0, 0), (0, 0), (-15, 15)), padding_mode='zeros'),
    ])
    vol = load_tf('CTChest.nii.gz').unsqueeze(0)
    # vol.requires_grad_(True)

    tf = [torch.tensor([
        [-3500, 0.0],
        [-200, 0.0],
        [200, 0.05],
        [1535, 0.5],
        [3071, 0.65],
    ]).cuda()]
    tf[0][:, 0] = (tf[0][:, 0] + 3500) / 7000

    xp = tf[0][:, 0]
    yp = tf[0][:, -1]
    a = piecewise_linear_channelwise(vol.cuda(), xp.unsqueeze(0), yp.unsqueeze(0))

    vol = vol.cuda()
    hu = vol * (3024 - (-3524)) + (-3524)
    mu = torch.clamp(0.05 * (1.0 + hu / 800.0), min=0.0)

    ijk2ras = vol.meta['affine']
    ras2ijk = torch.inverse(ijk2ras)

    print(ijk2ras, ras2ijk)

    center = torch.ones(4).double()
    center[:3] = torch.as_tensor(vol.shape[2:]) // 2
    center = ijk2ras @ center
    print(center[:3])

    ren = VolumeRaycaster(scatter=None, resolution=(1024, 1024), i0=None, ray_samples=384).cuda().eval()
    # ren = torch.compile(ren)
    # vol = torch.rand(8, 1, 128, 128, 128).cuda()
    # vol.requires_grad_(True)

    view_mat = get_vtk_view_mat((0., 1000, -130.),
                                center[:3],
                                (0.0, 0.0, 1.), device='cuda').unsqueeze(0)

    # view_mat = get_vtk_view_mat((825.512239409456, -13.179125178309306, -150.8782530984467),
    #                             (0.7339149949892914, -69.45105638432082, -184.52283569821498),
    #                             (-0.0018398770598324777, -0.0012575743709173355, 0.9999975166764699))
    # print(view_mat.inverse())

    view_mat = view_mat.repeat(1, 1, 1)

    start = timer()
    out = ren(a.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk)
    torch.cuda.synchronize()
    end = timer()
    print(out.shape, end - start)

    plt.figure()
    plt.imshow(out[0, 0].detach().cpu().numpy(), cmap='gray')
    plt.show()

    start = timer()
    out = ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk)
    torch.cuda.synchronize()
    end = timer()
    print(out.shape, end - start)

    plt.figure()
    plt.imshow(out[0, 0].detach().cpu().numpy(), cmap='gray')
    plt.show()

    start = timer()
    out = ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk)
    torch.cuda.synchronize()
    end = timer()
    print(out.shape, end - start)

    for _ in range(10):
        with torch.no_grad():
            ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk)
    torch.cuda.synchronize()

    with torch.no_grad():
        out_triton = ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
        out_old = ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=False)
        print(f'Old and Triton match: {torch.allclose(out_old, out_triton, atol=1e-4)}')
        print(f'Old and Triton Max: {(out_old - out_triton).abs().max()}, Mean: {(out_old - out_triton).abs().mean()}')
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(out_old[0, 0].detach().cpu().numpy(), cmap='gray')
        ax[1].imshow(out_triton[0, 0].detach().cpu().numpy(), cmap='gray')
        im = ax[2].imshow((out_old - out_triton)[0, 0].detach().cpu().numpy(), cmap='coolwarm')
        cbar = fig.colorbar(im, ax=ax[2])
        plt.show()

    mu_prof = mu.expand(1, 8, -1, -1, -1)

    ren.eval()
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,  # True adds overhead, only enable for deep dives
    ) as prof:
        for _ in range(10):  # enough iterations to smooth noise
            with record_function("vmap_forward"):
                with torch.no_grad():
                    out = ren.forward(mu_prof, view_mat=view_mat, ras2ijk=ras2ijk)
            torch.cuda.synchronize()  # must sync or CUDA ops appear instant
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
    ))

    # start = timer()
    # out.sum().backward()
    # torch.cuda.synchronize()
    # end = timer()
    # print(end-start)

    print(torch.cuda.memory_summary())

    # plt.figure()
    # plt.imshow(out[0,0].detach().cpu().numpy(), cmap='gray')
    # plt.show()
    #
    # plt.figure()
    # plt.imshow(out2[0,0].detach().cpu().numpy(), cmap='gray')
    # plt.show()

    # plt.figure()
    # plt.imshow(out[0,-1].detach().cpu().numpy() - out[0,0].detach().cpu().numpy(), cmap='gray')
    # plt.show()
    # print(torch.abs(out[0,-1] - out[0,0]).max())

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

    # ren = VolumeRaycaster(scatter=None, ray_samples=512, resolution=(1024,1024), i0=1e2).cuda().eval()

    print("############ Triton Version ############")

    start = timer()
    out = ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
    torch.cuda.synchronize()
    end = timer()
    torch.cuda.synchronize()
    end = timer()
    print(out.shape, end - start)

    with torch.no_grad():
        _ = ren(mu.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
        _ = ren(mu.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
        start = timer()
        out = ren(mu.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
        torch.cuda.synchronize()
        end = timer()
        print(out.shape, end - start)

    plt.figure()
    plt.imshow(out[0, 0].detach().cpu().numpy(), cmap='gray')
    plt.show()

    for _ in range(10):
        with torch.no_grad():
            ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
    torch.cuda.synchronize()

    ren.eval()
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,  # True adds overhead, only enable for deep dives
    ) as prof:
        for _ in range(10):  # enough iterations to smooth noise
            with record_function("triton_forward"):
                with torch.no_grad():
                    out = ren.forward(mu_prof, view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
            torch.cuda.synchronize()  # must sync or CUDA ops appear instant
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
    ))

    # start = timer()
    # out.sum().backward()
    # torch.cuda.synchronize()
    # end = timer()
    # print(end-start)

    print(torch.cuda.memory_summary())

    with torch.no_grad():
        with torch.autocast(device_type="cuda"):
            _ = ren(mu.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
            _ = ren(mu.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
            start = timer()
            out_triton = ren(mu.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True).float()
            torch.cuda.synchronize()
            end = timer()
            print("Triton fp16", end - start)

            start = timer()
            out_old = ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=False).float()
            torch.cuda.synchronize()
            end = timer()
            print("Orig fp16", end - start)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = ren(mu.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
            _ = ren(mu.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True)
            start = timer()
            out_triton_bf16 = ren(mu.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True).float()
            torch.cuda.synchronize()
            end = timer()
            print("Triton bf16", end - start)

            start = timer()
            out_old_bf16 = ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=False).float()
            torch.cuda.synchronize()
            end = timer()
            print("Orig bf16", end - start)

        print(f'Old and Triton match fp16: {torch.allclose(out_old, out_triton, atol=1e-4)}')
        print(
            f'Old and Triton Max fp16: {(out_old - out_triton).abs().max()}, Mean: {(out_old - out_triton).abs().mean()}')

        out_triton_fp32 = ren(mu.expand(1, 1, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=True).float()
        out_old_fp32 = ren(mu.expand(1, 8, -1, -1, -1), view_mat=view_mat, ras2ijk=ras2ijk, triton=False).float()

        print(f'Old match fp32/fp16: {torch.allclose(out_old, out_old_fp32, atol=1e-4)}')
        print(f'Triton match fp32/fp16: {torch.allclose(out_triton_fp32, out_triton, atol=1e-4)}')

        print(
            f'Triton Max fp16/32: {(out_triton_fp32 - out_triton).abs().max()}, Mean: {(out_triton_fp32 - out_triton).abs().mean()}')

        print(f'Triton match fp32/bf16: {torch.allclose(out_triton_fp32, out_triton_bf16, atol=1e-4)}')
        print(
            f'Triton Max bf16/32: {(out_triton_fp32 - out_triton_bf16).abs().max()}, Mean: {(out_triton_fp32 - out_triton_bf16).abs().mean()}')

    ren = VolumeRaycaster(scatter=None, resolution=(512, 512), i0=None).cuda()

    torch.manual_seed(0)
    mu = torch.randn(2, 8, 24, 16, device='cuda', dtype=torch.float64, requires_grad=True)

    # isolate the triton Function; gradcheck the full expanded graph in a second pass if you want
    ok = gradcheck(
        lambda m: ren(
            m.expand(1, 2, -1, -1, -1),
            view_mat=view_mat, ras2ijk=ras2ijk, triton=True,
        ),
        (mu,),
        eps=1e-6,
        atol=1e-4,  # looser than the fp64 default 1e-5 to absorb kernel fp noise
        rtol=1e-3,
        nondet_tol=1e-6,  # tl.atomic_add is order-nondeterministic
        fast_mode=True,
    )
    print("gradcheck passed:", ok)
