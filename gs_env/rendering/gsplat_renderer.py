"""Headless gsplat renderer for one already-resident Gaussian scene."""

from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
import sys

import numpy as np
import torch
from gsplat import rasterization

from gs_env.types import CameraIntrinsics, Observation
from .ply_loader import GaussianScene, load_gaussian_ply


class GSplatRenderer:
    """Render RGB, accumulated z-depth, and accumulated opacity from a c2w pose."""

    def __init__(self, device: str | torch.device = "cuda", background: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.device = torch.device(device)
        self.background = torch.tensor(background, dtype=torch.float32, device=self.device)
        self.scene: GaussianScene | None = None

    def load_scene(self, ply_path: str | Path) -> GaussianScene:
        """Load PLY once. Subsequent renders reuse the same GPU tensors."""
        self.scene = load_gaussian_ply(ply_path, self.device)
        return self.scene

    @property
    def is_loaded(self) -> bool:
        return self.scene is not None

    @torch.inference_mode()
    def render(self, camera_to_world: np.ndarray, intrinsics: CameraIntrinsics) -> Observation:
        if self.scene is None:
            raise RuntimeError("No scene loaded. Call load_scene() first.")
        c2w = np.asarray(camera_to_world, dtype=np.float32)
        if c2w.shape != (4, 4):
            raise ValueError(f"camera_to_world must have shape (4, 4), got {c2w.shape}")
        if not np.isfinite(c2w).all():
            raise ValueError("camera_to_world contains non-finite values")
        if intrinsics.width <= 0 or intrinsics.height <= 0 or intrinsics.fx <= 0 or intrinsics.fy <= 0:
            raise ValueError("Image size and focal lengths must be positive")

        world_to_camera = np.linalg.inv(c2w)
        viewmats = torch.from_numpy(world_to_camera).to(self.device).unsqueeze(0)
        ks = torch.tensor(
            [[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        # Some gsplat versions announce a first-use CUDA build on stdout.
        # Stdio MCP reserves stdout exclusively for JSON-RPC, so third-party
        # renderer diagnostics must be routed to stderr.
        with redirect_stdout(sys.stderr):
            rendered, alpha, _ = rasterization(
                self.scene.means,
                self.scene.quats,
                self.scene.scales,
                self.scene.opacities,
                self.scene.colors,
                viewmats,
                ks,
                intrinsics.width,
                intrinsics.height,
                sh_degree=self.scene.sh_degree,
                render_mode="RGB+D",
                packed=True,
            )
        output = rendered[0].float().cpu().numpy()
        return Observation(
            rgb=np.clip(output[..., :3], 0.0, 1.0),
            depth=output[..., 3],
            alpha=alpha[0, ..., 0].float().cpu().numpy(),
            camera_to_world=c2w.copy(),
            intrinsics=intrinsics,
        )
