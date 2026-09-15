"""Load standard 3D Gaussian Splatting PLY files into GPU-ready tensors.

The canonical format is the one emitted by the original 3DGS implementation:
``x/y/z``, ``f_dc_*``, ``f_rest_*``, ``opacity``, ``scale_*`` and ``rot_*``.
The source values are decoded here: opacity logits -> sigmoid, log-scales -> exp,
and rotations are normalized wxyz quaternions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from plyfile import PlyData


@dataclass(frozen=True)
class GaussianScene:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    sh_degree: int | None
    source_fields: tuple[str, ...]
    load_seconds: float

    @property
    def count(self) -> int:
        return int(self.means.shape[0])


def inspect_ply_fields(ply_path: str | Path) -> tuple[str, ...]:
    """Return vertex field names without decoding the Gaussian parameters."""
    ply = PlyData.read(str(ply_path), mmap="r")
    if "vertex" not in ply:
        raise ValueError("PLY has no 'vertex' element")
    return tuple(ply["vertex"].data.dtype.names or ())


def _ordered(fields: tuple[str, ...], prefix: str, required: bool = True) -> list[str]:
    matches = [field for field in fields if field.startswith(prefix)]
    try:
        matches.sort(key=lambda field: int(field.rsplit("_", 1)[1]))
    except (IndexError, ValueError) as error:
        raise ValueError(f"Fields with prefix '{prefix}' must have numeric suffixes: {matches}") from error
    if required and not matches:
        raise ValueError(f"Missing required PLY fields with prefix '{prefix}'")
    return matches


def _columns(vertex: np.ndarray, names: list[str]) -> np.ndarray:
    return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=1)


def load_gaussian_ply(ply_path: str | Path, device: str | torch.device = "cuda") -> GaussianScene:
    """Read a canonical 3DGS PLY once and upload its decoded tensors to ``device``."""
    started = perf_counter()
    path = Path(ply_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    ply = PlyData.read(str(path), mmap="r")
    if "vertex" not in ply:
        raise ValueError(f"{path} has no 'vertex' element")
    vertex = ply["vertex"].data
    fields = tuple(vertex.dtype.names or ())
    required = {"x", "y", "z", "opacity"}
    missing = sorted(required.difference(fields))
    if missing:
        raise ValueError(f"{path} is not a standard 3DGS PLY; missing {missing}. Fields: {fields}")

    dc_names = _ordered(fields, "f_dc_")
    if len(dc_names) != 3:
        raise ValueError(f"Expected f_dc_0..2, got {dc_names}")
    scale_names = _ordered(fields, "scale_")
    rotation_names = _ordered(fields, "rot_")
    if len(scale_names) != 3 or len(rotation_names) != 4:
        raise ValueError("Standard 3DGS requires three scale_* and four rot_* fields")
    rest_names = _ordered(fields, "f_rest_", required=False)
    if len(rest_names) % 3:
        raise ValueError(f"f_rest_* count must be divisible by 3, got {len(rest_names)}")

    means = _columns(vertex, ["x", "y", "z"])
    # Original 3DGS stores RGB SH coefficients as [R coefficients, G coefficients, B coefficients].
    dc = _columns(vertex, dc_names).reshape(-1, 1, 3)
    if rest_names:
        rest = _columns(vertex, rest_names).reshape(-1, 3, len(rest_names) // 3)
        rest = np.transpose(rest, (0, 2, 1))
        colors = np.concatenate((dc, rest), axis=1)
        sh_degree = int(round(np.sqrt(colors.shape[1]) - 1))
        if (sh_degree + 1) ** 2 != colors.shape[1]:
            raise ValueError(f"Invalid SH coefficient count: {colors.shape[1]}")
    else:
        colors = dc
        sh_degree = 0

    quat_np = _columns(vertex, rotation_names)
    quat_np /= np.maximum(np.linalg.norm(quat_np, axis=1, keepdims=True), 1e-12)
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    return GaussianScene(
        means=torch.from_numpy(means.copy()).to(target),
        quats=torch.from_numpy(quat_np.copy()).to(target),
        scales=torch.from_numpy(np.exp(_columns(vertex, scale_names)).copy()).to(target),
        opacities=torch.from_numpy((1.0 / (1.0 + np.exp(-_columns(vertex, ["opacity"]).reshape(-1)))).copy()).to(target),
        colors=torch.from_numpy(colors.copy()).to(target),
        sh_degree=sh_degree,
        source_fields=fields,
        load_seconds=perf_counter() - started,
    )
