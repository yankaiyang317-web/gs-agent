"""Visual encodings for MCP observations."""

from __future__ import annotations

import numpy as np

from gs_env.geometry.unproject import DEPTH_MIN_ALPHA
from gs_env.types import Observation


def depth_preview(observation: Observation) -> np.ndarray:
    """Return RGB pseudo-colour depth aligned pixel-for-pixel with observation RGB.

    Near depth is warm (red/yellow), far depth is cool (cyan/blue), and invalid
    or low-alpha pixels are black. Percentile scaling is per observation so the
    spatial layout is readable even across differently scaled scenes.
    """
    depth = observation.depth
    valid = np.isfinite(depth) & (depth > 0) & (observation.alpha >= DEPTH_MIN_ALPHA)
    image = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return image
    low, high = np.percentile(depth[valid], [2, 98])
    normalized = np.clip((depth - low) / max(high - low, 1e-8), 0, 1)
    # Five-stop palette indexed from near (warm) to far (cool).
    palette = np.array([[255, 48, 32], [255, 224, 48], [48, 210, 96], [48, 210, 255], [32, 48, 255]], dtype=np.float32)
    scaled = normalized * (len(palette) - 1)
    index = np.floor(scaled).astype(np.int32).clip(0, len(palette) - 2)
    fraction = (scaled - index)[..., None]
    colors = palette[index] * (1 - fraction) + palette[index + 1] * fraction
    image[valid] = colors[valid].astype(np.uint8)
    return image
