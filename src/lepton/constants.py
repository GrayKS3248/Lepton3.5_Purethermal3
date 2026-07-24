# -*- coding: utf-8 -*-
"""
Defines constants used by the package.

© Copyright, 2026 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

from typing import Final
import numpy as np

WIDTH: Final[int] = 160          # The width of the frame produced by the camera
HEIGHT: Final[int] = 120         # The height of the frame produced by the camera
SHAPE: Final[tuple[int, int]] = (WIDTH, HEIGHT)
RES: Final[int] = WIDTH * HEIGHT
TELEM_HEIGHT: Final[int] = 30    # The height of the telemetry readout in pixels
MASK_ALPHA: Final[float] = 0.6   # The alpha value to give to the detection mask
MASK_COLOR: Final[np.ndarray] = np.array([0.0, 1.0, 0.0]) # The mask color in RGB

# Add the constants to all
__all__ = [_x for _x in dir() if not _x.startswith("_") and _x.isupper()]
