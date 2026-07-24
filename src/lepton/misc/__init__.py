# -*- coding: utf-8 -*-
"""
Subpackage: misc initialization.

© Copyright, 2026 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

from .cmaps import Colormaps
from .detection import detect_fp_fronts

colormaps = Colormaps()

__all__ = [
    "colormaps",
    "detect_fp_fronts",
]
