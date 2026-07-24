# -*- coding: utf-8 -*-
"""
Package initialization.

© Copyright, 2026 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

from .constants import *
from .core.stream import Stream
from .scripts import leprun

__version__ = '1.0.1'

__all__ = [
    "Stream",
    "leprun"
]
for _x in constants.__all__:
    __all__.append(_x)
