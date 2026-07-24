# -*- coding: utf-8 -*-
# © Copyright, 2026 G. Schaer.
# SPDX-License-Identifier: GPL-3.0-only
"""
Package initialization.
"""

from .constants import *
from .core.stream import Stream
from .scripts import leprun

__version__ = '1.0.2'

__all__ = [
    "Stream",
    "leprun"
]
for _x in constants.__all__:
    __all__.append(_x)
