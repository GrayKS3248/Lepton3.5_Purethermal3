# -*- coding: utf-8 -*-
"""
Defines exception types to be used by the package.

© Copyright, 2026 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

class _DefaultException(Exception):
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload

    def __str__(self):
        return str(self.message)

class ShapeException(_DefaultException):
    """
    Raised when the captured image shape does not match the expected image shape.

    Attributes
    ----------
        message : string
            Explanation of the error.
        payload: 2 tuple of 2 tuple of ints
            A tuple of the captured image shape and the expected image shape.
    """

class CaptureException(_DefaultException):
    """
    Raised when the Lepton capture fails.

    Attributes
    ----------
        message : string
            Explanation of the error.
        payload: ndarray
            The captured data, if any.
    """

class CaptureTimeout(_DefaultException):
    """
    Raised when the Lepton capture timesout.

    Attributes
    ----------
        message : string
            Explanation of the error.
        payload: ndarray
            The captured data, if any.
    """

class UnknownCmapException(_DefaultException):
    """
    Raised when attempting to access unkown or unsupported colormap.

    Attributes
    ----------
        message : string
            Explanation of the error.
        payload: string
            The name of the attempted colormap.
    """
