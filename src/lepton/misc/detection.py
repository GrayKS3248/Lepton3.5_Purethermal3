# -*- coding: utf-8 -*-
# © Copyright, 2026 G. Schaer.
# SPDX-License-Identifier: GPL-3.0-only
"""
Provides functions for detecting features.
"""

import numpy as np
import cv2
from skimage.filters import (threshold_yen, threshold_multiotsu)
from scipy.ndimage import gaussian_filter

N_TEMPERATURE_CLASSES = 4

def detect_fp_fronts(ts, min_temp = 80.0):
    """
    Uses automatic thresholding of the temperature image, the gradient of the temperature image,
    and the time derivative of the temperature image sequence to estimate frontal polymerization
    front location.

    Parameters
    ----------
    ts : list of array of floats, shape( (m,n) )
        Time ordered temperature images in Celcius.
    min_temp : float, optional
        The minimum cutoff temperature for front definition in Celcius. The default is 80.

    Returns
    -------
    front_mask : array of bool, shape( (m,n) )
        A boolean mask of detected front instances.

    """
    # Get the candidate masks
    t_mask = _get_t_mask(ts[-1], min_temp)
    g_mask = _get_g_mask(ts[-1])

    # If only one temperature image was provided, front estimate cannot
    # use derivative of temperature. Return intersection of temperature
    # mask and gradient mask
    if len(ts) == 1:
        return t_mask & g_mask

    # Get the candidate dt mask
    dt_mask = _get_dt_mask(ts)

    # Return intersection of all masks
    return t_mask & g_mask & dt_mask

def _get_t_mask(temp, min_temp):
    # Apply multi-otsu thresholding to temperature to maximize inter-class variance
    # Candidate regions are those pixels between the lowest and highest threshold values
    shape = max(temp.shape) // 30
    shape -= shape % 2 - 1
    t = cv2.GaussianBlur(temp, (shape, )*2, 0)
    threshes = threshold_multiotsu(t[~np.isnan(t)], N_TEMPERATURE_CLASSES, nbins = 32)
    m = (t >= threshes[0]) & (t <= threshes[-1])

    # Apply the minimum cutoff temperature
    return m & (t>min_temp)

def _get_g_mask(temp):
    # Get the magnitude of the gradient of the temperature field
    dx, dy = np.gradient(temp)
    g = np.sqrt(dx * dx + dy * dy)
    shape = max(g.shape) // 30
    shape -= shape % 2 - 1
    g = cv2.GaussianBlur(g, (shape, )*2, 0)

    # Yen's method to maximize information entropy of mask
    return g > threshold_yen(g[~np.isnan(g)], nbins = 32)

def _get_dt_mask(temps):
    # Calculate the blurred temporal differential of the temperature
    # image sequence by spatiotemporal gaussian differentiation
    sigma = 0.036 * (max(temps[-1].shape) // 30) + 0.6
    dt = gaussian_filter(temps, sigma, order=(1, 0, 0), mode='nearest')[-1]

    # Isolate regions that got hotter. The front won't get colder.
    dt[dt < 0.0] = 0.0

    # Yen's method to maximize information entropy of mask
    return dt > threshold_yen(dt[~np.isnan(dt)], nbins = 512)
