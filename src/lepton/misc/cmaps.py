# -*- coding: utf-8 -*-
# © Copyright, 2026 G. Schaer.
# SPDX-License-Identifier: GPL-3.0-only
"""
Defines the colormaps used to visualize camera data.
"""

from compression import gzip
from pathlib import Path
import json
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import colormaps as _colormaps
from lepton.exceptions import UnknownCmapException

class Colormaps:
    """
    Stores all valid colormaps for access.

    Parameters
    ----------
    None.

    Attributes
    ----------
    None.

    """
    def __init__(self):
        # Load the default cmaps
        with gzip.open(
                Path(__file__).resolve().parent / Path("assets/cmaps.json.gz"),
                "rt",
                encoding="utf-8"
            ) as file:
            self._data = json.load(file)
        for k, v in self._data.items():
            self._data[k] = self._as_listed_cmap(v, k)

        # Load all matplotlib colormaps
        keys = list(_colormaps.keys()) + list(self._data.keys())
        vals = list(_colormaps.values()) + list(self._data.values())
        self._data = dict(zip(keys, vals))

    def __getitem__(self, key):
        try:
            return self._data[key]
        except KeyError as e:
            msg = "Attempted to access unkown or unsupported colormap."
            raise UnknownCmapException(msg, key) from e

    def __contains__(self, key):
        return key in self._data

    def _as_listed_cmap(self, fmap, name):
        vals = np.zeros((len(fmap), 4), dtype=float)
        vals[:,0] = [float((c >> 16) & 255) / 255.0 for c in fmap]
        vals[:,1] = [float((c >> 8) & 255) / 255.0 for c in fmap]
        vals[:,2] = [float(c & 255) / 255.0 for c in fmap]
        vals[:,3] = 1.0
        cmap = ListedColormap(vals, name=name)
        return cmap

    def get(self, key):
        """
        Returns the value of the item with the specified key.

        Parameters
        ----------
        key : string
            The key.

        Returns
        -------
        value
            The value of the item with the specified key.

        """
        return self._data.get(key, None)

    def keys(self):
        """
        Returns a view object. The view object contains the keys of Colormaps, as a list.

        Returns
        -------
        keys : dict_keys
            Colormap's keys.

        """
        return self._data.keys()

    def values(self):
        """
        Returns a view object. The view object contains the values of Colormaps, as a list.

        Returns
        -------
        values : dict_values
            Colormap's values.

        """
        return self._data.values()

    def items(self):
        """
        Returns a view object. The view object contains the key-value pairs of Colormaps,
        as tuples in a list.

        Returns
        -------
        items : dict_items
            Colormap's items.

        """
        return self._data.items()
