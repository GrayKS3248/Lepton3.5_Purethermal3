# -*- coding: utf-8 -*-
"""
Classes used to record camera stream and render video.

© Copyright, 2026 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

from threading import Thread
from collections import deque
from datetime import datetime
from pathlib import Path
import zipfile
import struct
import os
import json
from compression import gzip
from multiprocessing import Pool
import cv2
import numpy as np
import lepton
from . import ViewerImage

class FrameWriter:
    """
    Writes frame information to zip archive during camera stream and then extracts and deletes that
    archive when streaming is done

    Parameters
    ----------
    opts: dict
        A dictionary of options defined at the start of a stream. Must include the keys
        "dirpath": string
            The path to the directory in which the frame archive is made.
        "scale": float > 1
            The scale of the viewer window. Used to properly size the image.
        "cmap": matplotlib.colors.ListedColormap
            The colormap used to colorize the image.
        "record": bool
            When True, adds a recording circle to the top right of the image.

    Attributes
    ----------
    src_verts : list of tuples
        The coordinates of the corners of the ROI defined in viewer window coordinates.

    """
    def __init__(self, opts):
        parentpath = Path(opts["dirpath"])
        parentpath.mkdir(parents=True, exist_ok=True)
        fname = Path(datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        self._dirpath = parentpath / fname
        self._archive = None
        self._opts = opts
        self._archive_fnames = deque([])

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _extract(self):
        frames = {
            "frame_number": [],
            "frame_time": [],
            "temperature": [],
            "mask": [],
            "telemetry": [],
        }
        for f in self._archive.namelist():
            with self._archive.open(f) as file:
                dat = file.read()
                fmt_str = "II" + "H"*lepton.RES + "B"*lepton.RES + "B"*(len(dat)-(3*lepton.RES+8))
                dat = struct.unpack(fmt_str, dat)
                temp = dat[2:(lepton.RES + 2)]
                temp = np.array(temp).reshape(lepton.SHAPE[::-1]) * .01 - 273.15
                temp[temp > 250.0] = float('nan')
                temp[temp < -50.0] = float('nan')
                mask = dat[(lepton.RES + 2):(2*lepton.RES + 2)]
                mask = np.array(mask, dtype=bool).reshape(lepton.SHAPE[::-1])
                telem = dat[(2*lepton.RES + 2):]
                telem = json.loads(bytes(telem).decode("utf-8"))
                frames["frame_number"].append(dat[0])
                frames["frame_time"].append(dat[1])
                frames["temperature"].append(temp)
                frames["mask"].append(mask)
                frames["telemetry"].append(telem)

        self._dirpath.mkdir()
        makevideo(
            frames["temperature"],
            frames["telemetry"],
            frames["mask"],
            self._opts,
            self._dirpath / ("video.mp4")
        )
        for k, v in frames.items():
            if k == "telemetry":
                continue
            with gzip.open(self._dirpath / (k + ".npy.gz"), "wb") as f:
                np.save(f, np.array(v))
        with gzip.open(self._dirpath / ("telemetry.json.gz"), "wt", encoding="utf-8") as file:
            json.dump(frames["telemetry"], file, indent = 4)

    def add(self, data):
        """
        Adds a frame to the frame archive.

        Parameters
        ----------
        data: dict
            A dictionary containing the frame information. Must include the keys
            "num": int
                The frame number.
            "time": int
                The frame time in ms.
            "temperature": ndarray
                A float ndarray of the frame temperature in C.
            "telemetry": dict
                The frame telemetry
            "mask": ndarray
                A bool ndarray of the frame detection mask.

        Returns
        -------
        None.

        """
        if data is None:
            return
        n = np.array([data["num"], ], dtype=np.uint32)
        t = np.array([data["time"], ], dtype=np.uint32)
        temp = np.round(100.0 * (data["temperature"].flatten() + 273.15))
        temp[temp > 52315] = 52315 # 250 C
        temp[temp < 22315] = 22315 # -50 C
        temp[np.isnan(temp)] = 0   # Handle nans
        temp = temp.astype(np.uint16)
        try:
            mask = data["mask"].flatten().astype(np.uint8)
        except AttributeError:
            mask = np.zeros(lepton.RES, dtype=np.uint8)
        telem = json.dumps(data["telemetry"])
        msg = n.tobytes() + t.tobytes() + temp.tobytes() + mask.tobytes() + telem.encode("utf-8")
        fname = f"fr{data["num"]:07d}.dat"
        if not fname in self._archive_fnames:
            self._archive.writestr(fname, msg)
            self._archive_fnames.append(fname)

    def open(self):
        """
        Opens the frame writer's zip archive.

        Returns
        -------
        None.

        """
        self._archive = zipfile.ZipFile(
            self._dirpath.with_suffix(".zip"),
            "w",
            zipfile.ZIP_DEFLATED
        )

    def close(self):
        """
        Closes the frame writer's zip archive. Must be called when done.

        Returns
        -------
        None.

        """
        self._extract()
        self._archive.close()
        os.remove(self._dirpath.with_suffix(".zip"))
        self._archive = None

def _render_frame(frame):
    return cv2.cvtColor(ViewerImage(*frame).asuint8(), cv2.COLOR_BGR2RGB)

def _write_loop(frames, path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    i0 = ViewerImage(*frames[0]).asuint8()
    out = cv2.VideoWriter(path, fourcc, 30, (i0.shape[1], i0.shape[0]))
    try:
        with Pool() as pool:
            rendered_frames = pool.map(_render_frame, frames)
        for rendered_frame in rendered_frames:
            out.write(rendered_frame)
    finally:
        out.release()

def makevideo(temperature, telemetry, mask, opts, path):
    """
    Renders and saves a recording.

    Parameters
    ----------
    temperature: list[ndarray]
        A list of float ndarrays of the frames' temperatures in C.
    telemetry: list[dict]
        A list of the frames' telemetries.
    mask: list[ndarray]
        A list of bool ndarrays of the frames' detection masks.
    opts: dict
        A dictionary of options defined at the start of a stream. Must include the keys
        "scale": float > 1
            The scale of the viewer window. Used to properly size the image.
        "cmap": matplotlib.colors.ListedColormap
            The colormap used to colorize the image.
        "record": bool
            When True, adds a recording circle to the top right of the image.
    path: pathlib.Path
        The path of the video.

    Returns
    -------
    None.

    """
    if len(temperature) < 1:
        return
    t0 = telemetry[0]["Uptime (ms)"]
    temp_times = [t["Uptime (ms)"] - t0 for t in telemetry]
    frame_times = np.round(np.arange(0.0, round(temp_times[-1] + 100/3, 8), round(100/3, 8)), 4)
    temp_indices = frame_times[:, np.newaxis] - temp_times
    temp_indices = np.where(temp_indices > 0, temp_indices, np.inf).argmin(axis=1)
    frames = [(temperature[i], telemetry[i], mask[i], t0, opts) for i in temp_indices]
    thread = Thread(target=_write_loop, args=(frames, path, ))
    thread.start()
