# -*- coding: utf-8 -*-
"""
Example usage of the Lepton camera.

© Copyright, 2026 G. Schaer.
SPDX-License-Identifier: GPL-3.0-only
"""

import time
from lepton import Stream

if __name__ == "__main__":

    # 1. Instantiate a member of the Stream class
    stream = Stream(
        dev_idx = 0                    # The Lepton's device ID
    )

    # 2. Start the stream in non-blocking mode
    stream.start(
        blocking = False,              # If the stream runs in blocking or non-blocking mode
        record = True,                 # If the stream is recorded
        detect = False,                # Whether FP fronts are detected and labeled
        cmap = "magma",                # The selected colormap used by the viewer window
        scale = 1.25,                  # The size of the viewer window
        dirpath = "Lepton_Recordings", # The parent directory into which the recording data is saved
    )

    # Run continuously while the stream is not done for a maximum of 10 seconds
    start_time = time.monotonic()
    while not stream.is_complete():

        # 3. Get the current frame
        # Note: This will return None until the stream starts
        # Note: After calling Stream.start, it takes ~2 seconds to start as the Lepton boots
        frame = stream.get_frame()

        # 4. Extract the frame
        # Note: frame["mask"] is None if detect is False
        if not frame is None:
            curr_frame_number = frame["num"]
            curr_frame_time = frame["time"]
            curr_temperature_in_C = frame["temperature"]
            curr_detected_front_mask = frame["mask"]
            curr_frame_telemetry = frame["telemetry"]

        # 5. Terminate the stream if it has been going on for too long
        # Note: To terminate, the user can also press "esc" while the viewer window is active
        if time.monotonic() - start_time > 10.0:
            stream.terminate()

        # 6. Suspend loop execution for small amount of time to reduce resource consumption
        # Note: This does not effect the Lepton frame rate or viewer window
        time.sleep(0.05)
