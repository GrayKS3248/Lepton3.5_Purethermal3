import sys
sys.path.append("..")
from lepton import Lepton
from lepton import Videowriter
from lepton import decode_recording_data
sys.path.pop(-1)
import threading
import time
import matplotlib.pyplot as plt
import numpy as np

PORT = 0
CMAP = 'viridis'
SCALE_FACTOR = 4
RECORD = False
FPS = None
DETECT = True
MULTIFRAME = True
EQUALIZE = False

if __name__ == "__main__":   
    # Initialize lepton camera
    lepton = Lepton(PORT, CMAP, SCALE_FACTOR)
    
    # Begin streaming in a thread
    args = (FPS, DETECT, MULTIFRAME, EQUALIZE)
    if RECORD:
        thread1=threading.Thread(target=lepton.start_record, args=args)
    else:
        thread1=threading.Thread(target=lepton.start_stream, args=args)
    thread1.start()
    if lepton.wait_until_stream_active(timeout_ms=5000.0) < 0:
        lepton.emergency_stop()
    
    # Do other things while Lepton is streaming
    prev_frame = -1
    while lepton.is_streaming():
        frame_data = lepton.get_frame_data()
        if frame_data[0] > prev_frame and not frame_data[1] is None:
            temperature = frame_data[2]
            mask = frame_data[3]
            if np.any(mask):
                temperature[mask]=np.inf
            plt.imshow(temperature)
            plt.show()
            
        prev_frame = frame_data[0]
        time.sleep(0.0111) # Remove some CPU stress
        
    # Join the Lepton thread
    thread1.join()
    
    # Decode the recorded data
    if RECORD:
        writer = Videowriter()
        writer.make_video()
        raw_data = decode_recording_data()
