import sys
sys.path.append("..")
from lepton import Lepton
sys.path.pop(-1)
import threading
import time

PORT = 0
CMAP = 'ironbow'
RECORD = False
FPS = None
DETECT = True
MULTIFRAME = True
EQUALIZE = False

if __name__ == "__main__":   
    # Initialize lepton camera
    lepton = Lepton(PORT, CMAP, RECORD)
    
    # Begin streaming in a thread
    args = (FPS, DETECT, MULTIFRAME, EQUALIZE)
    thread1=threading.Thread(target=lepton.start_stream, args=args)
    thread1.start()
    lepton.wait_until_stream_active()
    
    # Do other things while Lepton is streaming
    prev_frame = -1
    while lepton.is_streaming():
        curr_frame = lepton.get_frame_number()
        curr_time = lepton.get_time()
        if curr_frame > prev_frame and not curr_time is None:
            print("Frame {} @ {:.3f}s".format(curr_frame, curr_time))
        prev_frame = curr_frame
        time.sleep(0.01) # Remove some CPU stress
        
    # Join the Lepton thread
    thread1.join()