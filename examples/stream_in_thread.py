import sys
sys.path.append("..")
from lepton import Lepton
from lepton import Videowriter
from lepton import decode_recording_data
sys.path.pop(-1)
import threading
import time

PORT = 0
CMAP = 'ironbow'
SCALE_FACTOR = 3
RECORD = True
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
        curr_frame = lepton.get_frame_number()
        curr_time = lepton.get_time()
        if curr_frame > prev_frame and not curr_time is None:
            print("Frame {} @ {:.3f}s".format(curr_frame, curr_time))
        prev_frame = curr_frame
        time.sleep(0.01) # Remove some CPU stress
        
    # Join the Lepton thread
    thread1.join()
    
    # Decode the recorded data
    if RECORD:
        writer = Videowriter()
        writer.make_video()
        data = decode_recording_data()
