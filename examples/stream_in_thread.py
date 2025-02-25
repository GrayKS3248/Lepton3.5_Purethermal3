import sys
sys.path.append("..")
from lepton import Lepton
sys.path.pop(-1)
import threading

PORT = 0
CMAP = 'ironbow'
RECORD = False
FPS = None
DETECT = False
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
    is_streaming = lepton.is_streaming()
    prev_frame = lepton.get_frame_number()
    while is_streaming:
        frame = lepton.get_frame_number()
        if frame > prev_frame:
            print("Frame {} @ {:.3f}s".format(frame, lepton.get_time()))
        prev_frame = frame
        is_streaming = lepton.is_streaming()
        
    # Join the Lepton thread
    thread1.join()