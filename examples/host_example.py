# Std modules
import threading

# External modules
import numpy as np

# Package modules
from lepton import Lepton
from lepton import Videowriter
from lepton import Host


# Global camera constants
CAMERA_PORT = 0
CMAP = 'black_hot'
SCALE_FACTOR = 3
RECORD = False
FPS = None
DETECT = False
MULTIFRAME = True
EQUALIZE = False

# Global socket constants
PORT = 8080 


def initialize():
    # Initialize lepton camera
    lepton = Lepton(CAMERA_PORT, CMAP, SCALE_FACTOR, False)
    
    # Begin streaming in a thread
    args = (FPS, DETECT, MULTIFRAME, EQUALIZE)
    if RECORD:
        thread=threading.Thread(target=lepton.start_record, args=args)
    else:
        thread=threading.Thread(target=lepton.start_stream, args=args)
    thread.start()

    return lepton, thread

def main(lepton):
    # Wait until the stream is active
    if lepton.wait_until_stream_active(timeout_ms=10000.0) < 0:
        lepton.emergency_stop()
        
    # Create a host socket to send captured data
    with Host() as host:
        host.bind(PORT)
        if not host.connect(timeout_func=lepton.is_streaming):
            if lepton.is_streaming():
                lepton.emergency_stop()
            return
        
        # While lepton is streaming, get and send frame data
        while lepton.is_streaming():
            frame_data = lepton.get_frame_data(focused_ok=True, encoded=True)
            if any([f is None for f in frame_data]): continue
        
            ret = host.send_msgs(frame_data)
            if ret != np.sum([len(dat) for dat in frame_data]):
                lepton.emergency_stop()
                break

def terminate(thread):
    # Join the Lepton thread
    thread.join()
    
    # Decode the recorded data
    if RECORD:
        writer = Videowriter()
        result, raw_data = writer.make_video()
        return raw_data


if __name__ == "__main__":   
    lepton, thread = initialize()
    main(lepton)
    raw_data = terminate(thread)



