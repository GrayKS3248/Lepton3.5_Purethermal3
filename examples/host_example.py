# Std modules
import sys
import threading

# External modules
import numpy as np

# Package modules
sys.path.append("..")
from lepton import Lepton
from lepton import Videowriter
from lepton import decode_recording_data
from oneway_socket import Host
sys.path.pop(-1)


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
        connected = host.connect(timeout_func=lepton.is_streaming)
        if not connected:
            lepton.emergency_stop()
            return
        
        # While lepton is streaming, get and send frame data
        while lepton.is_streaming():
            frame_data = lepton.get_frame_data(focused_ok=True, as_bytes=True)
            if frame_data[1] == b'': continue
        
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
        writer.make_video()
        raw_data = decode_recording_data()
        return raw_data


if __name__ == "__main__":   
    lepton, thread = initialize()
    main(lepton)
    raw_data = terminate(thread)



