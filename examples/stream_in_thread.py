# Std modules
import sys
import threading
import zlib

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
RECORD = True
FPS = None
DETECT = False
MULTIFRAME = True
EQUALIZE = False

# Global socket constants
PORT = 8080 


def initialize():
    # Initialize lepton camera
    lepton = Lepton(CAMERA_PORT, CMAP, SCALE_FACTOR)
    
    # Begin streaming in a thread
    args = (FPS, DETECT, MULTIFRAME, EQUALIZE)
    if RECORD:
        thread=threading.Thread(target=lepton.start_record, args=args)
    else:
        thread=threading.Thread(target=lepton.start_stream, args=args)
    thread.start()

    return lepton, thread

def to_msgs(frame_data):
    frame_num = frame_data[0]
    time_stamp = frame_data[1]
    temperature_C = frame_data[2]
    mask = frame_data[3]
    
    f_data = np.uint64(frame_num).tobytes()
    
    t_data = np.uint64(time_stamp).tobytes()
    
    t_mK = np.round(100*(temperature_C+273.15)).astype(np.uint16)
    T_data = np.insert(t_mK.flatten(),0,t_mK.shape).tobytes()
    T_data = zlib.compress(T_data)
    
    if mask is None: 
        m_data = b''
    else:
        m_data = np.insert(mask.flatten(),0,mask.shape).tobytes()
        m_data = zlib.compress(m_data)
    
    return  (f_data, t_data, T_data, m_data)

def main(lepton):
    # Wait until the stream is active
    if lepton.wait_until_stream_active(timeout_ms=5000.0) < 0:
        lepton.emergency_stop()
        
    # Create a host socket to send captured data
    with Host() as host:
        connected = host.connect(PORT, 
                                 timeout=1, timeout_func=lepton.is_streaming)
        
        prev_frame = -1
        while connected and lepton.is_streaming():
            
            frame_data = lepton.get_frame_data(focused_ok=True)
            if frame_data[0] > prev_frame and not frame_data[1] is None:
                msgs = to_msgs(frame_data)
                host.send_msgs(msgs)
                
            prev_frame = frame_data[0]

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



