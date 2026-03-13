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

# --- CALIBRATION SETTING ---
# If the camera is still slightly off, change this number.
USER_CALIBRATION_OFFSET = -60.0 

# Global socket constants
PORT = 8080 


def initialize():
    # Initialize lepton camera
    # The fourth argument 'False' disables internal AGC to keep raw data
    lepton = Lepton(CAMERA_PORT, CMAP, SCALE_FACTOR, False)
    
    # Begin streaming in a thread
    args = (FPS, DETECT, MULTIFRAME, EQUALIZE)
    if RECORD:
        thread = threading.Thread(target=lepton.start_record, args=args)
    else:
        thread = threading.Thread(target=lepton.start_stream, args=args)
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
            # Get raw 14-bit data (encoded=False for temperature accuracy)
            frame_data = lepton.get_frame_data(focused_ok=True, encoded=False)
            
            if frame_data is None: 
                continue
            
            # --- RADIOMETRIC PROCESSING ---
            calibrated_frames = []
            for raw_frame in frame_data:
                if raw_frame is None: 
                    continue
                
                # 1. Mask top 2 bits & convert to Celsius
                # 2. Add the calibration offset to fix the 60-degree error
                temp_c = ((raw_frame & 0x3FFF) / 100.0) - 273.15 + USER_CALIBRATION_OFFSET
                
                # 3. Convert back to float32 so the Host can send it as a data packet
                calibrated_frames.append(temp_c.astype(np.float32))
            # --- END PROCESSING ---
        
            # Send the corrected Celsius data to the host
            ret = host.send_msgs(calibrated_frames)
            
            # Calculate total expected length for verification
            expected_len = np.sum([f.nbytes for f in calibrated_frames])
            if ret != expected_len:
                # If data is lost during transmission, stop for safety
                lepton.emergency_stop()
                break

def terminate(thread):
    # Join the Lepton thread
    thread.join()
    
    # Decode the recorded data if needed
    if RECORD:
        writer = Videowriter()
        result, raw_data = writer.make_video()
        return raw_data


if __name__ == "__main__":   
    lepton, thread = initialize()
    main(lepton)
    raw_data = terminate(thread)
