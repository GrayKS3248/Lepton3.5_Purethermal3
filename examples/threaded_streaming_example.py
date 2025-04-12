# Std modules
import threading

# Package modules
from lepton import Lepton
from lepton import Videowriter


# Global camera constants
CAMERA_PORT = 0
CMAP = 'black_hot'
SCALE_FACTOR = 3
RECORD = False
FPS = None
DETECT = False
MULTIFRAME = True
EQUALIZE = False


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
    
    # Stream the lepton data
    while lepton.is_streaming():
        frame_data = lepton.get_frame_data(focused_ok=True)
        
        # These values will be None if attempt to get data from same frame
        frame_num = frame_data[0]
        frame_time_ms = frame_data[1]
        temperature_cK = frame_data[2]
        telemetry = frame_data[3]
        image = frame_data[4]
        mask = frame_data[5]

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



