# Package modules
from lepton import Client
from lepton import EOT
from lepton import NULL
from lepton import decode_frame_data


# Global socket constants
HOST = 'Laptop'
PORT = 8080 


def main():
    # Connect client to host
    with Client() as client:
        connected = client.connect(HOST, PORT)
        if not connected: return
        
        # Receive frame data until NULL or EOT sent
        while True:
            frame_data = client.recv_msgs()
            if frame_data == NULL: break
            if frame_data == EOT: break
        
            frame_data = decode_frame_data(frame_data)
            frame_num = tuple(frame_data[0].tolist())
            frame_time = tuple([round(float(f)*.001,3) for f in frame_data[1]])
            temperature = (0.01*frame_data[2].astype(float))-273.15
            telemetry = eval(frame_data[3][0])
            image = frame_data[4]
            mask = frame_data[5]
            
if __name__=="__main__":
    main()
        