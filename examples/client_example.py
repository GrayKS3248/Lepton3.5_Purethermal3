# Std modules
import zlib
import sys
from copy import copy

# External modules
import numpy as np

# Package modules
sys.path.append("..")
from oneway_socket import Client
from oneway_socket import EOT
from oneway_socket import NULL
sys.path.pop(-1)


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
            data = client.recv_msgs()
            if data == NULL: break
            if data == EOT: break
        
            # Decompress and format frame data
            t_mK = copy(data[2])
            t_mK = np.frombuffer(zlib.decompress(t_mK), dtype=np.uint16)
            t_mK = t_mK[2:].reshape(t_mK[:2])
            t_C = 0.01*t_mK-273.15
            
            mask = copy(data[3])
            if mask != b'':
                mask = np.frombuffer(zlib.decompress(mask), dtype=np.uint16)
                mask = mask[2:].reshape(mask[:2]).astype(bool)
            
if __name__=="__main__":
    main()
        