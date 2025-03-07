from oneway_socket import Client
from oneway_socket import EOT
import numpy as np
import zlib


HOST = 'Laptop'
PORT = 8080 


with Client() as client:
    client.connect('Laptop', 8080)
    
    while True:
        data = client.recv_msgs()
        if data == EOT: break
    
        data = data[2]
        data = np.frombuffer(zlib.decompress(data), dtype=np.uint16)
        t_mK = data[2:].reshape(data[:2])
        