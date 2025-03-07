# Std modules
import socket

# External modules
import numpy as np


EOT = b''


class Host:
    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client = None
    
    def connect(self, port=8080, timeout=10, timeout_func=None):
        self.server.bind((socket.gethostname(), port))
        self.server.listen(1)
        print("Connecting to client...")
        
        while True:
            try:
                self.server.settimeout(timeout)
                self.client, self.client_address = self.server.accept()
                print("Connected to {}".format(self.client_address[0]))
                return True
            
            except TimeoutError:
                if timeout_func is None or not timeout_func():
                    print("Timed out while attempting to connect.")
                    return False
                continue
    
    def disconnect(self):
        if not self.client is None:
            print("Disconnecting from client...")
            self.client.send(EOT)
            self.client.shutdown(socket.SHUT_RDWR)
            print("Disconnected.")
        print("Closing host socket...")
        self.server.close()
        print("Closed.")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
        
    def send_msgs(self, msgs):
        # Ensure no more than 255 messages are sent at once
        if len(msgs) > 255:
            return 0
        
        # Build and send preamble
        preamble = (np.uint8(len(msgs))).tobytes()
        for msg in msgs:
            preamble += np.uint32(len(msg)).tobytes()
        self.client.send(preamble)
        
        # Send the message
        totalsent = 0
        for msg in msgs:
            msglen = len(msg)
            sentlen = 0
            while sentlen < msglen:
                sent = self.client.send(msg[sentlen:])
                if sent == 0:
                    raise RuntimeError("Connection broken while sending")
                sentlen += sent
            totalsent += sentlen
        return totalsent

class Client:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    def connect(self, host, port):
        print("Connecting to host...")
        self.socket.connect((host, port))
        print("Connected.")
    
    def disconnect(self):
        print("Closing client socket...")
        self.socket.shutdown(socket.SHUT_RDWR)
        self.socket.close()
        print("Closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
        
    def _recv(self, msglen, chunk_size):
        # Read the msg in chunks
        chunks = []
        bytes_recd = 0
        while bytes_recd < msglen:
            chunk = self.socket.recv(min(msglen-bytes_recd, chunk_size))
            if chunk == b'':
                raise RuntimeError("Connection broken while receiving")
            chunks.append(chunk)
            bytes_recd += len(chunk)
        return b''.join(chunks)
        
    def recv_msgs(self, chunk_size=2048):
        # Get the number of sub messages
        header = self.socket.recv(1)
        if header == EOT: return EOT
        n_msgs = np.frombuffer(header, dtype=np.uint8)[0]
        
        # Get the msglens
        msglens = []
        for i in range(n_msgs):
            msglen = np.frombuffer(self.socket.recv(4), dtype=np.uint32)[0]
            msglens.append(msglen)
        
        # Read the messages
        msgs = []
        for msglen in msglens:
            msgs.append(self._recv(msglen, chunk_size))
        return msgs
