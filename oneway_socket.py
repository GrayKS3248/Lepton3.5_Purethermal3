# Std modules
import socket

# External modules
import numpy as np

# Package modules
from utilities import print_exception
from utilities import ESC


NULL = b'\x00'
EOT = b''


class MessageLengthException(Exception):
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload
        
    def __str__(self):
        return str(self.message)


class MessagesTypeException(Exception):
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload
        
    def __str__(self):
        return str(self.message)


class MessageTypeException(Exception):
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload
        
    def __str__(self):
        return str(self.message)


class Host:
    def __init__(self):
        self.hostsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clientsocket = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
    
    def bind(self, port=8080):
        hostname = socket.gethostname()
        print('Binding host socket... ', end='', flush=True)
        self.hostsocket.bind((hostname, port))
        msg = '\"{}\" bound to port {}.'.format(hostname, port)
        print(ESC.okcyan(msg), flush=True)
    
    def connect(self, port=8080, timeout=10.0, timeout_func=None):
        self.hostsocket.listen(1)
        print("Listening for client... ", end='', flush=True)
        
        while True:
            if not timeout_func is None: timeout = 1.0
            try:
                self.hostsocket.settimeout(timeout)
                self.clientsocket, client_address = self.hostsocket.accept()
                msg = "{}Connected to {}{}"
                print(msg.format(ESC.OKCYAN, client_address[0], ESC.ENDC), 
                      flush=True)
                return True
            
            except TimeoutError:
                if timeout_func is None:
                    self.clientsocket = None
                    msg = "{}Timed out while attempting to connect.{}"
                    print(msg.format(ESC.WARNING, ESC.ENDC), flush=True)
                    return False
            
                elif not timeout_func():
                    self.clientsocket = None
                    msg = "{}() expired while attempting to connect."
                    msg = msg.format(timeout_func.__name__)
                    print(ESC.warning(msg), flush=True)
                    return False
                
                continue
            
            except BaseException as e:
                msg = ("{}Connection failed due to an unknown exception.{}")
                print(msg.format(ESC.WARNING, ESC.ENDC), flush=True)
                print_exception(e, self.connect)
                return False
    
    def _reestablish(self, msgs):
        msg = "{}Attempting to reestablish connection... {}"
        print(msg.format(ESC.WARNING, ESC.ENDC), flush=True)
        reconnected = self.connect(timeout=10, timeout_func=None)
        if reconnected:
            return self.send_msgs(msgs)
        self.clientsocket = None
        msg = ("{}MESSAGE FAILED TO SEND: "
               "Connection broken while sending and "
               "reestablishment failed.{}").format(ESC.FAIL, ESC.ENDC)
        print(msg, flush=True)
        return -1
    
    def _send_msgs(self, msgs):
        # Ensure messages are type tuple
        if type(msgs)!=tuple:
            msg = "Invalid messages type. Expected {}. Got {}."
            raise MessagesTypeException(msg.format(tuple, type(msgs)), 
                                       payload=msgs)
        
        # Ensure no more than 255 messages are sent at once
        if len(msgs) > 255:
            msg = "Attempted to send more than 255 messages ({}) at once."
            raise MessageLengthException(msg.format(len(msgs)), payload=msgs)
            return -1
        
        # Ensure each message is type bytes
        for i, msg in enumerate(msgs):
            if type(msg)!=bytes:
                msg = "Message {} has invalid type. Expected {}. Got {}."
                raise MessageTypeException(msg.format(i, bytes, type(msg)), 
                                           payload=msgs)
        
        # Build and send preamble
        preamble = (np.uint8(len(msgs))).tobytes()
        for msg in msgs:
            preamble += np.uint32(len(msg)).tobytes()
        self.clientsocket.send(preamble)
        
        # Send the message
        totalsent = 0
        for msg in msgs:
            msglen = len(msg)
            sentlen = 0
            while sentlen < msglen:
                sent = self.clientsocket.send(msg[sentlen:])
                if sent == 0:
                    raise RuntimeError("Connection broken while sending")
                sentlen += sent
            totalsent += sentlen
        return totalsent
    
    def send_msgs(self, msgs):
        try:
            return self._send_msgs(msgs)
        
        except ConnectionResetError:
            msg = ("{}Connection was forcibly closed by the remote host. {}")
            print(msg.format(ESC.WARNING,ESC.ENDC), end='', flush=True)
            self.clientsocket = None
            return self._reestablish(msgs)
        
        except ConnectionAbortedError:
            msg = ("{}Connection was aborted by the software"
                   " in your host machine.{} ")
            print(msg.format(ESC.WARNING,ESC.ENDC), end='', flush=True)
            self.clientsocket = None
            return self._reestablish(msgs)
        
        except MessageLengthException as e:
            msg = "{}MESSAGE FAILED TO SEND: {}{}"
            print(msg.format(ESC.FAIL, str(e), ESC.ENDC), flush=True)
            return -1
        
        except MessagesTypeException as e:
            msg = "{}MESSAGE FAILED TO SEND: {}{}"
            print(msg.format(ESC.FAIL, str(e), ESC.ENDC), flush=True)
            return -1
        
        except MessageTypeException as e:
            msg = "{}MESSAGE FAILED TO SEND: {}{}"
            print(msg.format(ESC.FAIL, str(e), ESC.ENDC), flush=True)
            return -1
        
        except BaseException as e:
            msg = "{}AN UNKNOWN EXCEPTION CAUSED THE MESSAGE TO NOT BE SENT:{}"
            print(msg.format(ESC.FAIL, ESC.ENDC), flush=True)
            print_exception(e, self._send_msgs)
            return -1
                
    def disconnect(self):
        if not self.clientsocket is None:
            print("Disconnecting from client... ",end='', flush=True)
            self.clientsocket.send(EOT)
            self.clientsocket.shutdown(socket.SHUT_RDWR)
            self.clientsocket.close()
            print("{}Disconnected.{}".format(ESC.OKCYAN, ESC.ENDC), flush=True)
            self.clientsocket = None
        
        print("Closing host socket... ", end='', flush=True)
        self.hostsocket.close()
        print("{}Closed.{}".format(ESC.OKCYAN, ESC.ENDC), flush=True)


class Client:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
        
    def connect(self, host, port):
        print("Connecting to host... ", end='', flush=True)
        try:
            self.socket.connect((host, port))
            self.connected = True
            print("{}Connected.{}".format(ESC.OKCYAN, ESC.ENDC), flush=True)
            return True
        
        except ConnectionRefusedError:
            msg = ("{}No connection could be made because the target "
                   "machine actively refused it.{}")
            print(msg.format(ESC.WARNING, ESC.ENDC), flush=True)
            self.connected = False
            return False
        
        except BaseException as e:
            msg = ("{}Connection failed due to an unknown exception.{}")
            print(msg.format(ESC.WARNING, ESC.ENDC), flush=True)
            print_exception(e, self.connect)
            return False
        
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
        if not self.connected:
            msg = ("{}Attempted to receive messages while not connected. "
                   "Aborting.{} ")
            print(msg.format(ESC.WARNING,ESC.ENDC), flush=True)
            return NULL
        
        # Get the number of sub messages
        header = self.socket.recv(1)
        if header == EOT:
            print("Message stream terminated by host.", flush=True)
            return EOT
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
    
    def disconnect(self):
        if not self.connected: return
        print("Disconnecting from host... ",end='', flush=True)
        self.socket.shutdown(socket.SHUT_RDWR)
        self.socket.close()
        self.connected = False
        print("{}Disconnected.{}".format(ESC.OKCYAN, ESC.ENDC), flush=True)
