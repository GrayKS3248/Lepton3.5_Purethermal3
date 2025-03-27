# Std modules
import os
import ast
import zlib
from fractions import Fraction
from copy import copy
import textwrap
import inspect

# Package modules
from lepton.exceptions import InvalidNameException
from lepton.misc.utilities import safe_run, ESC

# External modules
import cv2
import numpy as np
import av


def decode_recording_data(dirpath='rec_data', telemetry_file='telem.json',
                          temperature_file='temperature.dat',
                          mask_file='mask.dat'):
    print("Decoding raw data... ", end='', flush=True)
    _read_DELIMed = Videowriter()._read_DELIMed
    _decode_bytes = Videowriter()._decode_bytes
    
    with open(os.path.join(dirpath, telemetry_file), 'r') as f:
        telems = _read_DELIMed(f, 'r')
    if telems == None:
        timestamps_ms = None
    else:
        telems=[ast.literal_eval(''.join(t)) for t in telems]
        timestamps_ms=[t['Uptime (ms)'] for t in telems]
    
    with open(os.path.join(dirpath, temperature_file), 'rb') as f:
        t_cK = _read_DELIMed(f, 'rb')
    if t_cK == None:
        temperatures_C = None
    else:
        temperatures_C = [0.01*_decode_bytes(t)-273.15 for t in t_cK]
    
    with open(os.path.join(dirpath, mask_file), 'rb') as f:
        masks = _read_DELIMed(f, 'rb')
    if masks != None:
        masks = [_decode_bytes(m, compressed=True) for m in masks]
    
    data = {'Temperature (C)' : temperatures_C,
            'Mask' : masks,
            'Timestamp (ms)' : timestamps_ms,
            'Telemetry': telems}
    print("{}Done.{}".format(ESC.OKCYAN, ESC.ENDC), flush=True)
    return data


class Videowriter():
    def __init__(self, rec_name='recording', dirpath='rec_data', 
                 telemetry_file='telem.json', image_file='image.dat'):
        self.REC_NAME = self._get_valid_name(rec_name)
        self.DIR_PATH = dirpath
        self.TELEMETRY_FILE = telemetry_file
        self.IMAGE_FILE = image_file
    
    def _read_DELIMed(self, f, mode='r'):
        data = []
        if mode == 'r':
            data = f.read().split('DELIM')
        elif mode == 'rb':
            data = f.read().split(b'DELIM')
            
        if len(data) > 1: return data[:-1]
        else: return None
    
    def _decode_bytes(self, byts, compressed=False):
        if compressed:
            nparr = np.frombuffer(zlib.decompress(byts), dtype=bool)
            return nparr.reshape((120,160))
        else:
            nparr = np.frombuffer(byts, np.byte)
            return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    def _get_valid_name_(self, rec_name):
        illegal_chars = ("\\", "/", "<", ">", ":", "|", "?", "*", ".")
        if any(illegal_char in rec_name for illegal_char in illegal_chars):
            msg = "Could not make file name \"{}\" valid. (Illegal characters)"
            msg = msg.format(rec_name)
            raise InvalidNameException(msg, (rec_name, -1))
        
        valid_name = '{}.avi'.format(rec_name)
        if not os.path.exists(valid_name): return valid_name
        
        max_append = 999
        for i in range(max_append):
            valid_name = '{}_{:03}.avi'.format(rec_name, i+1)
            if not os.path.exists(valid_name): return valid_name
        
        msg = "Could not make file name \"{}\" valid.".format(rec_name)
        raise InvalidNameException(msg, (rec_name, max_append))
    
    def _get_valid_name(self, rec_name):
        try: 
            valid_name = self._get_valid_name_(rec_name)
            return valid_name
        except InvalidNameException as e: 
            msg = '\n'.join(textwrap.wrap(str(e), 80))
            bars = ''.join(['-']*80)
            fnc_name = inspect.currentframe().f_code.co_name
            s = ("{}{}{}\n".format(ESC.FAIL,bars,ESC.ENDC),
                 "{}{}{}\n".format(ESC.FAIL,type(e).__name__,ESC.ENDC),
                 "In function: ",
                 "{}{}(){}\n".format(ESC.OKBLUE, fnc_name, ESC.ENDC),
                 "{}{}{}\n".format(ESC.WARNING,  msg, ESC.ENDC),)
            print(''.join(s), flush=True)
            rec_name = input('Please enter a different name: ')
            print("{}{}{}".format(ESC.FAIL,bars,ESC.ENDC), flush=True)
            return self._get_valid_name(rec_name)

    def _make_video(self):
        print("Writing video... ", end='', flush=True)
        with open(os.path.join(self.DIR_PATH, self.TELEMETRY_FILE), 'r') as f:
            telems = self._read_DELIMed(f, 'r')
        if telems == None:
            print("{}No video data found.{}".format(ESC.WARNING, ESC.ENDC),
                  flush = True)
            return
            
        telems = [ast.literal_eval(''.join(t)) for t in telems]
        with open(os.path.join(self.DIR_PATH, self.IMAGE_FILE), 'rb') as f:
            images = self._read_DELIMed(f, 'rb')
        images = [self._decode_bytes(i) for i in images]
        
        with av.open(self.REC_NAME, mode="w") as container:
            steam_is_set = False
            vid_stream = container.add_stream("h264", rate=33)
            vid_stream.pix_fmt = "yuv420p"
            vid_stream.bit_rate = 10_000_000
            vid_stream.codec_context.time_base = Fraction(1, 33)

            epoch = None
            prev_time = -np.inf
            for telem, image in zip(telems, images):
                if telem['Uptime (ms)']==0: continue
                if telem['Video format']=='': continue
                time = telem['Uptime (ms)']
                if epoch is None: epoch = time
                time = 0.001*(time - epoch)
                if time <= prev_time: continue
    
                if not steam_is_set:
                    vid_stream.width = image.shape[1]
                    vid_stream.height = image.shape[0]
                    steam_is_set = True
                
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                frame.pts = int(round(time/vid_stream.codec_context.time_base))
                for packet in vid_stream.encode(frame):
                    container.mux(packet)
                prev_time = copy(time)
        print("{}Done.{}".format(ESC.OKCYAN, ESC.ENDC), flush=True)

    def make_video(self):
        return safe_run(self._make_video)
            