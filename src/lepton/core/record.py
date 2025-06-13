# Std modules
import os
import zlib
from fractions import Fraction
import textwrap
import inspect

# Package modules
from lepton.exceptions import InvalidNameException
from lepton.misc.utilities import safe_run, ESC

# External modules
import cv2
import numpy as np
from numpy.ma import masked_array
import av


def encode_msg(msg, typ):
    # Ensure all data is of type numpy array
    data = np.array([msg])
    if len(data.shape) > 1:
        data = np.squeeze(data, axis=0)
    
    # Define data types
    uints = ['B', 'H', 'L', 'Q']
    ints = ['b', 'h', 'l', 'q']
    floats = ['e', 'f', 'd']
    bools = ['?']
    strs = ['U']
    all_types = uints + ints + floats + bools + strs
    
    # Get the target type character
    typ_char = np.dtype(typ).char
    if not typ_char in all_types:
        return b''
    
    # Split strings into chars
    if data.dtype.char == 'U':
        temp = []
        length = max([len(d) for d in data])
        for d in data:
            char_list = list(d)
            while len(char_list) < length:
                char_list.append(' ')
            temp.append(char_list)
        data = np.array(temp) 
    
    # Temporarily replace nan values with 0 so rounding and casting will work
    if data.dtype.char in floats:
        nan_mask = np.isnan(data)
        data[nan_mask] = 0
    else:
        nan_mask = np.zeros(data.shape, dtype=bool)

    # Round the data if going from float to int and type cast
    if data.dtype.char in floats and typ_char in uints+ints:
        data = np.round(data)
        
    # Convert to unicode code point if going from string to anything else
    if data.dtype.char == 'U' and typ_char != 'U':
        data = np.vectorize(ord)(data)
        
    # Type cast
    data = data.astype(typ)
    # try:
    #     with np.errstate(all='raise'):
    #         data = data.astype(typ)
    # except:
    #     data = data.astype(typ)
    #     try:
    #         print('-------------------')
    #         print(msg)
    #         print('-------------------')
    #     except:
    #         pass
    #     try:
    #         print('-------------------')
    #         print(data)
    #         print('-------------------')
    #     except:
    #         pass
    #     try:
    #         print('-------------------')
    #         print(typ)
    #         print('-------------------')
    #     except:
    #         pass
    
    # Replace any previous nans with typ min
    if np.any(nan_mask):
        if typ_char in ints:
            data[nan_mask] = np.iinfo(typ).min
        elif typ_char in floats:
            data[nan_mask] = np.finfo(typ).min
        else:
            return b''
    
    # Get the data shape and dimension
    shape = np.array(data.shape, dtype=np.uint16)
    dim = np.uint8(len(shape))
    
    # Flatten the data for encoding
    data = data.flatten()
    
    # Get the type code
    typ_code = dict(zip(all_types, np.arange(len(all_types), dtype=np.uint8)))
    code = typ_code[typ_char]
    
    # Build and compress the message
    msg = dim.tobytes() + shape.tobytes() + code.tobytes() + data.tobytes()
    msg = zlib.compress(msg)
    return msg

def decode_msg(msg):
    # Decompress and read the header
    data = zlib.decompress(msg)
    dim = np.frombuffer(data[0:1],np.uint8)[0]
    shape = np.frombuffer(data[1:1+2*dim],np.uint16)
    code = np.frombuffer(data[1+2*dim:2+2*dim],np.uint8)[0]
    
    # Define data types
    uints = ['B', 'H', 'L', 'Q']
    ints = ['b', 'h', 'l', 'q']
    floats = ['e', 'f', 'd']
    bools = ['?']
    strs = ['U1']
    all_types = uints + ints + floats + bools + strs
    
    # Build type code dictionary
    typ_code = dict(zip(np.arange(len(all_types), dtype=np.uint8), all_types))
    typ_char = typ_code[code]
    
    # Decode the message
    data = np.frombuffer(data[2+2*dim:],typ_char).reshape(shape)
    data = data.copy()
    
    # If type U1, convert from char list to string
    if typ_char == 'U1':
        data = np.array([''.join(d).rstrip() for d in data])
    
    # Replace nan values with masked array
    if typ_char in ints:
        nan_mask = data == np.iinfo(typ_char).min
        if np.any(nan_mask):
            data = masked_array(data, nan_mask)
    elif typ_char in floats:
        nan_mask = data == np.finfo(typ_char).min
        if np.any(nan_mask):
            data = masked_array(data, nan_mask)
    
    # Return the decoded data
    return data

def encode_frame_data(frame_data, types):
    msgs = []
    for d, t in zip(frame_data, types):
        msgs.append(encode_msg(d, t))
    return tuple(msgs)

def decode_frame_data(frame_data):
    msgs = []
    for d in frame_data:
        msgs.append(decode_msg(d))
    return tuple(msgs)
    
def _read_chunked(path):
    with open(path, 'rb') as f:
        data = []
        while True:
            msg_len = f.read(8)
            if len(msg_len) < 8:
                break
            msg = f.read(int(msg_len))
            data.append(decode_msg(msg))
    if len(data) > 0: return data
    else: return None

def decode_recording_data(dirpath='rec_data',
                          frame_number_file='frame_number.dat',
                          frame_time_file='frame_time.dat',
                          temperature_file='temperature.dat',
                          warped_temperature_file='warped_temperature.dat',
                          telemetry_file='telem.dat',
                          image_file='image.dat',
                          mask_file='mask.dat',
                          warped_mask_file='warped_mask.dat'):
    print("Decoding raw data... ", end='', flush=True)
    _frame_number = _read_chunked(os.path.join(dirpath, frame_number_file))
    _frame_time = _read_chunked(os.path.join(dirpath, frame_time_file))
    _temperature = _read_chunked(os.path.join(dirpath, temperature_file))
    _warped_temperature = _read_chunked(os.path.join(dirpath,
                                                     warped_temperature_file))
    _telemetry = _read_chunked(os.path.join(dirpath, telemetry_file))
    _image = _read_chunked(os.path.join(dirpath, image_file))
    _mask = _read_chunked(os.path.join(dirpath, mask_file))
    _warped_mask = _read_chunked(os.path.join(dirpath, warped_mask_file))
    
    if (_frame_number is None or _frame_time is None or _temperature is None or
        _warped_temperature is None or _telemetry is None or _image is None or 
        _mask is None or _warped_mask is None):
        print(ESC.warning('No or incomplete video data found.'), flush=True)
        return None
    
    zipped = zip(_frame_number, _frame_time, _temperature, _warped_temperature,
                 _telemetry, _image, _mask, _warped_mask)
    frame_number = []
    frame_time_s = []
    temperature_C = []
    warped_temperature_C = []
    telemetry = []
    image = []
    mask = []
    warped_mask = []
    for fn, ft, T, wT, t, i, m, wm in zipped:
        frame_number.append(tuple([int(d) for d in fn]))
        frame_time_s.append(tuple([round(0.001*float(d),3) for d in ft]))
        temperature_C.append(0.01*T.astype(float)-273.15)
        warped_temperature_C.append(0.01*wT.astype(float)-273.15)
        telemetry.append(eval(t[0]))
        image.append(i)
        mask.append(m)
        warped_mask.append(wm)
    
    data = {'Frame Number (Lepton, Capture)' : frame_number,
            'Frame Time (s) (Lepton, Wall)' : frame_time_s,
            'Temperature (C)' : temperature_C,
            'Warped Temperature (C)' : warped_temperature_C,
            'Telemetry' : telemetry,
            'Image' : image,
            'Mask' : mask,
            'Warped Mask' : warped_mask,}
            
    print("{}Done.{}".format(ESC.OKCYAN, ESC.ENDC), flush=True)
    return data        


class Videowriter():
    def __init__(self, rec_name='recording', dirpath='rec_data', 
                 telemetry_file='telem.json', image_file='image.dat'):
        self.REC_NAME = self._get_valid_name(rec_name)
        self.DIR_PATH = dirpath
        self.TELEMETRY_FILE = telemetry_file
        self.IMAGE_FILE = image_file
    
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

    def _make_video(self, playback_speed):
        frames = decode_recording_data(dirpath=self.DIR_PATH)
        
        print("Writing video... ", end='', flush=True)
        with av.open(self.REC_NAME, mode="w") as container:
            rate = min(120, max(20, int(np.round(30*playback_speed))))
            vid_stream = container.add_stream("h264", rate=rate)
            vid_stream.pix_fmt = "yuv420p"
            vid_stream.bit_rate = 10_000_000
            vid_stream.codec_context.time_base = Fraction(1, rate)
            vid_stream.width = frames['Image'][0].shape[1]
            vid_stream.height = frames['Image'][0].shape[0]

            # If LOS occured, lepton frame number and/or time will be 
            # non-monotonic. In this case, must use wall time.
            frm = np.array(frames['Frame Number (Lepton, Capture)'])[:,0]
            time = np.array(frames['Frame Time (s) (Lepton, Wall)'])[:,0]
            if np.any(np.diff(frm) <= 0.0) or np.any(np.diff(time) <= 0.0):
                time = np.array(frames['Frame Time (s) (Lepton, Wall)'])[:,1]
            epoch_time = time - time[0]
            
            speed_adjusted_epoch_time = epoch_time / playback_speed
            for t, image in zip(speed_adjusted_epoch_time, frames['Image']):
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                frame.pts = int(round(t/vid_stream.codec_context.time_base))
                for packet in vid_stream.encode(frame):
                    container.mux(packet)
        print("{}Done.{}".format(ESC.OKCYAN, ESC.ENDC), flush=True)
        return frames

    def make_video(self, playback_speed=1.0):
        return safe_run(self._make_video, args=(playback_speed,))
            