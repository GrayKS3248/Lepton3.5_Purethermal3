# External modules
import cv2
import numpy as np
import av

# Package modules
from detector import Detector
from cmaps import Cmaps 

# Std modules
import os
os.system("")
import ast
import struct
import zlib
import time
from contextlib import ExitStack
from collections import deque
import json
from fractions import Fraction
from copy import copy
import argparse
import traceback
import textwrap

def _safe_run(function, stop_function=None, args=(), stop_args=()):
    try: 
        function(*args)
        return 0
    except BaseException as e:
        if not stop_function is None: stop_function(*stop_args)
        msg = '\n'.join(textwrap.wrap(str(e), 80))
        bars = ''.join(['-']*80)
        s = ("{}{}{}\n".format(Clr.FAIL,bars,Clr.ENDC),
             "{}{}{}\n".format(Clr.FAIL,type(e).__name__,Clr.ENDC),
             "In function: ",
             "{}{}(){}\n".format(Clr.OKBLUE, function.__name__, Clr.ENDC),
             "{}{}{}\n".format(Clr.WARNING,  msg, Clr.ENDC),
             "{}{}{}".format(Clr.FAIL,bars,Clr.ENDC),)
        
        print("{}{}{}".format(Clr.FAIL, bars, Clr.ENDC))
        traceback.print_exc()
        print(''.join(s))
        return -1

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', help="Lepton camera port", 
                        type=int, default=0)
    parser.add_argument('-r', "--record", help="record data stream", 
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-n', "--name", help="name of saved video file", 
                        type=str, default='recording')
    parser.add_argument('-c', "--cmap", help="colormap used in viewer", 
                        default='ironbow', 
                        choices=['afmhot', 'arctic', 'black_hot', 'cividis', 
                                 'ironbow', 'inferno', 'magma',
                                 'outdoor_alert', 'rainbow', 'rainbow_hc',
                                 'viridis', 'white_hot'])
    parser.add_argument('-sf', "--scale-factor", 
                        help="the amount the captured image is scaled by",
                        type=int, default=3)
    parser.add_argument('-eq', "--equalize", 
                        help="apply histogram equalization to image", 
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-d', "--detect", help="if moving fronts are detected", 
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-m', "--multiframe", help="detection type", 
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-f', "--fps", help="target FPS of camera", 
                        type=int, default=None)

    args = parser.parse_args()
    return args

def _read_telem(f):
    data = []
    while True:
        char = f.read(1)
        if len(char)==0: return None
        data.append(char)
        if char == "}": break
    return ast.literal_eval(''.join(data))
    
def _read_tiff(f):
    data = None
    while True:
        bit = f.read(1)
        if data is None: 
            data = bit 
            continue
        data += bit
        if len(data)>3 and data[-1]==42 and data[-2]==73 and data[-3]==73:
            f.seek(-3,1)
            return data[0:-3]
        if bit==b'': 
            return data

def _decode_tiff(tiff_bytes):
    nparr = np.frombuffer(tiff_bytes, np.byte)
    return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

def decode_temperature_dat(dirpath='temp', telemetry_file='telem.json',
                           temperature_file='temperature.dat'):
    data = {'Temperature (C)' : [],
            'Timestamp (ms)' : [],}
    
    paths = [os.path.join(dirpath, telemetry_file),
             os.path.join(dirpath, temperature_file)]
    typ = ['r', 'rb']
    
    with ExitStack() as stack:
        fs = [stack.enter_context(open(f,t)) for (f,t) in zip(paths, typ)]
        
        epoch_ms = None
        prev_time_stamp_ms = -np.inf
        while True:
            telem = _read_telem(fs[0])
            tiff = _read_tiff(fs[1])
            if telem is None or tiff is None: break
        
            if telem['Uptime (ms)']==0: continue
            if telem['Video format']=='': continue
    
            time_ms = telem['Uptime (ms)']
            if epoch_ms is None: epoch_ms = time_ms
            time_stamp_ms = (time_ms - epoch_ms)
            if time_stamp_ms <= prev_time_stamp_ms: continue    
            prev_time_stamp_ms = copy(time_stamp_ms)

            temperature_mK = _decode_tiff(tiff)
            if temperature_mK is None: continue
            temperature_C = 0.01*temperature_mK.astype(float)-273.15
            data['Temperature (C)'].append(temperature_C)
            data['Timestamp (ms)'].append(time_stamp_ms)
            
    return data

def _read_cmask(f):
    data = None
    while True:
        bit = f.read(1)
        if bit==b'': 
            return data
        if data is None: 
            data = bit 
            continue
        data += bit
        if (len(data)>3 and 
            data[-1]==68 and
            data[-2]==78 and
            data[-3]==69 and
            data[-4]==73): 
            return data[:-4]

def _decode_cmask(cmask):
    mask = np.frombuffer(zlib.decompress(cmask), dtype=bool)
    return mask.reshape((120,160))

def decode_mask_dat(dirpath='temp', telemetry_file='telem.json',
                    mask_file='mask.dat'):
    data = {'Mask' : [],
            'Timestamp (ms)' : [],}
    
    paths = [os.path.join(dirpath, telemetry_file),
             os.path.join(dirpath, mask_file)]
    typ = ['r', 'rb']
    
    with ExitStack() as stack:
        fs = [stack.enter_context(open(f,t)) for (f,t) in zip(paths, typ)]
        
        epoch_ms = None
        prev_time_stamp_ms = -np.inf
        while True:
            telem = _read_telem(fs[0])
            cmask = _read_cmask(fs[1])
            if telem is None or cmask is None: break
        
            if telem['Uptime (ms)']==0: continue
            if telem['Video format']=='': continue
    
            time_ms = telem['Uptime (ms)']
            if epoch_ms is None: epoch_ms = time_ms
            time_stamp_s = (time_ms - epoch_ms)
            if time_stamp_s <= prev_time_stamp_ms: continue    
            prev_time_stamp_ms = copy(time_stamp_s)

            mask = _decode_cmask(cmask)
            if mask is None: continue
            data['Mask'].append(mask)
            data['Timestamp (ms)'].append(time_stamp_s)
            
    return data


class ImageShapeException(Exception):
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload
        
    def __str__(self):
        return str(self.message)


class TimeoutException(Exception):
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload
        
    def __str__(self):
        return str(self.message)


class Clr:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Capture():
    def __init__(self, port, target_fps):
        self.PORT = port
        self.IMAGE_SHP = (160, 120)
        try:
            self.TARGET_DT = 1.0 / target_fps
        except:
            self.TARGET_DT = None
            
        self.prev_frame_time = self._time()
        
    def __del__(self):
        self.cap.release()
    
    def __enter__(self):
        self.cap = cv2.VideoCapture(self.PORT + cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.IMAGE_SHP[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.IMAGE_SHP[1]+2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()
    
    def _time(self):
        return cv2.getTickCount()/cv2.getTickFrequency()
        
    def _wait_4_frametime(self):
        try: 
            while True:
                if (self._time()-self.prev_frame_time)>=self.TARGET_DT: return
        except:
            return
        
    def _decode_data(self, raw_data):
        temperature_C = raw_data[:-2] * 0.01 - 273.15
        row_A = raw_data[-2,:]        
        
        telem_revision = '{}.{}'.format(*struct.unpack("<2b", row_A[0]))
        uptime_ms = struct.unpack("<I", row_A[1:3])[0]
        status = struct.unpack("<I", row_A[3:5])[0]
        FFC_desired = status & 8
        if FFC_desired == 0: FFC_desired = "not desired"
        elif FFC_desired == 8: FFC_desired = "desired"
        FFC_state = status & 48
        if FFC_state == 0: FFC_state = "never commanded"
        elif FFC_state == 16: FFC_state = "imminent"
        elif FFC_state == 32: FFC_state = "in progress"
        elif FFC_state == 48: FFC_state = "complete"
        AGC_state = status & 4096
        if AGC_state == 0: AGC_state = "disabled"
        elif AGC_state == 4096: AGC_state = "enabled"
        shutter_lockout = status & 32768
        if shutter_lockout == 0: shutter_lockout = "not locked out"
        elif shutter_lockout == 32768: shutter_lockout = "locked out"
        overtemp_shutdown = status & 1048576
        if overtemp_shutdown == 0: overtemp_shutdown = "not imminent"
        elif overtemp_shutdown == 1048576: overtemp_shutdown = "within 10s"
        serial = struct.unpack("<2Q", row_A[5:13])[0]
        rev = struct.unpack("<6Bh", row_A[13:17])
        GPP_rev = '{}.{}.{}'.format(rev[0], rev[1], rev[2])
        DSP_rev = '{}.{}.{}'.format(rev[3], rev[4], rev[5])
        frame_count = struct.unpack("<I", row_A[20:22])[0]
        FPA_temp_C = struct.unpack("<H", row_A[24])[0]*0.01 - 273.15
        FPA_temp_C = round(FPA_temp_C,2)
        housing_temp_C = struct.unpack("<H", row_A[26])[0]*0.01 - 273.15
        housing_temp_C = round(housing_temp_C,2)
        FPA_temp_C_last_FFC = struct.unpack("<H", row_A[29])[0]*0.01 - 273.15
        FPA_temp_C_last_FFC = round(FPA_temp_C_last_FFC,2)
        uptime_ms_last_FFC =  struct.unpack("<H", row_A[30])[0]
        house_temp_C_last_FFC = struct.unpack("<H", row_A[32])[0]
        house_temp_C_last_FFC = house_temp_C_last_FFC*0.01 - 273.15
        house_temp_C_last_FFC = round(house_temp_C_last_FFC,2)
        AGC_ROI_tlbr = struct.unpack("<4H", row_A[34:38])
        AGC_clip_hi_px_ct = struct.unpack("<H", row_A[38])[0]
        AGC_clip_lo_px_ct = struct.unpack("<H", row_A[39])[0]
        video_format = struct.unpack("<I", row_A[72:74])[0]
        if video_format == 3: video_format = 'RGB888'
        elif video_format == 7: video_format = 'RAW14'
        else: video_format = ''
        mn_temp_C = round(np.min(temperature_C), 2)
        me_temp_C = round(np.mean(temperature_C), 2)
        mx_temp_C = round(np.max(temperature_C), 2)

        telemetry = {'Telemetry version' : telem_revision,
                     'Uptime (ms)' : uptime_ms,
                     'FFC desired' : FFC_desired,
                     'FFC state' : FFC_state,
                     'AGC state' : AGC_state,
                     'Shutter lockout' : shutter_lockout,
                     'Overtemp shutdown' : overtemp_shutdown,
                     'Serial number' : serial,
                     'g++ version' : GPP_rev,
                     'dsp version' : DSP_rev,
                     'Frame count since reboot' : frame_count,
                     'FPA temperature (C)' : FPA_temp_C,
                     'Housing temperature (C)' : housing_temp_C,
                     'FPA temperature at last FFC (C)' : FPA_temp_C_last_FFC,
                     'Uptime at last FFC (ms)' : uptime_ms_last_FFC,
                     'Housing temperature at last FFC' : house_temp_C_last_FFC,
                     'AGC ROI (top left bottom right)' : AGC_ROI_tlbr,
                     'AGC clip high' : AGC_clip_hi_px_ct,
                     'AGC clip low' : AGC_clip_lo_px_ct,
                     'Video format' : video_format,
                     'Frame min temperature (C)' : mn_temp_C,
                     'Frame mean temperature (C)' : me_temp_C,
                     'Frame max temperature (C)' : mx_temp_C,}         

        return temperature_C, telemetry
    
    def read(self):
        self._wait_4_frametime()
        res, im = self.cap.read()
        self.prev_frame_time = self._time()
        
        if im.shape[0]!=self.IMAGE_SHP[1]+2 or im.shape[1]!=self.IMAGE_SHP[0]:
            shp = (im.shape[0]-2,im.shape[1])
            msg = ("Captured image shape {} does not equal "
                   "expected image shape {}. Are you sure the selected "
                   "port is correct? NOTE: If captured image shape is "
                   "(61, 80) the Lepton may be seated incorrectly and you "
                   "should reseat its socket.")
            msg = msg.format(shp, self.IMAGE_SHP)
            raise ImageShapeException(msg, payload=(shp, self.IMAGE_SHP))
        
        if res:
            return self._decode_data(im)
        else:
            return self._decode_data(np.zeros((self.IMAGE_SHP[1]+2,
                                               self.IMAGE_SHP[0]),
                                              dtype=np.uint16))


class Lepton():
    def __init__(self, camera_port, cmap, scale_factor):
        self.PORT = camera_port
        self.CMAP = Cmaps[cmap]
        self.SHOW_SCALE = scale_factor
        self.BUFFER_SIZE = 3
        
        self.detector = Detector()
        
        self.temperature_C_buffer = deque()
        self.telemetry_buffer = deque()
        self.image_buffer = deque()
        self.mask_buffer = deque()
        
        self.frame_number = 0
        self.flag_streaming = False
        self.flag_recording = False
        self.flag_emergency_stop = False
    
    def _read_cap(self):
        temperature_C, telemetry = self.cap.read()
        self.temperature_C_buffer.append(temperature_C)
        self.telemetry_buffer.append(telemetry)
    
    def _detect_front(self, detect_fronts, multiframe):
        if not detect_fronts or len(self.temperature_C_buffer)<1:
            self.mask_buffer.append(None)
            return
        
        if multiframe:
            mask=self.detector.front(self.temperature_C_buffer, 'kmeans')
        else:
            mask=self.detector.front([self.temperature_C_buffer[-1]], 'kmeans')
        self.mask_buffer.append(mask)
    
    def _normalize_temperature(self, temperature_C, alpha=0.0, beta=1.0,
                               equalize=True):
        mn = np.min(temperature_C)
        mx = np.max(temperature_C)
        rn = mx-mn
        if rn==0.0: return temperature_C
        
        norm = (temperature_C-mn) * ((beta-alpha)/(mx-mn)) + alpha
        if not equalize: return norm
        eq = cv2.equalizeHist(np.round(norm*255).astype(np.uint8))
        return eq.astype(float) / 255.0

    def _temperature_2_image(self, equalize):
        image = self._normalize_temperature(self.temperature_C_buffer[-1],
                                            equalize=equalize)
        image = self.CMAP(image)
        image = np.round(255.0*image[:,:,:-1]).astype(np.uint8)
        self.image_buffer.append(image)
    
    def _uptime_str(self):
        telemetry = self.telemetry_buffer[-1]
        hrs = telemetry['Uptime (ms)']/3600000.0
        mns = 60.0*(hrs-np.floor(hrs))
        scs = 60.0*(mns-np.floor(mns))
        mss = 1000.0*(scs - np.floor(scs))
        hrs = int(np.floor(hrs))
        mns = int(np.floor(mns))
        scs = int(np.floor(scs))
        mss = int(np.floor(mss))
        return "{:02d}:{:02d}:{:02d}:{:03d}".format(hrs,mns,scs,mss)
    
    def _temperature_range_str(self):
        telemetry = self.telemetry_buffer[-1]
        mn = '({:0>6.2f})'.format(telemetry['Frame min temperature (C)'])
        i=1
        while mn[i]=='0' and mn[i+1]!='.':
            mn=' {}{}'.format(mn[:i], mn[i+1:])
            i+=1
        me = '| {:0>6.2f} |'.format(telemetry['Frame mean temperature (C)'])
        i=2
        while me[i]=='0' and me[i+1]!='.':
            me=' {}{}'.format(me[:i], me[i+1:])
            i+=1
        mx = '({:0>6.2f})'.format(telemetry['Frame max temperature (C)'])
        i=1
        while mx[i]=='0' and mx[i+1]!='.':
            mx=' {}{}'.format(mx[:i], mx[i+1:])
            i+=1
        return "{} {} {} C".format(mn, me, mx)
    
    def _fps_str(self):
        if len(self.telemetry_buffer)<self.BUFFER_SIZE:
            return 'FPS: ---'
        
        frame_times = []
        for i in range(self.BUFFER_SIZE):
            telemetry = self.telemetry_buffer[i-self.BUFFER_SIZE]
            frame_times.append(telemetry['Uptime (ms)'])
        delta = np.mean(np.diff(frame_times))*0.001
        if delta <= 0.0: return 'FPS: ---'
        return 'FPS: {:.2f}'.format(1.0/delta)
            
    def _telemetrize_image(self, image):
        shp = (image.shape[0]+30,image.shape[1],image.shape[2])
        telimg = np.zeros(shp).astype(np.uint8)
        telimg[:-30,:,:] = image
        
        uptime_pos = (int(np.round(telimg.shape[1]/64)), telimg.shape[0]-10)
        range_pos = (telimg.shape[1]-255, telimg.shape[0]-10)
        fps_pos = (int(np.round(0.5*(range_pos[0]+uptime_pos[0])))+20, 
                   telimg.shape[0]-10)
        
        telimg = cv2.putText(telimg, self._uptime_str(), uptime_pos, 
                             cv2.FONT_HERSHEY_PLAIN , 1, (255,255,255), 1, 
                             cv2.LINE_AA)
        telimg = cv2.putText(telimg, self._temperature_range_str(), range_pos, 
                             cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1,
                             cv2.LINE_AA)
        telimg = cv2.putText(telimg, self._fps_str(), fps_pos,
                             cv2.FONT_HERSHEY_PLAIN , 1, (255,255,255), 1,
                             cv2.LINE_AA)
        
        if self.telemetry_buffer[-1]['FFC state']=='imminent':
            telimg = cv2.rectangle(telimg,(5,5),(35,25),[0,0,0],-1)
            telimg = cv2.putText(telimg, "FFC", (6,21),
                                cv2.FONT_HERSHEY_PLAIN , 1, (255,255,255), 1,
                                cv2.LINE_AA)
        return telimg
    
    def _show(self):
        image = self.image_buffer[-1]
        mask = self.mask_buffer[-1]
        if not mask is None:
            image[mask] = [0,255,0]
        shp = (image.shape[1]*self.SHOW_SCALE, image.shape[0]*self.SHOW_SCALE)
        scaled_image = cv2.resize(image, shp, interpolation=cv2.INTER_LINEAR)
        telem_img = self._telemetrize_image(scaled_image)
        self.image_buffer[-1] = copy(telem_img)
        if self.flag_recording:
            show_img = cv2.circle(telem_img,
                                  (telem_img.shape[1]-10,10),5,[255,0,0],-1)
            cv2.imshow('Recording: FLIR Lepton',
                       cv2.cvtColor(show_img,cv2.COLOR_BGR2RGB))
        else:
            cv2.imshow('FLIR Lepton',cv2.cvtColor(telem_img,cv2.COLOR_BGR2RGB))

    def _esc_pressed(self):
        if cv2.waitKey(1) == 27:
            return True
        return False    

    def _estop_stream(self):
        print(Clr.WARNING+"Emergency stopping stream... "+Clr.ENDC, end="")
        self.flag_emergency_stop = True
        self.flag_streaming = False
        cv2.destroyAllWindows()
        print(Clr.FAIL+"Stopped."+Clr.ENDC)

    def _capture_frame(self, detect_fronts, multiframe, equalize):
        self._read_cap()
        self._detect_front(detect_fronts, multiframe)
        self._temperature_2_image(equalize)
        self._show()
        self.frame_number += 1
        self.flag_streaming = not self._esc_pressed()
        
    def _trim_buffers(self):
        while len(self.temperature_C_buffer) > self.BUFFER_SIZE:
            self.temperature_C_buffer.popleft()
        while len(self.telemetry_buffer) > self.BUFFER_SIZE:
            self.telemetry_buffer.popleft()
        while len(self.image_buffer) > self.BUFFER_SIZE:
            self.image_buffer.popleft()
        while len(self.mask_buffer) > self.BUFFER_SIZE:
            self.mask_buffer.popleft()

    def _stream(self, fps, detect_fronts, multiframe, equalize):
        with Capture(self.PORT, fps) as self.cap:
            
            if self.flag_emergency_stop:
                self._estop_stream()
                return
            
            self.flag_streaming = True
            while self.flag_streaming:
                
                if self.flag_emergency_stop:
                    self._estop_stream()
                    return
                
                self._capture_frame(detect_fronts, multiframe, equalize)
                self._trim_buffers()
            cv2.destroyAllWindows()
        
    def _estop_record(self):
        self._estop_stream()
        print(Clr.WARNING+"Emergency stopping record... "+Clr.ENDC, end="")
        self.flag_emergency_stop = True
        self.flag_recording = False
        print(Clr.FAIL+"Stopped."+Clr.ENDC)
    
    def _min_buf_len(self):
        return min(len(self.temperature_C_buffer), len(self.telemetry_buffer),
                   len(self.image_buffer), len(self.mask_buffer),)
    
    def _write_frame(self, T_file, t_file, i_file, m_file,
                     ignore_buf_min=False):
        if self._min_buf_len()<=self.BUFFER_SIZE and not ignore_buf_min: 
            return
        
        temperature_C = self.temperature_C_buffer[0]
        temperature_mK = np.round(100.*(temperature_C+273.15))
        temperature_mK = temperature_mK.astype(np.uint16)
        encode_param = [int(cv2.IMWRITE_TIFF_COMPRESSION), 
                        cv2.IMWRITE_TIFF_COMPRESSION_LZW]
        T_img = cv2.imencode('.tiff', temperature_mK, encode_param)[1]
        T_img = T_img.tobytes()
        T_file.write(T_img)
        T_file.write(b'\0')
        
        telemetry=self.telemetry_buffer[0]
        json.dump(telemetry, t_file)
        t_file.write('\n')
        
        image=cv2.imencode('.png', self.image_buffer[0])[1].tobytes()
        i_file.write(image)
        
        mask=self.mask_buffer[0]
        if not mask is None: 
            m_file.write(zlib.compress(mask.tobytes()))
            m_file.write(b'IEND')
        
        self.temperature_C_buffer.popleft()
        self.telemetry_buffer.popleft()
        self.image_buffer.popleft()
        self.mask_buffer.popleft()
    
    def _record(self, fps, detect_fronts, multiframe, equalize):
        dirname = 'temp'
        os.makedirs(dirname, exist_ok=True)
        fnames = ['temperature.dat', 'telem.json', 'image.dat', 'mask.dat']
        typ = ['wb', 'w', 'wb', 'wb']
        
        with (Capture(self.PORT, fps) as self.cap,
              open(os.path.join(dirname, fnames[0]), typ[0]) as T_file,
              open(os.path.join(dirname, fnames[1]), typ[1]) as t_file,
              open(os.path.join(dirname, fnames[2]), typ[2]) as i_file,
              open(os.path.join(dirname, fnames[3]), typ[3]) as m_file,):
            
            if self.flag_emergency_stop:
                self._estop_record()
                return
            
            self.flag_streaming = True
            self.flag_recording = True
            while self.flag_streaming:
                
                if self.flag_emergency_stop:
                    self._estop_record()
                    return
                
                self._capture_frame(detect_fronts, multiframe, equalize)
                self._write_frame(T_file, t_file, i_file, m_file)        
            cv2.destroyAllWindows()
            
            while self._min_buf_len() > 0:
                self._write_frame(T_file, t_file, i_file, m_file,
                                  ignore_buf_min=True)
            self.recording=False
    
    def _wait_until(self, condition, timeout_ms, dt_ms):
        epoch_s = time.time()
        timeout_s = 0.001*timeout_ms
        dt_s = 0.001*dt_ms
        while not condition():
            if (time.time()-epoch_s) > timeout_s:
                string = "Function _wait_until({}) timed out at {} ms."
                raise TimeoutException(string.format(condition.__name__, 
                                                     timeout_ms), 
                                       timeout_s)
            time.sleep(dt_s)
            if self.flag_emergency_stop: break

    def _ready_to_record(self):
        return self.flag_streaming or len(self.image_buffer)>1
    
    def emergency_stop(self):
        if not self.flag_emergency_stop:
            self.flag_emergency_stop = True
            st="WARNING: Emergency stop command received. "
            print(Clr.FAIL+st+Clr.ENDC)        
    
    def is_streaming(self):
        return self.flag_streaming
        
    def start_stream(self, fps=None, detect_fronts=False, multiframe=True, 
                     equalize=False):
        return _safe_run(self._stream, self._estop_stream,
                         args=(fps, detect_fronts, multiframe, equalize, ))    

    def wait_until_stream_active(self, timeout_ms=5000.0, dt_ms=10.0):
        return _safe_run(self._wait_until, args=(self._ready_to_record,
                                                 timeout_ms, dt_ms))    

    def is_recording(self):
        return self.flag_recording

    def start_record(self, fps=None, detect_fronts=False, multiframe=True, 
                     equalize=False):
        return _safe_run(self._record, self._estop_record, 
                         args=(fps, detect_fronts, multiframe, equalize))
    
    def get_temperature(self):
        if len(self.temperature_C_buffer) > 0:
            return self.temperature_C_buffer[-1]
        return None
    
    def get_front_mask(self):
        if len(self.mask_buffer) > 0:
            return self.mask_buffer[-1]
        return None
    
    def get_time(self):
        if len(self.telemetry_buffer) > 0:
            t = self.telemetry_buffer[-1]['Uptime (ms)']*0.001
            return round(t, 3)
        return None
    
    def get_frame_number(self):
        return self.frame_number
    

class Videowriter():
    def __init__(self):
        pass
    
    def _read_png_chunk(self, f):
        preamble = f.read(8)
        lng, typ = struct.unpack('>I4s', preamble)
        dat = f.read(lng)
        chksum = zlib.crc32(dat, zlib.crc32(struct.pack('>4s', typ)))
        postamble = f.read(4)
        crc, = struct.unpack('>I', postamble)
        if crc != chksum:
            raise Exception('chunk checksum failed {} != {}'.format(crc,
                chksum))
        return typ, preamble+dat+postamble
    
    def _read_png(self, f):
        png = f.read(8)
        if len(png) == 0: return None
        while True:
            chunk_type, chunk = self._read_png_chunk(f)
            png = png + chunk
            if chunk_type == b'IEND': break
        return png
    
    def _decode_png(self, png_bytes):
        nparr = np.frombuffer(png_bytes, np.byte)
        return cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
    
    def _read_telem(self, f):
        data = []
        while True:
            char = f.read(1)
            if len(char)==0: return None
            data.append(char)
            if char == "}": break
        return ast.literal_eval(''.join(data))
    
    def _make_video(self, rec_name, dirpath, telemetry_file, image_file):
        paths = [os.path.join(dirpath, telemetry_file),
                 os.path.join(dirpath, image_file)]
        
        typ = ['r', 'rb']
        with ExitStack() as stack:
            valid_name = '{}.avi'.format(rec_name)
            if os.path.exists(valid_name):
                for i in range(999):
                    valid_name = '{}_{:03}.avi'.format(rec_name, i+1)
                    if not os.path.exists(valid_name): break
            
            fs = [stack.enter_context(open(f,t)) for (f,t) in zip(paths, typ)]
            fs.append(stack.enter_context(av.open(valid_name, mode="w")))
            
            steam_is_set = False
            vid_stream = fs[-1].add_stream("h264", rate=1000)
            vid_stream.pix_fmt = "yuv420p"
            vid_stream.codec_context.time_base = Fraction(1, 1000)
            
            epoch = None
            prev_time = -np.inf
            while True:
                telem = self._read_telem(fs[0])
                png = self._read_png(fs[1])
                if telem is None or png is None: break
            
                if telem['Uptime (ms)']==0: continue
                if telem['Video format']=='': continue
                time = telem['Uptime (ms)']
                if epoch is None: epoch = time
                time = 0.001*(time - epoch)
                if time <= prev_time: continue

                image = self._decode_png(png)
                if not steam_is_set:
                    vid_stream.width = image.shape[1]
                    vid_stream.height = image.shape[0]
                    steam_is_set = True
                
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                frame.pts = int(round(time/vid_stream.codec_context.time_base))
                for packet in vid_stream.encode(frame):
                    fs[-1].mux(packet)
                prev_time = copy(time)

    def make_video(self, rec_name='recording', dirpath='temp', 
                   telemetry_file='telem.json', image_file='image.dat'):
        return _safe_run(self._make_video, 
                         args=(rec_name, dirpath, telemetry_file, image_file))

if __name__ == "__main__":   
    args = _parse_args()
    
    if not args.fps is None and args.fps < 5:
        wstr="Target FPS set below 5 can result in erroneous video rendering."
        print(Clr.WARNING+'WARNING: '+wstr+Clr.ENDC)

    lepton = Lepton(args.port, args.cmap, args.scale_factor)
    if not args.record:
        print("Streaming...")
        res = lepton.start_stream(fps=args.fps, detect_fronts=args.detect, 
                            multiframe=args.multiframe, equalize=args.equalize)
        if res >= 0: 
            print("Stream done.")
        
    else:
        print("Recording...")
        res = lepton.start_record(fps=args.fps, detect_fronts=args.detect,
                            multiframe=args.multiframe, equalize=args.equalize)
        if res >= 0:
            print('Record done.')
            writer = Videowriter()
            print('Writing video...')
            writer.make_video(rec_name=args.name)
            print('Writing done.')
