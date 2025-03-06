# External modules
import cv2
import numpy as np
import av
from scipy.signal import find_peaks

# Package modules
from detector import Detector
from cmaps import Cmaps 

# Std modules
import os
import ast
import struct
import zlib
import time
from collections import deque
import json
from fractions import Fraction
from copy import copy
import argparse
import traceback
import textwrap
import inspect
from dataclasses import dataclass
from threading import Lock


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
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-n', "--name", help="name of saved video file", 
                        type=str, default="recording")
    parser.add_argument('-c', "--cmap", help="colormap used in viewer", 
                        default='black_hot', 
                        choices=['afmhot', 'arctic', 'black_hot', 'cividis', 
                                 'ironbow', 'inferno', 'magma',
                                 'outdoor_alert', 'rainbow', 'rainbow_hc',
                                 'viridis', 'white_hot'])
    parser.add_argument('-sf', "--scale-factor", 
                        help="the amount the captured image is scaled by",
                        type=int, default=3)
    parser.add_argument('-eq', "--equalize", 
                        help="apply histogram equalization to image", 
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-d', "--detect", help="if moving fronts are detected", 
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-m', "--multiframe", help="detection type", 
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('-f', "--fps", help="target FPS of camera", 
                        type=int, default=None)

    args = parser.parse_args()
    return args

def decode_recording_data(dirpath='temp', telemetry_file='telem.json',
                          temperature_file='temperature.dat',
                          mask_file='mask.dat'):
    _read_DELIMed = Videowriter()._read_DELIMed
    _decode_bytes = Videowriter()._decode_bytes
    
    with open(os.path.join(dirpath, telemetry_file), 'r') as f:
        telems = _read_DELIMed(f, 'r')
    telems=[ast.literal_eval(''.join(t)) for t in telems]
    timestamps_ms=[t['Uptime (ms)'] for t in telems]
    
    with open(os.path.join(dirpath, temperature_file), 'rb') as f:
        temperatures_mK = _read_DELIMed(f, 'rb')
    temperatures_C = [0.01*_decode_bytes(t)-273.15 for t in temperatures_mK]
    
    with open(os.path.join(dirpath, mask_file), 'rb') as f:
        masks = _read_DELIMed(f, 'rb')
    masks = [_decode_bytes(m, compressed=True) for m in masks]
    
    data = {'Temperature (C)' : temperatures_C,
            'Mask' : masks,
            'Timestamp (ms)' : timestamps_ms,
            'Telemetry': telems}
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


class InvalidNameException(Exception):
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload
        
    def __str__(self):
        return str(self.message)


class BufferLengthException(Exception):
    def __init__(self, message, payload=None):
        self.message = message
        self.payload = payload
        
    def __str__(self):
        return str(self.message)


@dataclass
class Clr:
    HEADER: str = '\033[95m'
    OKBLUE: str = '\033[94m'
    OKCYAN: str = '\033[96m'
    OKGREEN: str = '\033[92m'
    WARNING: str = '\033[93m'
    FAIL: str = '\033[91m'
    ENDC: str = '\033[0m'
    BOLD: str = '\033[1m'
    UNDERLINE: str = '\033[4m'


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
        temp_C = raw_data[:-2] * 0.01 - 273.15
        
        row_A = raw_data[-2,:80]
        row_B = raw_data[-2,80:]
        row_C = raw_data[-1,:80]
        adat=struct.unpack("<bbIIQ8x6Bh6xI4xHxxH4xHHxxHxx6H64xI12x", row_A)
        bdat=struct.unpack("<38x8H106x", row_B)
        cdat=struct.unpack("<10x5H8xHH12x4H44x?x9H44x", row_C)
        
        status = ['', ]*5
        if adat[3] & 8 == 0: status[0] = "not desired"
        elif adat[3] & 8 == 8: status[0] = "desired"
        
        if adat[3] & 48 == 0: status[1] = "never commanded"
        elif adat[3] & 48 == 16: status[1] = "imminent"
        elif adat[3] & 48 == 32: status[1] = "in progress"
        elif adat[3] & 48 == 48: status[1] = "complete"
        
        if adat[3] & 4096 == 0: status[2] = "disabled"
        elif adat[3] & 4096 == 4096: status[2] = "enabled"
        
        if adat[3] & 32768 == 0: status[3] = "not locked out"
        elif adat[3] & 32768 == 32768: status[3] = "locked out"
        
        if adat[3] & 1048576 == 0: status[4] = "not imminent"
        elif adat[3] & 1048576 == 1048576: status[4] = "within 10s"

        video_format = ''
        if adat[24] == 3: video_format = 'RGB888'
        elif adat[24] == 7: video_format = 'RAW14'
        
        gain_mode = ''
        if cdat[0] == 0: gain_mode = 'high'
        elif cdat[0] == 1: gain_mode = 'low'
        elif cdat[0] == 2: gain_mode = 'auto'
        
        eff_gain_mode = ''
        if cdat[1] == 0: eff_gain_mode = 'high'
        elif cdat[1] == 1: eff_gain_mode = 'low'
        if cdat[0] != 2: eff_gain_mode = 'not in auto mode'
        
        desired_gain_mode = ''
        if cdat[2] == 0: desired_gain_mode = gain_mode
        elif cdat[2] == 1 and cdat[0] == 0: desired_gain_mode = 'low'
        elif cdat[2] == 1 and cdat[0] == 1: desired_gain_mode = 'high'
        
        telemetry = {
            'Telemetry version':'{}.{}'.format(adat[0], adat[1]),
            'Uptime (ms)':adat[2],
            'FFC desired':status[0],
            'FFC state':status[1],
            'AGC state':status[2],
            'Shutter lockout':status[3],
            'Overtemp shutdown':status[4],
            'Serial number':adat[4],
            'g++ version':'{}.{}.{}'.format(adat[5],adat[6],adat[7]),
            'dsp version':'{}.{}.{}'.format(adat[9],adat[10],adat[11]),
            'Frame count since reboot':adat[12],
            'FPA temperature (C)':round(adat[13]*0.01 - 273.15, 2),
            'Housing temperature (C)':round(adat[14]*0.01 - 273.15, 2),
            'FPA temperature at last FFC (C)':round(adat[15]*0.01-273.15, 2),
            'Uptime at last FFC (ms)':adat[16],
            'Housing temperature at last FFC':round(adat[17]*0.01-273.15,2),
            'AGC ROI (top left bottom right)':adat[18:22],
            'AGC clip high':adat[22],
            'AGC clip low':adat[23],
            'Video format':video_format,
            'Frame min temperature (C)':float(round(np.min(temp_C),2)),
            'Frame mean temperature (C)':float(round(np.mean(temp_C), 2)),
            'Frame max temperature (C)':float(round(np.max(temp_C), 2)),
            'Assumed emissivity':round(bdat[0]/8192,2),
            'Assumed background temperature (C)':round(0.01*bdat[1]-273.15,2),
            'Assumed atmospheric transmission':round(bdat[2]/8192,2),
            'Assumed atmospheric temperature (C)':round(0.01*bdat[3]-273.15,2),
            'Assumed window transmission':round(bdat[4]/8192,2),
            'Assumed window reflection':round(bdat[5]/8192,2),
            'Assumed window temperature (C)':round(0.01*bdat[6]-273.15,2),
            'Assumed reflected temperature (C)':round(0.01*bdat[7]-273.15,2),
            'Gain mode':gain_mode,
            'Effective gain mode':eff_gain_mode,
            'Desired gain mode':desired_gain_mode,
            'Temperature switch high gain to low gain (C)':cdat[3],
            'Temperature switch low gain to high gain (C)':cdat[4],
            'Population switch high gain to low gain (%)':cdat[5],
            'Population switch low gain to high gain (%)':cdat[6],
            'Gain mode ROI (top left bottom right)':cdat[7:11],
            'TLinear enabled':str(cdat[11]),
            'TLinear resolution':round(-0.09*cdat[12]+0.1,2),
            'Spotmeter max temperature (C)':round(0.01*cdat[13]-273.15,2),
            'Spotmeter mean temperature (C)':round(0.01*cdat[14]-273.15,2),
            'Spotmeter min temperature (C)':round(0.01*cdat[15]-273.15,2),
            'Spotmeter population (px)':cdat[16],
            'Spotmeter ROI (top left bottom right)':cdat[17:],}
        return temp_C, telemetry
    
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
        self.WINDOW_NAME = 'Lepton 3.5 on Purethermal 3'
        self.LOCK = Lock()
        
        self.detector = Detector()
        
        self.temperature_C_buffer = deque()
        self.telemetry_buffer = deque()
        self.image_buffer = deque()
        self.mask_buffer = deque()
        self.frame_number = 0
        
        self.flag_streaming = False
        self.flag_recording = False
        self.flag_emergency_stop = False
        self.flag_focus_box = False
        self.flag_modding_AR = True
        
        self.focus_box_AR = 1.33333333
        self.focus_box_size = 0.50
        self.focus_box = [(), (), (), ()]
        self.subject_quad = [(np.nan,np.nan), (np.nan,np.nan), 
                             (np.nan,np.nan), (np.nan,np.nan)]
        self.subject_next_vert = (np.nan,np.nan)
        self.homography = None
        self.inv_homography = None

    def _mouse_callback(self, event, x, y, flags, param):
        if not self.flag_focus_box: return
        
        if event == cv2.EVENT_MOUSEWHEEL and self.homography is None:
            if flags > 0:
                if self.flag_modding_AR:
                    self.focus_box_AR += 0.01
                else:
                    self.focus_box_size += 0.01
            else:
                if self.flag_modding_AR:
                    self.focus_box_AR -= 0.01
                else:
                    self.focus_box_size -= 0.01
            self.focus_box_size = np.clip(self.focus_box_size, 0.0, 1.0)
            self.focus_box_AR = np.clip(self.focus_box_AR, 0.0, 
                                        1.333333333/self.focus_box_size)

        if event == cv2.EVENT_RBUTTONDOWN and self.homography is None:
            self.flag_modding_AR = not self.flag_modding_AR
                
        if event == cv2.EVENT_LBUTTONDOWN:
            if (np.nan, np.nan) in self.subject_quad:
                insert_at = self.subject_quad.index((np.nan, np.nan))
                self.subject_quad[insert_at] = (x,y)
            
        if event == cv2.EVENT_MOUSEMOVE:
            self.subject_next_vert = np.array([x,y])
    
    def _warped_element(self, buffer, return_buffer=False):
        is_warped = self.flag_focus_box and not self.homography is None
        if not is_warped and return_buffer: return list(buffer)
        if not is_warped and not return_buffer: return copy(buffer[-1])
        
        buffer_len = len(buffer)
        warped_buffer = []
        for i in range(buffer_len):
            if i > 2: break
            element = copy(buffer[buffer_len-1-i])
            shp = (element.shape[1]*self.SHOW_SCALE,
                   element.shape[0]*self.SHOW_SCALE)
            element = cv2.resize(element, shp)
            element = cv2.warpPerspective(element, self.homography, shp)
            (l,t), (r,b) = self.focus_box[0], self.focus_box[2]
            if not return_buffer: return element[t:b+1,l:r+1]
            warped_buffer.append(element[t:b+1,l:r+1])
        warped_buffer.reverse()
        return warped_buffer
    
    def _detect_front(self, detect_fronts, multiframe):
        if not detect_fronts or len(self.temperature_C_buffer)<1:
            self.mask_buffer.append(None)
            return
        
        if multiframe:
            temps = self._warped_element(self.temperature_C_buffer,
                                         return_buffer=True)
        else:
            temps = [self._warped_element(self.temperature_C_buffer,
                                         return_buffer=False)]
        mask = self.detector.front(temps, 'kmeans')
        
        if self.flag_focus_box and not self.inv_homography is None:
            shp = (120*self.SHOW_SCALE,160*self.SHOW_SCALE)
            fmask = np.zeros(shp)
            (l,t), (r,b) = self.focus_box[0], self.focus_box[2]
            fmask[t:b+1,l:r+1] = mask.astype(float)
            fmask = cv2.warpPerspective(fmask, self.inv_homography, shp[::-1])
            mask = cv2.resize(fmask, (160,120)) >= 0.25
        self.mask_buffer.append(mask)
    
    def _normalize_temperature(self, temperature_C, alpha=0.0, beta=1.0,
                               equalize=True):
        mn = np.min(temperature_C)
        mx = np.max(temperature_C)
        rn = mx - mn
        if rn==0.0: return np.zeros(temperature_C.shape)
        norm = (temperature_C-mn) * ((beta-alpha)/(mx-mn)) + alpha
        if not equalize: return norm
        
        quantized = np.round(norm*255).astype(np.uint8)
        hist = cv2.calcHist([quantized.flatten()],[0],None,[256],[0,256])
        P = (hist / 19200.0).flatten()
        median_hist =  cv2.medianBlur(P, 3).flatten()
        F = median_hist[median_hist>0]
        local_maxizers = find_peaks(F)[0]
        global_maximizer = np.argmax(F)
        T = np.median(F[local_maxizers[local_maxizers>=global_maximizer]])
        P[P>T] = T
        FT = np.cumsum(P)
        DT = np.floor(255*FT/FT[-1]).astype(np.uint8)
        eq = DT[quantized] / 255.0
        return  eq

    def _temperature_2_image(self, equalize):
        image = self._normalize_temperature(self.temperature_C_buffer[-1],
                                            equalize=equalize)
        image = 255.0 * self.CMAP(image)[:,:,:-1]
        image = np.round(image).astype(np.uint8)
        self.image_buffer.append(image)
    
    def _draw_subject_quad(self, image):
        lines = []
        for i in range(4):
            j = (i+1) % 4
            lines.append([self.subject_quad[i], self.subject_quad[j]])
        lines = np.array(lines)
        
        next_vert_at = np.all(np.isnan(lines[:,1,:]),axis=1)
        if any(next_vert_at):
            next_vert_at = np.argmax(next_vert_at)
            lines[next_vert_at,1,:] = self.subject_next_vert
        
        roi_image = copy(image)
        for i, line in enumerate(lines):
            if np.any(np.isnan(line)) and i!=3: break
            if i==3 and np.any(np.isnan(line)):
                srt = np.round(lines[i-1][1]).astype(int)
            else:
                srt = np.round(line[0]).astype(int)
            end = np.round(line[1]).astype(int)
            if all(srt==end): continue
            roi_image = cv2.line(roi_image, srt, end, (255,0,255), 1) 
            
        return roi_image
    
    def _draw_focus_box(self, image, quad_incomplete):
        img_h, img_w = image.shape[0], image.shape[1] 
        box_h = int(np.round(self.focus_box_size*img_h))
        box_w = int(np.round(self.focus_box_AR*box_h))
        l = int(0.5*(img_w - box_w))
        t = int(0.5*(img_h - box_h))
        r = l + box_w - 1
        b = t + box_h - 1
        self.focus_box = [(l,t),(l,b),(r,b),(r,t)]
        
        color = [0,255,255] if quad_incomplete else [255,0,255]
        fb_image = cv2.rectangle(image, self.focus_box[0], self.focus_box[2],
                                 color, 1)
        if not quad_incomplete: return fb_image
        
        cnr=[i for i,s in enumerate(self.subject_quad) if s!=(np.nan, np.nan)]
        cnr = len(cnr)
        if cnr < 4:
            fb_image = cv2.circle(fb_image, self.focus_box[cnr], 
                                  3, [255,0,255], -1)
        
        if self.flag_modding_AR:
            txt = 'AR: {:.2f}'.format(self.focus_box_AR)
            fb_image = cv2.rectangle(fb_image,(l+2,t+2),(l+75,t+15),[0,0,0],-1)
        else:
            txt = 'Size: {:.2f}'.format(self.focus_box_size)
            fb_image = cv2.rectangle(fb_image,(l+2,t+2),(l+89,t+15),[0,0,0],-1)
        fb_image = cv2.putText(fb_image, txt, (l+4,t+14),
                               cv2.FONT_HERSHEY_PLAIN , 1, (255,255,255), 1,
                               cv2.LINE_AA)
        
        return fb_image
    
    def _focus_box(self, image):
        if not self.flag_focus_box: return image, False
        
        quad_incomplete = np.any(np.isnan(self.subject_quad))
        if quad_incomplete:
            quad_image = self._draw_subject_quad(image)
            return self._draw_focus_box(quad_image, quad_incomplete), False
        
        if self.homography is None:
            xs = np.array(self.subject_quad)
            ys = np.array(self.focus_box)
            self.homography, _ = cv2.findHomography(xs, ys)
            self.inv_homography = np.linalg.inv(self.homography)

        shp = (image.shape[1], image.shape[0])
        warped_image = cv2.warpPerspective(image, self.homography, shp)
        return self._draw_focus_box(warped_image, quad_incomplete), True
    
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
        
    def _get_show_image(self):
        image = copy(self.image_buffer[-1])
        mask = self.mask_buffer[-1]
        if not mask is None:
            image[mask] = [0,255,0]
                
        shp = (image.shape[1]*self.SHOW_SCALE, image.shape[0]*self.SHOW_SCALE)
        image = cv2.resize(image, shp)
        
        show_im, warped = self._focus_box(image)
        rec_im = copy(show_im) if warped else image
        rec_im = self._telemetrize_image(rec_im)
        self.image_buffer[-1] = rec_im
        
        if self.flag_recording:
            show_im = cv2.circle(show_im, (show_im.shape[1]-10,10), 5,
                                 [255,0,0], -1)
        show_im = self._telemetrize_image(show_im)
        show_im = cv2.cvtColor(show_im, cv2.COLOR_BGR2RGB)
        return show_im

    def _trim_buffers(self):
        while len(self.temperature_C_buffer) > self.BUFFER_SIZE:
            self.temperature_C_buffer.popleft()
        while len(self.telemetry_buffer) > self.BUFFER_SIZE:
            self.telemetry_buffer.popleft()
        while len(self.image_buffer) > self.BUFFER_SIZE:
            self.image_buffer.popleft()
        while len(self.mask_buffer) > self.BUFFER_SIZE:
            self.mask_buffer.popleft()

    def _keypress_callback(self, wait=1):      
        key = cv2.waitKeyEx(wait)

        if key == ord('f'):
            self.flag_focus_box = not self.flag_focus_box
    
        if key == ord('r'):
            self.subject_quad = [(np.nan,np.nan), (np.nan,np.nan), 
                                 (np.nan,np.nan), (np.nan,np.nan)]
            self.subject_next_vert = (np.nan,np.nan)
            self.homography = None
            self.inv_homography = None
    
        if key == 27:
            self.flag_streaming = False

    def _estop_stream(self):
        print(Clr.WARNING+"Emergency stopping stream... "+Clr.ENDC, end="")
        self.flag_emergency_stop = True
        self.flag_streaming = False
        cv2.destroyAllWindows()
        print(Clr.FAIL+"Stopped."+Clr.ENDC)

    def _stream(self, fps, detect_fronts, multiframe, equalize):
        with Capture(self.PORT, fps) as self.cap:
            if self.flag_emergency_stop:
                self._estop_stream()
                return
            
            self.flag_streaming = True
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE) 
            cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)
            while self.flag_streaming:
                if self.flag_emergency_stop:
                    self._estop_stream()
                    return
                
                temperature_C, telemetry = self.cap.read()
                with self.LOCK:
                    self.temperature_C_buffer.append(temperature_C)
                    self.telemetry_buffer.append(telemetry)
                    self._detect_front(detect_fronts, multiframe)
                    self._temperature_2_image(equalize)
                    image = self._get_show_image()
                    self._trim_buffers()
                    self.frame_number += 1
                    
                cv2.imshow(self.WINDOW_NAME, image) 
                self._keypress_callback()
                
            cv2.destroyAllWindows()
    
    def _buf_len(self):
        l1 = len(self.temperature_C_buffer)
        l2 = len(self.telemetry_buffer)
        l3 = len(self.image_buffer)
        l4 = len(self.mask_buffer)
        if (l1==l2 and l2==l3 and l3==l4): return l1
        
        msg = ("An error occured while validating buffer lengths. "
               "Temperature buffer: {}, Telemetry buffer: {}, "
               "Image buffer: {}, Mask buffer: {}. "
               "This can occur when non thread safe functions are called "
               "while in thread.").format(l1, l2, l3, l4)
        payload = (self.temperature_C_buffer,
                   self.telemetry_buffer,
                   self.image_buffer,
                   self.mask_buffer,)
        raise BufferLengthException(msg, payload=payload)
    
    def _get_writable_frame(self, ignore_buf_min):
        buffer_length = self._buf_len()
        if buffer_length <= self.BUFFER_SIZE and not ignore_buf_min:
            return (None, None, None, None, )
        
        if buffer_length == 0:
            return (None, None, None, None, )

        temperature_C = self.temperature_C_buffer.popleft()
        telemetry = self.telemetry_buffer.popleft()
        image = self.image_buffer.popleft()
        mask = self.mask_buffer.popleft()
        return (temperature_C, telemetry, image, mask, )
    
    def _write_frame(self, frame_data, files):
        if all(d is None for d in frame_data): return
        temperature_C = frame_data[0]
        telemetry = frame_data[1]
        image = frame_data[2]
        mask = frame_data[3]
        
        temperature_mK = np.round(100.*(temperature_C+273.15))
        temperature_mK = temperature_mK.astype(np.uint16)
        encode_param = [int(cv2.IMWRITE_TIFF_COMPRESSION), 
                        cv2.IMWRITE_TIFF_COMPRESSION_LZW]
        T_img = cv2.imencode('.tiff', temperature_mK, encode_param)[1]
        T_img = T_img.tobytes()
        files[0].write(T_img)
        files[0].write(b'DELIM')
        
        json.dump(telemetry, files[1])
        files[1].write('DELIM')
        
        image=cv2.imencode('.png', image)[1].tobytes()
        files[2].write(image)
        files[2].write(b'DELIM')
        
        if not mask is None: 
            files[3].write(zlib.compress(mask.tobytes()))
            files[3].write(b'DELIM')
    
    def _estop_record(self):
        self._estop_stream()
        print(Clr.WARNING+"Emergency stopping record... "+Clr.ENDC, end="")
        self.flag_emergency_stop = True
        self.flag_recording = False
        print(Clr.FAIL+"Stopped."+Clr.ENDC)
    
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
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE) 
            cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)
            files = (T_file, t_file, i_file, m_file, )
            while self.flag_streaming:
                if self.flag_emergency_stop:
                    self._estop_record()
                    return
                
                temperature_C, telemetry = self.cap.read()
                with self.LOCK:
                    self.temperature_C_buffer.append(temperature_C)
                    self.telemetry_buffer.append(telemetry)
                    self._detect_front(detect_fronts, multiframe)
                    self._temperature_2_image(equalize)
                    image = self._get_show_image()
                    frame_data = self._get_writable_frame(ignore_buf_min=False)
                    self.frame_number += 1
                
                self._write_frame(frame_data, files)
                cv2.imshow(self.WINDOW_NAME, image) 
                self._keypress_callback()
                
            cv2.destroyAllWindows()
            
            with self.LOCK:
                term_frame_data = []
                while self._buf_len() > 0:
                    frame_data = self._get_writable_frame(ignore_buf_min=True)
                    term_frame_data.append(frame_data)
            for frame_data in term_frame_data:
                self._write_frame(frame_data, files)
                
            self.recording=False    
    
    def emergency_stop(self):
        if not self.flag_emergency_stop:
            self.flag_emergency_stop = True
            msg="{}EMERGENCY STOP COMMAND RECEIVED{}"
            print(msg.format(Clr.FAIL, Clr.ENDC))        

    def start_stream(self, fps=None, detect_fronts=False, multiframe=True, 
                     equalize=False):
        return _safe_run(self._stream, self._estop_stream,
                         args=(fps, detect_fronts, multiframe, equalize, ))     

    def start_record(self, fps=None, detect_fronts=False, multiframe=True, 
                     equalize=False):
        return _safe_run(self._record, self._estop_record, 
                         args=(fps, detect_fronts, multiframe, equalize))
    
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

    def _buffers_populated(self):
        with self.LOCK:
            return self._buf_len() > 1
    
    def wait_until_stream_active(self, timeout_ms=5000.0, dt_ms=11.1):
        return _safe_run(self._wait_until, args=(self._buffers_populated,
                                                 timeout_ms, dt_ms))   
    
    def is_streaming(self):
        return copy(self.flag_streaming)
    
    def is_recording(self):
        return copy(self.flag_recording)
    
    def get_frame_data(self, focused_ok=False):
        with self.LOCK:
            frame_number = copy(self.frame_number)
            if self._buf_len() == 0:
                return (frame_number, None, None, None, )
            
            time = round(copy(self.telemetry_buffer[-1]['Uptime (ms)'])*.001,3)
            if not focused_ok:
                temperature_C = copy(self.temperature_C_buffer[-1])
                mask = copy(self.mask_buffer[-1])
                return (frame_number, time, temperature_C, mask, )
            
            temperature_C = self._warped_element(self.temperature_C_buffer, 
                                                 return_buffer=False)
            mask = self._warped_element(self.mask_buffer, 
                                        return_buffer=False)
            return (frame_number, time, temperature_C, mask, )
    

class Videowriter():
    def __init__(self):
        pass
    
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
            s = ("{}{}{}\n".format(Clr.FAIL,bars,Clr.ENDC),
                 "{}{}{}\n".format(Clr.FAIL,type(e).__name__,Clr.ENDC),
                 "In function: ",
                 "{}{}(){}\n".format(Clr.OKBLUE, fnc_name, Clr.ENDC),
                 "{}{}{}\n".format(Clr.WARNING,  msg, Clr.ENDC),)
            print(''.join(s))
            rec_name = input('Please enter a different name: ')
            print("{}{}{}".format(Clr.FAIL,bars,Clr.ENDC))
            return self._get_valid_name(rec_name)

    def _make_video(self, rec_name, dirpath, telemetry_file, image_file):
        valid_name = self._get_valid_name(rec_name)
        
        with open(os.path.join(dirpath, telemetry_file), 'r') as f:
            telems = self._read_DELIMed(f, 'r')
        telems = [ast.literal_eval(''.join(t)) for t in telems]
        with open(os.path.join(dirpath, image_file), 'rb') as f:
            images = self._read_DELIMed(f, 'rb')
        images = [self._decode_bytes(i) for i in images]
        
        with av.open(valid_name, mode="w") as container:
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
            