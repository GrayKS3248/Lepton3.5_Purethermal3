# Std modules
import os
import struct
import zlib
import time
from collections import deque
from copy import copy
from threading import Lock

# Package modules
from lepton.exceptions import (ImageShapeException,
                               BufferLengthException,
                               TimeoutException,)
from lepton.misc.cmaps import Cmaps
from lepton.misc.utilities import safe_run, ESC
from lepton.core.detector import Detector
from lepton.core.record import encode_frame_data

# External modules
import cv2
import numpy as np
from scipy.signal import find_peaks


class Capture():
    def __init__(self, port, target_fps, overlay):
        self.PORT = port
        self.IMAGE_SHP = (160, 120)
        try:
            self.TARGET_DT = 1.0 / target_fps
        except:
            self.TARGET_DT = None
        
        self.FLAG_OVERLAY = overlay
        if self.FLAG_OVERLAY:
            parent = os.path.dirname(os.path.realpath(__file__))
            overlay_dirpath = os.path.join(parent, r'_media')
            with open(os.path.join(overlay_dirpath,'overlay.cdat'), 'rb') as f:
                bindat = f.read()
            decoded = zlib.decompress(bindat).split(b'DELIM')
            shape = np.frombuffer(decoded[0], dtype=np.uint8)
            decoded.pop()
            decoded.pop(0)
            self.OVERLAY_FRMS = [np.frombuffer(d,dtype=np.uint8).reshape(shape) 
                                 for d in decoded]
            self.OVERLAY_LEN = len(self.OVERLAY_FRMS)
            self.overlay_n = 0
            
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
    
    def _overlay(self, img):
        overlay_img = -1*np.ones((160,122))
        foreground = self.OVERLAY_FRMS[self.overlay_n]
        overlay_img[:,34:-35] = foreground
        overlay_img = np.flip(overlay_img.T,axis=1)
        theta = 0.05
        tx = 33.
        ty = 0.
        sx = 0.
        sy = -0.22
        p1 = -0.0001
        p2 = -0.003
        sf = -0.28
        Hs = np.array([[1.+sf, 0.,    0.],
                       [0.,    1.+sf, 0.],
                       [0.,    0.,    1.]])
        He = np.array([[np.cos(theta), -np.sin(theta), tx],
                       [np.sin(theta),  np.cos(theta), ty],
                       [0.,             0.,            1.]])
        Ha = np.array([[1., sy, 0.],
                       [sx, 1., 0.],
                       [0., 0., 1.]])
        Hp = np.array([[1., 0., 0.],
                       [0., 1., 0.],
                       [p1, p2, 1.]])
        H = Hs@He@Ha@Hp
        overlay_img = cv2.warpPerspective(overlay_img, H, overlay_img.T.shape,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=-1)
        overlay_mask = overlay_img < 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        overlay_mask = cv2.dilate(overlay_mask.astype(np.uint8), 
                                  kernel, iterations=4)
        overlay_mask = np.logical_not(overlay_mask)
        overlay_mask = overlay_mask.astype(bool)
        overlay_img = (overlay_img.astype(float)/255.) * 145.0 + 20.0
        overlay_img = np.round(100*(overlay_img+273.15)).astype(np.uint16)
        img[overlay_mask] = overlay_img[overlay_mask]
        self.overlay_n = (self.overlay_n + 1) % self.OVERLAY_LEN
        return img
    
    def _decode_data(self, raw_data):
        temp_C = raw_data[:-2] * 0.01 - 273.15
        
        row_A = raw_data[-2,:80]
        row_B = raw_data[-2,80:]
        row_C = raw_data[-1,:80]
        adat=struct.unpack("<bbII16c8B6xI5H4xHIH2x6H64xIH10x", row_A)
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

        serial_number = b''.join(adat[4:20]).hex()

        gpp_version = '{}.{}.{}'.format(adat[20],adat[21],adat[22])
        dsp_version = '{}.{}.{}'.format(adat[24],adat[25],adat[26])
        
        video_format = ''
        if adat[43] == 3: video_format = 'RGB888'
        elif adat[43] == 7: video_format = 'RAW14'
        
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
            'Serial number (hex)':serial_number,
            'gpp version':gpp_version,
            'dsp version':dsp_version,
            'Frame count since reboot':adat[28],
            'Frame mean':adat[29],
            'FPA temperature (counts)':adat[30],
            'FPA temperature (C)':round(adat[31]*0.01 - 273.15, 2),
            'Housing temperature (counts)':adat[32],
            'Housing temperature (C)':round(adat[33]*0.01 - 273.15, 2),
            'FPA temperature at last FFC (C)':round(adat[34]*0.01-273.15, 2),
            'Uptime at last FFC (ms)':adat[35],
            'Housing temperature at last FFC (C)':round(adat[36]*.01-273.15,2),
            'AGC ROI (top left bottom right)':adat[37:41],
            'AGC clip high':adat[41],
            'AGC clip low':adat[42],
            'Video format':video_format,
            'Number of frames used for FFC':2**adat[44],
            'Frame temperature min (C)':float(round(np.min(temp_C),2)),
            'Frame temperature mean (C)':float(round(np.mean(temp_C), 2)),
            'Frame temperature max (C)':float(round(np.max(temp_C), 2)),
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
        return (temp_C, telemetry, )
    
    def reacquire(self):
        self.cap.release()
        time.sleep(5.0) # Wait to require for camera reboot
        self.cap = cv2.VideoCapture(self.PORT + cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.IMAGE_SHP[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.IMAGE_SHP[1]+2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    
    def read(self):
        self._wait_4_frametime()
        res, im = self.cap.read()
        self.prev_frame_time = self._time()
        
        if not res:
            return (False, None, None, )
        
        if im.shape[0]!=self.IMAGE_SHP[1]+2 or im.shape[1]!=self.IMAGE_SHP[0]:
            shp = (im.shape[0]-2,im.shape[1])
            msg = ("Captured image shape {} does not equal "
                   "expected image shape {}. Are you sure the selected "
                   "port is correct? NOTE: If captured image shape is "
                   "(61, 80) the Lepton may be seated incorrectly and you "
                   "should reseat its socket.")
            msg = msg.format(shp, self.IMAGE_SHP)
            raise ImageShapeException(msg, payload=(shp, self.IMAGE_SHP))
        
        if self.FLAG_OVERLAY:
            im = self._overlay(im)
        
        return (True, ) + self._decode_data(im)


class Lepton():
    def __init__(self, camera_port, cmap, scale_factor, overlay):
        # CONSTANTS (DO NOT TOUCH)
        self.PORT = camera_port
        self.CMAP = Cmaps[cmap]
        self.SHOW_SCALE = scale_factor
        self.OVERLAY = overlay
        self.BUFFER_SIZE = 5
        self.WINDOW_NAME = 'Lepton 3.5 on Purethermal 3'
        self._LOCK = Lock()
        
        # THREAD SAFE BUFFERS (INTERACT ONLY THROUGH PUBLIC FUNCTIONS)
        self._frame_number_buffer = deque()
        self._frame_time_buffer = deque()
        self._temperature_C_buffer = deque()
        self._telemetry_buffer = deque()
        self._image_buffer = deque()
        self._mask_buffer = deque()
        
        # THREAD SAFE INTERNAL FLAGS (INTERACT ONLY THROUGH PUBLIC FUNCTIONS)
        self._flag_streaming = False
        self._flag_recording = False
        self._flag_emergency_stop = False
        
        # THREAD UNSAFE INTERNAL FLAGS (DO NOT TOUCH)
        self._flag_focus_box = False
        self._flag_modding_AR = True
        self._flag_modding_fast = False
        
        # FRONT DETECTOR
        self.detector = Detector()
        
        # THREAD UNSAFE CLASS VARIABLES FOR HOMOGRAPHY TRANSFORM (DO NOT TOUCH)
        self._focus_box_AR = 1.33333333
        self._focus_box_size = 0.50
        self._focus_box = [(), (), (), ()]
        self._subject_quad = [(np.nan,np.nan), (np.nan,np.nan), 
                             (np.nan,np.nan), (np.nan,np.nan)]
        self._subject_next_vert = (np.nan,np.nan)
        self._homography = None
        self._scaled_homography = None
        self._inv_homography = None
        self._inv_scaled_homography = None
        
        # THREAD UNSAFE CLASS VARIABLES FOR FRAME TRACKING (DO NOT TOUCH)
        self._wall_epoch = np.nan
        self._frame_count = np.nan
        self._frame_num_prev_send = np.nan

    def _mouse_callback(self, event, x, y, flags, param):
        if not self._flag_focus_box: return
        
        if event == cv2.EVENT_MBUTTONDOWN and self._homography is None:
            self._flag_modding_fast = not self._flag_modding_fast
        
        if event == cv2.EVENT_MOUSEWHEEL and self._homography is None:
            rate = (0.1 if self._flag_modding_fast else 0.01)
            if flags > 0:
                if self._flag_modding_AR:
                    self._focus_box_AR += rate
                else:
                    self._focus_box_size += rate
            else:
                if self._flag_modding_AR:
                    self._focus_box_AR -= rate
                else:
                    self._focus_box_size -= rate
            self._focus_box_size = np.clip(self._focus_box_size, 0.01, 1.0)
            self._focus_box_AR = np.clip(self._focus_box_AR, 0.01, 
                                        min(4./(3.*self._focus_box_size),9.99))

        if event == cv2.EVENT_RBUTTONDOWN and self._homography is None:
            self._flag_modding_AR = not self._flag_modding_AR
                
        if event == cv2.EVENT_LBUTTONDOWN:
            if (np.nan, np.nan) in self._subject_quad:
                insert_at = self._subject_quad.index((np.nan, np.nan))
                self._subject_quad[insert_at] = (x,y)
            
        if event == cv2.EVENT_MOUSEMOVE:
            self._subject_next_vert = np.array([x,y])
    
    def _warp_element(self, element):
        is_warped = self._flag_focus_box and not self._homography is None
        
        if element is None or not is_warped:
            return copy(element)
        
        typ = element.dtype
        if typ==bool:
            warped = cv2.warpPerspective(element.astype(np.uint8), 
                                         self._scaled_homography,
                                         (160, 120))
        else:
            warped = cv2.warpPerspective(element,
                                         self._scaled_homography, 
                                         (160, 120))
        (l,t), (r,b) = self._focus_box[0], self._focus_box[2]
        l = int(np.round(l/self.SHOW_SCALE))
        t = int(np.round(t/self.SHOW_SCALE))
        r = int(np.round(r/self.SHOW_SCALE))
        b = int(np.round(b/self.SHOW_SCALE))
        warped = warped[t:b+1,l:r+1].astype(typ)
        return warped
    
    def _warp_deque(self, buffer, n=3):
        buffer_len = len(buffer)
        warped_buffer = []
        for i in range(buffer_len):
            if i == n: break
            warped_element = self._warp_element(buffer[buffer_len-1-i])
            warped_buffer.append(warped_element)
        warped_buffer.reverse()
        return warped_buffer
    
    def _dewarp_element(self, element, thresh=0.25):
        if (element is None or 
            not self._flag_focus_box or self._inv_homography is None):
            return copy(element)
        
        (l,t), (r,b) = self._focus_box[0], self._focus_box[2]
        l = int(np.round(l/self.SHOW_SCALE))
        t = int(np.round(t/self.SHOW_SCALE))
        r = int(np.round(r/self.SHOW_SCALE))
        b = int(np.round(b/self.SHOW_SCALE))
        typ = element.dtype
        if typ == bool:
            dewarped = np.zeros((120, 160), dtype=np.uint8)
            dewarped[t:b+1,l:r+1] = element.astype(np.uint8)
        else:
            dewarped = np.zeros((120, 160), dtype=typ)
            dewarped[t:b+1,l:r+1] = element
        dewarped = cv2.warpPerspective(dewarped,
                                       self._inv_scaled_homography,
                                       (160, 120)).astype(typ)
        return dewarped
    
    def _detect_front(self, detect_fronts, multiframe):
        if not detect_fronts or len(self._temperature_C_buffer)==0:
            self._mask_buffer.append(None)
            return
        
        if multiframe:
            temps = self._warp_deque(self._temperature_C_buffer)
        else:
            temps = [self._warp_element(self._temperature_C_buffer[-1])]
        mask = self.detector.front(temps, 'kmeans')
        self._mask_buffer.append(self._dewarp_element(mask))
    
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
        F_prime = F[local_maxizers[local_maxizers>=global_maximizer]]
        if len(F_prime) == 0:
            return norm
        else:
            T = np.median(F[local_maxizers[local_maxizers>=global_maximizer]])
        P[P>T] = T
        FT = np.cumsum(P)
        DT = np.floor(255*FT/FT[-1]).astype(np.uint8)
        eq = DT[quantized] / 255.0
        return  eq

    def _temperature_2_image(self, equalize):
        image = self._normalize_temperature(self._temperature_C_buffer[-1],
                                            equalize=equalize)
        image = np.round(255.*self.CMAP(image)[:,:,:-1]).astype(np.uint8)
        self._image_buffer.append(image)
    
    def _draw_subject_quad(self, image):
        lines = []
        for i in range(4):
            j = (i+1) % 4
            lines.append([self._subject_quad[i], self._subject_quad[j]])
        lines = np.array(lines)
        
        next_vert_at = np.all(np.isnan(lines[:,1,:]),axis=1)
        if any(next_vert_at):
            next_vert_at = np.argmax(next_vert_at)
            lines[next_vert_at,1,:] = self._subject_next_vert
        
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
        box_h = int(np.round(self._focus_box_size*img_h))
        box_w = int(np.round(self._focus_box_AR*box_h))
        l = int(0.5*(img_w - box_w))
        t = int(0.5*(img_h - box_h))
        r = l + box_w - 1
        b = t + box_h - 1
        self._focus_box = [(l,t),(l,b),(r,b),(r,t)]
        
        color = [0,255,255] if quad_incomplete else [255,0,255]
        fb_image = cv2.rectangle(image, self._focus_box[0], self._focus_box[2],
                                 color, 1)
        if not quad_incomplete: return fb_image
        
        cnr=[i for i,s in enumerate(self._subject_quad) if s!=(np.nan, np.nan)]
        cnr = len(cnr)
        if cnr < 4:
            fb_image = cv2.circle(fb_image, self._focus_box[cnr], 
                                  3, [255,0,255], -1)
        
        if self._flag_modding_AR:
            txt = 'AR: {:.2f}'.format(self._focus_box_AR)
            fb_image = cv2.rectangle(fb_image,(l+2,t+2),(l+75,t+15),[0,0,0],-1)
        else:
            txt = 'Size: {:.2f}'.format(self._focus_box_size)
            fb_image = cv2.rectangle(fb_image,(l+2,t+2),(l+89,t+15),[0,0,0],-1)
        fb_image = cv2.putText(fb_image, txt, (l+4,t+14),
                               cv2.FONT_HERSHEY_PLAIN , 1, (255,255,255), 1,
                               cv2.LINE_AA)
        
        return fb_image
    
    def _get_focus_box(self, image):
        if not self._flag_focus_box: return image, False
        
        quad_incomplete = np.any(np.isnan(self._subject_quad))
        if quad_incomplete:
            quad_image = self._draw_subject_quad(image)
            return self._draw_focus_box(quad_image, quad_incomplete), False
        
        if self._homography is None:
            xs = np.array(self._subject_quad)
            ys = np.array(self._focus_box)
            self._homography, _ = cv2.findHomography(xs, ys)
            self._scaled_homography = copy(self._homography)
            self._scaled_homography[0,-1] /= self.SHOW_SCALE
            self._scaled_homography[1,-1] /= self.SHOW_SCALE
            self._scaled_homography[2,0] *= self.SHOW_SCALE
            self._scaled_homography[2,1] *= self.SHOW_SCALE
            self._inv_homography = np.linalg.inv(self._homography)
            self._inv_scaled_homography=np.linalg.inv(self._scaled_homography)

        shp = (image.shape[1], image.shape[0])
        warped_image = cv2.warpPerspective(image, self._homography, shp)
        return self._draw_focus_box(warped_image, quad_incomplete), True
    
    def _uptime_str(self):
        telemetry = self._telemetry_buffer[-1]
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
        telemetry = self._telemetry_buffer[-1]
        mn = '({:0>6.2f})'.format(telemetry['Frame temperature min (C)'])
        i=1
        while mn[i]=='0' and mn[i+1]!='.':
            mn=' {}{}'.format(mn[:i], mn[i+1:])
            i+=1
        me = '| {:0>6.2f} |'.format(telemetry['Frame temperature mean (C)'])
        i=2
        while me[i]=='0' and me[i+1]!='.':
            me=' {}{}'.format(me[:i], me[i+1:])
            i+=1
        mx = '({:0>6.2f})'.format(telemetry['Frame temperature max (C)'])
        i=1
        while mx[i]=='0' and mx[i+1]!='.':
            mx=' {}{}'.format(mx[:i], mx[i+1:])
            i+=1
        return "{} {} {} C".format(mn, me, mx)
    
    def _fps_str(self):
        if len(self._telemetry_buffer)<self.BUFFER_SIZE:
            return 'FPS: ---'
        
        frame_times = []
        for i in range(self.BUFFER_SIZE):
            telemetry = self._telemetry_buffer[i-self.BUFFER_SIZE]
            frame_times.append(telemetry['Uptime (ms)'])
        if len(frame_times) <= 1:
            delta = 0.0
        else:
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
        
        if self._telemetry_buffer[-1]['FFC state']=='imminent':
            telimg = cv2.rectangle(telimg,(5,5),(35,25),[0,0,0],-1)
            telimg = cv2.putText(telimg, "FFC", (6,21),
                                cv2.FONT_HERSHEY_PLAIN , 1, (255,255,255), 1,
                                cv2.LINE_AA)
        return telimg
        
    def _get_show_image(self):
        image = copy(self._image_buffer[-1])
        mask = self._mask_buffer[-1]
        if not mask is None:
            image[mask] = [0,255,0]
                
        shp = (image.shape[1]*self.SHOW_SCALE, image.shape[0]*self.SHOW_SCALE)
        image = cv2.resize(image, shp, interpolation=cv2.INTER_NEAREST)
        
        show_im, warped = self._get_focus_box(image)
        rec_im = copy(show_im) if warped else image
        rec_im = self._telemetrize_image(rec_im)
        self._image_buffer[-1] = rec_im
        
        if self._flag_recording:
            show_im = cv2.circle(show_im, (show_im.shape[1]-10,10), 5,
                                 [255,0,0], -1)
        show_im = self._telemetrize_image(show_im)
        show_im = cv2.cvtColor(show_im, cv2.COLOR_BGR2RGB)
        return show_im

    def _buf_len(self):
        l1 = len(self._frame_number_buffer)
        l2 = len(self._frame_time_buffer)
        l3 = len(self._temperature_C_buffer)
        l4 = len(self._telemetry_buffer)
        l5 = len(self._image_buffer)
        l6 = len(self._mask_buffer)
        if (l1==l2 and l2==l3 and l3==l4 and l4==l5 and l5==l6): return l1
        
        msg = ("An error occured while validating buffer lengths. "
               "Frame number buffer: {}, Frame time buffer: {},"
               "Temperature buffer: {}, Telemetry buffer: {}, "
               "Image buffer: {}, Mask buffer: {}. "
               "This can occur when non thread safe functions are called "
               "while in thread.").format(l1, l2, l3, l4, l5, l6)
        payload = (self._frame_number_buffer,
                   self._frame_time_buffer,
                   self._temperature_C_buffer,
                   self._telemetry_buffer,
                   self._image_buffer,
                   self._mask_buffer,)
        raise BufferLengthException(msg, payload=payload)

    def _trim_buffers(self):
        for i in range(self.BUFFER_SIZE, self._buf_len()):
            self._frame_number_buffer.popleft()
            self._frame_time_buffer.popleft()
            self._temperature_C_buffer.popleft()
            self._telemetry_buffer.popleft()
            self._image_buffer.popleft()
            self._mask_buffer.popleft()

    def _keypress_callback(self, wait=1):      
        key = cv2.waitKeyEx(wait)

        if key == ord('f'):
            self._flag_focus_box = not self._flag_focus_box
    
        if key == ord('r'):
            self._subject_quad = [(np.nan,np.nan), (np.nan,np.nan), 
                                  (np.nan,np.nan), (np.nan,np.nan)]
            self._subject_next_vert = (np.nan,np.nan)
            self._homography = None
            self._scaled_homography = None
            self._inv_homography = None
            self._inv_scaled_homography = None
    
        if key == 27:
            self._flag_streaming = False

    def _estop_stream(self):
        msg = "Emergency stopping stream... "
        print(ESC.fail(msg), end="", flush=True)
        self._flag_emergency_stop = True
        self._flag_streaming = False
        cv2.destroyAllWindows()
        print(ESC.OKCYAN+"Stopped."+ESC.ENDC, flush=True)

    def _stream(self, fps, detect_fronts, multiframe, equalize):
        with Capture(self.PORT, fps, self.OVERLAY) as self.cap:
            if self._flag_emergency_stop:
                with self._LOCK:
                    self._estop_stream()
                time.sleep(1.0) # Wait for other tasks out of thread
                msg = "Stream emergency stopped before starting."
                print(ESC.fail(msg), flush=True)
                return
            
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE) 
            cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)
            with self._LOCK:
                self._flag_streaming = True
            print(ESC.header("Stream started."), flush=True)
            print(ESC.header(''.join(['-']*60)), flush=True)
            self._wall_epoch = time.time()
            self._frame_count = 0
            while self._flag_streaming:
                if self._flag_emergency_stop:
                    with self._LOCK:
                        self._estop_stream()
                    time.sleep(1.0) # Wait for other tasks out of thread
                    print(ESC.header(''.join(['-']*60)), flush=True)
                    print(ESC.header("Stream ended in emergency stop."), 
                          flush=True)
                    return
                
                res, temperature_C, telemetry = self.cap.read()
                wall_dt = 1000.*(time.time()-self._wall_epoch)
                wall_dt = int(np.round(wall_dt))
                # LOS
                if not res:
                    self.cap.reacquire()
                    self._frame_num_prev_send = np.nan
                    continue
                # If the frame is captured during camera boot, ignore it
                # If the frame is an FFC frame, ignore it
                if (telemetry['FFC state'] == 'never commanded' or
                    telemetry['FFC state'] == 'in progress'):
                    continue
                
                with self._LOCK:
                    # Check for FFC in progress by un-updated frame number
                    # or frame time
                    frame_num = telemetry['Frame count since reboot']
                    frame_tim = telemetry['Uptime (ms)']
                    if (self._buf_len()>0 and 
                        (self._frame_number_buffer[-1][0]==frame_num or
                         self._frame_time_buffer[-1][0]==frame_tim)):
                        continue
                    
                    self._frame_number_buffer.append((frame_num,
                                                      self._frame_count))
                    self._frame_time_buffer.append((frame_tim, 
                                                    wall_dt))
                    self._temperature_C_buffer.append(temperature_C)
                    self._telemetry_buffer.append(telemetry)
                    self._detect_front(detect_fronts, multiframe)
                    self._temperature_2_image(equalize)
                    image = self._get_show_image()
                    self._trim_buffers()
                    self._keypress_callback()
                    
                cv2.imshow(self.WINDOW_NAME, image) 
                self._frame_count += 1
                
            cv2.destroyAllWindows()
        time.sleep(1.0) # Wait for other tasks out of thread
        print(ESC.header(''.join(['-']*60)), flush=True)
        print(ESC.header("Stream ended normally."), flush=True)
    
    def _get_writable_frame(self, ignore_buf_min):
        buffer_length = self._buf_len()
        if (buffer_length <= self.BUFFER_SIZE and not ignore_buf_min or
            buffer_length == 0):
            return (None, None, None, None, None, )

        # No need to copy because pop operation
        frame_num = self._frame_number_buffer.popleft()
        frame_time_ms = self._frame_time_buffer.popleft()
        temperature_C = self._temperature_C_buffer.popleft()
        temperature_cK = 100.*(temperature_C+273.15)
        telemetry = str(self._telemetry_buffer.popleft())
        image = self._image_buffer.popleft()
        mask = self._mask_buffer.popleft()
        if mask is None:
            mask = np.zeros(temperature_C.shape, dtype=bool)
        return (frame_num,frame_time_ms,temperature_cK,telemetry,image,mask,)
    
    def _write_chunked_frame(self, frame_data, files):
        if all(d is None for d in frame_data): return
        encoded = encode_frame_data(frame_data, ['L', 'L', 'H', 'U', 'B', '?'])
        for f, e in zip(files, encoded):
            msg_len = bytes('{:08d}'.format(len(e)), 'utf-8')
            f.write(msg_len + e)
            
    def _estop_record(self):
        self._estop_stream()
        msg = "Emergency stopping record... "
        print(ESC.fail(msg), end="", flush=True)
        self._flag_emergency_stop = True
        self._flag_recording = False
        print(ESC.OKCYAN+"Stopped."+ESC.ENDC, flush=True)
    
    def _record(self, fps, detect_fronts, multiframe, equalize):
        dirname = 'rec_data'
        os.makedirs(dirname, exist_ok=True)
        fnames = ['frame_number.dat', 'frame_time.dat', 'temperature.dat', 
                  'telem.dat', 'image.dat', 'mask.dat']
        
        with (Capture(self.PORT, fps, self.OVERLAY) as self.cap,
              open(os.path.join(dirname, fnames[0]), 'wb') as fn_file,
              open(os.path.join(dirname, fnames[1]), 'wb') as ft_file,
              open(os.path.join(dirname, fnames[2]), 'wb') as T_file,
              open(os.path.join(dirname, fnames[3]), 'wb') as t_file,
              open(os.path.join(dirname, fnames[4]), 'wb') as i_file,
              open(os.path.join(dirname, fnames[5]), 'wb') as m_file,):
            if self._flag_emergency_stop:
                with self._LOCK:
                    self._estop_record()
                time.sleep(1.0) # Wait for other tasks out of thread
                msg = "Recording emergency stopped before starting."
                print(ESC.fail(msg), flush=True)
                return

            files = (fn_file, ft_file, T_file, t_file, i_file, m_file, )
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE) 
            cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)
            with self._LOCK:
                self._flag_streaming = True
                self._flag_recording = True
            print(ESC.header("Recording started."), flush=True)
            print(ESC.header(''.join(['-']*60)), flush=True)
            self._wall_epoch = time.time()
            self._frame_count = 0
            while self._flag_streaming:
                if self._flag_emergency_stop:
                    with self._LOCK:
                        self._estop_record()
                    time.sleep(1.0) # Wait for other tasks out of thread
                    print(ESC.header(''.join(['-']*60)), flush=True)
                    print(ESC.header("Recording ended in emergency stop."),
                          flush=True)
                    return
                
                res, temperature_C, telemetry = self.cap.read()
                wall_dt = 1000.*(time.time()-self._wall_epoch)
                wall_dt = int(np.round(wall_dt))
                # LOS
                if not res:
                    self.cap.reacquire()
                    self._frame_num_prev_send = np.nan
                    continue
                # If the frame is captured during camera boot, ignore it
                # If the frame is an FFC frame, ignore it
                if (telemetry['FFC state'] == 'never commanded' or
                    telemetry['FFC state']=='in progress'):
                    continue
                
                with self._LOCK:
                    # Check for FFC in progress by un-updated frame number
                    # or frame time
                    frame_num = telemetry['Frame count since reboot']
                    frame_tim = telemetry['Uptime (ms)']
                    if (self._buf_len()>0 and 
                        (self._frame_number_buffer[-1][0]==frame_num or
                         self._frame_time_buffer[-1][0]==frame_tim)):
                        continue
                    
                    self._frame_number_buffer.append((frame_num,
                                                      self._frame_count))
                    self._frame_time_buffer.append((frame_tim, 
                                                    wall_dt))
                    self._temperature_C_buffer.append(temperature_C)
                    self._telemetry_buffer.append(telemetry)
                    self._detect_front(detect_fronts, multiframe)
                    self._temperature_2_image(equalize)
                    image = self._get_show_image()
                    frame_data = self._get_writable_frame(ignore_buf_min=False)
                    self._keypress_callback()
                    
                self._write_chunked_frame(frame_data, files)
                cv2.imshow(self.WINDOW_NAME, image) 
                self._frame_count += 1
                
            cv2.destroyAllWindows()
            
            with self._LOCK:
                term_frame_data = []
                while self._buf_len() > 0:
                    frame_data = self._get_writable_frame(ignore_buf_min=True)
                    term_frame_data.append(frame_data)
                self._flag_recording = False  
            for frame_data in term_frame_data:
                self._write_chunked_frame(frame_data, files)
            
        time.sleep(1.0) # Wait for other tasks out of thread
        print(ESC.header(''.join(['-']*60)), flush=True)
        print(ESC.header("Recording ended normally."), flush=True)

    def emergency_stop(self):
        with self._LOCK:
            if not self._flag_emergency_stop:
                self._flag_emergency_stop = True
                msg="{}EMERGENCY STOP COMMAND RECEIVED{}"
                print(msg.format(ESC.FAIL, ESC.ENDC), flush=True)        

    def stop(self):
        with self._LOCK:
            self._flag_streaming = False

    def start_stream(self, fps=None, detect_fronts=False, multiframe=True, 
                     equalize=False):
        res = safe_run(self._stream, self._estop_stream,
                        args=(fps, detect_fronts, multiframe, equalize, ))   
        if res[0] < 0:
            time.sleep(1.0) # Wait for other tasks out of thread
            print(ESC.header(''.join(['-']*60)), flush=True)
            msg = "Streaming ended in emergency stop due to exception."
            print(ESC.header(msg),flush=True)
            
        return res[0]

    def start_record(self, fps=None, detect_fronts=False, multiframe=True, 
                     equalize=False):
        res = safe_run(self._record, self._estop_record, 
                       args=(fps, detect_fronts, multiframe, equalize))
        if res[0] < 0:
            time.sleep(1.0) # Wait for other tasks out of thread
            print(ESC.header(''.join(['-']*60)), flush=True)
            msg = "Recording ended in emergency stop due to exception."
            print(ESC.header(msg),flush=True)
            
        return res[0]
    
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
            if self._flag_emergency_stop: break

    def _buffers_populated(self):
        with self._LOCK:
            return self._buf_len() > 0
    
    def wait_until_stream_active(self, timeout_ms=5000., dt_ms=25.):
        return safe_run(self._wait_until, args=(self._buffers_populated,
                                                 timeout_ms, dt_ms))[0] 
    
    def _frame_data_to_bytes(self, frame_data):
        frame_num = frame_data[0]
        frame_time_s = frame_data[1]
        temperature_C = frame_data[2]
        mask = frame_data[3]
        
        if frame_num is None:
            b_frame_num = b''
        else:
            b_frame_num = self._encode(frame_num, np.uint32)[:-5]
        
        if frame_time_s is None:
            b_frame_time_ms = b''
        else:
            b_frame_time_ms = [1000.*s for s in frame_time_s]
            b_frame_time_ms = self._encode(b_frame_time_ms, np.uint32)[:-5]
        
        if temperature_C is None:
            b_temperature_cK = b''
        else:
            b_temperature_cK = 100.*(temperature_C+273.15)
            b_temperature_cK = self._encode(b_temperature_cK, np.uint16)[:-5]
        
        if mask is None: 
            b_mask = b''
        else:
            b_mask = self._encode(mask, bool)[:-5]
        
        return  (b_frame_num, b_frame_time_ms, b_temperature_cK, b_mask, )

    def _get_frame_data(self, focused_ok):
        with self._LOCK:
            if self._buf_len() == 0:
                return (None, None, None, None, None, None, )
            
            frame_num = copy(self._frame_number_buffer[-1])
            if (not np.isnan(self._frame_num_prev_send) and
                frame_num[0] <= self._frame_num_prev_send):
                return (None, None, None, None, None, None, )
            self._frame_num_prev_send = frame_num[0]
            
            frame_time_ms = copy(self._frame_time_buffer[-1])
            temperature_C = copy(self._temperature_C_buffer[-1])
            temperature_cK = (100.*(temperature_C+273.15)).astype(np.uint16)
            telemetry = str(copy(self._telemetry_buffer[-1]))
            image = copy(self._image_buffer[-1])
            mask = copy(self._mask_buffer[-1])
            if mask is None:
                mask = np.zeros(temperature_cK.shape, dtype=bool)
            is_warped = self._flag_focus_box and not self._homography is None
            
        if not focused_ok or not is_warped:
            return (frame_num, frame_time_ms, temperature_cK,
                    telemetry, image, mask,)
        
        warped_temperature_C = self._warp_element(temperature_C)
        warped_temperature_cK = 100.*(warped_temperature_C+273.15)
        warped_temperature_cK = np.round(warped_temperature_cK)
        warped_temperature_cK = warped_temperature_cK.astype(np.uint16)
        warped_mask = self._warp_element(mask)
        
        return (frame_num, frame_time_ms, warped_temperature_cK,
                telemetry, image, warped_mask,)
    
    def get_frame_data(self, focused_ok=False, encoded=False):
        frame_data = self._get_frame_data(focused_ok)
        if encoded and not any([f is None for f in frame_data]):
            frame_data = encode_frame_data(frame_data, 
                                           ['L', 'L', 'H', 'U', 'B', '?'])
        return frame_data
    
    def is_streaming(self):
        with self._LOCK:
            return copy(self._flag_streaming)
    
    def is_recording(self):
        with self._LOCK:
            return copy(self._flag_recording)
            