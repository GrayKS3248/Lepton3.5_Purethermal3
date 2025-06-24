# External modules
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


class Detector():
    def __init__(self, T0=0.0, a0=0.0, Hr=3.50e5, Cp=1600.0,
                 n_iter=12, epsilon=0.04):
        """
        Initialize a front detector set up to detect fronts of DCPD with 100ppm
        GC2.

        Parameters
        ----------
        T0 : float, optional
            The estimated initial temperature in celcius. Only used in the
            'temperature' front detection method. The default is 0.0.
        a0 : float, optional
            The estimated initial cure. Only used in the
            'temperature' front detection method. The default is 0.0.
        Hr : float, optional
            The enthalpy of reaction in J/Kg. Only used in the
            'temperature' front detection method. The default is 3.50e5.
        Cp : float, optional
            The specific heat in J/Kg-K. Only used in the
            'temperature' front detection method. The default is 1600.0.
        n_iter : int, optional
            Stop the kmeans algorithm after the specified number of iterations,
            n_iter. The default is 12.
        epsilon : float, optional
            Stop the kmeans algorithm if specified accuracy, epsilon, 
            is reached. The default is 0.04.
            
        Returns
        -------
        None.

        """
        self.T0 = T0 # C
        self.a0 = a0 # -
        self.hr = Hr # J - Kg^{-1}
        self.cp = Cp # J - Kg^{-1} - K^{-1}
        
        # Set kmeans clustering parameters
        self.criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 
                         n_iter, 
                         epsilon)
        self.flags = cv2.KMEANS_RANDOM_CENTERS
        
        
    def _kmeans(self, temperatures, n_try=8, min_temp=50.0):
        """
        Uses automatic thresholding of the temperature image, the 
        gradient of the temperature image, and the time derivative
        of the temperature image sequence to estimate front location

        Parameters
        ----------
        temperature : list of array of floats, shape( (m,n) )
            Time ordered temperature images in Celcius.
        n_try : int, optional
            Number of times the kmeans algorithm is executed using different
            initial labellings. The default is 8.
        min_temp : float, optional
            The minimum cutoff temperature for front definition in Celcius. 
            The default is 35.0.

        Returns
        -------
        front_mask : array of bool, shape( (m,n) )
            A boolean mask of detected front instances.

        """       
        # Extract size information
        T_shape = temperatures[-1].shape
        
        # Blur the most recent temperature image
        blur_size = int(np.round(max(T_shape)/40.))
        blur_size = (blur_size-(blur_size%2-1),)*2
        bT = cv2.GaussianBlur(temperatures[-1], blur_size, 0)
        
        # Flatten the temperature for use in k means clustering
        # Convert to float32
        fbT = bT.flatten().reshape(-1,1).astype(np.float32)
        
        # Apply 4 mean thresholding on the temperature to detect
        # 1. Background
        # 2. Candidate front and bulk boundries
        # 3. Candidate front and bulk cured material at high temperatures 
        #    near to the front
        # 4. Bulk cured material at high temperatures far from the front
        _, T_lab, T_cen = cv2.kmeans(fbT,4,None,
                                     self.criteria,n_try,self.flags)
        T_lab = T_lab.reshape(T_shape)
        T_cen = T_cen.flatten()
        T_cen = sorted(range(len(T_cen)), key=lambda k: T_cen[k])
        
        # Get the candidate front masked based on only temperature
        T_mask = (T_lab==T_cen[1]) | (T_lab==T_cen[2])
        
        # Apply the minimum cutoff temperature
        if not min_temp is None:
            T_mask = T_mask & (temperatures[-1]>min_temp)
        
        # Get the L2 norm of the gradient of the temperature field
        Tdx, Tdy = np.gradient(temperatures[-1])
        gT = np.sqrt(np.square(Tdx)+np.square(Tdy))
        
        # Blur the L2 norm of the gradient
        bgT = cv2.GaussianBlur(gT, blur_size, 0)
        
        # Flatten the l2 norm for use in k means clustering
        # Convert to float32
        fbgT = bgT.flatten().reshape(-1,1).astype(np.float32)
        
        # Apply 4 mean thresholding on L2 norm of temperature to detect
        # 1. Bulk cured material and background
        # 2. Bulk cured material boundaries
        # 3. Cured material recently interacted with front and colder front
        # 4. Front candidate
        _, gT_lab, gT_cen = cv2.kmeans(fbgT,4,None,
                                       self.criteria,n_try,self.flags)
        gT_lab = gT_lab.reshape(T_shape)
        gT_cen = gT_cen.flatten()
        gT_cen = sorted(range(len(gT_cen)), key=lambda k: gT_cen[k])
        
        # Get the candidate front mask based on only the gradient
        gT_mask = (gT_lab==gT_cen[2]) | (gT_lab==gT_cen[3])
        
        # If only one temperature image was provided, front estimate cannot
        # use derivative of temperature. Return intersection of temperature
        # mask and gradient mask
        if len(temperatures) == 1:
            return T_mask & gT_mask

        # Calculate the blurred temporal differential of the temperature
        # image sequence by spatiotemporal gaussian differentiation
        bdT = -gaussian_filter(temperatures, (2,2,3,), order=(0,0,1), 
                               mode='nearest')[-1]
        
        # Isolate regions that got hotter. The front won't get colder.
        bdT[bdT<0.0]=0.0

        # Flatten the delta temperature for use in k means clustering
        # Convert to float32
        fbdT = bdT.flatten().reshape(-1,1).astype(np.float32)
        
        # Apply 3 means thresholding on delta temperature to detect
        # 1. Bulk cured material and background
        # 2. Low contrast candidate front locations
        # 3. High contrast candidate front locations
        _, dT_lab, dT_cen = cv2.kmeans(fbdT,3,None,
                                       self.criteria,n_try,self.flags)
        dT_lab = dT_lab.reshape(T_shape)
        dT_cen = dT_cen.flatten()
        dT_cen = sorted(range(len(dT_cen)), key=lambda k: dT_cen[k])
            
        # Get the candidate front mask based on only delta temperature
        dT_mask = (dT_lab==dT_cen[1]) | (dT_lab==dT_cen[2])
        
        # Determine if the dT mask is an outlier by FToA comparison
        return T_mask & gT_mask & dT_mask
    
    def _canny(self, temperature):
        """
        Uses Canny edge detection to get a front mask.

        Parameters
        ----------
        temperature : array of floats, shape( (m,n) )
            Temperature image in Celcius.

        Returns
        -------
        front_mask : array of bool, shape( (m,n) )
            A boolean mask of detected front instances.

        """
        # Convert to correct type
        temperature = temperature.astype(np.uint8)
        
        # Hand tuned Canny edge detection to get mask
        front_mask = cv2.Canny(temperature,
                               threshold1=300,
                               threshold2=300,
                               apertureSize=3).astype(bool)
        return front_mask
    
    def _sobel(self, temperature):
        """
        Modified Sobel edge detection method used to detect fronts.

        Parameters
        ----------
        temperature : array of floats, shape( (m,n) )
            Temperature image in Celcius.

        Returns
        -------
        front_mask : array of bool, shape( (m,n) )
            A boolean mask of detected front instances.

        """
        # Set hand tuned parameters
        blur = 50.
        tukey = 5.
        
        # Blur the input temperature image
        s0 = int(np.round(temperature.shape[0]/blur))
        s1 = int(np.round(temperature.shape[1]/blur))
        size = (s0-(s0%2-1),s1-(s1%2-1))
        blur = cv2.GaussianBlur(temperature, size, 0)
        
        # Calculate the sobel gradient of the temperature image
        dx,dy = np.gradient(blur)
        grad = np.sqrt(dx**2 + dy**2)
        
        # Isolate the outliers in the gradient
        p1 = np.percentile(grad,10)
        p3 = np.percentile(grad,90)
        ipr = p3 - p1
        upper = p3 + (ipr * tukey)
        
        # The front mask is outliers in the gradient
        front_mask = grad > upper
        return front_mask
    
    def _ftemp(self, temperature):
        """
        Estimate the front temperature and threshold the temperature image
        to get approximate front
        
        Parameters
        ----------
        temperature : array of floats, shape( (m,n) )
            Temperature image in Celcius.

        Returns
        -------
        front_mask : array of bool, shape( (m,n) )
            A boolean mask of detected front instances.

        """
        # Get the expect maximum front temperature
        ftemp_hi = self.T0 + 0.90*(self.hr/self.cp)*(1.0 - self.a0)
        
        # Get the expect minimum front temperature
        ftemp_lo = self.T0 + 0.70*(self.hr/self.cp)*(1.0 - self.a0)
        
        # Threshold the temperature image to get the front estimate
        return np.logical_and(temperature>=ftemp_lo, temperature<=ftemp_hi)
    
    def front(self, temperatures, method):
        """
        Given a set of sequential temperature images, detects front instances
        in the last temperature image in the sequence.

        Parameters
        ----------
        temperatures : list of arrays
            A list containing sequential temperature images in celcius that
            contain a front in them.
        method : string
            The detection method used. The four types include 'kmeans',
            'canny', 'sobel', and 'ftemp'. 
                
                'kmeans' uses an automatic thresholding technique based on the
                temperature field, gradient of the temperature field, and time
                derivative of the temperature field. 
                
                'canny' and 'sobel' use the Canny edge detection and Sobel edge
                detection methods on the temperature image, repsectively. 
                
                'ftemp' uses expect front temperature estimates based on
                cure kinetics and initial conditions to threshold the
                temperature image.

        Returns
        -------
        front_mask : Bool array, shape same as each temperature image
            A boolean array that indicates if each pixel is a front instance.

        """
        # K means clustering method
        if method == 'kmeans':
            front_mask = self._kmeans(temperatures)
            
        # Canny edge detection method
        if method == 'canny':
            front_mask = self._canny(temperatures[-1])
        
        # Sobel edge detection method
        if method == 'sobel':
            front_mask = self._sobel(temperatures[-1])
    
        if method == 'ftemp':
            front_mask = self._ftemp(temperatures[-1])
        
        # Return the detected front instances
        return front_mask
    
    
import os
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmap
def make(d, sd, ns, c=(None, None)):
    
    if c[0] is None and c[1] is None:
        T = [cv2.imread(os.path.join(d,sd,n))[:,:,0]*0.7059+20 
             for n in ns]
    elif c[0] is None and not c[1] is None:
        T = [cv2.imread(os.path.join(d,sd,n))[:,c[1][0]:c[1][1],0]*0.7059+20 
             for n in ns]
    elif not c[0] is None and c[1] is None:
        T = [cv2.imread(os.path.join(d,sd,n))[c[0][0]:c[0][1],:,0]*0.7059+20 
             for n in ns]
    else:
        T = [cv2.imread(os.path.join(d,sd,n))[c[0][0]:c[0][1],
                                              c[1][0]:c[1][1],0]*0.7059+20 
             for n in ns]
    m = Detector()._kmeans(T)
    T0 = cmap['inferno'](np.clip(np.round((T[-1]-20)*1.4166),0,255)/255)
    T0m = T0.copy()
    T0m[m] = [0., 1., 0., 1.]
    plt.imshow(T0m)
    plt.show()
    plt.clf()
    plt.close()
    return T0, T0m
if __name__ == "__main__":
    d = r'C:/Users/Grayson/Docs/Repos/FP_Feeback_Control/Experimental Dataset/Images/Raw'
    
    sd = 'Rec-0223'
    ns = ['frame_02650.png', 'frame_02660.png', 'frame_02770.png']
    T0, T0m = make(d, sd, ns, c=((56,-56),(112,-192)))
    
    sd = 'Rec-0175'
    ns = ['frame_02970.png', 'frame_02980.png', 'frame_02990.png']
    T1, T1m = make(d, sd, ns, c=((56,-56),(112,-192)))
    
    sd = 'Long Fiber Tow Center Edge'
    ns = ['frame_03000.png', 'frame_03010.png', 'frame_03020.png']
    T2, T2m = make(d, sd, ns, c=((56,-56),(112,-192)))

    
    d = r'C:\Users\Grayson\Docs\Repos\FP_Feedback_Control_Hardware'
    sd = ''
    ns = ['0.png', '1.png', '2.png']
    T3, T3m = make(d, sd, ns, c=(None,(250,-149)))
    
    T0m = cv2.cvtColor(np.round(T0m*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    T1m = cv2.cvtColor(np.round(T1m*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    T2m = cv2.cvtColor(np.round(T2m*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    T3m = cv2.cvtColor(np.round(T3m*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(d,'T0.png'), T0m)
    cv2.imwrite(os.path.join(d,'T1.png'), T1m)
    cv2.imwrite(os.path.join(d,'T2.png'), T2m)
    cv2.imwrite(os.path.join(d,'T3.png'), T3m)
    