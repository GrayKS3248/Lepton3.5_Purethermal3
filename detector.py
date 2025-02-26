import numpy as np
import cv2
from sklearn.cluster import KMeans

class Detector():
    def __init__(self, T0=0.0, a0=0.0, Hr=3.50e5, Cp=1600.0):
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

        Returns
        -------
        None.

        """
        self.T0 = T0 # C
        self.a0 = a0 # -
        self.hr = Hr # J - Kg^{-1}
        self.cp = Cp # J - Kg^{-1} - K^{-1}
    
    def _kmeans(self, temperatures, n_iter=12, epsilon=0.04, n_try=8):
        """
        Uses automatic thresholding of the temperature image, the 
        gradient of the temperature image, and the time derivative
        of the temperature image sequence to estimate front location

        Parameters
        ----------
        temperature : list of array of floats, shape( (m,n) )
            Temperature image in Celcius.
        n_iter : int, optional
            Stop the kmeans algorithm after the specified number of iterations,
            n_iter. The default is 12.
        epsilon : float, optional
            Stop the kmeans algorithm if specified accuracy, epsilon, 
            is reached. The default is 0.04.
        n_try : int, optional
            Number of times the kmeans algorithm is executed using different
            initial labellings. The default is 8.

        Returns
        -------
        front_mask : array of bool, shape( (m,n) )
            A boolean mask of detected front instances.

        """
        # Set kmeans clustering parameters
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 
                    n_iter, 
                    epsilon)
        flags = cv2.KMEANS_RANDOM_CENTERS
        
        # Extract the most recent temperature 
        t = temperatures[-1]
        
        # Blur the temperature image
        s0 = int(np.round(t.shape[0]/40.))
        s1 = int(np.round(t.shape[1]/40.))
        size = (s0-(s0%2-1),s1-(s1%2-1))
        bt = cv2.GaussianBlur(t, size, 0)
        
        # Flatten the temperature for use in k means clustering
        # Convert to float32
        l = bt.shape[0]
        w = bt.shape[1]
        fbt = bt.reshape((l*w,1)).astype(np.float32)
        
        # Apply 4 mean thresholding on the temperature to detect
        # 1. Background
        # 2. Candidate front and bulk boundries
        # 3. Candidate front and bulk cured material at high temperatures near 
        #    to the front
        # 4. Bulk cured material at high temperatures far from the front
        _, t_lab, t_cen = cv2.kmeans(fbt, 4, None, criteria, n_try, flags)
        t_cen = t_cen.flatten()
        sort_t_cen = np.array(sorted(zip(t_cen, np.arange(0, len(t_cen)))))
        
        # Get the candidate front masked based on only temperature
        t_lab = t_lab.reshape((l,w))
        t_mask = np.logical_or(t_lab==sort_t_cen[1,1],
                               t_lab==sort_t_cen[2,1])
        
        # Get the L2 norm of the gradient of the temperature field
        dx, dy = np.gradient(bt)
        g = np.sqrt(np.square(dx)+np.square(dy))
        
        # Blur the L2 norm of the gradient
        bg = cv2.GaussianBlur(g, size, 0)
        
        # Flatten the l2 norm for use in k means clustering
        # Convert to float32
        fbg = bg.reshape((l*w,1)).astype(np.float32)
        
        # Apply 4 mean thresholding on L2 norm of temperature to detect
        # 1. Bulk cured material and background
        # 2. Bulk cured material boundaries
        # 3. Cured material recently interacted with front
        # 4. Front candidate
        _, g_lab, g_cen = cv2.kmeans(fbg, 4, None, criteria, n_try, flags)
        g_cen = g_cen.flatten()
        sort_g_cen = np.array(sorted(zip(g_cen, np.arange(0, len(g_cen)))))
        
        # Get the candidate front mask based on only the gradient
        g_lab = g_lab.reshape((l,w))
        g_mask = g_lab==sort_g_cen[3,1]
        
        # If only one temperature image was provided, front estimate cannot
        # use derivative of temperature. Return intersection of temperature
        # mask and gradient mask
        if len(temperatures) == 1:
            front_mask = np.logical_and(t_mask, g_mask)
            return front_mask
        
        # Calculate the time derivative of the blurred temperature sequence
        # based on 3 step finite difference
        elif len(temperatures) == 2:
            bt_m1 = cv2.GaussianBlur(temperatures[0], size, 0)
            bdt = -1.0*bt_m1 + 1.0*bt
        elif len(temperatures) >= 3:
            bt_m2 = cv2.GaussianBlur(temperatures[-3], size, 0)
            bt_m1 = cv2.GaussianBlur(temperatures[-2], size, 0)
            bdt = 0.5*bt_m2 - 2.0*bt_m1 + 1.5*bt
        
        # Isolate regions that got hotter. The front won't get colder.
        bdt[bdt<0.0]=0.0
        
        # Flatten the delta temperature for use in k means clustering
        # Convert to float32
        fbdt = bdt.reshape((l*w,1)).astype(np.float32)
        
        # Apply 2 means thresholding on delta temperature to detect
        # 1. Bulk cured material and background
        # 2. Candidate front locations
        _, dt_lab, dt_cen = cv2.kmeans(fbdt, 2, None, criteria, n_try, flags)
        dt_cen = dt_cen.flatten()
        sort_dt_cen = np.array(sorted(zip(dt_cen, np.arange(0, len(dt_cen)))))
        
        # Determine the automatically detected thresholding boundaries
        dt_bnds = []
        for i in sort_dt_cen[:,1]:
            mn=np.min(fbdt[dt_lab==i])
            mx=np.max(fbdt[dt_lab==i])
            dt_bnds.append([mn,mx])
            
        # Get the candidate front mask based on only delta temperature
        dt_lab = dt_lab.reshape((l,w))
        dt_mask = dt_lab.reshape((l,w))==sort_dt_cen[1,1]
        
        # The front mask is the intersection of all masks
        front_mask = np.logical_and(t_mask, g_mask)
        front_mask = np.logical_and(front_mask, dt_mask)
            
        # Return the detected front mask
        return front_mask
    
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
    