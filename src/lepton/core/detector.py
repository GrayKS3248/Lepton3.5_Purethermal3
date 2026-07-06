# External modules
from copy import copy
import numpy as np
import cv2
from skimage.filters import threshold_yen as threshold
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

    def _get_kernel(self, shp, s=0.025, a=0.03, b=0.5):
        """
        Gets standard gaussian kernel parameters for blurring.

        Parameters
        ----------
        shp : tuple of (int, int)
            The shape of the input image.

        Returns
        -------
        size : tuple of (int, int)
        	Gaussian kernel size.
        sigma_x : float
            Gaussian kernel standard deviation in X direction.
        sigma_y : float
            Gaussian kernel standard deviation in Y direction.
        s : float, optional
            Tuning parameter. Decimal percent of maximum image dimension
            in pixels of gaussian filter kernel size. The default is 0.03.
        a : float, optional
            Tuning parameter. Multiplicative constant applied to kernel dimensions
            to get gaussian standard deviation. The default is 0.06.
        b : float, optional
            Tuning parameter. Addative constant applied to kernel dimensions
            to get gaussian standard deviation. The default is 0.40.

        """
        size = int(max(shp)*s)
        size = (size-(size%2-1),)*2
        sigma_x, sigma_y = tuple(np.array(size)*a + b)
        return size, sigma_x, sigma_y

    def _get_t_mask(self, temp, criteria, n_try, flags, min_temp):
        """
        Apply 4 mean thresholding on the temperature to detect
        0. Background
        1. Candidate front and bulk cured material at high temperatures
           near to the front
        2. Bulk cured material at high temperatures far from the front
        Define the t mask as 1

        Parameters
        ----------
        temp : mxn array-like
            The temperature field.
        criteria : 3 Tuple
            It is the iteration termination criteria. When this criteria is
            satisfied, algorithm iteration stops. See
            https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
        n_try : int
            Flag to specify the number of times the algorithm is executed using
            different initial labellings. The algorithm returns the labels that
            yield the best compactness.
        flags : int
            This flag is used to specify how initial centers are taken.
        min_temp : float or None.
            The minimum cutoff temperature below which pixels are no longer
            considered front. If None, ignored.

        Returns
        -------
        mask : mxn boolean array-like
            The candidate temperature-based front mask.

        """
        # Blur and flatten the temperature
        size, sigma_x, sigma_y = self._get_kernel(temp.shape)
        t = cv2.GaussianBlur(temp, ksize=size, sigmaX=sigma_x, sigmaY=sigma_y)
        t = t.flatten().reshape(-1,1).astype(np.float32)

        # Apply 4 mean thresholding on the temperature
        _, lab, cen = cv2.kmeans(t, 3, None, criteria, n_try, flags)
        lab = lab.reshape(temp.shape)
        cen = cen.flatten()
        cen = sorted(range(len(cen)), key=lambda k: cen[k])
        m = lab==cen[1]

        # Apply the minimum cutoff temperature
        if not min_temp is None:
            m = m & (temp>min_temp)
        return m

    def _get_g_mask(self, temp):
        """
        Apply 2 mean thresholding on L2 norm of the gradient of temperature
        to detect
        0. Bulk cured material and background
        1. Cured material recently interacted with front and colder front
           and front candidate
        Define the g mask as 1

        Parameters
        ----------
        temp : mxn array-like
            The temperature field.

        Returns
        -------
        mask : mxn boolean array-like
            The candidate gradient-based front mask.

        """
        # Get the L2 norm of the gradient of the temperature field
        dx, dy = np.gradient(temp)
        g = np.sqrt(dx*dx+dy*dy)

        # Blur and flatten the gradient
        size, sigma_x, sigma_y = self._get_kernel(temp.shape)
        g = cv2.GaussianBlur(g, ksize=size, sigmaX=sigma_x, sigmaY=sigma_y)

        # 2 means thresholding is identical to Yen's method
        return g > threshold(g)

    def _get_dt_mask(self, temps):
        """
        Apply 2 means thresholding on delta temperature to detect
        0. Bulk cured material and background
        1. Candidate front locations
        Define the dt mask as 1

        Parameters
        ----------
        temps : List of mxn array-like
            A time-ordered sequence of temperature images.

        Returns
        -------
        mask : mxn boolean array-like
            The candidate time derivative-based front mask.

        """
        # Calculate the blurred temporal differential of the temperature
        # image sequence by spatiotemporal gaussian differentiation
        _, sigma, _ = self._get_kernel(temps[-1].shape)
        dt = gaussian_filter(temps, 1.2*sigma, order=(1,0,0), mode='nearest')[-1]

        # Isolate regions that got hotter. The front won't get colder. Then
        # flatten
        dt[dt<0.0]=0.0

        # 2 means thresholding is identical to Yen's method
        return dt > threshold(dt)

    def _kmeans(self, ts, n_iter=32, epsilon=0.001, n_try=4, min_temp=50.0):
        """
        Uses automatic thresholding of the temperature image, the
        gradient of the temperature image, and the time derivative
        of the temperature image sequence to estimate front location

        Parameters
        ----------
        ts : list of array of floats, shape( (m,n) )
            Time ordered temperature images in Celcius.
        n_iter : int, optional
            Stop the kmeans algorithm after the specified number of iterations,
            n_iter. The default is 12.
        epsilon : float, optional
            Stop the kmeans algorithm if specified accuracy, epsilon,
            is reached. The default is 0.04.
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
        # Set kmeans criteria and flags
        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,n_iter,epsilon)
        flags = cv2.KMEANS_RANDOM_CENTERS

        # Get the candidate masks
        t_mask = self._get_t_mask(ts[-1], criteria, n_try, flags, min_temp)
        g_mask = self._get_g_mask(ts[-1])

        # If only one temperature image was provided, front estimate cannot
        # use derivative of temperature. Return intersection of temperature
        # mask and gradient mask
        if len(ts) == 1:
            return t_mask & g_mask

        # Get the candidate dt mask
        dt_mask = self._get_dt_mask(ts)
        t_mask_img = copy(ts[-1])
        t_mask_img[~t_mask]=20.0
        t_mask_img = np.round(255*(t_mask_img - 20.0)/160.0).astype(np.uint8)
        t_mask_img = np.reshape(np.tile(t_mask_img, 3), ts[-1].shape+(3,), order='F')
        cv2.imwrite('t_mask.png', t_mask_img)

        g_mask_img = copy(ts[-1])
        g_mask_img[~g_mask]=20.0
        g_mask_img = np.round(255*(g_mask_img - 20.0)/160.0).astype(np.uint8)
        g_mask_img = np.reshape(np.tile(g_mask_img, 3), ts[-1].shape+(3,), order='F')
        cv2.imwrite('g_mask.png', g_mask_img)

        dt_mask_img = copy(ts[-1])
        dt_mask_img[~dt_mask]=20.0
        dt_mask_img = np.round(255*(dt_mask_img - 20.0)/160.0).astype(np.uint8)
        dt_mask_img = np.reshape(np.tile(dt_mask_img, 3), ts[-1].shape+(3,), order='F')
        cv2.imwrite('dt_mask.png', dt_mask_img)

        mask_img = copy(ts[-1])
        mask_img = np.round(255*(mask_img - 20.0)/160.0).astype(np.uint8)
        mask_img = np.reshape(np.tile(mask_img, 3), ts[-1].shape+(3,), order='F')
        mask_img[t_mask & g_mask & dt_mask]=(0, 255, 0)
        cv2.imwrite('mask.png', mask_img)

        # Return intersection of all masks
        return t_mask & g_mask & dt_mask

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
