#imports
import numpy as np

class OverscanSubtraction(KPF0_Primitive):
    """
    This utility can perform various types of overscan subtraction on any given frame.
    """
    def __init__(self):
        """Initializes overscan subtraction utility.
        """

    def mean_subtraction(self, rawimage, overscan_reg): #should work now
        """Gets mean of overscan data, subtracts value from raw science image data.

        Args:
            rawimage(np.ndarray): Raw frame data
            overscan_reg(list): Overscan region

        Returns:
            raw_sub_os(np.ndarray): Raw image with overscan mean subtracted
        """
        raw_sub_os = np.copy(rawimage)
        raw_sub_os = raw_sub_os - np.mean(rawimage[:,overscan_reg].T,0,keepdims=True) #mean of overscan region (columns) of all rows, then transposed

        return raw_sub_os
    

    def linearfit_subtraction(self,rawimage, overscan_reg): #need to double check that this works w fixes
        """Performs linear fit on overscan data, subtracts fit values from raw science image data.

        Args:
            rawimage(np.ndarray): Raw frame data
            nlines(): Number of lines
            overscan_reg(list): Overscan region

        Returns:
            raw_sub_os(np.ndarray): Raw image with overscan fit subtracted
        """    
        nlines = rawimage.shape[0] #double check this
        raw_sub_os = np.copy(rawimage)
        fit = []
        fit_params=[]
        line_array = np.arange(nlines)

        fit_params.append(np.polyfit(line_array, np.mean(rawimage[:,overscan_reg].T,0),1))
        raw_sub_os = raw_sub_os - np.reshape(np.polyval(fit_params,line_array),(-1,1))
        
        return raw_sub_os

    #def polynomialfit_subtraction():