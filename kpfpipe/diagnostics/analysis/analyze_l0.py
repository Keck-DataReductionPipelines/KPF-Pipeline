import re
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from kpfpipe.modules.Utils.kpf_parse import HeaderParse, get_data_products_L0
from datetime import datetime

class AnalyzeL0:
    """
    Description:
        This class contains functions to analyze L0 images (storing them
        as attributes).

    Arguments:
        L0 - an L0 object

    Attributes:
        amp_names (list) - names of the amplifier regions present in the L0 image
        gain (dictionary) - gain values for amplifier regions (determined by Ashley Baker)
        regions (list of strings) - list of amplifier regions in L0
        nregions_green (int) - number of amplifier regions present in Green CCD image
        nregions_red (int) - number of amplifier regions present in Red CCD image
        green_present (boolean) - True if the Green CCD image is in the L0 object
        red_present (boolean) - True if the Red CCD image is in the L0 object
        read_noise_overscan (dictionary) - read noise estimates in e- determined 
            by std of overscan regions.  The dictionary is over the list of regions
        std_mad_norm_ratio_overscan (dictionary) - read noise metric equal to 
            the (0.7979*stdev/mad), where stdev is the standard deviation of 
            a given overscan region, mad is the mean absolute deviation of a 
            given overscan region.  This should be = 1.00 for Gaussian noise.
            The dictionary is over the list of regions.
         
    """

    def __init__(self, L0):
        self.L0 = copy.deepcopy(L0)
        self.data_products = get_data_products_L0(L0)
        primary_header = HeaderParse(L0, 'PRIMARY')
        self.header = primary_header.header
        self.name = primary_header.get_name()
        self.ObsID = primary_header.get_obsid()
        if ('Green' in self.data_products) or ('Red' in self.data_products):
            self.regions = [key for key in self.L0.extensions if 'AMP' in key]
            bad_regions = [] # remove regions with no data
            for region in self.regions:
                if len(self.L0[region].data) == 0:
                    bad_regions.append(region) # the bad region can't be removed in this loop, so a list is created
            if bad_regions != []:
                for region in bad_regions:
                    self.regions.remove(region)
            self.determine_regions() # sets self.green_present, self.red_present, nregions_green, nregions_red
            # TO-DO: read gain values from elsewhere in the repo
            self.gain = {
                 'GREEN_AMP1': 5.175,
                 'GREEN_AMP2': 5.208,
                 'GREEN_AMP3': 5.52,
                 'GREEN_AMP4': 5.39,
                 'RED_AMP1': 5.02,
                 'RED_AMP2': 5.27,
                 'RED_AMP3': 5.32,
                 'RED_AMP4': 5.23,
            }
            self.read_noise_overscan = {} # initialize dictionary
            self.std_mad_norm_ratio_overscan = {} # initialize dictionary
            try:
                self.measure_read_noise_overscan()
            except Exception as e:
                print(e)
                print('measure_read_noise_overscan() failed on ' + self.ObsID)
            try:
                self.measure_std_mad_norm_ratio_overscan()
            except Exception as e:
                print(e)
                print('measure_std_mad_norm_ratio_overscan() failed on ' + self.ObsID)
            
            self.read_speed, self.green_acf, self.red_acf, self.green_read_time, self.red_read_time = \
                  primary_header.get_read_speed()
            if self.green_read_time == 0:
                 self.green_read_time = None
            if self.red_read_time == 0:
                 self.red_read_time = None
    
    def reject_outliers(self, data, n=5.0):
        """
        This method performs an iterative n-sigma outlier rejection 
        and is used in measure_read_noise_overscan().
        
        Parameters:
            data (list or np.array): Input data set
            n (float): Number of standard deviations from the mean. 
                       Data points outside this range are considered outliers.
        
        Returns:
            np.array: Data set with outliers removed.
        """
        data = np.array(data)
        
        while True:
            mean, std = np.mean(data), np.std(data)
            outliers = (data < mean - n * std) | (data > mean + n * std)
            if not np.any(outliers):
                break
            data = data[~outliers]
        
        return data


    def determine_regions(self):
        """
        This method determines if the Green and Red CCD images are present in the 
        AnalyzeL0 object and counts the number of amplifier regions in each CCD image.
        These results are written to attributes of the AnalyzeL0 object.
        
        Parameters:
            None 

        Attributes:
            self.green_present (boolean) - True if the Green CCD image (of non-zero size) is in the L0 object
            self.red_present (boolean) - True if the Red CCD image (of non-zero size) is in the L0 object
            self.nregions_green (int) - number of amplifier regions present in Green CCD image
            self.nregions_red (int) - number of amplifier regions present in Red CCD image

        Returns:
            None
        """
        
        print('Testing...')
        
        # Determine if specific amplifier regions are present and count them
        for CHIP in ['GREEN', 'RED']:
            nregions = 0
            amp1_present = False
            amp2_present = False
            amp3_present = False
            amp4_present = False
            if CHIP + '_AMP1' in self.L0.extensions:
                if self.L0[CHIP + '_AMP1'].shape[0] > 0: # Sanity check that data is present
                    amp1_present = True
            if CHIP + '_AMP2' in self.L0.extensions:
                if self.L0[CHIP + '_AMP2'].shape[0] > 0:
                    amp2_present = True
            if CHIP + '_AMP3' in self.L0.extensions:
                if self.L0[CHIP + '_AMP3'].shape[0] > 0:
                    amp3_present = True
            if CHIP + '_AMP4' in self.L0.extensions:
                if self.L0[CHIP + '_AMP4'].shape[0] > 0:
                    amp4_present = True
            if not amp1_present:
                nregions = 0
            if amp1_present:
                nregions = 1
            if amp1_present & amp2_present:
                nregions = 2
            if amp1_present & amp2_present & amp3_present & amp4_present:
                nregions = 4
            if CHIP == 'GREEN':
                self.nregions_green = nregions
                if nregions >= 1:
                    self.green_present = True
                else:
                    self.green_present = True
            if CHIP == 'RED':
                self.nregions_red = nregions
                if nregions >= 1:
                    self.red_present = True
                else:
                    self.red_present = True


    def measure_read_noise_overscan(self, nparallel=30, nserial=50, nsigma=10, verbose=False): 
        """
        Measure read noise in the overscan region of a KPF CCD image. 
        Read noise is measured as the standard deviation of the pixel values, 
        after applying an n-sigma outlier rejection.
        
        Args:
            nparallel (integer) - overscan length in parallel direction 
                                  (30 pixels was final value in 2023)
            nserial (integer) - overscan length in serial direction 
            nsigma (float) - number of sigma for outlier rejection method
    
        Attributes:
            self.read_noise_overscan - dictionary of read noise values in self.regions
    
        Returns:
            None
        """
        
        for region in self.regions:
            CHIP = region.split('_')[0]
            NUM = region.split('AMP')[1]
            gain = self.gain[CHIP+'_AMP'+NUM]
            data = self.L0[region]
            if np.nanmedian(data) > 200*2**16:  # divide by 2**16 if needed
                data /= 2**16
            if region == 'GREEN_AMP2':
                data = data[:,::-1]  # flip so that overscan is on the right
            if region == 'GREEN_AMP3':
                data = data[::-1,:]
            if region == 'GREEN_AMP4':
                data = data[::-1,::-1]
            if region == 'RED_AMP2':
                data = data[:,::-1]
            if region == 'RED_AMP3':
                data = data 
            if region == 'RED_AMP4':
                data = data[:,::-1]
            overscan_region = data[5:2040 + nparallel-5,2044 + 10:2044 + nserial-10]
            vals = gain * self.reject_outliers(overscan_region.flat, nsigma)    
            self.read_noise_overscan[region] = np.std(vals)
            if verbose:
                print(f'Read noise({region}) = {self.read_noise_overscan[region]} e-')


    def measure_std_mad_norm_ratio_overscan(self, nparallel=30, nserial=50, nsigma=10, verbose=False): 
        """
        Measure a read noise metric equal to (0.7979*stdev/mad), where stdev 
        is the standard deviation of a given overscan region, mad is the mean 
        absolute deviation of a given overscan region.  This should be = 1.00 
        for Gaussian noise.  The dictionary is over the list of regions.
        
        Args:
            nparallel (integer) - overscan length in parallel direction 
                                  (30 pixels was final value in 2023)
            nserial (integer) - overscan length in serial direction 
            nsigma (float) - number of sigma for outlier rejection method
    
        Attributes:
            self.std_mad_norm_ratio_overscan - dictionary of non-Gaussian 
                read noise metrics, 0.7979 * stdev(region) / mad(region),
                where region is one of the overscan regions 
    
        Returns:
            None
        """

        for region in self.regions:
            CHIP = region.split('_')[0]
            NUM = region.split('AMP')[1]
            gain = self.gain[CHIP+'_AMP'+NUM]
            data = self.L0[region]
            if np.nanmedian(data) > 200*2**16:  # divide by 2**16 if needed
                data /= 2**16
            if region == 'GREEN_AMP2':
                data = data[:,::-1]  # flip so that overscan is on the right
            if region == 'GREEN_AMP3':
                data = data[::-1,:]
            if region == 'GREEN_AMP4':
                data = data[::-1,::-1]
            if region == 'RED_AMP2':
                data = data[:,::-1]
            if region == 'RED_AMP3':
                data = data 
            if region == 'RED_AMP4':
                data = data[:,::-1]
            overscan_region = data[5:2040 + nparallel-5,2044 + 10:2044 + nserial-10]
            vals = gain * self.reject_outliers(overscan_region.flat, nsigma)    
            std = np.std(vals)
            mad = np.mean(np.abs(vals - np.mean(vals)))
            special_number = 0.7978845608 # sqrt(2/pi)
            self.std_mad_norm_ratio_overscan[region] = special_number*std/mad
            if verbose:
                print(f'Ratio of 0.7979 * stdev({region}) / mad({region})  = {self.std_mad_norm_ratio_overscan[region]}')
