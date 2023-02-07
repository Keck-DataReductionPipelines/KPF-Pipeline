import numpy as np
import numpy.ma as ma
from astropy.io import fits
import matplotlib.pyplot as plt

# Pipeline dependencies
from kpfpipe.logger import *
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments

# Global read-only variables
DEFAULT_CFG_PATH = 'modules/Utils/overscan_subtract.cfg'
DEFAULT_CFG_PATH_RED = 'modules/Utils/overscan_subtract_red.cfg'
DEFAULT_CFG_PATH_GREEN = 'modules/Utils/overscan_subtract_green.cfg'

class OverscanSubtraction(KPF0_Primitive):

    """
    Description:
        This utility performs various types of overscan subtraction and then forms channel images
        into one full frame image (ffi).

        Assume the DATASEC FITS-keyword follows the normal convention of defining the inclusive ranges
        of the data section in terms of one-based column and row numbers, respectively.

        E.g., DATASEC = '[28:1179,1:4616]'

        In this example, the data are columns 28 through 1179, inclusive, and rows 1 through 4616, inclusive.

    Arguments:
        rawfile (L0 image object): Returned from call to kpf0_from_fits(...) function or this class (OverscanSubtraction).
        prl_overscan_reg (list of two int):  Defines row range of raw-image parallel overscan region, i.e.,
           start and end pixels of overscan after the channel image has been flipped so that the parallel overscan
           region is on top (e.g.,[4080,4180]).
           THIS PARAMETER IS CURRENTLY NOT USED BY THE CODE.
        srl_overscan_reg (list of two int):  Defines column range of raw-image serial overscan region, i.e.,
           start and end pixels of overscan after the channel image has been flipped so that the serial overscan (or
           post-overscan) region is on the right (e.g., [2040,2140]).
           THE VALUES OF THIS PARAMETER APPLY TO AFTER PRESCAN REGION HAS BEEN CHOPPED OFF.
           THIS MEANS srl_overscan_reg = [2040,2140] FOR GREEN_AMP1 WITH BIASEC1 = [1:4,1:2140] AND BIASEC3 = [2045:2144,1:2140].
        mode (str): Choice of method of overscan subtraction: median, polynomial, or clippedmean.
        order (int): If mode = 'polynomial', defines order of polynomial fit.
        oscan_clip_no (int): Number of pixels clipped from each edge of overscan.
        ref_output (object returned from class OrientationReference):  Contains output of ccd reference file.
        ffi_exts (list of str): Name of FITS extensions where full-frame images will be stored (e.g., ['GREEN_CCD'])
            NOTE: for KPF, the list contains just one element per call to this class because other input parameters
            may depend on the CCD.
        data_type (str): Type of data (e.g., KPF).
        prescan_reg (list of int): Prescan regions of raw image (e.g., [0,4]).  This is the python list indexes of the
            first columns in each row of the channel image that are to be chopped off (which can be derived from BIASSEC1).
            The last column is also chopped off unconditionally (not sure why).
        gain_key (str): Name of FITS keyword that stores amplifier gain (e-/DN).
        channel_datasec_ncols (int): Number of columns in data section of channel images for a given detector
            (default is 2040).
        channel_datasec_nrows (int): Number of rows in data section of channel images for a given detector
            (default is 2040 for GREEN_CCD or otherwise 4080 for RED_CCD).
        n_sigma (float): Number of sigmas for overscan-value outlier rejection (default is 2.5).
    """

    def __init__(self,action,context):

        """
        Initializes overscan subtraction utility.
        """

        KPF0_Primitive.__init__(self, action, context)
        self.rawfile = self.action.args[0]
        self.prl_overscan_reg = self.action.args[1]
        self.srl_overscan_reg = self.action.args[2]
        self.mode = self.action.args[3]
        self.order = self.action.args[4]
        self.oscan_clip_no = self.action.args[5]
        self.ref_output = self.action.args[6]
        self.ffi_exts = self.action.args[7]
        self.data_type = self.action.args[8]
        self.prescan_reg = self.action.args[9]
        self.gain_key = self.action.args[10]

        try:
            self.channel_datasec_ncols = self.action.args[11]
        except:
            self.channel_datasec_ncols = 2040

        try:
            self.channel_datasec_nrows = self.action.args[12]
        except:
            if self.ffi_exts[0] == 'GREEN_CCD':
                self.channel_datasec_nrows = 2040
            else:
                self.channel_datasec_nrows = 4080

        try:
            self.n_sigma = self.action.args[13]
        except:
            self.n_sigma = 2.5
            
        self.module_config_path = DEFAULT_CFG_PATH
        if self.ffi_exts[0] == 'RED_CCD':
            self.module_config_path = DEFAULT_CFG_PATH_RED
        elif self.ffi_exts[0] == 'GREEN_CCD':
            self.module_config_path = DEFAULT_CFG_PATH_GREEN
    
        print("{} class: self.module_config_path = {}".format(self.__class__.__name__,self.module_config_path))

        print("Starting logger...")
        self.logger = start_logger(self.__class__.__name__, self.module_config_path)

        if self.logger is not None:
            print("--->self.logger is not None...")
        else:
            print("--->self.logger is None...")

        self.logger.info('Started {}'.format(self.__class__.__name__))
        self.logger.debug('module_config_path = {}'.format(self.module_config_path))


        ########################################################################################
        """

        12/11/22
        Since the L0 FITS-headers do not have the DATASEC and BIASSEC keywords, we assume
        reasonable GUESSES for them here now as an expedient in the interum.
        There are four green readout channels and two red readout channels.
        The entire green or red CCD is 4288x4280 pixels with overscan regions included.
        The CCD gain varies with readout channel.

        Here is the useful information currently in the L0 readout-channel FITS extensions:
        EXTNAME = 'GREEN_AMP1'
        AMPSEC  = '[1:2144,1:2140]'
        CCDGAIN =                  5.2 / Current measured amplifier gain in e-/DN Abby S

        EXTNAME = 'GREEN_AMP2'
        AMPSEC  = '[2145:4288,1:2140]' / amplifier section
        CCDGAIN =                 5.32 / Current measured amplifier gain in e-/DN Abby S

        EXTNAME = 'GREEN_AMP3'
        AMPSEC  = '[1:2144,2141:4280]' / amplifier section
        CCDGAIN =                  5.3 / Current measured amplifier gain in e-/DN Abby S

        EXTNAME = 'GREEN_AMP4'
        AMPSEC  = '[2145:4288,2141:4280]' / amplifier section
        CCDGAIN =                  5.4 / Current measured amplifier gain in e-/DN Abby S

        EXTNAME = 'RED_AMP1'
        AMPSEC  = '[1:2144,1:4180]'    / amplifier section
        CCDGAIN =                  5.1 / Current measured amplifier gain in e-/DN Abby S

        EXTNAME = 'RED_AMP2'
        AMPSEC  = '[2145:4288,1:4180]' / amplifier section
        CCDGAIN =                 5.21 / Current measured amplifier gain in e-/DN Abby S


        Correct channel-image orientation ascertained by looking at a recent L0 channel image:

        Channel_ext    Flip-needed    Channel_key       Channel_row   Channel_col
        GREEN_AMP1     none               4                  1            1
        GREEN_AMP2     lr                 1                  1            2
        GREEN_AMP3     ud                 3                  2            1
        GREEN_AMP4     lr & ud            2                  2            2
        RED_AMP1       none               4                  1            1
        RED_AMP2       lr                 1                  1            2


        According to NEID documentation (https://neid.ipac.caltech.edu/docs/NEID-DRP/algorithms.html),
        which we will adopt here, there are overscan, prescan, and post readout regions, defined by
        BIASSEC1 (left), BIASSEC2 (top or bottom), and BIASSEC3 (right) keywords for each readout channel
        (in the original orientation of the channel image, not flipped yet for tiling).

        Reverse-engineered the DATASEC and BIASSEC# keywords from a recent L0 channel image:

        Channel_ext   DATASEC               BIASSEC1          BIASSEC2               BIASSEC3
        GREEN_AMP1    [5:2044,1:2040]       [1:4,1:2140]      [5:2044,2041:2140]     [2045:2144,1:2140]
        GREEN_AMP2    [101:2140,1:2040]     [1:100,1:2140]    [101:2140,2041:2140]   [2141:2144,1:2140]
        GREEN_AMP3    [5:2144,101:2140]     [1:4,1:2140]      [5:2044,1:100]         [2045:2144,1:2140]
        GREEN_AMP4    [101:2140,101:2140]   [1:100,1:2140]    [101:2140,1:100]       [2141:2144,1:2140]
        RED_AMP1      [5:2044,1:4080]       [1:4,1:4180]      [5:2044,4081:4180]     [2045:2144,1:4180]
        RED_AMP1      [101:2140,1:4080]     [1:100,1:4180]    [101:2140,4081:4180]   [2141:2144,1:4180]

        """
        ########################################################################################

    def overscan_arrays(self):

        """
        Makes array of overscan pixels. For example, if raw image including overscan region
        is 1000 pixels wide, and overscan region (when oriented with overscan on right and parallel
        overscan on bottom) is 100 pixels wide, it will make an array of values 900-1000.

        Returns:
            srl_overscan_pxs(np.ndarray): Array of serial overscan region pixels
            prl_overscan_pxs(np.ndarray): Array of parallel overscan region pixels
            srl_overscan_clipped(np.ndarray): Array of clipped serial overscan region pixels
            prl_overscan_clipped(np.ndarray): Array of clipped parallel overscan region pixels
        """

        srl_overscan_pxs = np.arange(self.srl_overscan_reg[0],self.srl_overscan_reg[1],1)
        prl_overscan_pxs = np.arange(self.prl_overscan_reg[0],self.prl_overscan_reg[1],1)
        srl_N_overscan = len(srl_overscan_pxs)
        prl_N_overscan = len(prl_overscan_pxs)
        srl_overscan_clipped = srl_overscan_pxs[self.oscan_clip_no:srl_N_overscan-1-self.oscan_clip_no]
        prl_overscan_clipped = prl_overscan_pxs[self.oscan_clip_no:prl_N_overscan-1-self.oscan_clip_no]

        return srl_overscan_pxs,prl_overscan_pxs,srl_overscan_clipped,prl_overscan_clipped

    def median_subtraction(self,image,overscan_reg):

        """
        Gets median of overscan data, subtracts value from raw science image data.

        Args:
            image(np.ndarray): Array of image data
            overscan_reg(np.ndarray): Array of pixel range of overscan relative to image pixel width

        Returns:
            raw_sub_os(np.ndarray): Raw image with overscan median subtracted
        """

        raw_sub_os = np.zeros_like(image)
        for row in range(0,raw_sub_os.shape[0]):
            raw_sub_os[row,:] = image[row,:] - np.median(image[row,overscan_reg],keepdims=True)

        return raw_sub_os

    def clippedmean_subtraction(self,image,overscan_reg,ext):

        """
        Gets clipped mean of overscan data, subtracts value from raw image data.
        Data-value clipping for outliers is defined at +/- n_sigma (float) times sigma of the
        data, given by a robust estimator of image-data dispersion, sigma = 0.5 * (p84 - p16).

        Computes the clipped mean over the entire post-overscan strip (not row by row).

        Args:
            image(np.ndarray): Array of image data
            overscan_reg(np.ndarray): Array of pixel range of overscan relative to image pixel width

        Returns:
            raw_sub_os(np.ndarray): Raw image with overscan clipped-mean subtracted
        """

        a = image[:,overscan_reg]
        n_sigma = self.n_sigma
        frame_data_shape = np.shape(a)

        med = np.median(a)
        p16 = np.percentile(a,16)
        p84 = np.percentile(a,84)
        sigma = 0.5 * (p84 - p16)
        mdmsg = med - n_sigma * sigma
        b = np.less(a,mdmsg)
        mdpsg = med + n_sigma * sigma
        c = np.greater(a,mdpsg)
        mask = b | c
        mx = ma.masked_array(a, mask)
        avg = ma.getdata(mx.mean())
        overscan_clipped_mean[ext] = avg.item()

        self.logger.debug('---->{}.clippedmean_subtraction(): ext,overscan_reg,n_sigma,p16,med,p84,sigma,avg = {},{},{},{},{},{},{},{}'.\
            format(self.__class__.__name__,ext,overscan_reg,n_sigma,p16,med,p84,sigma,avg))

        raw_sub_os = image - avg

        return raw_sub_os


    def polyfit_subtraction(self,image,overscan_reg): #need to double check that this works w fixes

        """Performs linear fit on overscan data, subtracts fit values from raw science image data.

        Args:
            image(np.ndarray): Array of image data
            overscan_reg(np.ndarray): Array of pixel range of overscan relative to image pixel width

        Returns:
            raw_sub_os(np.ndarray): Raw image with overscan fit subtracted
        """

        xx = np.arange(image.shape[0]) #double check this
        raw_sub_os = np.zeros(image.shape)
        means = []
        for row in xx:
            mean = np.mean(image[row,overscan_reg])
            means.append(mean)

        polyfit = np.polyfit(xx,means,self.order)
        polyval = np.polyval(polyfit,xx)
        reshape = np.reshape(polyval,(-1,1))
        for row in xx:
            raw_sub_os[row] = image[row] - reshape[row]

        return raw_sub_os

    def orientation_adjust(self,image,key): #check transposing

        """
        Extracts and flips images to regularlize readout orientation for overscan subtraction.

        Args:
            image(np.ndarray): Raw image with overscan region
            key(int): Orientation of image

        Returns:
            image_fixed(np.ndarray): Correctly-oriented raw image for overscan subtraction
                and overscan region removal
        """

        if key == 1: #flip lr
            image_fixed = np.flip(image,axis=1)
        if key == 2: #turn upside down and flip lr
            image_fixed = np.flip(np.flip(image,axis=0),axis=1)
        if key == 3: #turn upside down
            image_fixed = np.flip(image,axis=0)
        if key == 4: #no change
            image_fixed = image

        return image_fixed

    def generate_FFI(self, images,rows,columns):

        """
        Generates full frame image from channel images.

        Args:
            images(list): List of arrays corresponding to channel data
            rows(list): List of rows corresponding to channel image FFI location
            columns(list): List of columns corresponding to channel image FFI location.
                Ex: Quadrant row is 2, col is 2, means that image will go lower right corner in FFI.

        Returns:
            full_frame_img(np.ndarray): Assembled full frame image
        """

        all_img = list(zip(images,rows,columns))
        h_stacked = []

        for row in range(1,np.max(rows)+1):
            output = [img for img in all_img if img[1] == row]
            sorted_output = sorted(output,key = lambda x: x[2])
            a,b,c = list(zip(*sorted_output))
            h_stack = np.hstack(a)
            h_stacked.append(h_stack)

        full_frame_img = np.vstack(h_stacked)

        return full_frame_img

    def overscan_cut(self,osub_image,channel_datasec_nrows,channel_datasec_ncols):

        """
        Cuts overscan region off of overscan-subtracted image.

        Args:
            osub_image(np.ndarray): Image with overscan region subtracted.
            overscan_reg_srl(np.ndarray): Serial overscan region
            overscan_reg_prl(np.ndarray): Parallel overscan region

        Returns:
            image_cut(np.ndarray): Image with overscan region cut off.
        """

        image_cut = osub_image[:,0:channel_datasec_ncols]
        image_cut = image_cut[0:channel_datasec_nrows,:]

        return image_cut

    def run_oscan_subtraction(self,channel_imgs,channels,channel_keys,channel_rows,channel_cols,channel_exts):

        """
        Performs overscan subtraction steps, in order: orient frame, subtract overscan (method
        chosen by user) from correctly-oriented frame (overscan on right and bottom), cuts off overscan region.

        Args:
            channel_imgs(np.ndarray): All extension images that make up a single FFI
            channels(list): Channel number
            channel_keys(list): Channel orientation key values (1=overscan on bottom and left, 2=overscan on left and top
                3=overscan on top and right, 4=overscan on right and bottom)
            channel_rows(list): List of rows corresponding to channel image FFI location
            channel_cols(list): List of columns corresponding to channel image FFI location
            channel_exts(list): FITS extensions of images

        Returns:
            full_frame_img(np.ndarray): Stiched-together full frame image, with overscan subtracted and removed
        """

        # clip ends of overscan region
        srl_oscan_pxl_array,prl_oscan_pxl_array,srl_clipped_oscan,prl_clipped_oscan = self.overscan_arrays()

        # create empty list for final, overscan subtracted/cut arrays
        no_overscan_imgs = []

        n_channel_images = len(channel_imgs)
        self.logger.debug("=============> n_channel_images = {}".format(n_channel_images))

        for img,key,ext in zip(channel_imgs,channel_keys,channel_exts):

            ###gain addition###


            ##########

            self.logger.debug('---->{}.run_oscan_subtraction(): ext = {}'.\
                format(self.__class__.__name__,ext))

            new_img_w_prescan = self.orientation_adjust(img,key)
            new_img = new_img_w_prescan[:,self.prescan_reg[1]:-1]

            # overscan subtraction for chosen method
            if self.mode == 'median':
                raw_sub_os = self.median_subtraction(new_img,srl_clipped_oscan)
            elif self.mode == 'clippedmean':
                raw_sub_os = self.clippedmean_subtraction(new_img,srl_clipped_oscan,ext)
            elif self.mode == 'polynomial': # subtract linear fit of overscan
                raw_sub_os = self.polyfit_subtraction(new_img,srl_clipped_oscan)
            else:
                raise TypeError('Input overscan subtraction mode set to value outside options.')

            # chop off overscan and prescan - put into overscan subtraction utility
            new_img = self.overscan_cut(raw_sub_os,self.channel_datasec_nrows,self.channel_datasec_ncols)

            # put img back into original orientation
            og_oriented_img = self.orientation_adjust(new_img,key)
            plt.imshow(og_oriented_img)

            no_overscan_imgs.append(og_oriented_img)

        full_frame_img = self.generate_FFI(no_overscan_imgs,channel_rows,channel_cols)

        return full_frame_img

    def _perform(self):

        """
        Performs entire overscan subtraction utility.

        Returns:
            l0_obj(fits.hdulist): Original FITS.hdulist but with FFI extension(s) filled,
            and tiled-together channel extensions removed.
        """

        channels,channel_keys,channel_rows,channel_cols,channel_exts = self.ref_output

        if self.data_type == 'KPF':
            l0_obj = self.rawfile
            NoneType = type(None)

            try:
                if len(l0_obj[channel_exts[0]]) == 0:
                    return Arguments(l0_obj)
            except:
                pass
            try:
                if isinstance(l0_obj[channel_exts[0]],NoneType) == True:
                    return Arguments(l0_obj)
            except:
                pass

            else:
                frames_data = []
                for ext in channel_exts:
                    data = l0_obj[ext]
                    gain = l0_obj.header[ext][self.gain_key]
                    data = data / (2**16) #don't make hardcoded? only ok for now, output a warning here
                    #####
                    data_gain_corr = data * gain
                    frames_data.append(data_gain_corr)
                frames_data = np.array(frames_data)

                if self.mode == 'clippedmean':
                    global overscan_clipped_mean
                    overscan_clipped_mean = {}

                for frame in range(len(self.ffi_exts)):

                    self.logger.debug('---->{}._perform(): frame = {}'.\
                        format(self.__class__.__name__,frame))

                    single_frame_data = np.array_split(frames_data,len(self.ffi_exts))[frame]
                    full_frame_img = self.run_oscan_subtraction(single_frame_data,channels,channel_keys,channel_rows,channel_cols,channel_exts)
                    l0_obj[self.ffi_exts[frame]] = full_frame_img
                    l0_obj.header[self.ffi_exts[frame]]['BUNIT'] = ('electrons','Units of image data')

                    if self.mode == 'clippedmean':
                        i = 1
                        for key in overscan_clipped_mean:
                            keywrd = "OSCANV" + str(i)
                            keyval = overscan_clipped_mean[key]
                            keycmt = "Overscan clipped mean (e-), " + key
                            l0_obj.header[self.ffi_exts[frame]][keywrd] = (keyval, keycmt)
                            i = i + 1

                for ext in channel_exts:
                    l0_obj.del_extension(ext)

                self.logger.debug('---->{}._perform(): Done with overscan_subtraction.py; returning...'.\
                    format(self.__class__.__name__))

                return Arguments(l0_obj)
