#imports
import numpy as np
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments
class OverscanSubtraction(KPF0_Primitive):
    """
    This utility can perform various types of overscan subtraction and then form channel images 
    into one full frame image .
    """
    def __init__(self, action, context):
        """Initializes overscan subtraction utility.
        """
        KPF0_Primitive.__init__(self, action, context)
        self.channel_imgs = self.action.args[0] #raw channel images
        self.overscan_reg = self.action.args[1] #overscan region of raw image, start and end pixels of overscan
        self.mode = self.action.args[2] #defines which method of overscan subtraction
        self.keys = self.action.args[3] #defines orientation of input image
        self.oscan_clip_no = self.action.args[4] #amount of pixels clipped from each edge of overscan
        self.rows = self.action.args[5] #rows in FFI
        self.columns = self.action.args[6] #columns in FFI

    def overscan_arrays(self):
        """Makes array of overscan pixels. For example, if raw image including overscan region
        is 1000 pixels wide, and overscan region (when oriented with overscan on right and parallelscan
        on bottom) is 100 pixels wide, 
        """
        overscan_pxs = np.arange(self.overscan_reg[0],self.overscan_reg[1],1)
        N_overscan = len(overscan_pxs)
        overscan_clipped = overscan_pxs[self.oscan_clip_no:N_overscan-1-self.oscan_clip_no]  #put into config (5) # chop m:n samples off the ends of the oscan region
        return overscan_pxs,overscan_clipped

    def mean_subtraction(self,image,overscan_reg): #should work now
        """Gets mean of overscan data, subtracts value from raw science image data.

        Returns:
            raw_sub_os(np.ndarray): Raw image with overscan mean subtracted
        """
        raw_sub_os = np.copy(image)
        raw_sub_os = raw_sub_os - np.mean(image[:,overscan_reg].T,0,keepdims=True) 
        #mean of overscan region (columns) of all rows, then transposed

        return raw_sub_os
    

    def linearfit_subtraction(self,image,overscan_reg): #need to double check that this works w fixes
        """Performs linear fit on overscan data, subtracts fit values from raw science image data.

        Args:
            overscan_reg(np.ndarray): Array of pixel range of overscan relative to image pixel width

        Returns:
            raw_sub_os(np.ndarray): Raw image with overscan fit subtracted
        """    
        nlines = image.shape[0] #double check this
        raw_sub_os = np.copy(image)
        fit = []
        fit_params=[]
        line_array = np.arange(nlines)

        fit_params.append(np.polyfit(line_array, np.mean(image[:,overscan_reg].T,0),1))
        raw_sub_os = raw_sub_os - np.reshape(np.polyval(fit_params,line_array),(-1,1))
        
        return raw_sub_os


    def orientation_adjust(self,image,key): #check transposing
        """ Extracts and flips images to regularlize readout orientation for overscan subtraction.

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

    def generate_FFI(self, images):
        """Generates full frame image from channel images.

        Args:
            images(list): List of arrays corresponding to channel data

        Returns:
            full_frame_img(np.ndarray): Assembled full frame image
        """

        all_img = list(zip(images,self.rows,self.columns))
        h_stacked = []

        for row in range(1,np.max(self.rows)+1):
            output = [img for img in all_img if img[1] == row]
            sorted_output = sorted(output,key = lambda x: x[2])
            a,b,c = list(zip(*sorted_output))
            h_stack = np.hstack(a)
            h_stacked.append(h_stack)

        full_frame_img = np.vstack(h_stacked)

        return full_frame_img

    def overscan_cut(self, osub_image):
        """Cuts overscan region off of overscan-subtracted image.

        Args:
            osub_image(np.array):

        Returns:
            image_cut(np.array): 

        """
        image_cut = osub_image[:,0:self.overscan_reg[0]]

        return image_cut

    def _perform(self): #iterate through wrapped frames
        """Performs overscan subtraction steps, in order: orient frame, subtract overscan (method
        chosen by user) from correctly-oriented frame (overscan on right and bottom), cuts off overscan region.
        """
        # clip ends of overscan region 
        oscan_pxl_array,clipped_oscan = self.overscan_arrays()

        # create empty list for final, overscan subtracted/cut arrays
        no_overscan_imgs = []

        for img,key in zip(self.channel_imgs,self.keys):
            # orient img
            new_img = orientation_adjust(img,key)

            # overscan subtraction for chosen method
            if self.mode==0:
                raw_sub_os = self.mean_subtraction(new_img,clipped_oscan)

            elif self.mode==1: # subtract linear fit of overscan
                raw_sub_os = self.linearfit_subtraction(new_img,clipped_oscan)

            else:
                raise TypeError('Input overscan subtraction mode set to value outside options.')

            # chop off overscan and prescan - put into overscan subtraction utility
            new_img = self.overscan_cut(raw_sub_os)

            # put img back into original orientation 
            og_oriented_img = orientation_adjust(new_img,key)

            no_overscan_imgs.append(og_oriented_img)

        full_frame_img = generate_FFI(no_overscan_imgs)

        return Arguments(full_frame_img)





    

