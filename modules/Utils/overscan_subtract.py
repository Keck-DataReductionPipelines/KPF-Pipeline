
#imports
import numpy as np
from astropy.io import fits
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments
class OverscanSubtraction(KPF0_Primitive):
    """
    This utility can perform various types of overscan subtraction and then form channel images 
    into one full frame image .
    """
    def __init__(self,action,context):
        """Initializes overscan subtraction utility.
        """
        KPF0_Primitive.__init__(self, action, context)
        self.rawfile = self.action.args[0]
        #self.channel_imgs = self.action.args[0] #raw channel images
        self.overscan_reg = self.action.args[1] #overscan region of raw image, start and end pixels of overscan
        self.mode = self.action.args[2] #defines which method of overscan subtraction
        self.order = self.action.args[3] #if self.mode = 'polynomial', defines order of polynomial fit
        self.oscan_clip_no = self.action.args[4] #amount of pixels clipped from each edge of overscan
        self.ref_output=self.action.args[5] #output of ccd reference file
        self.ffi_exts=self.action.args[6] #fits extensions where ffis will be stored
        self.data_type=self.action.args[7] #data type, pertaining to instrument

    def overscan_arrays(self):
        """Makes array of overscan pixels. For example, if raw image including overscan region
        is 1000 pixels wide, and overscan region (when oriented with overscan on right and parallelscan
        on bottom) is 100 pixels wide, 
        """
        overscan_pxs = np.arange(self.overscan_reg[0],self.overscan_reg[1],1)
        N_overscan = len(overscan_pxs)
        overscan_clipped = overscan_pxs[self.oscan_clip_no:N_overscan-1-self.oscan_clip_no]
        return overscan_pxs,overscan_clipped

    def mean_subtraction(self,image,overscan_reg): #should work now
        """Gets mean of overscan data, subtracts value from raw science image data.

        Returns:
            raw_sub_os(np.ndarray): Raw image with overscan mean subtracted
        """
        #raw_sub_os = np.zeros((image.shape[0],image.shape[1]))
        raw_sub_os = np.zeros_like(image)
        for row in range(0,raw_sub_os.shape[0]):
            raw_sub_os[row] = image[row] - np.mean(image[row,overscan_reg],0,keepdims=True) 
        #mean of overscan region (columns) of all rows, then transposed

        return raw_sub_os
    

    def polyfit_subtraction(self,image,overscan_reg): #need to double check that this works w fixes
        """Performs linear fit on overscan data, subtracts fit values from raw science image data.

        Args:
            overscan_reg(np.ndarray): Array of pixel range of overscan relative to image pixel width

        Returns:
            raw_sub_os(np.ndarray): Raw image with overscan fit subtracted
        """    
        nlines = image.shape[0] #double check this
        # raw_sub_os = np.copy(image)
        raw_sub_os = np.zeros((image.shape[0],image.shape[1]))
        fit = []
        fit_params=[]
        line_array = np.arange(nlines)
        #if we do method = 1 (for linearfit), or do method = polynomial, then provide order
        fit_params.append(np.polyfit(line_array, np.mean(image.T[:,overscan_reg],self.order),1))
        raw_sub_os = image - np.reshape(np.polyval(fit_params,line_array),(-1,1))
        
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

    def generate_FFI(self, images,rows,columns):
        """Generates full frame image from channel images.

        Args:
            images(list): List of arrays corresponding to channel data

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

    def overscan_cut(self, osub_image,overscan_reg):
        """Cuts overscan region off of overscan-subtracted image.

        Args:
            osub_image(np.array):

        Returns:
            image_cut(np.array): 

        """
        image_cut = osub_image[:,0:overscan_reg[0]]

        return image_cut

    #def prescan_cut(self,osub_image,prescan_reg):
        #image_cut = osub_image[0:prescan_reg[0],:]
        #return image_cut

    def run_oscan_subtraction(self,channel_imgs,channels,channel_keys,channel_rows,channel_cols,channel_exts): #iterate through wrapped frames
        """Performs overscan subtraction steps, in order: orient frame, subtract overscan (method
        chosen by user) from correctly-oriented frame (overscan on right and bottom), cuts off overscan region.
        """

        # clip ends of overscan region 
        oscan_pxl_array,clipped_oscan = self.overscan_arrays()

        # create empty list for final, overscan subtracted/cut arrays
        no_overscan_imgs = []
        #print (channel_imgs.shape)
        for img,key in zip(channel_imgs,channel_keys):
            #print(img.shape)
            # orient img
            new_img = self.orientation_adjust(img,key)
            # overscan subtraction for chosen method
            if self.mode=='mean':
                raw_sub_os = self.mean_subtraction(new_img,clipped_oscan)

            elif self.mode=='polynomial': # subtract linear fit of overscan
                raw_sub_os = self.polyfit_subtraction(new_img,clipped_oscan)

            else:
                raise TypeError('Input overscan subtraction mode set to value outside options.')

            # chop off overscan and prescan - put into overscan subtraction utility
            new_img = self.overscan_cut(raw_sub_os,oscan_pxl_array)

            # put img back into original orientation 
            og_oriented_img = self.orientation_adjust(new_img,key)

            no_overscan_imgs.append(og_oriented_img)

        full_frame_img = self.generate_FFI(no_overscan_imgs,channel_rows,channel_cols)

        return full_frame_img

    def _perform(self):

        channels,channel_keys,channel_rows,channel_cols,channel_exts=self.ref_output
        l0_obj = KPF0.from_fits(self.rawfile,self.data_type)
        frames_data = []
        for ext in channel_exts:
            data = l0_obj[ext].data
            frames_data.append(data)
        frames_data = np.array(frames_data)

        #full_frame_images=[]
        for frame in range(len(self.ffi_exts)):
            single_frame_data = np.array_split(frames_data,len(self.ffi_exts))[frame]
            full_frame_img = self.run_oscan_subtraction(single_frame_data,channels,channel_keys,channel_rows,channel_cols,channel_exts)
            #full_frame_images.append(full_frame_img)
            l0_obj[self.ffi_exts[frame]].data = full_frame_img

        # l0_obj.receipt_add_entry('overscan subtraction', self.__module__, f'input_files={self.rawfile}', 'PASS')

        # if self.logger:
        #     self.logger.info("overscan subtraction: Receipt written")

        return Arguments(l0_obj)









    


