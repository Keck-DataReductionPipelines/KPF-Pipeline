#imports
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
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
        self.prl_overscan_reg = self.action.args[1] #overscan region of raw image, start and end pixels of overscan
        self.srl_overscan_reg = self.action.args[2]
        self.mode = self.action.args[3] #defines which method of overscan subtraction
        self.order = self.action.args[4] #if self.mode = 'polynomial', defines order of polynomial fit
        self.oscan_clip_no = self.action.args[5] #amount of pixels clipped from each edge of overscan
        self.ref_output = self.action.args[6] #output of ccd reference file
        self.ffi_exts = self.action.args[7] #fits extensions where ffis will be stored
        self.data_type = self.action.args[8] #data type, pertaining to instrument

    def overscan_arrays(self):
        """Makes array of overscan pixels. For example, if raw image including overscan region
        is 1000 pixels wide, and overscan region (when oriented with overscan on right and parallelscan
        on bottom) is 100 pixels wide, it will make an array of values 900-1000.

        Returns:
            srl_overscan_pxs(np.ndarray): Array of serial overscan region pixels
            prl_overscan_pxs(np.ndarray): Array of parallel overscan region pixels
            srl_overscan_clipped(np.ndarray): Array of clipped serial overscan region pixels
            prl_overscan_clipped(np.ndarray): Array of clipped parallel overscan region pixels
        """
        # if self.rawfile.header['NOSCN_S'] and self.rawfile.header['NOSCN_P']:

        # else:
      
        srl_overscan_pxs = np.arange(self.srl_overscan_reg[0],self.srl_overscan_reg[1],1)
        prl_overscan_pxs = np.arange(self.prl_overscan_reg[0],self.prl_overscan_reg[1],1)
        srl_N_overscan = len(srl_overscan_pxs)
        prl_N_overscan = len(prl_overscan_pxs)
        srl_overscan_clipped = srl_overscan_pxs[self.oscan_clip_no:srl_N_overscan-1-self.oscan_clip_no]
        prl_overscan_clipped = prl_overscan_pxs[self.oscan_clip_no:prl_N_overscan-1-self.oscan_clip_no]
        return srl_overscan_pxs,prl_overscan_pxs,srl_overscan_clipped,prl_overscan_clipped

    def mean_subtraction(self,image,overscan_reg): #should work now
        """Gets mean of overscan data, subtracts value from raw science image data.

        Args:
            image(np.ndarray): Array of image data
            overscan_reg(np.ndarray): Array of pixel range of overscan relative to image pixel width

        Returns:
            raw_sub_os(np.ndarray): Raw image with overscan mean subtracted
        """
        raw_sub_os = np.zeros_like(image)
        for row in range(0,raw_sub_os.shape[0]):
            raw_sub_os[row,:] = image[row,:] - np.mean(image[row,overscan_reg],keepdims=True) 
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

    def overscan_cut(self,osub_image,overscan_reg_srl,overscan_reg_prl):
        """Cuts overscan region off of overscan-subtracted image.

        Args:
            osub_image(np.ndarray): Image with overscan region subtracted.
            overscan_reg_srl(np.ndarray): Serial overscan region
            overscan_reg_prl(np.ndarray): Parallel overscan region

        Returns:
            image_cut(np.ndarray): Image with overscan region cut off.

        """
        image_cut = osub_image[:,0:overscan_reg_srl[0]]
        image_cut = image_cut [0:overscan_reg_prl[0],:]

        return image_cut

    def neid_setup_run(self,l0_obj,channel_exts):
        """Runs individual frame overscan subtraction and removal. In progress.

        Args:
            l0_obj (fits.HDUList): Raw file object.
            channel_exts (list): List of extensions with amplifiers. 
            
        Returns:
            array: Whole assembled image (full frame image)
        """
        detsize = self.rawfile.header['PRIMARY']['DETSIZE']
        detsize = detsize.replace('[','')
        detsize = detsize.replace(']','')
        a_detsize,b_detsize = detsize.split(',')
        a_detsize,b_detsize = int(a_detsize),int(b_detsize)
        whole_image = np.zeros((b_detsize,a_detsize))
        for ext in channel_exts:
            bias1 = self.rawfile.header[ext]['BIASSEC1']
            bias2 = self.rawfile.header[ext]['BIASSEC2']
            bias3 = self.rawfile.header[ext]['BIASSEC3']
            datasec = self.rawfile.header[ext]['DATASEC']
            detsec = self.rawfile.header[ext]['DETSEC']
        
            col_start_list = []
            col_end_list = []
            row_start_list = []
            row_end_list = []
            for section in (bias1,bias2,bias3):
                bias = section.replace('[','')
                bias = bias.replace(']','')
                a,b = bias.split(',')
                col_start,col_end = a.split(':')
                row_start,row_end = b.split(':')
                col_start,col_end,row_start,row_end = int(col_start),int(col_end),int(row_start),int(row_end)
                col_start_list.append(col_start)
                col_end_list.append(col_end)
                row_start_list.append(row_start)
                row_end_list.append(row_end)      
                
            #detsec
            detsec = detsec.replace('[','')
            detsec = detsec.replace(']','')
            a_det,b_det = detsec.split(',')
            aa_det,ab_det = a_det.split(':')
            ba_det,bb_det = b_det.split(':')
            aa_det,ab_det,ba_det,bb_det = int(aa_det),int(ab_det),int(ba_det),int(bb_det)    
            
            #datasec
            datasec = datasec.replace('[','')
            datasec = datasec.replace(']','')
            a_data,b_data = datasec.split(',')
            aa_data,ab_data = a_data.split(':')
            ba_data,bb_data = b_data.split(':')
            aa_data,ab_data,ba_data,bb_data = int(aa_data),int(ab_data),int(ba_data),int(bb_data)            
        
            self.rawfile.data[ext] = self.rawfile.data[ext][ba_data:bb_data,aa_data:ab_data]
            # perform poly/mean sub
            #if self.mode=='mean':
            whole_image[ba_det:bb_det,aa_det:ab_det] = self.rawfile.data[ext]
        return whole_image
            #elif self.mode=='polynomial': # subtract linear fit of overscan

        
    def run_oscan_subtraction(self,channel_imgs,channels,channel_keys,channel_rows,channel_cols,channel_exts):
        """Performs overscan subtraction steps, in order: orient frame, subtract overscan (method
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
        for img,key in zip(channel_imgs,channel_keys):
            new_img = self.orientation_adjust(img,key)
            # overscan subtraction for chosen method
            if self.mode=='mean':
                raw_sub_os = self.mean_subtraction(new_img,srl_clipped_oscan)

            elif self.mode=='polynomial': # subtract linear fit of overscan
                raw_sub_os = self.polyfit_subtraction(new_img,srl_clipped_oscan)

            else:
                raise TypeError('Input overscan subtraction mode set to value outside options.')
            # chop off overscan and prescan - put into overscan subtraction utility
            new_img = self.overscan_cut(raw_sub_os,srl_oscan_pxl_array,prl_oscan_pxl_array)
            # put img back into original orientation 
            og_oriented_img = self.orientation_adjust(new_img,key)
            plt.imshow(og_oriented_img)

            no_overscan_imgs.append(og_oriented_img)

        full_frame_img = self.generate_FFI(no_overscan_imgs,channel_rows,channel_cols)
        return full_frame_img

    def _perform(self):
        """Performs entire overscan subtraction utility.

        Returns:
            l0_obj(fits.hdulist): Original FITS.hdulist but with FFI extension(s) filled
        """
        channels,channel_keys,channel_rows,channel_cols,channel_exts=self.ref_output

        if self.data_type == 'KPF':
            l0_obj = self.rawfile
            frames_data = []
            for ext in channel_exts:
                data = l0_obj[ext]
                frames_data.append(data)
            frames_data = np.array(frames_data)
            #full_frame_images=[]
            for frame in range(len(self.ffi_exts)):
                single_frame_data = np.array_split(frames_data,len(self.ffi_exts))[frame]
                full_frame_img = self.run_oscan_subtraction(single_frame_data,channels,channel_keys,channel_rows,channel_cols,channel_exts)        
                #full_frame_images.append(full_frame_img)
                l0_obj[self.ffi_exts[frame]] = full_frame_img
                
        if self.data_type == 'NEID':
            l0_obj = self.rawfile
            whole_image = self.neid_setup_run(l0_obj,channel_exts)
            l0_obj[self.ffi_exts] = whole_image
        
        return Arguments(l0_obj)