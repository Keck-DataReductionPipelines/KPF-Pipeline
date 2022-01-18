import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import stats
import pandas as pd
from kpfpipe.models.level0 import KPF0
from kpfpipe.primitives.level0 import KPF0_Primitive
from keckdrpframework.models.arguments import Arguments
"""
Averaging several frames together reduces the impact of readout
noise and gives a more accurate estimate of the bias level. 
The master bias frame produced from this averaging - Dealing with CCD Data

"""
class FrameCombinePrimitive(KPF0_Primitive):
    def __init__(self, action, context):
        KPF0_Primitive.__init__(self, action, context)
        self.frame_type=self.action.args[0]
        self.L0_names=self.action.args[1]
        self.ffi_ext=self.action.args[2]
        self.data_type=self.action.args[3]
        args_keys = [item for item in action.args.iter_kw() if item != "name"]
        self.high_level_master = action.args['high_level_master'] if 'high_level_master' in args_keys else None
        self.quicklook = action.args['quicklook'] if 'quicklook' in args_keys else False

    def _perform(self):
        if self.logger:
            self.logger.info
            print("Frame Combine: frame type is",self.frame_type)
        if self.frame_type == 'bias':
            if self.quicklook == False:
                tester = KPF0.from_fits(self.L0_names[0])
                ext_list = []
                for i in tester.extensions.keys():
                    if i != 'GREEN_CCD' and i != 'RED_CCD' and i != 'PRIMARY' and i != 'RECEIPT' and i != 'CONFIG':
                        ext_list.append(i)
                master_holder = tester
                for ffi in self.ffi_ext:
                    frames_data=[]
                    for path in self.L0_names:
                        obj = KPF0.from_fits(path)
                        frames_data.append(obj[ffi])
                    frames_data = np.array(frames_data)
                    medians = np.median(frames_data,axis=0)
                    final_frame = medians
                    ### kpf master file creation ###
                    master_holder[ffi] = final_frame
                for ext in ext_list:#figure out where to open master_holder_path
                    master_holder.del_extension(ext)
                    
            if self.quicklook == True:
                for ffi in self.ffi_ext: 
                    color,ccd = ffi.split('_')
                    tester = KPF0.from_fits(self.L0_names[0])
                    masterbias = KPF0.from_fits(self.high_level_master) #where to kpf open fits?
                    counts_master = np.array(masterbias[ffi],'d')
                    counts = np.array(tester[ffi],'d')
                    
                    flatten_counts = np.ravel(counts)
                    flatten_counts_master = np.ravel(counts_master)
                    low,high = np.percentile(flatten_counts,[0.1,99.9])
                    low_master,high_master = np.percentile(flatten_counts_master,[0.1,99.9])
                    counts[(counts>high) | (counts<low)] = np.nan
                    counts_master[(counts_master>high) | (counts_master<low)] = np.nan
                    flatten_counts = np.ravel(counts)
                    flatten_counts_master = np.ravel(counts_master)
                    #print(np.nanmedian(flatten_counts),np.nanmean(flatten_counts)),
                    # np.nanmin(flatten_counts),np.nanmax(flatten_counts))
                    plt.figure(figsize=(5,4))
                    plt.imshow(counts)
                    plt.xlabel('x (pixel number)')
                    plt.ylabel('y (pixel number)')
                    plt.title('{} Bias'.format(color))
                    plt.colorbar(label = 'Counts')
                    plt.savefig('2D_bias_frame_{}.png'.format(color.lower()))
                    
                    plt.close()
                    plt.figure(figsize=(5,4))
                    plt.hist(flatten_counts, bins = 20,alpha =0.5, label = 'Bias Median: ' + '%4.1f' % np.nanmedian(flatten_counts),density = True)#
                    plt.hist(flatten_counts_master*np.random.normal(1,0.1,len(flatten_counts_master)), bins = 20,alpha =0.5, label = 'Master Bias Median: '+ '%4.1f' % np.nanmedian(flatten_counts_master), histtype='step',density = True, color = 'orange', linewidth = 1 )
                    #plt.text(0.1,0.2,np.nanmedian(flatten_counts))
                    plt.xlabel('Counts')
                    plt.title('{} Bias Histogram'.format(color))
                    plt.legend()
                    plt.savefig('Bias_histo_{}.png'.format(color.lower()))
        #####     
                
        if self.frame_type == 'flat':
            if self.quicklook == False:
                tester = KPF0.from_fits(self.L0_names[0])
                ext_list = []
                for i in tester.extensions.keys():
                    if i != 'GREEN_CCD' and i != 'RED_CCD' and i != 'PRIMARY' and i != 'RECEIPT' and i != 'CONFIG':
                        ext_list.append(i)
                master_holder = tester
                for ffi in self.ffi_ext:
                    frames_data=[]
                    for path in self.L0_names:
                        obj = KPF0.from_fits(path) #check this
                        frames_data.append(obj[ffi])
                    obj = KPF0.from_fits(self.L0_names[0]) #check this
                    master_holder[ffi] = obj[ffi]
                for ext in ext_list:
                    master_holder.del_extension(ext)
                for ffi in self.ffi_ext:
                    rows = np.shape(master_holder[ffi])[0]
                    cols = np.shape(master_holder[ffi])[1]
                    norm_flat= np.ones((rows,cols))
                    norm_flat = pd.DataFrame(norm_flat)
                    master_holder.create_extension((ffi+'_NORMALIZED'),ext_type=np.array)
                    master_holder[ffi+'_NORMALIZED'] = norm_flat
                    # print(np.shape(master_holder[ffi+'_NORMALIZED']))
                    
            if self.quicklook == True:
                for ffi in self.ffi_ext: 
                    color,ccd = ffi.split('_')
                    
                    tester = KPF0.from_fits(self.L0_names[0])
                    masterflat = KPF0.from_fits(self.high_level_master) #where to kpf open fits?
                    counts_master = np.array(masterflat[ffi],'d')
                    counts = np.array(tester[ffi],'d')
                    flatten_counts = np.ravel(counts)
                    flatten_counts_master = np.ravel(counts_master)
                    low, high = np.percentile(flatten_counts,[0.1,99.9])
                    low_master, high_master = np.percentile(flatten_counts_master,[0.1,99.9])

                    plt.figure(figsize=(5,4))
                    plt.imshow(np.log10(counts),vmin = 0)
                    plt.colorbar(label = 'log(Counts)')
                    plt.xlabel('x (pixel number)')
                    plt.ylabel('y (pixel number)')
                    plt.title('{} Flat'.format(color))
                    #plt.show()
                    plt.savefig('2D_Flat_frame_{}.png'.format(color.lower()))

                    plt.close()
                    plt.figure(figsize=(5,4))
                    plt.hist(np.log10(flatten_counts), bins = 20, alpha = 0.5, 
                            label = 'Flat', range = [-2,6],density = True)
                    plt.hist(np.log10(flatten_counts_master), bins = 20, alpha = 0.5, label = 'Master Flat', 
                            range = [-2,6],density = True, histtype='step', color = 'orange', linewidth = 1)
                    plt.xlabel('log(Counts)')
                    plt.title('{} Flat Histogram'.format(color))
                    plt.legend()
                    plt.savefig('Flat_histo_{}.png'.format(color.lower()))   
                    
            # for ffi in self.ffi_ext:
            #     frames_data=[]
            #     for path in self.L0_names:
            #         obj = KPF0.from_fits(path) #check this
            #         frames_data.append(obj[ffi])
                # frames_data = np.array(frames_data)
                # medians = np.median(frames_data,axis=0) #taking pixel median
                # mmax,mmin = medians.max(),medians.min()
                # norm_meds = (medians - mmin)/(mmax - mmin) #normalizing
                # final_frame = norm_meds
                # ### kpf master file creation ###
                # master_holder[ffi] = final_frame
            # for ext in ext_list:
            #     master_holder.del_extension(ext)

            # plt.figure()
            # plt.imshow(final_frame)
            # plt.colorbar()
            # plt.savefig('final_frame_{}.pdf'.format(self.ffi_ext))
            # plt.close()
            # plt.figure()
            # plt.imshow(master_holder[self.ffi_ext])
            # plt.colorbar()
            # plt.savefig('master_holder_{}.pdf'.format(self.ffi_ext))
            # plt.close()
        return Arguments(master_holder)