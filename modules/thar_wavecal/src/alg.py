#imports
import numpy as np
import os
from scipy import signal, constants
from astropy.io import fits
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre

from modules.wavelength_cal.src.alg import LFCWaveCalibration

class ThArCalibration:
    """
    """

    def __init__(self, config=None, logger=None):
        configpull=ConfigHandler(config,'PARAM')
        self.search_within = configpull.get_config_value('peak_search_range', 1)
        self.fit_order = configpull.get_config_value('fit_order', 6)
        self.gaussian_fit_width = configpull.get_config_value('gaussian_fit_width', 10)
        self.min_order = configpull.get_config_value('min_order', 3)
        self.max_order = configpull.get_config_value('max_order', 117)
        self.n_zoom_sections = configpull.get_config_value('n_zoom_sections', 20)
        self.end_pixels_to_clip = configpull.get_config_value('clip_end_pxls', 1500)
        self.subplot_size = configpull.get_config_value('subplot_size', (6,20))
        self.saveplots = configpull.get_config_value('saveplots', 'ThAr_plots')

    def find_and_fit_peaks(self,flux,linelist,line_pixels_expected,plot_toggle,savefig):
        """[summary]

        Args:
            flux ([type]): [description]
            linelist ([type]): [description]
            line_pixels_expected ([type]): [description]
            savefig ([type]): [description]
        """
        peaks,_ = signal.find_peaks(flux)
        print('{} peaks detected ({} lines in input list).'.format(len(peaks),len(linelist)))

        num_input_lines = len(linelist)  
        num_pixels = len(flux)

        missed_lines = 0
        coefs = np.zeros((4,num_input_lines))
        for i in np.arange(num_input_lines):
            line_location = line_pixels_expected[i]
            potential_matches = peaks[(
                (peaks > line_location - self.search_within ) &
                (peaks < line_location + self.search_within)
            )]
            num_matches = len(potential_matches)
            if num_matches == 0:
                missed_lines += 1
            elif num_matches > 0:

                if num_matches > 1:
                    raise ValueError('Multiple peaks detected in a 2-pixel radius. Something went wrong with the fitting.')
                else:
                    peak_pixel = potential_matches[0]

                if peak_pixel < gaussian_fit_width:
                    first_fit_pixel = 0
                else:
                    first_fit_pixel = peak_pixel - gaussian_fit_width
                
                if peak_pixel > num_pixels:
                    last_fit_pixel = num_pixels
                else:
                    last_fit_pixel = peak_pixel + gaussian_fit_width

                #fit gaussian to matched peak location
                coefs[:,i] = LFC_wavelength_cal.fit_gaussian(
                    np.arange(first_fit_pixel,last_fit_pixel),
                    flux[first_fit_pixel:last_fit_pixel])

                amp = coefs[0,i]
                if amp < 0:
                    missed_lines += 1
        
        if plot_toggle = True:
            zoom_section_pixels = (num_pixels // self.n_zoom_sections)
            _, ax_list = plt.subplots(self.n_zoom_sections,1,figsize=(self.subplot_size))
            ax_list[0].set_title('({} missed lines)'.format(missed_lines))
            for i, ax in enumerate(ax_list):
                ax.plot(
                    np.arange(num_pixels[i*zoom_section_pixels:(i+1)*zoom_section_pixels]
                    flux[i*zoom_section_pixels:(i+1)*zoom_section_pixels],color='k',alpha=.1))

                ax.scatter(
                    coefs[1,:][
                        (coefs[1,:] > i * zoom_section_pixels) & 
                        (coefs[1,:] < (i+1) * zoom_section_pixels)
                    ], 
                    coefs[0,:][
                        (coefs[1,:] > i * zoom_section_pixels) & 
                        (coefs[1,:] < (i+1) * zoom_section_pixels)
                    ] + 
                    coefs[3,:][
                        (coefs[1,:] > i * zoom_section_pixels) & 
                        (coefs[1,:] < (i+1) * zoom_section_pixels)
                    ],
                    color='red'
                )
                ax.set_yscale('log')

            plt.tight_layout()
            plt.savefig('{}/spectrum_and_peaks.png'.format(savefig), dpi=250)
            plt.close()

        return coefs

    def fit_polynomial(self,peak_pixels,vacuum_wavelens,n_pixels):
        """[summary]

        Args:
            peak_pixels ([type]): [description]
            vacuum_wavelens ([type]): [description]
            n_pixels ([type]): [description]
        """
        leg_out = Legendre.fit(
            peak_pixels[peak_pixels > 0], vacuum_wavelens[peak_pixels >0], fit_order
        )
        wl_soln_for_order = leg_out(np.arange(n_pixels))
        
        return wl_soln_for_order


    def calculate_precision(self,our_wls,other_wls,num_lines_fit,plot_toggle,savefig):
        """[summary]

        Args:
            our_wls ([type]): [description]
            other_wls ([type]): [description]
            num_lines_fit ([type]): [description]
            savefig ([type]): [description]
        """
        residuals = ((our_wls - other_wls) * constants.c) / other_wls
        precision = np.std(residuals)/np.sqrt(num_lines_fit)

        if plot_toggle = True:
            plt.figure()
            plt.plot(residuals)
            plt.save_fig('{}/rv_precision.png'.format(savefig), dpi=250)
            plt.close()

        return precision, residuals

    def run_on_all_orders(self,flux,redman_w,redman_i,linelist_sub,other_wls,plot_toggle):
        """[summary]

        Args:
            flux ([type]): [description]
            redman_w ([type]): [description]
            redman_i ([type]): [description]
            linelist_sub ([type]): [description]
            other_wls ([type]): [description]
            plot_toggle ([type]): [description]
        """
        if plot_toggle = True:
            _, summary_ax = plt.subplots()
            summary_ax.set_xlabel('pixel')
            summary_ax.set_ylabel('Our WLS - Their WLS [m/s]')
            summary_ax.set_ylim(-500, 500)

        num_pixels = len(flux[0])

        for order_num in np.arange(first_order,last_order+1):
            print('\nRunning Order {}!'.format(order_num))

            if plot_toggle = True:
                order_plt_path = '{}/order{}'.format(self.saveplots,order_num)
                if not os.path.exists(order_plt_path):
                    os.mkdir(order_plt_path)

            min_order_wl = np.min(linelist_sub[order_num]['known_wavelengths_vac'])
            max_order_wl = np.max(linelist_sub[order_num]['known_wavelengths_vac'])
            in_order_mask = ((redman_w > min_order_wl) & (redman_w < max_order_wl))

            if plot_toggle = True:
                _, ax = plt.subplots(figsize=(12,6))
                plt.plot(
                    redman_w[in_order_mask], 
                    redman_i[in_order_mask],
                    color='purple'
                )
                ax.vlines(
                    linelist_sub[order_num]['known_wavelengths_vac'], 
                    ymin=0, 
                    ymax=0.5 * np.max(redman_i[in_order_mask]),
                    color='grey', alpha=0.5
                )
                plt.title('Order {} Linelist'.format(order_num))
                plt.xlabel('wavelength [$\\rm \AA$]')
                plt.ylabel('intensity')
                plt.savefig('{}/linelist.png'.format(order_plt_path), dpi=250)
                plt.close()

                # plot the full spectrum
                plt.figure(figsize=(20,10))
                plt.title('ThAr Spectra')
                im = plt.imshow(flux, aspect='auto')
                im.set_clim(0, 20000)
                plt.xlabel('pixel')
                plt.ylabel('order number')
                plt.savefig('{}/extracted_spectra.png'.format(SAVEPLOTS), dpi=250)
                plt.close()

                # plot the spectrum for the order of interest
                plt.figure()
                plt.plot(flux[order_num])  
                plt.title('Order {} Spectrum'.format(order_num))
                plt.xlabel('pixel')
                plt.ylabel('flux')
                plt.tight_layout()
                plt.savefig('{}/extracted_spectra.png'.format(order_plt_path), dpi=250)
                plt.close()
            
            gauss_fit_coefs = self.find_and_fit_peaks(
                flux[order_num],
                linelist_sub[order_num]['known_wavelengths_vac'],
                linelist_sub[order_num]['line_positions'],
                order_plt_path
            )

            wls = self.fit_polynomial(
                gauss_fit_coefs[1,:],
                linelist_sub[order_num]['known_wavelengths_vac'],
                len(flux[order_num])
            )

            num_lines_fit = len(gauss_fit_coefs[1,:][gauss_fit_coefs[1,:] > 0])
            precision,residuals = self.calculate_precision(wls,other_wls[order_num],
                num_lines_fit,order_plt_path)

            print('Order {} Precision: {:.2f} m/s.'.format(order_num, precision))

            if plot_toggle = True:
                summary_ax.plot(
                    np.arange(self.end_pixels_to_clip, num_pixels - self.end_pixels_to clip),
                    residuals[self.end_pixels_to_clip:num_pixels - self.end_pixels_to_clip]
                )
        
        if plot_toggle = True:
            plt.savefig('{}/wls_comp.png'.format(self.saveplots), dpi=250)
            plt.close()

        





    
                
        