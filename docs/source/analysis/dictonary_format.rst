.. |br| raw:: html

   <br />

Format of Wavelength Solution Dictionaries
==========================================

This page explains the organization of wavelength solution (WLS) dictionaries.  
These files are only used for inspection and analysis of the wavelength solutions, 
not by the pipeline itself as part of recipes.
  
Organization of WLS Dictionaries
--------------------------------
WLS dictionaries are (currently) per CCD, so there will be two WLS dictionaries per WLS.  (We might update this.)
They are organized hierarchically with by orderlet, order, line, as shown the example below:

``wls_dict`` = { |br|
  ``wls_processing_date``: datetime of running WLS code |br|
  ``cal_type``: 'ThAr' or 'LFC' (etalon not included yet) |br|
  ``orderlets``: { dictionary of ``orderlet_dict`` dictionaries named 'CAL', 'SCI1', 'SCI2', 'SCI3', 'SKY' }  |br|
  }

``orderlet_dict`` = { |br|
  ``full_name``: 'REDCAL' or 'REDSCI1' or 'REDSCI2' or 'REDSCI3' or 'REDSKY' (or similar GREEN`) |br|
  ``orderlet``: 'CAL' or 'SCI1' or 'SCI2' or 'SCI3' or 'SKY', |br|
  ``chip``: 'RED' or 'GREEN', |br|
  ``norders``: number of orders, |br|
  ``orders``: { dictionary of ``orderlet_dict`` dictionaries named 0 to ``norders-1`` } |br|
  }

``order_dict`` = { |br|
  ``ordernum``: order number, |br|
  ``echelle_order``: echelle order number (from the grating equation; not ``ordernum``), |br|
  ``flux``: Numpy array of flux values (length = ``npixels``), |br|
  ``n_pixels``: number of pixels in order (= 4080), |br|
  ``num_detected_peaks``: number of fitted peaks in order,  |br|
  ``initial_wls``: Numpy array of initial wavelengths (length = ``npixels``), |br|
  ``fitted_wls``: Numpy array of fitted wavelengths (length = ``npixels``), |br|
  ``known_wavelengths_vac``: Numpy array of wavelengths (Ang) from line list (length = ``num_detected_peaks``), |br|
  ``line_positions``: Numpy array of fitted line centers (length = ``num_detected_peaks``),  |br|
  ``rel_precision_cms``: estimated relative precision in cm/s, |br|
  ``abs_precision_cms``: estimated absolute precision in cm/s, |br|
  ``lines``: { dictionary of ``line_dict`` dictionaries named 0 to ``num_detected_peaks-1`` } |br|
  }

``line_dict`` = { |br|
  ``amp``: amplitude parameter value from the fit, |br|
  ``mu``: line center parameter value from the fit, |br|
  ``sig``: line width parameter value from the fit, |br|
  ``const``: contact offset parameter from the fit, |br|
  ``covar``: 2-dimensional covariance matrix from the fit, |br|
  ``data``: data (intensities with implied pixel index starting at 0), |br|
  ``model``: best fit model of ``data``, |br|
  ``quality``: 'good' or 'bad', |br|
  ``chi2``: chi^2 value for fit, |br|
  ``rms``: root mean square value for fit, |br|
  ``mu_diff``: difference in pixels between initial guess and fit, |br|
  ``lambda_fit``: best-fit wavelength (Ang), |br|
  }

Accessing parts WLS Dictionaries
--------------------------------
Below are examples of accessing the hierarchical structure of a WLS dictionary.

To print the number of type of calibration spectrum (LFC or ThAr), one would type |br|
``> print(wls_dict['cal_type'])``

To print the number of orders in the orderlet CAL, one would type |br|
``> print(wls_dict['orderlets']['CAL']['norders'])``

To print the echelle order number in order 3, of orderlet CAL, one would type |br|
``> print(wls_dict['orderlets']['CAL']['orders'][3]['echelle_order'])``
  
To print the best-fit wavelength of line 2, in order 3, of orderlet CAL, one would type |br|
``> print(wls_dict['orderlets']['CAL']['orders'][3]['lines'][2]['lambda_fit'])``
  
