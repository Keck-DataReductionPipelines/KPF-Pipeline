.. |br| raw:: html

Format of Wavelength Solution Dictionaries
==========================================

This page explains the organization of wavelength solution (WLS) dictionaries.  
These files are only used for inspection and analysis of the wavelength solutions, 
not by the pipeline itself as part of recipes.
  
Organization of WLS Dictionaries
--------------------------------
WLS dictionaries are (currently) per CCD, so there will be two WLS dictionaries per WLS.  (We might update this.)
They are organized hierarchically with by orderlet, order, line, as shown the example below:

``red_thar_WLSDict`` = { |br|
  ``wls_processing_date``: datetime of running WLS code |br|
  ``cal_type``: 'ThAr' or 'LFC' (etalon not included yet) |br|

  orderlets: {
    'CAL':
      full_name: 'RED_CAL',
      orderlet: 'CAL',
      chip: 'RED',
      norders: 32,
      orders: {
        ordernum: 0,
        flux: [ -1.59750346  ... 7.02040416],
        initial_wls: [6057.6004352 ... 5985.06984392],
        echelle_order: 102,
        n_pixels: 4080,
        fitted_wls:	[6057.60083972 ... 5985.05723645]
        rel_precision_cms: 170.0354479170775
        abs_precision_cms: 15206.271268770482
        num_detected_peaks: 35
        known_wavelengths_vac: [6055.056876 6052.6571 ... 5987.924637]
        line_positions: [ 178.41693342 ... 3944.10304143]
        lines: {
          0: {
            amp: 824.2076897704594
            mu: 178.4169334249592
            sig: 1.6482448783761718
            const: -1.4816422789405224
            covar:[[ 4.03987293e+03  ...  2.26986027e+01]]
            data: [ 11.97709366 ... 5.9710143 ]
            model: [ -1.48161703 ... -1.48162795]
            quality: 'good'
            chi2: 0.9533437579975779
            rms: 14.197982419898949
            mu_diff:0.41693342495921115
            lambda_fit: 6055.056876


  
