KPF File Handling 
=================

Introduction
++++++++++++
The **kpfpipe.models** provide containers for the three levels of KPF data.

This section provides a quick overview of the basic input output interfaces
you can use for reading and writing FITS files. Since the data containers'
structures are heavily influenced by the FITS file structure, I would 
recommend getting familiar with FITS first before you start looking at this 
section. 

Reading and Writing FITS
++++++++++++++++++++++++
Opening a FITS file requires the relative path to the file from the 
``KPF-Pipeline`` directory, and the instrument type of the file::

  >>> from kpfpipe.levels.level1 import KPF1

  >>> # Specify a relative path to a NEID level1 FITS file
  >>> fn = 'resource/NEID/TAUCETI_20191217/L1/neidL1_20191217T023129.fits'

  >>> data = KPF1.from_fits(fn, 'NEID')

``data`` is now a ``KPF1`` class instance that contains all extentions in the
original fits file. Alternatively, one could initialize an empty ``KPF1``
instance, and read in the file::

  >>> from kpfpipe.models.level1 import KPF1

  >>> # Specify a relative path to a NEID level1 FITS file
  >>> fn = 'resource/NEID/TAUCETI_20191217/L0/neidTemp_2D20191217T023129.fits'

  >>> # Initialize an empty KPF1 instance
  >>> data = KPF1()

  >>> # Populate the instance with file content
  >>> data.read(fn, 'NEID')

By default, the data containers support **KPF** and **NEID** FITS files as input. 
``KPF1`` also supports **HARPS**. 

To save the data in a container to a FITS file, a file path relative to
the repository is required::

  >>> out_path = 'out.fits'
  >>> data.to_fits(out_path)

Only **KPF** instrument type is supported as output.

Level 0 Container
+++++++++++++++++

Level 0 container (``KPF0``) is designed for containing level 0 data, 
which are raw 2D flux arrays. The class
instance has two members meant to store two data arrays: ``data`` and ``variance``,
which are named after their original FIT HDU::

  >>> from kpfpipe.models.level0 import KPF0

  >>> # Specify a level 0 file path
  >>> fn = 'resource/NEID/TAUCETI_20191217/L0/neidTemp_2D20191217T023129.fits'

  >>> kpf0 = KPF0.from_fits(fn, 'NEID')
  >>> kpf0.data
  array([[ 5.23479557,  8.52808475,  5.23767757, ...,  6.85253429,
        3.42946506, 11.70948315],
      [ 9.24654007, -0.70817083,  4.28142214, ...,  3.61828065,
      -6.44078827, -8.11476994],
      [ 4.79551649, -3.50319409,  8.11039829, ...,  3.66612482,
        5.22005558, -1.43092608],
      ...,
      [ 0.6333552 ,  2.28048396,  3.95055294, ..., -4.8392477 ,
        3.34699869, -4.89255333],
      [ 0.87699944,  5.82612801,  5.8451972 , ..., -1.52732146,
        8.31192493,  5.03137302],
      [ 2.89919949,  4.54632807, 11.16939735, ...,  3.41376376,
        6.64100981, 11.62545776]])

  >>> kpf0.variance
  array([[23.55319595, 26.84648514, 23.55607796, ..., 26.47743416,
      23.05436516, 31.33438301],
      [27.5649395 , 19.02657127, 22.59982109, ..., 23.24318123,
      26.06568909, 27.7396698 ],
      [23.1139164 , 21.82159424, 26.42879868, ..., 23.29102516,
      24.84495544, 21.05582619],
      ...,
      [21.15425491, 22.80138397, 24.47145271, ..., 23.58814812,
      22.09589958, 23.64145279],
      [21.39789963, 26.34702873, 26.3660965 , ..., 20.27622223,
      27.06082535, 23.78027344],
      [23.42009926, 25.06722832, 31.69029617, ..., 22.16266441,
      25.38990974, 30.37435722]])

Both ``data`` and ``variance`` returns 2D ``numpy.ndarray``

The header cards in the original FITS files can be accessed by the ``header``
member. Since both ``data`` and ``variance`` HDU in the original FITS file 
contain cards, one must provide both the HDU name and the keyword to access 
a specific card::

  >>> # accessing 'NAXIS' keyword from 'DATA' HDU
  >>> kpf0.header['DATA']['NAXIS']
  2

.. note::

  All dictionary keywords in ``kpf0.header`` are capitalized

Level 1 Containers
++++++++++++++++++
Level 1 container are designed for containing KPF level 1 data, 
which are extracted 2D spectrums. Data in ``KPF1`` class instances are
divided into three members: ``flux``, ``wave``, and ``variance``, 
and are identified by their fiber source. As usual, each data array 
are contained in seperate HDUs in their original FITS file, and so each
data array will have their own header. These header information are
stored in ``header`` member 

To see the details of a level 1 data, including the identifier of each data
array, use the ``info()`` function::

  >>> from kpfpipe.models.level1 import KPF1

  >>> # Specify a relative path to a NEID level1 FITS file
  >>> fn = 'resource/NEID/TAUCETI_20191217/L1/neidL1_20191217T023129.fits'

  >>> data = KPF1.from_fits(fn, 'NEID')
  >>> data.info()
  |Header_Name          |# Cards              
  ========================================
  |PRIMARY              |                 737 
  |SCI1_FLUX            |                  12 
  |SKY_FLUX             |                  12 
  |CAL_FLUX             |                  12 
  |SCI1_VARIANCE        |                  12 
  |SKY_VARIANCE         |                  12 
  |CAL_VARIANCE         |                  12 
  |SCI1_WAVE            |                1976 
  |SKY_WAVE             |                1974 
  |CAL_WAVE             |                1974 

  |Data_Name            |Data_Dimension       |N_Segment            
  ============================================================
  |SCI1                 |(117, 9216)          |                   0 
  |SKY                  |(117, 9216)          |                   0 
  |CAL                  |(117, 9216)          |                   0 

There are two tables shown: header and data. Header table contains the 
identifiers of headers of all HDUs. Once again, to access a single card,
one must provide the HDU identifier and the card name::

  >>> # accessing 'NAXIS' keyword from 'PRIMARY' HDU
  >>> data.header['PRIMARY']['NAXIS']
  0

The data table shows all fiber names. One can access the wavelength, flux,
and variance of each fiber::

  >>> data.flux['SCI1']
  array([[ 12.65735184,  12.66835666,  -2.66851381, ..., -48.96005187,
          9.08421789, -13.27987995],
       [  2.16660226,  -1.82952628,  -3.17600617, ...,   2.34231246,
        -17.13232634, -18.71305914],
       [-16.22507516, -27.65861525,  -6.243906  , ...,  28.86555065,
         -6.38643012, -13.26284281],
       ...,
       [  8.67699076,  18.14360676,  15.90762685, ...,   8.71897884,
        -10.34135327, -18.07807723],
       [  5.05678074,  14.13767794,  13.90471246, ...,   8.69989658,
        -10.25584989, -18.15284538],
       [ -3.57320811,  10.23168702,  14.60409041, ...,  29.71688532,
         20.99341983,   4.94008738]])

  >>> data.wave['SCI1']
  array([[ 3570.89978118,  3570.90958484,  3570.91938798, ...,
         3640.00486311,  3640.01008949,  3640.01531536],
       [ 3592.03381895,  3592.04367995,  3592.05354044, ...,
         3661.54282994,  3661.54808722,  3661.553344  ],
       [ 3613.41943714,  3613.42935617,  3613.43927468, ...,
         3683.33719157,  3683.34248013,  3683.34776819],
       ...,
       [10841.54602243, 10841.57556288, 10841.60510173, ...,
        11049.6716447 , 11049.68750883, 11049.70337152],
       [11038.67597497, 11038.70605056, 11038.73612453, ...,
        11250.57086981, 11250.58702235, 11250.60317341],
       [11243.10705331, 11243.13768386, 11243.16831276, ...,
        11458.91081357, 11458.92726518, 11458.9437153 ]])

  >>> data.variance['SCI1']
  array([[278.44026742, 272.87932151, 248.73813999, ..., 295.14812921,
        280.72291295, 266.26799239],
       [251.1378077 , 257.42927428, 265.09744947, ..., 265.05607175,
        272.10232742, 270.17478618],
       [248.08973912, 258.92727153, 263.88708513, ..., 282.55548564,
        294.89935355, 269.18462131],
       ...,
       [282.62912444, 284.13973389, 277.17934537, ..., 261.50707301,
        253.59457378, 264.98627017],
       [274.7178235 , 294.16708465, 289.33425246, ..., 261.49450323,
        253.25218469, 265.11803493],
       [278.88337209, 278.956556  , 282.77325306, ..., 274.11735174,
        274.55312856, 272.11358181]])






