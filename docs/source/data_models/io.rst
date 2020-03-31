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
    >>> fn = 'resource/NEID/TAUCETI_20191217/L0/neidTemp_2D20191217T023129.fits'

    >>> data = KPF1.from_fits(fn, 'NEID')

``data`` is now a ``KPF1`` class instance that contains all extentions in the
original fits file. Alternatively, one could initialize an empty ``KPF1``
instance, and read in the file::

    >>> from kpfpipe.levels.level1 import KPF1

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

Level 0 container (``KPF0``) is designed for containing level 0 data. The class
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



