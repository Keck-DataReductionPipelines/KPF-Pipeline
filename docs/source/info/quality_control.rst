Quality Control
===============

The KPF DRP has several Quality Control (QC) methods that can be run on the L0, 2D, L1, and 2D objects.  
The QC tests are run during normal processing in the main recipe.  
The QC methods are defined in the ``QCDefinitions`` class in ``modules/quality_control/src/quality_control.py``.
The results of these QC checks are added to the primary headers of kpf objects, which are written to 2D, L1, and L2 FITS files (but not the L0 files, which, with rare exceptions, are not modified after data collection at WMKO).  
The FITS header keywords QC tests produce are defined in :doc:`data_format`.  
One can find the QC tests on the command line using a command like 

``> fitsheader -e 0 /data/kpf/L2/20230701/KP.20230701.49940.99_L2.fits | grep QC:``

The Jupyter Notebook linked in the bullet points below demonstrates some features of the Quality Control module.

.. toctree::
   :maxdepth: 0

.. nbsphinx::

   ../tutorials/KPF_Data_Tutorial_Quality_Control.ipynb
