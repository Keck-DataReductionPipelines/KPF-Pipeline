KPF Data Frequently Asked Questions (FAQ)
=========================================

How do I download my data?
--------------------------
The primary portal for KPF data access is the Keck Observatory Archive, which hosts Level 0, Level 1, and Level 2 data products.  Members of the California Planet Search (CPS) can also access KPF data products on the Jump portal or on the computer called shrek.


How do I cite the KPF DRP in scientific publications?
-----------------------------------------------------
See the `Readme file <https://github.com/Keck-DataReductionPipelines/KPF-Pipeline/blob/docs-pipeline-info/README.md>`_ for the DRP.


Is there a quick-start guide to plot and interpret KPF spectra and RVs?
-----------------------------------------------------------------------
See the section titled :ref:`label-tutorials` for tutorials on the various KPF data files.


Which RVs from the header should I use?
---------------------------------------
<add answer here>


What is the difference between the SCI1, SCI2, and SCI3 spectra?
----------------------------------------------------------------
KPF has a single object fiber called 'SCI', a fiber offset by 10 arcsec called 'SKY', and another fiber offset by 10 arcsec in the opposite direction called 'EM-SKY'.  The latter two fibers record the sky spectrum for the main spectrometer and the Exposure Meter, respectively.  The SCI fiber has a diameter of 1.14 arcsec and captures light from the target (usually a star).  Light injected into SCI is optically sliced inside KPF by a device called the Reformatter into three different 'slices' called SCI1, SCI2, and SCI3.  Images of the SCI2 slice have a nealry rectangular shape (as seen in the L0 and 2D spectra of laser frequency comb spectra).  The shapes of the SCI1 and SCI3 slices are the outer sections of an octagon.  SCI1, SCI2, and SCI3 have the same spectrum, but with slightly different line shapes and wavelength solutions (the maximum difference between SCI1 and SCI3 is ~0.2 pixel, which is about 200 m/s in Doppler units).  SCI1, SCI2, and SCI3 should be separately analyzed for precision measurements like radial velocity determination.  For stellar characterization, they can be added to form a composite spectrum.

Are KPF wavelengths in vacuum or air?
-------------------------------------
The KPF DRP produces L1 and L2 data with vacuum wavelengths.

