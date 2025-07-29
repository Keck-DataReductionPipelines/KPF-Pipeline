KPF Eras
========

Significant changes to the Keck Planet Finder instrument are designated by increments of the KPFERA keyword.  
Each value of KPFERA spans a date range.  
Half integer values (e.g., 1.5) are meant to be periods of engineering work while integer values (e.g., 1.0) are meant to be "science eras" during which the instrument is stable.  
Smaller instrument changes are indicated by 0.1 increments indicate smaller instrument changes (e.g., 2.6 vs. 2.5).

The table below displays the contents of the file ``static/kpfera_definitions.csv`` in this repository, which defines the time period of KPF eras and has notes about the instrument changes betweent them.  

Users should expect offsets between the RV times series of a given source between half-integer and larger increments of KPFERA (e.g., 1.0 -> 1.5).  The offset will not be the same for every star due to differences in spectral content as a function of wavelength.  Thus, it is unlikely that a prescription for offests can be determined at high RV precision.  A future version of this page will quantify differences between RVs in KPF eras.

.. csv-table::
    :header-rows: 1
    :file: ../../../static/kpfera_definitions.csv
    :class: wrap-table
