Recipe
======

Data reduction is accomplished through a series of processing steps.
Exactly which steps, in what order, and under what circumstances is determined by a text file called a *recipe*.

Basics
------

In the KPF, recipe files have a syntax that is based on the python programming language syntax, but a recipe is not a python file, in the sense that the python interpreter is not what parses and runs it, but rather the KPF Pipeline and KDRP Framework.

Here is an example of a simple recipe:

::

    # simple recipe example
    from core.primitives import read_config, from_fits, to_fits
    from level0.primitives import flatfield remove_cosmic_rays, corrects_bad_pixels

    config = read_config(config_filename, command_line_args)

    lev0 = from_fits(config.filename1)
    lev0 = flatfield(lev0,config)
    lev0 = remove_cosmic_rays(lev0,config)
    lev0 = correct_bad_pixels(lev0,config)
    to_fits(config.filename2,lev0)

The actual work of data reduction is done by *primitives*, in this example *flatfield*, *remove_cosmic_rays* and *correct_bad_pixels*. In addition, it uses infrastructure primitives *read_config*, *from_fits* and *to_fits* to read the configuration and to read and write the actual data in the form of FITS files. The writer of a recipe need not be intimately familiar with the internals of each primitive used, but does need to know what arguments the primitive expects, typically a *data set*, but sometimes also additional information, and what it delivers in return, also typically a data set, but sometimes something else or in addition.

Any primitive that is used in a recipe must first be *imported*. Importing a primitive adds it to table internal to the pipeline, and ensures that recipe processing can continue after that primitive runs.

*Variables* can be created just by assigning a value to a name, such as *config* and *lev0* in the example above. A variable must be created before it can be used; otherwise it is an error.

    *Best Practice:* Reuse of variables is encouraged, especially when the size of the data the variable is holding is large. In the example above, *lev0* is a Level 0 data set, which can be quite large. If each successive processing step were assigned to different variables, e.g. *lev0a*, *lev0b*, etc., each variable would contain a copy of the data set, using a large and unnecessary amount of memory. However, recommend against using one variable hold a variety of different types of data, although, as in python, such a practice does not constitute an error. It simply makes the recipe hard to follow.
    
    The memory is reclaimed after the recipe processing ends. In a long recipe, memory can be reclaimed earlier by assigning *None* to the variable. In the example below, after processing of the level 0 data set has been completed and written to a FITS file, and a level 1 (spectrum) data set has been generated, the level 0 data set held in *lev0* is no longer needed, so the memory holding it is freed by assigning *None* to *lev0*. Additional data reduction processing could then continue using the data set in *lev1*.

::

    # slightly more complex recipe example
    from core.primitives import read_config, from_fits, to_fits
    from level0.primitives import flatfield, remove_cosmic_rays, correct_bad_pixels
    from level1.primitives import reduce_spectrum

    config = read_config(config_filename,command_line_args)

    lev0 = from_fits(config.filename1)
    lev0 = flatfield(lev0,config)
    lev0 = remove_cosmic_rays(lev0,config)
    lev0 = correct_bad_pixels(lev0,config)
    to_fits(filename2,lev0)
    lev1 = reduce_spectrum(lev0,config)
    # reclaim memory, so we don't waste space with two copies
    lev0 = None
    to_fits(filename3,lev1)

Conditional Execution of Primitives
-----------------------------------

During data reduction in the KPF pipeline, there may be situations where processing should be done only under certain conditions, or using different algorithms or parameters. The KPF pipeline supports the use of conditional expressions in the recipe to handle these kinds of circumstances. Consider the following example:

::

    # example of conditional processing in a recipe
    from core.primitives import read_config, from_fits, to_fits
    from level0.primitives import flatfield_low_SNR, flatfield_high_SNR
    from level0.utilities import eval_SNR

    config = read_config(config_file,command_line_args)

    lev0 = from_fits(config.lev0_from_fits.filename)
    lev0, SNR = eval_SNR(lev0)
    if SNR < config.SNR_thresh:
        lev0 = flatfield_low_SNR(lev0,config)
    else:
        lev0 = flatfield_high_SNR(lev0,config)
    to_fits(filename2,lev0)

In this example, if the signal-to-noise ratio (SNR) is above a threshold, a different flatfield algorithm is used that is more tolerant of noise. The first requirement for conditional processing is to gain access to the value that needs to be tested.  In this case, the primitive *eval_SNR* is used for that purpose.

    *Design Note*: We chose not to include awareness of the details of the data set models in the pipeline recipe software, so that changes to the data set do not require changes to the recipe code. Rather, small primitives need to be written to provide the recipe access to fields of the data set. Only those access primitives would need to be updated should the data set model change in an incompatible way.



Limitations
-----------


Implementation Details
----------------------

