Running an example
==================

Once you have the ``KPF-Pipeline`` setup, you may want to run a simple
example to confirm that everything is properly installed. A rather
trivial example is already provided in ``KPF-Pipeline/examples``.

To begin, go into terminal and make sure that you are currently in the
``KPF-Pipeline`` directory. Run the following command::

    kpf

This should invoke the ``kpfpipe/cli.py``. As a response, it should print
to terminal the usaga message::

    usage: kpf [-h] recipe config_file
    kpf: error: the following arguments are required: recipe, config_file

As shown, the ``kpf`` command takes two mandatory files as input:
``recipe``, the recipe file, and ``config``, the configurationfile. See
:doc:`input` for more detail.

To use the example recipe and configuration files, run the following
command::

    kpf examples/recipe.py example/default.cfg

This will run the modules specified in ``recipe.py`` under the configurations
specified in ``default.cfg``. The output should show::

    [KPF-Pipe][INFO]:Logger started
    2020-01-27 13:16:26:KPF-Pipe:INFO: Logger started
    [KPF-Pipe][INFO]:Finished initializting Pipeline
    2020-01-27 13:16:26:KPF-Pipe:INFO: Finished initializting Pipeline
    [DRPFrame][INFO]:Event to action ('evaluate_recipe', 'evaluating_recipe', None)
    2020-01-27 13:16:26:DRPFrame:INFO: Event to action ('evaluate_recipe', 'evaluating_recipe', None)
    [DRPFrame][INFO]:Framework main loop started
    2020-01-27 13:16:26:DRPFrame:INFO: Framework main loop started
    [DRPFrame][INFO]:Executing action evaluate_recipe
    2020-01-27 13:16:26:DRPFrame:INFO: Executing action evaluate_recipe
    [DRPFrame][INFO]:Action evaluate_recipe done
    2020-01-27 13:16:26:DRPFrame:INFO: Action evaluate_recipe done
    [DRPFrame][INFO]:Event to action ('TFAMakeTemplate', 'TEST', None)
    2020-01-27 13:16:26:DRPFrame:INFO: Event to action ('TFAMakeTemplate', 'TEST', None)
    [DRPFrame][INFO]:Executing action TFAMakeTemplate
    2020-01-27 13:16:26:DRPFrame:INFO: Executing action TFAMakeTemplate
    [TFATemp][INFO]:logger started
    2020-01-27 13:16:26:TFATemp:INFO: logger started
    [TFATemp][INFO]:beginning to create template
    2020-01-27 13:16:26:TFATemp:INFO: beginning to create template
    [TFATemp][INFO]:preliminary file used: HARPS.2007-04-09T09_51_56.458_e2ds_A.fits
    2020-01-27 13:16:26:TFATemp:INFO: preliminary file used: HARPS.2007-04-09T09_51_56.458_e2ds_A.fits
    [TFATemp][INFO]:(1/5) processing HARPS.2007-04-07T09_18_58.055_e2ds_A.fits
    2020-01-27 13:16:26:TFATemp:INFO: (1/5) processing HARPS.2007-04-07T09_18_58.055_e2ds_A.fits
    [TFATemp][INFO]:(2/5) processing HARPS.2007-04-04T09_17_51.376_e2ds_A.fits
    2020-01-27 13:16:27:TFATemp:INFO: (2/5) processing HARPS.2007-04-04T09_17_51.376_e2ds_A.fits
    [TFATemp][INFO]:(3/5) processing HARPS.2007-04-08T09_11_58.465_e2ds_A.fits
    2020-01-27 13:16:28:TFATemp:INFO: (3/5) processing HARPS.2007-04-08T09_11_58.465_e2ds_A.fits
    [TFATemp][INFO]:(4/5) processing HARPS.2007-04-06T09_02_30.189_e2ds_A.fits
    2020-01-27 13:16:28:TFATemp:INFO: (4/5) processing HARPS.2007-04-06T09_02_30.189_e2ds_A.fits
    [TFATemp][INFO]:(5/5) processing HARPS.2007-04-09T09_51_56.458_e2ds_A.fits
    2020-01-27 13:16:29:TFATemp:INFO: (5/5) processing HARPS.2007-04-09T09_51_56.458_e2ds_A.fits
    [TFATemp][INFO]:finised making templated
    2020-01-27 13:16:30:TFATemp:INFO: finised making templated
    [DRPFrame][INFO]:Action TFAMakeTemplate done
    2020-01-27 13:16:30:DRPFrame:INFO: Action TFAMakeTemplate done
    [DRPFrame][INFO]:Event to action ('exit_loop', 'exiting...', None)
    2020-01-27 13:16:30:DRPFrame:INFO: Event to action ('exit_loop', 'exiting...', None)
    [DRPFrame][INFO]:Executing action exit_loop
    2020-01-27 13:16:30:DRPFrame:INFO: Executing action exit_loop
    [KPF-Pipe][INFO]:exiting pipeline...
    2020-01-27 13:16:30:KPF-Pipe:INFO: exiting pipeline...

If you are seeing these messages, then **congraduations!** The package is installed
successfully. What you are seeing here are log messages from three loggers:
one for the framework, one for the pipeline, and one for the module. Refer to
:doc:`logging` for more information.
