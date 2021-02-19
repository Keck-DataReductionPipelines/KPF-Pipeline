Running a Simple Module
=======================

In the ``/examples`` directory, you should find several pairs of recipe
and configuration files. For now, we focus on a recipe file, ``simple.recipe``,
and a configuraton file, ``docs_simple.cfg``. 

The ``simple.recipe`` contains two lines::

    from modules.Trivial.KPFM_Trivial import KPFModExample
    KPFModExample()

This states that the ``KPFModExample``,
located at ``modules/Trivial/KPFM_Trivial.py``, is running as the
only module in this pipeline. 

The configuration file, ``docs_simple.cfg``,  contains::

    # Pipeline logger configurations
    [LOGGER]
    start_log = True
    log_path = logs/pipe_log.log
    log_level = info
    log_verbose = True

    # for recipe
    [ARGUMENT]

    [MODULE_CONFIGS]

Note that sections ``[ARGUMENT]`` and ``[MODULE_CONFIGS]`` are empty, since 
the module we are executing does not need any of these settings.
However, the sections must still exist in the configuration file
for the pipeline to be properly initialized.

To run this pair of files, execute the following command::

    kpf examples/simple.recipe examples/docs_simple.cfg 

If everything runs smoothly, you should see the following
printed to terminal::

    2021-02-03 18:23:17:DRPF:INFO:
    2021-02-03 18:23:17:DRPF:INFO: Initialization Framework cwd=<KPF-Pipeline directory>
    [KPF-Pipe][INFO]:Logger started
    [KPF-Pipe][INFO]:Finished initializing Pipeline
    2021-02-03 18:23:17:DRPF:INFO: Framework main loop started
    [KPF-Pipe][INFO]:Module: subrecipe_depth = 0
    [KPF-Pipe][INFO]:Added KPFModExample from modules.Trivial.KPFM_Trivial to event_table
    [KPF-Pipe][INFO]:Queued KPFModExample with args "name": KPFModExample_args; awaiting return.
    2021-02-03 18:23:17:DRPF:INFO: Event completed: name start_recipe, action start_recipe, arg name undef, recurr False
    [KPF-Pipe][INFO]:exiting pipeline...
    2021-02-03 18:23:17:DRPF:INFO: Event failed: name exit, action exit_loop, arg name undef, recurr False
    [KPFModExample] missing log configuration...not starting a new logger
    [KPFModExample] Performed!
    2021-02-03 18:23:17:DRPF:INFO: Event failed: name KPFModExample, action KPFModExample, arg name KPFModExample_args, recurr False
    [KPF-Pipe][INFO]:Module: subrecipe_depth = 0
    2021-02-03 18:23:17:DRPF:INFO: Event completed: name resume_recipe, action resume_recipe, arg name KPFModExample_args, recurr False
    2021-02-03 18:23:18:DRPF:INFO: No new events - do nothing
    2021-02-03 18:23:18:DRPF:INFO: Exiting main loop

Continue to :doc:`logging` for explanations on what these messages mean.
