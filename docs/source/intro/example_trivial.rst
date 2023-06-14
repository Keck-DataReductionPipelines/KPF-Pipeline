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

For more detail regarding the structure of KPF Pipeline configuration file, please refer to :doc:`config`. 

To run this pair of files, execute the following command::

    kpf -r examples/simple.recipe -c examples/docs_simple.cfg 

If everything runs smoothly, you should see the following
printed to terminal::

    [KPF-Pipe][INFO]:Pipeline logger started
    [KPF-Pipe][INFO]:Finished initializing Pipeline
    [KPF-Pipe][INFO]:Pipeline logger started
    [KPF-Pipe][INFO]:Pipeline logger started
    [KPF-Pipe][INFO]:Finished initializing Pipeline
    [KPF-Pipe][INFO]:Finished initializing Pipeline
    [KPF-Pipe][INFO]:Starting new log with path: pipeline_20230613.log
    [KPF-Pipe][INFO]:Starting new log with path: pipeline_20230613.log
    [pipeline_20230613.log][INFO]:*************** Executing recipe examples/simple.recipe ***************
    [pipeline_20230613.log][INFO]:Module: subrecipe_depth = 0
    [pipeline_20230613.log][INFO]:Added KPFModExample from modules.Trivial.KPFM_Trivial to event_table
    [pipeline_20230613.log][INFO]:Queued KPFModExample with args "name": KPFModExample_args; awaiting return.
    [KPFModExample] missing log configuration...not starting a new logger
    [KPFModExample] Performed!
    [pipeline_20230613.log][INFO]:Module: subrecipe_depth = 0
    [pipeline_20230613.log][INFO]:Assign: result <- KPFModExample done, type: str
    [pipeline_20230613.log][INFO]:exiting pipeline...

Continue to :doc:`logging` for explanations on what these messages mean.
