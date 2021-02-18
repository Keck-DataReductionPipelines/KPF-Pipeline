Running a Simple Module
=======================

In the ``/example`` directory, you shoudl find two pairs of recipe
and configuration files. For now, we focus on: ``recipe_simple.py``
and ``deffault_simple.cfg``

The ``recipe_simple.py`` contains a single line::

    context.push_event('KPFModExample', action.args)

This states that the ``KPFModExample``,
located at ``modules/Trivial/KPFM_Trivial.py``, is running as the
only module in this pipeline. 

The ``default_simple.cfg`` configuration contains::

    # Pipeline logger configurations
    [LOGGER]
    start_log = True
    log_path = logs/pipe_log.log
    log_level = info
    log_verbose = True

    # Framework related configurations
    [FRAMEWORK]
    config_path = configs/framework.cfg
    log_config = configs/framework_logger.cfg

    # Pipeline parameters
    [PIPELINE]
    pipeline_name = KPF-Pipe
    mod_search_path = modules

    # file
    [ARGUMENT]

    [MODULE_CONFIGS]

Note that the sections ``[ARGUMENT]`` and ``[MODULE_CONFIGS]`` are empty, since 
the module we are executing does not require any of these settings.
However, the sections must still be present in the configuration file
for the pipeline to be properly initialized.

To run this pairs of files, execute the following command::

    kpf examples/recipe_simple.py examples/default_simple.cfg 

If everything runs smoothly, you should see the following
printed to terminal::

    [KPF-Pipe][INFO]:Logger started
    2020-02-02 17:02:25:KPF-Pipe:INFO: Logger started
    [KPF-Pipe][INFO]:Finished initializting Pipeline
    2020-02-02 17:02:25:KPF-Pipe:INFO: Finished initializting Pipeline
    [DRPFrame][INFO]:Event to action ('evaluate_recipe', 'evaluating_recipe', None)
    2020-02-02 17:02:25:DRPFrame:INFO: Event to action ('evaluate_recipe', 'evaluating_recipe', None)
    [DRPFrame][INFO]:Framework main loop started
    2020-02-02 17:02:25:DRPFrame:INFO: Framework main loop started
    [DRPFrame][INFO]:Executing action evaluate_recipe
    2020-02-02 17:02:25:DRPFrame:INFO: Executing action evaluate_recipe
    [DRPFrame][INFO]:Action evaluate_recipe done
    2020-02-02 17:02:25:DRPFrame:INFO: Action evaluate_recipe done
    [DRPFrame][INFO]:Event to action ('KPFModExample', 'EXAMPLE', None)
    2020-02-02 17:02:25:DRPFrame:INFO: Event to action ('KPFModExample', 'EXAMPLE', None)
    [DRPFrame][INFO]:Executing action KPFModExample
    2020-02-02 17:02:25:DRPFrame:INFO: Executing action KPFModExample
    [KPFModExample] missing log configuration.. not starting logger
    [KPFModExample] Performed!
    [DRPFrame][INFO]:Action KPFModExample done
    2020-02-02 17:02:26:DRPFrame:INFO: Action KPFModExample done
    [DRPFrame][INFO]:Event to action ('exit_loop', 'exiting...', None)
    2020-02-02 17:02:26:DRPFrame:INFO: Event to action ('exit_loop', 'exiting...', None)
    [DRPFrame][INFO]:Executing action exit_loop
    2020-02-02 17:02:26:DRPFrame:INFO: Executing action exit_loop
    [KPF-Pipe][INFO]:exiting pipeline...
    2020-02-02 17:02:26:KPF-Pipe:INFO: exiting pipeline...

Continue to :doc:`logging` for explainations on what these messages mean
