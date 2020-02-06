Logging
=======

In this package, each non-trivial python instance has a logger. This means
that if you are running a single module as the entire pipeline, there
will be three loggers: one for the framework, one for the pipeline, and
one for the module you are running. Each modules also gets its own
logger instance, and while all logger instances are seperate, they are
all initialized in the same way, which is specified in ``kpfpipe/logger.py``.

Message Format
++++++++++++++
The log messages typically follows this format::

    [%(process_name)][%(log_level)]: %(message)

The ``%(process_name)`` refer to the name of the python instance that logged
this message. The ``%(log_level)`` refers to the level of 
message being logged. This is the
same level as the python logging package. see `logging level
<https://docs.python.org/3/library/logging.html#logging-levels>`_ for
more details. 

Logging Configurations
++++++++++++++++++++++
For the pipeline and module configuration files, there should be a section
called ``LOGGER`` that contains all logging related configurations. A
typical ``LOGGER`` section looks like:

    [LOGGER]
    start_log = True
    log_path = logs/pipe_log.log
    log_level = info
    log_verbose = True

``start_log`` specifies whether you actually want to start the logger for your
pipeline/specific module. If this is set to ``False``, then this python instance
will not log any messages to file or to console. 
This value is by default ``True``, and I recommend that it
always be set to ``True`` unless there is a good reason not to start the logger.

``log_path`` specifies the default log file path. If a ``.log`` file of the same
name already exists, then it will be overwritten. If this parameter is missing, or
a ``None`` is provided, then the logger will not log to a file

``log_level`` specifies the level of messages you would like to log. This is the 
same level as the ``logging`` class in python. It is ``info`` by default

``log_verbose`` specifies whether to print the messages to terminal. By default
this is set to ``True``

.. note:: any invalid values in any of the logging configuration will have the
    same effect as setting ``start_log = False``.

Example
+++++++
If you followed the directions in :doc:`example_trivial`, then
your terminal should now look like this::

    [KPF-Pipe][INFO]:Logger started
    2020-02-02 17:02:25:KPF-Pipe:INFO: Logger started
    [KPF-Pipe][INFO]:Finished initializing Pipeline
    2020-02-02 17:02:25:KPF-Pipe:INFO: Finished initializing Pipeline
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

What you are seeing are actually log messages from three different Logger
instances: the one from the framework ``DRPFrame``,
pipeline ``KPF-Pipe``, and module ``KPFModExample``.

.. note:: you may notice that each log messages are repeated twice, first \
    in the described format, then in a different format that contains date 
    and time. This bug is due to how a logger instance is setup in the 
    framework source code, and is something that I am still trying to fix


