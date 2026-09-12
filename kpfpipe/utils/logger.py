"""Pipeline logging setup: one UT-timestamped log file per invocation.

Implements the WMKO logging requirements (DRP-RUN-07/08/09/12): a
user-configurable log directory, all logs under one parent directory, and
per-invocation log files that concurrent pipeline instances never share.

Handlers are installed on the root logger so that module loggers
(``logging.getLogger(__name__)``), the ``py.warnings`` bridge, and
third-party libraries all reach the same file. Two sibling entry points
install them, never at import time:

- ``setup_logging`` -- the single-recipe leaf runner
  (scripts/processing/reduce.py, the ``kpfpipe run`` entry) calls it once per
  reduction, writing that recipe's per-unit log; its console echo defaults to
  stderr.
- ``setup_batch_logging`` -- the fan-out orchestrators (masters.py/science.py)
  call it once per invocation, writing a batch-summary log of the dispatch's own
  decision points; its console echo is pinned to stdout so an operator can watch
  batch progress live. A thin wrapper over ``setup_logging`` that also returns
  this run's id, which the orchestrator forwards to bind its children to it.

Library code only ever calls ``logging.getLogger(__name__)``; with no handlers
installed (e.g. recipes driven directly by tests) records are simply dropped.
"""

import logging
import os
import sys
import time

# One line per record: UT timestamp, level, logger name, message.
LOG_FORMAT = "%(asctime)s.%(msecs)03dZ %(levelname)-8s %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"

# Chatty third-party loggers pinned to WARNING so a DEBUG run stays readable.
_THIRD_PARTY_PINS = ("matplotlib", "PIL")

# Collision-suffix retries before giving up (fail loudly, never spin forever).
_MAX_COLLISION_RETRIES = 1000

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Handlers installed by setup_logging, so repeated setup/teardown never
# duplicates or leaks them (module-private state, reset by teardown_logging).
_installed_handlers = []

# Root-logger level before the first setup_logging, restored by teardown.
_prior_root_level = None


class _BatchConsoleFilter(logging.Filter):
    """Trims the batch orchestrator's *terminal echo* -- not its log file.

    A batch run's live stdout is meant to show the orchestrator's own decision
    trail, but INFO records propagated up from library loggers in the driver
    process (discovery, FITS/astropy chatter) make it noisy. So below WARNING,
    only records from the orchestrator's own code reach the console -- the
    ``scripts.*`` namespace (masters/science/_dispatch/plots) or ``__main__`` (a
    driver launched as ``python -m scripts.processing.<name>``, whose module
    logger is named ``__main__``). WARNING and above always pass, so per-unit
    failures and warnings are never hidden. Attached to the console handler only,
    so the batch *log file* keeps every record.
    """

    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return True
        return record.name == "__main__" or record.name.startswith("scripts.")


def get_level(name):
    """Map a level name ('debug' ... 'critical', any case) to its ``logging``
    int; raises ``ValueError`` on an unknown name."""
    try:
        return _LEVELS[str(name).upper()]
    except KeyError:
        raise ValueError(
            f"unknown log level {name!r}; expected one of {sorted(_LEVELS)}"
        ) from None


def _ut_stamp(start_time=None):
    """UT ``YYYYMMDDTHHMMSS`` -- the stamp in every run directory and log filename."""
    return time.strftime("%Y%m%dT%H%M%S", start_time or time.gmtime())


def log_filename(log_dir, recipe_name, target, start_time=None):
    """Build the unique per-invocation log path (does not create the file).

    The layout is ``{log_dir}/kpf_{recipe_name}_{target}_{YYYYMMDDTHHMMSS}.log``,
    the stamp in UT.

    Parameters
    ----------
    log_dir : str
        This run's own directory, ``{configured log_dir}/{run_id}`` as
        ``setup_logging`` joins it: one run's logs land together, under the one
        configured parent (DRP-RUN-09).
    recipe_name : str
        Short recipe identifier, e.g. 'science' or 'masters'.
    target : str
        The reduction target: obs_id, datecode, or 'run' when neither applies.
    start_time : time.struct_time or None
        UT start time of the invocation; None means ``time.gmtime()`` now.

    Returns
    -------
    str
        The absolute log-file path.

    Raises
    ------
    ValueError
        If ``log_dir`` is empty or not a string.
    """
    if not log_dir or not isinstance(log_dir, str):
        raise ValueError(f"log_dir must be a non-empty string; got {log_dir!r}")
    fn = f"kpf_{recipe_name}_{target}_{_ut_stamp(start_time)}.log"
    return os.path.abspath(os.path.join(log_dir, fn))


def setup_logging(
    log_dir,
    recipe_name,
    target,
    run_id=None,
    level="INFO",
    console=True,
    stream=None,
    console_filter=None,
):
    """Install per-invocation file (+ optional console) handlers on root.

    Writes ``{log_dir}/{run_id}/kpf_{recipe_name}_{target}_{stamp}.log``, creating
    the run directory as needed.

    - Tears down any handlers a previous setup_logging installed, so
      repeated calls never duplicate handlers.
    - Opens the log file with exclusive create; on a name collision (two
      instances starting the same second) it retries with a numeric suffix
      (``.1``, ``.2``, ...) so concurrent instances never share a file
      (DRP-RUN-12).
    - Formats records with UT timestamps (``time.gmtime``).
    - Sets the root logger level, pins chatty third-party loggers to
      WARNING, and calls ``logging.captureWarnings(True)`` so any
      third-party/stdlib ``warnings.warn`` still lands in the log at WARNING
      (pipeline code logs recoverable conditions via ``logger.warning``
      directly; DRP-RUN-08).

    Parameters
    ----------
    log_dir : str
        The configured parent log directory (DRP-RUN-07/09).
    recipe_name : str
        Short recipe identifier, e.g. 'science' or 'masters'.
    target : str
        The reduction target: obs_id, datecode, or 'run' when neither applies.
    run_id : str or None
        This run's directory name: a launching script forwards its own so the
        whole process tree logs together; None mints ``run_{stamp}``.
    level : str
        Logging level name; INFO is the production level.
    console : bool
        Also mirror records to a console via a StreamHandler.
    stream : file-like or None
        Console destination when ``console`` is true; ``None`` means stderr
        (``logging.StreamHandler``'s default). The batch orchestrators pass
        ``sys.stdout`` so their live progress stays on stdout.
    console_filter : logging.Filter or None
        Optional filter attached to the console handler only (never the file
        handler), to trim what the terminal echoes. ``setup_batch_logging`` passes
        one to quiet sub-WARNING library chatter on the batch console; ``None``
        (the leaf default) echoes every record at the root level.

    Returns
    -------
    str
        The absolute path of the created log file.

    Raises
    ------
    ValueError
        If ``log_dir`` is empty/not a string, or ``level`` is an unknown level name.
    FileExistsError
        If a unique log file cannot be created after the collision retries.
    """
    global _prior_root_level
    teardown_logging()

    level_int = get_level(level)
    # Before the join, which would mask an unset log_dir: os.path.join("", run_id)
    # is a truthy relative path, so log_filename's own check would never fire.
    if not log_dir or not isinstance(log_dir, str):
        raise ValueError(f"log_dir must be a non-empty string; got {log_dir!r}")
    run_dir = os.path.join(log_dir, run_id or f"run_{_ut_stamp()}")
    log_path = log_filename(run_dir, recipe_name, target)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = _open_file_handler(log_path)
    file_handler.set_name("kpfpipe_file")
    log_path = file_handler.baseFilename

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
    formatter.converter = time.gmtime

    root = logging.getLogger()
    _prior_root_level = root.level
    root.setLevel(level_int)

    new_handlers = [file_handler]
    if console:
        console_handler = logging.StreamHandler(stream)  # stream=None -> stderr
        console_handler.set_name("kpfpipe_console")
        if console_filter is not None:
            console_handler.addFilter(console_filter)
        new_handlers.append(console_handler)
    for handler in new_handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
        _installed_handlers.append(handler)

    for name in _THIRD_PARTY_PINS:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.captureWarnings(True)
    return log_path


def setup_batch_logging(log_dir, label, run_id=None, level="INFO", console=True):
    """Resolve this run's id and install per-invocation handlers for a batch driver.

    Sibling to ``setup_logging``, which stays the leaf-only, per-recipe entry:
    same machinery and layout, but for the fan-out drivers. Called once at the top
    of an orchestrator's ``main()`` (``masters``/``science``/``timeseries``;
    ``label`` is the stage name). Writes ``kpf_{label}_batch_{stamp}.log`` -- the
    batch's own decision points, from units dispatched to the failure sentinels --
    alongside each unit's per-reduction log, never in place of it. Its console echo
    goes to ``sys.stdout``, filtered by ``_BatchConsoleFilter`` so the live view
    stays the driver's narration while the file keeps every record.

    Parameters
    ----------
    log_dir : str
        The configured parent log directory (DRP-RUN-07/09).
    label : str
        Short orchestrator identifier, e.g. 'masters' or 'science'.
    run_id : str or None
        A launching script's run id, used verbatim; None mints ``{label}_{stamp}``.
    level : str
        Logging level name; INFO is the production level.
    console : bool
        Also mirror records to stdout via a StreamHandler.

    Returns
    -------
    (str, str)
        This run's id -- forward it to child scripts as ``--run_id`` so they log
        beside this batch -- and the absolute path of the created batch log file.
    """
    run_id = run_id or f"{label}_{_ut_stamp()}"
    return run_id, setup_logging(
        log_dir,
        recipe_name=label,
        target="batch",
        run_id=run_id,
        level=level,
        console=console,
        stream=sys.stdout,
        console_filter=_BatchConsoleFilter(),
    )


def teardown_logging():
    """Remove and close every handler setup_logging installed.

    Also restores the default ``warnings.showwarning`` via
    ``logging.captureWarnings(False)``. Safe to call when nothing is
    installed. Primarily for tests; the CLI relies on process exit.
    """
    global _prior_root_level
    root = logging.getLogger()
    while _installed_handlers:
        handler = _installed_handlers.pop()
        root.removeHandler(handler)
        handler.close()
    if _prior_root_level is not None:
        root.setLevel(_prior_root_level)
        _prior_root_level = None
    logging.captureWarnings(False)


def _open_file_handler(log_path):
    """Exclusively create the log file, suffixing ``.1``, ``.2``, ... on
    collision; the returned FileHandler records the winning path as
    ``.baseFilename``."""
    for i in range(_MAX_COLLISION_RETRIES + 1):
        candidate = log_path if i == 0 else f"{log_path}.{i}"
        try:
            return logging.FileHandler(candidate, mode="x", encoding="utf-8")
        except FileExistsError:
            continue
    raise FileExistsError(
        f"could not create a unique log file after {_MAX_COLLISION_RETRIES} "
        f"retries; last tried {candidate!r}"
    )
