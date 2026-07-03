"""Pipeline logging setup: one UT-timestamped log file per invocation.

Implements the WMKO logging requirements (DRP-RUN-07/08/09/12): a
user-configurable log directory, all logs under one parent directory, and
per-invocation log files that concurrent pipeline instances never share.

Handlers are installed on the root logger so that module loggers
(``logging.getLogger(__name__)``), the ``py.warnings`` bridge, and
third-party libraries all reach the same file. Setup happens exactly once,
in the CLI entry point (tools/cli.py) -- never at import time. Library code
only ever calls ``logging.getLogger(__name__)``; with no handlers installed
(e.g. recipes driven directly by tests) records are simply dropped.
"""

import logging
import os
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


def get_level(name):
    """Map a level name ('debug' ... 'critical', any case) to its logging int.

    Parameters
    ----------
    name : str
        A standard logging level name, case-insensitive.

    Returns
    -------
    int
        The corresponding ``logging`` level constant.
    """
    try:
        return _LEVELS[str(name).upper()]
    except KeyError:
        raise ValueError(
            f"unknown log level {name!r}; expected one of {sorted(_LEVELS)}"
        ) from None


def build_log_path(log_dir, recipe_name, target, start_time=None):
    """Build the unique per-invocation log path (does not create the file).

    The layout is ``{log_dir}/{YYYYMMDD}/kpf_{recipe_name}_{target}_
    {YYYYMMDDTHHMMSS}.log`` with both date components in UT. The date
    subdirectory keeps every log under the one configured parent directory
    (DRP-RUN-09) while the per-invocation filename records what ran and when.

    Parameters
    ----------
    log_dir : str
        The configured parent log directory (DRP-RUN-07).
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
    """
    if not log_dir or not isinstance(log_dir, str):
        raise ValueError(f"log_dir must be a non-empty string; got {log_dir!r}")
    if start_time is None:
        start_time = time.gmtime()
    datecode = time.strftime("%Y%m%d", start_time)
    stamp = time.strftime("%Y%m%dT%H%M%S", start_time)
    fn = f"kpf_{recipe_name}_{target}_{stamp}.log"
    return os.path.abspath(os.path.join(log_dir, datecode, fn))


def setup_logging(log_dir, recipe_name, target, level="INFO", console=True):
    """Install per-invocation file (+ optional stderr) handlers on root.

    - Tears down any handlers a previous setup_logging installed, so
      repeated calls never duplicate handlers.
    - Creates the ``{log_dir}/{YYYYMMDD}/`` directory as needed.
    - Opens the log file with exclusive create; on a name collision (two
      instances starting the same second) it retries with a numeric suffix
      (``.1``, ``.2``, ...) so concurrent instances never share a file
      (DRP-RUN-12).
    - Formats records with UT timestamps (``time.gmtime``).
    - Sets the root logger level, pins chatty third-party loggers to
      WARNING, and calls ``logging.captureWarnings(True)`` so every
      ``warnings.warn`` lands in the log at WARNING with its ``file:lineno``
      source identification (DRP-RUN-08).

    Parameters
    ----------
    log_dir : str
        The configured parent log directory (DRP-RUN-07).
    recipe_name : str
        Short recipe identifier, e.g. 'science' or 'masters'.
    target : str
        The reduction target: obs_id, datecode, or 'run' when neither applies.
    level : str
        Logging level name; INFO is the production level.
    console : bool
        Also mirror records to stderr via a StreamHandler.

    Returns
    -------
    str
        The absolute path of the created log file.
    """
    global _prior_root_level
    teardown_logging()

    level_int = get_level(level)
    log_path = build_log_path(log_dir, recipe_name, target)
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
        console_handler = logging.StreamHandler()  # stderr
        console_handler.set_name("kpfpipe_console")
        new_handlers.append(console_handler)
    for handler in new_handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
        _installed_handlers.append(handler)

    for name in _THIRD_PARTY_PINS:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.captureWarnings(True)
    return log_path


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
