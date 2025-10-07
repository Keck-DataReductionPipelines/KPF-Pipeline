"""
polly

log

Set up the package's logging module. By default this logs into a single file in my
directory, as well as to the console.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger("Polly")

# create file handler which logs even debug messages
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

log_file = os.getenv('KPF_POLLY_LOG_FILE')
if log_file is None:
    # Default path is for execution outside of Docker container on shrek.
    log_file = Path("/scr/jpember/polly_outputs/polly.log")
file_handler = logging.FileHandler(log_file)

file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# create console handler with a higher log level
stdout_formatter = logging.Formatter("%(message)s")
stdout = logging.StreamHandler()
stdout.setLevel(logging.INFO)
stdout.setFormatter(stdout_formatter)
logger.addHandler(stdout)

# Set default logger level
logger.setLevel(logging.INFO)
