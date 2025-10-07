"""
polly

log

Set up the package's logging module. By default this logs into a single file in my
directory, as well as to the console.
"""

import logging

logger = logging.getLogger("Polly")

# create file handler which logs even debug messages
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler("/data/analysis/polly.log")
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
