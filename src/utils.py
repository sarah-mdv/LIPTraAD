import logging
from pathlib import Path
from datetime import datetime
import getpass
from typing import List, Tuple
import pandas as pd
import numpy as np

"""
This file implements commonly used methods. In addition to that it also stores
the paths which are used throughout the entire project for files or directories.

Change the path here to adjust the loading and saving behaviour of ALL modules
in the pipeline.
"""

LOGGER = logging.getLogger(__name__)

SRC_DIR = Path(__file__).parent.resolve()
DATA_DIR = SRC_DIR / Path("data")
LOG_DIR = SRC_DIR / Path("logs")
PREPROCESSING_DIR = DATA_DIR / Path("preprocessed")

TADPOLE_DIR = DATA_DIR / Path("TADPOLE")
ADNIMERGE = TADPOLE_DIR / Path("ADNIMERGE.csv")


def _setup_logger(log_level: int) -> None:
    """
    Setting up logger such that a console handler forwards log statements to
    the console which match LOG_LEVEL (CL argument) and file handler which logs
    all messages independent of their log level.
    """
    logger = logging.getLogger(__package__)
    console_handler = logging.StreamHandler()
    date = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    log_path = LOG_DIR / Path(f"{date}.log")
    if not log_path.parent.exists():
        log_path.parent.mkdir(parents=True)
    file_handler = logging.FileHandler(str(log_path))

    # set the log level of the LOGGER to debug such that no messages are discarded
    logger.setLevel(logging.DEBUG)
    # what is print in the console should match the level specified by -v{v}
    console_handler.setLevel(log_level)
    # in the file we want all log messages again
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s- %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
