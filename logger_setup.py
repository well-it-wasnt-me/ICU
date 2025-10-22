"""
Logger Setup Module.

Configures logging for the application and provides a global logger instance.
"""

import logging


def setup_logging(log_file='face_recognition.log'):
    """
    Set up logging for the application.

    Logs are configured to output both to a file and to the console with a specified format.

    :param log_file: The filename for the log file.
    :return: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# Instantiate a global logger
logger = setup_logging()
