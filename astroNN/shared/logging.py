import logging
import threading
import sys

_logger = None
_logger_lock = threading.Lock()


def get_logger():
    """Return logger instance."""
    global _logger

    # Use double-checked locking to avoid taking lock unnecessarily.
    if _logger:
        return _logger

    _logger_lock.acquire()

    try:
        if _logger:
            return _logger

        logger = logging.getLogger("astroNN")

        _logger = logger
        _handler = logging.StreamHandler(sys.stdout)
        _handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
        logger.addHandler(_handler)
        return _logger

    finally:
        _logger_lock.release()


def log(level, msg, *args, **kwargs):
    get_logger().log(level, msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    get_logger().debug(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    get_logger().error(msg, *args, **kwargs)


def fatal(msg, *args, **kwargs):
    get_logger().fatal(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    get_logger().info(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    get_logger().warning(msg, *args, **kwargs)
