# This file is just for compatibility with xmris
import sys


def set_log_level(level: str = "", verbose: bool = True):
    try:
        from loguru import logger

        logger.remove()
        logger.add(sys.stdout, level=level.upper())

    except ImportError:
        print(
            "Loguru is not installed. Please install it to use logging features. "
            "You can do this by running 'uv add loguru'."
        )
