# src/utils/logger_setup.py

import logging
import sys # For standard output/error streams: sys.stdout, sys.stderr
import traceback # For printing stack traces

DEFAULT_LOG_LEVEL = logging.INFO  # Default log level (can be moved to config later)
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' # Log format
DATE_FORMAT = '%Y-%m-%d %H:%M:%S' # Time format

def setup_logger(name: str, level: int = DEFAULT_LOG_LEVEL) -> logging.Logger | None: # Return type may be None
    try:
        logger = logging.getLogger(name) # Get (or create) a logger with the given name
        logger.setLevel(level) # Only messages >= level are passed to handlers

        handler_exists = any( # Prevent adding duplicate StreamHandlers to stdout (avoids duplicate console logs)
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
        for h in logger.handlers)

        if not handler_exists:
            ch = logging.StreamHandler(sys.stdout) # StreamHandler sends logs to a stream (stdout/stderr)
            ch.setLevel(level) # Handler has its own level filter

            formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT) # Formatter defines final log output format
            ch.setFormatter(formatter)

            logger.addHandler(ch) # Logger can have multiple handlers

        return logger # Return configured logger
    except Exception as e:
        error_message = f"ERROR in '{__name__}': Failed to setup logger '{name}': {e}\n"
        sys.stderr.write(error_message)
        sys.stderr.write(traceback.format_exc() + "\n") # Print stack trace to stderr
        return None # Explicitly return None to indicate setup failure