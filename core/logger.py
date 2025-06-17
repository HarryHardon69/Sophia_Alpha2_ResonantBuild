"""
Core logging setup for the Sophia_Alpha2_ResonantBuild application.

This module initializes a root logger with handlers for console and file output,
providing a centralized logging facility.
"""

import logging
import logging.handlers # For MemoryHandler
import sys
import atexit # To register flush on exit
from config import config # To get SYSTEM_LOG_PATH and buffering settings

# 1. Initialize Root Logger
# It's generally recommended to get the root logger and configure it,
# or get a specific application logger like logging.getLogger('sophia_alpha2').
# For simplicity in a smaller application, configuring the root logger is often done.
logger = logging.getLogger() # Get root logger
logger.setLevel(logging.DEBUG) # Set root logger level (handlers can have different levels)

# 2. Create Formatter
# Example: %(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 3. Create StreamHandler for Console Output
stream_handler = logging.StreamHandler(sys.stdout) # Or sys.stderr for errors
stream_handler.setLevel(logging.INFO) # Log INFO and above to console
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# 4. Create FileHandler for Logging to File
# Ensure the log directory exists (config.ensure_path should ideally be called,
# but config itself might use the logger, creating a circular dependency if not handled carefully.
# For now, assume config.SYSTEM_LOG_PATH's directory is ensured by config.py itself at startup)
try:
    # config.ensure_path(config.SYSTEM_LOG_PATH) # Ensure parent directory exists
    # This might be problematic if logger is imported into config.py too early.
    # A better place for ensure_path for SYSTEM_LOG_PATH might be in config.py
    # after paths are defined but before logger is potentially used by config.py.
    # Or, have a dedicated setup function.
    # For now, we trust SYSTEM_LOG_PATH is valid or will be created by config.py's own ensure_path calls.

    file_handler = logging.FileHandler(config.SYSTEM_LOG_PATH, mode='a') # 'a' for append
    file_handler.setLevel(logging.DEBUG) # Log DEBUG and above to file
    file_handler.setFormatter(formatter)

    if config.ENABLE_LOG_BUFFERING:
        memory_handler = logging.handlers.MemoryHandler(
            capacity=config.LOG_BUFFER_CAPACITY,
            flushLevel=logging.ERROR, # Flush on ERROR or higher, or when capacity is reached
            target=file_handler,
            flushOnClose=True # Ensure flush when handler is closed (e.g. by logging.shutdown)
        )
        memory_handler.setLevel(logging.DEBUG) # Capture DEBUG and above in memory
        memory_handler.setFormatter(formatter) # Formatter might not be strictly needed here if target formats
        logger.addHandler(memory_handler)

        if config.FLUSH_LOG_ON_EXIT:
            atexit.register(memory_handler.flush) # Ensure buffer is flushed at python exit

        # Keep a reference to the memory handler if explicit flush from main is ever needed
        # logger.memory_handler = memory_handler
        logger.info(f"File logging configured with MemoryHandler (capacity: {config.LOG_BUFFER_CAPACITY}).")
    else:
        logger.addHandler(file_handler) # Add FileHandler directly if buffering is disabled
        logger.info("File logging configured without MemoryHandler (buffering disabled).")

except Exception as e:
    # Use a basic print here if logger setup itself fails.
    print(f"CRITICAL (logger.py): Failed to initialize FileHandler/MemoryHandler for logging. Error: {e}", file=sys.stderr)
    # Optionally, re-raise or exit if file logging is critical.

# Example usage (primarily for testing the logger setup directly)
if __name__ == '__main__': # pragma: no cover
    logger.debug("This is a debug message (should go to file, possibly buffered).")
    logger.info("This is an info message (should go to console and file, possibly buffered).")
    logger.warning("This is a warning message (should go to console and file).")
    logger.error("This is an error message (should go to console and file).")
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("This is an exception message (should go to console and file with traceback).")

    print(f"\nLogging configured. Check console output and '{config.SYSTEM_LOG_PATH}' for messages.")
