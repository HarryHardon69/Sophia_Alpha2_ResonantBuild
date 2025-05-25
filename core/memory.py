"""
Handles Sophia_Alpha2's memory operations.

This module includes functionalities for:
- Storing and retrieving interaction memories and ingested knowledge.
- Calculating concept novelty based on spatial and textual features.
- Managing a persistent knowledge graph (`knowledge_graph.json`).
- Filtering memories based on ethical alignment and novelty thresholds.
"""

import os
import sys
import json
import hashlib
import time # For timestamps and test delays
import datetime # For timestamps
import numpy as np # For numerical operations, e.g., novelty calculation

# Attempt to import configuration from the parent package
try:
    from .. import config
except ImportError:
    # Fallback for standalone execution or testing
    print("Memory.py: Could not import 'config' from parent package. Attempting relative import for standalone use.")
    try:
        import config
        print("Memory.py: Successfully imported 'config' directly (likely for standalone testing).")
    except ImportError:
        print("Memory.py: Failed to import 'config' for standalone use. Critical error.")
        config = None # Placeholder

# Further module-level constants or setup can go here.

# --- Module-Level Logging ---
LOG_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40, "critical": 50} # Duplicated from brain for now, consider centralizing

def _log_memory_event(event_type: str, data: dict, level: str = "info"):
    """
    Logs a structured system event from the memory module.
    Data is serialized to JSON.
    Respects LOG_LEVEL from config.
    """
    if not config or not hasattr(config, 'SYSTEM_LOG_PATH') or not hasattr(config, 'LOG_LEVEL'):
        print(f"Warning (memory.py): Config not available for _log_memory_event. Event: {event_type}, Data: {data}")
        # Avoid print loop if logging itself is the source of config issue for logging
        if event_type == "logging_error" and "Config not available" in data.get("error",""):
            return
        # Fallback to print if essential config for logging is missing
        print(f"MEMORY_EVENT ({level.upper()}): {event_type} - Data: {json.dumps(data, default=str)}")
        return

    # Re-check numeric_level calculation, ensure config.LOG_LEVEL is valid
    config_log_level_str = getattr(config, 'LOG_LEVEL', 'INFO').lower()
    numeric_level = LOG_LEVELS.get(level.lower(), LOG_LEVELS["info"])
    config_numeric_level = LOG_LEVELS.get(config_log_level_str, LOG_LEVELS["info"])

    if numeric_level < config_numeric_level:
        return # Skip logging if event level is below configured log level

    try:
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "module": "memory",
            "event_type": event_type,
            "level": level.upper(),
            "data": data
        }
        
        # Ensure the log directory exists (config.ensure_path should handle this for SYSTEM_LOG_PATH at import)
        # This might be called before config fully initializes ensure_path if there's an early log.
        # However, config.SYSTEM_LOG_PATH itself implies its parent dir should be ensured by config.py's own init.
        
        # Ensure SYSTEM_LOG_PATH's directory exists before writing
        # This is a safeguard in case config.py's ensure_path hasn't run or is insufficient
        log_file_path = config.SYSTEM_LOG_PATH
        log_dir = os.path.dirname(log_file_path)
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception as e_mkdir:
                # Very first log event might not be able to create dir if perms are wrong.
                print(f"CRITICAL (memory.py): Could not create log directory {log_dir}. Error: {e_mkdir}", file=sys.stderr)
                return # Cannot log to file if dir creation fails.

        with open(log_file_path, 'a') as f:
            f.write(json.dumps(log_entry, default=str) + '\n') # Use default=str for non-serializable data
            
    except Exception as e:
        # Fallback to print if logging to file fails
        print(f"Error logging memory event to file: {e}", file=sys.stderr)
        print(f"Original memory event: {event_type}, Data: {data}", file=sys.stderr)
        # Also log the error itself using the same mechanism, but to avoid recursion, check event_type
        if event_type != "logging_error":
             _log_memory_event("logging_error", {"error": str(e), "original_event": event_type}, level="error")

if __name__ == '__main__':
    print("core/memory.py loaded.")
    if config:
        print(f"Config module successfully imported. KNOWLEDGE_GRAPH_PATH: {getattr(config, 'KNOWLEDGE_GRAPH_PATH', 'N/A')}")
    else:
        print("Config module failed to import. Memory module functionality will be limited.")
    print(f"NumPy version: {np.__version__}")
