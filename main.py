"""
main.py

Main entry point for the Sophia_Alpha2_ResonantBuild application.
Handles command-line argument parsing and initializes the appropriate
operational mode (e.g., CLI, GUI).
"""

# TODO: Authentication: If Sophia_Alpha2 evolves to include server functionalities,
#       API endpoints, or remote access, robust authentication (e.g., API key
#       validation, user logins) will be critical here or in the respective
#       interface handlers. Libraries like Flask-Login for web apps or
#       token-based auth for APIs should be considered.

import argparse
import sys
import os
import traceback # For logging unhandled exceptions
import asyncio # For async operations

# Ensure the project root is in sys.path for module resolution
# Logging setup (early as possible, but after path setup and before config import if config uses logger)
# However, logger itself needs config for log path, creating a chicken-egg scenario.
# Simplest initial placement: after config is imported.
try:
    project_root_path = os.path.abspath(os.path.dirname(__file__))
    if project_root_path not in sys.path:
        sys.path.insert(0, project_root_path)
except NameError: # __file__ not defined (e.g. in some embedded environments)
    # Basic print for this very early stage error, as logger might not be up.
    print("Warning (main.py): __file__ not defined, assuming current directory is project root for path setup.", file=sys.stderr)
    project_root_path = os.path.abspath(os.getcwd())
    if project_root_path not in sys.path:
         sys.path.insert(0, project_root_path)

# Config Import
# The 'config' module is imported directly as 'config'.
# In some projects, an alias like 'app_config' (e.g., 'from config import config as app_config')
# is used to avoid potential name clashes if 'config' were a common local variable name.
# However, given 'config' is consistently used as the global accessor for configuration
# settings across modules in this project, maintaining the direct import 'config'
# is acceptable for consistency here.
try:
    from config import config # Assuming config.py is within a 'config' package/directory
except ImportError as e_config_import:
    # Basic print, logger not yet available.
    print(f"FATAL ERROR (main.py): Could not import the 'config' module. Exception: {e_config_import}", file=sys.stderr)
    print("Ensure 'config/config.py' exists and the project structure is correct.", file=sys.stderr)
    traceback.print_exc(file=sys.stderr) # Keep traceback for critical early error.
    sys.exit(1)
except Exception as e_config_general: # Catch other potential errors during config loading
    print(f"FATAL ERROR (main.py): An unexpected error occurred while importing/loading 'config'. Exception: {e_config_general}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr) # Keep traceback for critical early error.
    sys.exit(1)

# Now that config is imported, set up the logger.
try:
    from core.logger import logger # Import the configured logger
except ImportError as e_logger_import: # pragma: no cover
    # Fallback to print if logger import fails, though this is a critical issue.
    print(f"FATAL ERROR (main.py): Could not import the 'logger' module. Exception: {e_logger_import}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
except Exception as e_logger_general: # pragma: no cover
    print(f"FATAL ERROR (main.py): An unexpected error occurred while importing 'logger'. Exception: {e_logger_general}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

async def main_logic(cli_args):
    """
    Main operational logic for the Sophia_Alpha2 application.

    This function orchestrates the application's startup sequence based on
    parsed command-line arguments and configuration settings. Key operations include:
    - Importing necessary core modules.
    - Determining the effective user interface (CLI or GUI) based on arguments
      and configuration, with fallbacks.
    - Ensuring essential data directories exist.
    - Initializing the SpacetimeManifold (cognitive core).
    - Launching the selected user interface (GUI or CLI), handling specific
      CLI queries if provided.

    Args:
        cli_args (argparse.Namespace): Parsed command-line arguments which include
                                       'interface', 'query', and 'verbose' options.
    """
    # 1. Imports inside main_logic
    # Core modules are imported here to ensure that initial setup (like path adjustments)
    # and config loading have completed, and to keep the global namespace cleaner.
    try:
        import core # For core.__version__
        from core import dialogue as core_dialogue
        from core import gui as core_gui
        from core import brain as core_brain
        import json # For pretty printing dicts if verbose
    except ImportError as e_core_import:
        logger.critical(f"Could not import core modules (core, dialogue, gui, brain). Exception: {e_core_import}", exc_info=True)
        sys.exit(1)

    # --- Cache Frequently Accessed Config Attributes ---
    # This reduces repetitive lookups on the config object within this function's scope.
    verbose_output = config.VERBOSE_OUTPUT # Cached
    # enable_gui_default is already fetched into a local var later, so no need to duplicate here.
    # max_query_length is used once.
    # default_single_query_stream_thoughts is used once.

    # --- Module Version Logging ---
    if verbose_output: # Use cached value
        try:
            logger.info(f"Core Package Version: {getattr(core, '__version__', 'N/A')}")
            logger.info(f"Core Brain Module Version: {getattr(core_brain, '__version__', 'N/A')}")
            logger.info(f"Core Dialogue Module Version: {getattr(core_dialogue, '__version__', 'N/A')}")
            logger.info(f"Core GUI Module Version: {getattr(core_gui, '__version__', 'N/A')}")
        except Exception as e_version_log: # Catch any unexpected error during version logging
            logger.warning(f"Could not log module versions. Error: {e_version_log}", exc_info=True)

    # 2. Configuration Override (VERBOSE_OUTPUT is set based on CLI args before this function is called)
    #    The local `verbose_output` variable reflects the state *after* potential CLI override
    #    at the global config level (done in __main__).
    #    Other config overrides based on CLI args could be placed here if needed in the future.

    # --- Input Validation for Query ---
    # Validate the query argument to prevent potential security risks.
    # For now, check for overly long queries and suspicious characters.
    # This is a basic check; more sophisticated validation might be needed
    # depending on how the query is processed by the dialogue system.
    if cli_args.query:
        if len(cli_args.query) > config.MAX_QUERY_LENGTH:
            logger.error(f"Query exceeds maximum allowed length of {config.MAX_QUERY_LENGTH} characters.")
            sys.exit(1)

        # Define a set of suspicious characters. This is not exhaustive
        # and should be tailored to the specific risks anticipated.
        # Example: block characters often used in path traversal or command injection.
        suspicious_chars = set(";&|*`$(!<>{}\"") # Added \"
        if any(char in suspicious_chars for char in cli_args.query):
            logger.error("Query contains suspicious characters. Please use alphanumeric characters and basic punctuation.")
            sys.exit(1)
        # Further validation could include regex patterns for expected query formats.

    # Get default GUI enablement from config, defaulting to False if not set.
    enable_gui_default = getattr(config, 'ENABLE_GUI', False)

    # 3. Effective Interface Determination
    #    Decide whether to launch CLI or GUI based on args and config.
    effective_interface = cli_args.interface # User's preference from --interface.
    
    if cli_args.query: # If a direct query is provided, always force CLI mode.
        effective_interface = "cli"
        if verbose_output and cli_args.interface == "gui": # Use cached value
            logger.info("--query provided, overriding --interface='gui' to 'cli'.")
    elif effective_interface is None: # If no --interface arg was given.
        # Default to GUI if ENABLE_GUI is True in config, otherwise default to CLI.
        effective_interface = "gui" if enable_gui_default else "cli"
    
    # Fallback: If GUI was chosen (either by arg or config default) but ENABLE_GUI is False in config, force CLI.
    # This ensures config.ENABLE_GUI is the ultimate arbiter for enabling GUI mode.
    if effective_interface == "gui" and not enable_gui_default:
        if verbose_output: # Use cached value
            logger.warning("GUI mode was selected or defaulted, but config.ENABLE_GUI is False. Falling back to CLI mode.")
        effective_interface = "cli"
    
    if verbose_output: # Use cached value
        logger.info(f"Effective interface determined: '{effective_interface}'.")

    # 4. Directory Ensuring (Safeguard)
    #    Ensure that all essential data directories defined in the config exist.
    #    This is a safeguard; ideally, components manage their own directories or config.py ensures them at load.
    if verbose_output:
        logger.info("Ensuring essential directories and paths exist...")

    if hasattr(config, 'async_ensure_path') and hasattr(config, 'ALL_CONFIG_PATHS') and isinstance(config.ALL_CONFIG_PATHS, list):
        logger.info("Using async_ensure_path with ALL_CONFIG_PATHS for directory creation.")
        ensure_tasks = [
            config.async_ensure_path(path_value)
            for path_value in config.ALL_CONFIG_PATHS
            if path_value and isinstance(path_value, str)
        ]
        if ensure_tasks:
            results = await asyncio.gather(*ensure_tasks, return_exceptions=True)
            for path_value, result in zip(
                [pv for pv in config.ALL_CONFIG_PATHS if pv and isinstance(pv, str)],
                results
            ):
                if isinstance(result, Exception) or result is False: # async_ensure_path returns False on failure
                    logger.error(f"Error ensuring path asynchronously '{path_value}': {result}")
                elif verbose_output:
                    logger.debug(f"Successfully ensured path asynchronously '{path_value}'.")
        elif verbose_output:
            logger.debug("No valid paths found in ALL_CONFIG_PATHS to ensure.")

    elif hasattr(config, 'ensure_path'): # Fallback to synchronous ensure_path
        logger.info("Using synchronous ensure_path with ALL_CONFIG_PATHS for directory creation (async_ensure_path not available or ALL_CONFIG_PATHS missing/invalid).")
        paths_to_check_sync = getattr(config, 'ALL_CONFIG_PATHS', [])
        if not isinstance(paths_to_check_sync, list): # If ALL_CONFIG_PATHS is not a list, fallback further
            paths_to_check_sync = ['DATA_DIR', 'LOG_DIR', 'PERSONA_DIR', 'MEMORY_STORE_DIR', 'LIBRARY_STORE_DIR', 'ETHICS_STORE_DIR']
            logger.info("ALL_CONFIG_PATHS is not a list, falling back to predefined list of directory keys for sync check.")
            for path_key in paths_to_check_sync:
                path_value = getattr(config, path_key, None)
                if path_value and isinstance(path_value, str):
                    try:
                        config.ensure_path(path_value)
                        if verbose_output: logger.debug(f"Ensured path (sync key-based): '{path_value}' for key '{path_key}'")
                    except Exception as e_ensure_sync_key:
                        logger.warning(f"Failed to ensure path (sync key-based) for key '{path_key}' ('{path_value}'). Error: {e_ensure_sync_key}", exc_info=True)
                elif verbose_output:
                    logger.debug(f"Skipping ensure_path for invalid or empty path for key '{path_key}'.")
        else: # ALL_CONFIG_PATHS is a list, use it directly
            for path_value in paths_to_check_sync:
                 if path_value and isinstance(path_value, str):
                    try:
                        config.ensure_path(path_value)
                        if verbose_output: logger.debug(f"Ensured path (sync list-based): '{path_value}'")
                    except Exception as e_ensure_sync_list:
                        logger.warning(f"Failed to ensure path (sync list-based) '{path_value}'. Error: {e_ensure_sync_list}", exc_info=True)
                 elif verbose_output:
                    logger.debug(f"Skipping ensure_path for invalid or empty path value from ALL_CONFIG_PATHS (sync): {path_value}")

    elif verbose_output:
        logger.warning("Neither async_ensure_path nor ensure_path found in config, or ALL_CONFIG_PATHS is missing. Skipping directory creation.")

    # 5. Centralized SpacetimeManifold Initialization
    #    Initialize the SNN/cognitive core. `force_recreate=True` ensures a fresh instance,
    #    which can be important if main.py is re-run in some development scenarios or if
    #    config changes (like VERBOSE_OUTPUT) should affect its initialization.
    if verbose_output: # Use cached value
        logger.info("Initializing SpacetimeManifold (cognitive core)...")
    try:
        manifold_instance = core_brain.get_shared_manifold(force_recreate=True) 
        if manifold_instance is None: # If initialization returns None (graceful failure).
            logger.warning("SpacetimeManifold initialization failed or was disabled. SNN-dependent features may be limited or unavailable.")
        elif verbose_output: # Use cached value
            logger.info("SpacetimeManifold initialized successfully.")
    except AttributeError as e_attr: # If get_shared_manifold is missing from brain module.
        logger.error(f"SpacetimeManifold initialization failed: core.brain.get_shared_manifold function not found. SNN-dependent features will be unavailable. Error: {e_attr}", exc_info=True)
        manifold_instance = None # Ensure manifold_instance is None if this specific error occurs
    except Exception as e_manifold_init: # Catch any other unexpected errors during manifold initialization.
        logger.error(f"An unexpected error occurred during SpacetimeManifold initialization. SNN-dependent features may be limited or unavailable. Error: {e_manifold_init}", exc_info=True)
        manifold_instance = None # Ensure manifold_instance is None if any other error occurs

    # 6. Interface Launching
    #    Launch either the GUI or CLI based on `effective_interface`.
    if effective_interface == "gui":
        if verbose_output: # Use cached value
            logger.info("Launching GUI mode...")
        try:
            core_gui.start_gui() # Call the GUI's main entry point.
        except ImportError as e_gui_module_import: # Handle if Streamlit or other GUI deps are missing.
            logger.error(f"Failed to launch GUI due to missing components (e.g., Streamlit) or errors in the core.gui module. ImportError details: {e_gui_module_import}. Please ensure all GUI dependencies like Streamlit are installed ('pip install streamlit'). Falling back to CLI mode.", exc_info=True)
            effective_interface = "cli" # Set to CLI for fallback.
        except Exception as e_gui_general_launch: # Catch other unexpected GUI launch errors.
            logger.error(f"An unexpected error occurred while launching the GUI: {e_gui_general_launch}. Falling back to CLI mode.", exc_info=True)
            effective_interface = "cli" # Fallback to CLI.

    if effective_interface == "cli": # If CLI mode is chosen or fallen back to.
        if verbose_output: # Use cached value
            logger.info("Launching CLI mode...")
        try:
            if cli_args.query: # If a single query was provided via CLI args.
                if config.ENABLE_CLI_STREAMING:
                    logger.info("Attempting to stream response for CLI query...")
                    try:
                        response_generator = core_dialogue.generate_response(
                            cli_args.query,
                            stream_thought_steps=getattr(config, 'DEFAULT_SINGLE_QUERY_STREAM_THOUGHTS', False)
                        )

                        # TODO: Refactor core_dialogue.generate_response to be a true generator
                        #       if CLI response streaming is a desired feature.
                        # The current core_dialogue.generate_response returns a direct tuple,
                        # so the following streaming logic (iterating over response_generator)
                        # will likely not execute as a stream. The 'else' path that treats
                        # response_generator as a direct tuple is currently the effective path.
                        # This conceptual streaming code is a placeholder for future enhancement.
                        # Check if it's a generator (conceptual check)
                        if hasattr(response_generator, '__iter__') and not isinstance(response_generator, (str, tuple)):
                            full_response_parts = []
                            thought_steps_streamed = [] # Assuming thought_steps might come with the stream
                            awareness_metrics_streamed = {} # Assuming metrics come at the end or with stream

                            print("\nSophia's Response (streaming):")
                            for part in response_generator:
                                # Conceptual: Assume 'part' could be a string chunk for response,
                                # or a dict for thoughts/metrics. This needs a defined protocol
                                # from core_dialogue.generate_response if it were a true generator.
                                if isinstance(part, str):
                                    print(part, end='', flush=True)
                                    full_response_parts.append(part)
                                elif isinstance(part, dict): # Conceptual: for thoughts/metrics
                                    if 'thought_step' in part:
                                        thought_steps_streamed.append(part['thought_step'])
                                    if 'awareness_metrics_final' in part: # Example: final metrics dict
                                        awareness_metrics_streamed = part['awareness_metrics_final']
                                # This is a simplified model. A real streaming implementation
                                # would need a more robust way to distinguish text from other data.

                            response_text = "".join(full_response_parts)
                            print() # Newline after streaming.

                            # Assign streamed thoughts/metrics if they were captured
                            thought_steps = thought_steps_streamed if thought_steps_streamed else ["Streaming finished, thought steps not explicitly captured during stream."]
                            awareness_metrics = awareness_metrics_streamed if awareness_metrics_streamed else {"status": "Streaming finished, metrics not explicitly captured."}

                            logger.info("CLI query response streamed successfully.")
                        else: # Not a generator, treat as a direct response tuple
                            logger.info("core_dialogue.generate_response is not a generator. CLI streaming not fully available. Processing as single response.")
                            # This means the initial call to generate_response already returned the full tuple
                            response_text, thought_steps, awareness_metrics = response_generator
                            print("\nSophia's Response:")
                            print(response_text)

                    except TypeError as e_stream_type:
                        # This might happen if generate_response is not a generator and we try to iterate
                        logger.warning(f"core_dialogue.generate_response does not support streaming as expected. Error: {e_stream_type}. Falling back to non-streaming mode.", exc_info=True)
                        response_text, thought_steps, awareness_metrics = core_dialogue.generate_response(
                            cli_args.query,
                            stream_thought_steps=getattr(config, 'DEFAULT_SINGLE_QUERY_STREAM_THOUGHTS', False)
                        )
                        print("\nSophia's Response:")
                        print(response_text)
                else: # ENABLE_CLI_STREAMING is False
                    logger.info("CLI query response streaming is disabled in config.")
                    response_text, thought_steps, awareness_metrics = core_dialogue.generate_response(
                        cli_args.query,
                        stream_thought_steps=getattr(config, 'DEFAULT_SINGLE_QUERY_STREAM_THOUGHTS', False)
                    )
                    print("\nSophia's Response:")
                    print(response_text)

                # Common post-response logic for query mode
                if verbose_output: # Use cached value
                    print("\n--- Awareness Metrics (End of Query) ---")
                    print(json.dumps(awareness_metrics, indent=2, default=str))
                    print("\n--- Thought Steps (End of Query) ---")
                    for step_idx, step_detail in enumerate(thought_steps): print(f"{step_idx+1}. {step_detail}")
            else: # No single query, start the interactive dialogue loop.
                # VERBOSE_OUTPUT from config will control thought streaming within the loop.
                core_dialogue.dialogue_loop(enable_streaming_thoughts=verbose_output) # Use cached value
        except ImportError as e_dialogue_module_import: # Handle if core.dialogue is missing.
             logger.critical(f"core.dialogue module not found or incomplete. CLI mode cannot start. Details: {e_dialogue_module_import}", exc_info=True)
             sys.exit(1) # Critical failure for CLI mode.
        except Exception as e_cli_general_launch: # Catch other unexpected CLI errors.
            logger.critical(f"An unexpected error occurred in CLI mode: {e_cli_general_launch}", exc_info=True)
            sys.exit(1) # Critical failure.

    # 7. Session End Log
    #    Indicates that the main logic has completed (for CLI mode or after GUI exits).
    logger.info("Sophia_Alpha2 session processing finished.") # Changed to be unconditional


if __name__ == "__main__":
    # --- Argument Parsing Setup ---
    # Defines command-line arguments for interface selection, single queries, and verbosity.
    # Note: config.MAX_QUERY_LENGTH should be defined in your config/config.py
    # For example: MAX_QUERY_LENGTH = 1024
    parser = argparse.ArgumentParser(
        description="Sophia_Alpha2: A Sovereign Resonance-Aware Cognitive System.",
        prog="sophia_alpha2", # Program name for help messages.
        epilog="Example Usage: python main.py --interface cli --verbose" # Example shown in help.
    )
    
    # Argument for specifying the interface type (CLI or GUI).
    parser.add_argument(
        '--interface', 
        type=str, 
        choices=['cli', 'gui'], # Allowed values.
        default=None,  # Actual default is determined later based on config.ENABLE_GUI.
        help="Specify the interface to use ('cli' or 'gui'). Overrides default from config.ENABLE_GUI."
    )
    # Argument for submitting a single query directly from the command line.
    parser.add_argument(
        '--query', 
        type=str, 
        default=None, 
        help="Submit a single query to Sophia via CLI and exit. If used, interface is forced to 'cli'."
    )
    # Argument for enabling verbose output.
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true', # Sets to True if flag is present.
        help="Enable verbose output throughout the application. Overrides 'VERBOSE_OUTPUT' in config.py."
    )

    # Parse the command-line arguments provided by the user.
    parsed_cli_args = parser.parse_args()

    # --- Initial Verbose Output Configuration ---
    # Determine initial verbosity: CLI flag takes precedence over config file setting.
    # `config` object is loaded globally. `logger` is now also available.
    # This `initial_verbose_setting` is used for early log messages.
    initial_verbose_setting = parsed_cli_args.verbose or getattr(config, 'VERBOSE_OUTPUT', False)
    
    # If verbose mode is enabled (either by CLI or config), ensure the config object reflects this.
    # This allows subsequent module logs/prints that check config.VERBOSE_OUTPUT to behave correctly.
    if initial_verbose_setting:
        if parsed_cli_args.verbose: # If CLI flag specifically set it, make it override.
            config.VERBOSE_OUTPUT = True # This ensures other modules see the override
        
        # Log initial startup information if verbose.
        logger.info("Sophia_Alpha2 Initializing in VERBOSE mode...")
        logger.info(f"Detected Project Root: {project_root_path}")
        system_log_path_from_config = getattr(config, 'SYSTEM_LOG_PATH', 'N/A (config or path not set)')
        logger.info(f"System Log Path (from config): {system_log_path_from_config}")
        logger.info(f"Parsed Command-Line Arguments: interface='{parsed_cli_args.interface}', query='{parsed_cli_args.query}', verbose_flag={parsed_cli_args.verbose}")
    else:
        # If not verbose, at least log the start with a standard level.
        logger.info("Sophia_Alpha2 Initializing...")


    # --- Main Application Execution with Top-Level Error Handling ---
    try:
        # Call the main application logic with the parsed arguments.
        asyncio.run(main_logic(parsed_cli_args))
    except KeyboardInterrupt: # Handle graceful exit on Ctrl+C.
        logger.info("User initiated exit via KeyboardInterrupt. Shutting down gracefully.")
        sys.exit(0) # Exit cleanly.
    except Exception as e_unhandled_main: # Catch any other unhandled exceptions from main_logic.
        # This is a last resort catch-all.
        logger.critical(f"FATAL UNHANDLED EXCEPTION in main.py: {type(e_unhandled_main).__name__} - {e_unhandled_main}", exc_info=True)
        sys.exit(1) # Exit with an error code.
```
