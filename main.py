"""
main.py

Main entry point for the Sophia_Alpha2_ResonantBuild application.
Handles command-line argument parsing and initializes the appropriate
operational mode (e.g., CLI, GUI).
"""

import argparse
import sys
import os
import traceback # For logging unhandled exceptions

# Ensure the project root is in sys.path for module resolution
try:
    project_root_path = os.path.abspath(os.path.dirname(__file__))
    if project_root_path not in sys.path:
        sys.path.insert(0, project_root_path)
except NameError: # __file__ not defined (e.g. in some embedded environments)
    print("Warning (main.py): __file__ not defined, assuming current directory is project root for path setup.", file=sys.stderr)
    project_root_path = os.path.abspath(os.getcwd())
    if project_root_path not in sys.path:
         sys.path.insert(0, project_root_path)

# Config Import
try:
    from config import config # Assuming config.py is within a 'config' package/directory
except ImportError as e_config_import:
    print(f"FATAL ERROR (main.py): Could not import the 'config' module. Exception: {e_config_import}", file=sys.stderr)
    print("Ensure 'config/config.py' exists and the project structure is correct.", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
except Exception as e_config_general: # Catch other potential errors during config loading
    print(f"FATAL ERROR (main.py): An unexpected error occurred while importing/loading 'config'. Exception: {e_config_general}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# Placeholder for a global logger if main decides to set one up.
# For now, specific modules handle their own logging or print.

def main_logic(cli_args):
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
        from core import dialogue as core_dialogue
        from core import gui as core_gui
        from core import brain as core_brain
        import json # For pretty printing dicts if verbose
    except ImportError as e_core_import:
        print(f"FATAL ERROR (main.py): Could not import core modules (dialogue, gui, brain). Exception: {e_core_import}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # 2. Configuration Override (VERBOSE_OUTPUT is set based on CLI args before this function is called)
    #    Other config overrides based on CLI args could be placed here if needed in the future.
    
    # Get default GUI enablement from config, defaulting to False if not set.
    enable_gui_default = getattr(config, 'ENABLE_GUI', False)

    # 3. Effective Interface Determination
    #    Decide whether to launch CLI or GUI based on args and config.
    effective_interface = cli_args.interface # User's preference from --interface.
    
    if cli_args.query: # If a direct query is provided, always force CLI mode.
        effective_interface = "cli"
        if config.VERBOSE_OUTPUT and cli_args.interface == "gui":
            print("INFO (main.py): --query provided, overriding --interface='gui' to 'cli'.")
    elif effective_interface is None: # If no --interface arg was given.
        # Default to GUI if ENABLE_GUI is True in config, otherwise default to CLI.
        effective_interface = "gui" if enable_gui_default else "cli"
    
    # Fallback: If GUI was chosen (either by arg or config default) but ENABLE_GUI is False in config, force CLI.
    # This ensures config.ENABLE_GUI is the ultimate arbiter for enabling GUI mode.
    if effective_interface == "gui" and not enable_gui_default:
        if config.VERBOSE_OUTPUT: 
            print("WARNING (main.py): GUI mode was selected or defaulted, but config.ENABLE_GUI is False. Falling back to CLI mode.")
        effective_interface = "cli"
    
    if config.VERBOSE_OUTPUT: 
        print(f"INFO (main.py): Effective interface determined: '{effective_interface}'.")

    # 4. Directory Ensuring (Safeguard)
    #    Ensure that all essential data directories defined in the config exist.
    #    This is a safeguard; ideally, components manage their own directories or config.py ensures them at load.
    if config.VERBOSE_OUTPUT: 
        print("INFO (main.py): Ensuring essential directories exist as per configuration...")
    essential_dirs_to_check = ['DATA_DIR', 'LOG_DIR', 'PERSONA_DIR', 'MEMORY_STORE_DIR', 'LIBRARY_STORE_DIR', 'ETHICS_STORE_DIR']
    for dir_config_key in essential_dirs_to_check:
        directory_path_from_config = getattr(config, dir_config_key, None)
        if directory_path_from_config and isinstance(directory_path_from_config, str):
            try:
                # config.ensure_path is expected to create the directory if it doesn't exist.
                # It should handle both file paths (ensuring parent dir) and directory paths.
                # For directory paths, it's often expected they end with os.sep, but a robust
                # ensure_path might handle this. If config.ensure_path is not available (e.g. in
                # minimal test configs), this might be skipped or error.
                if hasattr(config, 'ensure_path'):
                    config.ensure_path(directory_path_from_config) 
                elif config.VERBOSE_OUTPUT: # Log if ensure_path utility is missing from config
                     print(f"WARNING (main.py): config.ensure_path utility not found. Cannot ensure directory for {dir_config_key} ('{directory_path_from_config}').")
            except Exception as e_ensure_dir: # Catch any error during directory creation.
                print(f"WARNING (main.py): Failed to ensure directory for {dir_config_key} ('{directory_path_from_config}'). Error: {e_ensure_dir}", file=sys.stderr)
        elif config.VERBOSE_OUTPUT: # Log if directory attribute is missing or invalid in config.
            print(f"INFO (main.py): Directory attribute '{dir_config_key}' not found or invalid in config. Skipping directory creation for it.")

    # 5. Centralized SpacetimeManifold Initialization
    #    Initialize the SNN/cognitive core. `force_recreate=True` ensures a fresh instance,
    #    which can be important if main.py is re-run in some development scenarios or if
    #    config changes (like VERBOSE_OUTPUT) should affect its initialization.
    if config.VERBOSE_OUTPUT: 
        print("INFO (main.py): Initializing SpacetimeManifold (cognitive core)...")
    try:
        manifold_instance = core_brain.get_shared_manifold(force_recreate=True) 
        if manifold_instance is None: # If initialization returns None (failure).
            print("WARNING (main.py): SpacetimeManifold initialization failed. SNN-dependent features may be limited or unavailable.", file=sys.stderr)
        elif config.VERBOSE_OUTPUT:
            print("INFO (main.py): SpacetimeManifold initialized successfully.")
    except AttributeError: # If get_shared_manifold is missing from brain module (e.g., import issues).
        print("ERROR (main.py): core.brain.get_shared_manifold function not found. The brain module might be outdated, incomplete, or failed to import correctly.", file=sys.stderr)
    except Exception as e_manifold_init: # Catch any other unexpected errors during manifold initialization.
        print(f"ERROR (main.py): An unexpected error occurred during SpacetimeManifold initialization: {e_manifold_init}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # Print full traceback for debugging.

    # 6. Interface Launching
    #    Launch either the GUI or CLI based on `effective_interface`.
    if effective_interface == "gui":
        if config.VERBOSE_OUTPUT: 
            print("INFO (main.py): Launching GUI mode...")
        try:
            core_gui.start_gui() # Call the GUI's main entry point.
        except ImportError as e_gui_module_import: # Handle if Streamlit or other GUI deps are missing.
            print("ERROR (main.py): Failed to launch GUI due to missing components (e.g., Streamlit) or errors in the core.gui module.", file=sys.stderr)
            print(f"ImportError details: {e_gui_module_import}. Please ensure all GUI dependencies like Streamlit are installed ('pip install streamlit').", file=sys.stderr)
            print("INFO (main.py): Falling back to CLI mode due to GUI launch error.", file=sys.stderr)
            effective_interface = "cli" # Set to CLI for fallback.
        except Exception as e_gui_general_launch: # Catch other unexpected GUI launch errors.
            print(f"ERROR (main.py): An unexpected error occurred while launching the GUI: {e_gui_general_launch}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("INFO (main.py): Falling back to CLI mode due to GUI launch error.", file=sys.stderr)
            effective_interface = "cli" # Fallback to CLI.

    if effective_interface == "cli": # If CLI mode is chosen or fallen back to.
        if config.VERBOSE_OUTPUT: 
            print("INFO (main.py): Launching CLI mode...")
        try:
            if cli_args.query: # If a single query was provided via CLI args.
                # Generate a single response and print it.
                response_text, thought_steps, awareness_metrics = core_dialogue.generate_response(
                    cli_args.query, 
                    stream_thought_steps=getattr(config, 'DEFAULT_SINGLE_QUERY_STREAM_THOUGHTS', False)
                )
                print("\nSophia's Response:")
                print(response_text)
                # If verbose, also print awareness metrics and thought steps.
                if config.VERBOSE_OUTPUT:
                    print("\n--- Awareness Metrics (End of Query) ---")
                    print(json.dumps(awareness_metrics, indent=2, default=str)) # Use default=str for non-serializables.
                    print("\n--- Thought Steps (End of Query) ---")
                    for step_idx, step_detail in enumerate(thought_steps): print(f"{step_idx+1}. {step_detail}")
            else: # No single query, start the interactive dialogue loop.
                # VERBOSE_OUTPUT from config will control thought streaming within the loop.
                core_dialogue.dialogue_loop(enable_streaming_thoughts=config.VERBOSE_OUTPUT)
        except ImportError as e_dialogue_module_import: # Handle if core.dialogue is missing.
             print(f"ERROR (main.py): core.dialogue module not found or incomplete. CLI mode cannot start. Details: {e_dialogue_module_import}", file=sys.stderr)
             sys.exit(1) # Critical failure for CLI mode.
        except Exception as e_cli_general_launch: # Catch other unexpected CLI errors.
            print(f"ERROR (main.py): An unexpected error occurred in CLI mode: {e_cli_general_launch}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1) # Critical failure.

    # 7. Session End Log
    #    Indicates that the main logic has completed (for CLI mode or after GUI exits).
    if config.VERBOSE_OUTPUT: 
        print("INFO (main.py): Sophia_Alpha2 session processing finished.")


if __name__ == "__main__":
    # --- Argument Parsing Setup ---
    # Defines command-line arguments for interface selection, single queries, and verbosity.
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
    # `config` object is loaded globally at the top of main.py.
    # This `initial_verbose_setting` is used for early log messages before `main_logic` might further refine it.
    initial_verbose_setting = parsed_cli_args.verbose or getattr(config, 'VERBOSE_OUTPUT', False)
    
    # If verbose mode is enabled (either by CLI or config), ensure the config object reflects this.
    # This allows subsequent module logs/prints that check config.VERBOSE_OUTPUT to behave correctly.
    if initial_verbose_setting:
        if parsed_cli_args.verbose: # If CLI flag specifically set it, make it override.
            config.VERBOSE_OUTPUT = True 
        
        # Print initial startup information if verbose.
        print("INFO (main.py): Sophia_Alpha2 Initializing in VERBOSE mode...")
        # `project_root_path` is defined globally at the start of the script.
        print(f"INFO (main.py): Detected Project Root: {project_root_path}")
        system_log_path_from_config = getattr(config, 'SYSTEM_LOG_PATH', 'N/A (config or path not set)')
        print(f"INFO (main.py): System Log Path (from config): {system_log_path_from_config}")
        print(f"INFO (main.py): Parsed Command-Line Arguments: interface='{parsed_cli_args.interface}', query='{parsed_cli_args.query}', verbose_flag={parsed_cli_args.verbose}")

    # --- Main Application Execution with Top-Level Error Handling ---
    try:
        # Call the main application logic with the parsed arguments.
        main_logic(parsed_cli_args)
    except KeyboardInterrupt: # Handle graceful exit on Ctrl+C.
        # Use the initially determined verbose setting for consistency, as config.VERBOSE_OUTPUT might change.
        output_stream = sys.stderr if initial_verbose_setting else sys.stdout
        print("\nINFO (main.py): User initiated exit via KeyboardInterrupt. Shutting down gracefully.", file=output_stream)
        sys.exit(0) # Exit cleanly.
    except Exception as e_unhandled_main: # Catch any other unhandled exceptions from main_logic.
        # This is a last resort catch-all.
        print(f"FATAL UNHANDLED EXCEPTION in main.py: {type(e_unhandled_main).__name__} - {e_unhandled_main}", file=sys.stderr)
        print("Full Traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr) # Print full traceback to stderr.
        # Attempting to log to a file here is risky if config/logging itself has issues.
        # Stderr is the most reliable output at this point.
        sys.exit(1) # Exit with an error code.
```
