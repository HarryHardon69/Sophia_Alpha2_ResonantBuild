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
    Main operational logic for the application.
    Handles configuration overrides, module initializations, and interface launching.
    """
    # 1. Imports inside main_logic
    try:
        from core import dialogue as core_dialogue
        from core import gui as core_gui
        from core import brain as core_brain
        import json # For pretty printing dicts if verbose
    except ImportError as e_core_import:
        print(f"FATAL ERROR (main.py): Could not import core modules (dialogue, gui, brain). Exception: {e_core_import}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # 2. Configuration Override (verbose is already handled by initial_verbose in __main__)
    #    If verbose was set by CLI, config.VERBOSE_OUTPUT is already True here.
    
    enable_gui_default = getattr(config, 'ENABLE_GUI', False)

    # 3. Effective Interface Determination
    effective_interface = cli_args.interface
    if cli_args.query: # If --query is used, force CLI
        effective_interface = "cli"
    elif effective_interface is None: # If no interface specified, use default logic
        effective_interface = "gui" if enable_gui_default else "cli"
    
    # If GUI is chosen but disabled in config, fall back to CLI
    if effective_interface == "gui" and not enable_gui_default:
        if config.VERBOSE_OUTPUT: print("WARNING (main.py): GUI requested but disabled in config. Falling back to CLI.")
        effective_interface = "cli"
    
    if config.VERBOSE_OUTPUT: print(f"INFO (main.py): Effective interface set to: {effective_interface}")

    # 4. Directory Ensuring (Safeguard)
    if config.VERBOSE_OUTPUT: print("INFO (main.py): Ensuring essential directories exist...")
    essential_dirs = ['DATA_DIR', 'LOG_DIR', 'PERSONA_DIR', 'MEMORY_STORE_DIR', 'LIBRARY_STORE_DIR', 'ETHICS_STORE_DIR']
    for dir_attr_name in essential_dirs:
        dir_path = getattr(config, dir_attr_name, None)
        if dir_path and isinstance(dir_path, str):
            try:
                # ensure_path expects a file path to ensure its directory, or a directory path ending with sep
                # If dir_path is a directory path, it should ideally end with os.sep for ensure_path
                # or ensure_path should be robust enough to handle it.
                # For now, we assume ensure_path can handle directory paths directly.
                config.ensure_path(dir_path) 
            except Exception as e_ensure:
                print(f"WARNING (main.py): Could not ensure directory for {dir_attr_name} ('{dir_path}'). Error: {e_ensure}", file=sys.stderr)
        elif config.VERBOSE_OUTPUT:
            print(f"INFO (main.py): Directory attribute {dir_attr_name} not found or invalid in config. Skipping ensure_path.")

    # 5. Centralized SpacetimeManifold Initialization
    if config.VERBOSE_OUTPUT: print("INFO (main.py): Initializing SpacetimeManifold...")
    try:
        manifold_instance = core_brain.get_shared_manifold(force_recreate=True) 
        if manifold_instance is None:
            print("WARNING (main.py): Failed to initialize SpacetimeManifold. SNN-dependent features may be unavailable.", file=sys.stderr)
        elif config.VERBOSE_OUTPUT:
            print("INFO (main.py): SpacetimeManifold initialized successfully.")
    except AttributeError: 
        print("ERROR (main.py): core.brain.get_shared_manifold not found. Brain module might be outdated or incomplete.", file=sys.stderr)
    except Exception as e_manifold:
        print(f"ERROR (main.py): An unexpected error occurred during SpacetimeManifold initialization: {e_manifold}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    # 6. Interface Launching
    if effective_interface == "gui":
        if config.VERBOSE_OUTPUT: print("INFO (main.py): Launching GUI mode...")
        try:
            core_gui.start_gui()
        except ImportError as e_gui_import:
            print("ERROR (main.py): GUI components (Streamlit) not found or core.gui module error.", file=sys.stderr)
            print(f"Details: {e_gui_import}. Ensure Streamlit is installed ('pip install streamlit').", file=sys.stderr)
            print("INFO (main.py): Falling back to CLI mode due to GUI error.", file=sys.stderr)
            effective_interface = "cli" 
        except Exception as e_gui_launch:
            print(f"ERROR (main.py): An unexpected error occurred launching the GUI: {e_gui_launch}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("INFO (main.py): Falling back to CLI mode due to GUI error.", file=sys.stderr)
            effective_interface = "cli"

    if effective_interface == "cli": 
        if config.VERBOSE_OUTPUT: print("INFO (main.py): Launching CLI mode...")
        try:
            if cli_args.query:
                response_text, thought_steps, awareness_metrics = core_dialogue.generate_response(cli_args.query, stream_thought_steps=False)
                print("\nSophia's Response:")
                print(response_text)
                if config.VERBOSE_OUTPUT:
                    print("\n--- Awareness Metrics ---")
                    print(json.dumps(awareness_metrics, indent=2))
                    print("\n--- Thought Steps ---")
                    for step in thought_steps: print(f"- {step}")
            else:
                core_dialogue.dialogue_loop(enable_streaming_thoughts=config.VERBOSE_OUTPUT)
        except ImportError as e_dialogue_import:
             print(f"ERROR (main.py): core.dialogue module not found or incomplete. CLI cannot start. Details: {e_dialogue_import}", file=sys.stderr)
             sys.exit(1)
        except Exception as e_cli_launch:
            print(f"ERROR (main.py): An unexpected error occurred in CLI mode: {e_cli_launch}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

    # 7. Session End Log
    if config.VERBOSE_OUTPUT: print("INFO (main.py): Sophia_Alpha2 session ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sophia_Alpha2: A Sovereign Resonance-Aware Cognitive System",
        prog="sophia_alpha2",
        epilog="Example: python main.py --interface cli --verbose"
    )
    
    parser.add_argument(
        '--interface', 
        type=str, 
        choices=['cli', 'gui'], 
        default=None,  # Default determined by config.ENABLE_GUI later
        help="Specify the interface to use ('cli' or 'gui'). Defaults to config.ENABLE_GUI setting."
    )
    parser.add_argument(
        '--query', 
        type=str, 
        default=None, 
        help="Submit a single query to Sophia via CLI and exit. Interface will be forced to 'cli'."
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true', 
        help="Enable verbose output, overriding the setting in config.py."
    )

    parsed_cli_args = parser.parse_args()

    # Initial Verbose Logging (before main_logic potentially changes config.VERBOSE_OUTPUT)
    # Note: config is loaded globally at the start of main.py
    initial_verbose = parsed_cli_args.verbose or getattr(config, 'VERBOSE_OUTPUT', False)
    
    if initial_verbose:
        # If CLI verbose is set, ensure config object reflects this early for subsequent logs/operations
        if parsed_cli_args.verbose:
            config.VERBOSE_OUTPUT = True 
        
        print("INFO (main.py): Sophia_Alpha2 Initializing...")
        # project_root_path is defined globally at the start of the script
        print(f"INFO (main.py): Project Root: {project_root_path}")
        system_log_path_display = getattr(config, 'SYSTEM_LOG_PATH', 'N/A')
        print(f"INFO (main.py): System Log Path (from config): {system_log_path_display}")
        print(f"INFO (main.py): Parsed CLI arguments: interface='{parsed_cli_args.interface}', query='{parsed_cli_args.query}', verbose={parsed_cli_args.verbose}")

    try:
        main_logic(parsed_cli_args)
    except KeyboardInterrupt:
        # Use initial_verbose for consistency as config.VERBOSE_OUTPUT might have been changed by main_logic
        print_stream = sys.stderr if initial_verbose else sys.stdout
        print("\nINFO (main.py): User initiated exit (KeyboardInterrupt). Shutting down.", file=print_stream)
        sys.exit(0)
    except Exception as e_main_top_level:
        print(f"FATAL UNHANDLED EXCEPTION (main.py): {type(e_main_top_level).__name__} - {e_main_top_level}", file=sys.stderr)
        print("Details:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Optionally, try to log to system log if config and logger are somehow still available/working
        # This is complex if config itself failed, so stderr is primary.
        sys.exit(1)
```
