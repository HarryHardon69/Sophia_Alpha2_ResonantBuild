"""
Manages Sophia_Alpha2's identity, traits, operational mode, and awareness state.

This module defines the Persona class, which encapsulates Sophia's characteristics
and her evolving understanding of her own operational state based on metrics
received from the cognitive core (brain.py). It handles persistence of this
state to a profile file.
"""

import os
import sys
import json
import datetime
import traceback # For detailed error logging in load_state

# Attempt to import configuration from the parent package
try:
    from .. import config
except ImportError:
    # Fallback for standalone execution or testing
    print("Persona.py: Could not import 'config' from parent package. Attempting relative import for standalone use.")
    try:
        import config
        print("Persona.py: Successfully imported 'config' directly (likely for standalone testing).")
    except ImportError as e_config:
        print(f"Persona.py: Failed to import 'config' for standalone use. Critical error: {e_config}")
        config = None # Placeholder, operations requiring config will fail

class Persona:
    """
    Manages Sophia_Alpha2's identity, traits, operational mode, and awareness state.
    Handles persistence of this state to a profile JSON file.
    """
    def __init__(self):
        # Default attributes
        default_name = "Sophia_Alpha2_Default"
        self.name = default_name
        self.mode = "reflective"  # e.g., reflective, learning, active_problem_solving
        self.traits = ["CuriosityDriven", "EthicallyMinded", "ResonanceAware", "Developmental"]
        
        # Awareness metrics - initialized with sensible defaults
        self.awareness = {
            "curiosity": 0.5,
            "context_stability": 0.5,
            "self_evolution_rate": 0.0,
            "coherence": 0.0,
            "active_llm_fallback": False,
            "primary_concept_coord": None  # Expected to be a 4-tuple (x,y,z,t_intensity) or None
        }

        default_profile_filename = "persona_profile_default.json"
        # Fallback profile path determination (module's directory/data/private if possible)
        # This is a bit tricky as __file__ might not be reliable in all execution contexts (e.g. bundled)
        try:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            fallback_data_dir = os.path.join(os.path.dirname(module_dir), 'data', 'private') # Assumes core/ is one level down from project root
        except NameError: # __file__ not defined (e.g. interactive session, some bundled scenarios)
            fallback_data_dir = os.path.join('.', 'data', 'private') # CWD/data/private as a last resort

        self.profile_path = os.path.join(fallback_data_dir, default_profile_filename)


        if config:
            self.name = getattr(config, 'PERSONA_NAME', default_name)
            # Construct profile path using config attributes if available
            # config.PERSONA_PROFILE_PATH should be the full, correct path from config.py
            self.profile_path = getattr(config, 'PERSONA_PROFILE_PATH', self.profile_path) # Use pre-calculated fallback if not in config
            
            if hasattr(config, 'ensure_path'):
                try:
                    config.ensure_path(self.profile_path)
                except Exception as e_ensure:
                    print(f"Warning (Persona __init__): Error ensuring profile path '{self.profile_path}' via config.ensure_path: {e_ensure}", file=sys.stderr)
            else:
                # Manual ensure_path if config.ensure_path is not available
                try:
                    profile_dir = os.path.dirname(self.profile_path)
                    if profile_dir and not os.path.exists(profile_dir):
                        os.makedirs(profile_dir, exist_ok=True)
                except Exception as e_manual_mkdir:
                     print(f"Warning (Persona __init__): Error manually ensuring profile directory for '{self.profile_path}': {e_manual_mkdir}", file=sys.stderr)
        else: 
            print("Warning (Persona __init__): Config module not loaded. Persona will use default name and profile path.", file=sys.stderr)
            # Manual ensure for default path if no config, using the already determined self.profile_path
            try:
                profile_dir = os.path.dirname(self.profile_path)
                if profile_dir and not os.path.exists(profile_dir):
                    os.makedirs(profile_dir, exist_ok=True)
            except Exception as e_manual_mkdir_noconf:
                print(f"Warning (Persona __init__): Error manually ensuring default profile directory for '{self.profile_path}': {e_manual_mkdir_noconf}", file=sys.stderr)

        self.load_state() 
        
        if config and getattr(config, 'VERBOSE_OUTPUT', False): # Check VERBOSE_OUTPUT from config if available
            print(f"Persona initialized: Name='{self.name}', Mode='{self.mode}', Profile='{self.profile_path}'", file=sys.stderr)
            print(f"Initial Awareness: {self.awareness}", file=sys.stderr)

    def save_state(self):
        """
        Saves the current persona state to the profile JSON file.
        Includes name, mode, traits, and the full awareness dictionary.
        """
        state_to_save = {
            "name": self.name,
            "mode": self.mode,
            "traits": self.traits,
            "awareness": self.awareness, # This will include primary_concept_coord
            "last_saved": datetime.datetime.utcnow().isoformat() + "Z"
        }

        try:
            # Ensure directory exists one last time before writing (though __init__ tries)
            profile_dir = os.path.dirname(self.profile_path)
            if profile_dir and not os.path.exists(profile_dir):
                os.makedirs(profile_dir, exist_ok=True)
                if config and getattr(config, 'VERBOSE_OUTPUT', False):
                    print(f"Info (Persona save_state): Created missing profile directory: {profile_dir}", file=sys.stderr)

            with open(self.profile_path, 'w') as f:
                json.dump(state_to_save, f, indent=2)
            
            if config and getattr(config, 'VERBOSE_OUTPUT', False):
                print(f"Info (Persona save_state): Persona state saved to {self.profile_path}", file=sys.stderr)
                
        except IOError as e_io:
            print(f"Error (Persona save_state): IOError saving persona state to {self.profile_path}: {e_io}", file=sys.stderr)
            # Optionally, log this to a system log if available via config
            # _log_system_event("persona_save_failure", {"path": self.profile_path, "error": str(e_io)}, level="error") # If logging is added
        except Exception as e:
            print(f"Error (Persona save_state): Unexpected error saving persona state to {self.profile_path}: {e}", file=sys.stderr)
            # _log_system_event("persona_save_failure_unknown", {"path": self.profile_path, "error": str(e)}, level="critical")

    def _initialize_default_state_and_save(self):
        """
        Resets the persona attributes to their default values and saves the state.
        This is typically called if a profile file is not found or is invalid.
        """
        if config and getattr(config, 'VERBOSE_OUTPUT', False):
            print(f"Info (Persona): Initializing default persona state for '{getattr(self, 'profile_path', 'N/A')}'.", file=sys.stderr)

        default_name = "Sophia_Alpha2_Default_Reset" # Slightly different to indicate it was reset
        # Use config values if config is loaded, otherwise hardcoded defaults from __init__
        self.name = getattr(config, 'PERSONA_NAME', default_name) if config else default_name
        self.mode = "reflective"
        self.traits = ["CuriosityDriven", "EthicallyMinded", "ResonanceAware", "Developmental"]
        
        self.awareness = {
            "curiosity": 0.5,
            "context_stability": 0.5,
            "self_evolution_rate": 0.0,
            "coherence": 0.0,
            "active_llm_fallback": False,
            "primary_concept_coord": None 
        }
        
        # self.profile_path should already be set by __init__
        # If it wasn't (e.g. config totally failed), save_state might have issues,
        # but it has its own fallbacks/error handling.

        if config and getattr(config, 'VERBOSE_OUTPUT', False):
            print(f"Info (Persona): Default state re-initialized. Name='{self.name}', Mode='{self.mode}'. Attempting save.", file=sys.stderr)
            
        self.save_state() # Persist these defaults

    def load_state(self):
        """
        Loads persona state from the profile JSON file.
        Handles file not found, empty file, malformed JSON, and older formats
        by gracefully applying available data and defaulting for missing fields.
        If loading fails critically or no profile exists, initializes a default state.
        """
        if not hasattr(self, 'profile_path') or not self.profile_path:
            # This should not happen if __init__ runs correctly
            print("Error (Persona load_state): profile_path not set. Cannot load.", file=sys.stderr)
            self._initialize_default_state_and_save() # Initialize and save a default
            return

        if not os.path.exists(self.profile_path) or os.path.getsize(self.profile_path) == 0:
            if config and getattr(config, 'VERBOSE_OUTPUT', False):
                print(f"Info (Persona load_state): Profile file not found or empty: '{self.profile_path}'. Initializing default state.", file=sys.stderr)
            self._initialize_default_state_and_save()
            return

        try:
            with open(self.profile_path, 'r') as f:
                loaded_state = json.load(f)

            if not isinstance(loaded_state, dict):
                print(f"Warning (Persona load_state): Profile data in '{self.profile_path}' is not a dictionary. Initializing default state.", file=sys.stderr)
                self._initialize_default_state_and_save()
                return

            # Load top-level attributes if present, otherwise keep __init__ defaults
            self.name = loaded_state.get("name", self.name)
            self.mode = loaded_state.get("mode", self.mode)
            self.traits = loaded_state.get("traits", self.traits)
            if not isinstance(self.traits, list) or not all(isinstance(t, str) for t in self.traits):
                print(f"Warning (Persona load_state): Loaded 'traits' is not a list of strings. Using default traits.", file=sys.stderr)
                self.traits = ["CuriosityDriven", "EthicallyMinded", "ResonanceAware", "Developmental"] # Default

            # Load awareness metrics, applying type validation and defaults for missing keys
            loaded_awareness = loaded_state.get("awareness", {})
            if not isinstance(loaded_awareness, dict):
                print(f"Warning (Persona load_state): Loaded 'awareness' data is not a dictionary. Using default awareness.", file=sys.stderr)
                loaded_awareness = {} # Ensure it's a dict to prevent errors below

            # Iterate over the keys defined in self.awareness by __init__ to ensure all expected keys are present
            # and correctly typed, using loaded values where valid.
            default_awareness_keys = {
                "curiosity": float, "context_stability": float, "self_evolution_rate": float,
                "coherence": float, "active_llm_fallback": bool, "primary_concept_coord": 'tuple_4_float_or_none'
            }

            for key, expected_type in default_awareness_keys.items():
                if key in loaded_awareness:
                    loaded_value = loaded_awareness[key]
                    if expected_type == float:
                        try:
                            self.awareness[key] = float(loaded_value)
                        except (ValueError, TypeError):
                            print(f"Warning (Persona load_state): Invalid type for awareness key '{key}'. Expected float, got {type(loaded_value)}. Using default.", file=sys.stderr)
                            # Default already set in self.awareness by __init__ if key was bad
                    elif expected_type == bool:
                        self.awareness[key] = bool(loaded_value) # Handles "true", "false", 0, 1 etc.
                    elif key == "primary_concept_coord": # Special handling for 'tuple_4_float_or_none'
                        if isinstance(loaded_value, (list, tuple)) and len(loaded_value) == 4:
                            try:
                                self.awareness[key] = tuple(float(v) for v in loaded_value)
                            except (ValueError, TypeError):
                                print(f"Warning (Persona load_state): Invalid numeric types in 'primary_concept_coord' list/tuple. Using None.", file=sys.stderr)
                                self.awareness[key] = None
                        elif loaded_value is None:
                            self.awareness[key] = None
                        else:
                            print(f"Warning (Persona load_state): Invalid format for 'primary_concept_coord'. Expected list/tuple of 4 or None. Got {type(loaded_value)}. Using None.", file=sys.stderr)
                            self.awareness[key] = None
                # If key not in loaded_awareness, the default from __init__ remains.
            
            if config and getattr(config, 'VERBOSE_OUTPUT', False):
                print(f"Info (Persona load_state): Persona state loaded successfully from {self.profile_path}", file=sys.stderr)
                print(f"Loaded Awareness: {self.awareness}", file=sys.stderr)

        except json.JSONDecodeError as e_json:
            print(f"Error (Persona load_state): Malformed JSON in profile file '{self.profile_path}': {e_json}. Initializing default state.", file=sys.stderr)
            self._initialize_default_state_and_save()
        except Exception as e:
            print(f"Error (Persona load_state): Unexpected error loading persona state from '{self.profile_path}': {e}. Initializing default state.", file=sys.stderr)
            # Ensure traceback is imported for full error details
            import traceback # Added import here
            traceback.print_exc(file=sys.stderr) # Print stack trace for unexpected errors
            self._initialize_default_state_and_save()

    def get_intro(self) -> str:
        """
        Generates an introductory statement for the persona, including name,
        mode, key awareness metrics, and focus intensity if available.
        """
        intro_parts = [f"Name: {self.name}", f"Mode: {self.mode}"]
        
        # Add awareness metrics
        curiosity = self.awareness.get("curiosity", 0.0)
        coherence = self.awareness.get("coherence", 0.0)
        intro_parts.append(f"Curiosity: {curiosity:.2f}")
        intro_parts.append(f"Coherence: {coherence:.2f}")

        # Add Focus Intensity (T-value) if primary_concept_coord is valid
        primary_coord = self.awareness.get("primary_concept_coord")
        if isinstance(primary_coord, (list, tuple)) and len(primary_coord) == 4:
            try:
                # The T-value is the 4th element (index 3)
                # This T-value is the *coordinate* value.
                # The raw intensity (0-1) is more intuitive for display if available.
                # Let's check if brain_awareness_metrics stored 'raw_t_intensity'
                # For now, we'll assume primary_coord[3] is what we need to show,
                # or that it's related to the raw intensity directly.
                # If primary_coord[3] is already scaled to manifold range, it might be large.
                # The prompt for brain.py's bootstrap_concept_from_llm stated:
                # "t_coord_intensity: intensity (0 to 1) * (self.range / 2) - used as a coordinate"
                # "raw_intensity = np.clip(float(llm_data["intensity"]), 0.0, 1.0)"
                # So, primary_coord[3] is the scaled one. To get back to 0-1 raw intensity for display:
                # raw_t_display = primary_coord[3] / (config.MANIFOLD_RANGE / 2.0) if config.MANIFOLD_RANGE else primary_coord[3]
                # However, 'raw_t_intensity_at_storage' is also stored in the node by memory.py
                # And brain.think() `awareness_metrics` should ideally pass the raw 0-1 `raw_t_intensity`.
                # Let's assume `self.awareness` might have `raw_t_intensity` from an update,
                # otherwise, we fall back to a message or don't display T if it's ambiguous from coord.
                # For now, let's assume primary_coord[3] is the *coordinate value*.
                # The requirement is "Focus Intensity (T): [value]". This implies the raw 0-1 intensity.
                # If only coord[3] is available, and it's scaled, we'd need config.MANIFOLD_RANGE to de-normalize.
                # Let's design get_intro to prefer a dedicated raw_t_intensity if available in self.awareness,
                # which update_awareness should aim to populate.
                # For now, if only primary_coord[3] (the coordinate) is there, we state it's the coordinate T-value.
                # A better approach: update_awareness should store both coord and raw_t_intensity.
                # Assuming update_awareness stores 'primary_concept_t_raw' if available from brain.
                
                t_value_display = None
                if 'raw_t_intensity' in self.awareness and isinstance(self.awareness['raw_t_intensity'], (int, float)):
                     t_value_display = self.awareness['raw_t_intensity']
                elif primary_coord[3] is not None : # Fallback to using the coordinate if raw not available
                    # This could be a large number if it's a coordinate.
                    # The requirement is "Focus Intensity (T): [value]". This implies the raw 0-1 intensity.
                    # If only coord[3] is available, and it's scaled, we'd need config.MANIFOLD_RANGE to de-normalize.
                    # For now, just show the coordinate value if raw isn't there.
                    # Or, better, state it's a coordinate.
                    # The prompt says: "extract its T-value (intensity, index 3)"
                    # Let's assume for now this means the raw 0-1 intensity that *should* be in awareness.
                    # If it's not, we skip. update_awareness needs to ensure it's there.
                    # For this implementation, let's assume 'primary_concept_coord' in awareness IS the raw 4-tuple (x,y,z, raw_t_intensity_0_to_1)
                    # This simplifies get_intro, but update_awareness needs to ensure this format.
                    # The brain.think() passes primary_concept_coord which *is* the scaled one.
                    # This means update_awareness needs to de-normalize or brain needs to pass raw T separately.
                    # RESOLUTION: For get_intro, I will assume primary_concept_coord[3] is the *raw 0-1 intensity*
                    # as this makes the most sense for a "Focus Intensity (T): [value]" display.
                    # This means `update_awareness` must ensure `self.awareness['primary_concept_coord'][3]` is the raw T.
                    # The brain's `awareness_metrics` has `primary_concept_coord` (scaled) AND `raw_t_intensity`.
                    # So `update_awareness` should ideally store `raw_t_intensity` separately or ensure `primary_concept_coord[3]` is the raw one.
                    # For now, let's assume `update_awareness` stores `raw_t_intensity` in `primary_concept_coord[3]`.
                    
                    # Re-evaluating based on task: "awareness_metrics['primary_concept_coord'] is a 4-tuple of FLOATS (X,Y,Z,T_Intensity_RAW_0_to_1)."
                    # This means primary_coord[3] IS the raw T intensity.
                    t_value_display = float(primary_coord[3])

                if t_value_display is not None:
                    intro_parts.append(f"Focus Intensity (T): {t_value_display:.2f}")

            except (ValueError, TypeError, IndexError):
                 # Silently ignore if coord[3] is not a valid number for T-intensity display
                pass 
                
        return " | ".join(intro_parts)

    def update_awareness(self, brain_awareness_metrics: dict):
        """
        Updates the persona's awareness state based on metrics from brain.think().

        Args:
            brain_awareness_metrics: A dictionary containing awareness metrics
                                     from the brain module. Expected keys include:
                                     'curiosity', 'context_stability', 
                                     'self_evolution_rate', 'coherence',
                                     'active_llm_fallback', 'primary_concept_coord' (scaled),
                                     and potentially 'raw_t_intensity' (0-1 scale).
        """
        if not isinstance(brain_awareness_metrics, dict):
            if config and getattr(config, 'VERBOSE_OUTPUT', False):
                print(f"Warning (Persona update_awareness): Invalid brain_awareness_metrics type: {type(brain_awareness_metrics)}. No update.", file=sys.stderr)
            return

        if config and getattr(config, 'VERBOSE_OUTPUT', False):
            print(f"Info (Persona update_awareness): Received brain metrics: {brain_awareness_metrics}", file=sys.stderr)

        changed = False
        original_awareness_str = str(self.awareness) # For change detection

        # Iterate over expected keys in self.awareness and update from brain_metrics if valid
        awareness_keys_to_process = {
            "curiosity": float, "context_stability": float, "self_evolution_rate": float,
            "coherence": float, "active_llm_fallback": bool
            # primary_concept_coord is handled specially
        }

        for key, expected_type in awareness_keys_to_process.items():
            if key in brain_awareness_metrics:
                new_value = brain_awareness_metrics[key]
                try:
                    if expected_type == float:
                        new_value = float(new_value)
                    elif expected_type == bool:
                        new_value = bool(new_value)
                    
                    if self.awareness.get(key) != new_value:
                        self.awareness[key] = new_value
                        changed = True
                except (ValueError, TypeError):
                    if config and getattr(config, 'VERBOSE_OUTPUT', False):
                        print(f"Warning (Persona update_awareness): Invalid type for metric '{key}'. Expected {expected_type}, got {type(new_value)}. Not updated.", file=sys.stderr)
        
        # Special handling for primary_concept_coord
        # We want to store: (scaled_x, scaled_y, scaled_z, raw_t_intensity_0_to_1)
        # Brain's awareness_metrics provides 'primary_concept_coord' (scaled) and 'raw_t_intensity' (0-1).
        
        brain_primary_coord_scaled = brain_awareness_metrics.get("primary_concept_coord")
        brain_raw_t_intensity = brain_awareness_metrics.get("raw_t_intensity") # This was added to brain.think() return for clarity

        new_coord_for_persona = None
        valid_coord_received = False

        if isinstance(brain_primary_coord_scaled, (list, tuple)) and len(brain_primary_coord_scaled) == 4:
            if brain_raw_t_intensity is not None: # Prefer brain's explicit raw_t_intensity
                try:
                    # Construct (scaled_x, scaled_y, scaled_z, raw_t_0_1)
                    new_coord_for_persona = (
                        float(brain_primary_coord_scaled[0]),
                        float(brain_primary_coord_scaled[1]),
                        float(brain_primary_coord_scaled[2]),
                        float(brain_raw_t_intensity) # Use the raw 0-1 intensity here
                    )
                    valid_coord_received = True
                except (ValueError, TypeError, IndexError):
                    if config and getattr(config, 'VERBOSE_OUTPUT', False):
                        print(f"Warning (Persona update_awareness): Invalid numeric types in brain's primary_concept_coord or raw_t_intensity. Coord: {brain_primary_coord_scaled}, RawT: {brain_raw_t_intensity}", file=sys.stderr)
            else: # Fallback if brain_raw_t_intensity is not provided (older brain version?)
                  # In this case, assume brain_primary_coord_scaled[3] IS the raw T (0-1), as per get_intro's assumption.
                  # This path makes get_intro's assumption more critical for brain to fulfill if raw_t_intensity isn't separate.
                try:
                    new_coord_for_persona = tuple(float(v) for v in brain_primary_coord_scaled)
                    # Ensure the T-value (coord[3]) is clipped to 0-1 if it's meant to be raw intensity here.
                    new_coord_for_persona = (new_coord_for_persona[0], new_coord_for_persona[1], new_coord_for_persona[2], 
                                             max(0.0, min(1.0, new_coord_for_persona[3])))
                    valid_coord_received = True
                    if config and getattr(config, 'VERBOSE_OUTPUT', False):
                        print(f"Warning (Persona update_awareness): 'raw_t_intensity' not found in brain metrics. Assuming primary_concept_coord[3] is raw T (0-1) and clipping.", file=sys.stderr)
                except (ValueError, TypeError, IndexError):
                    if config and getattr(config, 'VERBOSE_OUTPUT', False):
                        print(f"Warning (Persona update_awareness): Invalid numeric types in brain's primary_concept_coord when assuming direct use for raw T. Coord: {brain_primary_coord_scaled}", file=sys.stderr)
        elif brain_primary_coord_scaled is None and 'primary_concept_coord' in brain_awareness_metrics: # Explicitly None
            new_coord_for_persona = None # Set to None
            valid_coord_received = True # Valid in the sense that None is an acceptable value
        
        if valid_coord_received:
            if self.awareness.get("primary_concept_coord") != new_coord_for_persona:
                self.awareness["primary_concept_coord"] = new_coord_for_persona
                changed = True
        elif 'primary_concept_coord' in brain_awareness_metrics: # If key was there but data was bad
            if config and getattr(config, 'VERBOSE_OUTPUT', False):
                print(f"Warning (Persona update_awareness): Received malformed 'primary_concept_coord': {brain_primary_coord_scaled}. Retaining existing or None.", file=sys.stderr)


        if changed:
            if config and getattr(config, 'VERBOSE_OUTPUT', False):
                print(f"Info (Persona update_awareness): Awareness changed. Old: {original_awareness_str}, New: {self.awareness}", file=sys.stderr)
            self.save_state()
        else:
            if config and getattr(config, 'VERBOSE_OUTPUT', False):
                print(f"Info (Persona update_awareness): No change in awareness metrics.", file=sys.stderr)

if __name__ == '__main__':
    # --- Test Utilities ---
    class TempConfigOverride:
        """Temporarily overrides attributes in the config module for testing persona.py."""
        def __init__(self, temp_configs_dict):
            self.temp_configs = temp_configs_dict
            self.original_values = {}

        def __enter__(self):
            if not config:
                print("CRITICAL_TEST_ERROR (persona.py): Config module not loaded in TempConfigOverride.", file=sys.stderr)
                raise ImportError("Config module not loaded. Cannot override attributes for testing.")
            
            for key, value in self.temp_configs.items():
                if hasattr(config, key):
                    self.original_values[key] = getattr(config, key)
                else:
                    self.original_values[key] = None # Attribute didn't exist
                setattr(config, key, value)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not config:
                return
            
            for key, original_value in self.original_values.items():
                current_value_in_config = getattr(config, key, None)
                if original_value is not None: # If there was an original value, restore it.
                    if current_value_in_config != original_value:
                        setattr(config, key, original_value)
                elif key in self.temp_configs: # Attribute did not exist before but was set by this context manager.
                    if hasattr(config, key): # Check if it's still there
                        delattr(config, key)
            
            self.original_values.clear()
            self.temp_configs.clear()

    # --- Original __main__ preamble (or new test execution logic will start here) ---
    print("core/persona.py loaded as main script.")
    if config:
        print(f"Config module successfully imported. PERSONA_NAME: {getattr(config, 'PERSONA_NAME', 'N/A')}")
        print(f"Config PERSONA_PROFILE path: {getattr(config, 'PERSONA_PROFILE_PATH', 'N/A')}") # Corrected to PERSONA_PROFILE_PATH
        # Ensure paths for testing if config is available
        if hasattr(config, 'ensure_path') and hasattr(config, 'PERSONA_PROFILE_PATH'): # Corrected
            try:
                # For tests, PERSONA_PROFILE will be overridden. 
                # Ensure the default log path's directory from config exists if tests don't override it for system logs.
                if hasattr(config, 'SYSTEM_LOG_PATH'):
                    config.ensure_path(config.SYSTEM_LOG_PATH) 
            except Exception as e_ensure:
                print(f"Warning (persona.py __main__): Error ensuring default SYSTEM_LOG_PATH via config.ensure_path: {e_ensure}", file=sys.stderr)
    else:
        print("Config module failed to import. Persona module functionality will be limited.")
