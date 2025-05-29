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
    # print("Persona.py: Could not import 'config' from parent package. Attempting relative import for standalone use.")
    try:
        import config # type: ignore
        # print("Persona.py: Successfully imported 'config' directly (likely for standalone testing).")
    except ImportError as e_config:
        # print(f"Persona.py: Failed to import 'config' for standalone use. Critical error: {e_config}")
        config = None # Placeholder, operations requiring config will fail

class Persona:
    """
    Manages Sophia_Alpha2's identity, traits, operational mode, and awareness state.
    Handles persistence of this state to a profile JSON file.
    """
    def __init__(self):
        """
        Initializes the Persona instance.
        
        Sets default attributes, determines the profile path (favoring configuration
        if available, otherwise using a fallback), ensures the profile directory
        exists, and loads the persona state from the profile file (or initializes
        a default state if loading fails).
        """
        # Default attributes that will be used if not overridden by config or loaded state.
        default_name = getattr(config, 'DEFAULT_PERSONA_NAME', "Sophia_Alpha2_Default")
        self.name = default_name # Will be overridden by config.PERSONA_NAME if available
        self.mode = getattr(config, 'DEFAULT_PERSONA_MODE', "reflective")
        self.traits = getattr(config, 'DEFAULT_PERSONA_TRAITS', ["CuriosityDriven", "EthicallyMinded", "ResonanceAware", "Developmental"])
        
        # Awareness metrics - initialized with sensible defaults from config or hardcoded fallbacks
        default_awareness_metrics = getattr(config, 'DEFAULT_PERSONA_AWARENESS', {
            "curiosity": 0.5, "context_stability": 0.5, "self_evolution_rate": 0.0,
            "coherence": 0.0, "active_llm_fallback": False, "primary_concept_coord": None
        })
        self.awareness = default_awareness_metrics.copy()


        default_profile_filename = getattr(config, 'DEFAULT_PERSONA_PROFILE_FILENAME', "persona_profile_default.json")
        # Fallback profile path determination
        try:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            # Assumes core/ is one level down from project root which contains data/private
            fallback_data_dir = os.path.join(os.path.dirname(module_dir), 'data', 'private')
        except NameError: # __file__ not defined
            fallback_data_dir = os.path.join('.', 'data', 'private') # CWD/data/private

        # Initialize profile_path with a default. It will be overridden by config if available.
        self.profile_path = os.path.join(fallback_data_dir, default_profile_filename)

        # --- Profile Path Resolution and Directory Creation ---
        if config: # If the main config module is loaded.
            self.name = getattr(config, 'PERSONA_NAME', default_name) # Use persona name from config if set.
            # Prioritize PERSONA_PROFILE_PATH from config for the profile file.
            self.profile_path = getattr(config, 'PERSONA_PROFILE_PATH', self.profile_path)
            
            # Use config.ensure_path utility (if available) to create the profile directory.
            # This utility is expected to handle parent directory creation.
            if hasattr(config, 'ensure_path'):
                try:
                    config.ensure_path(self.profile_path) 
                except Exception as e_ensure:
                    # Log warning if config.ensure_path fails.
                    print(f"Warning (Persona __init__): Error ensuring profile path '{self.profile_path}' via config.ensure_path: {e_ensure}", file=sys.stderr)
            else:
                # Manual fallback for directory creation if config.ensure_path is missing.
                # This might occur in minimal testing environments where config is a stub.
                try:
                    profile_dir = os.path.dirname(self.profile_path)
                    # Only attempt to create if profile_dir is a valid path (not empty, e.g., for relative filenames in CWD).
                    if profile_dir and not os.path.exists(profile_dir): 
                        os.makedirs(profile_dir, exist_ok=True)
                except Exception as e_manual_mkdir:
                     print(f"Warning (Persona __init__): Error manually ensuring profile directory for '{self.profile_path}': {e_manual_mkdir}", file=sys.stderr)
        else: 
            # Scenario: config module is not loaded (e.g., standalone testing or critical import failure).
            # Persona will use its default name and the fallback profile_path.
            # Attempt to create the directory for this fallback path.
            # Suppress print warnings if running under the test harness (indicated by _called_from_test).
            if not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                 print("Warning (Persona __init__): Config module not loaded. Persona using default name and profile path. Attempting to ensure default path.", file=sys.stderr)
            try:
                profile_dir = os.path.dirname(self.profile_path)
                if profile_dir and not os.path.exists(profile_dir):
                    os.makedirs(profile_dir, exist_ok=True)
            except Exception as e_manual_mkdir_noconf: # Catch errors during manual directory creation.
                if not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                    print(f"Warning (Persona __init__): Error manually ensuring default profile directory for '{self.profile_path}': {e_manual_mkdir_noconf}", file=sys.stderr)

        # Load persona state from the determined profile_path.
        # If loading fails (e.g., file not found, corrupted), _initialize_default_state_and_save() is called within load_state().
        self.load_state() 
        
        # Log initialization details if verbose output is enabled in config and not running under test harness.
        if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
            print(f"Persona initialized: Name='{self.name}', Mode='{self.mode}', Profile='{self.profile_path}'", file=sys.stderr)
            print(f"Initial Awareness: {self.awareness}", file=sys.stderr)

    def save_state(self):
        """
        Saves the current persona state to the profile JSON file.
        Includes name, mode, traits, and the full awareness dictionary.
        The `primary_concept_coord` in awareness is stored as (scaled_x, scaled_y, scaled_z, raw_t_intensity_0_to_1).
        """
        # Compile all relevant persona attributes into a dictionary for serialization.
        state_to_save = {
            "name": self.name,
            "mode": self.mode,
            "traits": self.traits,
            "awareness": self.awareness, # This includes primary_concept_coord with raw T-intensity.
            "last_saved": datetime.datetime.utcnow().isoformat() + "Z" # Timestamp of save.
        }

        try:
            # Ensure the directory for the profile file exists before attempting to write.
            # This is a safeguard, as __init__ should have already created it.
            profile_dir = os.path.dirname(self.profile_path)
            if profile_dir and not os.path.exists(profile_dir):
                os.makedirs(profile_dir, exist_ok=True)
                # Log directory creation if verbose and not under test.
                if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                    print(f"Info (Persona save_state): Created missing profile directory: {profile_dir}", file=sys.stderr)

            # Write the state to the JSON file with pretty printing (indent=2).
            with open(self.profile_path, 'w') as f:
                json.dump(state_to_save, f, indent=2)
            
            # Log successful save if verbose and not under test.
            if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print(f"Info (Persona save_state): Persona state saved to {self.profile_path}", file=sys.stderr)
                
        except IOError as e_io:
            if not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print(f"Error (Persona save_state): IOError saving persona state to {self.profile_path}: {e_io}", file=sys.stderr)
        except Exception as e:
            if not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print(f"Error (Persona save_state): Unexpected error saving persona state to {self.profile_path}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)


    def _initialize_default_state_and_save(self):
        """
        Resets the persona attributes to their default values and saves the state.
        This is typically called if a profile file is not found or is invalid.
        The `primary_concept_coord` defaults to None.
        """
        if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
            print(f"Info (Persona): Initializing default persona state for '{getattr(self, 'profile_path', 'N/A')}'.", file=sys.stderr)

        default_name_reset = getattr(config, 'DEFAULT_PERSONA_NAME_RESET', "Sophia_Alpha2_Default_Reset")
        self.name = getattr(config, 'PERSONA_NAME', default_name_reset) if config else default_name_reset
        self.mode = getattr(config, 'DEFAULT_PERSONA_MODE', "reflective")
        self.traits = getattr(config, 'DEFAULT_PERSONA_TRAITS', ["CuriosityDriven", "EthicallyMinded", "ResonanceAware", "Developmental"])
        
        default_awareness_metrics_reset = getattr(config, 'DEFAULT_PERSONA_AWARENESS', {
            "curiosity": 0.5, "context_stability": 0.5, "self_evolution_rate": 0.0,
            "coherence": 0.0, "active_llm_fallback": False, "primary_concept_coord": None
        })
        self.awareness = default_awareness_metrics_reset.copy()
        
        if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
            print(f"Info (Persona): Default state re-initialized. Name='{self.name}', Mode='{self.mode}'. Attempting save.", file=sys.stderr)
            
        self.save_state()

    def load_state(self):
        """
        Loads persona state from the profile JSON file.
        Handles file not found, empty file, malformed JSON, and older formats.
        `primary_concept_coord` is validated as a 4-tuple of numbers or None.
        If loading fails critically, initializes a default state.
        """
        # Check if profile_path is set; essential for loading.
        if not hasattr(self, 'profile_path') or not self.profile_path:
            if not (hasattr(sys, '_called_from_test') and sys._called_from_test): # Suppress during tests.
                print("Error (Persona load_state): Persona.profile_path is not set. Cannot load state.", file=sys.stderr)
            self._initialize_default_state_and_save() # Fallback to default if path is missing.
            return

        # Check if the profile file exists and is not empty.
        if not os.path.exists(self.profile_path) or os.path.getsize(self.profile_path) == 0:
            if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print(f"Info (Persona load_state): Profile file '{self.profile_path}' not found or empty. Initializing default state.", file=sys.stderr)
            self._initialize_default_state_and_save() # Initialize if no file or empty.
            return

        try:
            # Attempt to open and load JSON data from the profile file.
            with open(self.profile_path, 'r') as f:
                loaded_state = json.load(f)

            # Validate that the loaded data is a dictionary.
            if not isinstance(loaded_state, dict):
                if not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                    print(f"Warning (Persona load_state): Profile data in '{self.profile_path}' is not a dictionary. Initializing default state.", file=sys.stderr)
                self._initialize_default_state_and_save()
                return

            # Load basic attributes, falling back to existing defaults if keys are missing.
            self.name = loaded_state.get("name", self.name)
            self.mode = loaded_state.get("mode", self.mode)
            
            # Load 'traits', ensuring it's a list of strings.
            loaded_traits = loaded_state.get("traits", self.traits)
            if isinstance(loaded_traits, list) and all(isinstance(t, str) for t in loaded_traits):
                self.traits = loaded_traits
            else: # If traits are malformed, revert to default and log warning.
                if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                    print(f"Warning (Persona load_state): Loaded 'traits' is not a list of strings. Using default traits.", file=sys.stderr)
                self.traits = ["CuriosityDriven", "EthicallyMinded", "ResonanceAware", "Developmental"] # Default traits

            # --- Load 'awareness' dictionary with careful validation for each key ---
            loaded_awareness_data = loaded_state.get("awareness", {}) # Get awareness dict, default to empty.
            if not isinstance(loaded_awareness_data, dict): # Ensure it's a dictionary.
                if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                    print(f"Warning (Persona load_state): 'awareness' data in profile is not a dictionary. Using default awareness state.", file=sys.stderr)
                loaded_awareness_data = {} # Use empty dict to trigger defaults for all awareness fields.

            # Template for expected types of awareness metrics (excluding primary_concept_coord).
            default_awareness_template = { 
                "curiosity": float, "context_stability": float, "self_evolution_rate": float,
                "coherence": float, "active_llm_fallback": bool
            }
            # Iterate and load each awareness metric, validating type and falling back to current default if invalid.
            for key, expected_type in default_awareness_template.items():
                if key in loaded_awareness_data:
                    value_from_file = loaded_awareness_data[key]
                    try: # Attempt type conversion.
                        if expected_type == float:
                            self.awareness[key] = float(value_from_file)
                        elif expected_type == bool:
                            self.awareness[key] = bool(value_from_file)
                        # Add other types if necessary
                    except (ValueError, TypeError): # If conversion fails, log warning and keep existing default.
                        if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                            print(f"Warning (Persona load_state): Invalid type for awareness key '{key}'. Expected {expected_type}, got {type(value_from_file)}. Using current default: {self.awareness.get(key)}.", file=sys.stderr)
                # If key not in loaded_awareness_data, the default from __init__ remains.
                
            # Special handling for 'primary_concept_coord'.
            # It can be None or a 4-tuple of floats.
            loaded_coord = loaded_awareness_data.get("primary_concept_coord")
            if loaded_coord is None: # Explicitly None in file.
                self.awareness["primary_concept_coord"] = None
            elif isinstance(loaded_coord, (list, tuple)) and len(loaded_coord) == 4: # Check type and length.
                try: # Convert elements to float.
                    self.awareness["primary_concept_coord"] = tuple(float(v) for v in loaded_coord)
                except (ValueError, TypeError): # If any element is not convertible to float.
                    if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                        print(f"Warning (Persona load_state): 'primary_concept_coord' in profile contains non-numeric values: {loaded_coord}. Setting to None.", file=sys.stderr)
                    self.awareness["primary_concept_coord"] = None # Fallback to None.
            else: # Malformed (not None, not list/tuple of 4).
                if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                    print(f"Warning (Persona load_state): 'primary_concept_coord' in profile is malformed: {loaded_coord}. Expected list/tuple of 4 numbers or None. Setting to None.", file=sys.stderr)
                self.awareness["primary_concept_coord"] = None # Fallback to None.

            # Log successful load if verbose and not under test.
            if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print(f"Info (Persona load_state): Persona state loaded successfully from {self.profile_path}", file=sys.stderr)
                print(f"Loaded Awareness after validation: {self.awareness}", file=sys.stderr)

        except json.JSONDecodeError as e_json: # Handle malformed JSON file.
            if not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print(f"Error (Persona load_state): Malformed JSON in profile file '{self.profile_path}': {e_json}. Initializing default state.", file=sys.stderr)
            self._initialize_default_state_and_save() # Fallback to default.
        except Exception as e: # Catch any other unexpected errors during loading.
            if not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print(f"Error (Persona load_state): Unexpected error loading persona state from '{self.profile_path}': {e}. Initializing default state.", file=sys.stderr)
                traceback.print_exc(file=sys.stderr) # Log full traceback for debugging.
            self._initialize_default_state_and_save() # Fallback to default.

    def get_intro(self) -> str:
        """
        Generates an introductory statement for the persona.
        Includes name, mode, key awareness metrics, and focus intensity (T-value) if available.
        The T-value is the raw 0-1 intensity from primary_concept_coord[3].
        """
        intro_parts = [f"Name: {self.name}", f"Mode: {self.mode}"]
        
        curiosity = self.awareness.get("curiosity", 0.0)
        coherence = self.awareness.get("coherence", 0.0)
        intro_parts.append(f"Curiosity: {curiosity:.2f}")
        intro_parts.append(f"Coherence: {coherence:.2f}")

        primary_coord = self.awareness.get("primary_concept_coord")
        if isinstance(primary_coord, (list, tuple)) and len(primary_coord) == 4:
            try:
                raw_t_intensity = float(primary_coord[3])
                intro_parts.append(f"Focus Intensity (T): {raw_t_intensity:.2f}")
            except (ValueError, TypeError, IndexError):
                if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                    print(f"Warning (Persona get_intro): Could not parse T-intensity from primary_concept_coord[3]: {primary_coord[3]}", file=sys.stderr)
                pass 
                
        return " | ".join(intro_parts)

    def update_awareness(self, brain_awareness_metrics: dict):
        """
        Updates the persona's awareness state based on metrics from brain.think().
        Persona's self.awareness['primary_concept_coord'] will store:
        (scaled_x, scaled_y, scaled_z, raw_t_intensity_0_to_1).
        """
        # Validate input type.
        if not isinstance(brain_awareness_metrics, dict):
            if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print(f"Warning (Persona update_awareness): Invalid brain_awareness_metrics type: {type(brain_awareness_metrics)}. Expected dict. No update performed.", file=sys.stderr)
            return

        if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
            print(f"Info (Persona update_awareness): Received brain metrics for update: {brain_awareness_metrics}", file=sys.stderr)

        changed = False # Flag to track if any awareness metric actually changes value.
        # Store a JSON representation of current awareness for detailed logging if changes occur.
        original_awareness_json = json.dumps(self.awareness, sort_keys=True, default=str)


        # Define keys and their expected types for standard awareness metrics.
        metric_keys_to_process = {
            "curiosity": float, "context_stability": float, "self_evolution_rate": float,
            "coherence": float, "active_llm_fallback": bool
        }
        # Process each standard metric.
        for key, expected_type in metric_keys_to_process.items():
            if key in brain_awareness_metrics: # If the metric is present in the input.
                new_value = brain_awareness_metrics[key]
                current_value = self.awareness.get(key) # Current value in persona.
                try:
                    # Attempt to cast the new value to its expected type.
                    if expected_type == float:
                        typed_new_value = float(new_value)
                    elif expected_type == bool:
                        typed_new_value = bool(new_value)
                    else: # Should not happen with current template, but for extensibility.
                        typed_new_value = new_value 

                    # If the new typed value is different from current, update and mark as changed.
                    if current_value != typed_new_value:
                        self.awareness[key] = typed_new_value
                        changed = True
                except (ValueError, TypeError): # Handle type conversion errors.
                    if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                        print(f"Warning (Persona update_awareness): Invalid type for metric '{key}'. Expected {expected_type}, got {type(new_value)}. Retaining existing value: {current_value}", file=sys.stderr)
        
        # --- Special Handling for 'primary_concept_coord' ---
        # This coordinate is a 4-tuple: (scaled_x, scaled_y, scaled_z, raw_t_intensity_0_to_1).
        # The brain might send 'primary_concept_coord' (scaled x,y,z, and potentially scaled t)
        # AND a separate 'raw_t_intensity' (expected to be 0-1).
        # We need to construct the final awareness coordinate using scaled x,y,z and the raw_t_intensity.
        
        scaled_coord_from_brain = brain_awareness_metrics.get("primary_concept_coord")
        raw_t_from_brain = brain_awareness_metrics.get("raw_t_intensity") # Expected 0-1 intensity
        
        new_awareness_coord_to_set = self.awareness.get("primary_concept_coord") # Start with current value.
        coord_update_attempted = False # Flag to track if we even tried to update coords.

        if "primary_concept_coord" in brain_awareness_metrics: # If brain provided any coordinate data.
            coord_update_attempted = True
            if scaled_coord_from_brain is None: # If brain explicitly sets coord to None.
                if self.awareness.get("primary_concept_coord") is not None: # If it was previously set.
                    new_awareness_coord_to_set = None
                    changed = True
            elif isinstance(scaled_coord_from_brain, (list, tuple)) and len(scaled_coord_from_brain) == 4:
                # Valid structure for scaled_coord_from_brain.
                try:
                    scaled_x = float(scaled_coord_from_brain[0])
                    scaled_y = float(scaled_coord_from_brain[1])
                    scaled_z = float(scaled_coord_from_brain[2])
                    final_raw_t = None # This will be the 4th element of the stored awareness coord.

                    # Determine final_raw_t, prioritizing 'raw_t_from_brain'.
                    if raw_t_from_brain is not None:
                        try:
                            final_raw_t = float(raw_t_from_brain)
                            # Validate and clip raw_t_intensity to ensure it's within [0,1].
                            if not (0.0 <= final_raw_t <= 1.0):
                                if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                                     print(f"Warning (Persona update_awareness): Received 'raw_t_intensity' {final_raw_t} outside [0,1] range. Clipping.", file=sys.stderr)
                                final_raw_t = np.clip(final_raw_t, 0.0, 1.0) # Use np.clip for robust clamping.
                        except (ValueError, TypeError): # If raw_t_from_brain is not a valid float.
                            if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                                print(f"Warning (Persona update_awareness): Invalid type for 'raw_t_intensity': {raw_t_from_brain}. Attempting fallback for T-intensity.", file=sys.stderr)
                            final_raw_t = None # Mark as None to trigger fallback logic.

                    # Fallback logic if final_raw_t is still None (i.e., raw_t_from_brain was missing or invalid).
                    if final_raw_t is None: 
                        if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                            print(f"Info (Persona update_awareness): 'raw_t_intensity' from brain metrics missing or invalid. Attempting to use scaled_coord_from_brain[3] as raw T-intensity if it's in [0,1] range.", file=sys.stderr)
                        
                        fallback_t_candidate = float(scaled_coord_from_brain[3]) # The 4th element from brain's scaled coord.
                        # Check if this fallback candidate is a valid raw intensity (0-1).
                        if 0.0 <= fallback_t_candidate <= 1.0:
                            final_raw_t = fallback_t_candidate
                            if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                                print(f"Info (Persona update_awareness): Using scaled_coord_from_brain[3] ({fallback_t_candidate:.4f}) as raw T-intensity.", file=sys.stderr)
                        else: # Fallback candidate is also not a valid raw intensity.
                            if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                                print(f"Warning (Persona update_awareness): Fallback T-intensity from scaled_coord_from_brain[3] ({fallback_t_candidate:.4f}) is outside [0,1] range. Defaulting raw T-intensity to 0.0 for awareness state.", file=sys.stderr)
                            final_raw_t = 0.0 # Default to 0.0 if no valid source.
                    
                    # Construct the new coordinate tuple for awareness state.
                    constructed_coord = (scaled_x, scaled_y, scaled_z, final_raw_t)
                    # Check if this newly constructed coordinate is different from the current one.
                    if self.awareness.get("primary_concept_coord") != constructed_coord:
                        new_awareness_coord_to_set = constructed_coord
                        changed = True
                
                except (ValueError, TypeError, IndexError) as e: # Catch errors if scaled_coord_from_brain elements are invalid.
                    if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                        print(f"Warning (Persona update_awareness): Malformed 'primary_concept_coord' data from brain: {scaled_coord_from_brain}. Error: {e}. Retaining existing primary_concept_coord.", file=sys.stderr)
            
            else: # scaled_coord_from_brain is not None and not a valid list/tuple of 4.
                if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                    print(f"Warning (Persona update_awareness): Received 'primary_concept_coord' from brain is neither None nor a valid 4-element list/tuple: {scaled_coord_from_brain}. Retaining existing primary_concept_coord.", file=sys.stderr)
                
            # After all processing, if an update to the coordinate was attempted and resulted in a change:
            if coord_update_attempted and self.awareness.get("primary_concept_coord") != new_awareness_coord_to_set:
                self.awareness["primary_concept_coord"] = new_awareness_coord_to_set
                # 'changed' flag would have been set if new_awareness_coord_to_set differs from original.

        # If any part of the awareness state has changed, save the entire state.
        if changed:
            new_awareness_json = json.dumps(self.awareness, sort_keys=True, default=str)
            if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print(f"Info (Persona update_awareness): Awareness state has changed.", file=sys.stderr)
                print(f"  Old awareness state (JSON): {original_awareness_json}", file=sys.stderr) 
                print(f"  New awareness state (JSON): {new_awareness_json}", file=sys.stderr) 
            self.save_state() # Persist changes.
        else: # No changes detected.
            if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print(f"Info (Persona update_awareness): No change in awareness metrics after processing: {brain_awareness_metrics}", file=sys.stderr)

# --- Self-Testing Block ---
if __name__ == "__main__":
    sys._called_from_test = True # Suppress normal operational print statements

    # Ensure local Persona is used, not one from PYTHONPATH if different
    current_module_persona = Persona 

    # --- Test Utilities ---
    class TempConfigOverride:
        """
        A context manager for temporarily overriding attributes in the global `config` module
        or a mock `config` object if the global `config` is None.

        This is useful for testing different configurations without permanently altering
        the `config` object or needing to reload modules. It handles cases where the
        global `config` might be None (e.g., during standalone module testing before
        full application initialization).
        """
        def __init__(self, temp_configs_dict: dict):
            """
            Initializes the TempConfigOverride context manager.

            Args:
                temp_configs_dict (dict): A dictionary where keys are attribute names (str)
                                          to be overridden in the `config` module, and values
                                          are the temporary values for these attributes.
            """
            self.temp_configs = temp_configs_dict
            self.original_values = {} # Stores original values of overridden attributes.
            self.config_module_was_none = False # Flag if global `config` was initially None.
            self.original_global_config = None # Stores the reference to the original global `config`.

        def __enter__(self):
            """
            Sets up the temporary configuration overrides when entering the context.

            If the global `config` is None, a temporary dummy `config` object is created
            for the duration of the context. Original attribute values are stored, and
            temporary values are set.

            Returns:
                The `config` object (either the original or the temporary dummy)
                with overrides applied.
            """
            global config # Allow modification of the global 'config' variable.
            self.original_global_config = config # Store the original global config.

            current_config_module = config
            if current_config_module is None: # If global 'config' is not loaded.
                self.config_module_was_none = True
                class DummyConfig: pass # Create a simple placeholder class.
                current_config_module = DummyConfig() # Assign the dummy to be used in this context.
                config = current_config_module # Update global 'config' to this dummy for the test's duration.
            
            # Apply temporary configurations.
            for key, value in self.temp_configs.items():
                if hasattr(current_config_module, key): # If attribute exists.
                    self.original_values[key] = getattr(current_config_module, key)
                else: # Attribute does not exist, will be added temporarily.
                    self.original_values[key] = "__ATTR_NOT_SET__" # Sentinel.
                setattr(current_config_module, key, value) # Set the temporary value.
            return current_config_module # Return the modified config object.

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Restores the original configuration when exiting the context.

            Reverts changes made to the `config` object, removing temporarily added
            attributes or restoring original values. If a dummy `config` was used,
            the global `config` is restored to its original state (e.g., None).
            """
            global config # Allow modification of the global 'config' variable.
            
            current_config_module_being_restored = config # The config object that was modified.

            # Restore original attribute values.
            for key, original_value in self.original_values.items():
                if original_value == "__ATTR_NOT_SET__": # If attribute was temporarily added.
                    if hasattr(current_config_module_being_restored, key):
                        delattr(current_config_module_being_restored, key) # Remove it.
                else: # Attribute existed before, restore its original value.
                    setattr(current_config_module_being_restored, key, original_value)
            
            # Restore the original global 'config' object itself.
            # This is crucial if `config` was temporarily replaced by a DummyConfig.
            config = self.original_global_config

    TEST_PROFILE_FILENAME = "test_persona_profile.json"
    # Place test profile in the same directory as this script for simplicity
    TEST_PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), TEST_PROFILE_FILENAME)

    def delete_test_profile(profile_path: str = TEST_PROFILE_PATH):
        """
        Deletes the specified test profile file if it exists.

        Args:
            profile_path (str, optional): The path to the test profile file.
                                          Defaults to `TEST_PROFILE_PATH`.
        """
        if os.path.exists(profile_path):
            try:
                os.remove(profile_path)
            except OSError as e:
                print(f"Warning: Could not delete test profile {profile_path}: {e}", file=sys.stderr)

    test_results = {"passed": 0, "failed": 0, "details": []}

    def _run_test(test_func: callable, *args):
        """
        Wrapper function to execute a single test case.
        Manages printing test status, accumulating results, and error handling.
        Ensures test profile cleanup after each test.

        Args:
            test_func (callable): The test function to execute.
            *args: Arguments to pass to the test function.
        """
        test_name = test_func.__name__
        print(f"\n--- Running {test_name} ---")
        try:
            test_func(*args)
            test_results["passed"] += 1
            test_results["details"].append(f"[PASS] {test_name}")
            print(f"[PASS] {test_name}")
        except AssertionError as e_assert: # Catch assertion errors specifically.
            test_results["failed"] += 1
            error_info = traceback.format_exc()
            test_results["details"].append(f"[FAIL] {test_name}: AssertionError: {e_assert}\n{error_info}")
            print(f"[FAIL] {test_name}: AssertionError: {e_assert}\n{error_info}", file=sys.stderr)
        except Exception as e_general: # Catch any other exceptions.
            test_results["failed"] += 1
            error_info = traceback.format_exc()
            test_results["details"].append(f"[FAIL] {test_name}: Exception: {e_general}\n{error_info}")
            print(f"[FAIL] {test_name}: Exception: {e_general}\n{error_info}", file=sys.stderr)
        finally:
            delete_test_profile() # Cleanup after each test.


    # --- Test Scenario Implementations ---

    def test_initialization():
        """
        Tests the `Persona` class initialization, including:
        - Default attribute values.
        - Loading of persona name from config.
        - Correct structure of the initial `awareness` dictionary.
        - Creation of the persona profile file upon initialization.
        - Correct content of the `get_intro()` method for a new persona.
        """
        test_persona_name = "TestSophia"
        with TempConfigOverride({"PERSONA_PROFILE_PATH": TEST_PROFILE_PATH, 
                                 "VERBOSE_OUTPUT": False, 
                                 "PERSONA_NAME": test_persona_name,
                                 "ensure_path": lambda path: os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) and not os.path.exists(os.path.dirname(path)) else None}):
            # `delete_test_profile()` is called by `_run_test`'s finally block.
            persona = current_module_persona() # Initializes and should save a default profile.

            assert persona.name == test_persona_name, f"Name mismatch: expected {test_persona_name}, got {persona.name}"
            assert persona.mode == "reflective", "Default mode mismatch"
            assert persona.traits == ["CuriosityDriven", "EthicallyMinded", "ResonanceAware", "Developmental"], "Default traits mismatch"
            
            expected_awareness_keys = {"curiosity", "context_stability", "self_evolution_rate", "coherence", "active_llm_fallback", "primary_concept_coord"}
            assert set(persona.awareness.keys()) == expected_awareness_keys, "Awareness keys mismatch"
            assert isinstance(persona.awareness["curiosity"], float) and persona.awareness["curiosity"] == 0.5, "Default curiosity"
            assert isinstance(persona.awareness["active_llm_fallback"], bool) and not persona.awareness["active_llm_fallback"], "Default llm_fallback"
            assert persona.awareness["primary_concept_coord"] is None, "Default primary_concept_coord should be None"

            intro = persona.get_intro()
            assert "Focus Intensity (T):" not in intro, "Focus Intensity should not be in intro for new persona"
            assert test_persona_name in intro, "Persona name not in intro"

            assert os.path.exists(TEST_PROFILE_PATH), "Persona profile file was not created upon initialization"
            with open(TEST_PROFILE_PATH, 'r') as f:
                saved_state = json.load(f)
            assert saved_state["name"] == test_persona_name, "Saved name in profile mismatch"
            assert saved_state["awareness"]["primary_concept_coord"] is None, "Saved PCC in profile mismatch"

    def test_update_awareness():
        """
        Tests the `update_awareness` method with various scenarios:
        - Update with a full set of valid brain metrics.
        - Update with a partial set of metrics.
        - Handling of malformed `primary_concept_coord` from brain metrics.
        - Handling of `primary_concept_coord` being explicitly set to None.
        - Fallback logic for `raw_t_intensity` if not directly provided but inferable
          from `primary_concept_coord[3]` (and validation of that inference).
        """
        with TempConfigOverride({"PERSONA_PROFILE_PATH": TEST_PROFILE_PATH, 
                                 "VERBOSE_OUTPUT": False, 
                                 "ensure_path": lambda path: os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) and not os.path.exists(os.path.dirname(path)) else None}):
            # `delete_test_profile()` is called by `_run_test`'s finally block.
            persona = current_module_persona()

            # Scenario 1: Valid Full Metrics from brain.
            brain_metrics_valid = { 
                "curiosity": 0.8, "coherence": 0.75, "active_llm_fallback": True, 
                "primary_concept_coord": (1.0, 2.0, 3.0, 75.0), # Brain might send scaled T-coord as 4th element
                "raw_t_intensity": 0.75, # Explicit raw T-intensity (0-1) is preferred.
                "context_stability": 0.85, "self_evolution_rate": 0.05 
            }
            persona.update_awareness(brain_metrics_valid)
            # Persona stores (scaled_x, scaled_y, scaled_z, raw_t_intensity_0_to_1)
            assert persona.awareness["primary_concept_coord"] == (1.0, 2.0, 3.0, 0.75), f"PCC S1: {persona.awareness['primary_concept_coord']}"
            assert "Focus Intensity (T): 0.75" in persona.get_intro(), f"Intro S1: {persona.get_intro()}"
            assert persona.awareness["curiosity"] == 0.8, "S1 Curiosity"
            assert persona.awareness["active_llm_fallback"] is True, "S1 LLM Fallback"

            # Scenario 2: Partial Metrics update.
            persona.update_awareness({"curiosity": 0.9, "self_evolution_rate": 0.1})
            assert persona.awareness["curiosity"] == 0.9, "S2 Curiosity"
            assert persona.awareness["self_evolution_rate"] == 0.1, "S2 SER"
            # Previous valid primary_concept_coord should persist if not updated.
            assert persona.awareness["primary_concept_coord"] == (1.0, 2.0, 3.0, 0.75), "PCC S2 (should remain from S1)"

            # Scenario 3: Malformed primary_concept_coord from Brain (e.g., wrong type).
            original_pcc_s2 = persona.awareness['primary_concept_coord'] # Store before malformed update.
            persona.update_awareness({"primary_concept_coord": "not-a-tuple", "raw_t_intensity": "bad-type"})
            assert persona.awareness['primary_concept_coord'] == original_pcc_s2, "PCC S3 (should remain from S2 due to malformed input)"

            # Scenario 4: primary_concept_coord is explicitly None from Brain.
            persona.update_awareness({"primary_concept_coord": None, "raw_t_intensity": None})
            assert persona.awareness['primary_concept_coord'] is None, "PCC S4 (should be None as per brain metrics)"
            assert "Focus Intensity (T):" not in persona.get_intro(), "Intro S4 (no T-intensity as PCC is None)"
            
            # Scenario 5: Missing 'raw_t_intensity', but brain's primary_concept_coord[3] is a valid raw T-intensity (0-1 range).
            persona.update_awareness({
                "primary_concept_coord": (4.0, 5.0, 6.0, 0.88), # Assume brain's 4th element is raw T here.
                # "raw_t_intensity": is missing from brain_metrics.
            })
            assert persona.awareness["primary_concept_coord"] == (4.0, 5.0, 6.0, 0.88), f"PCC S5: {persona.awareness['primary_concept_coord']}"
            assert "Focus Intensity (T): 0.88" in persona.get_intro(), f"Intro S5: {persona.get_intro()}"

            # Scenario 6: Missing 'raw_t_intensity', AND brain's primary_concept_coord[3] is NOT a valid raw T-intensity (e.g., still scaled).
            # Persona should default raw T-intensity to 0.0 in this ambiguous case.
            persona.update_awareness({
                "primary_concept_coord": (7.0, 8.0, 9.0, 88.0), # Assume brain's 4th element is scaled (not 0-1).
                 # "raw_t_intensity": is missing.
            })
            assert persona.awareness["primary_concept_coord"] == (7.0, 8.0, 9.0, 0.0), f"PCC S6: {persona.awareness['primary_concept_coord']}"
            assert "Focus Intensity (T): 0.00" in persona.get_intro(), f"Intro S6: {persona.get_intro()}"


    def test_save_load_cycle():
        """
        Tests the full cycle of saving a persona's state and then loading it
        into a new persona instance. Verifies that all attributes, especially
        the `awareness` dictionary and `primary_concept_coord` (with raw T-intensity),
        are correctly persisted and restored.
        """
        with TempConfigOverride({"PERSONA_PROFILE_PATH": TEST_PROFILE_PATH, 
                                 "VERBOSE_OUTPUT": False,
                                 "ensure_path": lambda path: os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) and not os.path.exists(os.path.dirname(path)) else None}):
            # `delete_test_profile()` is called by `_run_test`'s finally block.
            persona1 = current_module_persona()
            
            metrics_to_save = { 
                "curiosity": 0.66, 
                "primary_concept_coord": (10.0, 20.0, 30.0, 99.0), # Example scaled coord from brain
                "raw_t_intensity": 0.99 # Explicit raw T-intensity
            }
            persona1.update_awareness(metrics_to_save) # This will also call save_state()
            persona1_awareness_snapshot = persona1.awareness.copy() 
            
            # Create a new persona instance, which should load from the profile saved by persona1.
            persona2 = current_module_persona()

            assert persona2.awareness == persona1_awareness_snapshot, \
                f"Awareness mismatch after load: P1: {persona1_awareness_snapshot}, P2: {persona2.awareness}"
            # Specifically check the reconstructed primary_concept_coord in persona2.
            expected_pcc_after_load = (10.0, 20.0, 30.0, 0.99) # Scaled x,y,z + raw T
            assert persona2.awareness['primary_concept_coord'] == expected_pcc_after_load, \
                f"PCC mismatch after load. Expected: {expected_pcc_after_load}, Got: {persona2.awareness['primary_concept_coord']}"
            assert "Focus Intensity (T): 0.99" in persona2.get_intro(), f"Intro mismatch after load: {persona2.get_intro()}"

    def test_load_old_format_profile():
        """
        Tests loading a persona profile that might be in an older format
        (e.g., missing some newer `awareness` fields like `primary_concept_coord`).
        Ensures that missing fields are initialized to their defaults.
        """
        with TempConfigOverride({"PERSONA_PROFILE_PATH": TEST_PROFILE_PATH,
                                 "VERBOSE_OUTPUT": False,
                                 "ensure_path": lambda path: os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) and not os.path.exists(os.path.dirname(path)) else None}):
            # Manually write an old format profile.
            old_profile_data = {
                "name": "OldSophia", "mode": "archaic", "traits": ["Classic"], 
                "awareness": {"curiosity": 0.1, "coherence": 0.2} # Missing primary_concept_coord, etc.
            }
            with open(TEST_PROFILE_PATH, 'w') as f:
                json.dump(old_profile_data, f)
            
            persona = current_module_persona() # Load from the manually created old profile.
            
            assert persona.name == "OldSophia", "Name from old profile"
            assert persona.mode == "archaic", "Mode from old profile"
            assert persona.traits == ["Classic"], "Traits from old profile"
            assert persona.awareness['curiosity'] == 0.1, "Curiosity from old profile"
            assert persona.awareness['coherence'] == 0.2, "Coherence from old profile"
            # Check that fields not present in the old profile are set to defaults.
            assert persona.awareness['primary_concept_coord'] is None, "PCC from old profile (should be default None)"
            assert persona.awareness['context_stability'] == 0.5, "Context stability (should be default)"
            assert "Focus Intensity (T):" not in persona.get_intro(), "Intro from old profile (should not have T-intensity)"

    def test_load_malformed_profile():
        """
        Tests loading various types of malformed persona profiles:
        - Empty profile file.
        - File with invalid JSON.
        - File with valid JSON but malformed `primary_concept_coord` (e.g., wrong length, non-numeric elements).
        Ensures that the persona initializes to a default state in these cases.
        """
        with TempConfigOverride({"PERSONA_PROFILE_PATH": TEST_PROFILE_PATH,
                                 "VERBOSE_OUTPUT": False,
                                 "ensure_path": lambda path: os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) and not os.path.exists(os.path.dirname(path)) else None}):
            default_persona_name_after_reset = "Sophia_Alpha2_Default_Reset" 
            
            # Scenario 1: Empty File.
            with open(TEST_PROFILE_PATH, 'w') as f: f.write("") 
            persona_empty = current_module_persona()
            assert persona_empty.name == default_persona_name_after_reset, f"Empty file: Name {persona_empty.name}"
            assert persona_empty.awareness["primary_concept_coord"] is None, "Empty file: PCC should be None"

            # Scenario 2: Invalid JSON.
            delete_test_profile() # Clean before next sub-test.
            with open(TEST_PROFILE_PATH, 'w') as f: f.write("{invalid_json: ")
            persona_invalid_json = current_module_persona()
            assert persona_invalid_json.name == default_persona_name_after_reset, f"Invalid JSON: Name {persona_invalid_json.name}"
            assert persona_invalid_json.awareness["primary_concept_coord"] is None, "Invalid JSON: PCC should be None"

            # Scenario 3: Malformed primary_concept_coord (wrong length).
            delete_test_profile()
            malformed_pcc_profile_len = {"name": "TestPCC_Len", "awareness": {"primary_concept_coord": [1, 2, 3]}}
            with open(TEST_PROFILE_PATH, 'w') as f: json.dump(malformed_pcc_profile_len, f)
            persona_malformed_pcc_len = current_module_persona()
            assert persona_malformed_pcc_len.name == "TestPCC_Len", f"Malformed PCC (len): Name {persona_malformed_pcc_len.name}"
            assert persona_malformed_pcc_len.awareness["primary_concept_coord"] is None, \
                f"Malformed PCC (len): PCC should be None, got {persona_malformed_pcc_len.awareness['primary_concept_coord']}"

            # Scenario 4: Malformed primary_concept_coord (non-numeric elements).
            delete_test_profile()
            malformed_pcc_profile_type = {"name": "TestPCC_Type", "awareness": {"primary_concept_coord": [1, 2, 3, "not-a-number"]}}
            with open(TEST_PROFILE_PATH, 'w') as f: json.dump(malformed_pcc_profile_type, f)
            persona_malformed_pcc_type = current_module_persona()
            assert persona_malformed_pcc_type.name == "TestPCC_Type", f"Malformed PCC (type): Name {persona_malformed_pcc_type.name}"
            assert persona_malformed_pcc_type.awareness["primary_concept_coord"] is None, \
                f"Malformed PCC (type): PCC should be None, got {persona_malformed_pcc_type.awareness['primary_concept_coord']}"


    # --- Test Runner ---
    print("Starting Persona Class Self-Tests...")
    
    _run_test(test_initialization)
    _run_test(test_update_awareness)
    _run_test(test_save_load_cycle)
    _run_test(test_load_old_format_profile)
    _run_test(test_load_malformed_profile)

    print("\n--- Test Summary ---")
    for detail in test_results["details"]:
        print(detail.splitlines()[0]) # Print only the main pass/fail line for summary
    
    print(f"\nTotal Passed: {test_results['passed']}")
    print(f"Total Failed: {test_results['failed']}")

    # Clean up one last time
    delete_test_profile()
    del sys._called_from_test # Clean up the helper attribute

    if test_results["failed"] > 0:
        print("\nSOME TESTS FAILED.")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED.")
        sys.exit(0)
```
