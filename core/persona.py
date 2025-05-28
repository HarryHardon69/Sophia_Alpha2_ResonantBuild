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
        # Default attributes
        default_name = "Sophia_Alpha2_Default"
        self.name = default_name
        self.mode = "reflective"  # e.g., reflective, learning, active_problem_solving
        self.traits = ["CuriosityDriven", "EthicallyMinded", "ResonanceAware", "Developmental"]
        
        # Awareness metrics - initialized with sensible defaults
        # primary_concept_coord is (scaled_x, scaled_y, scaled_z, raw_t_intensity_0_to_1)
        self.awareness = {
            "curiosity": 0.5,
            "context_stability": 0.5,
            "self_evolution_rate": 0.0,
            "coherence": 0.0,
            "active_llm_fallback": False,
            "primary_concept_coord": None
        }

        default_profile_filename = "persona_profile_default.json"
        # Fallback profile path determination
        try:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            # Assumes core/ is one level down from project root which contains data/private
            fallback_data_dir = os.path.join(os.path.dirname(module_dir), 'data', 'private')
        except NameError: # __file__ not defined
            fallback_data_dir = os.path.join('.', 'data', 'private') # CWD/data/private

        # Initialize profile_path with a default. It will be overridden by config if available.
        self.profile_path = os.path.join(fallback_data_dir, default_profile_filename)

        if config:
            self.name = getattr(config, 'PERSONA_NAME', default_name)
            # Use config.PERSONA_PROFILE_PATH for the profile path
            self.profile_path = getattr(config, 'PERSONA_PROFILE_PATH', self.profile_path)
            
            if hasattr(config, 'ensure_path'):
                try:
                    config.ensure_path(self.profile_path) # Ensures the directory for profile_path exists
                except Exception as e_ensure:
                    print(f"Warning (Persona __init__): Error ensuring profile path '{self.profile_path}' via config.ensure_path: {e_ensure}", file=sys.stderr)
            else:
                # Manual ensure_path if config.ensure_path is not available (e.g. standalone testing without full config object)
                try:
                    profile_dir = os.path.dirname(self.profile_path)
                    if profile_dir and not os.path.exists(profile_dir): # Only create if profile_dir is not empty (e.g. not current dir)
                        os.makedirs(profile_dir, exist_ok=True)
                except Exception as e_manual_mkdir:
                     print(f"Warning (Persona __init__): Error manually ensuring profile directory for '{self.profile_path}': {e_manual_mkdir}", file=sys.stderr)
        else: 
            # When config is None, we are likely in a standalone test.
            # Ensure the profile directory exists for the default path.
            if not (hasattr(sys, '_called_from_test') and sys._called_from_test): # Avoid print during automated tests
                 print("Warning (Persona __init__): Config module not loaded. Persona will use default name and profile path. Attempting to ensure default path.", file=sys.stderr)
            try:
                profile_dir = os.path.dirname(self.profile_path)
                if profile_dir and not os.path.exists(profile_dir):
                    os.makedirs(profile_dir, exist_ok=True)
            except Exception as e_manual_mkdir_noconf:
                if not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                    print(f"Warning (Persona __init__): Error manually ensuring default profile directory for '{self.profile_path}': {e_manual_mkdir_noconf}", file=sys.stderr)

        self.load_state() 
        
        if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
            print(f"Persona initialized: Name='{self.name}', Mode='{self.mode}', Profile='{self.profile_path}'", file=sys.stderr)
            print(f"Initial Awareness: {self.awareness}", file=sys.stderr)

    def save_state(self):
        """
        Saves the current persona state to the profile JSON file.
        Includes name, mode, traits, and the full awareness dictionary.
        The `primary_concept_coord` in awareness is stored as (scaled_x, scaled_y, scaled_z, raw_t_intensity_0_to_1).
        """
        state_to_save = {
            "name": self.name,
            "mode": self.mode,
            "traits": self.traits,
            "awareness": self.awareness,
            "last_saved": datetime.datetime.utcnow().isoformat() + "Z"
        }

        try:
            profile_dir = os.path.dirname(self.profile_path)
            if profile_dir and not os.path.exists(profile_dir):
                os.makedirs(profile_dir, exist_ok=True)
                if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                    print(f"Info (Persona save_state): Created missing profile directory: {profile_dir}", file=sys.stderr)

            with open(self.profile_path, 'w') as f:
                json.dump(state_to_save, f, indent=2)
            
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

        default_name = "Sophia_Alpha2_Default_Reset"
        self.name = getattr(config, 'PERSONA_NAME', default_name) if config else default_name
        self.mode = "reflective"
        self.traits = ["CuriosityDriven", "EthicallyMinded", "ResonanceAware", "Developmental"]
        
        self.awareness = { # Default awareness state
            "curiosity": 0.5,
            "context_stability": 0.5,
            "self_evolution_rate": 0.0,
            "coherence": 0.0,
            "active_llm_fallback": False,
            "primary_concept_coord": None 
        }
        
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
        if not hasattr(self, 'profile_path') or not self.profile_path:
            if not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print("Error (Persona load_state): profile_path not set. Cannot load.", file=sys.stderr)
            self._initialize_default_state_and_save()
            return

        if not os.path.exists(self.profile_path) or os.path.getsize(self.profile_path) == 0:
            if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print(f"Info (Persona load_state): Profile file '{self.profile_path}' not found or empty. Initializing default state.", file=sys.stderr)
            self._initialize_default_state_and_save()
            return

        try:
            with open(self.profile_path, 'r') as f:
                loaded_state = json.load(f)

            if not isinstance(loaded_state, dict):
                if not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                    print(f"Warning (Persona load_state): Profile data in '{self.profile_path}' is not a dictionary. Initializing default state.", file=sys.stderr)
                self._initialize_default_state_and_save()
                return

            self.name = loaded_state.get("name", self.name)
            self.mode = loaded_state.get("mode", self.mode)
            
            loaded_traits = loaded_state.get("traits", self.traits)
            if isinstance(loaded_traits, list) and all(isinstance(t, str) for t in loaded_traits):
                self.traits = loaded_traits
            else:
                if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                    print(f"Warning (Persona load_state): Loaded 'traits' is not a list of strings. Using default traits.", file=sys.stderr)
                self.traits = ["CuriosityDriven", "EthicallyMinded", "ResonanceAware", "Developmental"] # Default

            loaded_awareness_data = loaded_state.get("awareness", {})
            if not isinstance(loaded_awareness_data, dict):
                if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                    print(f"Warning (Persona load_state): 'awareness' data in profile is not a dictionary. Using default awareness.", file=sys.stderr)
                loaded_awareness_data = {} 

            default_awareness_template = { 
                "curiosity": float, "context_stability": float, "self_evolution_rate": float,
                "coherence": float, "active_llm_fallback": bool
            }

            for key, expected_type in default_awareness_template.items():
                if key in loaded_awareness_data:
                    value = loaded_awareness_data[key]
                    try:
                        if expected_type == float:
                            self.awareness[key] = float(value)
                        elif expected_type == bool:
                            self.awareness[key] = bool(value)
                    except (ValueError, TypeError):
                        if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                            print(f"Warning (Persona load_state): Invalid type for awareness key '{key}'. Expected {expected_type}, got {type(value)}. Using default from __init__ ({self.awareness.get(key)}).", file=sys.stderr)
                
            loaded_coord = loaded_awareness_data.get("primary_concept_coord")
            if loaded_coord is None:
                self.awareness["primary_concept_coord"] = None
            elif isinstance(loaded_coord, (list, tuple)) and len(loaded_coord) == 4:
                try:
                    self.awareness["primary_concept_coord"] = tuple(float(v) for v in loaded_coord)
                except (ValueError, TypeError):
                    if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                        print(f"Warning (Persona load_state): 'primary_concept_coord' in profile has non-numeric values: {loaded_coord}. Setting to None.", file=sys.stderr)
                    self.awareness["primary_concept_coord"] = None 
            else: 
                if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                    print(f"Warning (Persona load_state): 'primary_concept_coord' in profile is malformed: {loaded_coord}. Expected list/tuple of 4 or None. Setting to None.", file=sys.stderr)
                self.awareness["primary_concept_coord"] = None 

            if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print(f"Info (Persona load_state): Persona state loaded successfully from {self.profile_path}", file=sys.stderr)
                print(f"Loaded Awareness: {self.awareness}", file=sys.stderr)

        except json.JSONDecodeError as e_json:
            if not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print(f"Error (Persona load_state): Malformed JSON in profile file '{self.profile_path}': {e_json}. Initializing default state.", file=sys.stderr)
            self._initialize_default_state_and_save()
        except Exception as e:
            if not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print(f"Error (Persona load_state): Unexpected error loading persona state from '{self.profile_path}': {e}. Initializing default state.", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            self._initialize_default_state_and_save()

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
        if not isinstance(brain_awareness_metrics, dict):
            if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print(f"Warning (Persona update_awareness): Invalid brain_awareness_metrics type: {type(brain_awareness_metrics)}. No update.", file=sys.stderr)
            return

        if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
            print(f"Info (Persona update_awareness): Received brain metrics: {brain_awareness_metrics}", file=sys.stderr)

        changed = False
        original_awareness_json = json.dumps(self.awareness, sort_keys=True)

        metric_keys_to_process = {
            "curiosity": float, "context_stability": float, "self_evolution_rate": float,
            "coherence": float, "active_llm_fallback": bool
        }
        for key, expected_type in metric_keys_to_process.items():
            if key in brain_awareness_metrics:
                new_value = brain_awareness_metrics[key]
                current_value = self.awareness.get(key)
                try:
                    if expected_type == float:
                        typed_new_value = float(new_value)
                    elif expected_type == bool:
                        typed_new_value = bool(new_value)
                    else: 
                        typed_new_value = new_value 

                    if current_value != typed_new_value:
                        self.awareness[key] = typed_new_value
                        changed = True
                except (ValueError, TypeError):
                    if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                        print(f"Warning (Persona update_awareness): Invalid type for metric '{key}'. Expected {expected_type}, got {type(new_value)}. Retaining existing value: {current_value}", file=sys.stderr)
        
        scaled_coord_from_brain = brain_awareness_metrics.get("primary_concept_coord")
        raw_t_from_brain = brain_awareness_metrics.get("raw_t_intensity")
        
        new_awareness_coord_to_set = self.awareness.get("primary_concept_coord") 
        coord_update_attempted = False

        if "primary_concept_coord" in brain_awareness_metrics: 
            coord_update_attempted = True
            if scaled_coord_from_brain is None:
                if self.awareness.get("primary_concept_coord") is not None:
                    new_awareness_coord_to_set = None
                    changed = True
            elif isinstance(scaled_coord_from_brain, (list, tuple)) and len(scaled_coord_from_brain) == 4:
                try:
                    scaled_x = float(scaled_coord_from_brain[0])
                    scaled_y = float(scaled_coord_from_brain[1])
                    scaled_z = float(scaled_coord_from_brain[2])
                    final_raw_t = None

                    if raw_t_from_brain is not None:
                        try:
                            final_raw_t = float(raw_t_from_brain)
                            if not (0.0 <= final_raw_t <= 1.0):
                                if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                                     print(f"Warning (Persona update_awareness): Received 'raw_t_intensity' {final_raw_t} outside 0-1 range. Clipping.", file=sys.stderr)
                                final_raw_t = max(0.0, min(1.0, final_raw_t))
                        except (ValueError, TypeError):
                            if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                                print(f"Warning (Persona update_awareness): Invalid type for 'raw_t_intensity': {raw_t_from_brain}. Will attempt fallback.", file=sys.stderr)
                            final_raw_t = None 

                    if final_raw_t is None: 
                        if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                            print(f"Warning (Persona update_awareness): 'raw_t_intensity' missing or invalid in brain metrics. Falling back to scaled_coord[3] if it's in 0-1 range.", file=sys.stderr)
                        
                        fallback_t = float(scaled_coord_from_brain[3]) 
                        if 0.0 <= fallback_t <= 1.0:
                            final_raw_t = fallback_t
                            if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                                print(f"Info (Persona update_awareness): Using scaled_coord[3] ({fallback_t:.4f}) as raw T-intensity.", file=sys.stderr)
                        else:
                            if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                                print(f"Warning (Persona update_awareness): Fallback scaled_coord[3] ({fallback_t:.4f}) is outside 0-1 range. Cannot determine raw T-intensity. Setting T to 0.0 in awareness coord.", file=sys.stderr)
                            final_raw_t = 0.0 
                    constructed_coord = (scaled_x, scaled_y, scaled_z, final_raw_t)
                    if self.awareness.get("primary_concept_coord") != constructed_coord:
                        new_awareness_coord_to_set = constructed_coord
                        changed = True
                
                except (ValueError, TypeError, IndexError) as e:
                    if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                        print(f"Warning (Persona update_awareness): Malformed 'primary_concept_coord' data: {scaled_coord_from_brain}. Error: {e}. Retaining existing.", file=sys.stderr)
            
            else: 
                if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                    print(f"Warning (Persona update_awareness): Received 'primary_concept_coord' is not None and not a list/tuple of 4: {scaled_coord_from_brain}. Retaining existing.", file=sys.stderr)
                
            if coord_update_attempted: 
                 if self.awareness.get("primary_concept_coord") != new_awareness_coord_to_set :
                    self.awareness["primary_concept_coord"] = new_awareness_coord_to_set
        if changed:
            new_awareness_json = json.dumps(self.awareness, sort_keys=True)
            if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print(f"Info (Persona update_awareness): Awareness changed.", file=sys.stderr)
                print(f"  Old awareness subset: {original_awareness_json}", file=sys.stderr) 
                print(f"  New awareness state: {new_awareness_json}", file=sys.stderr) 
            self.save_state()
        else:
            if config and getattr(config, 'VERBOSE_OUTPUT', False) and not (hasattr(sys, '_called_from_test') and sys._called_from_test):
                print(f"Info (Persona update_awareness): No change in awareness metrics after processing: {brain_awareness_metrics}", file=sys.stderr)

# --- Self-Testing Block ---
if __name__ == "__main__":
    sys._called_from_test = True # Suppress normal operational print statements

    # Ensure local Persona is used, not one from PYTHONPATH if different
    current_module_persona = Persona 

    # --- Test Utilities ---
    class TempConfigOverride:
        def __init__(self, temp_configs_dict):
            self.temp_configs = temp_configs_dict
            self.original_values = {}
            self.config_module = config # Use the 'config' resolved at module level

        def __enter__(self):
            if not self.config_module:
                # Create a dummy config object if it's None (e.g., standalone without ..config)
                class DummyConfig: pass
                self.config_module = DummyConfig()
                # If 'config' was None globally, set it to this dummy for the duration of the test
                # This is tricky because 'config' is a global in Persona's scope.
                # For simplicity, we'll rely on the tests passing this dummy config module
                # to Persona instances if needed, or Persona handling config=None.
                # The tests will primarily set attributes on this temporary config_module.
                global config
                self.original_global_config = config
                config = self.config_module
                # print("TempConfigOverride: Created dummy config module.", file=sys.stderr)


            for key, value in self.temp_configs.items():
                if hasattr(self.config_module, key):
                    self.original_values[key] = getattr(self.config_module, key)
                else:
                    self.original_values[key] = "__ATTR_NOT_SET__" # Sentinel for attributes that didn't exist
                setattr(self.config_module, key, value)
            return self.config_module

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not self.config_module:
                return

            for key, original_value in self.original_values.items():
                if original_value == "__ATTR_NOT_SET__":
                    if hasattr(self.config_module, key):
                        delattr(self.config_module, key)
                else:
                    setattr(self.config_module, key, original_value)
            
            if hasattr(self, 'original_global_config'): # Restore original global config if it was changed
                global config
                config = self.original_global_config

    TEST_PROFILE_FILENAME = "test_persona_profile.json"
    # Place test profile in the same directory as this script for simplicity
    TEST_PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), TEST_PROFILE_FILENAME)

    def delete_test_profile(profile_path=TEST_PROFILE_PATH):
        if os.path.exists(profile_path):
            try:
                os.remove(profile_path)
            except OSError as e:
                print(f"Warning: Could not delete test profile {profile_path}: {e}", file=sys.stderr)

    test_results = {"passed": 0, "failed": 0, "details": []}

    def _run_test(test_func, *args):
        test_name = test_func.__name__
        print(f"\n--- Running {test_name} ---")
        try:
            test_func(*args)
            test_results["passed"] += 1
            test_results["details"].append(f"[PASS] {test_name}")
            print(f"[PASS] {test_name}")
        except AssertionError as e:
            test_results["failed"] += 1
            error_info = traceback.format_exc()
            test_results["details"].append(f"[FAIL] {test_name}: {e}\n{error_info}")
            print(f"[FAIL] {test_name}: Assertion Error: {e}\n{error_info}", file=sys.stderr)
        except Exception as e:
            test_results["failed"] += 1
            error_info = traceback.format_exc()
            test_results["details"].append(f"[FAIL] {test_name}: Exception: {e}\n{error_info}")
            print(f"[FAIL] {test_name}: Exception: {e}\n{error_info}", file=sys.stderr)
        finally:
            # Clean up profile after each test to ensure independence,
            # unless a test specifically needs to persist it for a subsequent check by another test (not ideal).
            # For these tests, cleanup is generally good.
            delete_test_profile()


    # --- Test Scenario Implementations ---

    def test_initialization():
        test_persona_name = "TestSophia"
        with TempConfigOverride({"PERSONA_PROFILE_PATH": TEST_PROFILE_PATH, 
                                 "VERBOSE_OUTPUT": False, 
                                 "PERSONA_NAME": test_persona_name,
                                 "ensure_path": lambda path: os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None}):
            delete_test_profile()
            persona = current_module_persona()

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

            assert os.path.exists(TEST_PROFILE_PATH), "Persona profile file was not created"
            with open(TEST_PROFILE_PATH, 'r') as f:
                saved_state = json.load(f)
            assert saved_state["name"] == test_persona_name
            assert saved_state["awareness"]["primary_concept_coord"] is None

    def test_update_awareness():
        with TempConfigOverride({"PERSONA_PROFILE_PATH": TEST_PROFILE_PATH, 
                                 "VERBOSE_OUTPUT": False, # Set to True for debugging this test
                                 "ensure_path": lambda path: os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None}):
            delete_test_profile()
            persona = current_module_persona()

            # Scenario 1: Valid Full Metrics
            brain_metrics_valid = { 
                "curiosity": 0.8, "coherence": 0.75, "active_llm_fallback": True, 
                "primary_concept_coord": (1.0, 2.0, 3.0, 75.0), # Brain sends scaled T-coord
                "raw_t_intensity": 0.75, 
                "context_stability": 0.85, "self_evolution_rate": 0.05 
            }
            persona.update_awareness(brain_metrics_valid)
            assert persona.awareness["primary_concept_coord"] == (1.0, 2.0, 3.0, 0.75), f"PCC S1: {persona.awareness['primary_concept_coord']}"
            assert "Focus Intensity (T): 0.75" in persona.get_intro(), f"Intro S1: {persona.get_intro()}"
            assert persona.awareness["curiosity"] == 0.8
            assert persona.awareness["active_llm_fallback"] is True

            # Scenario 2: Partial Metrics
            persona.update_awareness({"curiosity": 0.9, "self_evolution_rate": 0.1})
            assert persona.awareness["curiosity"] == 0.9, "Curiosity S2"
            assert persona.awareness["self_evolution_rate"] == 0.1, "SER S2"
            assert persona.awareness["primary_concept_coord"] == (1.0, 2.0, 3.0, 0.75), "PCC S2 (should remain from S1)"

            # Scenario 3: Malformed primary_concept_coord from Brain
            original_pcc = persona.awareness['primary_concept_coord']
            persona.update_awareness({"primary_concept_coord": "not-a-tuple", "raw_t_intensity": "bad-type"})
            assert persona.awareness['primary_concept_coord'] == original_pcc, "PCC S3 (should remain from S2)"

            # Scenario 4: primary_concept_coord is None from Brain
            persona.update_awareness({"primary_concept_coord": None, "raw_t_intensity": None})
            assert persona.awareness['primary_concept_coord'] is None, "PCC S4 (should be None)"
            assert "Focus Intensity (T):" not in persona.get_intro(), "Intro S4 (no T-intensity)"
            
            # Scenario 5: Missing raw_t_intensity, but scaled_coord[3] is valid 0-1 (heuristic fallback)
            persona.update_awareness({
                "primary_concept_coord": (4.0, 5.0, 6.0, 0.88), # scaled_coord[3] is 0.88
                # "raw_t_intensity": missing
            })
            assert persona.awareness["primary_concept_coord"] == (4.0, 5.0, 6.0, 0.88), f"PCC S5: {persona.awareness['primary_concept_coord']}"
            assert "Focus Intensity (T): 0.88" in persona.get_intro(), f"Intro S5: {persona.get_intro()}"

            # Scenario 6: Missing raw_t_intensity, and scaled_coord[3] is NOT valid 0-1 (heuristic fallback fails, T defaults to 0.0)
            persona.update_awareness({
                "primary_concept_coord": (7.0, 8.0, 9.0, 88.0), # scaled_coord[3] is 88.0 (not 0-1)
                 # "raw_t_intensity": missing
            })
            assert persona.awareness["primary_concept_coord"] == (7.0, 8.0, 9.0, 0.0), f"PCC S6: {persona.awareness['primary_concept_coord']}"
            assert "Focus Intensity (T): 0.00" in persona.get_intro(), f"Intro S6: {persona.get_intro()}"


    def test_save_load_cycle():
        with TempConfigOverride({"PERSONA_PROFILE_PATH": TEST_PROFILE_PATH, 
                                 "VERBOSE_OUTPUT": False,
                                 "ensure_path": lambda path: os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None}):
            delete_test_profile()
            persona1 = current_module_persona()
            
            metrics_to_save = { 
                "curiosity": 0.66, 
                "primary_concept_coord": (10.0, 20.0, 30.0, 99.0), # Scaled T-coord from brain
                "raw_t_intensity": 0.99 # Raw T from brain
            }
            persona1.update_awareness(metrics_to_save)
            persona1_awareness = persona1.awareness.copy() # Shallow copy is fine for this flat dict
            
            # persona1.save_state() is called by update_awareness if changed.
            # Create a new persona instance, which should load from the saved profile.
            persona2 = current_module_persona()

            assert persona2.awareness == persona1_awareness, f"Awareness mismatch after load: P1: {persona1_awareness}, P2: {persona2.awareness}"
            assert persona2.awareness['primary_concept_coord'] == (10.0, 20.0, 30.0, 0.99), f"PCC mismatch after load: {persona2.awareness['primary_concept_coord']}"
            assert "Focus Intensity (T): 0.99" in persona2.get_intro(), f"Intro mismatch after load: {persona2.get_intro()}"

    def test_load_old_format_profile():
        with TempConfigOverride({"PERSONA_PROFILE_PATH": TEST_PROFILE_PATH,
                                 "VERBOSE_OUTPUT": False,
                                 "ensure_path": lambda path: os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None}):
            delete_test_profile()
            # Manually write an old format profile (missing some awareness fields, esp. primary_concept_coord)
            old_profile_data = {"name": "OldSophia", "mode": "archaic", "traits": ["Classic"], "awareness": {"curiosity": 0.1, "coherence": 0.2}}
            with open(TEST_PROFILE_PATH, 'w') as f:
                json.dump(old_profile_data, f)
            
            persona = current_module_persona()
            
            assert persona.name == "OldSophia", "Name from old profile"
            assert persona.mode == "archaic", "Mode from old profile"
            assert persona.traits == ["Classic"], "Traits from old profile"
            assert persona.awareness['curiosity'] == 0.1, "Curiosity from old profile"
            assert persona.awareness['coherence'] == 0.2, "Coherence from old profile"
            # Fields not in old profile should have defaults
            assert persona.awareness['primary_concept_coord'] is None, "PCC from old profile (should be None)"
            assert persona.awareness['context_stability'] == 0.5, "Context stability default" # Assuming 0.5 is default
            assert "Focus Intensity (T):" not in persona.get_intro(), "Intro from old profile"

    def test_load_malformed_profile():
        with TempConfigOverride({"PERSONA_PROFILE_PATH": TEST_PROFILE_PATH,
                                 "VERBOSE_OUTPUT": False,
                                 "ensure_path": lambda path: os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None}):
            default_persona_name = "Sophia_Alpha2_Default_Reset" # Name after _initialize_default_state_and_save
            
            # Scenario 1: Empty File
            delete_test_profile()
            with open(TEST_PROFILE_PATH, 'w') as f:
                f.write("") # Create empty file
            persona_empty = current_module_persona()
            assert persona_empty.name == default_persona_name, f"Empty file: Name {persona_empty.name}"
            assert persona_empty.awareness["primary_concept_coord"] is None, "Empty file: PCC"

            # Scenario 2: Invalid JSON
            delete_test_profile()
            with open(TEST_PROFILE_PATH, 'w') as f:
                f.write("{invalid_json: ")
            persona_invalid_json = current_module_persona()
            assert persona_invalid_json.name == default_persona_name, f"Invalid JSON: Name {persona_invalid_json.name}"
            assert persona_invalid_json.awareness["primary_concept_coord"] is None, "Invalid JSON: PCC"

            # Scenario 3: Malformed primary_concept_coord in File (e.g., wrong length)
            delete_test_profile()
            malformed_pcc_profile = {"name": "TestPCC", "awareness": {"primary_concept_coord": [1, 2, 3]}} # List of 3, not 4
            with open(TEST_PROFILE_PATH, 'w') as f:
                json.dump(malformed_pcc_profile, f)
            persona_malformed_pcc = current_module_persona()
            assert persona_malformed_pcc.name == "TestPCC", f"Malformed PCC file: Name {persona_malformed_pcc.name}"
            assert persona_malformed_pcc.awareness["primary_concept_coord"] is None, f"Malformed PCC file: PCC should be None, got {persona_malformed_pcc.awareness['primary_concept_coord']}"

            # Scenario 4: Malformed primary_concept_coord in File (e.g., non-numeric)
            delete_test_profile()
            malformed_pcc_profile_non_numeric = {"name": "TestPCC2", "awareness": {"primary_concept_coord": [1, 2, 3, "not-a-number"]}}
            with open(TEST_PROFILE_PATH, 'w') as f:
                json.dump(malformed_pcc_profile_non_numeric, f)
            persona_malformed_pcc_nn = current_module_persona()
            assert persona_malformed_pcc_nn.name == "TestPCC2"
            assert persona_malformed_pcc_nn.awareness["primary_concept_coord"] is None, f"Malformed PCC (non-numeric) file: PCC should be None, got {persona_malformed_pcc_nn.awareness['primary_concept_coord']}"


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
