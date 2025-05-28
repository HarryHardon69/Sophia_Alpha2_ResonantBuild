"""
core/dialogue.py

Manages dialogue interactions, user input processing, and response generation
for Sophia_Alpha2.
"""

import sys
import os
import datetime
import json
import time
import traceback

# --- Configuration Import ---
try:
    from .. import config
except ImportError:
    # Fallback for standalone execution
    print("Dialogue.py: Could not import 'config' from parent package. Attempting for standalone use.")
    try:
        import config as app_config # Use an alias to avoid conflict if 'config' is a local var name
        config = app_config
        print("Dialogue.py: Successfully imported 'config' directly.")
    except ImportError:
        print("Dialogue.py: Failed to import 'config'. Critical error.")
        config = None # Placeholder

# --- Core Component Imports with Fallbacks ---
_BRAIN_THINK_AVAILABLE = False
try:
    from .brain import think
    _BRAIN_THINK_AVAILABLE = True
    if config and getattr(config, 'VERBOSE_OUTPUT', False) and not getattr(sys, '_IS_TEST_RUNNING', False):
        print("Dialogue.py: Successfully imported 'think' from core.brain")
except ImportError as e:
    if not getattr(sys, '_IS_TEST_RUNNING', False):
        print(f"Dialogue.py: Failed to import 'think' from core.brain: {e}. Core thinking capabilities will be unavailable.")
    def think(*args, **kwargs): # Mock function
        if not getattr(sys, '_IS_TEST_RUNNING', False):
            print("Warning (dialogue.py): Using fallback/mock think(). Brain processing is not available.")
        return (["Mock brain response: Brain not available."], "Brain not available", {"error": "Brain module not loaded"})

_PERSONA_AVAILABLE = False
try:
    from .persona import Persona
    _PERSONA_AVAILABLE = True
    if config and getattr(config, 'VERBOSE_OUTPUT', False) and not getattr(sys, '_IS_TEST_RUNNING', False):
        print("Dialogue.py: Successfully imported 'Persona' from core.persona")
except ImportError as e:
    if not getattr(sys, '_IS_TEST_RUNNING', False):
        print(f"Dialogue.py: Failed to import 'Persona' from core.persona: {e}. Persona management will be unavailable.")
    Persona = None # type: ignore

_MEMORY_MODULE_AVAILABLE = False
_MEMORY_STORE_MEMORY_AVAILABLE = False
try:
    from . import memory as memory_module # For accessing _knowledge_graph
    from .memory import store_memory 
    _MEMORY_MODULE_AVAILABLE = True
    _MEMORY_STORE_MEMORY_AVAILABLE = True
    if config and getattr(config, 'VERBOSE_OUTPUT', False) and not getattr(sys, '_IS_TEST_RUNNING', False):
        print("Dialogue.py: Successfully imported 'memory' module and 'store_memory' from core.memory")
except ImportError as e:
    if not getattr(sys, '_IS_TEST_RUNNING', False):
        print(f"Dialogue.py: Failed to import from core.memory: {e}. Memory storage and graph access will be unavailable.")
    memory_module = None # type: ignore
    def store_memory(*args, **kwargs): # Mock function
        if not getattr(sys, '_IS_TEST_RUNNING', False):
            print("Warning (dialogue.py): Using fallback/mock store_memory(). Memory storage is not available.")
        return False

_ETHICS_MODULE_AVAILABLE = False
_ETHICS_SCORE_ETHICS_AVAILABLE = False
_ETHICS_TRACK_TRENDS_AVAILABLE = False
try:
    from . import ethics as ethics_module # For accessing _ethics_db
    from .ethics import score_ethics, track_trends
    _ETHICS_MODULE_AVAILABLE = True
    _ETHICS_SCORE_ETHICS_AVAILABLE = True
    _ETHICS_TRACK_TRENDS_AVAILABLE = True
    if config and getattr(config, 'VERBOSE_OUTPUT', False) and not getattr(sys, '_IS_TEST_RUNNING', False):
        print("Dialogue.py: Successfully imported 'ethics' module, 'score_ethics', 'track_trends' from core.ethics")
except ImportError as e:
    if not getattr(sys, '_IS_TEST_RUNNING', False):
        print(f"Dialogue.py: Failed to import from core.ethics: {e}. Ethical processing will be limited.")
    ethics_module = None # type: ignore
    def score_ethics(*args, **kwargs): # Mock function
        if not getattr(sys, '_IS_TEST_RUNNING', False):
            print("Warning (dialogue.py): Using fallback/mock score_ethics(). Ethical scoring is not available.")
        return 0.5 
    def track_trends(*args, **kwargs): # Mock function
        if not getattr(sys, '_IS_TEST_RUNNING', False):
            print("Warning (dialogue.py): Using fallback/mock track_trends(). Trend tracking is not available.")
        return "Trends unavailable"

_LIBRARY_AVAILABLE = False
_SUMMARIZE_TEXT_AVAILABLE = False
_IS_VALID_COORDINATE_AVAILABLE = False
_MITIGATOR_AVAILABLE = False
_CORE_EXCEPTION_AVAILABLE = False

try:
    from . import library as library_module # For accessing KNOWLEDGE_LIBRARY
    from .library import (
        Mitigator, KNOWLEDGE_LIBRARY, CoreException, 
        BrainError, PersonaError, MemoryError, EthicsError, 
        LibraryError, DialogueError, NetworkError, ConfigError,
        summarize_text, is_valid_coordinate
    )
    _LIBRARY_AVAILABLE = True
    _SUMMARIZE_TEXT_AVAILABLE = True
    _IS_VALID_COORDINATE_AVAILABLE = True
    _MITIGATOR_AVAILABLE = True
    _CORE_EXCEPTION_AVAILABLE = True
    if config and getattr(config, 'VERBOSE_OUTPUT', False) and not getattr(sys, '_IS_TEST_RUNNING', False):
        print("Dialogue.py: Successfully imported components from core.library")
except ImportError as e:
    if not getattr(sys, '_IS_TEST_RUNNING', False):
        print(f"Dialogue.py: Failed to import from core.library: {e}. Library utilities and custom exceptions will be unavailable.")
    library_module = None # type: ignore
    Mitigator = None # type: ignore
    KNOWLEDGE_LIBRARY = {} # Placeholder
    if 'CoreException' not in globals(): CoreException = type('CoreException', (Exception,), {}) # type: ignore
    if 'BrainError' not in globals(): BrainError = type('BrainError', (CoreException,), {}) # type: ignore
    if 'PersonaError' not in globals(): PersonaError = type('PersonaError', (CoreException,), {}) # type: ignore
    if 'MemoryError' not in globals(): MemoryError = type('MemoryError', (CoreException,), {}) # type: ignore
    if 'EthicsError' not in globals(): EthicsError = type('EthicsError', (CoreException,), {}) # type: ignore
    if 'LibraryError' not in globals(): LibraryError = type('LibraryError', (CoreException,), {}) # type: ignore
    if 'DialogueError' not in globals(): DialogueError = type('DialogueError', (CoreException,), {}) # type: ignore
    if 'NetworkError' not in globals(): NetworkError = type('NetworkError', (CoreException,), {}) # type: ignore
    if 'ConfigError' not in globals(): ConfigError = type('ConfigError', (CoreException,), {}) # type: ignore
    
    def summarize_text(text: str, max_length: int = 100) -> str:
        if not getattr(sys, '_IS_TEST_RUNNING', False): print("Warning (dialogue.py): Using fallback/mock summarize_text().")
        if not text: return ""
        return text[:max_length-3] + "..." if len(text) > max_length else text
    
    def is_valid_coordinate(coord: tuple | list) -> bool:
        if not getattr(sys, '_IS_TEST_RUNNING', False): print("Warning (dialogue.py): Using fallback/mock is_valid_coordinate().")
        return isinstance(coord, (list,tuple)) and (3 <= len(coord) <=4) and all(isinstance(n, (int,float)) for n in coord)


# --- Logging Function ---
def _log_dialogue_event(event_type: str, data: dict, level: str = "info"):
    log_message_data = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "module": "dialogue",
        "level": level.upper(),
        "event_type": event_type,
        "data": data
    }
    log_message_str = json.dumps(log_message_data) + "\n"
    
    log_path_to_use = None
    min_log_level_str = "info" 
    ensure_path_func = None

    if config:
        log_path_to_use = getattr(config, 'SYSTEM_LOG_PATH', None)
        min_log_level_str = getattr(config, 'LOG_LEVEL', "info")
        ensure_path_func = getattr(config, 'ensure_path', None)

    level_map = {"debug": 0, "info": 1, "warning": 2, "error": 3, "critical": 4}
    current_event_level_val = level_map.get(level.lower(), 1)
    min_log_level_val = level_map.get(min_log_level_str.lower(), 1)
    
    is_test_running = getattr(sys, '_IS_TEST_RUNNING', False) 

    if current_event_level_val < min_log_level_val and not is_test_running: 
        return 

    if log_path_to_use:
        try:
            if ensure_path_func:
                 ensure_path_func(log_path_to_use)
            else: 
                log_dir = os.path.dirname(log_path_to_use)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
            with open(log_path_to_use, 'a', encoding='utf-8') as f:
                f.write(log_message_str)
        except Exception as e:
            if not is_test_running:
                sys.stderr.write(f"Error writing to system log ({log_path_to_use}): {e}\n")
                sys.stderr.write(log_message_str) 
    else:
        if not is_test_running:
            sys.stderr.write(log_message_str)

# --- Shared Persona Instance Management ---
_dialogue_persona_instance: Persona | None = None

def get_dialogue_persona() -> Persona | None:
    global _dialogue_persona_instance
    if _dialogue_persona_instance is None:
        if not _PERSONA_AVAILABLE or Persona is None:
            _log_dialogue_event("get_persona_failed", {"reason": "Persona class not imported successfully."}, level="error")
            return None
        try:
            _dialogue_persona_instance = Persona()
            _log_dialogue_event("dialogue_persona_initialized", {"persona_name": _dialogue_persona_instance.name if _dialogue_persona_instance else "Unknown"}, level="info")
        except Exception as e:
            _log_dialogue_event("dialogue_persona_init_error", {"error": str(e), "trace": traceback.format_exc()}, level="critical")
            _dialogue_persona_instance = None 
            return None
    return _dialogue_persona_instance

# --- Main Dialogue Functions ---
def generate_response(user_input: str, stream_thought_steps: bool = False) -> tuple[str, list, dict]:
    _log_dialogue_event("generate_response_start", {"user_input_snippet": user_input[:100]})
    persona_instance = get_dialogue_persona()
    if persona_instance is None:
        _log_dialogue_event("generate_response_error", {"reason": "Persona module not available."}, level="critical")
        return ("Error: Persona module not available. Cannot generate response.", [], {"error": "Persona unavailable"})

    awareness_metrics = {
        "curiosity": 0.1, "context_stability": 0.3, "self_evolution_rate": 0.0,
        "coherence": 0.0, "active_llm_fallback": True, 
        "primary_concept_coord": None, "raw_t_intensity": 0.0, "snn_error": None
    }
    thought_steps: list[str] = ["Dialogue Manager: Initializing response sequence."]
    brain_response_text = "No response generated due to an internal issue." 

    try:
        if _BRAIN_THINK_AVAILABLE:
            thought_steps_brain, brain_response_text_from_module, brain_awareness = think(user_input, stream_thought_steps=stream_thought_steps)
            brain_response_text = brain_response_text_from_module 
            if isinstance(brain_awareness, dict): awareness_metrics.update(brain_awareness)
            else: _log_dialogue_event("brain_awareness_invalid_type", {"type": type(brain_awareness).__name__}, level="warning")
            if isinstance(thought_steps_brain, list): thought_steps.extend(thought_steps_brain)
            else: _log_dialogue_event("brain_thought_steps_invalid_type", {"type": type(thought_steps_brain).__name__}, level="warning")
            awareness_metrics["active_llm_fallback"] = brain_awareness.get("active_llm_fallback", True) if isinstance(brain_awareness, dict) else True
        else:
            thought_steps.append("Dialogue Manager: Brain 'think' function not available. Using default response.")
            awareness_metrics["snn_error"] = "Brain module not available"
    except Exception as e_brain:
        _log_dialogue_event("brain_think_error", {"error": str(e_brain), "trace": traceback.format_exc()}, level="error")
        brain_response_text = f"My apologies, I encountered an issue while processing that: {str(e_brain)}"
        awareness_metrics["snn_error"] = str(e_brain)
        awareness_metrics["active_llm_fallback"] = True 
        thought_steps.append(f"Error during brain.think: {str(e_brain)}")

    try:
        if _PERSONA_AVAILABLE and persona_instance:
            persona_instance.update_awareness(awareness_metrics)
            thought_steps.append("Dialogue Manager: Persona awareness updated.")
        else: thought_steps.append("Dialogue Manager: Persona instance or module not available for awareness update.")
    except Exception as e_persona_update:
        _log_dialogue_event("persona_update_awareness_error", {"error": str(e_persona_update), "trace": traceback.format_exc()}, level="error")
        thought_steps.append(f"Error updating persona awareness: {str(e_persona_update)}")

    ethical_score = 0.5 
    concept_summary_for_ethics = summarize_text(user_input if awareness_metrics.get("active_llm_fallback") else brain_response_text, 100)
    action_description_for_ethics = summarize_text(brain_response_text, 200)
    try:
        if _ETHICS_SCORE_ETHICS_AVAILABLE:
            calculated_score = score_ethics(awareness_metrics, concept_summary_for_ethics, action_description_for_ethics)
            if isinstance(calculated_score, (int, float)) and 0.0 <= calculated_score <= 1.0: ethical_score = float(calculated_score)
            else: _log_dialogue_event("ethics_score_invalid_type", {"score": calculated_score, "type": type(calculated_score).__name__}, level="warning")
            thought_steps.append(f"Dialogue Manager: Ethical score calculated: {ethical_score:.2f}")
        else: thought_steps.append("Dialogue Manager: Ethical scoring not available. Using default score.")
    except Exception as e_ethics_score:
        _log_dialogue_event("ethics_score_error", {"error": str(e_ethics_score), "trace": traceback.format_exc()}, level="error")
        thought_steps.append(f"Error during ethical scoring: {str(e_ethics_score)}")

    try:
        if _MEMORY_STORE_MEMORY_AVAILABLE:
            primary_coord = awareness_metrics.get("primary_concept_coord")
            if is_valid_coordinate(primary_coord):
                concept_name_for_memory = summarize_text(user_input, 30).replace("...", "").strip() or "interaction_summary"
                intensity_for_memory = awareness_metrics.get("raw_t_intensity", 0.0)
                memory_data = {
                    "concept_name": concept_name_for_memory, "concept_coord": primary_coord,
                    "summary": summarize_text(f"User: {user_input} | Sophia: {brain_response_text}", 200),
                    "intensity": float(intensity_for_memory) if isinstance(intensity_for_memory, (int,float)) else 0.0,
                    "ethical_alignment": ethical_score,
                }
                memory_entry_id = store_memory(**memory_data) # type: ignore
                _log_dialogue_event("memory_store_attempt", {"entry_id": memory_entry_id if memory_entry_id else "failed", "concept": concept_name_for_memory})
                thought_steps.append(f"Dialogue Manager: Memory storage attempted for '{concept_name_for_memory}'. ID: {memory_entry_id}")
            else: thought_steps.append("Dialogue Manager: Primary concept coordinate not valid for memory storage.")
        else: thought_steps.append("Dialogue Manager: Memory storage not available.")
    except Exception as e_memory_store:
        _log_dialogue_event("memory_store_error", {"error": str(e_memory_store), "trace": traceback.format_exc()}, level="error")
        thought_steps.append(f"Error storing memory: {str(e_memory_store)}")

    current_mode = getattr(persona_instance, 'mode', 'N/A').upper()
    final_response = f"[{current_mode}|E:{ethical_score:.2f}] {brain_response_text}"
    mitigation_applied_flag = False
    try:
        if _MITIGATOR_AVAILABLE and Mitigator:
            mitigation_eth_threshold = getattr(config, 'MITIGATION_ETHICAL_THRESHOLD', 0.3) if config else 0.3
            alignment_eth_threshold = getattr(config, 'ETHICAL_ALIGNMENT_THRESHOLD', 0.5) if config else 0.5
            if ethical_score < mitigation_eth_threshold:
                mitigator_instance = Mitigator()
                mitigated_text = mitigator_instance.moderate_ethically_flagged_content(brain_response_text, ethical_score, strict_mode=True)
                final_response = f"[{current_mode}|E:{ethical_score:.2f}] [MITIGATED] {mitigated_text}"
                mitigation_applied_flag = True
                thought_steps.append("Dialogue Manager: Mitigation applied (strict).")
            elif ethical_score < alignment_eth_threshold:
                final_response = f"[{current_mode}|E:{ethical_score:.2f}] [CAUTION] {brain_response_text}" 
                mitigation_applied_flag = True 
                thought_steps.append("Dialogue Manager: Caution applied to response.")
        else: thought_steps.append("Dialogue Manager: Mitigation utilities not available.")
    except Exception as e_mitigation:
        _log_dialogue_event("mitigation_error", {"error": str(e_mitigation), "trace": traceback.format_exc()}, level="error")
        thought_steps.append(f"Error during content mitigation: {str(e_mitigation)}")

    try:
        if _ETHICS_TRACK_TRENDS_AVAILABLE:
            trends_summary = track_trends(ethical_score=ethical_score, context=concept_summary_for_ethics) 
            _log_dialogue_event("ethics_trends_updated", {"summary": trends_summary if trends_summary else "No summary"}, level="debug")
            thought_steps.append(f"Dialogue Manager: Ethical trends updated. Summary: {trends_summary}")
        else: thought_steps.append("Dialogue Manager: Ethical trend tracking not available.")
    except Exception as e_ethics_trends:
        _log_dialogue_event("ethics_track_trends_error", {"error": str(e_ethics_trends), "trace": traceback.format_exc()}, level="error")
        thought_steps.append(f"Error tracking ethical trends: {str(e_ethics_trends)}")

    _log_dialogue_event("generate_response_end", {"final_response_snippet": final_response[:100], "mitigation_applied": mitigation_applied_flag})
    return (final_response, thought_steps, awareness_metrics)

def dialogue_loop(enable_streaming_thoughts: bool = None):
    _log_dialogue_event("dialogue_loop_start", {})
    persona_instance = get_dialogue_persona()
    if persona_instance is None:
        print("CRITICAL: Persona module could not be initialized. Dialogue loop cannot start.", file=sys.stderr)
        _log_dialogue_event("dialogue_loop_critical_fail", {"reason": "Persona unavailable at start"}, level="critical")
        return

    print("Sophia_Alpha2 Dialogue Interface. Type '!help' for commands, 'quit' or 'exit' to end.")
    if enable_streaming_thoughts is not None: stream_thoughts_cli = enable_streaming_thoughts
    elif config: stream_thoughts_cli = getattr(config, 'VERBOSE_OUTPUT', False)
    else: stream_thoughts_cli = False
    print(f"Thought Streaming: {'ON' if stream_thoughts_cli else 'OFF'}. Type '!stream' to toggle.")

    while True:
        try:
            current_persona = get_dialogue_persona()
            if current_persona is None:
                print("CRITICAL: Persona became unavailable during loop. Exiting.", file=sys.stderr)
                _log_dialogue_event("dialogue_loop_persona_lost", {}, level="critical")
                break
            awareness = getattr(current_persona, 'awareness', {})
            prompt_mode = getattr(current_persona, 'mode', 'N/A').upper()
            prompt_name = getattr(current_persona, 'name', 'Sophia')
            prompt_curiosity = awareness.get('curiosity', 0.0)
            prompt_coherence = awareness.get('coherence', 0.0)
            prompt = f"{prompt_name}({prompt_mode}|A:{prompt_curiosity:.1f},C:{prompt_coherence:.1f})> "
            user_input = input(prompt)
        except KeyboardInterrupt: print("\nExiting dialogue loop (KeyboardInterrupt)..."); break
        except EOFError: print("\nExiting dialogue loop (EOFError)..."); break

        user_input_lower = user_input.strip().lower()
        if user_input_lower in ["quit", "exit"]: break
        elif user_input_lower == "!stream":
            stream_thoughts_cli = not stream_thoughts_cli
            print(f"Thought Streaming: {'ON' if stream_thoughts_cli else 'OFF'}")
        elif user_input_lower == "!persona":
            if current_persona and hasattr(current_persona, 'get_intro'):
                print(current_persona.get_intro())
                print(json.dumps(getattr(current_persona, 'awareness', {}), indent=2))
            else: print("Persona details unavailable.")
        elif user_input_lower == "!ethicsdb":
            if _ETHICS_MODULE_AVAILABLE and ethics_module and hasattr(ethics_module, '_ethics_db'):
                print(f"Ethics DB (first 5 entries): {list(ethics_module._ethics_db.items())[:5]}") 
                print(f"Total ethics entries: {len(ethics_module._ethics_db)}")
            else: print("Ethics DB details unavailable or module not loaded.")
        elif user_input_lower == "!memgraph":
            if _MEMORY_MODULE_AVAILABLE and memory_module and hasattr(memory_module, '_knowledge_graph'):
                kg = memory_module._knowledge_graph
                print(f"Memory Graph: Nodes={len(kg.get('nodes',[]))}, Edges={len(kg.get('edges',[]))}")
            else: print("Memory Graph details unavailable or module not loaded.")
        elif user_input_lower == "!library":
            if _LIBRARY_AVAILABLE and library_module:
                print(f"Knowledge Library Entries: {len(library_module.KNOWLEDGE_LIBRARY)}")
            else: print("Knowledge Library details unavailable or module not loaded.")
        elif user_input_lower == "!help":
            print("\nAvailable Commands:\n  !help          - Show this help message.\n  !stream        - Toggle streaming of thought steps.\n  !persona       - Display current persona information.\n  !ethicsdb      - Display summary of ethics database (debug).\n  !memgraph      - Display summary of memory graph (debug).\n  !library       - Display summary of knowledge library (debug).\n  quit / exit    - End the dialogue session.\n")
        else:
            if not user_input.strip(): continue
            print("Thinking...")
            try:
                final_response, thought_steps, _ = generate_response(user_input, stream_thought_steps=stream_thoughts_cli)
                print(f"\n{final_response}")
                if stream_thoughts_cli and thought_steps:
                    print("\n--- Thought Process ---")
                    for i, step in enumerate(thought_steps): print(f"{i+1}. {step}")
                    print("--- End of Thoughts ---\n")
            except CoreException as e_core: 
                _log_dialogue_event("dialogue_loop_core_exception", {"error": str(e_core), "type": type(e_core).__name__}, level="error")
                print(f"A known error occurred: ({type(e_core).__name__}) {str(e_core)}")
            except Exception as e_unexpected:
                _log_dialogue_event("dialogue_loop_unexpected_exception", {"error": str(e_unexpected), "trace": traceback.format_exc()}, level="critical")
                print(f"An unexpected error occurred: {str(e_unexpected)}. Please check logs.")
    print("Dialogue session ended.")
    _log_dialogue_event("dialogue_loop_end", {})

# --- Comprehensive Self-Testing Block ---
_IS_TEST_RUNNING = False 

if __name__ == "__main__":
    _IS_TEST_RUNNING = True 
    # print(f"INFO (dialogue.py __main__): _IS_TEST_RUNNING set to {_IS_TEST_RUNNING}")

    import unittest
    import unittest.mock as mock
    import io
    import contextlib # For redirect_stdout

    # --- Test Utilities ---
    class TempConfigOverride:
        def __init__(self, temp_configs_dict):
            self.temp_configs = temp_configs_dict
            self.original_values = {}
            self.config_module_was_none = False
            self.original_global_config = None

        def __enter__(self):
            global config 
            self.original_global_config = config 
            if config is None:
                self.config_module_was_none = True
                class DummyConfig: pass
                config = DummyConfig()
            for key, value in self.temp_configs.items():
                if hasattr(config, key): self.original_values[key] = getattr(config, key)
                else: self.original_values[key] = "__ATTR_NOT_SET__"
                setattr(config, key, value)
            return config

        def __exit__(self, exc_type, exc_val, exc_tb):
            global config 
            current_config_module = config 
            for key, original_value in self.original_values.items():
                if original_value == "__ATTR_NOT_SET__":
                    if hasattr(current_config_module, key): delattr(current_config_module, key)
                else: setattr(current_config_module, key, original_value)
            config = self.original_global_config

    TEST_DIALOGUE_SYSTEM_LOG = "test_dialogue_system_log.txt"
    TEST_PERSONA_PROFILE = "test_dialogue_persona_profile.json"
    # Add other test file paths if ethics/memory/library are to be tested for file creation
    TEST_ETHICS_DB = "test_dialogue_ethics.db"
    TEST_MEMORY_KG = "test_dialogue_knowledge_graph.json"
    TEST_LIBRARY_LOG = "test_dialogue_library_log.json"

    def delete_test_files():
        files_to_delete = [TEST_DIALOGUE_SYSTEM_LOG, TEST_PERSONA_PROFILE, TEST_ETHICS_DB, TEST_MEMORY_KG, TEST_LIBRARY_LOG]
        for f_path in files_to_delete:
            if os.path.exists(f_path):
                try: os.remove(f_path)
                except OSError as e: print(f"Warning: Could not delete test file {f_path}: {e}", file=sys.stderr)

    def setup_test_environment(test_specific_configs=None):
        global _dialogue_persona_instance, KNOWLEDGE_LIBRARY # For library module direct access in debug
        if _LIBRARY_AVAILABLE and library_module: # Reset library's KNOWLEDGE_LIBRARY if library module was imported
             library_module.KNOWLEDGE_LIBRARY = {}
             library_module._library_dirty_flag = False
        
        # Reset other relevant singletons or module-level states if necessary
        # (e.g., memory._knowledge_graph, ethics._ethics_db if they are directly manipulated)
        # This is a simplified approach; a full framework would manage this more cleanly.
        if _MEMORY_MODULE_AVAILABLE and memory_module: memory_module._knowledge_graph = {"nodes": [], "edges": []}
        if _ETHICS_MODULE_AVAILABLE and ethics_module: ethics_module._ethics_db = {}


        delete_test_files()
        _dialogue_persona_instance = None # Reset shared persona instance for dialogue tests

        base_test_configs = {
            "SYSTEM_LOG_PATH": TEST_DIALOGUE_SYSTEM_LOG,
            "PERSONA_PROFILE_PATH": TEST_PERSONA_PROFILE,
            "ETHICS_DB_PATH": TEST_ETHICS_DB, # Assuming ethics module uses this
            "KNOWLEDGE_GRAPH_PATH": TEST_MEMORY_KG, # Assuming memory module uses this
            "LIBRARY_LOG_PATH": TEST_LIBRARY_LOG, # Assuming library module uses this
            "VERBOSE_OUTPUT": False,
            "LOG_LEVEL": "debug", 
            "MITIGATION_ETHICAL_THRESHOLD": 0.4, # Example for testing
            "ETHICAL_ALIGNMENT_THRESHOLD": 0.6, # Example for testing
            "REQUIRE_PUBLIC_STORAGE_CONSENT": False, # Simplify tests by default
            "ensure_path": lambda path_to_ensure: os.makedirs(os.path.dirname(path_to_ensure), exist_ok=True) if os.path.dirname(path_to_ensure) else None,
        }
        if test_specific_configs: base_test_configs.update(test_specific_configs)
        return TempConfigOverride(base_test_configs)

    class TestDialogueModule(unittest.TestCase):
        def setUp(self):
            # This ensures that for each test method, we have a clean environment.
            # The TempConfigOverride is applied per test method using `with`.
            # Resetting the persona instance ensures each test starts fresh.
            global _dialogue_persona_instance
            _dialogue_persona_instance = None 
            delete_test_files() # Clean files before each test method too

        @mock.patch('core.dialogue.track_trends')
        @mock.patch('core.dialogue.Mitigator')
        @mock.patch('core.dialogue.store_memory')
        @mock.patch('core.dialogue.score_ethics')
        @mock.patch('core.dialogue.Persona') 
        @mock.patch('core.dialogue.think')
        def test_generate_response_snn_success_high_ethics(self, mock_think, MockPersonaClass, mock_score_ethics, mock_store_memory, MockMitigatorClass, mock_track_trends):
            test_configs = setup_test_environment()
            with test_configs as current_config:
                # Configure Persona mock
                mock_persona_instance = MockPersonaClass.return_value
                mock_persona_instance.name = "TestSophia"
                mock_persona_instance.mode = "TestMode"
                mock_persona_instance.awareness = {"curiosity": 0.5, "coherence": 0.8}
                
                # Configure think mock for SNN success
                mock_think.return_value = (["SNN thought step 1"], "SNN response", {"active_llm_fallback": False, "curiosity": 0.9})
                mock_score_ethics.return_value = 0.9 # High ethics

                response, thoughts, awareness = generate_response("test input")

                mock_think.assert_called_once_with("test input", stream_thought_steps=False)
                mock_persona_instance.update_awareness.assert_called_once()
                # Example check for awareness update:
                # self.assertTrue(mock_persona_instance.update_awareness.call_args[0][0]['active_llm_fallback'] == False)
                mock_score_ethics.assert_called_once()
                mock_store_memory.assert_called_once()
                MockMitigatorClass.return_value.moderate_ethically_flagged_content.assert_not_called()
                mock_track_trends.assert_called_once()
                self.assertIn("[TESTMODE|E:0.90] SNN response", response)
                self.assertNotIn("[CAUTION]", response)
                self.assertNotIn("[MITIGATED]", response)

        @mock.patch('core.dialogue.track_trends')
        @mock.patch('core.dialogue.Mitigator')
        @mock.patch('core.dialogue.store_memory')
        @mock.patch('core.dialogue.score_ethics')
        @mock.patch('core.dialogue.Persona')
        @mock.patch('core.dialogue.think')
        def test_generate_response_ethical_caution(self, mock_think, MockPersonaClass, mock_score_ethics, mock_store_memory, MockMitigatorClass, mock_track_trends):
            test_configs = setup_test_environment({"ETHICAL_ALIGNMENT_THRESHOLD": 0.6, "MITIGATION_ETHICAL_THRESHOLD": 0.3}) # Ensure threshold allows caution
            with test_configs as current_config:
                mock_persona_instance = MockPersonaClass.return_value
                mock_persona_instance.name="TestSophia"; mock_persona_instance.mode="TestMode"
                mock_think.return_value = ([], "Ethical caution response", {"active_llm_fallback": False})
                mock_score_ethics.return_value = 0.5 # Caution range (between 0.3 and 0.6)
                
                response, _, _ = generate_response("caution input")
                self.assertIn("[CAUTION]", response)
                MockMitigatorClass.return_value.moderate_ethically_flagged_content.assert_not_called()

        @mock.patch('core.dialogue.track_trends')
        @mock.patch('core.dialogue.Mitigator')
        @mock.patch('core.dialogue.store_memory')
        @mock.patch('core.dialogue.score_ethics')
        @mock.patch('core.dialogue.Persona')
        @mock.patch('core.dialogue.think')
        def test_generate_response_ethical_mitigation(self, mock_think, MockPersonaClass, mock_score_ethics, mock_store_memory, MockMitigatorClass, mock_track_trends):
            test_configs = setup_test_environment({"MITIGATION_ETHICAL_THRESHOLD": 0.4}) # Ensure mitigation triggers
            with test_configs as current_config:
                mock_persona_instance = MockPersonaClass.return_value
                mock_persona_instance.name="TestSophia"; mock_persona_instance.mode="TestMode"
                mock_mitigator_instance = MockMitigatorClass.return_value
                mock_mitigator_instance.moderate_ethically_flagged_content.return_value = "Mitigated content text"
                mock_think.return_value = ([], "Original unethical response", {"active_llm_fallback": False})
                mock_score_ethics.return_value = 0.2 # Low ethics, triggers mitigation
                
                response, _, _ = generate_response("mitigation input")
                mock_mitigator_instance.moderate_ethically_flagged_content.assert_called_once()
                self.assertIn("[MITIGATED] Mitigated content text", response)

        @mock.patch('core.dialogue.Persona')
        @mock.patch('core.dialogue.think')
        def test_generate_response_brain_error(self, mock_think, MockPersonaClass):
            test_configs = setup_test_environment()
            with test_configs:
                mock_persona_instance = MockPersonaClass.return_value
                mock_persona_instance.name="TestSophia"; mock_persona_instance.mode="TestMode"
                mock_think.side_effect = Exception("Simulated brain error")
                
                response, _, awareness = generate_response("input causing brain error")
                self.assertIn("My apologies, I encountered an issue", response)
                self.assertTrue(awareness['active_llm_fallback'])
                self.assertEqual(awareness['snn_error'], "Simulated brain error")
                mock_persona_instance.update_awareness.assert_called_once() # Still called

        @mock.patch('core.dialogue.get_dialogue_persona')
        def test_generate_response_persona_unavailable(self, mock_get_persona):
            test_configs = setup_test_environment()
            with test_configs:
                mock_get_persona.return_value = None
                response, _, _ = generate_response("any input")
                self.assertIn("Error: Persona module not available", response)

        @mock.patch('builtins.input')
        @mock.patch('core.dialogue.generate_response') 
        def test_dialogue_loop_commands(self, mock_generate_response, mock_input):
            # Capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output # Redirect stdout
            
            test_configs = setup_test_environment({"VERBOSE_OUTPUT": False}) # For predictable !stream toggle
            with test_configs:
                # Mock persona for the loop
                mock_persona_for_loop = mock.MagicMock(spec=Persona)
                mock_persona_for_loop.name = "LoopSophia"
                mock_persona_for_loop.mode = "LoopMode"
                mock_persona_for_loop.awareness = {"curiosity": 0.1, "coherence": 0.2}
                mock_persona_for_loop.get_intro.return_value = "LoopSophia Intro"

                with mock.patch('core.dialogue.get_dialogue_persona', return_value=mock_persona_for_loop):
                    # Test !stream
                    mock_input.side_effect = ["!stream", "!stream", "quit"]
                    dialogue_loop()
                    output = captured_output.getvalue()
                    self.assertIn("Thought Streaming: ON", output)
                    self.assertIn("Thought Streaming: OFF", output) # Toggled back

                    # Test !persona
                    captured_output.truncate(0); captured_output.seek(0) # Clear buffer
                    mock_input.side_effect = ["!persona", "quit"]
                    dialogue_loop()
                    self.assertIn("LoopSophia Intro", captured_output.getvalue())
                    
                    # Test !help
                    captured_output.truncate(0); captured_output.seek(0)
                    mock_input.side_effect = ["!help", "quit"]
                    dialogue_loop()
                    self.assertIn("Available Commands:", captured_output.getvalue())

                    # Test regular input calls generate_response
                    captured_output.truncate(0); captured_output.seek(0)
                    mock_generate_response.return_value = ("Mocked response", ["mock thought"], {})
                    mock_input.side_effect = ["hello sophia", "quit"]
                    dialogue_loop(enable_streaming_thoughts=False) # Test with streaming off initially
                    mock_generate_response.assert_called_with("hello sophia", stream_thought_steps=False)
                    self.assertIn("Mocked response", captured_output.getvalue())
            
            sys.stdout = sys.__stdout__ # Restore stdout

    # --- Test Runner Logic ---
    print("Starting Dialogue Module Self-Tests...")
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDialogueModule))
    
    runner = unittest.TextTestRunner(verbosity=2)
    results = runner.run(suite)
    
    delete_test_files() # Clean up after all tests
    _IS_TEST_RUNNING = False 

    if results.wasSuccessful():
        print("\nALL DIALOGUE MODULE TESTS PASSED.")
        sys.exit(0)
    else:
        print("\nDIALOGUE MODULE TESTS FAILED.")
        sys.exit(1)
```
