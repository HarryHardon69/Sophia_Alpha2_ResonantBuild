"""
core/dialogue.py

Manages dialogue interactions, user input processing, and response generation
for Sophia_Alpha2.
"""

import sys
import os
import datetime
import json
import traceback
import threading
_persona_instance_lock = threading.Lock()
# 'time' module is used only in the __main__ test block, not in the module's core logic.
# It can be imported there if specific test timing is needed.

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
        # Return a dictionary structure similar to what the real function might return in an error/default state
        return {
            "trend_direction": "unavailable_mock_fallback",
            "data_points_used": 0,
            "last_updated": datetime.datetime.utcnow().isoformat() + "Z"
            # Add other keys expected by consumers if necessary, with default values
        }

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
    """
    Logs a structured event from the dialogue module.

    Similar to logging functions in other core modules, this function formats
    log data as JSON and writes to the system log file specified in config,
    respecting the configured LOG_LEVEL. Includes fallback to stderr if
    config or file writing fails. Suppresses print output during test runs
    identified by `sys._IS_TEST_RUNNING`.

    Args:
        event_type (str): The type of event being logged (e.g., "generate_response_start").
        data (dict): A dictionary containing data relevant to the event.
        level (str, optional): The severity level of the log ("debug", "info", 
                               "warning", "error", "critical"). Defaults to "info".
    """
    log_message_data = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z", # ISO 8601 UTC timestamp
        "module": "dialogue", # Identifies the source module of the log
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
_dialogue_persona_instance: Persona | None = None # Cached global instance of Persona.

def get_dialogue_persona() -> Persona | None:
    """
    Retrieves or initializes the shared Persona instance for the dialogue module.

    This function implements a singleton pattern for the Persona object within
    the dialogue module's scope. If the Persona class was not successfully
    imported, or if instantiation fails, it logs an error and returns None.

    Returns:
        Persona | None: The shared Persona instance, or None if unavailable/failed.
    """
    global _dialogue_persona_instance
    if _dialogue_persona_instance is None: # Quick check without lock first (double-checked locking pattern)
        with _persona_instance_lock: # Acquire lock
            if _dialogue_persona_instance is None: # Check again after acquiring lock
                if not _PERSONA_AVAILABLE or Persona is None:
                    _log_dialogue_event("get_persona_failed", {"reason": "Persona class was not imported successfully."}, level="error")
                    return None
                try:
                    _dialogue_persona_instance = Persona()
                    _log_dialogue_event("dialogue_persona_initialized",
                                        {"persona_name": _dialogue_persona_instance.name if _dialogue_persona_instance else "UnknownName"},
                                        level="info")
                except Exception as e:
                    _log_dialogue_event("dialogue_persona_init_error", {"error_message": str(e), "traceback": traceback.format_exc()}, level="critical")
                    _dialogue_persona_instance = None
                    # No return here, let it fall through to return the (now None) _dialogue_persona_instance
    return _dialogue_persona_instance

# --- Main Dialogue Functions ---
def generate_response(user_input: str, stream_thought_steps: bool = False) -> tuple[str, list, dict]:
    """
    Orchestrates the generation of a response to user input.

    This involves:
    1. Fetching the Persona instance.
    2. Calling the brain's `think` method to process the input and get initial response and awareness metrics.
    3. Updating the Persona's awareness based on these metrics.
    4. Calculating an ethical score for the context.
    5. Storing a memory of the interaction.
    6. Applying content mitigation/moderation if necessary based on ethics.
    7. Tracking ethical trends.
    8. Formatting and returning the final response, thought steps, and awareness metrics.

    Args:
        user_input (str): The text input from the user.
        stream_thought_steps (bool, optional): If True, the brain's `think` method
                                               might provide more verbose thought steps.
                                               Defaults to False.

    Returns:
        tuple: A tuple containing:
            - final_response (str): The generated response to the user.
            - thought_steps (list): A list of strings detailing the internal processing.
            - awareness_metrics (dict): The latest awareness metrics after processing.
    """
    _log_dialogue_event("generate_response_start", {"user_input_snippet": user_input[:100]}) # Log start and input snippet.
    
    # Step 1: Get the shared Persona instance.
    persona_instance = get_dialogue_persona()
    if persona_instance is None: # Critical failure if persona is not available.
        _log_dialogue_event("generate_response_error", {"reason": "Persona instance is unavailable."}, level="critical")
        # Return an error message and default/empty state.
        return ("Error: System persona is currently unavailable. Cannot generate response.", 
                ["Dialogue Manager: Persona unavailable."], 
                {"error": "Persona unavailable"})

    # Initialize default awareness metrics and thought log.
    # These will be updated by results from brain, ethics, etc.
    awareness_metrics = getattr(config, 'DEFAULT_AWARENESS_METRICS_DIALOGUE', {
        "curiosity": 0.1, "context_stability": 0.3, "self_evolution_rate": 0.0,
        "coherence": 0.0, "active_llm_fallback": True,
        "primary_concept_coord": None, "raw_t_intensity": 0.0, "snn_error": None
    }).copy() # Use .copy() to prevent modification of the global default dict

    thought_steps: list[str] = ["Dialogue Manager: Initializing response sequence."]
    brain_response_text = getattr(config, 'DEFAULT_DIALOGUE_ERROR_BRAIN_RESPONSE', "System did not generate a specific response due to an internal processing issue.")

    # Step 2: Process input with the brain's `think` method.
    try:
        if _BRAIN_THINK_AVAILABLE: # Check if `think` function was imported successfully.
            # Call brain.think() to get initial response and awareness metrics.
            thought_steps_brain, brain_response_text_from_module, brain_awareness_metrics = think(user_input, stream_thought_steps=stream_thought_steps)
            brain_response_text = brain_response_text_from_module # Update response text.
            
            # Update overall awareness_metrics with those from the brain.
            if isinstance(brain_awareness_metrics, dict): awareness_metrics.update(brain_awareness_metrics)
            else: _log_dialogue_event("brain_awareness_invalid_type", {"type_received": type(brain_awareness_metrics).__name__}, level="warning")
            
            # Append brain's thought steps to the main log.
            if isinstance(thought_steps_brain, list): thought_steps.extend(thought_steps_brain)
            else: _log_dialogue_event("brain_thought_steps_invalid_type", {"type_received": type(thought_steps_brain).__name__}, level="warning")
            
            # Specifically update active_llm_fallback status from brain's report.
            awareness_metrics["active_llm_fallback"] = brain_awareness_metrics.get("active_llm_fallback", True) if isinstance(brain_awareness_metrics, dict) else True
        else: # Brain's `think` function is not available (e.g., import failed).
            thought_steps.append("Dialogue Manager: Brain 'think' function unavailable. Proceeding with default response logic.")
            awareness_metrics["snn_error"] = "Brain module or 'think' function not loaded." # Record error.
    except Exception as e_brain_think: # Handle any unexpected errors from brain.think().
        _log_dialogue_event("brain_think_error", {"error_message": str(e_brain_think), "traceback": traceback.format_exc()}, level="error")
        brain_response_text = f"My apologies, I encountered an issue while processing your input: {str(e_brain_think)}"
        awareness_metrics["snn_error"] = str(e_brain_think) # Record the error.
        awareness_metrics["active_llm_fallback"] = True # Assume fallback if brain processing failed.
        thought_steps.append(f"Dialogue Manager: Error during brain.think() execution: {str(e_brain_think)}")

    # Step 3: Update Persona's awareness state.
    try:
        if _PERSONA_AVAILABLE and persona_instance: # Check if Persona class and instance are valid.
            persona_instance.update_awareness(awareness_metrics) # Update with latest metrics.
            thought_steps.append("Dialogue Manager: Persona awareness state updated.")
        else: thought_steps.append("Dialogue Manager: Persona instance or module unavailable for awareness update.")
    except Exception as e_persona_update_awareness:
        _log_dialogue_event("persona_update_awareness_error", {"error_message": str(e_persona_update_awareness), "traceback": traceback.format_exc()}, level="error")
        thought_steps.append(f"Dialogue Manager: Error updating persona awareness: {str(e_persona_update_awareness)}")

    # Step 4: Calculate ethical score for the current context.
    ethical_score_value = getattr(config, 'DEFAULT_DIALOGUE_ETHICAL_SCORE_FALLBACK', 0.5)
    # Determine summary for ethics: use user input if LLM fallback, else brain's response.
    summary_len_short = getattr(config, 'DIALOGUE_SUMMARY_LENGTH_SHORT', 100)
    summary_len_action = getattr(config, 'DIALOGUE_SUMMARY_LENGTH_ACTION', 200)
    summary_text_for_ethics = summarize_text(user_input if awareness_metrics.get("active_llm_fallback") else brain_response_text, summary_len_short)
    action_text_for_ethics = summarize_text(brain_response_text, summary_len_action) # Brain's response as the "action".
    try:
        if _ETHICS_SCORE_ETHICS_AVAILABLE: # Check if `score_ethics` function is available.
            calculated_score = score_ethics(awareness_metrics, concept_summary=summary_text_for_ethics, action_description=action_text_for_ethics)
            # Validate and use the returned score.
            if isinstance(calculated_score, (int, float)) and 0.0 <= calculated_score <= 1.0: ethical_score_value = float(calculated_score)
            else: _log_dialogue_event("ethics_score_invalid_type", {"score_received": calculated_score, "type_received": type(calculated_score).__name__}, level="warning")
            thought_steps.append(f"Dialogue Manager: Ethical score calculated: {ethical_score_value:.2f}")
        else: thought_steps.append("Dialogue Manager: Ethical scoring function unavailable. Using default ethical score.")
    except Exception as e_ethics_scoring_call:
        _log_dialogue_event("ethics_score_error", {"error_message": str(e_ethics_scoring_call), "traceback": traceback.format_exc()}, level="error")
        thought_steps.append(f"Dialogue Manager: Error during ethical scoring: {str(e_ethics_scoring_call)}")

    # Step 5: Store a memory of the interaction.
    try:
        if _MEMORY_STORE_MEMORY_AVAILABLE: # Check if `store_memory` function is available.
            primary_coord_for_memory = awareness_metrics.get("primary_concept_coord")
            # Store memory only if valid coordinates are available (from SNN processing).
            if _IS_VALID_COORDINATE_AVAILABLE and is_valid_coordinate(primary_coord_for_memory):
                concept_name_max_len = getattr(config, 'MAX_CONCEPT_NAME_FOR_MEMORY_LEN', 30)
                default_concept_name = getattr(config, 'DEFAULT_MEMORY_CONCEPT_NAME', "interaction_summary")
                concept_name_for_memory_storage = summarize_text(user_input, concept_name_max_len).replace("...", "").strip() or default_concept_name
                
                intensity_for_memory_storage = awareness_metrics.get("raw_t_intensity", 0.0) # Raw T-intensity (0-1).
                
                memory_summary_max_len = getattr(config, 'MAX_MEMORY_SUMMARY_LEN', 200)
                # Prepare data packet for memory storage.
                memory_data_packet = {
                    "concept_name": concept_name_for_memory_storage, "concept_coord": primary_coord_for_memory,
                    "summary": summarize_text(f"User: {user_input} | Sophia: {brain_response_text}", memory_summary_max_len), # Interaction summary.
                    "intensity": float(intensity_for_memory_storage) if isinstance(intensity_for_memory_storage, (int,float)) else 0.0,
                    "ethical_alignment": ethical_score_value, # Store with calculated ethical score.
                }
                memory_entry_id = store_memory(**memory_data_packet) # type: ignore # Store the memory.
                _log_dialogue_event("memory_store_attempt", {"entry_id": memory_entry_id if memory_entry_id else "Store_Failed", "concept_name": concept_name_for_memory_storage})
                thought_steps.append(f"Dialogue Manager: Memory storage attempted for '{concept_name_for_memory_storage}'. Entry ID: {memory_entry_id}")
            else: thought_steps.append("Dialogue Manager: Primary concept coordinates not valid for memory storage.")
        else: thought_steps.append("Dialogue Manager: Memory storage function unavailable.")
    except Exception as e_memory_storage_call:
        _log_dialogue_event("memory_store_error", {"error_message": str(e_memory_storage_call), "traceback": traceback.format_exc()}, level="error")
        thought_steps.append(f"Dialogue Manager: Error during memory storage: {str(e_memory_storage_call)}")

    # Step 6: Apply content mitigation/moderation if needed.
    # The final response includes persona mode and ethical score.
    current_persona_mode = getattr(persona_instance, 'mode', 'N/A').upper()
    final_response_text = f"[{current_persona_mode}|E:{ethical_score_value:.2f}] {brain_response_text}"
    mitigation_was_applied = False
    try:
        if _MITIGATOR_AVAILABLE and Mitigator: # Check if Mitigator class is available.
            # Get mitigation thresholds from config (or use defaults).
            mitigation_trigger_threshold = getattr(config, 'MITIGATION_ETHICAL_THRESHOLD', 0.3) if config else 0.3
            caution_threshold = getattr(config, 'ETHICAL_ALIGNMENT_THRESHOLD', 0.5) if config else 0.5
            
            if ethical_score_value < mitigation_trigger_threshold: # If score is below stricter mitigation threshold.
                mitigator_instance = Mitigator() # Create Mitigator instance.
                mitigated_text_content = mitigator_instance.moderate_ethically_flagged_content(brain_response_text, ethical_score_value, strict_mode=True)
                final_response_text = f"[{current_persona_mode}|E:{ethical_score_value:.2f}] [MITIGATED] {mitigated_text_content}"
                mitigation_was_applied = True
                thought_steps.append("Dialogue Manager: Content mitigation applied (strict threshold).")
            elif ethical_score_value < caution_threshold: # If score is below general alignment (caution) threshold.
                final_response_text = f"[{current_persona_mode}|E:{ethical_score_value:.2f}] [CAUTION] {brain_response_text}" 
                mitigation_was_applied = True # Technically a form of content presentation adjustment.
                thought_steps.append("Dialogue Manager: Caution applied to response presentation.")
        else: thought_steps.append("Dialogue Manager: Mitigation utilities (Mitigator class) not available.")
    except Exception as e_mitigation_application:
        _log_dialogue_event("mitigation_error", {"error_message": str(e_mitigation_application), "traceback": traceback.format_exc()}, level="error")
        thought_steps.append(f"Dialogue Manager: Error during content mitigation: {str(e_mitigation_application)}")

    # Step 7: Track ethical trends.
    try:
        if _ETHICS_TRACK_TRENDS_AVAILABLE: # Check if `track_trends` function is available.
            # Call track_trends with the current ethical score and context summary.
            # Note: The original `track_trends` signature might need adjustment if it doesn't take these args.
            # For this review, assuming it's compatible or adapted. If not, this call might error or be ineffective.
            # Based on ethics.py, track_trends() does not take arguments directly, it reads from its DB.
            # So, the score_ethics call which saves to DB is the input to track_trends.
            trends_summary_report = track_trends() # This will use the score just logged by score_ethics.
            _log_dialogue_event("ethics_trends_updated", {"summary_report": trends_summary_report if trends_summary_report else "No summary report"}, level="debug")
            thought_steps.append(f"Dialogue Manager: Ethical trends updated. Current trend: {trends_summary_report.get('trend_direction', 'N/A') if isinstance(trends_summary_report,dict) else 'N/A'}")
        else: thought_steps.append("Dialogue Manager: Ethical trend tracking function unavailable.")
    except Exception as e_ethics_trends_call:
        _log_dialogue_event("ethics_track_trends_error", {"error_message": str(e_ethics_trends_call), "traceback": traceback.format_exc()}, level="error")
        thought_steps.append(f"Dialogue Manager: Error tracking ethical trends: {str(e_ethics_trends_call)}")

    _log_dialogue_event("generate_response_end", {"final_response_snippet": final_response_text[:100], "mitigation_applied_flag": mitigation_was_applied})
    return (final_response_text, thought_steps, awareness_metrics) # Return final composed response.

def dialogue_loop(enable_streaming_thoughts: bool = None):
    _log_dialogue_event("dialogue_loop_start", {})
    persona_instance = get_dialogue_persona()
    if persona_instance is None:
    """
    Manages the main interactive command-line loop for Sophia_Alpha2.

    Handles user input, command parsing (e.g., !help, !stream),
    calls `generate_response` for actual dialogue, and prints outputs.
    The loop continues until the user types 'quit' or 'exit'.

    Args:
        enable_streaming_thoughts (bool, optional): Overrides the initial setting for
                                                    streaming thought steps. If None,
                                                    it defaults to `config.VERBOSE_OUTPUT`.
    """
    _log_dialogue_event("dialogue_loop_start", {})
    
    # Attempt to get the persona instance at the start of the loop.
    persona_instance = get_dialogue_persona()
    if persona_instance is None: # Critical: If persona can't be loaded, loop cannot run.
        print("CRITICAL: Persona module could not be initialized. Dialogue loop cannot start.", file=sys.stderr)
        _log_dialogue_event("dialogue_loop_critical_fail", {"reason": "Persona instance unavailable at loop start."}, level="critical")
        return

    print("Sophia_Alpha2 Dialogue Interface. Type '!help' for commands, 'quit' or 'exit' to end.")
    
    # Determine initial state for streaming thought steps.
    if enable_streaming_thoughts is not None:
        stream_thoughts_cli_active = enable_streaming_thoughts
    elif config: # If config is available, use VERBOSE_OUTPUT.
        stream_thoughts_cli_active = getattr(config, 'VERBOSE_OUTPUT', False)
    else: # Default to False if no config.
        stream_thoughts_cli_active = False
    print(f"Thought Streaming: {'ON' if stream_thoughts_cli_active else 'OFF'}. Type '!stream' to toggle.")

    # Main input loop.
    while True:
        try:
            # Refresh persona instance in case it was reset or changed externally (though unlikely in current design).
            current_persona = get_dialogue_persona()
            if current_persona is None: # If persona becomes unavailable mid-loop.
                print("CRITICAL: Persona became unavailable during loop. Exiting.", file=sys.stderr)
                _log_dialogue_event("dialogue_loop_persona_lost", {"reason": "Persona instance became None mid-loop."}, level="critical")
                break
            
            # Construct the command prompt string using current persona details.
            current_awareness_state = getattr(current_persona, 'awareness', {})
            prompt_persona_mode = getattr(current_persona, 'mode', 'N/A').upper()
            prompt_persona_name = getattr(current_persona, 'name', 'Sophia')
            prompt_persona_curiosity = current_awareness_state.get('curiosity', 0.0)
            prompt_persona_coherence = current_awareness_state.get('coherence', 0.0)
            command_prompt = f"{prompt_persona_name}({prompt_persona_mode}|A:{prompt_persona_curiosity:.1f},C:{prompt_persona_coherence:.1f})> "
            
            user_input_str = input(command_prompt) # Get user input.
        except KeyboardInterrupt: # Handle Ctrl+C gracefully.
            print("\nExiting dialogue loop (KeyboardInterrupt)...")
            _log_dialogue_event("dialogue_loop_ended_interrupt", {})
            break
        except EOFError: # Handle end-of-file (e.g., piped input).
            print("\nExiting dialogue loop (EOFError)...")
            _log_dialogue_event("dialogue_loop_ended_eof", {})
            break

        user_input_cleaned_lower = user_input_str.strip().lower() # Process for command checking.

        # Command Handling
        if user_input_cleaned_lower in ["quit", "exit"]: # Exit command.
            break
        elif user_input_cleaned_lower == "!stream": # Toggle thought streaming.
            stream_thoughts_cli_active = not stream_thoughts_cli_active
            print(f"Thought Streaming: {'ON' if stream_thoughts_cli_active else 'OFF'}")
        elif user_input_cleaned_lower == "!persona": # Display persona info.
            if current_persona and hasattr(current_persona, 'get_intro'):
                print(current_persona.get_intro()) # Display formatted intro.
                # Display raw awareness dict for debugging.
                print(json.dumps(getattr(current_persona, 'awareness', {}), indent=2, default=str))
            else: print("Persona details unavailable.")
        elif user_input_cleaned_lower == "!ethicsdb": # Debug: Display ethics DB summary.
            if _ETHICS_MODULE_AVAILABLE and ethics_module and hasattr(ethics_module, '_ethics_db'):
                # Display a summary or sample of the ethics DB.
                print(f"Ethics DB (first 5 score entries): {ethics_module._ethics_db.get('ethical_scores', [])[:5]}") 
                print(f"Total ethics score entries: {len(ethics_module._ethics_db.get('ethical_scores', []))}")
                print(f"Trend Analysis: {ethics_module._ethics_db.get('trend_analysis', {})}")
            else: print("Ethics DB details unavailable (module or _ethics_db not loaded).")
        elif user_input_cleaned_lower == "!memgraph": # Debug: Display memory graph summary.
            if _MEMORY_MODULE_AVAILABLE and memory_module and hasattr(memory_module, '_knowledge_graph'):
                kg_state = memory_module._knowledge_graph
                print(f"Memory Graph: Nodes={len(kg_state.get('nodes',[]))}, Edges={len(kg_state.get('edges',[]))}")
            else: print("Memory Graph details unavailable (module or _knowledge_graph not loaded).")
        elif user_input_cleaned_lower == "!library": # Debug: Display library summary.
            if _LIBRARY_AVAILABLE and library_module: # Check if library_module itself was imported.
                # KNOWLEDGE_LIBRARY is a global in library_module.
                print(f"Knowledge Library Entries: {len(library_module.KNOWLEDGE_LIBRARY)}")
            else: print("Knowledge Library details unavailable (module not loaded).")
        elif user_input_cleaned_lower == "!help": # Display help message.
            print("\nAvailable Commands:\n  !help          - Show this help message.\n  !stream        - Toggle streaming of thought steps.\n  !persona       - Display current persona information.\n  !ethicsdb      - Display summary of ethics database (debug).\n  !memgraph      - Display summary of memory graph (debug).\n  !library       - Display summary of knowledge library (debug).\n  quit / exit    - End the dialogue session.\n")
        else: # Not a command, process as dialogue input.
            if not user_input_str.strip(): continue # Skip empty input lines.
            print("Thinking...") # Indicate processing.
            try:
                # Generate response using the main orchestration function.
                final_response_text, thought_steps_log, _ = generate_response(user_input_str, stream_thought_steps=stream_thoughts_cli_active)
                print(f"\n{final_response_text}") # Print final response.
                # If streaming is active, print thought steps.
                if stream_thoughts_cli_active and thought_steps_log:
                    print("\n--- Thought Process ---")
                    for i, step_detail in enumerate(thought_steps_log): print(f"{i+1}. {step_detail}")
                    print("--- End of Thoughts ---\n")
            except CoreException as e_core_dialogue: # Handle known custom exceptions from core modules.
                _log_dialogue_event("dialogue_loop_core_exception", {"error_message": str(e_core_dialogue), "exception_type": type(e_core_dialogue).__name__}, level="error")
                print(f"A known system error occurred: ({type(e_core_dialogue).__name__}) {str(e_core_dialogue)}")
            except Exception as e_unexpected_dialogue: # Handle any other unexpected errors.
                _log_dialogue_event("dialogue_loop_unexpected_exception", {"error_message": str(e_unexpected_dialogue), "traceback": traceback.format_exc()}, level="critical")
                print(f"An unexpected critical error occurred: {str(e_unexpected_dialogue)}. Please check system logs for details.")
    
    print("Dialogue session ended.")
    _log_dialogue_event("dialogue_loop_end", {})

# --- Comprehensive Self-Testing Block ---
_IS_TEST_RUNNING = False 

if __name__ == "__main__":
    # This block allows the script to be run directly for testing,
    # ensuring that relative imports within the 'core' package can be resolved.
    if __package__ is None or __package__ == '':
        import sys
        from pathlib import Path
        # Add the parent directory of 'core' to sys.path
        # This assumes the script is in 'core/dialogue.py' and we want to add the directory containing 'core'
        parent_dir = Path(__file__).resolve().parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        # Try to re-evaluate __package__ or set it, though direct execution might still limit some package features.
        # For this script, modifying sys.path is often sufficient for imports.
        __package__ = "core" # Attempt to set package context for relative imports

    _IS_TEST_RUNNING = True 

    import unittest
    import unittest.mock as mock
    import io
    import contextlib # For redirect_stdout

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

            if config is None: # If global 'config' is not loaded.
                self.config_module_was_none = True
                class DummyConfig: pass # Create a simple placeholder class.
                config = DummyConfig() # Assign the dummy to the global 'config'.
            
            # Apply temporary configurations.
            for key, value in self.temp_configs.items():
                if hasattr(config, key): # If attribute exists.
                    self.original_values[key] = getattr(config, key)
                else: # Attribute does not exist, will be added temporarily.
                    self.original_values[key] = "__ATTR_NOT_SET__" # Sentinel.
                setattr(config, key, value) # Set the temporary value.
            return config # Return the modified config object.

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
            config = self.original_global_config

    TEST_DIALOGUE_SYSTEM_LOG = "test_dialogue_system_log.txt"
    TEST_PERSONA_PROFILE = "test_dialogue_persona_profile.json"
    # Add other test file paths if ethics/memory/library are to be tested for file creation
    TEST_ETHICS_DB = "test_dialogue_ethics.db"
    TEST_MEMORY_KG = "test_dialogue_knowledge_graph.json"
    TEST_LIBRARY_LOG = "test_dialogue_library_log.json"

    def delete_test_files():
        """
        Deletes predefined test files created during test execution.
        This helps ensure a clean state for subsequent test runs.
        Prints warnings if deletion fails.
        """
        files_to_delete = [
            TEST_DIALOGUE_SYSTEM_LOG, TEST_PERSONA_PROFILE, 
            TEST_ETHICS_DB, TEST_MEMORY_KG, TEST_LIBRARY_LOG
        ]
        for f_path in files_to_delete:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except OSError as e:
                    print(f"Warning: Could not delete test file {f_path}: {e}", file=sys.stderr)

    def setup_test_environment(test_specific_configs: dict = None) -> TempConfigOverride:
        """
        Prepares the testing environment for dialogue module tests.

        This involves:
        1.  Resetting global states of imported modules (KNOWLEDGE_LIBRARY, _knowledge_graph,
            _ethics_db, _dialogue_persona_instance) to ensure test isolation.
        2.  Deleting any existing test files (logs, DBs, profiles) created from previous runs.
        3.  Constructing a configuration dictionary for `TempConfigOverride`, including
            paths to test-specific files and default test settings.

        Args:
            test_specific_configs (dict, optional): A dictionary of configuration settings
                                                    specific to the current test, which will
                                                    override or augment the base test configs.
                                                    Defaults to None.

        Returns:
            TempConfigOverride: An instance of the context manager initialized with the
                                combined test configurations.
        """
        global _dialogue_persona_instance # Allow modification of the shared persona instance.
        
        # Reset relevant states in imported modules to ensure test isolation.
        # This is a simplified approach; a more robust test framework might handle
        # module reloading or more granular fixture management.
        if _LIBRARY_AVAILABLE and library_module: 
             library_module.KNOWLEDGE_LIBRARY = {} # Reset in-memory library.
             library_module._library_dirty_flag = False
        
        if _MEMORY_MODULE_AVAILABLE and memory_module:
            memory_module._knowledge_graph = {"nodes": [], "edges": []} # Reset in-memory graph.
        
        if _ETHICS_MODULE_AVAILABLE and ethics_module:
            ethics_module._ethics_db = {} # Reset in-memory ethics database.
        
        delete_test_files() # Clear out any files from previous test runs.
        _dialogue_persona_instance = None # Reset the shared persona instance for the dialogue module.

        # Define base configurations for testing.
        base_test_configs = {
            "SYSTEM_LOG_PATH": TEST_DIALOGUE_SYSTEM_LOG,
            "PERSONA_PROFILE_PATH": TEST_PERSONA_PROFILE,
            "ETHICS_DB_PATH": TEST_ETHICS_DB, 
            "KNOWLEDGE_GRAPH_PATH": TEST_MEMORY_KG, 
            "LIBRARY_LOG_PATH": TEST_LIBRARY_LOG, 
            "VERBOSE_OUTPUT": False, # Keep tests quieter by default.
            "LOG_LEVEL": "debug",    # Capture detailed logs during tests.
            "MITIGATION_ETHICAL_THRESHOLD": 0.4, # Example threshold for testing.
            "ETHICAL_ALIGNMENT_THRESHOLD": 0.6,  # Example threshold for testing.
            "REQUIRE_PUBLIC_STORAGE_CONSENT": False, # Simplify tests by disabling consent by default.
            # Mock implementation of ensure_path for config, as the real one might depend on project structure.
            "ensure_path": lambda path_to_ensure: os.makedirs(os.path.dirname(path_to_ensure), exist_ok=True) if os.path.dirname(path_to_ensure) and not os.path.exists(os.path.dirname(path_to_ensure)) else None,
        }
        # Merge with any test-specific configurations provided.
        if test_specific_configs:
            base_test_configs.update(test_specific_configs)
        
        return TempConfigOverride(base_test_configs)

    class TestDialogueModule(unittest.TestCase):
        """
        Unit tests for the core.dialogue module.
        Uses unittest.mock to isolate dependencies and simulate various scenarios.
        """
        def setUp(self):
            """
            Sets up the test environment before each test method.
            This involves resetting the shared dialogue persona instance and cleaning
            up any test files that might have been created.
            The `TempConfigOverride` is applied within each test method using a `with` statement
            to ensure configuration changes are scoped to that specific test.
            """
            global _dialogue_persona_instance
            _dialogue_persona_instance = None # Ensure a fresh persona state for each test.
            delete_test_files() # Clean test files before each test.

        @mock.patch('core.dialogue.track_trends')
        @mock.patch('core.dialogue.Mitigator')
        @mock.patch('core.dialogue.store_memory')
        @mock.patch('core.dialogue.score_ethics')
        @mock.patch('core.dialogue.Persona') 
        @mock.patch('core.dialogue.think')
        def test_generate_response_snn_success_high_ethics(self, mock_think, MockPersonaClass, mock_score_ethics, mock_store_memory, MockMitigatorClass, mock_track_trends):
            """
            Tests `generate_response` for a successful SNN processing path with high ethical score.
            Ensures no mitigation is applied and all components (think, persona, ethics, memory) are called.
            """
            test_configs_context = setup_test_environment() # Get the context manager instance.
            with test_configs_context as current_config: # Enter the context.
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
            """
            Tests `generate_response` when the ethical score falls into the "caution" range.
            Ensures the response is prefixed with "[CAUTION]" and no stricter mitigation is applied.
            """
            test_configs_context = setup_test_environment({"ETHICAL_ALIGNMENT_THRESHOLD": 0.6, "MITIGATION_ETHICAL_THRESHOLD": 0.3}) # Ensure threshold allows caution
            with test_configs_context as current_config:
                mock_persona_instance = MockPersonaClass.return_value
                mock_persona_instance.name="TestSophia"; mock_persona_instance.mode="TestMode"
                mock_think.return_value = ([], "Ethical caution response", {"active_llm_fallback": False})
                mock_score_ethics.return_value = 0.5 # Caution range (between 0.3 and 0.6 for this test's config)
                
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
            """
            Tests `generate_response` when the ethical score is low enough to trigger content mitigation.
            Ensures the Mitigator is called and the response indicates mitigation.
            """
            test_configs_context = setup_test_environment({"MITIGATION_ETHICAL_THRESHOLD": 0.4}) # Ensure mitigation triggers
            with test_configs_context as current_config:
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
            """
            Tests `generate_response` when the `think` function raises an exception.
            Ensures an error message is returned, LLM fallback is indicated, and the error is logged in awareness.
            """
            test_configs_context = setup_test_environment()
            with test_configs_context:
                mock_persona_instance = MockPersonaClass.return_value
                mock_persona_instance.name="TestSophia"; mock_persona_instance.mode="TestMode"
                mock_think.side_effect = Exception("Simulated brain error")
                
                response, _, awareness = generate_response("input causing brain error")
                self.assertIn("My apologies, I encountered an issue", response)
                self.assertTrue(awareness['active_llm_fallback'])
                self.assertEqual(awareness['snn_error'], "Simulated brain error")
                mock_persona_instance.update_awareness.assert_called_once() # Still called to update with error state

        @mock.patch('core.dialogue.get_dialogue_persona')
        def test_generate_response_persona_unavailable(self, mock_get_persona):
            """
            Tests `generate_response` when the persona instance cannot be retrieved.
            Ensures a critical error message is returned.
            """
            test_configs_context = setup_test_environment()
            with test_configs_context:
                mock_get_persona.return_value = None # Simulate persona being unavailable
                response, _, awareness = generate_response("any input")
                self.assertIn("Error: System persona is currently unavailable.", response)
                self.assertIn("Persona unavailable", awareness.get("error", ""))


        @mock.patch('builtins.input')
        @mock.patch('core.dialogue.generate_response') 
        def test_dialogue_loop_commands(self, mock_generate_response, mock_input):
            """
            Tests the `dialogue_loop` function's command handling capabilities.
            Simulates user input for commands like '!stream', '!persona', '!help',
            and verifies the expected output or behavior. Also tests regular input
            dispatching to `generate_response`.
            """
            # Capture print output to check console messages.
            captured_output = io.StringIO()
            sys.stdout = captured_output # Redirect stdout to the StringIO buffer.
            
            test_configs_context = setup_test_environment({"VERBOSE_OUTPUT": False}) # For predictable !stream toggle.
            with test_configs_context:
                # Mock persona for the loop to avoid dependency on actual Persona state.
                mock_persona_for_loop = mock.MagicMock(spec=Persona)
                mock_persona_for_loop.name = "LoopSophia"
                mock_persona_for_loop.mode = "LoopMode"
                mock_persona_for_loop.awareness = {"curiosity": 0.1, "coherence": 0.2}
                mock_persona_for_loop.get_intro.return_value = "LoopSophia Intro"

                # Patch get_dialogue_persona to return our mock.
                with mock.patch('core.dialogue.get_dialogue_persona', return_value=mock_persona_for_loop):
                    # Test !stream command (toggles twice).
                    mock_input.side_effect = ["!stream", "!stream", "quit"]
                    dialogue_loop()
                    output = captured_output.getvalue()
                    self.assertIn("Thought Streaming: ON", output)
                    self.assertIn("Thought Streaming: OFF", output) # Toggled back.

                    # Test !persona command.
                    captured_output.truncate(0); captured_output.seek(0) # Clear buffer for next check.
                    mock_input.side_effect = ["!persona", "quit"]
                    dialogue_loop()
                    self.assertIn("LoopSophia Intro", captured_output.getvalue())
                    
                    # Test !help command.
                    captured_output.truncate(0); captured_output.seek(0)
                    mock_input.side_effect = ["!help", "quit"]
                    dialogue_loop()
                    self.assertIn("Available Commands:", captured_output.getvalue())

                    # Test regular input calls generate_response.
                    captured_output.truncate(0); captured_output.seek(0)
                    mock_generate_response.return_value = ("Mocked response", ["mock thought"], {})
                    mock_input.side_effect = ["hello sophia", "quit"]
                    dialogue_loop(enable_streaming_thoughts=False) # Test with streaming off initially.
                    mock_generate_response.assert_called_with("hello sophia", stream_thought_steps=False)
                    self.assertIn("Mocked response", captured_output.getvalue())
            
            sys.stdout = sys.__stdout__ # Restore stdout to its original stream.

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
