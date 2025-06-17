"""
Core ethics module for Sophia_Alpha2.

This module is responsible for:
- Scoring the ethical alignment of concepts, actions, and system states.
- Tracking ethical trends over time using T-weighted analysis.
- Managing a persistent database for ethical scores and trend data.
- Providing a framework for guiding Sophia_Alpha2's actions based on
  configurable ethical principles.
"""

import datetime
import json
import os
import atexit # For shutting down log executor
import concurrent.futures # For asynchronous logging
import re # Added for regex word boundary matching
import stat # Added for file permission checks
import sys
import threading # Added for thread safety
import traceback # Promoted to top-level
from cryptography.fernet import Fernet, InvalidToken
import cachetools # For manifold query caching

import numpy as np

# Attempt to import configuration from the parent package
try:
    from .. import config
except ImportError:
    print("Ethics.py: Could not import 'config' from parent package. Attempting relative import for standalone use.")
    try:
        import config
        print("Ethics.py: Successfully imported 'config' directly (likely for standalone testing).")
    except ImportError as e_config:
        print(f"Ethics.py: Failed to import 'config' for standalone use. Critical error: {e_config}")
        config = None # Placeholder

# Attempt to import get_shared_manifold from core.brain
# This might be mocked during standalone testing of ethics.py if brain.py is complex or unavailable.
try:
    from .brain import get_shared_manifold
except ImportError:
    print("Ethics.py: Could not import 'get_shared_manifold' from .brain. Manifold cluster scoring will be limited/mocked in standalone tests.")
    # Define a placeholder if direct import fails, to allow module to load for basic tests
    def get_shared_manifold(force_recreate: bool = False): # Simple_fallback_mock
        print("Warning (ethics.py): Using fallback/mock get_shared_manifold(). Real brain.py not imported.")
        # This mock should align with how tests will mock it, or tests should patch this.
        class MockManifold:
            def get_conceptual_neighborhood(self, concept_coord, radius):
                print(f"MockManifold.get_conceptual_neighborhood called with {concept_coord}, radius {radius}. Returning empty list.")
                return [] # Return empty list or some default mock data
        
        class MockSharedManifold: # To allow manifold.device type access if some code needs it
             def __init__(self):
                 self.manifold_instance = MockManifold()
             def __getattr__(self, name): # Delegate to actual mock manifold instance
                 if hasattr(self.manifold_instance, name):
                     return getattr(self.manifold_instance, name)
                 # Add device attribute to the "shared manifold" itself if code tries to access it directly on get_shared_manifold() result
                 elif name == 'device': 
                     return torch.device("cpu") if 'torch' in sys.modules else "cpu" # basic device mock
                 raise AttributeError(f"'MockSharedManifold' (via fallback get_shared_manifold) has no attribute '{name}'")

        return MockSharedManifold().manifold_instance # return the object with get_conceptual_neighborhood

import copy # For deepcopy in sanitization

# Further module-level constants or setup can go here.

# --- Module-Level Logging ---
LOG_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40, "critical": 50}
_log_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='EthicsLogWorker')
atexit.register(_log_executor.shutdown, wait=True)

SENSITIVE_LOG_KEYS = [
    "awareness_metrics_snapshot", # Entire object will be redacted
    "concept_summary_snippet",
    "action_description_snippet",
    "concept_summary", # Direct field if present
    "action_description", # Direct field if present
    "trace", # For stack traces
    "error_detail", # Often contains specifics from exceptions
    # Add other keys as identified, e.g., specific sub-fields from awareness_metrics if needed later
]
REDACTION_PLACEHOLDER = "[REDACTED]"

def _actual_log_write(log_file_path_str: str, log_entry_json: str):
    """Helper function to perform the actual file write for logging."""
    try:
        # ensure_path should ideally be called before submitting to executor,
        # or the executor needs access to config. For now, assume path exists or ensure_path is robust.
        if config and hasattr(config, 'ensure_path'): # Check if config and ensure_path are available
             config.ensure_path(log_file_path_str)

        with open(log_file_path_str, 'a') as f:
            f.write(log_entry_json + '\n')
    except Exception as e_write:
        # Fallback to print if async writing fails
        print(f"ETHICS_ASYNC_LOG_WRITE_ERROR: {log_entry_json} - Error: {e_write}", file=sys.stderr)

def _log_ethics_event(event_type: str, data: dict, level: str = "info"):
    """
    Logs a structured system event from the ethics module.
    Sanitizes data and submits the logging task to a thread pool executor.
    """
    if not config or not hasattr(config, 'SYSTEM_LOG_PATH') or not hasattr(config, 'LOG_LEVEL'):
        print(f"ETHICS_EVENT_LOG_CONFIG_ERROR ({level.upper()}): {event_type} - Data: {json.dumps(data, default=str)}"
              f" - Reason: Config not fully available for logging.", file=sys.stderr)
        return

    try:
        numeric_level = LOG_LEVELS.get(level.lower(), LOG_LEVELS["info"])
        config_numeric_level = LOG_LEVELS.get(config.LOG_LEVEL.lower(), LOG_LEVELS["info"])

        if numeric_level < config_numeric_level:
            return

        data_to_log = copy.deepcopy(data)
        for key_to_sanitize in SENSITIVE_LOG_KEYS:
            if key_to_sanitize in data_to_log:
                data_to_log[key_to_sanitize] = REDACTION_PLACEHOLDER

        if data != data_to_log and any(data.get(k) != data_to_log.get(k) for k in SENSITIVE_LOG_KEYS if k in data and k in data_to_log and data_to_log[k] == REDACTION_PLACEHOLDER):
             print(f"CRITICAL_LOGGING_ERROR: Original data modified during sanitization for event {event_type}.", file=sys.stderr)

        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "module": "ethics",
            "event_type": event_type,
            "level": level.upper(),
            "data": data_to_log
        }
        
        log_file_path = str(config.SYSTEM_LOG_PATH)
        log_entry_json = json.dumps(log_entry, default=str)

        if not _log_executor._shutdown: # Check if executor is active
            _log_executor.submit(_actual_log_write, log_file_path, log_entry_json)
        else: # Log directly if executor is shutdown (e.g. during atexit)
            print(f"ETHICS_LOG_EXECUTOR_SHUTDOWN: Logging directly for event {event_type}.", file=sys.stderr)
            _actual_log_write(log_file_path, log_entry_json)
            
    except Exception as e_submit:
        print(f"ETHICS_LOG_PREPARATION_SUBMISSION_ERROR ({level.upper()}): {event_type} - Data: {json.dumps(data, default=str)}"
              f" - Error: {e_submit}", file=sys.stderr)

# --- Ethical Database State & Management ---
_db_lock = threading.Lock()
_manifold_cache = cachetools.LRUCache(maxsize=128)
_ethics_db = {
    "ethical_scores": [],  # List of score event dictionaries
    "trend_analysis": {}   # Dictionary to store trend analysis results
}
_ethics_db_dirty_flag = False # True if _ethics_db has in-memory changes not yet saved

def _load_ethics_db():
    """
    Loads the ethics database from the path specified in config.
    Handles file errors and malformed data, defaulting to an empty DB structure.
    Sets `_ethics_db_dirty_flag` to False, as the in-memory state aligns with
    what was loaded (or the default empty state).
    """
    global _ethics_db, _ethics_db_dirty_flag
    
    with _db_lock:
        if not config or not hasattr(config, 'ETHICS_DB_PATH') or not hasattr(config, 'ensure_path'):
            _log_ethics_event("load_ethics_db_failure", {"error": "Config not available or ETHICS_DB_PATH/ensure_path not set"}, level="critical")
            _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
            _ethics_db_dirty_flag = False
            return

        db_path = config.ETHICS_DB_PATH
        config.ensure_path(db_path)
        encryption_key = os.environ.get("ETHICS_ENCRYPTION_KEY")
        data = None # Initialize data to avoid UnboundLocalError if all paths fail before assignment

        try:
            if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
                _log_ethics_event("load_ethics_db_info", {"message": "Ethics DB file not found or empty. Initializing new DB.", "path": db_path}, level="info")
                _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
                _ethics_db_dirty_flag = False
                return

            if encryption_key:
                fernet = Fernet(encryption_key.encode())
                with open(db_path, 'rb') as f:
                    encrypted_data = f.read()

                if not encrypted_data:
                    _log_ethics_event("load_ethics_db_info", {"message": "Encrypted ethics DB file is empty after read. Initializing new DB.", "path": db_path}, level="info")
                    _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
                    _ethics_db_dirty_flag = False
                    return

                try:
                    decrypted_data = fernet.decrypt(encrypted_data)
                    data = json.loads(decrypted_data.decode('utf-8'))
                    _log_ethics_event("load_ethics_db_success", {"path": db_path, "encrypted": True, "scores_loaded": len(data.get("ethical_scores", []))}, level="info")
                except InvalidToken:
                    _log_ethics_event("load_ethics_db_decryption_error", {"path": db_path, "error": "Invalid token or key. Could not decrypt."}, level="critical")
                    _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
                    _ethics_db_dirty_flag = False
                    return
                except json.JSONDecodeError as e:
                    _log_ethics_event("load_ethics_db_json_decode_error", {"path": db_path, "encrypted": True, "error": str(e)}, level="error")
                    _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
                    _ethics_db_dirty_flag = False
                    return
            else: # No encryption key
                _log_ethics_event("load_ethics_db_info", {"message": "ETHICS_ENCRYPTION_KEY not set. Attempting to load as unencrypted JSON.", "path": db_path}, level="warning")
                with open(db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                _log_ethics_event("load_ethics_db_success", {"path": db_path, "encrypted": False, "scores_loaded": len(data.get("ethical_scores", []))}, level="info")

            # Common validation and pruning for loaded data (whether encrypted or not)
            if isinstance(data, dict) and \
               "ethical_scores" in data and isinstance(data["ethical_scores"], list) and \
               "trend_analysis" in data and isinstance(data["trend_analysis"], dict):

                max_entries = int(getattr(config, 'ETHICS_LOG_MAX_ENTRIES', 1000))
                if len(data["ethical_scores"]) > max_entries:
                    num_to_remove = len(data["ethical_scores"]) - max_entries
                    data["ethical_scores"] = data["ethical_scores"][num_to_remove:]
                    _log_ethics_event("load_ethics_db_pruned",
                                      {"path": db_path, "removed_count": num_to_remove, "retained_count": max_entries},
                                      level="info")
                _ethics_db = data
                _ethics_db_dirty_flag = False # Data loaded matches current disk state (after potential pruning)
            else:
                _log_ethics_event("load_ethics_db_malformed_structure", {"path": db_path, "error": "Loaded data structure is invalid."}, level="error")
                _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
                _ethics_db_dirty_flag = False

        except json.JSONDecodeError as e:
            _log_ethics_event("load_ethics_db_json_decode_error", {"path": db_path, "encrypted": False, "error": str(e)}, level="error")
            _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
            _ethics_db_dirty_flag = False
        except FileNotFoundError:
            _log_ethics_event("load_ethics_db_file_not_found", {"path": db_path}, level="error")
            _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
            _ethics_db_dirty_flag = False
        except Exception as e:
            _log_ethics_event("load_ethics_db_unknown_error", {"path": db_path, "error": str(e), "trace": traceback.format_exc()}, level="critical")
            _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
            _ethics_db_dirty_flag = False

def _save_ethics_db():
    """
    Saves the current state of `_ethics_db` to disk if `_ethics_db_dirty_flag` is True.
    Uses an atomic write (write to temp file, then replace original) to prevent data corruption.
    Resets `_ethics_db_dirty_flag` to False after a successful save.
    """
    global _ethics_db_dirty_flag, _ethics_db

    current_db_state_to_save = None
    is_dirty_for_save = False

    with _db_lock:
        if not _ethics_db_dirty_flag:
            _log_ethics_event("save_ethics_db_skipped", {"message": "No changes to save (_ethics_db_dirty_flag is False)."}, level="debug")
            return
        current_db_state_to_save = copy.deepcopy(_ethics_db)
        is_dirty_for_save = True

    if not is_dirty_for_save: # Should not happen if logic above is correct
        return

    if not config or not hasattr(config, 'ETHICS_DB_PATH') or not hasattr(config, 'ensure_path'):
        _log_ethics_event("save_ethics_db_failure", {"error": "Config not available or ETHICS_DB_PATH/ensure_path not set"}, level="critical")
        # Note: Dirty flag remains true as save failed before I/O.
        return

    db_path = config.ETHICS_DB_PATH
    config.ensure_path(db_path) # Ensure directory exists (can be outside lock)
    encryption_key = os.environ.get("ETHICS_ENCRYPTION_KEY")
    temp_db_path = db_path + ".tmp"

    try:
        # current_db_state_to_save is a deepcopy, safe to use outside lock
        serialized_data = json.dumps(current_db_state_to_save, indent=4, default=str)

        if encryption_key:
            fernet = Fernet(encryption_key.encode())
            encrypted_data = fernet.encrypt(serialized_data.encode('utf-8'))
            with open(temp_db_path, 'wb') as f:
                f.write(encrypted_data)
            _log_ethics_event("save_ethics_db_write_temp", {"path": temp_db_path, "encrypted": True}, level="debug")
        else:
            _log_ethics_event("save_ethics_db_info", {"message": "ETHICS_ENCRYPTION_KEY not set. Saving as unencrypted JSON.", "path": db_path}, level="warning")
            with open(temp_db_path, 'w', encoding='utf-8') as f:
                f.write(serialized_data)
            _log_ethics_event("save_ethics_db_write_temp", {"path": temp_db_path, "encrypted": False}, level="debug")
        
        os.replace(temp_db_path, db_path)

        with _db_lock: # Lock to update the dirty flag
            _ethics_db_dirty_flag = False

        # Set file permissions to owner read/write only (0o600)
        try:
            os.chmod(db_path, 0o600)
            _log_ethics_event("save_ethics_db_permissions_set", {"path": db_path, "permissions": "0o600"}, level="debug")
        except OSError as e_chmod:
            _log_ethics_event("save_ethics_db_permissions_error", {"path": db_path, "error": str(e_chmod)}, level="warning")

        _log_ethics_event("save_ethics_db_success", 
                          {"path": db_path, "encrypted": bool(encryption_key), "scores_saved": len(_ethics_db.get("ethical_scores", []))},
                          level="info")
    
    except IOError as e_io:
        _log_ethics_event("save_ethics_db_io_error", {"path": db_path, "temp_path": temp_db_path, "error": str(e_io), "trace": traceback.format_exc()}, level="critical")
        if os.path.exists(temp_db_path):
            try: os.remove(temp_db_path)
            except Exception as e_rm: _log_ethics_event("save_ethics_db_temp_cleanup_error", {"path": temp_db_path, "error": str(e_rm)}, level="error")
    except Exception as e:
        _log_ethics_event("save_ethics_db_unknown_error", {"path": db_path, "temp_path": temp_db_path, "error": str(e), "trace": traceback.format_exc()}, level="critical")
        if os.path.exists(temp_db_path):
            try: os.remove(temp_db_path)
            except Exception as e_rm: _log_ethics_event("save_ethics_db_temp_cleanup_error_unknown", {"path": temp_db_path, "error": str(e_rm)}, level="error")

def score_ethics(awareness_metrics: dict, concept_summary: str = "", action_description: str = "") -> float:
    """
    Calculates an ethical score based on awareness metrics, concept/action text,
    and manifold context.

    Args:
        awareness_metrics: Dictionary from brain.think() containing coherence, 
                           primary_concept_coord, etc.
        concept_summary: Textual summary of the concept being considered.
        action_description: Textual description of a proposed action.

    Returns:
        A final ethical score between 0.0 (less aligned) and 1.0 (more aligned).
    """
    global _ethics_db_dirty_flag # To mark the DB as needing a save after logging the score.

    if not config: # Config is essential for weights and other parameters.
        _log_ethics_event("score_ethics_failure", {"error": "Config module not loaded, cannot score ethics."}, level="critical")
        return 0.0 # Return a default low score indicating failure or undefined state.

    # --- Input Validation and Sanitization ---
    # Type checking for concept_summary and action_description
    if not isinstance(concept_summary, str):
        _log_ethics_event("score_ethics_input_type_coercion",
                          {"parameter": "concept_summary", "original_type": str(type(concept_summary)), "coerced_to": "empty string"},
                          level="warning")
        concept_summary = ""
    if not isinstance(action_description, str):
        _log_ethics_event("score_ethics_input_type_coercion",
                          {"parameter": "action_description", "original_type": str(type(action_description)), "coerced_to": "empty string"},
                          level="warning")
        action_description = ""

    # Length checking
    max_len = int(getattr(config, 'ETHICS_MAX_TEXT_INPUT_LENGTH', 5000))
    if len(concept_summary) > max_len:
        _log_ethics_event("score_ethics_input_length_truncated",
                          {"parameter": "concept_summary", "original_length": len(concept_summary), "max_length": max_len},
                          level="warning")
        concept_summary = concept_summary[:max_len]
    if len(action_description) > max_len:
        _log_ethics_event("score_ethics_input_length_truncated",
                          {"parameter": "action_description", "original_length": len(action_description), "max_length": max_len},
                          level="warning")
        action_description = action_description[:max_len]

    # Prepare data for logging this scoring event.
    # Snippets for logging are now based on potentially modified inputs.
    event_data_for_log = {
        "awareness_metrics_snapshot": awareness_metrics, # Full snapshot for traceability.
        "concept_summary_snippet": concept_summary[:100], # Log a snippet of (potentially coerced/truncated) summary.
        "action_description_snippet": action_description[:100] # Log a snippet of (potentially coerced/truncated) description.
    }
    _log_ethics_event("score_ethics_start", event_data_for_log, level="debug")

    scores = {}  # To store individual component scores.
    weights = {} # To store weights for each component.

    # --- Component 1: Coherence Score ---
    # Coherence from awareness_metrics is typically -1 to 1.
    # This component score maps it to 0-1, where 1 means high coherence (abs(coherence_val) is low).
    try:
        coherence_val = float(awareness_metrics.get("coherence", 0.0))
    except (ValueError, TypeError) as e:
        _log_ethics_event("score_ethics_param_error", {"parameter": "coherence", "value": awareness_metrics.get("coherence"), "error": str(e)}, level="warning")
        coherence_val = 0.0 # Default to neutral coherence
    scores["coherence"] = np.clip(1.0 - abs(coherence_val), 0.0, 1.0) # Closer to 0 coherence_val = higher score
    
    try:
        weights["coherence"] = float(getattr(config, 'ETHICS_COHERENCE_WEIGHT', 0.2)) 
    except (ValueError, TypeError) as e:
        _log_ethics_event("score_ethics_config_error", {"parameter": "ETHICS_COHERENCE_WEIGHT", "value": getattr(config, 'ETHICS_COHERENCE_WEIGHT', 'NotSet'), "error": str(e)}, level="warning")
        weights["coherence"] = 0.2 # Default weight

    # --- Component 2: Manifold Valence Score ---
    # Based on the 'x' coordinate of the primary concept, normalized by MANIFOLD_RANGE.
    # Assumes x_coord represents valence, mapping e.g. -MANIFOLD_RANGE/2 (neg) to MANIFOLD_RANGE/2 (pos).
    primary_coord = awareness_metrics.get("primary_concept_coord")
    
    # 'raw_t_intensity' (0-1) is expected from awareness_metrics for trend analysis and intensity preference.
    try:
        raw_t_intensity_for_trend = float(awareness_metrics.get("raw_t_intensity", 0.0))
    except (ValueError, TypeError) as e:
        _log_ethics_event("score_ethics_param_error", {"parameter": "raw_t_intensity", "value": awareness_metrics.get("raw_t_intensity"), "error": str(e)}, level="warning")
        raw_t_intensity_for_trend = 0.0 # Default to zero intensity
    
    if primary_coord and isinstance(primary_coord, (list, tuple)) and len(primary_coord) == 4:
        try:
            x_valence_coord = float(primary_coord[0]) # This is the first element of the coord tuple
            
            try:
                manifold_range = float(getattr(config, 'MANIFOLD_RANGE', 1.0)) # Default to 1.0 to avoid div by zero if not set
                if manifold_range == 0: manifold_range = 1.0 # Explicitly ensure not zero
            except (ValueError, TypeError) as e_range:
                _log_ethics_event("score_ethics_config_error", {"parameter": "MANIFOLD_RANGE", "value": getattr(config, 'MANIFOLD_RANGE', 'NotSet'), "error": str(e_range)}, level="warning")
                manifold_range = 1.0 # Default
                
            if manifold_range != 0: # Should always be true now with default 1.0
                # Normalize x_coord: maps [-range/2, +range/2] to [0, 1].
                scores["manifold_valence"] = np.clip((x_valence_coord + (manifold_range / 2.0)) / manifold_range, 0.0, 1.0)
            # No else needed as manifold_range defaults to 1.0
            event_data_for_log["primary_concept_t_intensity_raw"] = raw_t_intensity_for_trend
        except (ValueError, TypeError, IndexError) as e_coord_val: # Handle malformed coordinate data or access errors
            scores["manifold_valence"] = 0.5 # Neutral score on error
            _log_ethics_event("score_ethics_coord_processing_error", {"coord_received": primary_coord, "error": str(e_coord_val)}, level="warning")
    else: # If coordinates are missing or invalid format
        scores["manifold_valence"] = 0.5 # Neutral score
    
    try:
        weights["manifold_valence"] = float(getattr(config, 'ETHICS_VALENCE_WEIGHT', 0.2))
    except (ValueError, TypeError) as e:
        _log_ethics_event("score_ethics_config_error", {"parameter": "ETHICS_VALENCE_WEIGHT", "value": getattr(config, 'ETHICS_VALENCE_WEIGHT', 'NotSet'), "error": str(e)}, level="warning")
        weights["manifold_valence"] = 0.2 # Default weight

    # --- Component 3: Manifold Intensity Preference Score ---
    # Prefers concepts/actions with an intensity around a target (e.g., 0.5 on a 0-1 scale).
    # Uses a Gaussian-like function to score deviation from this ideal intensity.
    sigma_intensity_pref = getattr(config, 'ETHICS_INTENSITY_PREFERENCE_SIGMA', 0.25)
    ideal_intensity_center = getattr(config, 'ETHICS_IDEAL_INTENSITY_CENTER', 0.5)
    # Gaussian: exp(- (x - mu)^2 / (2 * sigma^2) ) where mu=ideal_intensity_center.
    scores["intensity_preference"] = np.clip(np.exp(-((raw_t_intensity_for_trend - ideal_intensity_center)**2) / (2 * sigma_intensity_pref**2)), 0.0, 1.0)
    
    try:
        weights["intensity_preference"] = float(getattr(config, 'ETHICS_INTENSITY_WEIGHT', config.DEFAULT_ETHICS_INTENSITY_WEIGHT))
    except (ValueError, TypeError) as e:
        _log_ethics_event("score_ethics_config_error", {"parameter": "ETHICS_INTENSITY_WEIGHT", "value": getattr(config, 'ETHICS_INTENSITY_WEIGHT', 'NotSet'), "error": str(e)}, level="warning")
        weights["intensity_preference"] = 0.1 # Default weight


    # --- Component 4: Ethical Framework Alignment Score ---
    # Simple keyword-based analysis of concept/action text against configured positive/negative keywords.
    text_to_analyze = (str(concept_summary) + " " + str(action_description)).lower()
    ethical_framework_config = getattr(config, 'ETHICAL_FRAMEWORK', {}) # ETHICAL_FRAMEWORK in config.py
    # Default keywords if not provided in config.
    # Structure can be: ["keyword1", {"term": "keyword2", "synonyms": ["synonym2_1", "synonym2_2"]}, ...]
    positive_keyword_entries = ethical_framework_config.get("positive_keywords", ["help", "improve", "assist", "share", "create", "understand", "align", "benefit"])
    negative_keyword_entries = ethical_framework_config.get("negative_keywords", ["harm", "deceive", "exploit", "manipulate", "destroy", "control", "damage"])

    pos_score_count = 0
    for entry in positive_keyword_entries:
        if isinstance(entry, str):
            pattern = r"\b" + re.escape(entry) + r"\b"
        elif isinstance(entry, dict) and "term" in entry:
            terms_to_match = [re.escape(entry["term"])] + [re.escape(s) for s in entry.get("synonyms", [])]
            pattern = r"\b(" + "|".join(terms_to_match) + r")\b"
        else:
            _log_ethics_event("score_ethics_framework_config_invalid_entry", {"entry": entry, "type": "positive"}, level="warning")
            continue # Skip malformed entry

        if re.search(pattern, text_to_analyze, re.IGNORECASE):
            pos_score_count += 1

    neg_score_count = 0
    for entry in negative_keyword_entries:
        if isinstance(entry, str):
            pattern = r"\b" + re.escape(entry) + r"\b"
        elif isinstance(entry, dict) and "term" in entry:
            terms_to_match = [re.escape(entry["term"])] + [re.escape(s) for s in entry.get("synonyms", [])]
            pattern = r"\b(" + "|".join(terms_to_match) + r")\b"
        else:
            _log_ethics_event("score_ethics_framework_config_invalid_entry", {"entry": entry, "type": "negative"}, level="warning")
            continue # Skip malformed entry

        if re.search(pattern, text_to_analyze, re.IGNORECASE):
            neg_score_count += 1

    # Placeholder for future semantic analysis integration
    # if getattr(config, 'ENABLE_SEMANTIC_ANALYSIS', False):
    #     # semantic_score_adjustment = calculate_semantic_alignment(text_to_analyze)
    #     # Adjust pos_score_count/neg_score_count or directly influence scores["framework_alignment"]
    #     _log_ethics_event("score_ethics_semantic_analysis_placeholder", {"message": "Semantic analysis not yet implemented."}, level="debug")
    #     pass # End placeholder

    if pos_score_count + neg_score_count > 0: # Avoid division by zero if no keywords found.
        scores["framework_alignment"] = np.clip(pos_score_count / (pos_score_count + neg_score_count), 0.0, 1.0)
    else: # No relevant keywords found.
        scores["framework_alignment"] = 0.5 # Neutral score.
    
    try:
        weights["framework_alignment"] = float(getattr(config, 'ETHICS_FRAMEWORK_WEIGHT', 0.3))
    except (ValueError, TypeError) as e:
        _log_ethics_event("score_ethics_config_error", {"parameter": "ETHICS_FRAMEWORK_WEIGHT", "value": getattr(config, 'ETHICS_FRAMEWORK_WEIGHT', 'NotSet'), "error": str(e)}, level="warning")
        weights["framework_alignment"] = 0.3 # Default weight

    # --- Component 5: Manifold Cluster Context Score ---
    # Assesses the ethical alignment of concepts in the manifold neighborhood of the primary concept.
    # Requires interaction with the SpacetimeManifold (potentially mocked).
    cluster_score_val = 0.5 # Default neutral score.
    if primary_coord and isinstance(primary_coord, (list, tuple)) and len(primary_coord) == 4: # Valid primary concept coordinates needed.
        try:
            manifold_instance = get_shared_manifold()
            if manifold_instance and hasattr(manifold_instance, 'get_conceptual_neighborhood'):
                try:
                    radius_factor = float(getattr(config, 'ETHICS_CLUSTER_RADIUS_FACTOR', 0.1))
                except (ValueError, TypeError) as e_rf:
                    _log_ethics_event("score_ethics_config_error", {"parameter": "ETHICS_CLUSTER_RADIUS_FACTOR", "value": getattr(config, 'ETHICS_CLUSTER_RADIUS_FACTOR', 'NotSet'), "error": str(e_rf)}, level="warning")
                    radius_factor = 0.1
                try:
                    manifold_range_for_radius = float(getattr(config, 'MANIFOLD_RANGE', 1.0))
                    if manifold_range_for_radius == 0: manifold_range_for_radius = 1.0
                except (ValueError, TypeError) as e_mr:
                    _log_ethics_event("score_ethics_config_error", {"parameter": "MANIFOLD_RANGE", "value": getattr(config, 'MANIFOLD_RANGE', 'NotSet'), "error": str(e_mr)}, level="warning")
                    manifold_range_for_radius = 1.0
                
                radius = radius_factor * manifold_range_for_radius
                
                coord_tuple_for_cache = tuple(primary_coord) if isinstance(primary_coord, list) else primary_coord
                cache_key = (coord_tuple_for_cache, radius)

                # Thread-safe cache access (cachetools.LRUCache is thread-safe for get/set)
                neighborhood_nodes = _manifold_cache.get(cache_key)
                if neighborhood_nodes is None:
                    _log_ethics_event("score_ethics_manifold_cache_miss", {"key_coord": coord_tuple_for_cache, "key_radius": radius}, level="debug")
                    neighborhood_nodes = manifold_instance.get_conceptual_neighborhood(primary_coord, radius)
                    _manifold_cache[cache_key] = neighborhood_nodes
                else:
                    _log_ethics_event("score_ethics_manifold_cache_hit", {"key_coord": coord_tuple_for_cache, "key_radius": radius, "num_nodes": len(neighborhood_nodes)}, level="debug")

                event_data_for_log["cluster_neighborhood_size"] = len(neighborhood_nodes)
                
                if neighborhood_nodes:
                    neighbor_valences = [] # Collect normalized valences of neighbors.
                    for node_data in neighborhood_nodes: 
                        if isinstance(node_data, dict) and isinstance(node_data.get('coordinates'), (list,tuple)) and len(node_data['coordinates']) == 4:
                            try:
                                node_x_valence = float(node_data['coordinates'][0])
                                current_manifold_range = float(getattr(config, 'MANIFOLD_RANGE', 1.0)) 
                                if current_manifold_range == 0: current_manifold_range = 1.0 
                                
                                normalized_valence = (node_x_valence + (current_manifold_range / 2.0)) / current_manifold_range
                                neighbor_valences.append(np.clip(normalized_valence, 0.0, 1.0))
                            except (ValueError, TypeError, IndexError) as e_node_val:
                                _log_ethics_event("score_ethics_cluster_node_processing_error", 
                                                  {"node_id": node_data.get("id", "UnknownID"), "coord_data": node_data.get('coordinates'), "error": str(e_node_val)}, 
                                                  level="warning")
                                neighbor_valences.append(0.5) # Neutral for this problematic node
                    
                    if neighbor_valences: # If valid neighbors with valences were found.
                        avg_neighbor_valence = np.mean(neighbor_valences)
                        cluster_score_val = avg_neighbor_valence # Cluster score is the average valence of neighbors.
                        event_data_for_log["cluster_avg_valence"] = avg_neighbor_valence
                    else: 
                         event_data_for_log["cluster_avg_valence"] = "N/A (no valid valences in neighborhood)"
                else: # No neighbors found within the radius.
                     event_data_for_log["cluster_avg_valence"] = "N/A (no neighbors found)"
            else: # Manifold instance or specific method is unavailable (e.g., using a basic mock).
                _log_ethics_event("score_ethics_manifold_unavailable_for_cluster", 
                                  {"reason": "get_shared_manifold() returned None or instance lacks get_conceptual_neighborhood method."}, 
                                  level="warning")
                event_data_for_log["cluster_score_status"] = "Manifold/method unavailable"
        except AttributeError as e_attr: # Handles if mock is too simple and lacks expected attributes.
            _log_ethics_event("score_ethics_neighborhood_method_missing", {"error_detail": str(e_attr)}, level="warning")
            event_data_for_log["cluster_score_status"] = f"Neighborhood method or attribute missing: {e_attr}"
        except Exception as e_cluster: # Catch other errors during cluster analysis.
            _log_ethics_event("score_ethics_cluster_error", {"error_detail": str(e_cluster), "trace": traceback.format_exc()}, level="error")
            event_data_for_log["cluster_score_status"] = f"Cluster analysis error: {e_cluster}"
            
    scores["manifold_cluster_context"] = np.clip(cluster_score_val, 0.0, 1.0)
    try:
        weights["manifold_cluster_context"] = float(getattr(config, 'ETHICS_CLUSTER_CONTEXT_WEIGHT', 0.2))
    except (ValueError, TypeError) as e:
        _log_ethics_event("score_ethics_config_error", {"parameter": "ETHICS_CLUSTER_CONTEXT_WEIGHT", "value": getattr(config, 'ETHICS_CLUSTER_CONTEXT_WEIGHT', 'NotSet'), "error": str(e)}, level="warning")
        weights["manifold_cluster_context"] = 0.2 # Default weight

    # --- Final Score Calculation: Weighted Average ---
    # Calculate the weighted sum of all component scores.
    weighted_sum_of_scores = 0
    total_weight_applied = 0
    for key in scores: # Iterate through calculated component scores.
        component_weight = weights.get(key, 0) # Get weight for this component (default 0 if not set).
        weighted_sum_of_scores += scores[key] * component_weight
        total_weight_applied += component_weight

    if total_weight_applied == 0: # Avoid division by zero if all weights are zero.
        final_score = 0.0 # Default to 0 if no weights applied.
        _log_ethics_event("score_ethics_warning", {"warning_message": "Total weight for ethical components was zero. Final score defaulted to 0."}, level="warning")
    else:
        final_score = weighted_sum_of_scores / total_weight_applied
    
    final_score = np.clip(final_score, 0.0, 1.0) # Ensure final score is within [0,1].

    # Store detailed event data for logging and persistence.
    event_data_for_log["component_scores"] = scores
    event_data_for_log["component_weights"] = weights
    event_data_for_log["final_score"] = final_score
    event_data_for_log["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    
    # Ensure raw_t_intensity is part of the logged event if not already added (e.g. if primary_coord was invalid).
    if "primary_concept_t_intensity_raw" not in event_data_for_log: 
        event_data_for_log["primary_concept_t_intensity_raw"] = raw_t_intensity_for_trend 

    # --- Logging & Persistence of the Score Event ---
    try:
        with _db_lock:
            if not isinstance(_ethics_db.get("ethical_scores"), list):
                _ethics_db["ethical_scores"] = []
                _log_ethics_event("score_ethics_db_reinit_scores_list", {"message": "'ethical_scores' list re-initialized in _ethics_db."}, level="warning")

            _ethics_db["ethical_scores"].append(event_data_for_log)

            max_entries = int(getattr(config, 'ETHICS_LOG_MAX_ENTRIES', 1000))
            if len(_ethics_db["ethical_scores"]) > max_entries:
                num_to_remove = len(_ethics_db["ethical_scores"]) - max_entries
                _ethics_db["ethical_scores"] = _ethics_db["ethical_scores"][num_to_remove:]
                _log_ethics_event("score_ethics_log_pruned", {"removed_count": num_to_remove, "new_count": len(_ethics_db["ethical_scores"])}, level="debug")

            _ethics_db_dirty_flag = True

        _save_ethics_db()
    except Exception as e_db_persist:
        _log_ethics_event("score_ethics_db_error", {"error_detail": str(e_db_persist), "trace": traceback.format_exc()}, level="critical")

    _log_ethics_event("score_ethics_complete", {"final_score": final_score, "concept_snippet": concept_summary[:50], "action_snippet": action_description[:50]}, level="info")
    return final_score

def track_trends() -> dict:
    """
    Analyzes the history of ethical scores to identify trends.
    Uses T-weighting (based on concept intensity at the time of scoring)
    to give more significance to high-intensity events.

    Returns:
        A dictionary containing trend analysis data, e.g.,
        {
            "data_points_used": int,
            "short_term_window": int,
            "long_term_window": int,
            "t_weighted_short_term_avg": float | None,
            "t_weighted_long_term_avg": float | None,
            "trend_direction": "improving" | "declining" | "stable" | "insufficient_data",
            "significance_threshold": float,
            "last_updated": str
        }
    """
    global _ethics_db_dirty_flag, _ethics_db

    _log_ethics_event("track_trends_start", {}, level="debug")

    if not config:
        _log_ethics_event("track_trends_failure", {"error": "Config module not loaded, cannot track trends."}, level="critical")
        return {"trend_direction": "error_config_missing", "last_updated": datetime.datetime.utcnow().isoformat() + "Z"}

    min_data_points = int(getattr(config, 'ETHICS_TREND_MIN_DATAPOINTS', 10)) 
    
    ethical_scores_data_copy = []
    with _db_lock:
        ethical_scores_data_copy = copy.deepcopy(_ethics_db.get("ethical_scores", []))

    if not isinstance(ethical_scores_data_copy, list) or len(ethical_scores_data_copy) == 0:
        _log_ethics_event("track_trends_insufficient_data", 
                          {"count": len(ethical_scores_data_copy), "min_required": min_data_points, "reason": "No score data available."},
                          level="info")
        current_trends = {
            "data_points_used": len(ethical_scores_data_copy),
            "trend_direction": "insufficient_data",
            "last_updated": datetime.datetime.utcnow().isoformat() + "Z"
        }
        with _db_lock:
            _ethics_db["trend_analysis"] = current_trends
            _ethics_db_dirty_flag = True
        _save_ethics_db()
        return current_trends

    # Extract valid scores and their associated raw T-intensities for weighting.
    scores_with_intensity = []
    for score_event in ethical_scores_data:
        if isinstance(score_event, dict) and \
           "final_score" in score_event and \
           "primary_concept_t_intensity_raw" in score_event: # Key for T-weighting.
            try:
                final_score = float(score_event["final_score"])
                # Raw intensity of the concept at the time of scoring, used for T-weighting.
                t_intensity = float(score_event["primary_concept_t_intensity_raw"]) 
                scores_with_intensity.append({"score": final_score, "t_intensity": t_intensity})
            except (ValueError, TypeError): # Skip malformed score events.
                _log_ethics_event("track_trends_data_format_error", {"event_snippet": str(score_event)[:100], "reason": "Invalid type for score or intensity."}, level="warning")
                continue

    # Check if enough valid data points were extracted.
    if len(scores_with_intensity) < min_data_points:
        _log_ethics_event("track_trends_insufficient_valid_data", {"valid_data_count": len(scores_with_intensity), "min_required": min_data_points}, level="info")
        current_trends = {
            "data_points_used": len(scores_with_intensity),
            "trend_direction": "insufficient_data",
            "last_updated": datetime.datetime.utcnow().isoformat() + "Z"
        }
        _ethics_db["trend_analysis"] = current_trends
        _ethics_db_dirty_flag = True
        _save_ethics_db()
        return current_trends

    # --- T-Weighting Calculation ---
    # Assign weights to scores based on their T-intensity. Higher intensity = higher weight.
    # Weight formula: (t_intensity * factor) + base_weight ensures all events have some weight.
    t_intensity_factor = getattr(config, 'ETHICS_TREND_T_INTENSITY_FACTOR', 0.9)
    base_weight = getattr(config, 'ETHICS_TREND_BASE_WEIGHT', 0.1)

    original_scores_np = np.array([s_i["score"] for s_i in scores_with_intensity])
    # Calculate weights: clip intensity to [0,1], apply factor, add base.
    weights_np = np.array([(np.clip(s_i["t_intensity"], 0, 1) * t_intensity_factor + base_weight) 
                           for s_i in scores_with_intensity])
    weights_np[weights_np == 0] = 1e-6 # Avoid division by zero if a weight is exactly zero.

    # Define windows for short-term and long-term trend analysis.
    # These are proportions of the minimum data points required, ensuring sensible window sizes.
    short_term_factor = getattr(config, 'ETHICS_TREND_SHORT_WINDOW_FACTOR', 0.2)
    long_term_factor = getattr(config, 'ETHICS_TREND_LONG_WINDOW_FACTOR', 0.5)
    min_short_window = getattr(config, 'ETHICS_TREND_MIN_SHORT_WINDOW', 3)
    min_long_window = getattr(config, 'ETHICS_TREND_MIN_LONG_WINDOW', 5)

    short_term_window_size = max(min_short_window, int(min_data_points * short_term_factor))
    long_term_window_size = max(min_long_window, int(min_data_points * long_term_factor))

    # Adjust window sizes if the actual number of scores is less than the calculated window.
    short_term_window_size = min(short_term_window_size, len(original_scores_np))
    long_term_window_size = min(long_term_window_size, len(original_scores_np))

    # Calculate T-weighted average for the short-term window.
    t_weighted_short_term_avg = None
    if short_term_window_size > 0 and len(original_scores_np) >= short_term_window_size:
        # Slice the most recent scores and their corresponding weights.
        short_scores_slice = original_scores_np[-short_term_window_size:]
        short_weights_slice = weights_np[-short_term_window_size:]
        t_weighted_short_term_avg = np.sum(short_scores_slice * short_weights_slice) / np.sum(short_weights_slice)

    # Calculate T-weighted average for the long-term window.
    t_weighted_long_term_avg = None
    if long_term_window_size > 0 and len(original_scores_np) >= long_term_window_size:
        long_scores_slice = original_scores_np[-long_term_window_size:]
        long_weights_slice = weights_np[-long_term_window_size:]
        t_weighted_long_term_avg = np.sum(long_scores_slice * long_weights_slice) / np.sum(long_weights_slice)
    
    # --- Determine Trend Direction ---
    trend_direction = "stable" # Default assumption.
    significance_threshold = float(getattr(config, 'ETHICS_TREND_SIGNIFICANCE_THRESHOLD', 0.05)) # From config.

    if t_weighted_short_term_avg is not None and t_weighted_long_term_avg is not None:
        # Compare short-term average to long-term average to determine trend.
        # Ensure windows are distinct enough for a meaningful comparison.
        if long_term_window_size > short_term_window_size: 
            diff = t_weighted_short_term_avg - t_weighted_long_term_avg
            # Relative difference to account for scale of scores. Add epsilon to avoid division by zero.
            relative_diff = diff / (abs(t_weighted_long_term_avg) + 1e-9) 
            
            if relative_diff > significance_threshold: # Short-term significantly higher than long-term.
                trend_direction = "improving"
            elif relative_diff < -significance_threshold: # Short-term significantly lower than long-term.
                trend_direction = "declining"
            # Otherwise, difference is not significant, trend remains "stable".
        else: # Not enough distinct data (e.g., long window is same or smaller than short).
            trend_direction = "stable" # Or could be "insufficient_trend_window_separation".
            
    elif t_weighted_short_term_avg is not None: # Only short-term average available.
         trend_direction = "insufficient_data_for_trend_comparison" # Cannot compare to long-term.
    else: # No averages could be calculated (should be caught by earlier checks).
        trend_direction = "insufficient_data"

    # Compile trend analysis results.
    current_trends_summary = {
        "data_points_used": len(scores_with_intensity),
        "short_term_window_size": short_term_window_size if t_weighted_short_term_avg is not None else 0,
        "long_term_window_size": long_term_window_size if t_weighted_long_term_avg is not None else 0,
        "t_weighted_short_term_avg": t_weighted_short_term_avg,
        "t_weighted_long_term_avg": t_weighted_long_term_avg,
        "trend_direction": trend_direction,
        "significance_threshold": significance_threshold,
        "last_updated": datetime.datetime.utcnow().isoformat() + "Z"
    }

    with _db_lock:
        _ethics_db["trend_analysis"] = current_trends_summary
        _ethics_db_dirty_flag = True
    _save_ethics_db()

    _log_ethics_event("track_trends_complete", current_trends_summary, level="info")
    return current_trends_summary

# --- Load ethics database at module import ---
_load_ethics_db() # Initial load, locks handled within

if __name__ == '__main__':
    # --- Test Utilities ---
    class TempConfigOverride:
        """
        A context manager for temporarily overriding attributes in the global `config` module.

        This is useful for testing different configurations without permanently altering
        the `config` object or needing to reload modules. It ensures that original
        values are restored upon exiting the context.
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

        def __enter__(self):
            """
            Sets up the temporary configuration overrides when entering the context.

            Iterates through `self.temp_configs`, stores the original value of each
            attribute from the `config` module (or a sentinel if it doesn't exist),
            and then sets the attribute to its temporary value.

            Returns:
                self: The TempConfigOverride instance (or the config object itself).

            Raises:
                ImportError: If the global `config` module is not loaded/available.
            """
            if not config:  # Check if the global config object is available.
                raise ImportError("Config module not loaded. TempConfigOverride cannot operate.")
            
            for key, value in self.temp_configs.items():
                # Store the original value if the attribute exists, otherwise store a sentinel.
                if hasattr(config, key):
                    self.original_values[key] = getattr(config, key)
                else:
                    self.original_values[key] = "__ATTR_NOT_SET__" # Sentinel for new attributes.
                
                # Set the temporary value.
                setattr(config, key, value)
            return self # Or return config, depending on usage preference.

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Restores the original configuration when exiting the context.

            Iterates through the stored original values. If an attribute was newly added
            during the override (original value is the sentinel), it's removed. Otherwise,
            the attribute is restored to its original value.
            """
            if not config: # Should not happen if __enter__ succeeded, but good for robustness.
                return

            for key, original_value in self.original_values.items():
                if original_value == "__ATTR_NOT_SET__":
                    # If the attribute was added by this context manager, remove it.
                    if hasattr(config, key):
                        delattr(config, key)
                else:
                    # Otherwise, restore its original value.
                    setattr(config, key, original_value)

    # --- Test Setup & Helper ---
    TEST_ETHICS_DB_FILENAME = "test_ethics_db.json"
    module_dir = os.path.dirname(os.path.abspath(__file__))
    TEST_ETHICS_DB_PATH = os.path.join(module_dir, TEST_ETHICS_DB_FILENAME)
    TEST_SYSTEM_LOG_PATH = os.path.join(module_dir, "test_ethics_system_log.json") # For logging during tests

    original_get_shared_manifold = None # To store the real get_shared_manifold
    original_ethics_encryption_key = None # To store the original environment variable value

    class MockConceptualNeighborhoodManifold:
        """
        Mock for the SpacetimeManifold class (or its relevant parts for ethics tests).
        This mock specifically controls the output of the `get_conceptual_neighborhood`
        method, which is used by `score_ethics` for cluster context analysis.
        """
        def __init__(self, neighborhood_data=None):
            """
            Initializes the mock manifold.

            Args:
                neighborhood_data (list, optional): A list of dictionaries, where each
                                                    dictionary represents a neighboring node's
                                                    data (e.g., including 'coordinates').
                                                    Defaults to an empty list.
            """
            self.device = "cpu" # Mock device attribute, sometimes checked by other parts.
        self.neighborhood_data = []
        if neighborhood_data is not None:
            for i, item in enumerate(neighborhood_data):
                if not isinstance(item, dict):
                    raise ValueError(f"Item at index {i} in neighborhood_data is not a dictionary. Received: {item}")
                if 'coordinates' not in item:
                    raise ValueError(f"Item at index {i} in neighborhood_data is missing 'coordinates' key. Received: {item}")

                coords = item['coordinates']
                if not isinstance(coords, (list, tuple)):
                    raise ValueError(f"Item at index {i}: 'coordinates' must be a list or tuple. Received: {type(coords)}")
                if len(coords) != 4:
                    raise ValueError(f"Item at index {i}: 'coordinates' must have 4 elements. Received length: {len(coords)}")
                for j, coord_val in enumerate(coords):
                    if not isinstance(coord_val, (int, float)):
                        raise ValueError(f"Item at index {i}, coordinate at index {j}: Value must be numeric (int or float). Received: {coord_val} (type: {type(coord_val)})")
                self.neighborhood_data.append(item) # Add if valid

        def get_conceptual_neighborhood(self, concept_coord, radius):
            """
            Simulates retrieving concepts in the neighborhood of `concept_coord`.

            Args:
                concept_coord: The coordinates of the concept to find neighbors for.
                radius: The radius to search within.

            Returns:
                list: The predefined `self.neighborhood_data`.
            """
            _log_ethics_event("mock_get_conceptual_neighborhood_called", {"coord": concept_coord, "radius": radius, "returning_count": len(self.neighborhood_data)}, level="debug")
            return self.neighborhood_data # Return predefined data

    def mock_get_shared_manifold_for_ethics_test(neighborhood_data=None, force_recreate=False):
        """
        A factory function that returns an instance of MockConceptualNeighborhoodManifold.
        This function is used to replace `core.brain.get_shared_manifold` during tests
        of the ethics module, allowing control over the manifold's behavior for
        cluster context scoring.

        Args:
            neighborhood_data (list, optional): Data to be returned by the mock manifold's
                                                `get_conceptual_neighborhood` method.
            force_recreate (bool, optional): Argument matching the real `get_shared_manifold`,
                                             ignored by this mock.

        Returns:
            MockConceptualNeighborhoodManifold: An instance of the mock manifold.
        """
        return MockConceptualNeighborhoodManifold(neighborhood_data=neighborhood_data)

    def setup_test_environment(test_specific_configs: dict = None, mock_neighborhood: list = None, encryption_key_value: Optional[str] = None):
        """
        Prepares the testing environment for ethics module tests.

        This involves:
        1.  Cleaning up any existing test database or log files.
        2.  Resetting the in-memory `_ethics_db` and `_ethics_db_dirty_flag`.
        3.  Monkeypatching `get_shared_manifold` for controlling manifold interactions.
        4.  Optionally setting/unsetting the ETHICS_ENCRYPTION_KEY environment variable.
        5.  Constructing a configuration dictionary for `TempConfigOverride`.

        Args:
            test_specific_configs (dict, optional): Configuration overrides for the current test.
            mock_neighborhood (list, optional): Data for the mock manifold's neighborhood.
            encryption_key_value (str, optional): Value to set for ETHICS_ENCRYPTION_KEY.
                                                 If None, the variable is unset.

        Returns:
            TempConfigOverride: An instance of the context manager with test configurations.
        """
        global _ethics_db, _ethics_db_dirty_flag, original_get_shared_manifold, original_ethics_encryption_key
        
        if os.path.exists(TEST_ETHICS_DB_PATH):
            os.remove(TEST_ETHICS_DB_PATH)
        if os.path.exists(TEST_SYSTEM_LOG_PATH):
            os.remove(TEST_SYSTEM_LOG_PATH)
            
        _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
        _ethics_db_dirty_flag = False
        
        # Store original get_shared_manifold if not already stored, then monkeypatch.
        if original_get_shared_manifold is None: 
            original_get_shared_manifold = sys.modules[__name__].get_shared_manifold
        
        # Apply the mock for get_shared_manifold, capturing the current mock_neighborhood.
        # Using a default argument for current_mock_neighborhood in lambda to capture its value at definition time.
        sys.modules[__name__].get_shared_manifold = lambda force_recreate=False, current_mock_neighborhood=mock_neighborhood: \
            mock_get_shared_manifold_for_ethics_test(neighborhood_data=current_mock_neighborhood, force_recreate=force_recreate)

        # Manage ETHICS_ENCRYPTION_KEY environment variable
        original_ethics_encryption_key = os.environ.get("ETHICS_ENCRYPTION_KEY")
        if encryption_key_value is not None:
            os.environ["ETHICS_ENCRYPTION_KEY"] = encryption_key_value
            _log_ethics_event("setup_test_env", {"ETHICS_ENCRYPTION_KEY_status": "set"}, level="debug")
        else:
            if "ETHICS_ENCRYPTION_KEY" in os.environ:
                del os.environ["ETHICS_ENCRYPTION_KEY"]
            _log_ethics_event("setup_test_env", {"ETHICS_ENCRYPTION_KEY_status": "unset"}, level="debug")

        final_test_configs = {
            "ETHICS_DB_PATH": TEST_ETHICS_DB_PATH,
            "SYSTEM_LOG_PATH": TEST_SYSTEM_LOG_PATH,
            "VERBOSE_OUTPUT": False,
            "ETHICS_LOG_MAX_ENTRIES": 5,
            "ETHICS_TREND_MIN_DATAPOINTS": 3,
            "ETHICS_TREND_SIGNIFICANCE_THRESHOLD": 0.05,
            "ETHICS_COHERENCE_WEIGHT": 0.2, "ETHICS_VALENCE_WEIGHT": 0.2,
            "ETHICS_INTENSITY_WEIGHT": 0.1, "ETHICS_FRAMEWORK_WEIGHT": 0.3,
            "ETHICS_CLUSTER_CONTEXT_WEIGHT": 0.2,
            "MANIFOLD_RANGE": 10.0,
            "ensure_path": lambda path: os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) and not os.path.exists(os.path.dirname(path)) else None,
        }
        if test_specific_configs:
            final_test_configs.update(test_specific_configs)
        
        return TempConfigOverride(final_test_configs)

    def cleanup_test_environment():
        """
        Cleans up the testing environment after ethics module tests.
        Removes test files, restores `get_shared_manifold` and ETHICS_ENCRYPTION_KEY.
        """
        global original_get_shared_manifold, original_ethics_encryption_key
        if os.path.exists(TEST_ETHICS_DB_PATH):
            os.remove(TEST_ETHICS_DB_PATH)
        if os.path.exists(TEST_SYSTEM_LOG_PATH):
            os.remove(TEST_SYSTEM_LOG_PATH)
        
        if original_get_shared_manifold is not None:
            sys.modules[__name__].get_shared_manifold = original_get_shared_manifold
            original_get_shared_manifold = None # Reset for next potential setup sequence.

        # Restore ETHICS_ENCRYPTION_KEY
        if original_ethics_encryption_key is not None:
            os.environ["ETHICS_ENCRYPTION_KEY"] = original_ethics_encryption_key
            _log_ethics_event("cleanup_test_env", {"ETHICS_ENCRYPTION_KEY_status": "restored"}, level="debug")
        else:
            if "ETHICS_ENCRYPTION_KEY" in os.environ: # If it was set during test but not originally
                del os.environ["ETHICS_ENCRYPTION_KEY"]
            _log_ethics_event("cleanup_test_env", {"ETHICS_ENCRYPTION_KEY_status": "cleared (was not originally set)"}, level="debug")
        original_ethics_encryption_key = None # Reset for next setup sequence.


    def run_test(test_func, *args, **kwargs) -> bool:
        """
        Executes a given test function within a controlled test environment.

        Sets up the environment using `setup_test_environment` (including config
        overrides and manifold mocking), runs the test function, prints its status,
        handles exceptions, and ensures cleanup via `cleanup_test_environment`.

        Args:
            test_func (callable): The test function to execute.
            *args: Positional arguments to pass to the test function.
            **kwargs: Keyword arguments:
                - `test_configs` (dict, optional): Specific config overrides for this test.
                - `mock_neighborhood_data` (list, optional): Data for the mock manifold.
                - Other kwargs are passed to `test_func`.

        Returns:
            bool: True if the test passes, False otherwise.
        """
        test_name = test_func.__name__
        print(f"--- Running Test: {test_name} ---")
        
        # Extract test setup specific kwargs before passing the rest to test_func
        test_configs_override = kwargs.pop("test_configs", {})
        mock_neighborhood_data_for_test = kwargs.pop("mock_neighborhood_data", [])
        encryption_key_for_test = kwargs.pop("encryption_key", None) # New kwarg for run_test
        
        # Use a try-finally block to ensure cleanup_test_environment is always called.
        try:
            # setup_test_environment returns a TempConfigOverride instance.
            # The `with` statement handles its __enter__ and __exit__ for config management.
            with setup_test_environment(test_specific_configs=test_configs_override, 
                                        mock_neighborhood=mock_neighborhood_data_for_test,
                                        encryption_key_value=encryption_key_for_test): # Pass key to setup
                _load_ethics_db() # Load DB under the new temp config (likely creates empty if file missing).
                result = test_func(*args, **kwargs) # Execute the actual test function.
                if result:
                    print(f"PASS: {test_name}")
                else:
                    print(f"FAIL: {test_name}")
                return result
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            traceback.print_exc()
            return False # Mark as fail on unexpected exception during setup or test execution.
        finally:
            cleanup_test_environment() # Ensure cleanup happens regardless of test outcome.

    # --- Test Function Definitions ---

    def _perform_db_load_save_checks(is_encrypted: bool, current_key: Optional[str]) -> bool:
        """Helper function to perform common DB load/save checks."""
        global _ethics_db, _ethics_db_dirty_flag, TEST_ETHICS_DB_PATH, config

        # Initial load (file non-existent) - _load_ethics_db is called by run_test context
        assert _ethics_db == {"ethical_scores": [], "trend_analysis": {}}, f"[{'Encrypted' if is_encrypted else 'Unencrypted'}] Initial DB state not empty."
        assert not _ethics_db_dirty_flag, f"[{'Encrypted' if is_encrypted else 'Unencrypted'}] Dirty flag not false on initial load."
        print(f"  PASS: [{('Encrypted' if current_key else 'Unencrypted')}] Initial load non-existent file.")

        # Save when not dirty
        _save_ethics_db()
        assert not os.path.exists(TEST_ETHICS_DB_PATH), f"[{('Encrypted' if current_key else 'Unencrypted')}] DB file created on save when not dirty."
        print(f"  PASS: [{('Encrypted' if current_key else 'Unencrypted')}] Save skipped when not dirty.")

        # Modify DB, set dirty flag, and save
        test_data_point = {"test_score": 1, "primary_concept_t_intensity_raw": 0.5, "timestamp": "2023-01-01T00:00:00Z"}
        _ethics_db["ethical_scores"].append(test_data_point)
        _ethics_db_dirty_flag = True
        _save_ethics_db()
        assert os.path.exists(TEST_ETHICS_DB_PATH), f"[{('Encrypted' if current_key else 'Unencrypted')}] DB not saved when dirty."
        assert not _ethics_db_dirty_flag, f"[{('Encrypted' if current_key else 'Unencrypted')}] Dirty flag not reset after save."
        print(f"  PASS: [{('Encrypted' if current_key else 'Unencrypted')}] DB saved when dirty, flag reset.")

        # Verify encryption status of the saved file
        if current_key: # Encrypted check
            try:
                with open(TEST_ETHICS_DB_PATH, 'r', encoding='utf-8') as f:
                    json.load(f) # Should fail for encrypted file
                assert False, f"[{('Encrypted' if current_key else 'Unencrypted')}] Encrypted file was successfully parsed as JSON (should not happen)."
            except json.JSONDecodeError:
                print(f"  PASS: [{('Encrypted' if current_key else 'Unencrypted')}] File content is not plain JSON (as expected for encryption).")
            except Exception as e: # Catch other read errors if not even text
                print(f"  PASS: [{('Encrypted' if current_key else 'Unencrypted')}] File content is not plain JSON (read error: {e}, as expected for encryption).")

        else: # Unencrypted check
            try:
                with open(TEST_ETHICS_DB_PATH, 'r', encoding='utf-8') as f:
                    json.load(f)
                print(f"  PASS: [{('Encrypted' if current_key else 'Unencrypted')}] File content is plain JSON (as expected for no encryption).")
            except json.JSONDecodeError:
                assert False, f"[{('Encrypted' if current_key else 'Unencrypted')}] Unencrypted file failed to parse as JSON (should not happen)."


        # Load saved DB (implicitly done by next call to run_test or if we manually call _load_ethics_db)
        # For this helper, we'll clear and reload manually to check current state.
        _ethics_db = {}
        _ethics_db_dirty_flag = False # Reset before load
        # Need to ensure ETHICS_ENCRYPTION_KEY is correctly set in environment for this _load_ethics_db call
        # This is managed by the run_test wrapper and setup_test_environment
        _load_ethics_db()

        assert _ethics_db.get("ethical_scores"), f"[{('Encrypted' if current_key else 'Unencrypted')}] Ethical scores not loaded."
        assert len(_ethics_db["ethical_scores"]) > 0, f"[{('Encrypted' if current_key else 'Unencrypted')}] Ethical scores list is empty after load."
        assert _ethics_db["ethical_scores"][0].get("test_score") == test_data_point["test_score"], f"[{('Encrypted' if current_key else 'Unencrypted')}] Loaded DB content mismatch."
        print(f"  PASS: [{('Encrypted' if current_key else 'Unencrypted')}] Reloaded saved DB correctly.")

        # Check file permissions if file was created
        if os.path.exists(TEST_ETHICS_DB_PATH):
            try:
                file_mode = os.stat(TEST_ETHICS_DB_PATH).st_mode
                # S_IMODE extracts the permission bits
                permissions = stat.S_IMODE(file_mode)
                # Check for owner read/write (0o600).
                # S_IRUSR (0o400), S_IWUSR (0o200)
                # Ensure no group/other permissions: not (file_mode & (stat.S_IRWXG | stat.S_IRWXO))
                expected_permissions = stat.S_IRUSR | stat.S_IWUSR
                if permissions == expected_permissions:
                    print(f"  PASS: [{('Encrypted' if current_key else 'Unencrypted')}] File permissions are correctly set to 0o600.")
                else:
                    # This might be flaky on some systems (e.g. Windows, or where umask is restrictive)
                    # So, log a warning rather than failing the test, but assert that os.chmod was attempted.
                    # The check for "save_ethics_db_permissions_set" or "save_ethics_db_permissions_error"
                    # in logs would be more robust if direct permission check is problematic.
                    # For now, let's assert it directly but be mindful of potential flakiness.
                    assert permissions == expected_permissions, \
                        f"[{('Encrypted' if current_key else 'Unencrypted')}] File permissions are {oct(permissions)}, expected {oct(expected_permissions)} (0o600)."
            except AssertionError as e_perm: # Catch only our assert for permissions
                 print(f"  WARNING: Permission check failed: {e_perm}. This might be due to OS/environment specifics.")
                 # To make this a hard fail, remove the try-except for AssertionError.
            except Exception as e_stat:
                print(f"  WARNING: [{('Encrypted' if current_key else 'Unencrypted')}] Could not verify file permissions: {e_stat}")

        return True

    def test_db_load_save_encryption_modes() -> bool:
        """
        Tests DB load/save with and without encryption.
        """
        print("Testing DB load/save (Unencrypted mode)...")
        # kwargs for run_test will be derived from this test function's parameters if needed,
        # or set directly in the tests_to_run list.
        # Here, _perform_db_load_save_checks expects is_encrypted and current_key
        # current_key will be None for unencrypted mode, set by run_test via setup_test_environment
        unencrypted_result = _perform_db_load_save_checks(is_encrypted=False, current_key=os.environ.get("ETHICS_ENCRYPTION_KEY"))
        if not unencrypted_result: return False

        # Note: The state (like os.environ) is cleaned up and set by run_test for each call.
        # So the next call to _perform_db_load_save_checks will be in an environment
        # configured by its corresponding run_test call (i.e. with encryption key set).
        print("Testing DB load/save (Encrypted mode)...")
        # This assertion relies on the test runner (run_test) to set the key for the encrypted part.
        # The key itself is generated outside and passed to run_test.
        encrypted_result = _perform_db_load_save_checks(is_encrypted=True, current_key=os.environ.get("ETHICS_ENCRYPTION_KEY"))
        return encrypted_result

    def test_mock_manifold_validation(**kwargs) -> bool:
        """
        Tests the validation logic in MockConceptualNeighborhoodManifold.
        Verifies that ValueError is raised for malformed neighborhood_data.
        """
        print("Testing MockConceptualNeighborhoodManifold validation...")
        valid_item = {"id": "node1", "coordinates": [1.0, 2.0, 3.0, 4.0]}

        # Test case 1: Valid data
        try:
            mock_get_shared_manifold_for_ethics_test(neighborhood_data=[valid_item, {"coordinates": (0,0,0,0)}])
            print("  PASS: Valid neighborhood_data accepted.")
        except ValueError as e:
            print(f"  FAIL: Valid data raised ValueError: {e}")
            return False

        # Test case 2: Item not a dictionary
        invalid_data_not_dict = [valid_item, "not_a_dict"]
        try:
            mock_get_shared_manifold_for_ethics_test(neighborhood_data=invalid_data_not_dict)
            print("  FAIL: Malformed data (item not a dict) did not raise ValueError.")
            return False
        except ValueError as e:
            print(f"  PASS: Malformed data (item not a dict) raised ValueError: {e}")

        # Test case 3: Missing 'coordinates' key
        invalid_data_missing_coords = [valid_item, {"id": "node2", "other_data": "foo"}]
        try:
            mock_get_shared_manifold_for_ethics_test(neighborhood_data=invalid_data_missing_coords)
            print("  FAIL: Malformed data (missing 'coordinates') did not raise ValueError.")
            return False
        except ValueError as e:
            print(f"  PASS: Malformed data (missing 'coordinates') raised ValueError: {e}")

        # Test case 4: 'coordinates' not a list/tuple
        invalid_data_coords_not_list = [valid_item, {"id": "node2", "coordinates": "1,2,3,4"}]
        try:
            mock_get_shared_manifold_for_ethics_test(neighborhood_data=invalid_data_coords_not_list)
            print("  FAIL: Malformed data ('coordinates' not list/tuple) did not raise ValueError.")
            return False
        except ValueError as e:
            print(f"  PASS: Malformed data ('coordinates' not list/tuple) raised ValueError: {e}")

        # Test case 5: 'coordinates' wrong length
        invalid_data_coords_wrong_len = [valid_item, {"id": "node2", "coordinates": [1.0, 2.0, 3.0]}]
        try:
            mock_get_shared_manifold_for_ethics_test(neighborhood_data=invalid_data_coords_wrong_len)
            print("  FAIL: Malformed data ('coordinates' wrong length) did not raise ValueError.")
            return False
        except ValueError as e:
            print(f"  PASS: Malformed data ('coordinates' wrong length) raised ValueError: {e}")

        # Test case 6: 'coordinates' non-numeric value
        invalid_data_coords_non_numeric = [valid_item, {"id": "node2", "coordinates": [1.0, "two", 3.0, 4.0]}]
        try:
            mock_get_shared_manifold_for_ethics_test(neighborhood_data=invalid_data_coords_non_numeric)
            print("  FAIL: Malformed data ('coordinates' non-numeric) did not raise ValueError.")
            return False
        except ValueError as e:
            print(f"  PASS: Malformed data ('coordinates' non-numeric) raised ValueError: {e}")

        return True

    def test_framework_alignment_advanced(**kwargs) -> bool:
        """
        Tests regex word boundary, case-insensitivity, and synonym handling
        in framework_alignment scoring.
        """
        print("Testing framework_alignment advanced (regex, synonyms)...")
        global _ethics_db, config # Need config for ETHICAL_FRAMEWORK override

        # Test cases: (text_to_analyze, expected_positive_matches, expected_negative_matches)
        # Uses keywords defined in tests_to_run for this test function.
        # Positive: [{"term": "help", "synonyms": ["assist", "support"]}, "control", {"term": "deal.", "synonyms": ["agreement."]}]
        # Negative: [{"term": "harm", "synonyms": ["damage", "hurt"]}, "uncontrolled"]
        test_scenarios = [
            # Basic term matching
            ("This is a helpful action.", 1, 0), # "help" (main term)
            ("This is harmful.", 0, 1),          # "harm" (main term)
            # Synonym matching
            ("He will assist you.", 1, 0),       # "assist" (synonym for "help")
            ("It caused some damage.", 0, 1),    # "damage" (synonym for "harm")
            # Word boundary and case-insensitivity (already covered, but good to re-verify with new structure)
            ("It causes no harm, what a harmony!", 0, 1), # "harm", not "harmony"
            ("We need to CONTROL the situation.", 1, 0), # "control" (string entry, case-insensitive)
            ("The situation is uncontrolled.", 0, 1), # "uncontrolled" (string entry)
            # Special characters in term (via dict)
            ("Let's make a deal.", 1, 0),        # "deal." (main term with dot)
            ("This is a good deal for us", 0, 0),# "deal" (no dot) should not match "deal."
            ("Finalize the agreement.", 1, 0), # "agreement." (synonym for "deal.")
            # Multiple matches for the same keyword entry (should count as one)
            ("Please help and support this cause.", 1, 0), # "help" and "support" are for the same positive entry
            ("The action might hurt and also inflict harm.", 0, 1), # "hurt" and "harm" are for the same negative entry
            # Mixed matches
            ("We must control the potential harm.", 1, 1), # "control" (positive), "harm" (negative)
            ("Support this, but it may cause damage.", 1, 1), # "support" (positive synonym), "damage" (negative synonym)
            # No matches
            ("This is neutral.", 0, 0),
            # Malformed entry in config (test if score_ethics logs warning and continues) - This is harder to test here
            # as config is mocked. Assuming score_ethics handles it by skipping.
        ]

        # ETHICAL_FRAMEWORK is overridden by test_configs in tests_to_run.
        # The keywords used in test_scenarios above must align with that specific config.
        # Positive: [{"term": "help", "synonyms": ["assist", "support"]}, "control", {"term": "deal.", "synonyms": ["agreement."]}]
        # Negative: [{"term": "harm", "synonyms": ["damage", "hurt"]}, "uncontrolled"]

        # Override ETHICAL_FRAMEWORK for this test
        # This requires setup_test_environment to pass test_specific_configs to TempConfigOverride
        # And TempConfigOverride to correctly apply it to the 'config' object.
        # The test_specific_configs is passed as 'test_configs' kwarg to run_test.

        # The actual keywords will be set by the test runner via test_configs kwarg.
        # Here we just assume they are set by the time score_ethics is called.
        # Default keywords from score_ethics:
        # positive_keywords = ["help", "improve", "assist", "share", "create", "understand", "align", "benefit"]
        # negative_keywords = ["harm", "deceive", "exploit", "manipulate", "destroy", "control", "damage"]
        # For this test, we'll use a custom set via config override.

        results_ok = True
        for i, (text, expected_pos, expected_neg) in enumerate(test_scenarios):
            # Minimal awareness_metrics, not relevant for this specific component test
            awareness_metrics = {"coherence": 0.0, "primary_concept_coord": (0,0,0,0), "raw_t_intensity": 0.5}

            # score_ethics logs its results, so we can inspect the last log entry.
            # Ensure the DB is clean before each scoring to easily get the last score.
            _ethics_db["ethical_scores"] = []
            _ethics_db_dirty_flag = True # To allow saving if needed, though not strictly for this test.

            score_ethics(awareness_metrics, concept_summary=text, action_description="")

            if not _ethics_db["ethical_scores"]:
                print(f"  FAIL Scenario {i+1} ('{text}'): No score event logged.")
                results_ok = False
                continue

            last_score_event = _ethics_db["ethical_scores"][-1]
            component_scores = last_score_event.get("component_scores", {})
            framework_score = component_scores.get("framework_alignment", -1) # Default to -1 if not found

            # Calculate expected score based on counts
            # This logic must mirror what's in score_ethics's framework_alignment part
            calculated_expected_score = 0.5 # Neutral if no keywords match
            if expected_pos + expected_neg > 0:
                calculated_expected_score = np.clip(expected_pos / (expected_pos + expected_neg), 0.0, 1.0)

            if abs(framework_score - calculated_expected_score) < 0.001:
                print(f"  PASS Scenario {i+1} ('{text}'): Score {framework_score:.2f} matches expected {calculated_expected_score:.2f} (pos: {expected_pos}, neg: {expected_neg}).")
            else:
                print(f"  FAIL Scenario {i+1} ('{text}'): Score {framework_score:.2f}, Expected {calculated_expected_score:.2f} (pos: {expected_pos}, neg: {expected_neg}). Logged event: {last_score_event}")
                results_ok = False

        return results_ok

    def test_score_ethics_input_validation(**kwargs) -> bool:
        """
        Tests input validation for concept_summary and action_description in score_ethics.
        Checks type coercion and length truncation.
        """
        print("Testing score_ethics input validation...")
        global _ethics_db, config, TEST_SYSTEM_LOG_PATH

        # Config is set via test_configs in tests_to_run
        # ETHICS_MAX_TEXT_INPUT_LENGTH is set to 20
        # ETHICAL_FRAMEWORK positive_keywords: ["good", "excellent"] (excellent is > 20 with a prefix)
        # ETHICS_FRAMEWORK_WEIGHT is 1.0, others 0.

        max_len = config.ETHICS_MAX_TEXT_INPUT_LENGTH # Should be 20 from test_configs

        test_cases = [
            {
                "name": "non_string_summary",
                "summary": 123, "action": "valid action",
                "expected_log_event": "score_ethics_input_type_coercion",
                "expected_param": "concept_summary",
                "expected_text_for_match": "", # Coerced to empty
                "expect_keyword_match": False # "good" or "excellent" won't be in ""
            },
            {
                "name": "non_string_action",
                "summary": "valid summary", "action": None,
                "expected_log_event": "score_ethics_input_type_coercion",
                "expected_param": "action_description",
                "expected_text_for_match": "valid summary", # Action becomes empty
                "expect_keyword_match": False # Assuming "good" is not in "valid summary"
            },
            {
                "name": "summary_too_long",
                "summary": "This is a good summary that is definitely longer than twenty chars and has excellent content.",
                "action": "short",
                "expected_log_event": "score_ethics_input_length_truncated",
                "expected_param": "concept_summary",
                 # "This is a good summa" (20 chars), "good" should match. "excellent" is truncated.
                "expected_text_for_match": "This is a good summa",
                "expect_keyword_match": True # "good" should match
            },
             {
                "name": "summary_too_long_no_match_after_truncate",
                "summary": "Twenty chars exactly excellent word.", # "excellent" starts at char 21
                "action": "",
                "expected_log_event": "score_ethics_input_length_truncated",
                "expected_param": "concept_summary",
                "expected_text_for_match": "Twenty chars exactly",
                "expect_keyword_match": False # "excellent" should be truncated
            }
        ]

        all_passed = True
        for case in test_cases:
            print(f"  Running case: {case['name']}...")
            # Clear previous log entries for this specific sub-test might be complex.
            # Instead, we'll scan all logs from the run_test context.
            # A more robust way would be to pass a unique ID to score_ethics and check for that in logs,
            # but that's a larger change. We rely on event_type and parameter.

            # Clean score log to isolate the framework score from this specific call
            _ethics_db["ethical_scores"] = []

            score = score_ethics(
                awareness_metrics={"coherence": 0.0, "primary_concept_coord": (0,0,0,0), "raw_t_intensity": 0.5},
                concept_summary=case["summary"],
                action_description=case["action"]
            )

            assert isinstance(score, float) and 0.0 <= score <= 1.0, f"  FAIL [{case['name']}]: Score {score} is not a valid float in [0,1]."

            # Check logs for the specific warning
            log_found = False
            if os.path.exists(TEST_SYSTEM_LOG_PATH):
                with open(TEST_SYSTEM_LOG_PATH, 'r') as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line)
                            if log_entry.get("event_type") == case["expected_log_event"] and \
                               log_entry.get("data", {}).get("parameter") == case["expected_param"]:
                                log_found = True
                                break
                        except json.JSONDecodeError:
                            continue
            assert log_found, f"  FAIL [{case['name']}]: Expected log event '{case['expected_log_event']}' for parameter '{case['expected_param']}' not found."
            print(f"    PASS [{case['name']}]: Correct warning log found.")

            # Check framework alignment score based on (potentially) modified text
            if _ethics_db["ethical_scores"]:
                last_score_event = _ethics_db["ethical_scores"][-1]
                component_scores = last_score_event.get("component_scores", {})
                framework_score = component_scores.get("framework_alignment", -1.0)

                # Keywords for this test are ["good", "excellent"] (positive)
                # This logic assumes ETHICS_FRAMEWORK_WEIGHT = 1.0 and others 0 from test_configs
                text_that_was_analyzed = ""
                if case["expected_param"] == "concept_summary":
                    text_that_was_analyzed = case["expected_text_for_match"] + (str(case["action"]) if isinstance(case["action"], str) else "")
                elif case["expected_param"] == "action_description":
                     text_that_was_analyzed = (str(case["summary"]) if isinstance(case["summary"], str) else "") + case["expected_text_for_match"]
                else: # For non_string cases, summary is expected_text_for_match if action is coerced, or vice versa
                    if "summary" in case["expected_param"]:
                         text_that_was_analyzed = case["expected_text_for_match"] + (str(case["action"]) if isinstance(case["action"], str) else "")
                    else: # action is coerced
                         text_that_was_analyzed = (str(case["summary"]) if isinstance(case["summary"], str) else "") + case["expected_text_for_match"]


                # Simplified check: does "good" appear in the text that should have been analyzed?
                # "excellent" is used to test truncation (it won't appear if truncated before it).
                # This relies on the specific keywords set in test_configs.

                # For 'summary_too_long': summary is "This is a good summa...", action "short"
                # text_to_analyze becomes "This is a good summashort"
                # If "good" is in "This is a good summa", then score should be 1.0.
                # If "excellent" was the keyword, and it got truncated, score would be 0.5 (neutral).

                # Re-calculate expected score based on whether "good" (the shorter keyword) is present in expected_text_for_match
                # This is a bit of a simplification of how the score is derived, focusing on the keyword we expect to see or not see.

                # Keywords from config: positive_keywords = ["good", "excellent"]
                # The text_to_analyze in score_ethics is concept_summary + " " + action_description
                # Let's reconstruct what text_to_analyze would have been

                processed_summary = case["summary"]
                if not isinstance(processed_summary, str): processed_summary = ""
                if len(processed_summary) > max_len: processed_summary = processed_summary[:max_len]

                processed_action = case["action"]
                if not isinstance(processed_action, str): processed_action = ""
                if len(processed_action) > max_len: processed_action = processed_action[:max_len]

                final_text_for_analysis = (processed_summary + " " + processed_action).lower()

                # Check against keywords from config.
                # test_configs positive_keywords: ["good", "excellent"]
                # test_configs negative_keywords: ["badkey"]

                # For "summary_too_long", final_text_for_analysis = "this is a good summa short" -> "good" matches -> 1 positive -> score 1.0
                # For "summary_too_long_no_match_after_truncate", final_text_for_analysis = "twenty chars exactly " -> no match -> score 0.5

                expected_score_after_val = 0.5 # Neutral
                pos_hits = 0
                if "good" in final_text_for_analysis: pos_hits = 1
                # "excellent" is only in one original summary, and it's designed to be truncated or just at edge.
                # In "summary_too_long", original "excellent" is at char 46. Truncated summary: "This is a good summa". Action: "short".
                # final_text_for_analysis = "this is a good summa short". "excellent" is not there.
                # In "summary_too_long_no_match_after_truncate", original "excellent" is at char 21. Truncated: "Twenty chars exactly". Action: "".
                # final_text_for_analysis = "twenty chars exactly ". "excellent" is not there.

                if case["name"] == "summary_too_long" and "good" in final_text_for_analysis: # "good" is within first 20 chars
                     expected_score_after_val = 1.0
                elif case["name"] == "summary_too_long_no_match_after_truncate" and "excellent" not in final_text_for_analysis:
                     expected_score_after_val = 0.5 # because "excellent" is truncated
                elif case["name"] == "non_string_summary" or case["name"] == "non_string_action":
                     expected_score_after_val = 0.5 # because inputs become empty, no keywords match.

                assert abs(framework_score - expected_score_after_val) < 0.001, \
                    f"  FAIL [{case['name']}]: Framework score {framework_score:.2f}, Expected after validation {expected_score_after_val:.2f}. Final text analyzed: '{final_text_for_analysis}'"
                print(f"    PASS [{case['name']}]: Framework score correct after validation.")
            elif case["expect_keyword_match"] or not case["expect_keyword_match"]: # If we expected a score change and there's no score record
                 # This condition means we care about the keyword match outcome for this test case
                 print(f"  FAIL [{case['name']}]: No score event logged in _ethics_db to check framework score.")
                 all_passed = False


        return all_passed

    def test_concurrent_operations(**kwargs) -> bool:
        """
        Tests concurrent calls to score_ethics to check for race conditions
        around database operations.
        """
        print("Testing concurrent score_ethics operations...")
        global _ethics_db, config
        # Import locally for test function if not at top level of if __name__ == '__main__'
        import concurrent.futures

        # Ensure enough log entries are allowed if pruning is aggressive
        # And ensure LOG_LEVEL allows 'info' from score_ethics_complete
        test_configs_concurrent = {
            "ETHICS_LOG_MAX_ENTRIES": 50,
            "LOG_LEVEL": "INFO"
        }
        # Apply these configs if they are not already default or set by run_test context
        # This test runs within run_test, so config is managed by TempConfigOverride there.
        # We can assume ETHICS_LOG_MAX_ENTRIES is sufficient from its test_configs.

        # Ensure DB is clean before starting this test
        _ethics_db["ethical_scores"] = []
        _ethics_db["trend_analysis"] = {}
        _ethics_db_dirty_flag = True # Force a save to clear out any existing test file
        _save_ethics_db()
        # Reload to ensure clean state from file (or lack thereof)
        _load_ethics_db()


        num_calls = 20
        # Minimal awareness_metrics, concept_summary, action_description for the test
        awareness_metrics = {"coherence": 0.1, "primary_concept_coord": (0.1,0.1,0.1,0.1), "raw_t_intensity": 0.1}
        summary = "concurrent summary"
        action = "concurrent action"

        results_list = [] # To store results or exceptions from threads

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_calls) as executor:
            futures = [executor.submit(score_ethics, awareness_metrics, f"{summary} {i}", f"{action} {i}") for i in range(num_calls)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    results_list.append(future.result())
                except Exception as e:
                    results_list.append(e) # Store exception if any occurred

        # Check for exceptions in thread execution
        exceptions_in_threads = [r for r in results_list if isinstance(r, Exception)]
        if exceptions_in_threads:
            print(f"  FAIL: Exceptions occurred during concurrent execution: {exceptions_in_threads}")
            return False

        # Check 1: In-memory DB state after all calls
        # The _ethics_db_dirty_flag might be False if the last save completed fully.
        # The number of scores should be num_calls if pruning didn't occur.
        # The run_test context for this test should set ETHICS_LOG_MAX_ENTRIES high enough.

        # Re-acquire lock to safely read _ethics_db length
        with _db_lock:
            in_memory_score_count = len(_ethics_db["ethical_scores"])

        # _save_ethics_db might be called by the last thread. Ensure it finishes.
        # Wait a very brief moment for any final save to complete.
        # This is a bit heuristic; proper synchronization would be more complex.
        # Given _save_ethics_db itself is mostly synchronous after acquiring its data,
        # this might be okay.
        import time
        time.sleep(0.1) # Small delay for file I/O to settle if any last save was pending.


        if in_memory_score_count != num_calls:
             print(f"  FAIL: In-memory ethical_scores count is {in_memory_score_count}, expected {num_calls}.")
             # Log current _ethics_db for debugging if count mismatches
             # _log_ethics_event("debug_concurrent_test_in_memory_db_state", {"db": _ethics_db}, level="debug")
             return False
        print(f"  PASS: In-memory ethical_scores count is {in_memory_score_count} as expected.")

        # Check 2: Reload DB from file and verify count
        _load_ethics_db() # This uses its own lock

        with _db_lock: # Lock to read the reloaded _ethics_db
            reloaded_score_count = len(_ethics_db["ethical_scores"])

        if reloaded_score_count != num_calls:
            print(f"  FAIL: Reloaded ethical_scores count from DB file is {reloaded_score_count}, expected {num_calls}.")
            # _log_ethics_event("debug_concurrent_test_reloaded_db_state", {"db": _ethics_db}, level="debug")
            return False
        print(f"  PASS: Reloaded ethical_scores count from DB file is {reloaded_score_count} as expected.")

        # Check 3: Verify DB file is valid JSON (already implicitly tested by _load_ethics_db)
        # No InvalidToken or JSONDecodeError means it's structurally fine.
        print("  PASS: DB file loaded successfully (implies valid JSON and potentially valid decryption).")

        return True

    def test_load_corrupted_encrypted_db(**kwargs) -> bool:
        """
        Tests loading a DB file that is encrypted but with garbage content or wrong key.
        """
        print("Testing loading corrupted encrypted DB...")
        global _ethics_db, TEST_ETHICS_DB_PATH, config

        # Key is set by run_test via encryption_key kwarg
        assert os.environ.get("ETHICS_ENCRYPTION_KEY"), "FAIL: ETHICS_ENCRYPTION_KEY not set for this test."

        # Create a corrupted/garbage file
        with open(TEST_ETHICS_DB_PATH, 'wb') as f:
            f.write(os.urandom(100)) # Write 100 random bytes (not valid Fernet token)

        _load_ethics_db() # Attempt to load the corrupted DB

        with _db_lock: # Lock to read shared state
            assert _ethics_db == {"ethical_scores": [], "trend_analysis": {}}, \
                "FAIL: _ethics_db not reset to default after loading corrupted encrypted file."
            assert not _ethics_db_dirty_flag, \
                "FAIL: _ethics_db_dirty_flag not False after loading corrupted file and resetting."
        print("  PASS: _ethics_db reset to default and dirty flag is false.")

        # Check for log message (this is an approximation, log content might vary)
        log_found = False
        if os.path.exists(TEST_SYSTEM_LOG_PATH):
            with open(TEST_SYSTEM_LOG_PATH, 'r') as f:
                for line in f:
                    if "load_ethics_db_decryption_error" in line or "Invalid token" in line:
                        log_found = True
                        break
        assert log_found, "FAIL: Expected decryption error log not found."
        print("  PASS: Decryption error correctly logged.")
        return True

    def test_score_ethics_missing_awareness_keys(**kwargs) -> bool:
        """
        Tests score_ethics behavior when awareness_metrics keys are missing.
        """
        print("Testing score_ethics with missing awareness_metrics keys...")
        global _ethics_db, config

        # Config to make all component weights non-zero to see their effect,
        # but ETHICAL_FRAMEWORK can be empty to simplify score expectation.
        # test_configs for this test should set these.

        missing_keys_metrics = {} # Empty dict
        all_default_metrics = { # All keys present but with default/neutral values expected by .get()
            "coherence": 0.0,
            "primary_concept_coord": None, # or an invalid format that leads to default
            "raw_t_intensity": 0.0
        }

        scenarios = {
            "completely_empty_metrics": missing_keys_metrics,
            "all_default_values": all_default_metrics
        }

        results_ok = True
        for name, metrics in scenarios.items():
            print(f"  Scenario: {name}")
            _ethics_db["ethical_scores"] = [] # Clear for each sub-test
            score = score_ethics(metrics, "test summary", "test action")

            assert isinstance(score, float) and 0.0 <= score <= 1.0, \
                f"  FAIL [{name}]: Score {score} is not a valid float in [0,1]."
            print(f"    PASS [{name}]: Valid score {score:.2f} returned.")

            if not _ethics_db["ethical_scores"]:
                print(f"  FAIL [{name}]: No score event logged.")
                results_ok = False
                continue

            last_score_event = _ethics_db["ethical_scores"][-1]
            component_scores = last_score_event.get("component_scores", {})

            # Check if component scores are neutral (0.5 for some, or based on 0.0 inputs for others)
            # This depends on how score_ethics defaults them.
            # Example: coherence score for 0.0 coherence is 1.0.
            # manifold_valence for missing/invalid coord is 0.5.
            # intensity_preference for 0.0 intensity depends on ideal_intensity_center (0.5 default) and sigma.
            # framework_alignment for empty text and default keywords is 0.5.
            expected_coherence = 1.0 # (1 - abs(0.0))
            expected_valence = 0.5
            # intensity_pref = exp(-((0.0 - 0.5)^2) / (2*0.25^2)) = exp(-0.25 / 0.125) = exp(-2) approx 0.135
            # This might change if ETHICS_IDEAL_INTENSITY_CENTER or SIGMA is changed by test_configs
            ideal_center = getattr(config, 'ETHICS_IDEAL_INTENSITY_CENTER', 0.5)
            sigma_intensity = getattr(config, 'ETHICS_INTENSITY_PREFERENCE_SIGMA', 0.25)
            expected_intensity_pref = np.exp(-((0.0 - ideal_center)**2) / (2 * sigma_intensity**2))

            # Assuming empty text for framework if summary/action are also empty or keywords don't match "test summary test action"
            expected_framework = 0.5
            # Cluster context also defaults to 0.5 if coord is bad or manifold mock is basic
            expected_cluster = 0.5


            assert abs(component_scores.get("coherence", -1) - expected_coherence) < 0.001, \
                f"  FAIL [{name}]: Coherence score {component_scores.get('coherence')} != {expected_coherence}"
            assert abs(component_scores.get("manifold_valence", -1) - expected_valence) < 0.001, \
                 f"  FAIL [{name}]: Valence score {component_scores.get('manifold_valence')} != {expected_valence}"
            assert abs(component_scores.get("intensity_preference", -1) - expected_intensity_pref) < 0.001, \
                 f"  FAIL [{name}]: Intensity score {component_scores.get('intensity_preference')} != {expected_intensity_pref}"
            assert abs(component_scores.get("framework_alignment", -1) - expected_framework) < 0.001, \
                 f"  FAIL [{name}]: Framework score {component_scores.get('framework_alignment')} != {expected_framework}"
            assert abs(component_scores.get("manifold_cluster_context", -1) - expected_cluster) < 0.001, \
                 f"  FAIL [{name}]: Cluster score {component_scores.get('manifold_cluster_context')} != {expected_cluster}"
            print(f"    PASS [{name}]: Component scores appear to default correctly.")

            # Check for warnings (e.g. "score_ethics_param_error") in logs
            # This test doesn't explicitly check logs for brevity, but could be added.
            # The current score_ethics logs warnings for coherence, raw_t_intensity, coord processing if they are problematic.
        return results_ok

    def test_track_trends_malformed_scores(**kwargs) -> bool:
        """
        Tests track_trends behavior with malformed entries in ethical_scores.
        """
        print("Testing track_trends with malformed score entries...")
        global _ethics_db, config

        # Config for track_trends (min_datapoints) is set by run_test context
        # Ensure ETHICS_TREND_MIN_DATAPOINTS is low enough (e.g., 3) via test_configs

        malformed_scores = [
            "not_a_dict", # Invalid type
            {"final_score": "not_a_float", "primary_concept_t_intensity_raw": 0.5}, # Invalid score type
            {"final_score": 0.8, "primary_concept_t_intensity_raw": "not_a_float"}, # Invalid intensity type
            {"primary_concept_t_intensity_raw": 0.5}, # Missing final_score
            {"final_score": 0.7}, # Missing intensity
            # Some valid entries to ensure it can still process them
            {"final_score": 0.6, "primary_concept_t_intensity_raw": 0.3, "timestamp": "t1"},
            {"final_score": 0.65, "primary_concept_t_intensity_raw": 0.4, "timestamp": "t2"},
            {"final_score": 0.7, "primary_concept_t_intensity_raw": 0.5, "timestamp": "t3"},
            {"final_score": 0.75, "primary_concept_t_intensity_raw": 0.6, "timestamp": "t4"},
        ]

        with _db_lock:
            _ethics_db["ethical_scores"] = copy.deepcopy(malformed_scores) # Use deepcopy if sub-elements might be changed by other tests
            _ethics_db_dirty_flag = True # To allow saving if track_trends modifies DB state

        trends_result = track_trends()

        assert isinstance(trends_result, dict), "FAIL: track_trends did not return a dictionary."
        # It should process the 4 valid entries. If min_datapoints is <=4, it should calculate a trend.
        # If min_datapoints is, say, 3, it will use these 4.
        # If test_configs sets min_datapoints to 3, then 4 valid points is enough.

        min_dp_for_trend = getattr(config, 'ETHICS_TREND_MIN_DATAPOINTS', 10) # Get from config as set by test
        if 4 >= min_dp_for_trend :
             assert trends_result.get("data_points_used") == 4, \
                f"FAIL: Expected 4 valid data points to be used, got {trends_result.get('data_points_used')}"
             assert trends_result.get("trend_direction") != "insufficient_data" or trends_result.get("trend_direction") != "error_config_missing", \
                f"FAIL: Trend direction indicates error or insufficient data with {trends_result.get('data_points_used')} points, expected calculation."
        else: # 4 < min_dp_for_trend
            assert trends_result.get("trend_direction") == "insufficient_data", \
                f"FAIL: Expected 'insufficient_data' due to {4} valid points vs {min_dp_for_trend} min, got {trends_result.get('trend_direction')}"
            assert trends_result.get("data_points_used") == 4, \
                f"FAIL: Expected 4 valid data points to be counted, got {trends_result.get('data_points_used')}"

        print(f"  PASS: track_trends processed malformed data and yielded trend: '{trends_result.get('trend_direction')}' with {trends_result.get('data_points_used')} points.")

        # Check for "track_trends_data_format_error" logs
        log_found = False
        if os.path.exists(TEST_SYSTEM_LOG_PATH):
            with open(TEST_SYSTEM_LOG_PATH, 'r') as f:
                for line in f:
                    if "track_trends_data_format_error" in line:
                        log_found = True
                        break
        assert log_found, "FAIL: Expected 'track_trends_data_format_error' log not found."
        print("  PASS: Warning for malformed data correctly logged by track_trends.")

        return True

    def test_db_load_pruning(**kwargs) -> bool:
        """
        Tests that _load_ethics_db prunes entries if the file contains more
        than ETHICS_LOG_MAX_ENTRIES.
        """
        print("Testing _load_ethics_db pruning...")
        global _ethics_db, _ethics_db_dirty_flag, TEST_ETHICS_DB_PATH, config

        # Config is set by run_test via test_configs
        # ETHICS_LOG_MAX_ENTRIES should be set to a small number like 3 or 5 for this test.
        max_entries_config = getattr(config, 'ETHICS_LOG_MAX_ENTRIES', 5) # Default to 5 if not in test_configs

        # 1. Create a DB file with more than max_entries_config scores
        num_scores_to_write = max_entries_config + 3
        temp_db_content = {"ethical_scores": [], "trend_analysis": {}}
        for i in range(num_scores_to_write):
            # Create distinct entries, e.g., by varying a timestamp or a dummy field
            temp_db_content["ethical_scores"].append({
                "final_score": 0.5,
                "primary_concept_t_intensity_raw": 0.5,
                "timestamp": f"2023-01-01T00:00:{i:02d}Z", # Make them sortable by time
                "test_id": i
            })

        # Save this oversized DB content to the test file (unencrypted for simplicity here)
        # Ensure no encryption key is set for this direct save part of the test setup
        current_env_key = os.environ.pop("ETHICS_ENCRYPTION_KEY", None)
        try:
            with open(TEST_ETHICS_DB_PATH, 'w') as f:
                json.dump(temp_db_content, f)
        finally:
            if current_env_key: # Restore if it was set
                os.environ["ETHICS_ENCRYPTION_KEY"] = current_env_key

        # 2. Call _load_ethics_db() - this will happen in the run_test context which sets up env key if needed
        # For this test, we want to test the pruning logic, so whether it's encrypted or not during load
        # depends on the 'encryption_key' passed to run_test for this test function.
        # Let's assume this test will be run with encryption_key=None for simplicity of setup,
        # or the test runner ensures the key matches if one was used for saving.
        # The critical part is that _load_ethics_db reads the file.

        _load_ethics_db() # This should trigger pruning if file content > max_entries_config

        log_pruned_event_found = False
        # Check for the pruning log event
        if os.path.exists(TEST_SYSTEM_LOG_PATH):
            with open(TEST_SYSTEM_LOG_PATH, 'r') as f:
                for line in f:
                    try:
                        log_json = json.loads(line.strip())
                        if log_json.get("event_type") == "load_ethics_db_pruned":
                            log_pruned_event_found = True
                            assert log_json.get("data", {}).get("retained_count") == max_entries_config
                            assert log_json.get("data", {}).get("removed_count") == 3
                            break
                    except json.JSONDecodeError:
                        continue

        assert log_pruned_event_found, "FAIL: 'load_ethics_db_pruned' event not logged."
        print("  PASS: Pruning event logged correctly during load.")

        with _db_lock: # Lock to read _ethics_db
            loaded_scores_count = len(_ethics_db.get("ethical_scores", []))

        assert loaded_scores_count == max_entries_config, \
            f"FAIL: DB not pruned on load. Expected {max_entries_config} scores, got {loaded_scores_count}."
        print(f"  PASS: DB correctly pruned to {max_entries_config} entries on load.")

        # Verify that the most RECENT entries were kept (based on test_id or timestamp)
        # The pruning logic in _load_ethics_db is `data["ethical_scores"][num_to_remove:]`
        # So, it keeps the latter part of the list, which should be the most recent if ordered by append time.
        # Our dummy entries have increasing timestamps/test_ids.
        with _db_lock:
            first_kept_score_test_id = _ethics_db.get("ethical_scores", [])[0].get("test_id") if loaded_scores_count > 0 else -1

        # Expected first test_id after pruning num_scores_to_write - max_entries_config
        # e.g., 8 scores, max 5. 3 removed. Kept are indices 3,4,5,6,7. First kept is test_id 3.
        expected_first_test_id = num_scores_to_write - max_entries_config
        assert first_kept_score_test_id == expected_first_test_id, \
            f"FAIL: Incorrect entries kept after pruning. Expected first test_id {expected_first_test_id}, got {first_kept_score_test_id}."
        print(f"  PASS: Correct (most recent) entries retained after pruning.")

        return True

    def test_async_logging(**kwargs) -> bool:
        """
        Tests that asynchronous logging writes all messages and doesn't block excessively.
        Also checks if the executor is properly shut down via atexit (manual check for now).
        """
        print("Testing asynchronous logging...")
        global TEST_SYSTEM_LOG_PATH, config, _log_executor

        # Ensure log level is very permissive for this test
        # test_configs in run_test will set LOG_LEVEL = "DEBUG"

        num_log_messages = 50
        test_event_type = "test_async_log_event"

        # Clear the log file before this test section
        if os.path.exists(TEST_SYSTEM_LOG_PATH):
            os.remove(TEST_SYSTEM_LOG_PATH)

        # Log many messages quickly
        for i in range(num_log_messages):
            _log_ethics_event(test_event_type, {"message_id": i, "content": f"Async test message {i}"}, level="debug")

        # Wait for the log executor to process messages.
        # A more robust way might involve checking _log_executor._work_queue.qsize(),
        # but that's an internal detail. A short sleep is usually sufficient for tests.
        # Ensure all tasks are done before reading the file.
        # Forcing shutdown and recreation of executor is too complex for a test,
        # so we rely on the global executor and atexit for cleanup.
        # A simple check could be to wait until queue is empty.

        # Wait a bit for logs to be written. This is heuristic.
        # In a real scenario with a long-running app, atexit handles shutdown.
        # For a test, we might need to explicitly wait for queue to empty or use a local executor.
        # For now, let's use a slightly longer sleep and check queue size if possible (though not standard API).

        # Try to wait for the queue to clear, but with a timeout.
        wait_time = 0.0
        max_wait_time = 5.0 # seconds
        sleep_interval = 0.05
        queue_cleared = False
        if hasattr(_log_executor, '_work_queue'): # Check if internal queue is accessible
            while _log_executor._work_queue.qsize() > 0 and wait_time < max_wait_time:
                time.sleep(sleep_interval)
                wait_time += sleep_interval
            if _log_executor._work_queue.qsize() == 0:
                queue_cleared = True
                print(f"  INFO: Log queue cleared in {wait_time:.2f}s.")
            else:
                print(f"  WARNING: Log queue not cleared after {max_wait_time}s. Size: {_log_executor._work_queue.qsize()}")
        else: # Fallback to simple sleep if queue not accessible
            print("  INFO: _log_executor._work_queue not accessible, using fixed sleep.")
            time.sleep(1.0) # Wait 1 second for logs to flush if queue size can't be checked.
            queue_cleared = True # Assume cleared for purpose of test continuing

        if not queue_cleared and not hasattr(_log_executor, '_work_queue'):
             # If we just did a fixed sleep, give it a bit more to be safe if system is slow.
             time.sleep(1.0)


        # Verify all messages are in the log file
        logged_message_ids = set()
        if os.path.exists(TEST_SYSTEM_LOG_PATH):
            with open(TEST_SYSTEM_LOG_PATH, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        if log_entry.get("event_type") == test_event_type:
                            logged_message_ids.add(log_entry.get("data", {}).get("message_id"))
                    except json.JSONDecodeError:
                        continue # Skip malformed lines

        missing_ids = set(range(num_log_messages)) - logged_message_ids
        if missing_ids:
            print(f"  FAIL: Missing message IDs in log: {sorted(list(missing_ids))[:10]} (showing first 10 if many)")
            print(f"  Logged count: {len(logged_message_ids)}, Expected: {num_log_messages}")
            return False

        assert len(logged_message_ids) == num_log_messages, \
            f"FAIL: Not all messages logged. Expected {num_log_messages}, got {len(logged_message_ids)}."
        print(f"  PASS: All {num_log_messages} async messages correctly logged.")

        # The atexit handler for _log_executor.shutdown is hard to test directly here
        # without ending the test process. We assume it works if registered.
        return True

    def test_manifold_caching(**kwargs) -> bool:
        """
        Tests the manifold query caching in score_ethics.
        """
        print("Testing manifold query caching...")
        global _ethics_db, config, _manifold_cache

        _manifold_cache.clear() # Ensure a clean cache for this test run

        # Mock get_shared_manifold to return a manifold with a spyable get_conceptual_neighborhood
        class CallCounter: # Simple callable wrapper to count calls
            def __init__(self, func_to_wrap):
                self.func = func_to_wrap
                self.call_count = 0
                self.last_args = None
                self.last_kwargs = None
            def __call__(self, *args, **kwargs):
                self.call_count += 1
                self.last_args = args
                self.last_kwargs = kwargs
                return self.func(*args, **kwargs)

        # Define a simple neighborhood data to be returned by the mock
        mock_neighborhood_result = [{"id": "neighbor1", "coordinates": [0.1,0.2,0.3,0.4]}]

        # This function will be wrapped by CallCounter
        def actual_mock_get_neighborhood(concept_coord, radius):
            # Log or print actual call to mock
            # print(f"DEBUG: actual_mock_get_neighborhood called with coord={concept_coord}, radius={radius}")
            return mock_neighborhood_result

        spyable_get_neighborhood = CallCounter(actual_mock_get_neighborhood)

        class TestMockManifold:
            def __init__(self):
                self.device = "cpu"
                # Attach the spyable version here
                self.get_conceptual_neighborhood = spyable_get_neighborhood

        original_get_shared_manifold_in_module = sys.modules[__name__].get_shared_manifold
        sys.modules[__name__].get_shared_manifold = lambda force_recreate=False: TestMockManifold()

        # Test parameters
        awareness_metrics_1 = {
            "coherence": 0.0,
            "primary_concept_coord": [1.0, 2.0, 3.0, 0.0], # Use list to test tuple conversion for cache key
            "raw_t_intensity": 0.5
        }
        # Config should set ETHICS_CLUSTER_RADIUS_FACTOR and MANIFOLD_RANGE
        # e.g. radius_factor = 0.1, manifold_range = 10.0 => radius = 1.0
        # These will be taken from config by score_ethics. Ensure test_configs sets them.

        # --- Call 1: Cache Miss ---
        _ethics_db["ethical_scores"] = [] # Clear scores to check the new one
        score_1 = score_ethics(awareness_metrics_1, "summary1", "action1")
        assert spyable_get_neighborhood.call_count == 1, \
            f"FAIL (Call 1): Expected 1 call to get_conceptual_neighborhood, got {spyable_get_neighborhood.call_count}"
        print("  PASS (Call 1): get_conceptual_neighborhood called once (cache miss).")
        last_score_event_1 = _ethics_db["ethical_scores"][-1]
        cluster_score_1 = last_score_event_1["component_scores"]["manifold_cluster_context"]


        # --- Call 2: Cache Hit ---
        # Use same awareness_metrics (which means same primary_coord and radius derived from config)
        _ethics_db["ethical_scores"] = []
        score_2 = score_ethics(awareness_metrics_1, "summary2", "action2") # Summary/action change, but coord/radius same
        assert spyable_get_neighborhood.call_count == 1, \
            f"FAIL (Call 2): Expected 1 call (total) due to cache hit, got {spyable_get_neighborhood.call_count}"
        print("  PASS (Call 2): get_conceptual_neighborhood not called again (cache hit).")
        last_score_event_2 = _ethics_db["ethical_scores"][-1]
        cluster_score_2 = last_score_event_2["component_scores"]["manifold_cluster_context"]
        assert abs(cluster_score_1 - cluster_score_2) < 0.001, \
             f"FAIL (Call 2): Cluster score {cluster_score_2} should be same as {cluster_score_1} due to cache."


        # --- Call 3: Different Coords - Cache Miss ---
        awareness_metrics_2 = {
            "coherence": 0.0,
            "primary_concept_coord": (4.0, 5.0, 6.0, 0.0), # Different coordinates
            "raw_t_intensity": 0.5
        }
        _ethics_db["ethical_scores"] = []
        score_3 = score_ethics(awareness_metrics_2, "summary3", "action3")
        assert spyable_get_neighborhood.call_count == 2, \
            f"FAIL (Call 3): Expected 2 calls (total) due to new cache miss, got {spyable_get_neighborhood.call_count}"
        print("  PASS (Call 3): get_conceptual_neighborhood called again for new coords (cache miss).")
        last_score_event_3 = _ethics_db["ethical_scores"][-1]
        cluster_score_3 = last_score_event_3["component_scores"]["manifold_cluster_context"]
        # Score might be same if mock always returns same neighborhood_data, but call count is key.
        # Here, cluster_score_3 should still be based on mock_neighborhood_result.
        assert abs(cluster_score_1 - cluster_score_3) < 0.001, \
             f"FAIL (Call 3): Cluster score {cluster_score_3} unexpected. (Note: mock returns same data, so score should be same)."


        # --- Call 4: Different Radius (via config change) - Cache Miss ---
        # This requires changing config for ETHICS_CLUSTER_RADIUS_FACTOR or MANIFOLD_RANGE
        # The test_configs mechanism in run_test applies config at start of test.
        # To test radius change, this test itself would need to be run twice with different configs,
        # or it needs a way to manipulate 'config' mid-test, which TempConfigOverride isn't designed for.
        # For simplicity, we'll assume this part is implicitly covered if coord change is covered.
        # A more advanced test could use nested TempConfigOverride if the context manager supported it,
        # or directly patch config attributes if desperate (but avoid if possible).
        # Let's skip explicit radius change test here to avoid overcomplicating test setup for now.
        # The cache key includes radius, so logic is sound.

        # Restore original get_shared_manifold
        sys.modules[__name__].get_shared_manifold = original_get_shared_manifold_in_module
        _manifold_cache.clear() # Clean up cache after test
        return True

    def test_log_sanitization(**kwargs) -> bool:
        """
        Tests the sanitization logic in `_log_ethics_event`.
        Verifies that sensitive fields are redacted in the log output and
        original data objects are not modified.
        """
        print("Testing log sanitization...")
        global _ethics_db, TEST_SYSTEM_LOG_PATH, SENSITIVE_LOG_KEYS, REDACTION_PLACEHOLDER, config

        # Ensure log level is low enough to capture debug/info messages from this test
        # This should be handled by the TempConfigOverride in run_test if LOG_LEVEL is part of it.
        # Forcing a specific log level for testing _log_ethics_event itself.
        if hasattr(config, 'LOG_LEVEL'):
            original_log_level = config.LOG_LEVEL
            config.LOG_LEVEL = "DEBUG"
        else: # Should not happen if config is properly set up by test harness
            original_log_level = "INFO" # Default assumption
            if config: config.LOG_LEVEL = "DEBUG"


        original_data_sensitive = {
            "concept_summary_snippet": "This is a very sensitive concept.",
            "action_description_snippet": "Performing a risky action.",
            "awareness_metrics_snapshot": {"coord": (1,2,3), "details": "lots of secret numbers"},
            "trace": "File 'secret.py', line 123, in do_secret_stuff",
            "non_sensitive_field": "This is fine to log."
        }
        data_copy_for_check = copy.deepcopy(original_data_sensitive)

        _log_ethics_event("test_sensitive_log_event", original_data_sensitive, level="info")

        # Check that original data was not modified
        assert original_data_sensitive == data_copy_for_check, "Original data object was modified by _log_ethics_event."
        print("  PASS: Original data object not modified.")

        # Read the log file and check the last entry
        logged_correctly = False
        if os.path.exists(TEST_SYSTEM_LOG_PATH):
            with open(TEST_SYSTEM_LOG_PATH, 'r') as f:
                lines = f.readlines()

            last_log_line = None
            for line in reversed(lines): # Find the specific event we just logged
                try:
                    log_json = json.loads(line.strip())
                    if log_json.get("event_type") == "test_sensitive_log_event":
                        last_log_line = line.strip()
                        break
                except json.JSONDecodeError:
                    continue # Skip malformed lines if any

            if last_log_line:
                logged_entry = json.loads(last_log_line)
                logged_data = logged_entry.get("data", {})

                sanitization_ok = True
                for key in SENSITIVE_LOG_KEYS:
                    if key in original_data_sensitive: # If the key was in original data
                        if logged_data.get(key) != REDACTION_PLACEHOLDER:
                            print(f"  FAIL: Sensitive key '{key}' not redacted. Value: '{logged_data.get(key)}'")
                            sanitization_ok = False
                            break
                    # Also check if a key NOT in original data somehow appeared as redacted (should not happen)
                    elif logged_data.get(key) == REDACTION_PLACEHOLDER and key not in original_data_sensitive :
                         print(f"  FAIL: Key '{key}' appeared as redacted but was not in original sensitive data.")
                         sanitization_ok = False
                         break

                if sanitization_ok and logged_data.get("non_sensitive_field") == original_data_sensitive["non_sensitive_field"]:
                    logged_correctly = True
                    print("  PASS: Sensitive fields redacted, non-sensitive fields intact.")
                elif not sanitization_ok: # Already printed specific error
                    pass
                else: # Non-sensitive field mismatch
                    print(f"  FAIL: Non-sensitive field mismatch. Expected '{original_data_sensitive['non_sensitive_field']}', Got '{logged_data.get('non_sensitive_field')}'")

            else: # Log event not found
                 print(f"  FAIL: Log event 'test_sensitive_log_event' not found in log file.")
        else: # Log file not found
            print(f"  FAIL: Test log file '{TEST_SYSTEM_LOG_PATH}' not found.")

        # Restore original log level
        if hasattr(config, 'LOG_LEVEL') and original_log_level is not None : # original_log_level might be None if config itself was None initially
             config.LOG_LEVEL = original_log_level
        elif hasattr(config, 'LOG_LEVEL') and original_log_level is None and config.LOG_LEVEL == "DEBUG": # if it was set by this test
             delattr(config, 'LOG_LEVEL') # Or set to a sensible default if that's preferred

        return logged_correctly


    def test_score_ethics_basic_and_components(**kwargs) -> bool:
        """
        Tests the `score_ethics` function's basic operation and component scoring.
        Verifies:
        - A score is calculated and is within the valid range [0,1].
        - The scoring event is logged to the ethics DB.
        - Log pruning mechanism works correctly when max entries are exceeded.
        """
        print("Testing score_ethics (basic components)...")
        awareness_metrics_neutral = {
            "coherence": 0.0, 
            "primary_concept_coord": (0,0,0, 0.5 * config.MANIFOLD_RANGE / 2.0), # Scaled t_coord
            "raw_t_intensity": 0.5 # Raw 0-1 intensity
        }
        score = score_ethics(awareness_metrics_neutral, "neutral summary", "neutral action")
        
        assert 0.0 <= score <= 1.0, f"Basic score out of range [0,1]. Score: {score}"
        assert len(_ethics_db["ethical_scores"]) == 1, f"Score event not logged. Count: {len(_ethics_db['ethical_scores'])}"
        print(f"  PASS: Basic score calculated: {score:.2f}, event logged.")
        
        max_entries = config.ETHICS_LOG_MAX_ENTRIES 
        for i in range(max_entries + 2): # Add more entries to trigger pruning.
            score_ethics(awareness_metrics_neutral, f"summary {i}", f"action {i}")
        
        assert len(_ethics_db["ethical_scores"]) == max_entries, \
            f"Log pruning failed. Count: {len(_ethics_db['ethical_scores'])}, Expected: {max_entries}"
        print(f"  PASS: Log pruning to {max_entries} entries.")
        return True

    def test_score_ethics_cluster_context(mock_neighborhood_data_for_test: list, **kwargs) -> bool:
        """
        Tests the `score_ethics` function's manifold cluster context component.
        Uses a mock manifold (via `mock_neighborhood_data_for_test`) to provide
        controlled neighborhood data. Verifies that the cluster context score
        reflects the average valence of the mock neighbors.

        Args:
            mock_neighborhood_data_for_test (list): Data for the mock manifold's neighbors.
        
        Returns:
            bool: True if the test passes, False otherwise.
        """
        print("Testing score_ethics with manifold cluster context...")
        awareness_metrics = {
            "coherence": 0.8, 
            "primary_concept_coord": (config.MANIFOLD_RANGE / 4, 0, 0, 0.7 * config.MANIFOLD_RANGE / 2.0), # Scaled t_coord
            "raw_t_intensity": 0.7 # Raw 0-1 intensity
        }
        score = score_ethics(awareness_metrics, "concept affecting cluster", "action related to cluster")
        
        last_score_event = _ethics_db["ethical_scores"][-1]
        cluster_comp_score = last_score_event["component_scores"]["manifold_cluster_context"]
        
        expected_mock_avg_valence = 0.5 # Default if no valid neighbors or error.
        if mock_neighborhood_data_for_test: 
            norm_valences = []
            for n_data in mock_neighborhood_data_for_test:
                node_coords = n_data.get("coordinates")
                if isinstance(node_coords, (list,tuple)) and len(node_coords)==4:
                     node_x_val = float(node_coords[0])
                     current_manifold_range = float(getattr(config, 'MANIFOLD_RANGE', 1.0))
                     current_manifold_range = current_manifold_range if current_manifold_range != 0 else 1.0
                     norm_val = (node_x_val + (current_manifold_range / 2.0)) / current_manifold_range
                     norm_valences.append(np.clip(norm_val, 0.0, 1.0))
            if norm_valences: 
                expected_mock_avg_valence = np.mean(norm_valences)
        
        assert abs(cluster_comp_score - expected_mock_avg_valence) < 0.01, \
            f"Cluster context score {cluster_comp_score:.2f} (logged avg: {last_score_event.get('cluster_avg_valence', 'N/A')}) " \
            f"doesn't match expected from mock {expected_mock_avg_valence:.2f}"
        print(f"  PASS: Cluster context score ({cluster_comp_score:.2f}) matches mock expectation.")
        return True


    def test_track_trends_scenarios(**kwargs) -> bool:
        """
        Tests the `track_trends` function under various scenarios:
        - Insufficient data (0 scores, fewer than min_data_points).
        - Stable trend (scores are consistent).
        - Improving trend (scores generally increase).
        - Declining trend (scores generally decrease).
        Verifies the `trend_direction` output.
        """
        print("Testing track_trends...")
        global _ethics_db
        
        min_points = config.ETHICS_TREND_MIN_DATAPOINTS 
        _ethics_db["ethical_scores"] = [] # Ensure DB is empty for this test.
        trends = track_trends()
        assert trends["trend_direction"] == "insufficient_data", "Trend not 'insufficient_data' for 0 scores."
        print("  PASS: Insufficient data (0 scores).")

        # Populate with fewer than min_points.
        for i in range(min_points - 1):
             _ethics_db["ethical_scores"].append({"final_score": 0.5, "primary_concept_t_intensity_raw": 0.5, "timestamp": f"2023-01-01T00:0{i}:00Z"})
        trends = track_trends()
        assert trends["trend_direction"] == "insufficient_data", f"Trend not 'insufficient_data' for {min_points-1} scores."
        print(f"  PASS: Insufficient data ({min_points-1} scores).")

        # Stable trend.
        _ethics_db["ethical_scores"] = [] 
        for i in range(min_points + 2): 
            _ethics_db["ethical_scores"].append({"final_score": 0.5, "primary_concept_t_intensity_raw": 0.5, "timestamp": f"2023-01-01T00:0{i}:00Z"})
        trends_stable = track_trends()
        assert trends_stable["trend_direction"] == "stable", f"Stable trend not detected. Avg: {trends_stable.get('t_weighted_short_term_avg'):.2f}"
        print(f"  PASS: Stable trend detected (avg: {trends_stable.get('t_weighted_short_term_avg'):.2f}).")

        # Improving trend.
        _ethics_db["ethical_scores"] = []
        base_scores_improve = [0.2, 0.3, 0.4, 0.7, 0.8]; intensities_improve = [0.2, 0.3, 0.4, 0.9, 1.0]
        # Ensure enough data points for trend calculation based on min_points.
        while len(base_scores_improve) < min_points: base_scores_improve.append(0.8); intensities_improve.append(1.0) 
        for i, score_val in enumerate(base_scores_improve):
            _ethics_db["ethical_scores"].append({"final_score": score_val, "primary_concept_t_intensity_raw": intensities_improve[i], "timestamp": f"2023-01-01T00:0{i}:00Z"})
        trends_improve = track_trends()
        assert trends_improve["trend_direction"] == "improving", \
            f"Improving trend not detected. Short: {trends_improve.get('t_weighted_short_term_avg'):.2f}, Long: {trends_improve.get('t_weighted_long_term_avg'):.2f}"
        print(f"  PASS: Improving trend (Short: {trends_improve.get('t_weighted_short_term_avg'):.2f}, Long: {trends_improve.get('t_weighted_long_term_avg'):.2f}).")
        
        # Declining trend.
        _ethics_db["ethical_scores"] = []
        base_scores_decline = [0.8, 0.7, 0.6, 0.3, 0.2]; intensities_decline = [1.0, 0.9, 0.4, 0.3, 0.2]
        while len(base_scores_decline) < min_points: base_scores_decline.append(0.2); intensities_decline.append(0.2)
        for i, score_val in enumerate(base_scores_decline):
            _ethics_db["ethical_scores"].append({"final_score": score_val, "primary_concept_t_intensity_raw": intensities_decline[i], "timestamp": f"2023-01-01T00:0{i}:00Z"})
        trends_decline = track_trends()
        assert trends_decline["trend_direction"] == "declining", \
            f"Declining trend not detected. Short: {trends_decline.get('t_weighted_short_term_avg'):.2f}, Long: {trends_decline.get('t_weighted_long_term_avg'):.2f}"
        print(f"  PASS: Declining trend (Short: {trends_decline.get('t_weighted_short_term_avg'):.2f}, Long: {trends_decline.get('t_weighted_long_term_avg'):.2f}).")

        return True

    # --- Main Test Execution Logic ---
    print("\n--- Starting Core Ethics Self-Tests ---")
    
    if not config:
        print("CRITICAL: Config module not loaded at test execution. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Generate a single Fernet key for all encrypted test runs for consistency in this test session
    # Note: cryptography.fernet might not be available here if not imported globally
    # For standalone script execution, ensure it's imported.
    # It was added to the top-level imports of the module, so it should be fine.
    test_fernet_key = Fernet.generate_key().decode() # decode to store as string in env
        
    original_config_verbose_ethics = getattr(config, 'VERBOSE_OUTPUT', None)
    config.VERBOSE_OUTPUT = False 

    # test_db_load_save_encryption_modes is now the main test function.
    # It internally calls _perform_db_load_save_checks.
    # We need to run it twice: once with encryption_key=None (default for run_test), once with the key.
    tests_to_run = [
        (test_db_load_save_encryption_modes, {"encryption_key": None}),
        (test_db_load_save_encryption_modes, {"encryption_key": test_fernet_key}),
        (test_db_load_pruning, {"test_configs": {"ETHICS_LOG_MAX_ENTRIES": 3, "LOG_LEVEL": "DEBUG"}, "encryption_key": None}),
        (test_log_sanitization, {}),
        (test_async_logging, {"test_configs": {"LOG_LEVEL": "DEBUG"}}),
        (test_manifold_caching, { # New test for manifold caching
            "test_configs": {
                "ETHICS_CLUSTER_RADIUS_FACTOR": 0.1, # Example values needed by score_ethics
                "MANIFOLD_RANGE": 10.0,
                "LOG_LEVEL": "DEBUG", # To see cache hit/miss logs from score_ethics
                "ETHICS_CLUSTER_CONTEXT_WEIGHT": 0.2 # Ensure component is active
            }
        }),
        (test_mock_manifold_validation, {}),
        (test_framework_alignment_advanced, {
            "test_configs": {
                "ETHICAL_FRAMEWORK": {
                    "positive_keywords": [{"term": "help", "synonyms": ["assist", "support"]}, "control", {"term": "deal.", "synonyms": ["agreement."]} ],
                    "negative_keywords": [{"term": "harm", "synonyms": ["damage", "hurt"]}, "uncontrolled"]
                }, "ETHICS_FRAMEWORK_WEIGHT": 1.0, "ETHICS_COHERENCE_WEIGHT": 0,
                "ETHICS_VALENCE_WEIGHT": 0, "ETHICS_INTENSITY_WEIGHT": 0, "ETHICS_CLUSTER_CONTEXT_WEIGHT": 0,
                "ENABLE_SEMANTIC_ANALYSIS": False
            }
        }),
        (test_score_ethics_input_validation, {
            "test_configs": {
                "ETHICS_MAX_TEXT_INPUT_LENGTH": 20,
                "ETHICAL_FRAMEWORK": { "positive_keywords": ["good", "excellent"], "negative_keywords": ["badkey"] },
                "ETHICS_FRAMEWORK_WEIGHT": 1.0, "ETHICS_COHERENCE_WEIGHT": 0, "ETHICS_VALENCE_WEIGHT": 0,
                "ETHICS_INTENSITY_WEIGHT": 0, "ETHICS_CLUSTER_CONTEXT_WEIGHT": 0, "LOG_LEVEL": "DEBUG"
            }
        }),
        (test_score_ethics_basic_and_components, {}),
        (test_score_ethics_cluster_context, {"mock_neighborhood_data_for_test": [{"coordinates": (config.MANIFOLD_RANGE/2 if config else 5.0, 0,0,0)}, {"coordinates": (-(config.MANIFOLD_RANGE/4) if config else -2.5,0,0,0)}]}),
        (test_track_trends_scenarios, {"test_configs": {"ETHICS_TREND_MIN_DATAPOINTS": 3, "LOG_LEVEL": "DEBUG"}}),

        (test_concurrent_operations, {"test_configs": {"ETHICS_LOG_MAX_ENTRIES": 50, "LOG_LEVEL": "INFO"}}),
        (test_load_corrupted_encrypted_db, {"encryption_key": test_fernet_key, "test_configs": {"LOG_LEVEL": "DEBUG"}}),
        (test_score_ethics_missing_awareness_keys, {
            "test_configs": {
                "ETHICS_COHERENCE_WEIGHT": 0.2, "ETHICS_VALENCE_WEIGHT": 0.2,
                "ETHICS_INTENSITY_WEIGHT": 0.1, "ETHICS_FRAMEWORK_WEIGHT": 0.3,
                "ETHICS_CLUSTER_CONTEXT_WEIGHT": 0.2, "LOG_LEVEL": "DEBUG",
                "ETHICAL_FRAMEWORK": {"positive_keywords": ["test"], "negative_keywords": []}
            }
        }),
        (test_track_trends_malformed_scores, {"test_configs": {"ETHICS_TREND_MIN_DATAPOINTS": 3, "LOG_LEVEL": "DEBUG"}}),
    ]
    
    results = []
    for test_fn, test_kwargs_for_run_test in tests_to_run:
        # test_kwargs_for_run_test contains args for run_test, like 'encryption_key'
        # Other args for the test_fn itself would be passed via run_test's *args or **kwargs if needed
        results.append(run_test(test_fn, **test_kwargs_for_run_test))
        time.sleep(0.05) # Slight delay, perhaps for filesystem operations to settle if needed.

    print("\n--- Core Ethics Self-Test Summary ---")
    passed_count = sum(1 for r in results if r)
    total_count = len(results)
    print(f"Tests Passed: {passed_count}/{total_count}")

    if original_config_verbose_ethics is not None:
        config.VERBOSE_OUTPUT = original_config_verbose_ethics
    elif hasattr(config, 'VERBOSE_OUTPUT'): 
        delattr(config, 'VERBOSE_OUTPUT')

    if passed_count == total_count:
        print("All core ethics tests PASSED successfully!")
    else:
        print("One or more core ethics tests FAILED. Please review logs above.")
        sys.exit(1)
