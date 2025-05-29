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
import sys
import traceback # Promoted to top-level

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

# Further module-level constants or setup can go here.

# --- Module-Level Logging ---
LOG_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40, "critical": 50}

def _log_ethics_event(event_type: str, data: dict, level: str = "info"):
    """
    Logs a structured system event from the ethics module.
    Data is serialized to JSON. Respects LOG_LEVEL from config.
    """
    # Check if config and necessary attributes are available
    if not config or not hasattr(config, 'SYSTEM_LOG_PATH') or not hasattr(config, 'LOG_LEVEL') or not hasattr(config, 'ensure_path'):
        # Fallback to print if essential config for logging is missing
        # Avoid using _log_ethics_event itself here to prevent recursion if it's the source of the config problem.
        print(f"ETHICS_EVENT_LOG_CONFIG_ERROR ({level.upper()}): {event_type} - Data: {json.dumps(data, default=str)}"
              f" - Reason: Config not fully available for logging.", file=sys.stderr)
        return

    try:
        numeric_level = LOG_LEVELS.get(level.lower(), LOG_LEVELS["info"])
        config_numeric_level = LOG_LEVELS.get(config.LOG_LEVEL.lower(), LOG_LEVELS["info"])

        if numeric_level < config_numeric_level:
            return # Skip logging if event level is below configured log level

        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "module": "ethics",
            "event_type": event_type,
            "level": level.upper(),
            "data": data
        }
        
        log_file_path = config.SYSTEM_LOG_PATH
        # Ensure the log directory exists. config.ensure_path expects a file path to ensure its parent dir.
        config.ensure_path(log_file_path) 

        with open(log_file_path, 'a') as f:
            f.write(json.dumps(log_entry, default=str) + '\n') # Use default=str for non-serializable data
            
    except Exception as e:
        # Fallback to print if logging to file fails
        print(f"ETHICS_EVENT_LOG_FILE_ERROR ({level.upper()}): {event_type} - Data: {json.dumps(data, default=str)}"
              f" - Error: {e}", file=sys.stderr)
        # Log the logging error itself, carefully to avoid recursion if this function is called again.
        if event_type != "logging_error_internal": # Avoid direct recursion
            # Construct minimal data for the error log to reduce risk of further serialization issues
            error_data = {"original_event": event_type, "logging_error_message": str(e)}
            if config and hasattr(config, 'SYSTEM_LOG_PATH'): # Check again if config is usable for this specific error log
                 _log_ethics_event("logging_error_internal", error_data, level="critical")
            else:
                 print(f"ETHICS_EVENT_LOG_CRITICAL_FAILURE: Cannot log internal logging error due to missing config. Original event: {event_type}", file=sys.stderr)

# --- Ethical Database State & Management ---
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
    
    # Critical check for config availability for database path and directory creation.
    if not config or not hasattr(config, 'ETHICS_DB_PATH') or not hasattr(config, 'ensure_path'):
        _log_ethics_event("load_ethics_db_failure", {"error": "Config not available or ETHICS_DB_PATH/ensure_path not set"}, level="critical")
        _ethics_db = {"ethical_scores": [], "trend_analysis": {}} # Default structure
        _ethics_db_dirty_flag = False # In-memory state is this default, so not "dirty".
        return

    db_path = config.ETHICS_DB_PATH
    config.ensure_path(db_path) # Ensures the directory for the DB file exists.

    try:
        # If DB file doesn't exist or is empty, initialize with a default structure.
        if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
            _log_ethics_event("load_ethics_db_info", {"message": "Ethics DB file not found or empty. Initializing new DB.", "path": db_path}, level="info")
            _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
            _ethics_db_dirty_flag = False # Freshly initialized, so not dirty.
            return

        # Attempt to open and load JSON data from the existing DB file.
        with open(db_path, 'r') as f:
            data = json.load(f)

        # Validate the basic structure of the loaded data.
        if isinstance(data, dict) and \
           "ethical_scores" in data and isinstance(data["ethical_scores"], list) and \
           "trend_analysis" in data and isinstance(data["trend_analysis"], dict):
            _ethics_db = data # Assign loaded data to the global DB variable.
            _ethics_db_dirty_flag = False # Successfully loaded, so in-memory is synchronized.
            _log_ethics_event("load_ethics_db_success", 
                              {"path": db_path, "scores_loaded": len(data["ethical_scores"])}, 
                              level="info")
        else: # Data does not conform to the expected structure.
            _log_ethics_event("load_ethics_db_malformed_structure", 
                              {"path": db_path, "error": "Root must be dict with 'ethical_scores' (list) and 'trend_analysis' (dict)."}, 
                              level="error")
            _ethics_db = {"ethical_scores": [], "trend_analysis": {}} # Reset to default.
            _ethics_db_dirty_flag = False # Defaulted, so considered clean.
            
    except json.JSONDecodeError as e: # Handle errors if JSON is invalid.
        _log_ethics_event("load_ethics_db_json_decode_error", {"path": db_path, "error": str(e)}, level="error")
        _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
        _ethics_db_dirty_flag = False
    except Exception as e: # Catch any other unexpected errors during file operations.
        _log_ethics_event("load_ethics_db_unknown_error", {"path": db_path, "error": str(e), "trace": traceback.format_exc()}, level="critical")
        _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
        _ethics_db_dirty_flag = False

def _save_ethics_db():
    """
    Saves the current state of `_ethics_db` to disk if `_ethics_db_dirty_flag` is True.
    Uses an atomic write (write to temp file, then replace original) to prevent data corruption.
    Resets `_ethics_db_dirty_flag` to False after a successful save.
    """
    global _ethics_db_dirty_flag

    if not _ethics_db_dirty_flag: # Only proceed if there are changes to save.
        _log_ethics_event("save_ethics_db_skipped", {"message": "No changes to save (_ethics_db_dirty_flag is False)."}, level="debug")
        return

    # Critical check for config availability.
    if not config or not hasattr(config, 'ETHICS_DB_PATH') or not hasattr(config, 'ensure_path'):
        _log_ethics_event("save_ethics_db_failure", {"error": "Config not available or ETHICS_DB_PATH/ensure_path not set"}, level="critical")
        # Do not reset dirty flag; changes are still pending.
        return

    db_path = config.ETHICS_DB_PATH
    config.ensure_path(db_path) # Ensure the directory for the DB file exists.

    temp_db_path = db_path + ".tmp" # Path for the temporary file.
    try:
        # Step 1: Write the current database to the temporary file.
        # `default=str` handles potential non-serializable items like datetime objects if not already ISO strings.
        with open(temp_db_path, 'w') as f:
            json.dump(_ethics_db, f, indent=4, default=str) 
        
        # Step 2: Atomically replace the original DB file with the temporary file.
        # os.replace is generally atomic and preferred over os.rename if destination might exist.
        os.replace(temp_db_path, db_path)

        _ethics_db_dirty_flag = False # Reset dirty flag only after successful write and replacement.
        _log_ethics_event("save_ethics_db_success", 
                          {"path": db_path, "scores_saved": len(_ethics_db["ethical_scores"])}, 
                          level="info")
    
    except IOError as e_io: # Handle file I/O errors (e.g., permission issues).
        _log_ethics_event("save_ethics_db_io_error", {"path": db_path, "temp_path": temp_db_path, "error": str(e_io), "trace": traceback.format_exc()}, level="critical")
        # Attempt to clean up the temporary file if it exists and an error occurred.
        if os.path.exists(temp_db_path):
            try: os.remove(temp_db_path)
            except Exception as e_rm: _log_ethics_event("save_ethics_db_temp_cleanup_error", {"path": temp_db_path, "error": str(e_rm)}, level="error")
    except Exception as e: # Handle other unexpected errors during the save process.
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

    # Prepare data for logging this scoring event.
    event_data_for_log = {
        "awareness_metrics_snapshot": awareness_metrics, # Full snapshot for traceability.
        "concept_summary_snippet": concept_summary[:100] if concept_summary else "", # Log a snippet.
        "action_description_snippet": action_description[:100] if action_description else "" # Log a snippet.
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
    positive_keywords = ethical_framework_config.get("positive_keywords", ["help", "improve", "assist", "share", "create", "understand", "align", "benefit"])
    negative_keywords = ethical_framework_config.get("negative_keywords", ["harm", "deceive", "exploit", "manipulate", "destroy", "control", "damage"])
    
    pos_score_count = sum(1 for kw in positive_keywords if kw in text_to_analyze)
    neg_score_count = sum(1 for kw in negative_keywords if kw in text_to_analyze)
    
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
            manifold_instance = get_shared_manifold() # Fetch the (potentially mocked) manifold instance.
            if manifold_instance and hasattr(manifold_instance, 'get_conceptual_neighborhood'):
                # Parameters for neighborhood query from config.
                try:
                    radius_factor = float(getattr(config, 'ETHICS_CLUSTER_RADIUS_FACTOR', 0.1))
                except (ValueError, TypeError) as e_rf:
                    _log_ethics_event("score_ethics_config_error", {"parameter": "ETHICS_CLUSTER_RADIUS_FACTOR", "value": getattr(config, 'ETHICS_CLUSTER_RADIUS_FACTOR', 'NotSet'), "error": str(e_rf)}, level="warning")
                    radius_factor = 0.1 # Default
                try:
                    manifold_range_for_radius = float(getattr(config, 'MANIFOLD_RANGE', 1.0))
                    if manifold_range_for_radius == 0: manifold_range_for_radius = 1.0 # Avoid zero radius
                except (ValueError, TypeError) as e_mr:
                    _log_ethics_event("score_ethics_config_error", {"parameter": "MANIFOLD_RANGE", "value": getattr(config, 'MANIFOLD_RANGE', 'NotSet'), "error": str(e_mr)}, level="warning")
                    manifold_range_for_radius = 1.0 # Default
                
                radius = radius_factor * manifold_range_for_radius
                
                neighborhood_nodes = manifold_instance.get_conceptual_neighborhood(primary_coord, radius)
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
        # Ensure the 'ethical_scores' list exists in the DB.
        if not isinstance(_ethics_db.get("ethical_scores"), list): 
            _ethics_db["ethical_scores"] = [] # Initialize if missing or wrong type.
            _log_ethics_event("score_ethics_db_reinit_scores_list", {"message": "'ethical_scores' list re-initialized in _ethics_db."}, level="warning")

        _ethics_db["ethical_scores"].append(event_data_for_log) # Add current score event.
        
        # Prune older entries if the log exceeds max_entries from config.
        max_entries = int(getattr(config, 'ETHICS_LOG_MAX_ENTRIES', 1000))
        if len(_ethics_db["ethical_scores"]) > max_entries:
            num_to_remove = len(_ethics_db["ethical_scores"]) - max_entries
            _ethics_db["ethical_scores"] = _ethics_db["ethical_scores"][num_to_remove:] # Keep most recent entries.
            _log_ethics_event("score_ethics_log_pruned", {"removed_count": num_to_remove, "new_count": len(_ethics_db["ethical_scores"])}, level="debug")

        _ethics_db_dirty_flag = True # Mark DB as changed.
        _save_ethics_db() # Attempt to save immediately.
    except Exception as e_db_persist: # Catch errors during DB interaction.
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
    global _ethics_db_dirty_flag # To mark DB as changed after updating trend_analysis.

    _log_ethics_event("track_trends_start", {}, level="debug")

    if not config: # Config is needed for trend parameters.
        _log_ethics_event("track_trends_failure", {"error": "Config module not loaded, cannot track trends."}, level="critical")
        return {"trend_direction": "error_config_missing", "last_updated": datetime.datetime.utcnow().isoformat() + "Z"}

    # Minimum number of data points required to perform trend analysis, from config.
    min_data_points = int(getattr(config, 'ETHICS_TREND_MIN_DATAPOINTS', 10)) 
    
    # Validate the ethical_scores data in the database.
    ethical_scores_data = _ethics_db.get("ethical_scores")
    if not isinstance(ethical_scores_data, list) or len(ethical_scores_data) == 0: # Check if list is empty too.
        _log_ethics_event("track_trends_insufficient_data", 
                          {"count": len(ethical_scores_data) if isinstance(ethical_scores_data, list) else "N/A (not a list)", 
                           "min_required": min_data_points, "reason": "No score data available."}, 
                          level="info")
        current_trends = {
            "data_points_used": len(ethical_scores_data) if isinstance(ethical_scores_data, list) else 0,
            "trend_direction": "insufficient_data",
            "last_updated": datetime.datetime.utcnow().isoformat() + "Z"
        }
        _ethics_db["trend_analysis"] = current_trends # Update DB with this status.
        _ethics_db_dirty_flag = True 
        _save_ethics_db() # Persist.
        return current_trends

    # Extract valid scores and their associated raw T-intensities for weighting.
    # Raw T-intensity (0-1) is expected to be logged with each score event from `score_ethics`.
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

    _ethics_db["trend_analysis"] = current_trends_summary # Update the database with new trend analysis.
    _ethics_db_dirty_flag = True # Mark DB as changed.
    _save_ethics_db() # Persist changes.

    _log_ethics_event("track_trends_complete", current_trends_summary, level="info")
    return current_trends_summary

# --- Load ethics database at module import ---
_load_ethics_db()

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
            self.neighborhood_data = neighborhood_data if neighborhood_data is not None else []
            self.device = "cpu" # Mock device attribute, sometimes checked by other parts.

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

    def setup_test_environment(test_specific_configs: dict = None, mock_neighborhood: list = None):
        """
        Prepares the testing environment for ethics module tests.

        This involves:
        1.  Cleaning up any existing test database or log files.
        2.  Resetting the in-memory `_ethics_db` and `_ethics_db_dirty_flag`.
        3.  Monkeypatching `get_shared_manifold` (if not already done or if mock_neighborhood changes)
            to use `mock_get_shared_manifold_for_ethics_test` for controlling manifold interactions.
        4.  Constructing a configuration dictionary for `TempConfigOverride`, including paths
            to test-specific files and default test settings for ethics parameters.

        Args:
            test_specific_configs (dict, optional): Configuration overrides specific to the current test.
            mock_neighborhood (list, optional): Data for the mock manifold's neighborhood.

        Returns:
            TempConfigOverride: An instance of the context manager with test configurations.
        """
        global _ethics_db, _ethics_db_dirty_flag, original_get_shared_manifold
        
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
        sys.modules[__name__].get_shared_manifold = lambda force_recreate=False, current_mock_neighborhood=mock_neighborhood: \
            mock_get_shared_manifold_for_ethics_test(neighborhood_data=current_mock_neighborhood, force_recreate=force_recreate)

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
        Removes test files and restores the original `get_shared_manifold` function.
        """
        global original_get_shared_manifold
        if os.path.exists(TEST_ETHICS_DB_PATH):
            os.remove(TEST_ETHICS_DB_PATH)
        if os.path.exists(TEST_SYSTEM_LOG_PATH):
            os.remove(TEST_SYSTEM_LOG_PATH)
        
        if original_get_shared_manifold is not None:
            sys.modules[__name__].get_shared_manifold = original_get_shared_manifold
            original_get_shared_manifold = None # Reset for next potential setup sequence.


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
        
        # Use a try-finally block to ensure cleanup_test_environment is always called.
        try:
            # setup_test_environment returns a TempConfigOverride instance.
            # The `with` statement handles its __enter__ and __exit__ for config management.
            with setup_test_environment(test_specific_configs=test_configs_override, 
                                        mock_neighborhood=mock_neighborhood_data_for_test):
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

    def test_db_load_save_and_dirty_flag() -> bool:
        """
        Tests the ethics database loading, saving, and dirty flag logic.
        Verifies:
        - Initial load of a non-existent file creates an empty DB state.
        - Saving is skipped if the dirty flag is not set.
        - Modifying the DB, setting the dirty flag, and saving creates the file.
        - Reloading the saved DB correctly restores its content.
        """
        print("Testing DB load/save and dirty flag logic...")
        global _ethics_db, _ethics_db_dirty_flag

        # 1. Initial load (file non-existent)
        # _load_ethics_db() is called by the `with setup_test_environment(...)` context manager.
        assert _ethics_db == {"ethical_scores": [], "trend_analysis": {}}, "Initial DB state not empty."
        assert not _ethics_db_dirty_flag, "Dirty flag not false on initial load."
        print("  PASS: Initial load non-existent file.")

        # 2. Save when not dirty (should skip writing the file).
        _save_ethics_db()
        assert not os.path.exists(TEST_ETHICS_DB_PATH), "DB file created on save when not dirty."
        print("  PASS: Save skipped when not dirty.")

        # 3. Modify DB, set dirty flag, and save.
        _ethics_db["ethical_scores"].append({"test_score": 1, "primary_concept_t_intensity_raw": 0.5, "timestamp": "2023-01-01T00:00:00Z"})
        _ethics_db_dirty_flag = True
        _save_ethics_db()
        assert os.path.exists(TEST_ETHICS_DB_PATH), "DB not saved when dirty."
        assert not _ethics_db_dirty_flag, "Dirty flag not reset after save."
        print("  PASS: DB saved when dirty, flag reset.")
        
        # 4. Load saved DB.
        _ethics_db = {} # Clear in-memory to force reload.
        _load_ethics_db() # This will load from TEST_ETHICS_DB_PATH.
        assert _ethics_db.get("ethical_scores"), "Ethical scores not loaded."
        assert _ethics_db["ethical_scores"][0].get("test_score") == 1, "Loaded DB content mismatch."
        print("  PASS: Reloaded saved DB correctly.")
        return True

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
        
    original_config_verbose_ethics = getattr(config, 'VERBOSE_OUTPUT', None)
    config.VERBOSE_OUTPUT = False 

    tests_to_run = [
        (test_db_load_save_and_dirty_flag, {}),
        (test_score_ethics_basic_and_components, {}),
        (test_score_ethics_cluster_context, {"mock_neighborhood_data_for_test": [{"coordinates": (config.MANIFOLD_RANGE/2, 0,0,0)}, {"coordinates": (-config.MANIFOLD_RANGE/4,0,0,0)}]}),
        (test_track_trends_scenarios, {}),
    ]
    
    results = []
    for test_fn, test_kwargs in tests_to_run:
        results.append(run_test(test_fn, **test_kwargs))
        time.sleep(0.05)

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
