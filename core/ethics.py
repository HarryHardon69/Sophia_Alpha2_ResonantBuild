"""
Core ethics module for Sophia_Alpha2.

This module is responsible for:
- Scoring the ethical alignment of concepts, actions, and system states.
- Tracking ethical trends over time using T-weighted analysis.
- Managing a persistent database for ethical scores and trend data.
- Providing a framework for guiding Sophia_Alpha2's actions based on
  configurable ethical principles.
"""

import os
import sys
import json
import datetime
import numpy as np # For numerical operations, e.g., weighted averages, std dev in trends

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
    Sets _ethics_db_dirty_flag to False.
    """
    global _ethics_db, _ethics_db_dirty_flag
    # Ensure traceback is imported for full error details, if not already at module level
    import traceback
    
    if not config or not hasattr(config, 'ETHICS_DB_PATH') or not hasattr(config, 'ensure_path'):
        _log_ethics_event("load_ethics_db_failure", {"error": "Config not available or ETHICS_DB_PATH/ensure_path not set"}, level="critical")
        _ethics_db = {"ethical_scores": [], "trend_analysis": {}} # Default structure
        _ethics_db_dirty_flag = False
        return

    db_path = config.ETHICS_DB_PATH
    config.ensure_path(db_path) # Ensure directory exists

    try:
        if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
            _log_ethics_event("load_ethics_db_info", {"message": "Ethics DB file not found or empty. Initializing new DB.", "path": db_path}, level="info")
            _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
            _ethics_db_dirty_flag = False 
            return

        with open(db_path, 'r') as f:
            data = json.load(f)

        # Validate structure
        if isinstance(data, dict) and            "ethical_scores" in data and isinstance(data["ethical_scores"], list) and            "trend_analysis" in data and isinstance(data["trend_analysis"], dict):
            _ethics_db = data
            _ethics_db_dirty_flag = False
            _log_ethics_event("load_ethics_db_success", 
                              {"path": db_path, "scores_loaded": len(data["ethical_scores"])}, 
                              level="info")
        else:
            _log_ethics_event("load_ethics_db_malformed_structure", 
                              {"path": db_path, "error": "Root must be dict with 'ethical_scores' (list) and 'trend_analysis' (dict)."}, 
                              level="error")
            _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
            _ethics_db_dirty_flag = False
            
    except json.JSONDecodeError as e:
        _log_ethics_event("load_ethics_db_json_decode_error", {"path": db_path, "error": str(e)}, level="error")
        _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
        _ethics_db_dirty_flag = False
    except Exception as e:
        _log_ethics_event("load_ethics_db_unknown_error", {"path": db_path, "error": str(e), "trace": traceback.format_exc()}, level="critical")
        _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
        _ethics_db_dirty_flag = False

def _save_ethics_db():
    """
    Saves the current state of _ethics_db to disk if changes have been made.
    Uses the _ethics_db_dirty_flag. Sets flag to False after successful save.
    Uses atomic writes.
    """
    global _ethics_db_dirty_flag
    # Ensure traceback is imported for full error details, if not already at module level
    import traceback

    if not _ethics_db_dirty_flag:
        _log_ethics_event("save_ethics_db_skipped", {"message": "No changes to save."}, level="debug")
        return

    if not config or not hasattr(config, 'ETHICS_DB_PATH') or not hasattr(config, 'ensure_path'):
        _log_ethics_event("save_ethics_db_failure", {"error": "Config not available or ETHICS_DB_PATH/ensure_path not set"}, level="critical")
        return

    db_path = config.ETHICS_DB_PATH
    config.ensure_path(db_path) # Ensure directory exists

    try:
        temp_db_path = db_path + ".tmp"
        with open(temp_db_path, 'w') as f:
            json.dump(_ethics_db, f, indent=4, default=str) # Use default=str for any non-serializable data
        
        if sys.platform == "win32":
            os.replace(temp_db_path, db_path)
        else:
            os.rename(temp_db_path, db_path)

        _ethics_db_dirty_flag = False
        _log_ethics_event("save_ethics_db_success", 
                          {"path": db_path, "scores_saved": len(_ethics_db["ethical_scores"])}, 
                          level="info")
    
    except IOError as e_io:
        _log_ethics_event("save_ethics_db_io_error", {"path": db_path, "error": str(e_io), "trace": traceback.format_exc()}, level="critical")
        if os.path.exists(temp_db_path):
            try: os.remove(temp_db_path)
            except Exception as e_rm: _log_ethics_event("save_ethics_db_temp_cleanup_error", {"path": temp_db_path, "error": str(e_rm)}, level="error")
    except Exception as e:
        _log_ethics_event("save_ethics_db_unknown_error", {"path": db_path, "error": str(e), "trace": traceback.format_exc()}, level="critical")
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
    global _ethics_db_dirty_flag
    # Ensure traceback is imported for full error details, if not already at module level
    import traceback 

    if not config:
        _log_ethics_event("score_ethics_failure", {"error": "Config not loaded"}, level="critical")
        return 0.0 # Default low score if config is missing

    event_data_for_log = {
        "awareness_metrics_snapshot": awareness_metrics,
        "concept_summary_snippet": concept_summary[:100] if concept_summary else "",
        "action_description_snippet": action_description[:100] if action_description else ""
    }
    _log_ethics_event("score_ethics_start", event_data_for_log, level="debug")

    scores = {}
    weights = {}

    # 1. Coherence Score (0-1, where 1 is perfect coherence)
    coherence_val = float(awareness_metrics.get("coherence", 0.0)) 
    scores["coherence"] = np.clip(1.0 - abs(coherence_val), 0.0, 1.0)
    weights["coherence"] = float(getattr(config, 'ETHICS_COHERENCE_WEIGHT', 0.2))

    # 2. Manifold Valence Score (0-1, where 1 is positive valence)
    primary_coord = awareness_metrics.get("primary_concept_coord")
    # Assuming awareness_metrics might contain 'raw_t_intensity' (0-1) for trend analysis.
    # This was planned for brain.think() to return.
    raw_t_intensity_for_trend = float(awareness_metrics.get("raw_t_intensity", 0.0)) 
    
    if primary_coord and isinstance(primary_coord, (list, tuple)) and len(primary_coord) == 4:
        try:
            x_valence_coord = float(primary_coord[0])
            manifold_range = float(getattr(config, 'MANIFOLD_RANGE', 0.0)) # Default to 0 if not set
            if manifold_range != 0:
                scores["manifold_valence"] = np.clip((x_valence_coord + (manifold_range / 2.0)) / manifold_range, 0.0, 1.0)
            else:
                scores["manifold_valence"] = 0.5 # Neutral if range is zero
            event_data_for_log["primary_concept_t_intensity_raw"] = raw_t_intensity_for_trend
        except (ValueError, TypeError):
            scores["manifold_valence"] = 0.5 
            _log_ethics_event("score_ethics_coord_error", {"coord": primary_coord}, level="warning")
    else:
        scores["manifold_valence"] = 0.5
    weights["manifold_valence"] = float(getattr(config, 'ETHICS_VALENCE_WEIGHT', 0.2))

    # 3. Manifold Intensity Preference Score (0-1, where 1 is optimal intensity e.g. 0.5)
    sigma_intensity_pref = 0.25 
    scores["intensity_preference"] = np.clip(np.exp(-((raw_t_intensity_for_trend - 0.5)**2) / (2 * sigma_intensity_pref**2)), 0.0, 1.0)
    weights["intensity_preference"] = float(getattr(config, 'ETHICS_INTENSITY_WEIGHT', 0.1))

    # 4. Framework Alignment Score (placeholder keyword analysis)
    text_to_analyze = (str(concept_summary) + " " + str(action_description)).lower()
    # Use ETHICAL_FRAMEWORK from config, which is a dict of principles with weights and descriptions
    # For keyword analysis, we'll look for keywords in descriptions or define specific keyword lists in config.
    # For simplicity, let's assume config.ETHICAL_FRAMEWORK might have 'positive_keywords' and 'negative_keywords' lists.
    ethical_framework_config = getattr(config, 'ETHICAL_FRAMEWORK', {}) # Default to empty dict
    positive_keywords = ethical_framework_config.get("positive_keywords", ["help", "improve", "assist", "share", "create", "understand", "align"])
    negative_keywords = ethical_framework_config.get("negative_keywords", ["harm", "deceive", "exploit", "manipulate", "destroy", "control"])
    
    pos_score = sum(1 for kw in positive_keywords if kw in text_to_analyze)
    neg_score = sum(1 for kw in negative_keywords if kw in text_to_analyze)
    
    if pos_score + neg_score > 0:
        scores["framework_alignment"] = np.clip(pos_score / (pos_score + neg_score), 0.0, 1.0)
    else:
        scores["framework_alignment"] = 0.5 
    weights["framework_alignment"] = float(getattr(config, 'ETHICS_FRAMEWORK_WEIGHT', 0.3))

    # 5. Manifold Cluster Context Score
    cluster_score = 0.5 
    if primary_coord and isinstance(primary_coord, (list, tuple)) and len(primary_coord) == 4:
        try:
            manifold_instance = get_shared_manifold() # This might be a mock if brain.py is not fully functional
            if manifold_instance and hasattr(manifold_instance, 'get_conceptual_neighborhood'):
                radius_factor = float(getattr(config, 'ETHICS_CLUSTER_RADIUS_FACTOR', 0.1))
                manifold_range = float(getattr(config, 'MANIFOLD_RANGE', 1.0)) # Default range to 1 to avoid div by zero if not set
                radius = radius_factor * manifold_range 
                
                neighborhood_nodes = manifold_instance.get_conceptual_neighborhood(primary_coord, radius)
                event_data_for_log["cluster_neighborhood_size"] = len(neighborhood_nodes)
                if neighborhood_nodes:
                    neighbor_valences = []
                    for node_data in neighborhood_nodes: 
                        if isinstance(node_data, dict) and isinstance(node_data.get('coordinates'), (list,tuple)) and len(node_data['coordinates']) == 4:
                             node_x_valence = float(node_data['coordinates'][0])
                             if manifold_range != 0:
                                 normalized_valence = (node_x_valence + (manifold_range / 2.0)) / manifold_range
                                 neighbor_valences.append(np.clip(normalized_valence, 0.0, 1.0))
                             else:
                                 neighbor_valences.append(0.5) 
                    
                    if neighbor_valences:
                        avg_neighbor_valence = np.mean(neighbor_valences)
                        cluster_score = avg_neighbor_valence
                        event_data_for_log["cluster_avg_valence"] = avg_neighbor_valence
                    else: 
                         event_data_for_log["cluster_avg_valence"] = "N/A (no valid valences in neighborhood)"
                else: 
                     event_data_for_log["cluster_avg_valence"] = "N/A (no neighbors found)"
            else: # Manifold instance or method not available (likely mock or import issue)
                _log_ethics_event("score_ethics_manifold_unavailable_for_cluster", 
                                  {"reason": "get_shared_manifold() returned None or no get_conceptual_neighborhood method."}, 
                                  level="warning")
                event_data_for_log["cluster_score_status"] = "Manifold or method unavailable"
        except AttributeError as e_attr: 
            _log_ethics_event("score_ethics_neighborhood_method_missing", {"error": str(e_attr)}, level="warning")
            event_data_for_log["cluster_score_status"] = f"Neighborhood method missing: {e_attr}"
        except Exception as e_cluster:
            _log_ethics_event("score_ethics_cluster_error", {"error": str(e_cluster), "trace": traceback.format_exc()}, level="error")
            event_data_for_log["cluster_score_status"] = f"Error: {e_cluster}"
            
    scores["manifold_cluster_context"] = np.clip(cluster_score, 0.0, 1.0)
    weights["manifold_cluster_context"] = float(getattr(config, 'ETHICS_CLUSTER_CONTEXT_WEIGHT', 0.2))

    # --- Final Score Calculation ---
    weighted_sum = 0
    total_weight = 0
    for key in scores:
        weighted_sum += scores[key] * weights.get(key, 0) 
        total_weight += weights.get(key, 0)

    if total_weight == 0:
        final_score = 0.0 
        _log_ethics_event("score_ethics_warning", {"warning": "Total weight for ethical components is zero."}, level="warning")
    else:
        final_score = weighted_sum / total_weight
    
    final_score = np.clip(final_score, 0.0, 1.0)
    event_data_for_log["component_scores"] = scores
    event_data_for_log["component_weights"] = weights
    event_data_for_log["final_score"] = final_score
    event_data_for_log["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    
    if "primary_concept_t_intensity_raw" not in event_data_for_log: # Ensure it's logged
        event_data_for_log["primary_concept_t_intensity_raw"] = raw_t_intensity_for_trend 

    # --- Logging & Persistence ---
    try:
        if not isinstance(_ethics_db.get("ethical_scores"), list): 
            _ethics_db["ethical_scores"] = []
            _log_ethics_event("score_ethics_db_reinit_scores_list", {}, level="warning")

        _ethics_db["ethical_scores"].append(event_data_for_log)
        
        max_entries = int(getattr(config, 'ETHICS_LOG_MAX_ENTRIES', 1000))
        if len(_ethics_db["ethical_scores"]) > max_entries:
            num_to_remove = len(_ethics_db["ethical_scores"]) - max_entries
            _ethics_db["ethical_scores"] = _ethics_db["ethical_scores"][num_to_remove:]
            _log_ethics_event("score_ethics_log_pruned", {"removed_count": num_to_remove, "new_count": len(_ethics_db["ethical_scores"])}, level="debug")

        _ethics_db_dirty_flag = True
        _save_ethics_db() # Attempt to save immediately
    except Exception as e_db:
        _log_ethics_event("score_ethics_db_error", {"error": str(e_db), "trace": traceback.format_exc()}, level="critical")

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
    global _ethics_db_dirty_flag
    # Ensure traceback is imported for full error details, if not already at module level
    import traceback

    _log_ethics_event("track_trends_start", {}, level="debug")

    if not config:
        _log_ethics_event("track_trends_failure", {"error": "Config not loaded"}, level="critical")
        return {"trend_direction": "error_config_missing", "last_updated": datetime.datetime.utcnow().isoformat() + "Z"}

    min_data_points = int(getattr(config, 'ETHICS_TREND_MIN_DATAPOINTS', 10)) # Default to 10
    
    # Ensure "ethical_scores" exists and is a list
    if not isinstance(_ethics_db.get("ethical_scores"), list) or len(_ethics_db.get("ethical_scores", [])) < 1: # Check actual list
        _log_ethics_event("track_trends_insufficient_data", {"count": len(_ethics_db.get("ethical_scores", [])), "min_required": min_data_points}, level="info")
        current_trends = {
            "data_points_used": len(_ethics_db.get("ethical_scores", [])),
            "trend_direction": "insufficient_data",
            "last_updated": datetime.datetime.utcnow().isoformat() + "Z"
        }
        _ethics_db["trend_analysis"] = current_trends 
        _ethics_db_dirty_flag = True 
        _save_ethics_db()
        return current_trends

    scores_with_intensity = []
    for score_event in _ethics_db["ethical_scores"]:
        if isinstance(score_event, dict) and            "final_score" in score_event and            "primary_concept_t_intensity_raw" in score_event:
            try:
                final_score = float(score_event["final_score"])
                t_intensity = float(score_event["primary_concept_t_intensity_raw"]) # This is the raw 0-1 intensity
                scores_with_intensity.append({"score": final_score, "t_intensity": t_intensity})
            except (ValueError, TypeError):
                _log_ethics_event("track_trends_data_format_error", {"event_snippet": str(score_event)[:100]}, level="warning")
                continue

    if len(scores_with_intensity) < min_data_points:
        _log_ethics_event("track_trends_insufficient_valid_data", {"count": len(scores_with_intensity), "min_required": min_data_points}, level="info")
        current_trends = {
            "data_points_used": len(scores_with_intensity),
            "trend_direction": "insufficient_data",
            "last_updated": datetime.datetime.utcnow().isoformat() + "Z"
        }
        _ethics_db["trend_analysis"] = current_trends
        _ethics_db_dirty_flag = True
        _save_ethics_db()
        return current_trends

    # T-weighting: weight = (t_intensity * factor) + base_weight
    t_intensity_factor = 0.9 
    base_weight = 0.1        

    # Extract original scores and weights separately for clarity in weighted average calculation
    original_scores = np.array([s_i["score"] for s_i in scores_with_intensity])
    weights = np.array([(np.clip(s_i["t_intensity"],0,1) * t_intensity_factor + base_weight) 
                        for s_i in scores_with_intensity])
    weights[weights == 0] = 1e-6 

    short_term_window = max(3, int(min_data_points * 0.2))
    long_term_window = max(5, int(min_data_points * 0.5))

    short_term_window = min(short_term_window, len(original_scores))
    long_term_window = min(long_term_window, len(original_scores))

    t_weighted_short_term_avg = None
    if short_term_window > 0 and len(original_scores) >= short_term_window :
        short_scores_slice = original_scores[-short_term_window:]
        short_weights_slice = weights[-short_term_window:]
        t_weighted_short_term_avg = np.sum(short_scores_slice * short_weights_slice) / np.sum(short_weights_slice)

    t_weighted_long_term_avg = None
    if long_term_window > 0 and len(original_scores) >= long_term_window:
        long_scores_slice = original_scores[-long_term_window:]
        long_weights_slice = weights[-long_term_window:]
        t_weighted_long_term_avg = np.sum(long_scores_slice * long_weights_slice) / np.sum(long_weights_slice)
    
    trend_direction = "stable"
    significance_threshold = float(getattr(config, 'ETHICS_TREND_SIGNIFICANCE_THRESHOLD', 0.05)) 

    if t_weighted_short_term_avg is not None and t_weighted_long_term_avg is not None:
        # Ensure we have enough separation between short and long term windows for meaningful comparison
        # e.g. if short_term_window is almost same as long_term_window, diff might not be useful.
        # This logic assumes short_term_window < long_term_window for trend.
        # If not, the "trend" is just the short_term_avg vs itself, which is stable.
        if long_term_window > short_term_window: # Only compare if windows are distinct enough
            diff = t_weighted_short_term_avg - t_weighted_long_term_avg
            # Use a small epsilon for stability if long_term_avg is near zero
            relative_diff = diff / (abs(t_weighted_long_term_avg) + 1e-9) 
            
            if relative_diff > significance_threshold:
                trend_direction = "improving"
            elif relative_diff < -significance_threshold:
                trend_direction = "declining"
        else: # Not enough distinct data for long/short comparison, or windows are same.
            trend_direction = "stable" # Or "insufficient_trend_data"
            
    elif t_weighted_short_term_avg is not None:
         # Only short term data, cannot determine trend against long term.
         trend_direction = "insufficient_data_for_trend_comparison"
    else: # No averages could be calculated
        trend_direction = "insufficient_data"


    current_trends = {
        "data_points_used": len(scores_with_intensity),
        "short_term_window": short_term_window if t_weighted_short_term_avg is not None else 0,
        "long_term_window": long_term_window if t_weighted_long_term_avg is not None else 0,
        "t_weighted_short_term_avg": t_weighted_short_term_avg,
        "t_weighted_long_term_avg": t_weighted_long_term_avg,
        "trend_direction": trend_direction,
        "significance_threshold": significance_threshold,
        "last_updated": datetime.datetime.utcnow().isoformat() + "Z"
    }

    _ethics_db["trend_analysis"] = current_trends
    _ethics_db_dirty_flag = True
    _save_ethics_db()

    _log_ethics_event("track_trends_complete", current_trends, level="info")
    return current_trends

# --- Load ethics database at module import ---
_load_ethics_db()

if __name__ == '__main__':
    # --- Test Utilities ---
    class TempConfigOverride:
        # ... (Assume TempConfigOverride class is defined here as per previous step) ...
        def __init__(self, temp_configs_dict):
            self.temp_configs = temp_configs_dict
            self.original_values = {}
        def __enter__(self):
            if not config: raise ImportError("Config module not loaded")
            for key, value in self.temp_configs.items():
                self.original_values[key] = getattr(config, key, None)
                setattr(config, key, value)
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            if not config: return
            for key, original_value in self.original_values.items():
                if hasattr(config,key) and original_value is None and key not in self.temp_configs: pass
                elif original_value is not None: setattr(config, key, original_value)
                elif hasattr(config,key): delattr(config,key)

    # --- Test Setup & Helper ---
    TEST_ETHICS_DB_FILENAME = "test_ethics_db.json"
    module_dir = os.path.dirname(os.path.abspath(__file__))
    TEST_ETHICS_DB_PATH = os.path.join(module_dir, TEST_ETHICS_DB_FILENAME)
    TEST_SYSTEM_LOG_PATH = os.path.join(module_dir, "test_ethics_system_log.json") # For logging during tests

    original_get_shared_manifold = None # To store the real get_shared_manifold

    class MockConceptualNeighborhoodManifold:
        """Mock for SpacetimeManifold to control get_conceptual_neighborhood output."""
        def __init__(self, neighborhood_data=None):
            self.neighborhood_data = neighborhood_data if neighborhood_data is not None else []
            self.device = "cpu" # Mock device attribute

        def get_conceptual_neighborhood(self, concept_coord, radius):
            _log_ethics_event("mock_get_conceptual_neighborhood_called", {"coord": concept_coord, "radius": radius, "returning_count": len(self.neighborhood_data)}, level="debug")
            return self.neighborhood_data # Return predefined data

    def mock_get_shared_manifold_for_ethics_test(neighborhood_data=None, force_recreate=False):
        """Returns an instance of the MockConceptualNeighborhoodManifold."""
        return MockConceptualNeighborhoodManifold(neighborhood_data=neighborhood_data)

    def setup_test_environment(test_specific_configs=None, mock_neighborhood=None):
        global _ethics_db, _ethics_db_dirty_flag, original_get_shared_manifold, get_shared_manifold
        
        if os.path.exists(TEST_ETHICS_DB_PATH):
            os.remove(TEST_ETHICS_DB_PATH)
        if os.path.exists(TEST_SYSTEM_LOG_PATH):
            os.remove(TEST_SYSTEM_LOG_PATH)
            
        _ethics_db = {"ethical_scores": [], "trend_analysis": {}}
        _ethics_db_dirty_flag = False
        
        # Store original and monkeypatch get_shared_manifold for this test run
        if original_get_shared_manifold is None: # Store only once
            original_get_shared_manifold = sys.modules[__name__].get_shared_manifold
        
        # Apply mock for get_shared_manifold
        # This lambda captures the current mock_neighborhood value for this specific test setup.
        sys.modules[__name__].get_shared_manifold = lambda force_recreate=False, current_mock_neighborhood=mock_neighborhood: mock_get_shared_manifold_for_ethics_test(neighborhood_data=current_mock_neighborhood, force_recreate=force_recreate)


        # Default test configs + any test-specific overrides
        final_test_configs = {
            "ETHICS_DB_PATH": TEST_ETHICS_DB_PATH,
            "SYSTEM_LOG_PATH": TEST_SYSTEM_LOG_PATH,
            "VERBOSE_OUTPUT": False, # Usually False for tests unless debugging a specific one
            "ETHICS_LOG_MAX_ENTRIES": 5, # Small for testing pruning
            "ETHICS_TREND_MIN_DATAPOINTS": 3,
            "ETHICS_TREND_SIGNIFICANCE_THRESHOLD": 0.05,
            # Default weights for testing, can be overridden by test_specific_configs
            "ETHICS_COHERENCE_WEIGHT": 0.2, "ETHICS_VALENCE_WEIGHT": 0.2,
            "ETHICS_INTENSITY_WEIGHT": 0.1, "ETHICS_FRAMEWORK_WEIGHT": 0.3,
            "ETHICS_CLUSTER_CONTEXT_WEIGHT": 0.2,
            "MANIFOLD_RANGE": 10.0 # Default test range for normalization in score_ethics
        }
        if test_specific_configs:
            final_test_configs.update(test_specific_configs)
        
        return TempConfigOverride(final_test_configs)

    def cleanup_test_environment():
        global original_get_shared_manifold, get_shared_manifold
        if os.path.exists(TEST_ETHICS_DB_PATH):
            os.remove(TEST_ETHICS_DB_PATH)
        if os.path.exists(TEST_SYSTEM_LOG_PATH):
            os.remove(TEST_SYSTEM_LOG_PATH)
        
        # Restore original get_shared_manifold
        if original_get_shared_manifold is not None:
            sys.modules[__name__].get_shared_manifold = original_get_shared_manifold


    def run_test(test_func, *args, **kwargs): # Allow kwargs for test-specific configs/mocks
        test_name = test_func.__name__
        print(f"--- Running Test: {test_name} ---")
        test_configs = kwargs.pop("test_configs", {})
        mock_neighborhood_data = kwargs.pop("mock_neighborhood_data", [])
        
        with setup_test_environment(test_specific_configs=test_configs, mock_neighborhood=mock_neighborhood_data):
            # _load_ethics_db() is called by setup_test_environment essentially by resetting _ethics_db
            # and then subsequent functions like score_ethics will interact with this in-memory _ethics_db.
            # If a test needs to load from a *pre-populated* test file, that needs special handling.
            # For now, tests assume starting with an empty DB state in memory.
            _load_ethics_db() # Ensure it loads from the overridden TEST_ETHICS_DB_PATH (likely non-existent initially)
            result = False # Default to False
            try:
                # Ensure traceback is imported for full error details if not already
                if 'traceback' not in sys.modules: import traceback
                result = test_func(*args, **kwargs)
                if result:
                    print(f"PASS: {test_name}")
                else:
                    print(f"FAIL: {test_name}")
            except Exception as e:
                print(f"ERROR in {test_name}: {e}")
                traceback.print_exc()
                result = False # Explicitly mark as fail on exception
            finally:
                cleanup_test_environment() # Clean up files and restore mocks
        return result

    # --- Test Function Definitions ---

    def test_db_load_save_and_dirty_flag():
        print("Testing DB load/save and dirty flag logic...")
        global _ethics_db, _ethics_db_dirty_flag

        # 1. Initial load (file non-existent)
        _load_ethics_db() 
        if _ethics_db != {"ethical_scores": [], "trend_analysis": {}} or _ethics_db_dirty_flag:
            print("FAIL: Initial load did not result in clean empty DB state.")
            return False
        print("  PASS: Initial load non-existent file.")

        # 2. Save when not dirty (should skip)
        _save_ethics_db()
        if os.path.exists(TEST_ETHICS_DB_PATH):
            print("FAIL: DB file created on save when not dirty.")
            return False
        print("  PASS: Save skipped when not dirty.")

        # 3. Modify DB and save
        _ethics_db["ethical_scores"].append({"test_score": 1, "primary_concept_t_intensity_raw": 0.5, "timestamp": "2023-01-01T00:00:00Z"})
        _ethics_db_dirty_flag = True
        _save_ethics_db()
        if not os.path.exists(TEST_ETHICS_DB_PATH) or _ethics_db_dirty_flag:
            print("FAIL: DB not saved when dirty, or dirty flag not reset.")
            return False
        print("  PASS: DB saved when dirty, flag reset.")
        
        # 4. Load saved DB
        _ethics_db = {} 
        _load_ethics_db()
        if not _ethics_db.get("ethical_scores") or _ethics_db["ethical_scores"][0].get("test_score") != 1:
            print(f"FAIL: Did not load saved DB correctly. Got: {_ethics_db}")
            return False
        print("  PASS: Reloaded saved DB correctly.")
        return True

    def test_score_ethics_basic_and_components(**kwargs): 
        print("Testing score_ethics (basic components)...")
        awareness_metrics_neutral = {"coherence": 0.0, "primary_concept_coord": (0,0,0,0.5 * config.MANIFOLD_RANGE / 2.0), "raw_t_intensity": 0.5}
        score = score_ethics(awareness_metrics_neutral, "neutral summary", "neutral action")
        
        if not (0.0 <= score <= 1.0):
            print(f"FAIL: Basic score out of range [0,1]. Score: {score}")
            return False
        if len(_ethics_db["ethical_scores"]) != 1:
            print(f"FAIL: Score event not logged. Count: {len(_ethics_db['ethical_scores'])}")
            return False
        print(f"  PASS: Basic score calculated: {score:.2f}, event logged.")
        
        max_entries = config.ETHICS_LOG_MAX_ENTRIES 
        for i in range(max_entries + 2): 
            score_ethics(awareness_metrics_neutral, f"summary {i}", f"action {i}")
        if len(_ethics_db["ethical_scores"]) != max_entries:
            print(f"FAIL: Log pruning failed. Count: {len(_ethics_db['ethical_scores'])}, Expected: {max_entries}")
            return False
        print(f"  PASS: Log pruning to {max_entries} entries.")
        return True

    def test_score_ethics_cluster_context(mock_neighborhood_data_for_test, **kwargs): # Renamed arg
        print("Testing score_ethics with manifold cluster context...")
        awareness_metrics = {"coherence": 0.8, "primary_concept_coord": (config.MANIFOLD_RANGE/4, 0,0,0.7), "raw_t_intensity": 0.7}
        score = score_ethics(awareness_metrics, "concept affecting cluster", "action related to cluster")
        
        last_score_event = _ethics_db["ethical_scores"][-1]
        cluster_comp_score = last_score_event["component_scores"]["manifold_cluster_context"]
        
        expected_mock_avg_valence = 0.5 # Default if no data or error
        if mock_neighborhood_data_for_test: 
            norm_valences = []
            for n_data in mock_neighborhood_data_for_test:
                node_coords = n_data.get("coordinates")
                if isinstance(node_coords, (list,tuple)) and len(node_coords)==4:
                     node_x_val = float(node_coords[0])
                     # Ensure MANIFOLD_RANGE is not zero for normalization
                     current_manifold_range = float(getattr(config, 'MANIFOLD_RANGE', 1.0))
                     if current_manifold_range == 0: current_manifold_range = 1.0 # Avoid div by zero
                     norm_val = (node_x_val + (current_manifold_range / 2.0)) / current_manifold_range
                     norm_valences.append(np.clip(norm_val, 0.0, 1.0))
            if norm_valences: expected_mock_avg_valence = np.mean(norm_valences)
        
        if abs(cluster_comp_score - expected_mock_avg_valence) > 0.01 : 
            print(f"FAIL: Cluster context score {cluster_comp_score:.2f} ({last_score_event['cluster_avg_valence']}) doesn't match expected from mock {expected_mock_avg_valence:.2f}")
            return False
        print(f"  PASS: Cluster context score ({cluster_comp_score:.2f}) matches mock expectation.")
        return True


    def test_track_trends_scenarios(**kwargs):
        print("Testing track_trends...")
        global _ethics_db
        
        min_points = config.ETHICS_TREND_MIN_DATAPOINTS 
        _ethics_db["ethical_scores"] = [] 
        trends = track_trends()
        if trends["trend_direction"] != "insufficient_data": return False
        print("  PASS: Insufficient data (0 scores).")

        for i in range(min_points - 1):
             _ethics_db["ethical_scores"].append({"final_score": 0.5, "primary_concept_t_intensity_raw": 0.5, "timestamp": f"2023-01-01T00:0{i}:00Z"})
        trends = track_trends()
        if trends["trend_direction"] != "insufficient_data": return False
        print(f"  PASS: Insufficient data ({min_points-1} scores).")

        _ethics_db["ethical_scores"] = [] 
        for i in range(min_points + 2): 
            _ethics_db["ethical_scores"].append({"final_score": 0.5, "primary_concept_t_intensity_raw": 0.5, "timestamp": f"2023-01-01T00:0{i}:00Z"})
        trends_stable = track_trends()
        if trends_stable["trend_direction"] != "stable": return False
        print(f"  PASS: Stable trend detected (avg: {trends_stable.get('t_weighted_short_term_avg'):.2f}).")

        _ethics_db["ethical_scores"] = []
        base_scores_improve = [0.2, 0.3, 0.4, 0.7, 0.8]; intensities_improve = [0.2, 0.3, 0.4, 0.9, 1.0]
        for i, score_val in enumerate(base_scores_improve):
            _ethics_db["ethical_scores"].append({"final_score": score_val, "primary_concept_t_intensity_raw": intensities_improve[i], "timestamp": f"2023-01-01T00:0{i}:00Z"})
        trends_improve = track_trends()
        if trends_improve["trend_direction"] != "improving": return False
        print(f"  PASS: Improving trend (Short: {trends_improve.get('t_weighted_short_term_avg'):.2f}, Long: {trends_improve.get('t_weighted_long_term_avg'):.2f}).")
        
        _ethics_db["ethical_scores"] = []
        base_scores_decline = [0.8, 0.7, 0.6, 0.3, 0.2]; intensities_decline = [1.0, 0.9, 0.4, 0.3, 0.2] 
        for i, score_val in enumerate(base_scores_decline):
            _ethics_db["ethical_scores"].append({"final_score": score_val, "primary_concept_t_intensity_raw": intensities_decline[i], "timestamp": f"2023-01-01T00:0{i}:00Z"})
        trends_decline = track_trends()
        if trends_decline["trend_direction"] != "declining": return False
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
    # Ensure traceback is available for run_test if not already imported at module level for some reason
    if 'traceback' not in sys.modules: import traceback
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
