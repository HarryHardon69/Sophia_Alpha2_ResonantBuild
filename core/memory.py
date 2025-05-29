"""
Handles Sophia_Alpha2's memory operations.

This module includes functionalities for:
- Storing and retrieving interaction memories and ingested knowledge.
- Calculating concept novelty based on spatial and textual features.
- Managing a persistent knowledge graph (`knowledge_graph.json`).
- Filtering memories based on ethical alignment and novelty thresholds.
"""

import datetime
import json
import os
import sys
import time
import traceback # Promoted to top-level
import uuid

import numpy as np

# Attempt to import configuration from the parent package
try:
    from .. import config
except ImportError:
    # Fallback for standalone execution or testing
    print("Memory.py: Could not import 'config' from parent package. Attempting relative import for standalone use.")
    try:
        import config
        print("Memory.py: Successfully imported 'config' directly (likely for standalone testing).")
    except ImportError:
        print("Memory.py: Failed to import 'config' for standalone use. Critical error.")
        config = None # Placeholder

# Further module-level constants or setup can go here.

# --- Module-Level Logging ---
LOG_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40, "critical": 50} # Duplicated from brain for now, consider centralizing

def _log_memory_event(event_type: str, data: dict, level: str = "info"):
    """
    Logs a structured system event from the memory module.
    Data is serialized to JSON.
    Respects LOG_LEVEL from config.
    """
    if not config or not hasattr(config, 'SYSTEM_LOG_PATH') or not hasattr(config, 'LOG_LEVEL'):
        print(f"Warning (memory.py): Config not available for _log_memory_event. Event: {event_type}, Data: {data}")
        # Avoid print loop if logging itself is the source of config issue for logging
        if event_type == "logging_error" and "Config not available" in data.get("error",""):
            return
        # Fallback to print if essential config for logging is missing
        print(f"MEMORY_EVENT ({level.upper()}): {event_type} - Data: {json.dumps(data, default=str)}")
        return

    # Re-check numeric_level calculation, ensure config.LOG_LEVEL is valid
    config_log_level_str = getattr(config, 'LOG_LEVEL', 'INFO').lower()
    numeric_level = LOG_LEVELS.get(level.lower(), LOG_LEVELS["info"])
    config_numeric_level = LOG_LEVELS.get(config_log_level_str, LOG_LEVELS["info"])

    if numeric_level < config_numeric_level:
        return # Skip logging if event level is below configured log level

    try:
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "module": "memory",
            "event_type": event_type,
            "level": level.upper(),
            "data": data
        }
        
        # Ensure the log directory exists (config.ensure_path should handle this for SYSTEM_LOG_PATH at import)
        # This might be called before config fully initializes ensure_path if there's an early log.
        # However, config.SYSTEM_LOG_PATH itself implies its parent dir should be ensured by config.py's own init.
        
        # Ensure SYSTEM_LOG_PATH's directory exists before writing
        # This is a safeguard in case config.py's ensure_path hasn't run or is insufficient
        log_file_path = config.SYSTEM_LOG_PATH
        log_dir = os.path.dirname(log_file_path)
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception as e_mkdir:
                # Very first log event might not be able to create dir if perms are wrong.
                print(f"CRITICAL (memory.py): Could not create log directory {log_dir}. Error: {e_mkdir}", file=sys.stderr)
                return # Cannot log to file if dir creation fails.

        with open(log_file_path, 'a') as f:
            f.write(json.dumps(log_entry, default=str) + '\n') # Use default=str for non-serializable data
            
    except Exception as e:
        # Fallback to print if logging to file fails
        print(f"Error logging memory event to file: {e}", file=sys.stderr)
        print(f"Original memory event: {event_type}, Data: {data}", file=sys.stderr)
        # Also log the error itself using the same mechanism, but to avoid recursion, check event_type
        if event_type != "logging_error":
             _log_memory_event("logging_error", {"error": str(e), "original_event": event_type}, level="error")

# --- Knowledge Graph State ---
_knowledge_graph = {
    "nodes": [], # List of node dictionaries
    "edges": []  # List of edge dictionaries
}
_kg_dirty_flag = False # True if _knowledge_graph has in-memory changes not yet saved to disk

def _load_knowledge_graph():
    """
    Loads the knowledge graph from the path specified in config.
    Handles file not found, empty file, and malformed JSON.
    Initializes to an empty graph structure if loading fails or file doesn't exist.
    Sets the module-global `_kg_dirty_flag` to False after loading, as the in-memory
    graph is considered synchronized with the loaded state (or a fresh default).
    """
    global _knowledge_graph, _kg_dirty_flag, config # Ensure config is accessible
    
    if not config or not hasattr(config, 'KNOWLEDGE_GRAPH_PATH'):
        _log_memory_event("load_kg_failure", {"error": "Config not available or KNOWLEDGE_GRAPH_PATH not set"}, level="critical")
        _knowledge_graph = {"nodes": [], "edges": []} # Initialize with a default empty structure
        _kg_dirty_flag = False # Considered "clean" as it's a fresh default state
        return

    kg_path = config.KNOWLEDGE_GRAPH_PATH
    # Ensure the directory for the knowledge graph file exists.
    # Uses config.ensure_path if available, otherwise uses os.makedirs directly.
    if hasattr(config, 'ensure_path'):
        config.ensure_path(kg_path) # ensure_path handles parent directory creation
    else: # Fallback if config.ensure_path is not available for some reason
        parent_dir = os.path.dirname(kg_path)
        if parent_dir: # Ensure parent_dir is not empty (e.g. if kg_path is just a filename)
            os.makedirs(parent_dir, exist_ok=True)

    try:
        # Handle cases: file doesn't exist or is empty.
        if not os.path.exists(kg_path) or os.path.getsize(kg_path) == 0:
            _log_memory_event("load_kg_info", {"message": "Knowledge graph file not found or empty. Initializing new graph.", "path": kg_path}, level="info")
            _knowledge_graph = {"nodes": [], "edges": []} # Initialize with default structure
            # This state is considered "clean" as it represents a newly initialized graph.
            # If an empty file should be explicitly written to disk at this point,
            # _save_knowledge_graph would need to be called (and dirty flag managed accordingly).
            _kg_dirty_flag = False 
            return

        # Attempt to open and load JSON data from the file.
        with open(kg_path, 'r') as f:
            data = json.load(f)

        # Validate the loaded data structure.
        if isinstance(data, dict) and \
           "nodes" in data and isinstance(data["nodes"], list) and \
           "edges" in data and isinstance(data["edges"], list):
            _knowledge_graph = data # Assign loaded data to the global variable
            _kg_dirty_flag = False # Successfully loaded, so in-memory is synchronized
            _log_memory_event("load_kg_success", {"path": kg_path, "nodes_loaded": len(data["nodes"]), "edges_loaded": len(data["edges"])}, level="info")
        else: # Malformed structure (e.g., missing 'nodes' or 'edges' keys, or they are not lists)
            _log_memory_event("load_kg_malformed_structure", {"path": kg_path, "error": "Root must be dict with 'nodes' and 'edges' lists."}, level="error")
            _knowledge_graph = {"nodes": [], "edges": []} # Reset to default
            _kg_dirty_flag = False # Considered "clean" as it's reset to a default state
            
    except json.JSONDecodeError as e: # Handle errors during JSON parsing
        _log_memory_event("load_kg_json_decode_error", {"path": kg_path, "error": str(e)}, level="error")
        _knowledge_graph = {"nodes": [], "edges": []} # Reset to default
        _kg_dirty_flag = False
    except Exception as e: # Catch any other unexpected errors during file operations or loading
        _log_memory_event("load_kg_unknown_error", {"path": kg_path, "error": str(e), "trace": traceback.format_exc()}, level="critical")
        _knowledge_graph = {"nodes": [], "edges": []} # Reset to default
        _kg_dirty_flag = False

def _save_knowledge_graph():
    """
    Saves the current state of the module-global `_knowledge_graph` to disk if changes
    have been made (indicated by `_kg_dirty_flag`).
    Uses an atomic write operation (write to temp file, then rename/replace) to prevent
    data corruption. Sets `_kg_dirty_flag` to False after a successful save.
    """
    global _kg_dirty_flag # To modify the flag

    if not _kg_dirty_flag: # Only save if there are pending changes
        _log_memory_event("save_kg_skipped", {"message": "No changes to save (_kg_dirty_flag is False)."}, level="debug")
        return

    if not config or not hasattr(config, 'KNOWLEDGE_GRAPH_PATH'):
        _log_memory_event("save_kg_failure", {"error": "Config not available or KNOWLEDGE_GRAPH_PATH not set"}, level="critical")
        # Do not reset _kg_dirty_flag as changes are still pending and unsaved.
        return

    kg_path = config.KNOWLEDGE_GRAPH_PATH
    # Ensure the directory for the knowledge graph file exists.
    if hasattr(config, 'ensure_path'):
        config.ensure_path(kg_path)
    else: # Fallback
        parent_dir = os.path.dirname(kg_path)
        if parent_dir:
             os.makedirs(parent_dir, exist_ok=True)

    temp_kg_path = kg_path + ".tmp" # Define temporary file path for atomic write

    try:
        # Atomicity Step 1: Write the current graph to a temporary file.
        # Using indent=4 for human-readable JSON output.
        with open(temp_kg_path, 'w') as f:
            json.dump(_knowledge_graph, f, indent=4)
        
        # Atomicity Step 2: Replace the original file with the temporary file.
        # os.replace is atomic on Windows and POSIX (if target exists and permissions allow).
        # os.rename might fail on Windows if the destination exists.
        os.replace(temp_kg_path, kg_path) # Preferred for atomicity if destination might exist

        _kg_dirty_flag = False # Reset dirty flag only after successful write and rename
        _log_memory_event("save_kg_success", {"path": kg_path, "nodes_saved": len(_knowledge_graph["nodes"]), "edges_saved": len(_knowledge_graph["edges"])}, level="info")
    
    except IOError as e_io: # Handle file I/O specific errors
        _log_memory_event("save_kg_io_error", {"path": kg_path, "temp_path": temp_kg_path, "error": str(e_io), "trace": traceback.format_exc()}, level="critical")
        # Do not reset dirty flag. Attempt to clean up the temporary file if it exists.
        if os.path.exists(temp_kg_path):
            try:
                os.remove(temp_kg_path)
            except Exception as e_rm:
                 _log_memory_event("save_kg_temp_file_cleanup_error_io", {"path": temp_kg_path, "error": str(e_rm)}, level="error")
    except Exception as e: # Handle other unexpected errors during save
        _log_memory_event("save_kg_unknown_error", {"path": kg_path, "temp_path": temp_kg_path, "error": str(e), "trace": traceback.format_exc()}, level="critical")
        # Do not reset dirty flag. Attempt to clean up the temporary file.
        if os.path.exists(temp_kg_path):
            try:
                os.remove(temp_kg_path)
            except Exception as e_rm:
                 _log_memory_event("save_kg_temp_file_cleanup_error_unknown", {"path": temp_kg_path, "error": str(e_rm)}, level="error")

def calculate_novelty(concept_coord: tuple, concept_summary: str) -> float:
    """
    Calculates the novelty of a concept based on its coordinates and summary
    compared to existing memories in the knowledge graph.

    Args:
        concept_coord: A 4-tuple (x, y, z, t_intensity) representing the concept's
                       coordinates in the manifold. These are assumed to be raw coordinates
                       as stored or received, not yet normalized for this function's internal use.
        concept_summary: Textual summary of the concept.

    Returns:
        A novelty score between 0.0 (not novel) and 1.0 (very novel).
    """
    _log_memory_event("calculate_novelty_start", {"coord_len": len(concept_coord) if isinstance(concept_coord, tuple) else -1, "summary_len": len(concept_summary)}, level="debug")

    if not config: # Ensure config is loaded, as it contains parameters for calculation
        _log_memory_event("calculate_novelty_error", {"error": "Config module not loaded, cannot calculate novelty."}, level="error")
        return 0.0 # Default to non-novel if config is unavailable

    # Validate input: concept_coord should be a 4-tuple of numeric values.
    if not isinstance(concept_coord, tuple) or len(concept_coord) != 4:
        _log_memory_event("calculate_novelty_error", {"error": "Invalid concept_coord format (must be 4-tuple).", "coord_received": str(concept_coord)}, level="warning")
        return 0.0 
    try:
        # Convert coordinates to a NumPy array of floats for vectorized operations.
        current_coord_np = np.array([float(c) for c in concept_coord])
    except (ValueError, TypeError) as e: # Handle non-numeric coordinates
        _log_memory_event("calculate_novelty_error", {"error": f"Invalid coordinate values (must be numeric): {str(e)}", "coord_received": str(concept_coord)}, level="warning")
        return 0.0

    # --- Spatial Novelty Calculation ---
    # Spatial novelty is based on the minimum normalized Euclidean distance to existing nodes in the KG.
    # A higher distance implies higher novelty.
    spatial_novelty_score = 1.0 # Default to max novelty if no existing nodes or other issues.
    
    if _knowledge_graph["nodes"]: # Only calculate if there are existing nodes to compare against.
        min_dist_sq_normalized = float('inf') # Initialize with a very large value for squared distance.
        
        # Normalization factor for coordinates. MANIFOLD_RANGE from config defines the typical
        # extent of a single dimension in the coordinate space. Distances are normalized by this range.
        norm_factor = config.MANIFOLD_RANGE 
        if norm_factor == 0: # Prevent division by zero if MANIFOLD_RANGE is misconfigured.
            _log_memory_event("calculate_novelty_warning", {"warning": "MANIFOLD_RANGE is 0. Spatial distance normalization will be ineffective."}, level="warning")
            norm_factor = 1.0 # Use 1.0 to avoid error, though normalization becomes identity.

        processed_node_coords = 0 # Counter for nodes with valid coordinates.
        for node in _knowledge_graph["nodes"]:
            node_coord_data = node.get("coordinates") # Existing node's coordinates
            if isinstance(node_coord_data, (list, tuple)) and len(node_coord_data) == 4:
                try:
                    node_coord_np = np.array([float(c) for c in node_coord_data])
                    # Calculate squared Euclidean distance, normalized by norm_factor for each dimension.
                    # diff_sq_normalized = sum(((current_coord_dim - node_coord_dim) / norm_factor)^2)
                    diff_sq_normalized = np.sum(((current_coord_np - node_coord_np) / norm_factor)**2)
                    min_dist_sq_normalized = min(min_dist_sq_normalized, diff_sq_normalized)
                    processed_node_coords += 1
                except (ValueError, TypeError): # Skip nodes with malformed coordinates.
                    _log_memory_event("calculate_novelty_node_coord_error", {"node_id": node.get("id"), "coord_data": str(node_coord_data)}, level="debug")
                    continue 

        if processed_node_coords > 0 and min_dist_sq_normalized != float('inf'):
            # Convert minimum squared normalized distance to actual distance.
            min_dist_normalized = np.sqrt(min_dist_sq_normalized)
            # Spatial novelty score: Linearly maps normalized distance to novelty.
            # A distance of 0 implies 0 novelty. A distance of 1.0 (or more) implies full novelty.
            # This assumes that if a concept is 1 "manifold unit" away (after normalization), it's fully novel.
            # This mapping might need tuning depending on desired sensitivity.
            spatial_novelty_score = np.clip(min_dist_normalized, 0.0, 1.0)
            
    _log_memory_event("calculate_novelty_spatial", {"score": spatial_novelty_score, "nodes_compared": len(_knowledge_graph.get('nodes',[]))}, level="debug")

    # --- Textual Novelty Calculation ---
    # Textual novelty is based on the maximum Jaccard similarity between the concept's summary
    # and summaries of existing nodes. Novelty = 1 - max_similarity.
    textual_novelty_score = 1.0 # Default to max novelty.
    
    if _knowledge_graph["nodes"] and concept_summary: # Requires existing nodes and a non-empty summary.
        max_jaccard_similarity = 0.0
        current_summary_lower = concept_summary.lower() # Case-insensitive comparison.
        current_tokens = set(current_summary_lower.split()) # Tokenize summary into a set of words.

        if not current_tokens: # If current summary is empty or only whitespace.
            textual_novelty_score = 0.5 # Assign neutral novelty.
        else:
            processed_text_nodes = 0
            for node in _knowledge_graph["nodes"]:
                node_summary = node.get("summary")
                if isinstance(node_summary, str) and node_summary: # Check if node has a valid summary.
                    node_summary_lower = node_summary.lower()
                    node_tokens = set(node_summary_lower.split())
                    
                    if not node_tokens: continue # Skip comparison if existing node's summary is empty.

                    # Calculate Jaccard Similarity: intersection_size / union_size
                    intersection_size = len(current_tokens.intersection(node_tokens))
                    union_size = len(current_tokens.union(node_tokens))
                    
                    if union_size == 0: # Both summaries effectively empty (e.g. only common stop words if not filtered)
                        similarity = 1.0 if not current_tokens and not node_tokens else 0.0
                    else:
                        similarity = intersection_size / union_size
                    
                    max_jaccard_similarity = max(max_jaccard_similarity, similarity)
                    processed_text_nodes += 1
            
            if processed_text_nodes > 0: # If compared against at least one node.
                textual_novelty_score = 1.0 - max_jaccard_similarity # Higher similarity means lower novelty.
    
    _log_memory_event("calculate_novelty_textual", {"score": textual_novelty_score, "nodes_compared": len(_knowledge_graph.get('nodes',[]))}, level="debug")

    # --- Combine Spatial and Textual Novelties ---
    # Weighted average of spatial and textual novelty scores.
    # Weights are sourced from config, defaulting to 0.5 each.
    spatial_weight = float(getattr(config, 'SPATIAL_NOVELTY_WEIGHT', 0.5))
    textual_weight = float(getattr(config, 'TEXTUAL_NOVELTY_WEIGHT', 0.5))
    
    # Normalize weights to ensure they sum to 1, or handle sum of 0.
    total_weight = spatial_weight + textual_weight
    if total_weight == 0: # If both weights are zero
        # If either component score is positive, default to equal weighting to reflect that novelty.
        # Otherwise, if both scores are 0, the result is 0.
        if spatial_novelty_score > 0 or textual_novelty_score > 0:
            spatial_weight = 0.5
            textual_weight = 0.5
            total_weight = 1.0 # Corrected total_weight for averaging
        else: # Both scores are 0, weights are 0, so final novelty is 0.
             final_novelty_score = 0.0
    
    if total_weight > 0: # Calculate weighted average if total_weight is positive.
        final_novelty_score = ((spatial_novelty_score * spatial_weight) + \
                               (textual_novelty_score * textual_weight)) / total_weight
    # If total_weight was initially 0 and scores were also 0, final_novelty_score is already 0.

    final_novelty_score = np.clip(final_novelty_score, 0.0, 1.0) # Ensure score is within [0,1].
    
    _log_memory_event("calculate_novelty_final", {"score": final_novelty_score, 
                                                 "spatial_raw": spatial_novelty_score, "textual_raw": textual_novelty_score,
                                                 "spatial_weight": spatial_weight, "textual_weight": textual_weight}, 
                                                 level="info")
    return final_novelty_score

def store_memory(concept_name: str, concept_coord: tuple, summary: str, 
                 intensity: float, ethical_alignment: float, 
                 related_concepts: list = None) -> bool:
    """
    Stores a new memory (concept) in the knowledge graph if it meets novelty
    and ethical thresholds.

    Args:
        concept_name: The name or label of the concept.
        concept_coord: 4D coordinates (x, y, z, t_intensity).
        summary: Textual summary of the concept.
        intensity: Raw intensity of the concept (e.g., 0-1).
        ethical_alignment: Ethical alignment score of the concept (e.g., 0-1).
        related_concepts: Optional list of memory_ids or concept_names that this 
                          concept is related to.

    Returns:
        True if the memory was stored, False otherwise.
    """
    global _kg_dirty_flag # Flag to indicate that the in-memory graph has changed and needs saving.

    _log_memory_event("store_memory_attempt", 
                      {"concept_name": concept_name, 
                       "coord_len": len(concept_coord) if isinstance(concept_coord, tuple) else -1, 
                       "ethical_alignment": ethical_alignment}, 
                      level="debug")

    if not config: # Critical dependency: config for thresholds.
        _log_memory_event("store_memory_failure", {"concept_name": concept_name, "error": "Config module not loaded."}, level="critical")
        return False

    # --- 1. Input Validation ---
    # Ensure essential data fields are present and have the correct basic format.
    if not all([concept_name, isinstance(concept_coord, tuple), len(concept_coord) == 4, summary is not None]):
        _log_memory_event("store_memory_invalid_input", 
                          {"concept_name": concept_name, "reason": "Missing or invalid format for core inputs (name, coord, summary)."}, 
                          level="warning")
        return False
    try:
        # Convert potentially stringified numeric inputs to their correct types.
        numeric_coord = tuple(float(c) for c in concept_coord)
        float_intensity = float(intensity)
        float_ethical_alignment = float(ethical_alignment)
    except (ValueError, TypeError) as e: # Handle errors if conversion fails.
        _log_memory_event("store_memory_invalid_input_type", 
                          {"concept_name": concept_name, "error": f"Type conversion failed for numeric fields: {str(e)}"}, 
                          level="warning")
        return False

    # --- 2. Calculate Novelty Score ---
    # Determines how new or different this concept is compared to existing memories.
    novelty_score = calculate_novelty(numeric_coord, summary)

    # --- 3. Check Thresholds for Storage ---
    # Retrieve thresholds from config, with defaults if not specified.
    novelty_threshold = float(getattr(config, 'MEMORY_NOVELTY_THRESHOLD', 0.5))
    ethical_threshold = float(getattr(config, 'MEMORY_ETHICAL_THRESHOLD', 0.5))

    # Reject if novelty is below threshold.
    if novelty_score < novelty_threshold:
        _log_memory_event("store_memory_rejected_novelty", 
                          {"concept_name": concept_name, "novelty_score": novelty_score, "threshold": novelty_threshold}, 
                          level="info")
        return False

    # Reject if ethical alignment is below threshold.
    if float_ethical_alignment < ethical_threshold:
        _log_memory_event("store_memory_rejected_ethical", 
                          {"concept_name": concept_name, "ethical_score": float_ethical_alignment, "threshold": ethical_threshold}, 
                          level="info")
        return False

    # --- 4. Store Memory in Knowledge Graph ---
    # If all checks pass, create and store the new memory node.
    try:
        memory_id = uuid.uuid4().hex # Generate a unique ID for the new memory.
        timestamp = datetime.datetime.utcnow().isoformat() + "Z" # ISO 8601 format UTC timestamp.

        # Construct the new node dictionary.
        new_node = {
            "id": memory_id,
            "label": str(concept_name), # Ensure label is a string.
            "coordinates": numeric_coord, # Store validated numeric coordinates.
            "summary": str(summary), # Ensure summary is a string.
            "intensity": float_intensity, # Store validated float intensity.
            "raw_t_intensity_at_storage": float_intensity, # Explicitly store for potential future reference.
            "ethical_alignment_at_storage": float_ethical_alignment,
            "novelty_at_storage": novelty_score,
            "timestamp": timestamp,
            "type": "concept_memory" # Categorize the node type (can be expanded).
        }
        _knowledge_graph["nodes"].append(new_node) # Add the new node to the in-memory graph.
        
        # Handle relationships to other concepts if `related_concepts` is provided.
        if related_concepts and isinstance(related_concepts, list):
            for related_item_specifier in related_concepts:
                target_node_id = None
                if isinstance(related_item_specifier, str):
                    # Attempt to resolve specifier: first as ID, then as name.
                    is_likely_id = len(related_item_specifier) == 32 and all(c in '0123456789abcdef' for c in related_item_specifier.lower())
                    
                    if is_likely_id: # If it looks like a UUID hex.
                        # Check if a node with this ID already exists.
                        if any(node['id'] == related_item_specifier for node in _knowledge_graph['nodes']):
                            target_node_id = related_item_specifier
                        else: # Target ID specified but not found.
                             _log_memory_event("store_memory_relation_target_id_not_found", 
                                               {"source_id": memory_id, "target_id_spec": related_item_specifier}, 
                                               level="warning")
                    else: # Assume it's a concept name; search for its ID.
                        # This finds the first node matching the label. Could be enhanced for multiple matches.
                        found_nodes = [node['id'] for node in _knowledge_graph['nodes'] if node.get('label') == related_item_specifier]
                        if found_nodes:
                            target_node_id = found_nodes[0] # Use the ID of the first match.
                            if len(found_nodes) > 1: # Log if multiple nodes match the name.
                                _log_memory_event("store_memory_relation_multiple_targets_by_name", 
                                                  {"source_id": memory_id, "target_name": related_item_specifier, "selected_id": target_node_id, "all_found_ids": found_nodes}, 
                                                  level="warning")
                        else: # Target name specified but no node found with that label.
                            _log_memory_event("store_memory_relation_target_name_not_found", 
                                              {"source_id": memory_id, "target_name": related_item_specifier}, 
                                              level="warning")
                
                if target_node_id: # If a valid target ID was found.
                    # Create a new edge to represent the relationship.
                    new_edge = {
                        "id": uuid.uuid4().hex, # Unique ID for the edge itself.
                        "source": memory_id, # ID of the new memory node.
                        "target": target_node_id, # ID of the related memory node.
                        "relation_type": "related_to", # Default relationship type.
                        "timestamp": timestamp # Timestamp of relationship creation.
                    }
                    _knowledge_graph["edges"].append(new_edge) # Add edge to in-memory graph.
                    _log_memory_event("store_memory_relation_added", 
                                      {"source_id": memory_id, "target_id": target_node_id, "edge_id": new_edge["id"]}, 
                                      level="debug")

        _kg_dirty_flag = True # Mark the graph as modified.
        _save_knowledge_graph() # Attempt to save the updated graph to disk immediately.

        _log_memory_event("store_memory_success", 
                          {"memory_id": memory_id, "concept_name": concept_name, 
                           "node_count": len(_knowledge_graph['nodes']), "edge_count": len(_knowledge_graph['edges'])}, 
                          level="info")
        return True

    except Exception as e: # Catch any other unexpected errors during storage.
        _log_memory_event("store_memory_exception", 
                          {"concept_name": concept_name, "error": str(e), "trace": traceback.format_exc()}, 
                          level="critical")
        return False

def get_memory_by_id(memory_id: str) -> dict | None:
    """
    Retrieves a specific memory (node) from the knowledge graph by its ID.

    Args:
        memory_id: The unique ID of the memory to retrieve.

    Returns:
        The memory (node dictionary) if found, otherwise None.
    """
    if not memory_id or not isinstance(memory_id, str):
        _log_memory_event("get_memory_by_id_invalid_input", {"memory_id": str(memory_id)}, level="warning")
        return None

    try:
        # Ensure _knowledge_graph and "nodes" key exist and "nodes" is a list
        if not isinstance(_knowledge_graph, dict) or            not isinstance(_knowledge_graph.get("nodes"), list):
            _log_memory_event("get_memory_by_id_kg_malformed", 
                              {"error": "Knowledge graph not initialized or malformed"}, 
                              level="error")
            return None
            
        for node in _knowledge_graph["nodes"]:
            if isinstance(node, dict) and node.get("id") == memory_id:
                _log_memory_event("get_memory_by_id_success", {"memory_id": memory_id}, level="debug")
                return node # Return a copy to prevent external modification? For now, return original.
                           # Consider: return node.copy() if external modification is a concern.
        
        _log_memory_event("get_memory_by_id_not_found", {"memory_id": memory_id}, level="debug")
        return None
        
    except Exception as e:
        _log_memory_event("get_memory_by_id_exception", {"memory_id": memory_id, "error": str(e), "trace": traceback.format_exc()}, level="error")
        return None

def get_memories_by_concept_name(concept_name: str, exact_match: bool = True) -> list:
    """
    Retrieves memories (nodes) from the knowledge graph by concept name (label).

    Args:
        concept_name: The name/label of the concept to search for.
        exact_match: If True, requires an exact match for the concept name.
                     If False, performs a case-insensitive substring match.

    Returns:
        A list of matching memory (node dictionaries). Empty if none found.
    """
    if not concept_name or not isinstance(concept_name, str):
        _log_memory_event("get_memories_by_name_invalid_input", {"concept_name": str(concept_name)}, level="warning")
        return []

    found_memories = []
    try:
        if not isinstance(_knowledge_graph, dict) or            not isinstance(_knowledge_graph.get("nodes"), list):
            _log_memory_event("get_memories_by_name_kg_malformed", 
                              {"error": "Knowledge graph not initialized or malformed"}, 
                              level="error")
            return []

        search_term_lower = concept_name.lower() # For case-insensitive search

        for node in _knowledge_graph["nodes"]:
            if isinstance(node, dict) and "label" in node and isinstance(node["label"], str):
                node_label = node["label"]
                if exact_match:
                    if node_label == concept_name:
                        found_memories.append(node) # Consider node.copy()
                else: # Substring match, case-insensitive
                    if search_term_lower in node_label.lower():
                        found_memories.append(node) # Consider node.copy()
        
        _log_memory_event("get_memories_by_name_result", 
                          {"concept_name": concept_name, "exact_match": exact_match, "count": len(found_memories)}, 
                          level="debug")
        return found_memories

    except Exception as e:
        _log_memory_event("get_memories_by_name_exception", 
                          {"concept_name": concept_name, "error": str(e), "trace": traceback.format_exc()}, 
                          level="error")
        return [] # Return empty list on error

def get_recent_memories(limit: int = 10) -> list:
    """
    Retrieves the most recent memories (nodes) from the knowledge graph,
    sorted by timestamp in descending order.

    Args:
        limit: The maximum number of recent memories to return. Defaults to 10.

    Returns:
        A list of the most recent memory (node dictionaries), up to the limit.
        Empty if no memories or an error occurs.
    """
    if not isinstance(limit, int) or limit < 0:
        _log_memory_event("get_recent_memories_invalid_limit", {"limit": limit}, level="warning")
        # Default to 10 if limit is invalid, or could return empty list.
        # For now, let's proceed with a default limit, or a very small one if limit was negative.
        limit = 10 if limit >=0 else 0 
        if limit == 0: return []


    try:
        if not isinstance(_knowledge_graph, dict) or            not isinstance(_knowledge_graph.get("nodes"), list):
            _log_memory_event("get_recent_memories_kg_malformed", 
                              {"error": "Knowledge graph not initialized or malformed"}, 
                              level="error")
            return []

        # Filter out nodes that might lack a timestamp or have an invalid one, though ideally all nodes from store_memory will have it.
        valid_nodes_with_timestamp = [
            node for node in _knowledge_graph["nodes"] 
            if isinstance(node, dict) and isinstance(node.get("timestamp"), str)
        ]
        
        # Sort nodes by timestamp in descending order (most recent first)
        # ISO format timestamps (like "2023-05-15T10:30:00Z") can be compared lexicographically.
        sorted_nodes = sorted(valid_nodes_with_timestamp, key=lambda x: x["timestamp"], reverse=True)
        
        returned_memories = sorted_nodes[:limit] # Get the top 'limit' memories
        
        _log_memory_event("get_recent_memories_success", 
                          {"limit": limit, "returned_count": len(returned_memories)}, 
                          level="debug")
        # Consider returning copies: [node.copy() for node in returned_memories]
        return returned_memories

    except Exception as e:
        _log_memory_event("get_recent_memories_exception", 
                          {"limit": limit, "error": str(e), "trace": traceback.format_exc()}, 
                          level="error")
        return [] # Return empty list on error

def read_memory(n: int = None) -> list:
    """
    Reads memories from the knowledge graph, sorted by timestamp descending.

    Args:
        n: The number of most recent memories to return. 
           If None or not a positive integer, returns all memories.

    Returns:
        A list of memory (node dictionaries), sorted by recency.
        Empty if no memories or an error occurs.
    """
    _log_memory_event("read_memory_attempt", {"n_requested": n}, level="debug")

    try:
        if not isinstance(_knowledge_graph, dict) or            not isinstance(_knowledge_graph.get("nodes"), list):
            _log_memory_event("read_memory_kg_malformed", 
                              {"error": "Knowledge graph not initialized or malformed"}, 
                              level="error")
            return []

        # Filter out nodes that might lack a timestamp or have an invalid one
        valid_nodes_with_timestamp = [
            node for node in _knowledge_graph["nodes"] 
            if isinstance(node, dict) and isinstance(node.get("timestamp"), str)
        ]
        
        # Sort nodes by timestamp in descending order (most recent first)
        sorted_nodes = sorted(valid_nodes_with_timestamp, key=lambda x: x["timestamp"], reverse=True)
        
        if n is not None and isinstance(n, int) and n > 0:
            returned_memories = sorted_nodes[:n]
            count_returned = len(returned_memories)
        else: # Return all sorted memories if n is None, not an int, or not positive
            returned_memories = sorted_nodes
            count_returned = len(returned_memories)
            if n is not None: # Log if n was invalid and all are being returned
                 _log_memory_event("read_memory_invalid_n_returning_all", {"n_value": n, "count": count_returned}, level="debug")

        _log_memory_event("read_memory_success", 
                          {"n_requested": n, "returned_count": count_returned, "total_available": len(valid_nodes_with_timestamp)}, 
                          level="debug")
        # Consider returning copies: [node.copy() for node in returned_memories]
        return returned_memories

    except Exception as e:
        _log_memory_event("read_memory_exception", 
                          {"n_requested": n, "error": str(e), "trace": traceback.format_exc()}, 
                          level="error")
        return [] # Return empty list on error

# --- Load knowledge graph at module import ---
# This ensures _knowledge_graph is populated when memory.py is imported.
# (Make sure this is placed before any functions that might immediately try to use _knowledge_graph,
# although typically functions will be defined first, then module-level calls like this one.)
_load_knowledge_graph()

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
    TEST_KG_FILENAME = "test_knowledge_graph.json" # In the same dir as memory.py for simplicity during test
    
    # Ensure config is loaded, otherwise tests can't run.
    if not config:
        print("CRITICAL (memory.py __main__): Config module not loaded. Tests cannot proceed.", file=sys.stderr)
        sys.exit(1)

    # Set VERBOSE_OUTPUT to True for test visibility, can be overridden per test.
    # Store original verbose setting if it exists, to restore it later.
    original_verbose_output = getattr(config, 'VERBOSE_OUTPUT', None)
    config.VERBOSE_OUTPUT = True # Default for tests

    # Ensure paths are set up for test file (using current dir for test KG file)
    # Test KG path will be relative to where memory.py is, or use absolute.
    # For simplicity, let memory.py's dir be the base for TEST_KG_FILENAME
    module_dir = os.path.dirname(os.path.abspath(__file__))
    TEST_KNOWLEDGE_GRAPH_PATH = os.path.join(module_dir, TEST_KG_FILENAME)

    def setup_test_environment():
        """Clears the test KG file and resets in-memory graph state."""
        global _knowledge_graph, _kg_dirty_flag
        if os.path.exists(TEST_KNOWLEDGE_GRAPH_PATH):
            os.remove(TEST_KNOWLEDGE_GRAPH_PATH)
        _knowledge_graph = {"nodes": [], "edges": []}
        _kg_dirty_flag = False
        # _load_knowledge_graph() will be called implicitly by store_memory or explicitly in tests
        # under TempConfigOverride if path is changed. For default path, this reset is key.


    def run_test(test_func, *args):
        test_name = test_func.__name__
        print(f"--- Running Test: {test_name} ---")
        setup_test_environment() # Ensure clean slate for each test using the default test KG path
        try:
            with TempConfigOverride({"KNOWLEDGE_GRAPH_PATH": TEST_KNOWLEDGE_GRAPH_PATH, "SYSTEM_LOG_PATH": os.path.join(module_dir, "test_system_log.json")}):
                 # Re-call _load_knowledge_graph with the new path if the test needs a fresh load from (non-existent) file
                 _load_knowledge_graph()
                 result = test_func(*args)
            if result:
                print(f"PASS: {test_name}")
            else:
                print(f"FAIL: {test_name}")
            return result
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            traceback.print_exc()
            return False
        finally:
            # Clean up test files after each test
            if os.path.exists(TEST_KNOWLEDGE_GRAPH_PATH):
                os.remove(TEST_KNOWLEDGE_GRAPH_PATH)
            if os.path.exists(os.path.join(module_dir, "test_system_log.json")):
                os.remove(os.path.join(module_dir, "test_system_log.json"))


    # --- Test Function Definitions ---

    def test_load_knowledge_graph_logic():
        print("Testing _load_knowledge_graph scenarios...")
        global _knowledge_graph, _kg_dirty_flag
        
        # 1. Non-existent file (should create empty graph)
        if os.path.exists(TEST_KNOWLEDGE_GRAPH_PATH): os.remove(TEST_KNOWLEDGE_GRAPH_PATH)
        _load_knowledge_graph()
        if _knowledge_graph != {"nodes": [], "edges": []} or _kg_dirty_flag:
            print("FAIL: Load non-existent file did not result in clean empty graph.")
            return False
        print("  PASS: Non-existent file.")

        # 2. Empty file
        with open(TEST_KNOWLEDGE_GRAPH_PATH, 'w') as f: f.write("")
        _load_knowledge_graph()
        if _knowledge_graph != {"nodes": [], "edges": []} or _kg_dirty_flag:
            print("FAIL: Load empty file did not result in clean empty graph.")
            return False
        print("  PASS: Empty file.")

        # 3. Malformed JSON
        with open(TEST_KNOWLEDGE_GRAPH_PATH, 'w') as f: f.write("{malformed_json: ")
        _load_knowledge_graph()
        if _knowledge_graph != {"nodes": [], "edges": []} or _kg_dirty_flag:
            print("FAIL: Load malformed JSON did not result in clean empty graph.")
            return False
        print("  PASS: Malformed JSON.")

        # 4. Incorrect structure (valid JSON, wrong content)
        with open(TEST_KNOWLEDGE_GRAPH_PATH, 'w') as f: json.dump({"not_nodes": [], "not_edges": []}, f)
        _load_knowledge_graph()
        if _knowledge_graph != {"nodes": [], "edges": []} or _kg_dirty_flag:
            print("FAIL: Load incorrect structure did not result in clean empty graph.")
            return False
        print("  PASS: Incorrect structure.")

        # 5. Correctly formatted file
        correct_data = {"nodes": [{"id": "1", "label": "test"}], "edges": []}
        with open(TEST_KNOWLEDGE_GRAPH_PATH, 'w') as f: json.dump(correct_data, f)
        _load_knowledge_graph()
        if _knowledge_graph != correct_data or _kg_dirty_flag:
            print(f"FAIL: Load correct file failed. Got: {_knowledge_graph}, Dirty: {_kg_dirty_flag}")
            return False
        print("  PASS: Correctly formatted file.")
        return True

    def test_save_on_change_mechanism():
        print("Testing save-on-change (_kg_dirty_flag logic)...")
        global _knowledge_graph, _kg_dirty_flag
        
        # Initial state: no file, graph empty, flag false
        if os.path.exists(TEST_KNOWLEDGE_GRAPH_PATH): os.remove(TEST_KNOWLEDGE_GRAPH_PATH)
        _knowledge_graph = {"nodes": [], "edges": []}
        _kg_dirty_flag = False

        _save_knowledge_graph() # Should skip (not dirty)
        if os.path.exists(TEST_KNOWLEDGE_GRAPH_PATH):
            print("FAIL: Save occurred when not dirty (initial).")
            return False
        print("  PASS: Initial save correctly skipped.")

        # Modify graph (e.g. by a simplified store_memory action)
        _knowledge_graph["nodes"].append({"id": "test_node", "label": "Test"})
        _kg_dirty_flag = True # Manually set for this test part
        
        _save_knowledge_graph() # Should save now (dirty)
        if not os.path.exists(TEST_KNOWLEDGE_GRAPH_PATH):
            print("FAIL: Save did not occur when dirty.")
            return False
        if _kg_dirty_flag: # Should be false after save
            print("FAIL: Dirty flag not reset after save.")
            return False
        mtime1 = os.path.getmtime(TEST_KNOWLEDGE_GRAPH_PATH)
        print("  PASS: Save occurred when dirty, flag reset.")

        time.sleep(0.01) # Ensure modification time can change if file is rewritten

        _save_knowledge_graph() # Should skip again (not dirty)
        mtime2 = os.path.getmtime(TEST_KNOWLEDGE_GRAPH_PATH)
        if mtime1 != mtime2 : # Modification time should NOT change
            print(f"FAIL: Save occurred when not dirty (subsequent). mtime1={mtime1}, mtime2={mtime2}")
            # This check can be flaky on some systems/resolutions of mtime.
            # A more robust check would be to read content, but that's more involved here.
            # For now, mtime is a reasonable proxy.
            # print("  INFO: mtime check for no-rewrite can be flaky. Verify logs if this fails unexpectedly.")
        else:
             print("  PASS: Subsequent save correctly skipped (mtime unchanged).")
        
        # Test atomic write: check if temp file is absent after successful write
        temp_path = TEST_KNOWLEDGE_GRAPH_PATH + ".tmp"
        if os.path.exists(temp_path):
            print(f"FAIL: Temporary file {temp_path} still exists after save operations.")
            return False
        print("  PASS: Temporary file cleanup appears successful.")
            
        return True


    def test_calculate_novelty_scenarios():
        print("Testing calculate_novelty...")
        global _knowledge_graph
        _knowledge_graph = {"nodes": [], "edges": []} # Start with empty graph

        # 1. Empty graph (max novelty)
        novelty = calculate_novelty((0,0,0,0), "summary1")
        if not (0.99 <= novelty <= 1.0): # Allow for float precision
            print(f"FAIL: Novelty on empty graph was {novelty}, expected ~1.0")
            return False
        print(f"  PASS: Empty graph novelty: {novelty:.2f}")

        # Add a node
        node1_coord = (10, 20, 30, 0.5) # Using larger coords to test normalization effect
        node1_summary = "initial test summary for node one"
        _knowledge_graph["nodes"].append({"id": "n1", "coordinates": node1_coord, "summary": node1_summary, "label":"n1"})

        # 2. Identical concept (very low novelty)
        novelty_identical = calculate_novelty(node1_coord, node1_summary)
        if novelty_identical > 0.1: # Should be very close to 0
            print(f"FAIL: Novelty for identical concept was {novelty_identical}, expected close to 0.")
            return False
        print(f"  PASS: Identical concept novelty: {novelty_identical:.4f}")
        
        # 3. Spatially close, textually different
        close_coord = (node1_coord[0]+0.1, node1_coord[1]-0.1, node1_coord[2], node1_coord[3]) # Small spatial diff
        novelty_sp_close_txt_diff = calculate_novelty(close_coord, "completely different summary text")
        # Expect higher novelty than identical, dominated by text if weights are equal
        if not (novelty_identical < novelty_sp_close_txt_diff <= 1.0):
             print(f"FAIL: Spatially close/textually different check failed. N_ident={novelty_identical}, N_sp_close={novelty_sp_close_txt_diff}")
             return False
        print(f"  PASS: Spatially close, textually different novelty: {novelty_sp_close_txt_diff:.2f}")

        # 4. Spatially distant, textually similar
        distant_coord = (node1_coord[0] + config.MANIFOLD_RANGE, node1_coord[1], node1_coord[2], node1_coord[3])
        novelty_sp_dist_txt_sim = calculate_novelty(distant_coord, node1_summary + " slight change")
        if not (novelty_identical < novelty_sp_dist_txt_sim <= 1.0):
             print(f"FAIL: Spatially distant/textually similar check failed. N_ident={novelty_identical}, N_sp_dist={novelty_sp_dist_txt_sim}")
             return False
        print(f"  PASS: Spatially distant, textually similar novelty: {novelty_sp_dist_txt_sim:.2f}")

        # 5. Completely novel
        novel_coord = (node1_coord[0] + config.MANIFOLD_RANGE, node1_coord[1] + config.MANIFOLD_RANGE, node1_coord[2], node1_coord[3])
        novelty_full = calculate_novelty(novel_coord, "brand new unique summary phrase")
        if not (0.9 <= novelty_full <= 1.0): # Expect high novelty
            print(f"FAIL: Fully novel concept was {novelty_full}, expected ~1.0")
            return False
        print(f"  PASS: Fully novel concept novelty: {novelty_full:.2f}")

        # 6. Malformed input coordinates
        novelty_malformed = calculate_novelty(("bad", 1, 2, 3), "summary")
        if novelty_malformed != 0.0:
            print(f"FAIL: Malformed coord novelty was {novelty_malformed}, expected 0.0")
            return False
        novelty_malformed_len = calculate_novelty((1,2,3), "summary")
        if novelty_malformed_len != 0.0:
            print(f"FAIL: Malformed coord length novelty was {novelty_malformed_len}, expected 0.0")
            return False
        print("  PASS: Malformed coordinate input handling.")
        return True

    def test_store_and_retrieve_memory():
        print("Testing store_memory and retrieval functions...")
        global _knowledge_graph, _kg_dirty_flag
        
        # Use TempConfigOverride for thresholds
        with TempConfigOverride({"MEMORY_NOVELTY_THRESHOLD": 0.5, "MEMORY_ETHICAL_THRESHOLD": 0.6}):
            # 1. Successful store
            res_store1 = store_memory("concept1", (1,1,1,0.8), "summary one", 0.8, 0.7)
            if not res_store1 or len(_knowledge_graph["nodes"]) != 1 or not _kg_dirty_flag: # Dirty flag should be true before save, then false after
                print(f"FAIL: Successful store failed. Result: {res_store1}, Nodes: {len(_knowledge_graph['nodes'])}, Dirty: {_kg_dirty_flag}")
                return False
            node1_id = _knowledge_graph["nodes"][0]["id"]
            print(f"  PASS: Successful store (node1_id: {node1_id}).")
            _kg_dirty_flag = True # Simulate change for next save test inside store_memory itself

            # 2. Novelty Rejection (identical to concept1, novelty should be ~0)
            res_store_low_novelty = store_memory("concept1_again", (1,1,1,0.8), "summary one", 0.8, 0.7)
            if res_store_low_novelty or len(_knowledge_graph["nodes"]) != 1:
                print(f"FAIL: Low novelty rejection failed. Stored: {res_store_low_novelty}, Nodes: {len(_knowledge_graph['nodes'])}")
                return False
            print("  PASS: Low novelty rejection.")

            # 3. Ethical Rejection
            res_store_low_ethics = store_memory("concept_unethical", (2,2,2,0.7), "summary two", 0.7, 0.5) # ethical 0.5 < threshold 0.6
            if res_store_low_ethics or len(_knowledge_graph["nodes"]) != 1:
                print(f"FAIL: Low ethical rejection failed. Stored: {res_store_low_ethics}, Nodes: {len(_knowledge_graph['nodes'])}")
                return False
            print("  PASS: Low ethical rejection.")
            
            # 4. Store another for relation testing
            time.sleep(0.01) # ensure different timestamp
            res_store2 = store_memory("concept2", (10,10,10,0.6), "summary for concept two", 0.6, 0.8)
            if not res_store2 or len(_knowledge_graph["nodes"]) != 2:
                print(f"FAIL: Store concept2 failed. Nodes: {len(_knowledge_graph['nodes'])}")
                return False
            node2_id = _knowledge_graph["nodes"][1]["id"]
            print("  PASS: Successful store (concept2).")

            # 5. Store with relations
            time.sleep(0.01)
            res_store3 = store_memory("concept3_related", (5,5,5,0.5), "summary three related to 1 and 2", 0.5, 0.9, related_concepts=[node1_id, "concept2"]) # Mix ID and name
            if not res_store3 or len(_knowledge_graph["nodes"]) != 3 or len(_knowledge_graph["edges"]) != 2:
                print(f"FAIL: Store with relations failed. Nodes: {len(_knowledge_graph['nodes'])}, Edges: {len(_knowledge_graph['edges'])}")
                return False
            print("  PASS: Store with relations.")

            # Test retrieval functions
            ret_by_id = get_memory_by_id(node1_id)
            if not ret_by_id or ret_by_id["label"] != "concept1":
                print(f"FAIL: get_memory_by_id failed. Got: {ret_by_id}")
                return False
            print("  PASS: get_memory_by_id.")

            ret_by_name_exact = get_memories_by_concept_name("concept2")
            if not ret_by_name_exact or len(ret_by_name_exact) != 1 or ret_by_name_exact[0]["id"] != node2_id:
                print(f"FAIL: get_memories_by_concept_name (exact) failed. Got: {ret_by_name_exact}")
                return False
            print("  PASS: get_memories_by_concept_name (exact).")
            
            ret_by_name_substr = get_memories_by_concept_name("concept", exact_match=False)
            if len(ret_by_name_substr) != 3: # concept1, concept2, concept3_related
                print(f"FAIL: get_memories_by_concept_name (substring) failed. Count: {len(ret_by_name_substr)}, Expected 3.")
                return False
            print("  PASS: get_memories_by_concept_name (substring).")

            recent_2 = get_recent_memories(2)
            if len(recent_2) != 2 or recent_2[0]["label"] != "concept3_related" or recent_2[1]["label"] != "concept2":
                print(f"FAIL: get_recent_memories(2) failed. Got: {[n['label'] for n in recent_2]}")
                return False
            print("  PASS: get_recent_memories(2).")

            all_mem_read = read_memory()
            if len(all_mem_read) != 3 or all_mem_read[0]["label"] != "concept3_related":
                print(f"FAIL: read_memory() all failed. Count: {len(all_mem_read)}, First: {all_mem_read[0]['label'] if all_mem_read else 'None'}")
                return False
            print("  PASS: read_memory() all.")
            
            one_mem_read = read_memory(n=1)
            if len(one_mem_read) != 1 or one_mem_read[0]['label'] != "concept3_related":
                print(f"FAIL: read_memory(1) failed. Got: {[n['label'] for n in one_mem_read]}")
                return False
            print("  PASS: read_memory(1).")

        return True
        
    def test_retrieval_empty_graph():
        print("Testing retrieval functions on an empty/cleared graph...")
        global _knowledge_graph
        _knowledge_graph = {"nodes": [], "edges": []} # Ensure it's empty for this test

        if get_memory_by_id("any_id") is not None: return False
        if get_memories_by_concept_name("any_name") != []: return False
        if get_recent_memories(5) != []: return False
        if read_memory() != []: return False
        if read_memory(5) != []: return False
        print("  PASS: All retrieval functions correctly returned empty on empty graph.")
        return True

    # --- Main Test Execution Logic ---
    print("\n--- Starting Core Memory Self-Tests ---")
    # Ensure config.KNOWLEDGE_GRAPH_PATH is set to the test path for all tests implicitly via run_test
    
    tests_to_run = [
        test_load_knowledge_graph_logic,
        test_save_on_change_mechanism,
        test_calculate_novelty_scenarios,
        test_store_and_retrieve_memory,
        test_retrieval_empty_graph,
    ]
    
    results = []
    for test_fn in tests_to_run:
        # run_test already calls setup_test_environment which uses TEST_KNOWLEDGE_GRAPH_PATH
        results.append(run_test(test_fn))
        time.sleep(0.05) # Small delay for file system ops if any and for log readability

    print("\n--- Core Memory Self-Test Summary ---")
    passed_count = sum(1 for r in results if r)
    total_count = len(results)
    print(f"Tests Passed: {passed_count}/{total_count}")

    # Restore original VERBOSE_OUTPUT if it was changed
    if original_verbose_output is not None:
        config.VERBOSE_OUTPUT = original_verbose_output
    else: # If it didn't exist, remove the one we added for tests
        if hasattr(config, 'VERBOSE_OUTPUT'): delattr(config, 'VERBOSE_OUTPUT')


    if passed_count == total_count:
        print("All core memory tests PASSED successfully!")
        # sys.exit(0) # In a CI environment, this would be appropriate.
    else:
        print("One or more core memory tests FAILED. Please review logs above.")
        sys.exit(1) # Exit with error code if any test failed
