"""
Core package for Sophia_Alpha2_ResonantBuild.

This package contains the central cognitive components of the system,
including the SpacetimeManifold (brain), memory systems, dialogue management, 
ethics engine, etc.
"""

# Version for the core package
__version__ = "0.1.0"

import sys # For stderr in the fallback case of think()

# --- Brain Exports ---
from .brain import SpacetimeManifold, get_shared_manifold

# Top-level convenience function for 'think' from brain module
def think(input_text: str, stream_thought_steps: bool = False) -> tuple[list, str, dict]:
    """
    Primary entry point for making Sophia_Alpha2 process input and generate a response.
    Retrieves the shared SpacetimeManifold instance and calls its think() method.

    Args:
        input_text (str): The text input to be processed.
        stream_thought_steps (bool, optional): If True, enables more verbose
                                               thought step logging from the brain.
                                               Defaults to False.

    Returns:
        tuple[list, str, dict]: A tuple containing:
            - thought_steps (list): A list of strings describing internal processing.
            - response_text (str): The generated textual response.
            - awareness_metrics (dict): A dictionary of current awareness metrics.
              In case of manifold initialization failure or other errors (like invalid input),
              this will contain default/error values including an 'snn_error' key
              or other relevant error indicators.
    """
    if not isinstance(input_text, str) or not input_text.strip():
        # Try to import config for default error messages, with a hardcoded fallback
        cfg = None
        try:
            from .. import config as cfg
        except ImportError:
            pass

        error_message = "Input validation failed: input_text must be a non-empty string."
        user_facing_error = "Invalid input provided. Please provide a valid text input."
        default_awareness = getattr(cfg, 'AWARENESS_ERROR_DEFAULTS', {
            "curiosity": 0, "context_stability": 0, "self_evolution_rate": 0,
            "coherence": 0, "active_llm_fallback": True,
            "primary_concept_coord": (0,0,0,0), "snn_error": "Invalid input"
        })
        print(error_message, file=sys.stderr) # Or use a proper logger
        return ([error_message], user_facing_error, default_awareness.copy())
    # TODO: Consider adding testing hooks or mechanisms here to allow for more granular testing of the think process, potentially mocking the manifold or inspecting intermediate states.
    # TODO: Optimize get_shared_manifold() with caching if manifold initialization is resource-intensive and the manifold instance can be reused across multiple 'think' calls, provided its state management allows for this.
    manifold = get_shared_manifold()
    if manifold:
        return manifold.think(input_text, stream_thought_steps=stream_thought_steps)
    else:
        # TODO: Implement retry logic for get_shared_manifold() if initialization can be transiently problematic. This might involve a few attempts with increasing backoff before failing completely.
        # TODO: Cache the config import. Importing 'config' repeatedly in this error path, or in the input validation path, might be inefficient if it's a frequent occurrence. Consider loading it once at the module level or caching its import result.
        # Try to import config to get default error messages, with a hardcoded fallback if unavailable
        cfg = None
        try:
            from .. import config as cfg
        except ImportError:
            pass # cfg remains None

        error_message = getattr(cfg, 'CRITICAL_MANIFOLD_ERROR_MSG', "CRITICAL: SpacetimeManifold not available. Cannot process thought.")
        # TODO: Sanitize USER_FACING_MANIFOLD_ERROR_MSG. While it's user-facing, ensure it doesn't inadvertently leak exploitable details about the internal state if errors become more specific in the future.
        user_facing_error = getattr(cfg, 'USER_FACING_MANIFOLD_ERROR_MSG', "I am currently unable to process thoughts due to an internal initialization issue.")
        default_awareness = getattr(cfg, 'AWARENESS_ERROR_DEFAULTS', {
            "curiosity": 0, "context_stability": 0, "self_evolution_rate": 0,
            "coherence": 0, "active_llm_fallback": True,
            "primary_concept_coord": (0,0,0,0), "snn_error": "Manifold not initialized"
        })
        
        # TODO: Align logging with a global LOG_LEVEL from config. Instead of print_to_stderr, use a proper logger that respects configured log levels (e.g., from config.LOG_LEVEL) to control verbosity and output streams for errors.
        # TODO: Ensure that 'error_message' and any other data logged to sys.stderr in this block do not contain sensitive operational details. Sanitize if necessary.
        print(error_message, file=sys.stderr)
        return (
            [error_message], 
            user_facing_error, 
            default_awareness.copy() # Return a copy to prevent modification of the default
        )

# --- Memory Exports ---
from .memory import (
    calculate_novelty,
    store_memory,
    get_memory_by_id,
    get_memories_by_concept_name,
    get_recent_memories,
    read_memory
)

# --- Ethics Exports ---
from .ethics import (
    score_ethics,
    track_trends
)

# --- Persona Exports ---
from .persona import Persona

# --- Library Exports ---
from .library import (
    # Public API functions
    store_knowledge,
    retrieve_knowledge_by_id,
    retrieve_knowledge_by_keyword,
    # Utility classes/functions
    Mitigator,
    sanitize_text,
    summarize_text,
    is_valid_coordinate,
    # Custom Exceptions
    CoreException, 
    BrainError, 
    PersonaError, 
    MemoryError, 
    EthicsError, 
    LibraryError, 
    DialogueError, 
    NetworkError, 
    ConfigError
)

# --- Dialogue Exports ---
from .dialogue import generate_response, dialogue_loop, get_dialogue_persona

# --- GUI Exports ---
from .gui import start_gui

__all__ = [
    # Brain components
    'SpacetimeManifold',
    'get_shared_manifold',
    'think',
    # Memory components
    'calculate_novelty',
    'store_memory',
    'get_memory_by_id',
    'get_memories_by_concept_name',
    'get_recent_memories',
    'read_memory',
    # Ethics components
    'score_ethics',
    'track_trends',
    # Persona component
    'Persona',
    # Library components
    'store_knowledge',
    'retrieve_knowledge_by_id',
    'retrieve_knowledge_by_keyword',
    'Mitigator',
    'sanitize_text',
    'summarize_text',
    'is_valid_coordinate',
    'CoreException', 
    'BrainError', 
    'PersonaError', 
    'MemoryError', 
    'EthicsError', 
    'LibraryError', 
    'DialogueError', 
    'NetworkError', 
    'ConfigError',
    # Dialogue components
    'generate_response',
    'dialogue_loop',
    'get_dialogue_persona',
    # GUI components
    'start_gui',
    # Other core components will be added here as they are developed
    # e.g., 'DialogueManager', 'PersonaManager', etc. 
    # Corrected placeholder from 'EthicsEngine' as it's now implemented.
]
