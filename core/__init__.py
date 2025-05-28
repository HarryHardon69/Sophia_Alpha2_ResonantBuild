"""
Core package for Sophia_Alpha2_ResonantBuild.

This package contains the central cognitive components of the system,
including the SpacetimeManifold (brain), memory systems, dialogue management, 
ethics engine, etc.
"""

import sys # For stderr in the fallback case of think()

# --- Brain Exports ---
from .brain import SpacetimeManifold, get_shared_manifold

# Top-level convenience function for 'think' from brain module
def think(input_text: str, stream_thought_steps: bool = False) -> tuple[list, str, dict]:
    """
    Primary entry point for making Sophia_Alpha2 process input and generate a response.
    Retrieves the shared SpacetimeManifold instance and calls its think() method.
    """
    manifold = get_shared_manifold()
    if manifold:
        return manifold.think(input_text, stream_thought_steps=stream_thought_steps)
    else:
        error_message = "CRITICAL: SpacetimeManifold not available. Cannot process thought."
        print(error_message, file=sys.stderr)
        return (
            [error_message], 
            "I am currently unable to process thoughts due to an internal initialization issue.", 
            {
                "curiosity": 0, "context_stability": 0, "self_evolution_rate": 0, 
                "coherence": 0, "active_llm_fallback": True, 
                "primary_concept_coord": (0,0,0,0), "snn_error": "Manifold not initialized"
            }
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
    # Other core components will be added here as they are developed
    # e.g., 'DialogueManager', 'PersonaManager', etc. 
    # Corrected placeholder from 'EthicsEngine' as it's now implemented.
]
