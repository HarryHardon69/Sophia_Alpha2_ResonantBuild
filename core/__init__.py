"""
Core package for Sophia_Alpha2_ResonantBuild.

This package contains the central cognitive components of the system,
including the SpacetimeManifold (brain), dialogue management, ethics engine, etc.
"""

import sys # Added for stderr in the fallback case of think()

from .brain import SpacetimeManifold, get_shared_manifold

# Top-level convenience function for 'think'
def think(input_text: str, stream_thought_steps: bool = False) -> tuple[list, str, dict]:
    """
    Primary entry point for making Sophia_Alpha2 process input and generate a response.
    Retrieves the shared SpacetimeManifold instance and calls its think() method.

    Args:
        input_text: The text input to be processed.
        stream_thought_steps: Whether to stream thought process steps to console (if verbose).

    Returns:
        A tuple containing:
            - thought_steps_list (list): Log of cognitive steps.
            - response_text_string (str): The generated response.
            - awareness_metrics_dict (dict): Dictionary of awareness metrics.
    """
    manifold = get_shared_manifold()
    if manifold:
        return manifold.think(input_text, stream_thought_steps=stream_thought_steps)
    else:
        # Fallback if manifold couldn't be initialized (e.g., critical config error)
        # This should be rare if config loads correctly.
        error_message = "CRITICAL: SpacetimeManifold not available. Cannot process thought."
        print(error_message, file=sys.stderr) # Also print to stderr
        return (
            [error_message], 
            "I am currently unable to process thoughts due to an internal initialization issue.", 
            {
                "curiosity": 0, "context_stability": 0, "self_evolution_rate": 0, 
                "coherence": 0, "active_llm_fallback": True, 
                "primary_concept_coord": (0,0,0,0), "snn_error": "Manifold not initialized"
            }
        )

__all__ = [
    'SpacetimeManifold',
    'get_shared_manifold',
    'think'
    # Other core components will be added here as they are developed
    # e.g., 'DialogueManager', 'EthicsEngine', etc.
]
