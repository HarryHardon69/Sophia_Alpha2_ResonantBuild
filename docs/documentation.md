# Sophia_Alpha2_ResonantBuild - Technical Documentation

This document provides more detailed technical information about the modules and components
of the Sophia_Alpha2_ResonantBuild project.

## Table of Contents
*   [Configuration (`config/config.py`)](#configuration-configconfigpy)
*   [Core Cognitive Engine (`core/brain.py`)](#core-cognitive-engine-corebrainpy)

---

## Configuration (`config/config.py`)
The `config/config.py` module centralizes all system-wide configuration settings. It uses environment variables for sensitive data and provides defaults for most parameters. Refer to the inline comments in `config/config.py` for details on each setting.

Key configurable aspects include:
*   Path definitions and helper utilities (`_PROJECT_ROOT`, `get_path`, `ensure_path`).
*   Resource profiles (`LOW`, `MODERATE`, `HIGH`) affecting SNN size and performance.
*   System behavior flags (`VERBOSE_OUTPUT`, `ENABLE_SNN`).
*   SNN parameters (learning rates, neuron properties, optimizer settings).
*   LLM provider selection and API details (OpenAI, LM Studio, Ollama, Mock).
*   Persona selection.
*   Ethics module parameters.
*   Memory and Library settings.
*   Execution environment (`DEVELOPMENT`, `PRODUCTION`).

---

## Core Cognitive Engine (`core/brain.py`)

The `core/brain.py` module implements the cognitive core of Sophia_Alpha2, known as the Spacetime Manifold.

### Overview
The `SpacetimeManifold` class is the primary component of `core/brain.py`. It utilizes a Spiking Neural Network (SNN) built with `snnTorch` to perform cognitive processing. Key functionalities include:

*   **Concept Bootstrapping:** Uses a configured Large Language Model (LLM) to fetch initial information (summary, valence, abstraction, relevance, intensity) about concepts. This data is then used to derive 4D coordinates (x, y, z, t_intensity) for placing the concept within the manifold.
*   **SNN Processing:** Input concepts, represented by their bootstrapped features or derived patterns, are processed through an SNN comprising a fully connected layer followed by Leaky Integrate-and-Fire (LIF) neurons.
*   **STDP/Hebbian Learning:** The synaptic weights of the SNN are updated based on Spike-Timing-Dependent Plasticity (STDP) or a Hebbian-like learning rule. This allows the manifold to learn and adapt from processed information. The `update_stdp` method implements this, influenced by the timing and correlation of pre- and post-synaptic activity.
*   **Awareness Metrics Generation:** The `think` method, which is the main public interface, orchestrates the SNN processing and calculates a set of "awareness metrics." These metrics provide insights into the manifold's state and processing dynamics. Key metrics include:
    *   `curiosity`: Reflects the SNN's activity level and dissonance (coherence).
    *   `context_stability`: Indicates the stability of SNN activity patterns.
    *   `self_evolution_rate`: Measures the degree of STDP-induced weight changes.
    *   `coherence`: Represents the overall consistency or resonance of the manifold's state.
    *   `active_llm_fallback`: Indicates if the SNN was bypassed and an LLM was used directly for the response.
    *   `primary_concept_coord`: The 4D coordinates of the main concept processed.
*   **Singleton Management:** A shared instance of `SpacetimeManifold` is managed via `get_shared_manifold()` for system-wide access.
*   **Logging:** Comprehensive logging of events, SNN steps, and errors is implemented via `_log_system_event`.

### Key Classes and Functions
*   **`SpacetimeManifold`**: The main class encapsulating all cognitive functionalities.
    *   `__init__(self)`: Initializes SNN components, loads configuration.
    *   `bootstrap_concept_from_llm(self, concept_name)`: Fetches and processes concept data from an LLM.
    *   `update_stdp(...)`: Implements the STDP learning rule.
    *   `warp_manifold(self, input_text)`: Runs the main SNN simulation loop with learning.
    *   `think(self, input_text, ...)`: Main entry point; processes input and returns response and awareness metrics.
*   **`get_shared_manifold()`**: Accessor for the singleton `SpacetimeManifold` instance.
*   **`think(input_text, ...)` (in `core/__init__.py`)**: A top-level convenience function that uses `get_shared_manifold().think(...)`.

### Configuration via `config.py`
`core/brain.py` relies heavily on `config/config.py` for its operational parameters. Key settings include:
*   `ENABLE_SNN`: Master switch to enable or disable SNN processing (fallback to LLM).
*   `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL`: For LLM concept bootstrapping.
*   `LLM_CONCEPT_PROMPT_TEMPLATE`: Provider-specific prompt structures.
*   `RESOURCE_PROFILE_TYPE`: Determines SNN size (`MAX_NEURONS`) and simulation parameters (`SNN_TIME_STEPS`).
*   SNN Learning Parameters: `HEBBIAN_LEARNING_RATE`, `STDP_WINDOW_MS`, `STDP_DEPRESSION_FACTOR`.
*   SNN Neuron Parameters: `SNN_LIF_BETA`, `SNN_LIF_THRESHOLD`, `SNN_SURROGATE_SLOPE`.
*   `COHERENCE_UPDATE_FACTOR`.
*   `VERBOSE_OUTPUT`: For streaming thought steps.

The module includes a self-testing suite (`if __name__ == "__main__":`) that verifies core functionalities.
