# Sophia_Alpha2_ResonantBuild - Technical Documentation

This document provides more detailed technical information about the modules and components
of the Sophia_Alpha2_ResonantBuild project.

## Table of Contents
*   [Configuration (`config/config.py`)](#configuration-configconfigpy)
*   [Core Cognitive Engine (`core/brain.py`)](#core-cognitive-engine-corebrainpy)
*   [Memory System (`core/memory.py`)](#memory-system-corememorypy)
*   [Persona Management (`core/persona.py`)](#persona-management-corepersonapy)

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
*   **Awareness Metrics Generation:** The `think` method, which is the main public interface, orchestrates the SNN processing and calculates a set of "awareness metrics." These metrics provide insights into the manifold's state and processing dynamics.
*   **Singleton Management:** A shared instance of `SpacetimeManifold` is managed via `get_shared_manifold()` for system-wide access.
*   **Logging:** Comprehensive logging of events, SNN steps, and errors is implemented via `_log_system_event`.

### Key Classes and Functions
*   **`SpacetimeManifold`**: The main class.
    *   `__init__(self)`: Initializes SNN components, loads configuration.
    *   `bootstrap_concept_from_llm(self, concept_name)`: Fetches and processes concept data from an LLM.
    *   `update_stdp(...)`: Implements the STDP learning rule.
    *   `warp_manifold(self, input_text)`: Runs the main SNN simulation loop with learning.
    *   `think(self, input_text, ...)`: Main entry point; processes input and returns response and awareness metrics.
*   **`get_shared_manifold()`**: Accessor for the singleton `SpacetimeManifold` instance.
*   **`think(input_text, ...)` (in `core/__init__.py`)**: A top-level convenience function.

### Configuration via `config.py`
Key settings for `core/brain.py` include:
*   `ENABLE_SNN`, `LLM_PROVIDER`, `LLM_MODEL`, `RESOURCE_PROFILE_TYPE`, SNN learning and neuron parameters, `COHERENCE_UPDATE_FACTOR`, `VERBOSE_OUTPUT`.

The module includes a self-testing suite (`if __name__ == "__main__":`).

---

## Memory System (`core/memory.py`)

The `core/memory.py` module is responsible for Sophia_Alpha2's memory operations, managing a persistent knowledge graph stored as a JSON file (`knowledge_graph.json`).

### Overview
This module allows the system to store and retrieve information about concepts, including their semantic properties, associated manifold coordinates, and relationships. Key functionalities include:

*   **Knowledge Graph Persistence:**
    *   Loads the knowledge graph from `config.KNOWLEDGE_GRAPH_PATH` at startup (`_load_knowledge_graph`). Handles file creation, empty files, and malformed JSON.
    *   Saves the knowledge graph to disk (`_save_knowledge_graph`) only when changes have been made (using a "dirty flag" mechanism for efficiency). Uses atomic writes to prevent data corruption.
*   **Novelty Calculation (`calculate_novelty`):**
    *   Determines the novelty of a new concept by comparing it to existing entries in the knowledge graph.
    *   Combines **spatial novelty** (based on 4D Euclidean distance in the manifold space, using normalized coordinates from `config.MANIFOLD_RANGE`) and **textual novelty** (based on Jaccard similarity of concept summaries).
    *   The contribution of spatial and textual components is weighted by `config.SPATIAL_NOVELTY_WEIGHT` and `config.TEXTUAL_NOVELTY_WEIGHT`.
*   **Memory Storage (`store_memory`):**
    *   Stores new concepts (nodes) in the knowledge graph if they meet criteria defined in `config.py`.
    *   Checks against `config.MEMORY_NOVELTY_THRESHOLD` (using `calculate_novelty`) and `config.MEMORY_ETHICAL_THRESHOLD` (using the provided `ethical_alignment` score).
    *   Stored nodes include a unique ID, label (concept name), 4D coordinates, summary, intensity, ethical alignment score at storage, novelty score at storage, timestamp, and type.
    *   Supports creating relationships (edges) between concepts if `related_concepts` (IDs or names) are provided.
*   **Memory Retrieval:** Provides several functions to access stored memories:
    *   `get_memory_by_id`: Retrieves a single memory by its unique ID.
    *   `get_memories_by_concept_name`: Retrieves memories by their label, supporting exact or case-insensitive substring matching.
    *   `get_recent_memories`: Retrieves the N most recent memories.
    *   `read_memory`: Retrieves all memories or the N most recent, sorted by timestamp (descending).
*   **Logging:** Uses `_log_memory_event` for detailed logging of operations.

### Public Functions (exposed via `core/__init__.py`)
*   **`calculate_novelty(concept_coord: tuple, concept_summary: str) -> float`**: Returns a novelty score (0.0-1.0).
*   **`store_memory(concept_name: str, concept_coord: tuple, summary: str, intensity: float, ethical_alignment: float, related_concepts: list = None) -> bool`**: Returns `True` if stored, `False` otherwise.
*   **`get_memory_by_id(memory_id: str) -> dict | None`**: Returns a node dictionary or `None`.
*   **`get_memories_by_concept_name(concept_name: str, exact_match: bool = True) -> list`**: Returns a list of node dictionaries.
*   **`get_recent_memories(limit: int = 10) -> list`**: Returns a list of node dictionaries.
*   **`read_memory(n: int = None) -> list`**: Returns a list of node dictionaries.

### Configuration via `config.py`
Key settings for `core/memory.py` include:
*   `KNOWLEDGE_GRAPH_PATH`: Path to the JSON file storing the knowledge graph.
*   `MANIFOLD_RANGE`: Used for normalizing coordinates in novelty calculation.
*   `SPATIAL_NOVELTY_WEIGHT`, `TEXTUAL_NOVELTY_WEIGHT`: Weights for combining novelty components.
*   `MEMORY_NOVELTY_THRESHOLD`: Minimum novelty score required to store a memory.
*   `MEMORY_ETHICAL_THRESHOLD`: Minimum ethical alignment score required to store a memory.

The module includes a comprehensive self-testing suite (`if __name__ == "__main__":`).

---

## Persona Management (`core/persona.py`)

The `core/persona.py` module defines and manages Sophia_Alpha2's identity, traits, operational mode, and awareness state. This is crucial for maintaining a consistent and evolving persona throughout her interactions and cognitive processes.

### Module Overview
`persona.py` encapsulates Sophia's characteristics and her dynamic understanding of her own operational state. This state is influenced by metrics received from the cognitive core (`core/brain.py`) and is persisted to a profile file, allowing Sophia's persona to be consistent across sessions and to evolve over time.

### `Persona` Class
This is the central class in `core/persona.py`.

*   **Purpose:** Manages all aspects of Sophia's identity, including her name, current operational mode, inherent traits, and a detailed dictionary of awareness metrics.
*   **Key Attributes:**
    *   `name` (str): The name of the persona, e.g., "Sophia_Alpha2_Default".
    *   `mode` (str): The current operational mode, such as "reflective", "learning", or "active_problem_solving".
    *   `traits` (list[str]): A list of strings defining core characteristics, e.g., `["CuriosityDriven", "EthicallyMinded"]`.
    *   `awareness` (dict): A dictionary holding various metrics that represent Sophia's current state of awareness. This includes:
        *   `curiosity` (float): Her current level of curiosity.
        *   `coherence` (float): The coherence of her current cognitive state.
        *   `context_stability` (float): Stability of the current context.
        *   `self_evolution_rate` (float): Rate of her own developmental change.
        *   `active_llm_fallback` (bool): Whether the system is currently relying on LLM fallbacks.
        *   `primary_concept_coord` (tuple | None): A 4-tuple `(x, y, z, t_intensity_raw)` representing the manifold coordinates of the current primary concept of focus, along with its raw T-intensity (0-1). This T-value, or Focus Intensity, indicates the intensity of focus on this concept. It is `None` if no concept is primary.

### Key Methods
*   **`__init__()`**: Initializes the Persona instance. It sets up default attributes (name, mode, traits, awareness metrics) and then attempts to load an existing persona state from the profile file defined by `config.PERSONA_PROFILE_PATH`. If no profile is found, or if it's malformed, it initializes with defaults and saves a new profile.
*   **`update_awareness(brain_awareness_metrics: dict)`**: Updates the `awareness` dictionary based on new metrics received from `core/brain.py` (typically after a `think()` cycle). This method handles the logic for integrating new values, including the `primary_concept_coord` and its associated raw T-intensity. Changes are automatically saved to the profile.
*   **`get_intro()`**: Generates a string that provides a brief introduction to the persona, including her name, mode, and key awareness metrics like curiosity, coherence, and the current Focus Intensity (T-value) if a primary concept is active.
*   **`save_state()` / `load_state()`**: These methods handle the persistence of the persona's state. `save_state()` writes the current attributes to the JSON profile file and is typically called automatically when changes occur (e.g., in `update_awareness` or after initialization if a new profile is created). `load_state()` reads from the profile file during initialization, gracefully handling missing or malformed files by applying defaults.

### Persistent Data (`persona_profile.json`)
The state of the `Persona` class is persisted in a JSON file.

*   **Path:** The path to this file is defined in `config.py` via the `config.PERSONA_PROFILE_PATH` setting. An example could be `data/private/persona_profile_default.json` or a user-configured path like `data/personas/Sophia_Alpha2_Default.json`.
*   **Structure:** The JSON file generally contains the following structure:
    ```json
    {
      "name": "Sophia_Alpha2_Default",
      "mode": "reflective",
      "traits": ["CuriosityDriven", "EthicallyMinded", "ResonanceAware", "Developmental"],
      "awareness": {
        "curiosity": 0.5,
        "context_stability": 0.5,
        "self_evolution_rate": 0.0,
        "coherence": 0.0,
        "active_llm_fallback": false,
        "primary_concept_coord": [0.123, -0.456, 0.789, 0.85]
      },
      "last_saved": "2023-10-27T10:30:00.123456Z"
    }
    ```
    *   If `primary_concept_coord` is not active or defined, its value will be `null` in the JSON file.
    *   The `primary_concept_coord` array stores the X, Y, Z coordinates (scaled values as used within the manifold) and the fourth element is the raw T-intensity (a float between 0.0 and 1.0) representing the focus level.

The module includes a comprehensive self-testing suite (`if __name__ == "__main__":`) which also serves as an example of how to interact with the `Persona` class.
```
