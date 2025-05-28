# Sophia_Alpha2_ResonantBuild - Technical Documentation

This document provides more detailed technical information about the modules and components
of the Sophia_Alpha2_ResonantBuild project.

## Table of Contents
*   [Configuration (`config/config.py`)](#configuration-configconfigpy)
*   [Core Cognitive Engine (`core/brain.py`)](#core-cognitive-engine-corebrainpy)
*   [Memory System (`core/memory.py`)](#memory-system-corememorypy)
*   [Persona Management (`core/persona.py`)](#persona-management-corepersonapy)
*   [Knowledge Library and Utilities (`core/library.py`)](#knowledge-library-and-utilities-corelibrarypy)
*   [Dialogue Management (`core/dialogue.py`)](#dialogue-management-coredialoguepy)

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

---

## Knowledge Library and Utilities (`core/library.py`)

The `core/library.py` module serves a dual purpose: it provides a collection of shared utility functions and custom exceptions for use across the `core` package, and it manages a persistent, curated knowledge library.

### Module Overview
This module is designed to centralize common functionalities like text processing, data validation, and a standardized exception hierarchy. Additionally, it offers a system for storing and retrieving structured knowledge entries, distinct from the more dynamic `core.memory` (knowledge graph). The library includes an ethical mitigation component (`Mitigator`) to moderate content.

### Key Features & Components

#### 1. Knowledge Persistence
*   **In-Memory Store (`KNOWLEDGE_LIBRARY`):** A Python dictionary that holds the loaded knowledge entries. Each key is an `entry_id` (a SHA256 hash of the content).
*   **Persistent Storage (`library_log.json`):** The knowledge library is persisted to a JSON file, typically located at `data/library_store/library_log.json` (path configured via `config.LIBRARY_LOG_PATH`).
*   **Loading (`_load_knowledge_library()`):** Loads the library from the JSON file into the `KNOWLEDGE_LIBRARY` dictionary when the module is imported. Handles file not found, empty files, and malformed JSON.
*   **Saving (`_save_knowledge_library()`):** Saves the `KNOWLEDGE_LIBRARY` to the JSON file if any changes have been made (tracked by `_library_dirty_flag`). Uses an atomic write process (write to temp file, then replace) to prevent data corruption.

#### 2. Utility Functions
*   **`sanitize_text(input_text: str) -> str`:** Removes leading/trailing whitespace and normalizes multiple internal whitespace characters to a single space.
*   **`summarize_text(text: str, max_length: int) -> str`:** Truncates text to a specified `max_length`, appending "..." if truncation occurs. Handles `None` or empty input.
*   **`is_valid_coordinate(coord: tuple | list) -> bool`:** Validates if the input is a list or tuple containing 3 or 4 numeric elements.

#### 3. Custom Exceptions
A hierarchy of custom exceptions is defined for standardized error handling within the `core` package:
*   **`CoreException(Exception)`:** Base class for all custom core exceptions.
*   Derivatives include: `BrainError`, `PersonaError`, `MemoryError`, `EthicsError`, `LibraryError`, `DialogueError`, `NetworkError`, and `ConfigError`, each intended for issues specific to their respective modules or concerns.

#### 4. `Mitigator` Class
*   **Purpose:** Provides mechanisms for ethical oversight and content moderation. It's designed to identify and reframe or flag content that may be ethically problematic.
*   **Initialization:** Loads thresholds (e.g., `ETHICAL_ALIGNMENT_THRESHOLD`, `MITIGATION_ETHICAL_THRESHOLD`) and lists of sensitive keywords/reframing phrases from `config.py`, with built-in defaults.
*   **Key Method (`moderate_ethically_flagged_content`):**
    *   Takes original text, an ethical score, and a strict mode flag as input.
    *   Evaluates the content based on the ethical score against configured thresholds and checks for sensitive keywords.
    *   If mitigation is triggered, it logs the event and returns a moderated string (e.g., reframing the content, providing a placeholder, or indicating review is needed).
    *   If no mitigation is needed, it returns the original text.

#### 5. Knowledge Management Functions
These functions form the public API for interacting with the knowledge library:
*   **`store_knowledge(content: str, is_public: bool, source_uri: str, author: str) -> str | None`:**
    *   **Role:** Adds a new piece of knowledge to the library.
    *   **Inputs:**
        *   `content` (str): The textual content.
        *   `is_public` (bool, default `False`): Indicates if the content is for public access.
        *   `source_uri` (str, optional): URI of the knowledge source.
        *   `author` (str, optional): Author of the knowledge.
    *   **Key Processes:**
        *   Validates input content.
        *   Generates an `entry_id` (SHA256 hash of content) and other metadata.
        *   Attempts to generate 4D manifold coordinates and raw T-intensity for the content by calling `bootstrap_concept_from_llm` from `core.brain` (via `get_shared_manifold`). Uses fallbacks if brain interaction fails.
        *   Calculates an ethical score for the content using `score_ethics` from `core.ethics`, providing relevant awareness metrics. Uses a default score on failure.
        *   Handles placeholder consent for public items if `config.REQUIRE_PUBLIC_STORAGE_CONSENT` is true.
        *   Constructs and stores the entry, then saves the library.
    *   **Output:** The `entry_id` if stored successfully, otherwise `None`.
*   **`retrieve_knowledge_by_id(entry_id: str) -> dict | None`:**
    *   Retrieves a specific knowledge entry using its unique `entry_id`.
*   **`retrieve_knowledge_by_keyword(keyword: str, search_public: bool, search_private: bool) -> list[dict]`:**
    *   Searches for knowledge entries containing the specified `keyword` (case-insensitive) within their `content_preview` or `full_content`.
    *   Allows filtering based on whether entries are public or private using the `search_public` and `search_private` boolean flags.

### Data Structure for `library_log.json`
The `library_log.json` file stores the `KNOWLEDGE_LIBRARY` as a JSON object, where each key is an `entry_id` (SHA256 hash), and the value is an object representing the knowledge entry.

**Example Entry:**
```json
{
  "entry_id_hash_example": {
    "id": "entry_id_hash_example",
    "timestamp": "2023-12-01T12:00:00.000000Z",
    "content_hash": "entry_id_hash_example",
    "content_preview": "A concise summary of the knowledge item...",
    "full_content": "The complete textual content of the knowledge item, which can be extensive.",
    "is_public": true,
    "source_uri": "https://example.com/original_article",
    "author": "Dr. Jane Doe",
    "coordinates": [0.123, -0.456, 0.789, 1.234],
    "raw_t_intensity": 0.85,
    "ethics_score": 0.92,
    "version": "1.0"
  }
  // ... other entries keyed by their IDs
}
```
*   **`coordinates`**: A 4-tuple `(x, y, z, t_coord)` representing the concept's position in the manifold (scaled values), or `null` if not generated.
*   **`raw_t_intensity`**: A float between 0.0 and 1.0 representing the T-value (intensity) derived from the brain's LLM bootstrapping, or `null`/`0.0` if not applicable.
*   **`ethics_score`**: A float between 0.0 and 1.0 representing the assessed ethical alignment of the content.

The module includes a comprehensive self-testing suite (`if __name__ == "__main__":`) to validate its functionalities.

---

## Dialogue Management (`core/dialogue.py`)

The `core/dialogue.py` module acts as the central nervous system for Sophia_Alpha2's interactions. It's responsible for processing user input, orchestrating the cognitive and ethical faculties of the system, and generating coherent, context-aware responses.

### Module Overview
`dialogue.py` is the primary orchestrator. It connects the `core.brain` (for thinking), `core.persona` (for state and identity), `core.ethics` (for moral alignment), `core.memory` (for recalling past interactions/knowledge), and `core.library` (for curated knowledge and utilities like content mitigation) to produce meaningful dialogue. It also provides the command-line interface (CLI) for direct user interaction.

### Key Functions

*   **`generate_response(user_input: str, stream_thought_steps: bool = False) -> tuple[str, list, dict]`**
    *   **Core Responsibility:** This function is the workhorse for processing a single piece of user input and generating Sophia's reply. It returns a tuple containing the final response string for the user, a list of strings detailing the internal "thought steps" taken, and a dictionary of the latest awareness metrics.
    *   **Orchestration Flow:**
        1.  **Persona Retrieval:** Obtains the current `Persona` instance.
        2.  **Brain Interaction:** Calls `core.brain.think()` with the user input to get the initial response, thought steps, and raw awareness metrics.
        3.  **Persona Update:** Updates the `Persona` instance's awareness state with the metrics from the brain.
        4.  **Ethical Scoring:** Uses `core.ethics.score_ethics()` to evaluate the ethical alignment of the generated response and context.
        5.  **Memory Storage:** Records key aspects of the interaction (user input, Sophia's response, ethical score, relevant coordinates) into the `core.memory` knowledge graph using `store_memory()`.
        6.  **Content Mitigation:** Employs the `Mitigator` class from `core.library` to moderate the brain's response if its ethical score falls below configured thresholds, potentially reframing or flagging the content.
        7.  **Ethical Trend Tracking:** Updates ethical trends using `core.ethics.track_trends()` based on the interaction's ethical score.
        8.  **Response Formatting:** Constructs the final response string, often prefixing it with persona mode and ethical score indicators.
    *   **Error Handling:** Each step in the orchestration is wrapped in try-except blocks to ensure graceful handling of failures in any dependent module, logging errors and attempting to provide a fallback response.

*   **`dialogue_loop(enable_streaming_thoughts: bool = None)`**
    *   **Core Responsibility:** Provides the main entry point for continuous, interactive command-line dialogue with Sophia_Alpha2.
    *   **Features:**
        *   **Dynamic Prompt:** Displays a prompt that includes the current persona's name, mode, and key awareness metrics (e.g., curiosity, coherence) to provide context to the user.
        *   **Special Commands:** Handles commands prefixed with `!`:
            *   `!stream`: Toggles the real-time display of thought steps from `generate_response`.
            *   `!persona`: Shows the current persona's detailed information.
            *   `!ethicsdb`, `!memgraph`, `!library`: Debug commands to inspect the current state of the ethics database, memory knowledge graph, and knowledge library respectively.
            *   `!help`: Displays a list of available commands.
            *   `quit` / `exit`: Terminates the dialogue loop.
        *   **Query Processing:** For any input not recognized as a command, it calls `generate_response()` to get Sophia's reply and then prints the response and, if enabled, the thought steps.

### Interaction Flow
`core/dialogue.py` serves as the central hub. When a user provides input via `dialogue_loop()`:
1.  The input is passed to `generate_response()`.
2.  `generate_response()` coordinates with `brain` for initial processing.
3.  `persona` is updated with new awareness data.
4.  `ethics` assesses the response.
5.  `memory` records the interaction.
6.  `library` (specifically `Mitigator`) may moderate the response.
7.  The final, potentially moderated, response is returned to `dialogue_loop()` and presented to the user.
This cycle ensures that all core components contribute to Sophia's behavior in a cohesive manner.

The module includes a comprehensive self-testing suite (`if __name__ == "__main__":`) to validate its functionalities, especially the complex orchestration within `generate_response` and the command handling in `dialogue_loop`.
```
