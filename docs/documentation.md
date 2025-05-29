# Sophia_Alpha2_ResonantBuild - Technical Documentation

This document provides more detailed technical information about the modules and components
of the Sophia_Alpha2_ResonantBuild project.

## Table of Contents
*   [Main Application Entry Point (`main.py`)](#main-application-entry-point-mainpy)
*   [Configuration (`config/config.py`)](#configuration-configconfigpy)
*   [Core Cognitive Engine (`core/brain.py`)](#core-cognitive-engine-corebrainpy)
*   [Persona Management (`core/persona.py`)](#persona-management-corepersonapy)
*   [Memory System (`core/memory.py`)](#memory-system-corememorypy)
*   [Knowledge Library and Utilities (`core/library.py`)](#knowledge-library-and-utilities-corelibrarypy)
*   [Ethical Framework (`core/ethics.py`)](#ethical-framework-coreethicspy)
*   [Dialogue Management (`core/dialogue.py`)](#dialogue-management-coredialoguepy)
*   [Graphical User Interface (`core/gui.py`)](#graphical-user-interface-coreguipy)

---

## Main Application Entry Point (`main.py`)

### Overview
The `main.py` script is the central and primary entry point for launching and managing the Sophia_Alpha2_ResonantBuild application. Its main responsibilities include:
*   Parsing command-line arguments to allow user control over the application's behavior at startup.
*   Ensuring the project's Python path is correctly set up for module resolution.
*   Loading the main configuration from `config/config.py`.
*   Overriding specific configurations (like `VERBOSE_OUTPUT`) based on CLI arguments.
*   Ensuring that essential data and logging directories are present by calling `config.ensure_path()` for relevant directory paths defined in `config.py`.
*   Performing critical initializations, such as creating the shared `SpacetimeManifold` instance from `core.brain`.
*   Determining and launching the appropriate user interface (Command-Line Interface or Graphical User Interface).

### Key Functionality

*   **Argument Parsing:**
    `main.py` uses the `argparse` module to handle command-line arguments:
    *   `--interface {cli,gui}`: Allows the user to explicitly choose between the Command-Line Interface (`cli`) or the Streamlit-based Graphical User Interface (`gui`). If not provided, the default behavior is determined by the `config.ENABLE_GUI` setting.
    *   `--query "Your question here"`: If provided, `main.py` submits this single query to Sophia_Alpha2 via the CLI, prints the response, and then exits. This forces the interface to 'cli'.
    *   `-v`, `--verbose`: Enables verbose logging output for the current session. This overrides the `VERBOSE_OUTPUT` setting in `config.py`.

*   **Configuration Handling:**
    *   The script robustly imports the `config` object from `config/config.py`. If this import fails, a fatal error is logged, and the application exits.
    *   The `--verbose` CLI argument directly influences `config.VERBOSE_OUTPUT`, allowing runtime control over logging verbosity.

*   **Interface Launching:**
    *   Based on the parsed arguments and the `config.ENABLE_GUI` setting, `main_logic` (a function within `main.py`) determines the `effective_interface`.
    *   **GUI Mode:** If "gui" is selected and enabled, `core.gui.start_gui()` is called. If `start_gui()` fails (e.g., Streamlit not installed or other GUI-specific errors), `main.py` logs the error and attempts to fall back to CLI mode.
    *   **CLI Mode:** If "cli" is selected (or as a fallback from GUI failure):
        *   If a `--query` was provided, `core.dialogue.generate_response()` is called directly (with `stream_thought_steps` potentially set by `config.DEFAULT_SINGLE_QUERY_STREAM_THOUGHTS`), and its output is printed.
        *   Otherwise, the interactive `core.dialogue.dialogue_loop()` is initiated.

### Error Handling
`main.py` includes top-level `try-except` blocks to catch critical errors during initialization (like config import failures) or during the execution of `main_logic`. These handlers print detailed error messages and tracebacks to `sys.stderr` and ensure the application exits with an appropriate status code.

---

## Configuration (`config/config.py`)

### Overview
The `config/config.py` module is the cornerstone for all system-wide configuration settings. It centralizes parameters that control application behavior, paths, resource usage, API interactions, and default values for various operational aspects. The primary design principle is to load configurations from environment variables first, falling back to predefined default values if the environment variables are not set or are invalid.

### Key Features
*   **Environment Variable Loading:** Most configuration parameters are sourced using `os.getenv('VAR_NAME', DEFAULT_VALUE)`. This allows for easy customization in different deployment environments without code changes.
*   **Centralized Defaults:** For every parameter loaded from an environment variable, a corresponding `DEFAULT_PARAMETER_NAME` constant is defined within `config.py`. This makes `config.py` the single source of truth for all default settings.
*   **Type Conversion with Error Handling:** Values retrieved from environment variables (which are strings) are explicitly converted to their appropriate types (e.g., `float`, `int`, `bool`). These conversions are wrapped in `try-except ValueError` blocks to catch invalid inputs, log a warning, and revert to the defined default, enhancing robustness.
*   **Path Management:**
    *   `_PROJECT_ROOT`: Dynamically determines the project's root directory.
    *   `get_path(relative_path)`: Constructs absolute paths relative to the project root.
    *   `ensure_path(file_or_dir_path)`: Utility function to create directories if they don't exist, used for ensuring data, log, and other essential directories are available.
*   **Categorized Configuration Sections:** Settings are grouped into logical sections such as:
    *   Path Configuration (e.g., `DATA_DIR`, `LOG_DIR`, `PERSONA_PROFILE_PATH`)
    *   Resource Management (e.g., `_RESOURCE_PROFILES`, `MANIFOLD_RANGE`)
    *   System Behavior (e.g., `VERBOSE_OUTPUT`, `ENABLE_SNN`, SNN parameters like `HEBBIAN_LEARNING_RATE`, `SNN_BATCH_SIZE`)
    *   API Keys and Endpoints (e.g., `ENABLE_LLM_API`, `LLM_PROVIDER`, `_LLM_CONFIG` with defaults like `DEFAULT_OPENAI_BASE_URL`)
    *   Ethics Module Configuration (e.g., `ETHICAL_FRAMEWORK`, `ETHICAL_ALIGNMENT_THRESHOLD`, various weights and thresholds for ethics scoring and trend analysis)
    *   Memory/Library Configuration (e.g., `MEMORY_NOVELTY_THRESHOLD`, `DEFAULT_KNOWLEDGE_COHERENCE`)
    *   Default values for other core modules (e.g., `AWARENESS_ERROR_DEFAULTS` for `core/__init__.py`, `DEFAULT_PERSONA_MODE` for `core/persona.py`, `GUI_RESPONSE_STREAMING_DELAY` for `core/gui.py`).
*   **Validation and Self-Test:** Includes `validate_config()` to check critical settings and `self_test_config_paths_and_creation()` to verify path utilities, typically run when `config.py` is executed directly.

### Usage by Other Modules
Other modules import the `config` object (e.g., `from .. import config`) and access parameters as attributes (e.g., `config.MANIFOLD_RANGE`, `config.LLM_TEMPERATURE`). The use of `getattr(config, 'PARAMETER_NAME', fallback_value)` is common in other modules to gracefully handle cases where a parameter might not be set in specific (e.g., testing) configurations, though `config.py` aims to provide comprehensive defaults.

---

## Core Cognitive Engine (`core/brain.py`)

### Overview
The `core/brain.py` module implements the cognitive core of Sophia_Alpha2, the `SpacetimeManifold`. It's responsible for SNN-based processing, LLM interaction for concept bootstrapping, learning through STDP, and generating awareness metrics.

### Key Classes and Functions
*   **`SpacetimeManifold`**:
    *   `__init__(self)`: Initializes SNN components (layers, neurons using `snnTorch`), loads parameters from `config` (e.g., `MANIFOLD_RANGE`, `SNN_INPUT_SIZE`, learning rates, `SNN_BATCH_SIZE`), and sets up the PyTorch device. Default values for these parameters are referenced from `config` (e.g., `config.DEFAULT_SNN_INPUT_SIZE`).
    *   `_mock_phi3_concept_data(self, concept_name)`: Internal helper providing fallback mock data for concepts.
    *   `_try_connect_llm(self)`: Checks LLM API connectivity based on `config.LLM_BASE_URL` and `config.LLM_CONNECTION_TIMEOUT`.
    *   `bootstrap_concept_from_llm(self, concept_name)`: Fetches concept data (summary, valence, etc.) from an LLM (configured via `config.LLM_PROVIDER`, `config.LLM_MODEL`, `config.LLM_CONCEPT_PROMPT_TEMPLATE`, `config.LLM_TEMPERATURE`). Uses `requests` for API calls. Implements regex fallback for parsing. Converts LLM data to float using internal error handling.
    *   `update_stdp(...)`: Implements STDP learning, using `config.HEBBIAN_LEARNING_RATE` (as `self.lr_stdp`), `config.STDP_WINDOW_MS`, `config.STDP_DEPRESSION_FACTOR`, and `config.COHERENCE_UPDATE_FACTOR`.
    *   `warp_manifold(self, input_text)`: Main SNN simulation loop. Activates input neurons based on `config.SNN_INPUT_ACTIVE_FRACTION`.
    *   `think(self, input_text, ...)`: Orchestrates `warp_manifold` or `bootstrap_concept_from_llm` (if SNN disabled or failed). Initializes awareness metrics using `config.DEFAULT_BRAIN_AWARENESS_METRICS`. Calculates curiosity and context stability using factors/defaults from `config` (e.g., `config.CURIOSITY_COHERENCE_FACTOR`).
*   **`get_shared_manifold()`**: Accessor for the singleton `SpacetimeManifold` instance.
*   **`reset_manifold_for_test()`**: Utility for testing.

### Data File Interactions
*   Does not directly read/write persistent data files itself but relies on `config` for parameters that might define paths for other modules.
*   Logs extensively to the system log file defined by `config.SYSTEM_LOG_PATH`.

### Key Configurable Parameters (from `config.py`)
*   `ENABLE_SNN`, `VERBOSE_OUTPUT`
*   `MANIFOLD_RANGE`, `RESOURCE_PROFILE` (for SNN dimensions, time steps)
*   SNN specific: `SNN_INPUT_SIZE`, `SNN_BATCH_SIZE`, `SNN_INPUT_ACTIVE_FRACTION`, `HEBBIAN_LEARNING_RATE`, `STDP_LEARNING_RATE`, `STDP_WINDOW_MS`, `STDP_DEPRESSION_FACTOR`, `SNN_LIF_BETA`, `SNN_LIF_THRESHOLD`, `SNN_SURROGATE_SLOPE`, `SNN_OPTIMIZER_LR`.
*   LLM related: `ENABLE_LLM_API`, `LLM_PROVIDER`, `LLM_TEMPERATURE`, `LLM_CONNECTION_TIMEOUT`, `LLM_REQUEST_TIMEOUT`, and settings within `_LLM_CONFIG` (API keys, base URLs, model names, prompt templates).
*   `DEFAULT_BRAIN_AWARENESS_METRICS`, `CURIOSITY_COHERENCE_FACTOR`, `CONTEXT_STABILITY_STD_DEV_FACTOR`, `DEFAULT_CONTEXT_STABILITY_SINGLE_READING`, `DEFAULT_CONTEXT_STABILITY_NO_READING`.

---

## Persona Management (`core/persona.py`)

### Overview
The `core/persona.py` module defines Sophia_Alpha2's identity, including her name, traits, operational mode, and an evolving set of "awareness" metrics. It ensures that this persona state is persisted and can be loaded across sessions.

### `Persona` Class
*   **`__init__(self)`**:
    *   Initializes default attributes for name, mode, traits, and the `awareness` dictionary. These primary defaults are sourced from `config.py` (e.g., `config.DEFAULT_PERSONA_NAME`, `config.DEFAULT_PERSONA_MODE`, `config.DEFAULT_PERSONA_TRAITS`, `config.DEFAULT_PERSONA_AWARENESS`).
    *   The actual persona name and profile path are determined by `config.PERSONA_NAME` and `config.PERSONA_PROFILE_PATH`.
    *   Ensures the profile directory exists (using `config.ensure_path` if available).
    *   Calls `self.load_state()` to load from the profile file or initialize a new default state.
*   **`save_state(self)`**: Saves the current state (name, mode, traits, awareness, last_saved timestamp) to the JSON profile specified by `self.profile_path`.
*   **`_initialize_default_state_and_save(self)`**: Resets attributes to their defaults (again, sourced from `config.py`, e.g., `config.DEFAULT_PERSONA_NAME_RESET`) and saves the state. Used for initial setup or if loading fails.
*   **`load_state(self)`**: Loads state from the JSON profile. Handles file-not-found, empty files, and malformed JSON by calling `_initialize_default_state_and_save()`. It carefully validates and type-converts loaded data, especially for the `awareness` dictionary and `primary_concept_coord`.
*   **`get_intro(self) -> str`**: Returns a formatted string summarizing the persona's current state.
*   **`update_awareness(self, brain_awareness_metrics: dict)`**: Updates the `self.awareness` dictionary based on metrics from `core.brain.think()`. Handles type validation and the specific structure of `primary_concept_coord`. Saves the state if changes occur.

### Data File Interactions
*   **Reads from and writes to:** The persona profile JSON file. The path is determined by `config.PERSONA_PROFILE_PATH`, which itself defaults based on `config.PERSONA_DIR` and `config.PERSONA_NAME`. A fallback path using `data/private/persona_profile_default.json` is used if `config` is entirely unavailable.

### Key Configurable Parameters (from `config.py`)
*   `PERSONA_NAME`: Defines the active persona and part of the default profile filename.
*   `PERSONA_PROFILE_PATH`: Explicit path to the persona profile JSON file.
*   `DEFAULT_PERSONA_NAME`, `DEFAULT_PERSONA_MODE`, `DEFAULT_PERSONA_TRAITS`, `DEFAULT_PERSONA_AWARENESS`, `DEFAULT_PERSONA_PROFILE_FILENAME`, `DEFAULT_PERSONA_NAME_RESET`: Provide the primary default values for persona attributes.
*   `VERBOSE_OUTPUT`: Controls informational print statements within the module (suppressed during tests).

---

## Memory System (`core/memory.py`)

### Overview
`core/memory.py` manages Sophia_Alpha2's persistent knowledge graph, which stores concepts as nodes and relationships as edges. It includes functionalities for novelty calculation, ethical filtering before storage, and various retrieval methods.

### Key Features & Functions
*   **Knowledge Graph Persistence (`_knowledge_graph`, `_load_knowledge_graph`, `_save_knowledge_graph`):**
    *   The graph is stored in-memory in `_knowledge_graph` and persisted as a JSON file specified by `config.KNOWLEDGE_GRAPH_PATH`.
    *   Atomic saves and robust loading (handles missing/empty/corrupt files) are implemented.
*   **`calculate_novelty(concept_coord: tuple, concept_summary: str) -> float`**:
    *   Calculates a novelty score (0-1) for a new concept.
    *   Combines spatial novelty (Euclidean distance to nearest existing node, normalized by `config.MANIFOLD_RANGE`) and textual novelty (1 - max Jaccard similarity with existing summaries).
    *   Weights for spatial vs. textual novelty are from `config.SPATIAL_NOVELTY_WEIGHT` and `config.TEXTUAL_NOVELTY_WEIGHT`.
    *   Uses `config.TEXTUAL_NOVELTY_EMPTY_SUMMARY_SCORE` if the input summary is empty.
*   **`store_memory(concept_name: str, concept_coord: tuple, summary: str, intensity: float, ethical_alignment: float, related_concepts: list = None) -> bool`**:
    *   Stores a concept if it meets novelty (`config.MEMORY_NOVELTY_THRESHOLD`) and ethical (`config.MEMORY_ETHICAL_THRESHOLD`) thresholds.
    *   Nodes include ID, label, coordinates, summary, intensity, ethical score, novelty score, timestamp, and type (e.g., `config.MEMORY_NODE_TYPE_CONCEPT`).
    *   Can create edges if `related_concepts` are provided, using `config.MEMORY_DEFAULT_RELATION_TYPE`.
*   **Retrieval Functions:**
    *   `get_memory_by_id(memory_id: str)`
    *   `get_memories_by_concept_name(concept_name: str, exact_match: bool = True)`
    *   `get_recent_memories(limit: int = config.DEFAULT_RECENT_MEMORIES_LIMIT)`
    *   `read_memory(n: int = None)`
*   **Logging:** Uses `_log_memory_event` to log to `config.SYSTEM_LOG_PATH`.

### Data File Interactions
*   **Reads from and writes to:** The knowledge graph JSON file, path defined by `config.KNOWLEDGE_GRAPH_PATH`.
*   **Writes to:** System log file defined by `config.SYSTEM_LOG_PATH`.

### Key Configurable Parameters (from `config.py`)
*   `KNOWLEDGE_GRAPH_PATH`
*   `MANIFOLD_RANGE` (for novelty calculation)
*   `SPATIAL_NOVELTY_WEIGHT`, `TEXTUAL_NOVELTY_WEIGHT`, `TEXTUAL_NOVELTY_EMPTY_SUMMARY_SCORE`
*   `MEMORY_NOVELTY_THRESHOLD`, `MEMORY_ETHICAL_THRESHOLD`
*   `MEMORY_NODE_TYPE_CONCEPT`, `MEMORY_DEFAULT_RELATION_TYPE`
*   `DEFAULT_RECENT_MEMORIES_LIMIT`

---

## Knowledge Library and Utilities (`core/library.py`)

### Overview
`core/library.py` provides shared utilities, custom exceptions, the `Mitigator` class for ethical content moderation, and manages a persistent curated knowledge library.

### Key Features & Components
*   **Custom Exceptions:** Defines a hierarchy of exceptions (e.g., `CoreException`, `BrainError`, `LibraryError`) for standardized error handling across `core` modules.
*   **Text Utilities:**
    *   `sanitize_text(input_text: str)`: Cleans whitespace.
    *   `summarize_text(text: str, max_length: int = config.DEFAULT_SUMMARY_MAX_LENGTH)`: Truncates text. The `max_length` is used by various calling functions, which source their specific limits from `config`.
    *   `is_valid_coordinate(coord: tuple | list)`: Validates 3D/4D numeric coordinates.
*   **`Mitigator` Class:**
    *   Moderates content based on ethical scores and sensitive keywords.
    *   Thresholds (`ETHICAL_ALIGNMENT_THRESHOLD`, `MITIGATION_ETHICAL_THRESHOLD`, `MITIGATION_SEVERE_ETHICAL_SCORE_THRESHOLD`, `MITIGATION_STRICT_CAUTION_ETHICAL_SCORE_THRESHOLD`) are sourced from `config`.
    *   Sensitive keywords and reframing phrases can be configured via `config` (e.g., `config.LIBRARY_SENSITIVE_KEYWORDS`), with hardcoded fallbacks.
    *   Uses `config.MITIGATION_LOG_SUMMARY_MAX_LENGTH` for logged summaries.
*   **Knowledge Library Persistence (`_load_knowledge_library`, `_save_knowledge_library`):**
    *   Manages `KNOWLEDGE_LIBRARY` (in-memory dict) persisted to `library_log.json` (path from `config.LIBRARY_LOG_PATH`).
    *   Features atomic saves and robust loading.
*   **`store_knowledge(...)`**:
    *   Stores curated knowledge entries.
    *   Assigns manifold coordinates (via `core.brain`) and ethical scores (via `core.ethics`).
    *   Uses `config.KNOWLEDGE_PREVIEW_MAX_LENGTH`, `config.KNOWLEDGE_COORD_CONCEPT_NAME_MAX_LENGTH`, `config.KNOWLEDGE_DEFAULT_COORD_CONCEPT_NAME`, `config.KNOWLEDGE_ENTRY_SCHEMA_VERSION`.
    *   Handles consent for public storage via `config.REQUIRE_PUBLIC_STORAGE_CONSENT`.
*   **Retrieval Functions (`retrieve_knowledge_by_id`, `retrieve_knowledge_by_keyword`):** Standard lookup functions.
*   **Logging:** Uses `_log_library_event` to log to `config.SYSTEM_LOG_PATH`.

### Data File Interactions
*   **Reads from and writes to:** `library_log.json` (path from `config.LIBRARY_LOG_PATH`).
*   **Writes to:** System log file defined by `config.SYSTEM_LOG_PATH`.

### Key Configurable Parameters (from `config.py`)
*   `LIBRARY_LOG_PATH`
*   `DEFAULT_SUMMARY_MAX_LENGTH`
*   `ETHICAL_ALIGNMENT_THRESHOLD`, `MITIGATION_ETHICAL_THRESHOLD`, `MITIGATION_SEVERE_ETHICAL_SCORE_THRESHOLD`, `MITIGATION_STRICT_CAUTION_ETHICAL_SCORE_THRESHOLD`
*   `LIBRARY_SENSITIVE_KEYWORDS`, `LIBRARY_REFRAMING_PHRASES` (keys for lists/dicts within config)
*   `MITIGATION_LOG_SUMMARY_MAX_LENGTH`
*   `KNOWLEDGE_PREVIEW_MAX_LENGTH`, `KNOWLEDGE_COORD_CONCEPT_NAME_MAX_LENGTH`, `KNOWLEDGE_DEFAULT_COORD_CONCEPT_NAME`, `KNOWLEDGE_ENTRY_SCHEMA_VERSION`
*   `REQUIRE_PUBLIC_STORAGE_CONSENT`
*   `DEFAULT_KNOWLEDGE_COHERENCE` (used by mock `score_ethics`)

---

## Ethical Framework (`core/ethics.py`)

### Overview
The `core/ethics.py` module is central to Sophia_Alpha2's ethical reasoning capabilities. It provides functionalities for scoring the ethical alignment of concepts and actions, tracking ethical trends over time, and managing a persistent database for this ethical data.

### Key Features & Functions
*   **Ethical Database (`_ethics_db`, `_load_ethics_db`, `_save_ethics_db`):**
    *   Manages an in-memory dictionary `_ethics_db` containing `ethical_scores` (a list of scoring events) and `trend_analysis` results.
    *   Persists this database to a JSON file specified by `config.ETHICS_DB_PATH`.
    *   Implements robust loading (handling missing/empty/corrupt files) and atomic saves.
    *   Log pruning for `ethical_scores` is based on `config.ETHICS_LOG_MAX_ENTRIES`.
*   **`score_ethics(awareness_metrics: dict, concept_summary: str = "", action_description: str = "") -> float`**:
    *   Calculates a composite ethical score (0.0-1.0) based on several components:
        1.  **Coherence Score:** Derived from `awareness_metrics['coherence']`. Weight: `config.ETHICS_COHERENCE_WEIGHT`.
        2.  **Manifold Valence Score:** Based on the 'x' coordinate of `awareness_metrics['primary_concept_coord']` and `config.MANIFOLD_RANGE`. Weight: `config.ETHICS_VALENCE_WEIGHT`.
        3.  **Manifold Intensity Preference Score:** Scores proximity of `awareness_metrics['raw_t_intensity']` to an ideal (defined by `config.ETHICS_IDEAL_INTENSITY_CENTER` and `config.ETHICS_INTENSITY_PREFERENCE_SIGMA`). Weight: `config.ETHICS_INTENSITY_WEIGHT`.
        4.  **Ethical Framework Alignment Score:** Keyword-based analysis of text against `config.ETHICAL_FRAMEWORK` (positive/negative keywords). Weight: `config.ETHICS_FRAMEWORK_WEIGHT`.
        5.  **Manifold Cluster Context Score:** Assesses average valence of neighboring concepts in the manifold (obtained via `core.brain.get_shared_manifold().get_conceptual_neighborhood`). Uses `config.ETHICS_CLUSTER_RADIUS_FACTOR`. Weight: `config.ETHICS_CLUSTER_CONTEXT_WEIGHT`.
    *   Logs detailed scoring event data and saves the updated ethics database.
*   **`track_trends() -> dict`**:
    *   Analyzes historical `ethical_scores` from the database to identify trends (improving, declining, stable).
    *   Uses T-weighting based on concept intensity (`primary_concept_t_intensity_raw` logged with each score) using `config.ETHICS_TREND_T_INTENSITY_FACTOR` and `config.ETHICS_TREND_BASE_WEIGHT`.
    *   Calculates short-term and long-term weighted averages based on window sizes derived from `config.ETHICS_TREND_MIN_DATAPOINTS` and factors (`config.ETHICS_TREND_SHORT_WINDOW_FACTOR`, `config.ETHICS_TREND_LONG_WINDOW_FACTOR`, `config.ETHICS_TREND_MIN_SHORT_WINDOW`, `config.ETHICS_TREND_MIN_LONG_WINDOW`).
    *   Trend determination uses `config.ETHICS_TREND_SIGNIFICANCE_THRESHOLD`.
    *   Stores analysis results in `_ethics_db["trend_analysis"]` and saves the database.
*   **Logging:** Uses `_log_ethics_event` to log to `config.SYSTEM_LOG_PATH`.

### Data File Interactions
*   **Reads from and writes to:** The ethics database JSON file, path defined by `config.ETHICS_DB_PATH`.
*   **Writes to:** System log file defined by `config.SYSTEM_LOG_PATH`.

### Key Configurable Parameters (from `config.py`)
*   `ETHICS_DB_PATH`
*   Weights for scoring: `ETHICS_COHERENCE_WEIGHT`, `ETHICS_VALENCE_WEIGHT`, `ETHICS_INTENSITY_WEIGHT`, `ETHICS_FRAMEWORK_WEIGHT`, `ETHICS_CLUSTER_CONTEXT_WEIGHT`.
*   `MANIFOLD_RANGE` (for valence and cluster scoring context).
*   `ETHICS_IDEAL_INTENSITY_CENTER`, `ETHICS_INTENSITY_PREFERENCE_SIGMA` (for intensity preference scoring).
*   `ETHICAL_FRAMEWORK` (dictionary defining principles, keywords).
*   `ETHICS_CLUSTER_RADIUS_FACTOR`.
*   Log/Trend parameters: `ETHICS_LOG_MAX_ENTRIES`, `ETHICS_TREND_MIN_DATAPOINTS`, `ETHICS_TREND_SIGNIFICANCE_THRESHOLD`, `ETHICS_TREND_T_INTENSITY_FACTOR`, `ETHICS_TREND_BASE_WEIGHT`, `ETHICS_TREND_SHORT_WINDOW_FACTOR`, `ETHICS_TREND_LONG_WINDOW_FACTOR`, `ETHICS_TREND_MIN_SHORT_WINDOW`, `ETHICS_TREND_MIN_LONG_WINDOW`.

---

## Dialogue Management (`core/dialogue.py`)

### Overview
`core/dialogue.py` is the central interaction hub for Sophia_Alpha2. It processes user input, orchestrates calls to other core modules (brain, persona, ethics, memory, library), and formulates the final response. It also implements the Command-Line Interface (CLI).

### Key Functions
*   **`generate_response(user_input: str, stream_thought_steps: bool = False) -> tuple[str, list, dict]`**:
    *   Orchestrates the entire response generation pipeline:
        1.  Retrieves the `Persona` instance (`get_dialogue_persona`).
        2.  Calls `core.brain.think()` for cognitive processing.
        3.  Updates `Persona` awareness.
        4.  Scores the interaction using `core.ethics.score_ethics`. Default fallback score from `config.DEFAULT_DIALOGUE_ETHICAL_SCORE_FALLBACK`. Summarization lengths for ethics context use `config.DIALOGUE_SUMMARY_LENGTH_SHORT` and `config.DIALOGUE_SUMMARY_LENGTH_ACTION`.
        5.  Stores a memory of the interaction via `core.memory.store_memory`. Max length for concept name and default name from `config.MAX_CONCEPT_NAME_FOR_MEMORY_LEN` and `config.DEFAULT_MEMORY_CONCEPT_NAME`. Memory summary length from `config.MAX_MEMORY_SUMMARY_LEN`.
        6.  Applies content moderation using `core.library.Mitigator` based on `config.MITIGATION_ETHICAL_THRESHOLD` and `config.ETHICAL_ALIGNMENT_THRESHOLD`.
        7.  Updates ethical trends via `core.ethics.track_trends`.
    *   Initializes awareness metrics using `config.DEFAULT_AWARENESS_METRICS_DIALOGUE`. Default error response uses `config.DEFAULT_DIALOGUE_ERROR_BRAIN_RESPONSE`.
*   **`dialogue_loop(enable_streaming_thoughts: bool = None)`**:
    *   Manages the interactive CLI loop.
    *   Handles special commands (`!help`, `!stream`, `!persona`, etc.).
    *   Uses `config.VERBOSE_OUTPUT` to determine initial thought streaming state.
*   **`get_dialogue_persona() -> Persona | None`**: Retrieves/initializes the shared `Persona` instance.

### Fallbacks and Logging
*   The module implements robust fallbacks for cases where core components (brain, persona, memory, ethics, library) fail to import, defining mock objects to allow the system to run in a degraded state or for testing.
*   Uses `_log_dialogue_event` for structured logging to `config.SYSTEM_LOG_PATH`.

### Data File Interactions
*   Indirectly interacts with data files through calls to other modules (e.g., `persona.save_state()`, `memory._save_knowledge_graph()`).
*   Writes to the system log file defined by `config.SYSTEM_LOG_PATH`.

### Key Configurable Parameters (from `config.py`)
*   `VERBOSE_OUTPUT` (influences default thought streaming in `dialogue_loop`).
*   `MITIGATION_ETHICAL_THRESHOLD`, `ETHICAL_ALIGNMENT_THRESHOLD` (used by `Mitigator` via `generate_response`).
*   `DEFAULT_AWARENESS_METRICS_DIALOGUE`, `DEFAULT_DIALOGUE_ERROR_BRAIN_RESPONSE`, `DEFAULT_DIALOGUE_ETHICAL_SCORE_FALLBACK`.
*   `MAX_CONCEPT_NAME_FOR_MEMORY_LEN`, `DEFAULT_MEMORY_CONCEPT_NAME`, `MAX_MEMORY_SUMMARY_LEN`.
*   `DIALOGUE_SUMMARY_LENGTH_SHORT`, `DIALOGUE_SUMMARY_LENGTH_ACTION`.
*   (Indirectly, all parameters used by the modules it calls).

---

## Graphical User Interface (`core/gui.py`)

### Overview
The `core/gui.py` module implements a web-based Graphical User Interface (GUI) using Streamlit, providing an alternative to the CLI for interacting with Sophia_Alpha2. It allows for chat-style interaction and visualization of some system states.

### Key Features & Functions
*   **Streamlit Integration:** Built entirely using Streamlit components.
*   **Initialization (`initialize_session_state`, `start_gui`):**
    *   `initialize_session_state`: Sets up Streamlit's session state variables for `dialogue_history`, `persona_instance`, `stream_thoughts_gui` preference (default from `config.VERBOSE_OUTPUT`), `last_thought_steps`, `last_awareness_metrics`, and `error_message`. It also handles the initial retrieval of the `Persona` instance via `core.dialogue.get_dialogue_persona()`.
    *   `start_gui`: Main entry point called by `main.py` to render the GUI.
*   **Interface Rendering (`render_main_interface`):**
    *   Displays persona information (name, mode, traits, awareness metrics) in a sidebar.
    *   Shows a snapshot of metrics from the last interaction.
    *   Provides controls: "Clear Dialogue History", "Reset Persona State".
    *   Presents the main chat interface where users can input queries.
    *   Displays dialogue history using `st.chat_message`.
    *   Includes an expander to show detailed "Thought Stream" if toggled.
    *   Simulates response streaming with a delay controlled by `config.GUI_RESPONSE_STREAMING_DELAY`.
*   **Backend Interaction:** Calls `core.dialogue.generate_response()` to process user input and get Sophia's reply.
*   **Fallbacks:** If core `dialogue` or `persona` components are unavailable, it uses mock objects to display error messages within the GUI.

### Data File Interactions
*   Does not directly interact with data files. All data persistence is handled by other modules (`persona`, `memory`, `library`, `ethics`) called via `core.dialogue`.

### Key Configurable Parameters (from `config.py`)
*   `PERSONA_NAME` (used for page title, defaults to 'Sophia_Alpha2').
*   `VERBOSE_OUTPUT` (sets the initial default for the "Show Thought Stream Expander" checkbox).
*   `GUI_RESPONSE_STREAMING_DELAY` (controls the simulated typing speed for responses).

---
