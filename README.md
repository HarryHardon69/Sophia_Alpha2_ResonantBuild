# Sophia_Alpha2_ResonantBuild

## Vision
Sophia_Alpha2_ResonantBuild aims to develop a modular, ethically-aligned cognitive architecture (Sophia_Alpha2) capable of sophisticated information processing, learning, and interaction. It leverages a Spiking Neural Network (SNN) core for cognitive functions, integrates with Large Language Models (LLMs) for knowledge bootstrapping, and emphasizes persistent memory, ethical content mitigation, and a dynamic persona. The system is designed for configurability and aims to explore concepts of artificial consciousness and resonant awareness in a single-user context initially.

## Key Features
*   **Modular Core Components:** Highly configurable modules for cognitive processing (`brain`), memory systems (`memory`), persona management (`persona`), curated knowledge (`library`), dialogue orchestration (`dialogue`), ethical framework (`ethics`), and user interface (`gui`).
*   **SNN-based Cognitive Architecture:** Utilizes a Spiking Neural Network (`SpacetimeManifold` in `core.brain`) for core cognitive processing, with LLM integration for concept bootstrapping and knowledge augmentation.
*   **Persistent Knowledge Graph:** Employs a graph-based memory (`core.memory`) for storing and retrieving conceptual relationships and interaction memories, with mechanisms for novelty calculation and ethical filtering.
*   **Curated Knowledge Library:** A persistent library (`core.library`) for storing validated and curated information, complete with metadata such as source, author, and ethical scoring.
*   **Dynamic Persona Management:** Features a dynamic persona (`core.persona`) whose awareness and operational mode evolve based on interaction metrics and SNN activity. Persona state is persisted.
*   **Ethical Framework and Mitigation:** Includes an ethical scoring system (`core.ethics`) and a `Mitigator` class (`core.library`) for content moderation and reframing based on configurable ethical principles.
*   **Configurable Logging and Resource Profiles:** Centralized configuration (`config/config.py`) allows for detailed logging setup and different resource profiles (low, moderate, high) to manage computational load.
*   **Dialogue Management:** Orchestrates interactions via `core.dialogue`, managing the flow between user input, cognitive processing, and response generation.
*   **User Interfaces:**
    *   Command-Line Interface (CLI) for direct interaction and scripting.
    *   Streamlit-based Graphical User Interface (GUI) for interactive chat and system monitoring.
*   **Centralized Application Entry Point:** `main.py` provides a unified way to start the application, parse arguments, and select the interface.

## System Architecture
Sophia_Alpha2_ResonantBuild is designed with a modular architecture to promote separation of concerns and maintainability.

*   **`main.py`**: The primary entry point. It handles command-line argument parsing, initializes the system-wide configuration, ensures essential directories are present, and launches the selected user interface (CLI or GUI). It also orchestrates the initial setup of the `SpacetimeManifold`.
*   **`config/config.py`**: Centralizes all system-wide configuration settings, including paths, resource profiles, API keys, persona details, ethical framework parameters, and operational flags. It supports loading configurations from environment variables with sensible defaults.
*   **`core/brain.py`**: Contains the `SpacetimeManifold` class, the cognitive core of the system. It implements the SNN using `snnTorch`, handles concept bootstrapping from LLMs (via `requests`), applies STDP/Hebbian learning rules, and generates awareness metrics based on SNN activity.
*   **`core/persona.py`**: Manages Sophia_Alpha2's identity (name, traits, operational mode) and her dynamic awareness state (curiosity, coherence, etc.). This state is influenced by metrics from the brain and is persisted to a JSON profile.
*   **`core/memory.py`**: Manages the system's persistent knowledge graph, stored as `knowledge_graph.json`. It's responsible for storing new memories (concepts) only if they meet novelty and ethical thresholds, calculating novelty, and providing functions to retrieve memories by ID, name, or recency.
*   **`core/library.py`**: Manages a curated, persistent knowledge library (`library_log.json`). It allows for storing detailed knowledge entries with metadata (source, author, ethical score). This module also provides shared utilities like text sanitization, summarization, coordinate validation, and the `Mitigator` class for ethical content moderation. Custom exceptions for the `core` package are defined here.
*   **`core/ethics.py`**: Responsible for the ethical dimension of the system. It scores the ethical alignment of concepts and actions based on awareness metrics and textual content. It also tracks ethical trends over time using a T-weighted analysis and manages a persistent database for these scores.
*   **`core/dialogue.py`**: Orchestrates the interaction flow. When user input is received, it calls the `core.brain` to process the input, updates the `core.persona` with new awareness metrics, consults `core.ethics` for an ethical score, potentially stores the interaction in `core.memory`, and uses `core.library`'s `Mitigator` for content moderation before generating a final response. It also contains the CLI loop.
*   **`core/gui.py`**: Provides a Streamlit-based web GUI for user interaction. It uses `core.dialogue` to process inputs and display responses, and also visualizes persona awareness and thought processes.

The flow generally starts with `main.py` setting up the environment and selected UI. User input, whether from CLI or GUI, is routed through `core.dialogue`. `core.dialogue` then coordinates with `core.brain` for thinking, `core.persona` for state, `core.ethics` for scoring, `core.memory` and `core.library` for knowledge, before formulating a response.

## Directory Structure
```
Sophia_Alpha2_ResonantBuild/
├── .gitignore
├── README.md
├── config/
│   ├── __init__.py
│   └── config.py      # System-wide configuration settings
├── core/
│   ├── __init__.py    # Core package initializer, exposes key functions & 'think' convenience fn
│   ├── brain.py       # Cognitive core: SpacetimeManifold (SNN), LLM bootstrap, learning
│   ├── dialogue.py    # Orchestrates dialogue flow, CLI, and calls to other core modules
│   ├── ethics.py      # Ethical framework, scoring, and trend analysis
│   ├── gui.py         # Streamlit-based GUI for interaction and visualization
│   ├── library.py     # Manages curated knowledge base, utilities, custom exceptions, Mitigator
│   ├── memory.py      # Manages the knowledge graph (storing, retrieving, novelty)
│   └── persona.py     # Manages Sophia_Alpha2's identity, traits, mode, and awareness
├── data/
│   ├── ethics_store/  # Contains ethics_db.json
│   │   └── .gitkeep
│   ├── library_store/ # Contains library_log.json (curated knowledge)
│   │   └── .gitkeep
│   ├── logs/          # Contains sophia_alpha2_system.log
│   │   └── .gitkeep
│   ├── memory_store/  # Contains knowledge_graph.json & memory_log.json
│   │   └── .gitkeep
│   ├── personas/      # Contains persona_profile.json (e.g., Sophia_Alpha2_Prime.json)
│   │   └── .gitkeep
│   ├── private/       # Fallback directory for persona profile if config not loaded
│   │   └── .gitkeep
│   └── public/        # Not currently used, placeholder for future public data
│       └── .gitkeep
├── docs/
│   ├── Phase2KB.md      # Project knowledge base and evolving specifications (conceptual)
│   └── documentation.md # Detailed technical documentation (conceptual)
├── interface/         # Placeholder for potential future alternative interface modules
│   └── __init__.py
├── main.py            # Primary application entry point
├── requirements.txt   # Python package dependencies
└── tests/             # Placeholder for future dedicated test suite
    ├── __init__.py
    └── .gitkeep
```

## Getting Started

### Prerequisites
*   **Python:** Python 3.9 or higher is recommended.
*   **Git:** For cloning the repository.
*   **pip:** Python package installer, usually comes with Python.
*   **Virtual Environment (Recommended):** To manage project dependencies in an isolated environment. `venv` is part of the Python standard library.

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/Sophia_Alpha2_ResonantBuild.git
    cd Sophia_Alpha2_ResonantBuild
    ```
    (Replace `https://github.com/your_username/Sophia_Alpha2_ResonantBuild.git` with the actual repository URL).

2.  **Create and activate a virtual environment (Recommended):**
    *   On Linux/macOS:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    *   On Windows:
        ```bash
        python -m venv .venv
        .venv\Scripts\activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration
The central configuration file is `config/config.py`. Many parameters can be overridden by environment variables. Key settings to review:

*   **LLM Provider (`LLM_PROVIDER`):**
    *   Choose from `"openai"`, `"lm_studio"`, `"ollama"`, or `"mock_for_snn_test"`.
    *   Set via environment variable `LLM_PROVIDER` or modify the default in `config.py`.
*   **API Keys and Endpoints:**
    *   **OpenAI:** Set `OPENAI_API_KEY` (required) and optionally `OPENAI_BASE_URL` as environment variables.
    *   **LM Studio:** Ensure LM Studio is running and accessible at `LM_STUDIO_BASE_URL` (default `http://localhost:1234/v1`). No API key is typically required.
    *   **Ollama:** Ensure Ollama service is running and accessible at `OLLAMA_BASE_URL` (default `http://localhost:11434/api`). No API key is typically required. Specify the model using `OLLAMA_MODEL`.
*   **Persona:**
    *   `PERSONA_NAME`: Sets the active persona (e.g., "Sophia_Alpha2_Prime").
    *   `PERSONA_PROFILE_PATH`: Path to the persona's JSON profile. Defaults are constructed based on `PERSONA_NAME` and `PERSONA_DIR`.
*   **General Behavior:**
    *   `ENABLE_GUI`: Set to `True` or `False` (as a string for environment variable, e.g., `ENABLE_GUI="True"`) to enable or disable the Streamlit GUI by default.
    *   `VERBOSE_OUTPUT`: Set to `True` or `False` to control detailed console logging. Can be overridden by the `--verbose` CLI flag.
    *   `LOG_LEVEL`: Controls the verbosity of file logging (e.g., "debug", "info", "warning").

Refer to `config/config.py` for a comprehensive list of all configurable parameters and their default values.

### Running the Application
The primary way to run Sophia_Alpha2 is through `main.py` from the project's root directory:

*   **Default Mode (GUI if enabled, else CLI):**
    ```bash
    python main.py
    ```
    This launches the GUI if `ENABLE_GUI` is `True` in `config.py` (or via environment variable). Otherwise, it defaults to the CLI.

*   **Force CLI Mode:**
    ```bash
    python main.py --interface cli
    ```
    Starts the interactive dialogue loop in your terminal.

*   **Force GUI Mode:**
    ```bash
    python main.py --interface gui
    ```
    Attempts to start the Streamlit-based GUI. If GUI is disabled via `config.ENABLE_GUI=False` or Streamlit is not functional, it will fall back to CLI mode.

*   **Single Query (CLI):**
    ```bash
    python main.py --query "Tell me about resonance."
    ```
    This submits a single query, prints the response, and exits. The interface is automatically set to 'cli'.

*   **Verbose Output:**
    ```bash
    python main.py --verbose
    ```
    or
    ```bash
    python main.py -v
    ```
    Enables detailed logging output for the current session, overriding the `VERBOSE_OUTPUT` setting in `config/config.py`. It can be combined with other arguments (e.g., `python main.py --interface cli --verbose`).

Using `python main.py` is the recommended way to start the application as it ensures all system initializations are correctly performed.

## Roadmap
*   **Phase 1: Initial Scaffolding & Core Structure Definition** (Complete)
    *   Project setup, directory structure, initial `config.py`.
*   **Phase 2: Single-user Coherence Foundation - Core Module Implementation** (Complete)
    *   Development of `core` modules: `brain` (SNN basics, LLM bootstrap), `memory` (KG persistence, novelty), `persona` (state, awareness), `library` (storage, mitigation utils), `ethics` (scoring, trends), `dialogue` (orchestration, CLI), `gui` (Streamlit UI).
    *   `main.py` integration for centralized startup and argument parsing.
    *   Refinement of error handling, configuration management, and internal APIs.
*   **Future Phases (Conceptual):**
    *   Advanced learning paradigms and SNN self-modification.
    *   Enhanced multi-modal input processing.
    *   More sophisticated ethical reasoning and dynamic framework adjustments.
    *   Comprehensive testing frameworks and CI/CD integration.
    *   Exploration of multi-user networking and shared awareness contexts.

## Contributing
Contribution guidelines will be established as the project matures. For now, please refer to `docs/Phase2KB.md` (if available, currently conceptual) or the main codebase for architectural insights if considering contributions. Key areas for immediate improvement include expanding test coverage and refining documentation.

## License
License: TBD (Likely MIT or Apache 2.0). A formal LICENSE file will be added in a future iteration.
