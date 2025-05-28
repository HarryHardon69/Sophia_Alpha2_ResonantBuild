# Sophia_Alpha2_ResonantBuild

## Vision
(Placeholder for Vision Statement - To be populated based on Phase2KB.md)

## Key Features
(Placeholder for Key Features - To be populated based on Phase2KB.md)
*   Modular core components (configurable, cognitive processing, memory, persona, knowledge library, dialogue management).
*   SNN-based cognitive architecture with LLM bootstrapping.
*   Persistent knowledge graph with novelty and ethical filtering.
*   Persistent Knowledge Library for curated information with ethical scoring.
*   Ethical Mitigation (`Mitigator` class) for content moderation.
*   Dialogue Management / Interaction Orchestration via CLI.
*   Streamlit-based GUI for interactive chat and system monitoring.
*   Centralized application entry point via `main.py` with CLI argument parsing.

## System Architecture
(Placeholder for System Architecture Overview - To be populated based on Phase2KB.md and further design)

The `main.py` script serves as the primary entry point for the Sophia_Alpha2 application. It handles command-line arguments, initializes configurations, ensures necessary directories are present, and launches the chosen user interface (CLI or GUI).

The `core.brain` module is a central component, housing the `SpacetimeManifold` class. This class implements a Spiking Neural Network (SNN) for cognitive processing, bootstraps concepts using Large Language Models (LLMs), incorporates STDP/Hebbian learning, and generates awareness metrics. `main.py` ensures this manifold is initialized at startup.

The `core.memory` module manages Sophia_Alpha2's knowledge graph, stored in `data/memory_store/knowledge_graph.json`. It handles storing new concepts based on novelty and ethical alignment, calculates concept novelty, and provides various functions for retrieving memories.

The `core.persona` module manages Sophia_Alpha2's identity, traits, operational mode, and evolving awareness state. Handles the persistence of this state to `persona_profile.json` (path typically defined in `config.PERSONA_PROFILE_PATH`).

The `core.library` module provides shared utility functions, ethical mitigation (`Mitigator` class), and manages the curated knowledge base (`library_log.json`), including ingestion with coordinate assignment and ethical scoring.

The `core.dialogue` module is the central interaction handler for Sophia_Alpha2. It orchestrates calls to the brain, persona, ethics, memory, and library modules to generate responses and manage the dialogue flow. Includes the main CLI loop for user interaction.

The `core.gui` module provides a Streamlit-based Graphical User Interface (GUI) for interacting with Sophia_Alpha2, visualizing thought processes, and displaying awareness metrics.

All core modules are designed to be extensively configurable via `config/config.py`.

## Directory Structure
```
Sophia_Alpha2_ResonantBuild/
├── .gitignore
├── README.md
├── config/
│   ├── __init__.py
│   └── config.py      # System-wide configuration settings
├── core/
│   ├── __init__.py    # Core package initializer, exposes key functions
│   ├── brain.py       # Cognitive core: SpacetimeManifold (SNN), LLM bootstrap, learning, awareness
│   ├── dialogue.py    # Orchestrates dialogue flow, CLI, and calls to other core modules
│   ├── ethics.py      # (Stub) For ethical framework and decision-making
│   ├── gui.py         # Streamlit-based GUI for interaction and visualization
│   ├── library.py     # Manages curated knowledge base, utilities, and ethical mitigation
│   ├── memory.py      # Manages the knowledge graph (storing, retrieving, novelty calculation)
│   └── persona.py     # Manages Sophia_Alpha2's identity, traits, mode, and awareness state
├── data/
│   ├── ethics_store/
│   │   └── .gitkeep
│   ├── library_store/ # Contains library_log.json (curated knowledge)
│   │   └── .gitkeep
│   ├── logs/
│   │   └── .gitkeep
│   ├── memory_store/  # Contains knowledge_graph.json
│   │   └── .gitkeep
│   ├── personas/
│   │   └── .gitkeep
│   ├── private/
│   │   └── .gitkeep
│   └── public/
│       └── .gitkeep
├── docs/
│   ├── Phase2KB.md      # Project knowledge base and evolving specifications
│   └── documentation.md # Detailed technical documentation
├── interface/
│   └── __init__.py
├── main.py            # Primary application entry point
├── requirements.txt
└── tests/
    ├── __init__.py
    └── .gitkeep
```

## Getting Started

### Prerequisites
(To be detailed: Python version, pip, virtual environment recommendations)

### Installation
(To be detailed: `git clone`, `cd Sophia_Alpha2_ResonantBuild`, `pip install -r requirements.txt`, environment variable setup for API keys if needed)

### Running the Application
The primary way to run Sophia_Alpha2 is through `main.py` from the project's root directory.

*   **Default Mode:**
    ```bash
    python main.py
    ```
    This will launch the GUI if `ENABLE_GUI` is set to `True` in `config/config.py`. Otherwise, it defaults to the Command-Line Interface (CLI).

*   **Force CLI Mode:**
    ```bash
    python main.py --interface cli
    ```
    This starts the interactive dialogue loop in your terminal.

*   **Force GUI Mode:**
    ```bash
    python main.py --interface gui
    ```
    This attempts to start the Streamlit-based GUI. If GUI is disabled in `config.py` or Streamlit is not installed, it will fall back to CLI mode.

*   **Single Query (CLI):**
    ```bash
    python main.py --query "Tell me about resonance."
    ```
    This submits a single query to Sophia, prints the response to the console, and then exits. The interface is automatically set to 'cli'.

*   **Verbose Output:**
    ```bash
    python main.py --verbose
    ```
    or
    ```bash
    python main.py -v
    ```
    This enables detailed logging output for the current session, overriding the `VERBOSE_OUTPUT` setting in `config/config.py`. It can be combined with other arguments, e.g., `python main.py --interface cli --verbose`.

Previously, `streamlit run core/gui.py` was an alternative way to start the GUI. While this might still work, using `python main.py` (with or without `--interface gui`) is now the recommended and centralized method as it ensures all system initializations managed by `main.py` are correctly performed.

## Roadmap
(Placeholder for Project Roadmap - To be populated later)
*   Phase 1: Initial Scaffolding (Complete)
*   Phase 2: Core module implementation (config.py, brain.py, memory.py, persona.py, library.py, dialogue.py, gui.py complete)
*   Phase 3: `main.py` integration (Complete). Implementation of ethics module.
*   Phase 4: ...

## Contributing
(Placeholder for Contribution Guidelines - To be populated later)

## License
(Placeholder for License Information - To be determined and added later)
